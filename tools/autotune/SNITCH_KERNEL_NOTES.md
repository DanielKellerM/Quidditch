# Snitch Kernel Notes — runtime reference + autotuner application

Engineer-facing reference for the Snitch-kernel autotuner (`tools/autotune`).
Source tree: `~/Projects/Quidditch/snitch_cluster/sw/`. All `file:line`
references are relative to that tree unless an absolute path is given.

**Hard fact that drives everything below:** in the canonical cluster (9 harts =
8 compute + 1 DM) the **DM core is the LAST hart index** and has **NO FPU**.
Compute cores `0..N-1` have FPUs (plus SSR + FREP). `SNRT_CLUSTER_DM_CORE_NUM`
is `1` (`impl/snitch_cluster_cfg.h.tpl:46`), so the partition is exactly
"`compute_core_num` compute harts + 1 trailing DM hart." Concrete corroboration:
`wake_dm` raises `snrt_int_cluster_set(1 << snrt_cluster_compute_core_num())`
(`src/dm.h:142`) — the wake-target bit is exactly `compute_core_num`, pinning the
DM core to the last hart index. Any floating-point op (`fmul.d`, `fadd.d`,
`fdiv.d`, `tanh`/`expf` via libm) executed on the DM core traps or falls through
to a scalar/degenerate path. This is the root cause of the autotuner's
integer-only Tier-1 reference (Part 2).

---

## Part 1 — Snitch runtime + kernel reference

### 1.1 Core-gating API — "how to avoid using the DM core"

The partition is purely index-based. `snrt_cluster_core_idx()` in
`[0, cluster_core_num)` is the value every predicate keys off; indices
`0..compute_core_num-1` are compute cores, the final index is the DM core.

The predicates below are `inline` definitions in `src/team.h`; `src/team.c` is a
list of `extern` re-declarations forcing one external emission of each; the
public forward declarations (what `harness.c.in` `#include`s) live in
`api/team_decls.h` — that is the header to include when re-gating the reference.

| Function | file:line | Semantics |
|----------|-----------|-----------|
| `snrt_hartid` | `src/team.h:23-27` | Raw `mhartid` CSR; system-global, includes base-hartid offset. |
| `snrt_cluster_core_num` | `src/team.h:43-45` | `SNRT_CLUSTER_CORE_NUM` = total cores/cluster **incl.** DM (e.g. 9). |
| `snrt_cluster_dm_core_num` | `src/team.h:116-118` | `SNRT_CLUSTER_DM_CORE_NUM`, normally 1 — the FPU-less DMA hart(s). |
| `snrt_cluster_compute_core_num` | `src/team.h:125-127` | `core_num - dm_core_num` = number of FPU compute cores. The partition boundary index. |
| `snrt_global_compute_core_num` | `src/team.h:70-72` | `cluster_num() * cluster_compute_core_num()`. Denominator for global work split that **excludes** DM cores. |
| `snrt_global_core_idx` | `src/team.h:79-81` | Zero-based system-wide index over ALL cores (compute + DM): `hartid() - base_hartid`. |
| `snrt_cluster_core_idx` | `src/team.h:107-109` | `global_core_idx() % cluster_core_num()`. **The gating value.** |
| `snrt_global_compute_core_idx` | `src/team.h:88-91` | `cluster_idx()*cluster_compute_core_num() + cluster_core_idx()` — dense compute-only global index for slicing work. **Valid only on compute cores** (on a DM core `cluster_core_idx() == compute_core_num`, so it runs one past the compute range — guard with `snrt_is_compute_core()` first). |
| `snrt_is_compute_core` | `src/team.h:134-136` | `cluster_core_idx() < cluster_compute_core_num()`. TRUE on FPU harts. **The "this hart runs the math" predicate.** |
| `snrt_is_dm_core` | `src/team.h:152-154` | `!snrt_is_compute_core()`. TRUE only on the LAST hart. The only place DMA is issued. |
| `snrt_cluster_is_last_compute_core` | `src/team.h:143-145` | `cluster_core_idx() == cluster_compute_core_num() - 1`. Highest-indexed COMPUTE core (NOT the DM core). Single-representative election (epilogue / single-writer). NB: the prompt's "snrt_is_last_compute_core" is named `snrt_cluster_is_last_compute_core` in this tree. |

**Barriers (the compute/DMA rejoin points):**

| Function | file:line | Semantics |
|----------|-----------|-----------|
| `snrt_cluster_hw_barrier` | `src/sync.h:204-206` | `csrr x0, barrier` — single-instruction HW barrier over **ALL** cluster harts (compute AND DM). Every hart must execute it or the cluster deadlocks. |
| `snrt_global_barrier` | `src/sync.h:290-301` | Hierarchical: hw_barrier → **only the DM core** does `snrt_inter_cluster_barrier()` → hw_barrier. DM core is the per-cluster inter-cluster delegate. |
| `snrt_global_sw_barrier` | `src/sync.h:267-278` | Same shape, forces the SW inter-cluster path. |

**To avoid the DM core:** keep all FPU/reference work inside
`if (snrt_is_compute_core()) { ... }`. The else-branch (== `snrt_is_dm_core()`,
the single last hart) is the only place DMA is issued. The DM core's body is
`dm_main` (`src/dm.h:174-241`; the DMA service loop proper is `while(!do_exit)`
at `:182-232`), a pure-DMA worker loop (queue pop →
`__builtin_sdma_start_oned/twod` → `wfi_dm` sleep when the queue is empty) — it
never runs an FPU op.

### 1.2 Canonical kernel structure

The literal "avoid the DM core" pattern, predicates at `src/team.h:134-154`,
cleanest bare exemplar `blas/gemv/src/main.c:32-49` (the compute step there
calls `gemv()` rather than inlining the SSR/FREP region; the skeleton below is
the stylized composite):

```c
if (snrt_is_dm_core()) {
    snrt_dma_start_1d(dst, src, bytes, ch);   // DMA operands L3 -> TCDM
    snrt_dma_wait_all();
}
snrt_cluster_hw_barrier();                     // tile resident
if (snrt_is_compute_core()) {
    // SSR + FREP math, sliced by snrt_global_compute_core_idx / cluster_core_idx
}
snrt_cluster_hw_barrier();
if (snrt_is_dm_core()) {
    snrt_dma_start_1d(out, res, bytes, ch);    // result TCDM -> L3
    snrt_dma_wait_all();
}
```

- **Work split across compute cores:** two strategies.
  - **STRIDED interleave** (stride = `compute_core_num`, keeps consecutive
    elements in distinct TCDM banks): `gemm.h:45-53` (`sc_st_gemm`: "Compute
    cores work not on contiguous blocks but on strided rows"), `axpy.h:22-24,50-55`.
  - **CONTIGUOUS block** (`frac = M/num`, `start = idx*frac`, last core mops up
    the remainder): `gemv.h:78-90`, `dot.h:112-116`.
- **SSR + FREP streaming compute region** (self-contained on one FPU core, no
  DMA): `blas/gemm/src/gemm_fp64.h:104-191` (SSR setup, `snrt_ssr_enable` at
  `:191`), `:201-236` (FREP body). Sequence: `snrt_ssr_loop_Nd` + `snrt_ssr_repeat` →
  `snrt_ssr_read/write` (bind base ptrs) → `snrt_ssr_enable` (ft0/ft1/ft2 become
  stream ports) → `frep.o` over unrolled `fmadd.d` reading ft0/ft1, writing ft2
  → `snrt_fpu_fence` → `snrt_ssr_disable`. SSR CSR accessor bottoms out at
  `src/ssr.h:164` (`scfgwi` in `write_ssr_cfg`); enable/disable
  `src/ssr.h:95/108` (`__builtin_ssr_enable/disable`, with a `csrsi/csrci ssr,1`
  fallback for the non-LLVM toolchain).
- **FREP encoding** (`frep.o rep, n_insn, stagger_cnt, stagger_mask`): `rep` is
  a GPR holding **count-1** (loop runs count times); `n_insn` is the body length
  (immediate); stagger fields rotate fp source regs per iter (`0,0` in GEMM,
  `3,0b0010` in SARIS stencils). Nestable. Canonical: `gemm_fp64.h:204,213`
  (`n_frep` = `(M*N/unroll)-1`, `K-3`); minimal `tests/src/gemm_frep.c:9-55` /
  `gemm_frep1d.c:9-55` confirm the `n_iter-1` encoding (`:49`).
- **Multi-accumulator unrolling = FMA-latency hiding:** `unroll = 8` (must be ≥
  FMA latency), `gemm_fp64.h:101`; `dot` uses 4-acc, `dot.h:40` (`frep.o ...,4`
  at `dot.h:62`). Sequencer capacity bounds
  (`SNRT_NUM_SEQUENCER_LOOPS/INSNS`) are hard search-space constraints
  (`gemm_fp64.h:201` guard).
- **Double-buffering (3-stage software pipeline):** `gemm.h:253-254` derives
  `comp_i=i-1, dma_out_i=i-2` from one loop index `i` (with `dma_in_i=i`);
  `buff_idx`/the `[0]/[1]` buffer pair ping-pongs; loop runs `num_tiles+2` iters
  (`gemm.h:244` `num_iters += 2`). `axpy.h:126-202` is the 1D version;
  `batchnorm.h:98-176` is the `read_buf/write_buf` toggle form (`:98-99,155-156`).
- **Barrier-count invariant:** ONE bottom barrier when double-buffered
  (`gemm.h:488`, `axpy.h:201`); single-buffer paths add extra barriers to
  serialize DMA-in → compute → DMA-out (`gemm.h:404` `if(!double_buffer)`,
  `axpy.h:153-154,179,197` `if(!DOUBLE_BUFFER)`). Any retuned variant MUST
  preserve this.
- **ROI / fence ordering:** `snrt_mcycle()` brackets the compute region
  (`gemm_fp64.h:195/309`, `axpy.h:182/193`, `dot.h:120/139`) on **compute
  cores**, never the DM core. `snrt_fpu_fence` (`src/ssr.h:39`) must precede the
  post-compute barrier so the DM core never DMAs out a half-written result; it
  must come **before** `snrt_ssr_disable` (`gemm_fp64.h:308-311`).

### 1.3 Per-family idioms

| Family | Reference kernel | Idiom |
|--------|------------------|-------|
| **gemm** | `gemm_fp64_opt` (`gemm_fp64.h:88-313`), `sc_st_gemm` (`gemm.h:40-68`), driver `gemm()` (`gemm.h:188-492`) | 4D/3D SSR (A,B read / C write) + `unroll=8` FREP `fmadd.d`; rows STRIDED across compute cores; tiled `dim/num_tiles`, double-buffered; `parallelize_k` does a log-tree cross-cluster DMA reduction (`gemm.h:464-484`). |
| **gemv (= matvec, M=1)** | `gemv()` `gemv.h:71-96`, `single_core_gemv` `:23-67` | CONTIGUOUS row block per compute core (last core takes remainder, `:78-83`); 2D SSR over `(n,m)`, broadcast `x` via SSR stride-0; per row `frep.o n-1` `fmadd` (`:55-59`) → one streamed dot product. The simplest non-trivial target. |
| **axpy (elementwise)** | `axpy_job` `axpy.h:70-203` (`axpy_opt:44`) | 3-stage pipelined 1D; STRIDE-interleave elements; 3 SSRs (x,y read / z write, `:53-55`), single `frep.o frac-1` `fmadd.d ft2,a,ft0,ft1` (`:60-63`). |
| **dot (reduction)** | `dot()` `dot.h:92-149` (`dot_seq_4_acc:40`) | DMA whole x,y in once (`:106-107`); each core a CONTIGUOUS chunk → partial via 4-acc SSR-FREP; barrier; core 0 serially sums partials under `snrt_fpu_fence` (`:131-135`). Map + serial/tree reduce. |
| **gelu (nonlinear)** | `gelu_fp64` `gelu.h:49-57`, `gelu_layer` `:60-97` | Per-element scalar activation, NO SSR/FREP; `tanh`-approx via **real libm `tanh`+`sqrt`** (`gelu_activation_fp64`, `gelu.h:29-31`) on the compute FPU. Transcendental-free i-BERT `sigmoid_gelu_fp64` exists (`gelu.h:36-46`) but its call site is commented out (`gelu.h:53`); `gelu_activation_fp64` is the live path (`gelu.h:54`). |
| **softmax (nonlinear+reduction)** | `softmax_fp32` `softmax.h:40-72`, `softmax_layer` `:80-135` | Numerically-stable 3-pass per row: scalar max-reduce (seed `-INFINITY`, `:48`) → `expf(x-max)` + sum-reduce (**libm `expf`**, `softmax.h:60`) → divide (`:66`). Each core owns whole rows → no cross-core reduction. |
| **layernorm (nonlinear+reduction)** | naive `layernorm_fp32.h:12-65`, opt `:67-221`, dispatch `layernorm.h:49-166` | Two sum-reductions (mean, var) + rsqrt. fp32 opt path: `vfsum.s` reductions inside `frep.o` (`:135-139`), UNROLL=4 (`:7`); rsqrt = `fsqrt.s`+`fdiv.s` then `vfcpka.s.s` broadcast (`:178-180`). The **widening `vfsumex.s.h` (fp16→fp32 / fp8→fp16→fp32) reductions live in the fp16/fp8 variants** (`layernorm_fp16.h:75-78`, `layernorm_fp8.h:82-89`), NOT in the fp32 opt. Naive scalar template uses libm `sqrtf` (`:52`). |
| **fusedconv** | `conv2d_fp32` `conv2d.h:924`, `bn_relu` `:287` | Conv epilogue (BatchNorm `y=kappa*x+lambda` via `vfmul`/`vfadd` + ReLU `vfmax.s ...,0`) fused **in-place** in ONE `frep` over the TCDM output (`:351-364`); no L2/HBM round-trip. `bn_relu` is the generic fused-affine+activation primitive. (A separate `kernels/dnn/fusedconv` also exists; this doc points at `conv2d.h`.) |
| **fused_concat_linear** | `..._optimized` `fused_concat_linear.h:83` | Concat→Linear with NO materialized concat buffer: concat axis becomes the K-tiling axis, `k_tiles=num_inputs` (`:102`), `parallelize_k=1` (`:104`) → per-input partials summed by the GEMM cross-cluster log reduction. `concat_layer` (`concat.h:35`) is the DMA-only fallback. |
| **flashattention_2 / mha** | `flashattention_2_fp32` `flashattention_2_fp32.h:8`, `mha_fp32` `mha_fp32.h:7` | QK^T → **online softmax** (running max/sum, rescale) → ×V, all in TCDM, S/P/stats/partial-O never staged to HBM; `B_r,B_c` block the row/col tiles (`:14-15,44-56`). MHA maps heads to clusters (`mha_fp32.h:9` `snrt_cluster_idx() < num_heads`) then feeds `fused_concat_linear` (`:33-39`) for the output projection. |

---

## Part 2 — Application to the autotuner

### 2.1 THE KEY UNBLOCK: gate the reference to a compute core

**Current state** (`tools/autotune/harness.c.in:42-46`):

```c
int main(void) {
  // (buffers declared via @@BUFFER_DECLS@@ at line 43)
  // The 8 compute cores park in the worker loop; the DM core drives below.
  if (!snrt_is_dm_core()) return quidditch_dispatch_enter_worker_loop();
  // ---- DM core only ----
```

Every non-DM hart returns into `quidditch_dispatch_enter_worker_loop()`
(`runtime/runtime/src/Quidditch/dispatch/dispatch.c:60-88`) and PARKS in a
`while(!exit){ snrt_cluster_hw_barrier(); ... }` loop, only waking to run the
dispatched kernel when the DM core releases the barrier and a workgroup is
assigned. **All** code after the gate — input fill, dispatch, AND the Tier-1
reference + compare (`harness.c.in:114-127`, `@@CHECK@@` at line 120) — runs
**only on the DM core**, which has no FPU (`team.h:152`
`snrt_is_dm_core = !snrt_is_compute_core`).

That single gate is the **sole reason** the reference is integer-only:
`gen_harness.py:82-138` emits a structured-integer reference
(`A[i,k]=i*K+k+1`, `B value=k*N+j+1`) and an in-harness CHECK that recomputes
`want` with `(int)A*(int)B` accumulated in an `int`, comparing `(int)C` — pure
`int mul/add` + an implicit `fcvt.w.d` on the read of the f64 buffer
(`gen_harness.py:128-138`). It `assert`s no int32 overflow and rejects the shape
otherwise (`gen_harness.py:96-99`). `fdiv.d`/`fmul.d`/`fadd.d` would trap on the
DM core. KERNELS.md spells out the invariant: *"DM core (hart_00008) has no FPU
— the in-harness Tier-1 reference stays integer (`fcvt.w.d` + int mul/add). All
floating-point checking is host-side (Tier-2)."*

**The lever:** the FPU lives on the compute harts (`snrt_is_compute_core()`,
`team.h:134`). Re-gating the reference compute onto a compute core would make an
**in-harness FP reference possible** — a real `rel+abs` tolerance compare
in-sim. That would collapse the "Tier-2 is host-only" restriction:
sigmoid/tanh, matvec/M=1 (int32 overflows for real K), large-K, and f32/f16
(KERNELS.md widening steps 3-5) would no longer need to be punted to a host
post-process.

**⚠ This is NOT a one-line predicate swap — and it is NOT free, because the
compute cores are already captured by the dispatch worker loop.** Concretely:

1. **Don't unconditionally `return enter_worker_loop()` for non-DM cores**
   (`harness.c.in:46`). A compute core that has entered
   `quidditch_dispatch_enter_worker_loop` is **stuck in
   `snrt_cluster_hw_barrier()`** (`dispatch.c:65-86`) until
   `quidditch_dispatch_quit()` sets `exit=true` and fires the releasing barrier
   (`dispatch.c:90-93`, called from `harness.c.in:112`). There is **no ad-hoc
   way to run arbitrary reference code on a parked compute core** mid-loop —
   the worker loop only ever calls `configuredKernel`. So the FP reference
   cannot simply "also run on hart 0"; it must run **after** `quit()` has
   released the compute cores, or via a dedicated post-dispatch barrier phase
   added to both the harness main and the worker contract.
2. **Respect the barrier/worker contract** (`dispatch.c:60-93`). Options: (a)
   run the FP reference on compute core 0 **after** `quidditch_dispatch_quit()`
   releases it — but `quit()` is the shutdown barrier, so this needs a custom
   post-quit rendezvous; or (b) add a new dispatched "reference" kernel and run
   it through the normal queue/execute path so it lands on a compute core
   legitimately. Either way the worker loop / barrier-count invariant (§1.1)
   must stay balanced or the cluster deadlocks.
3. **Share the I/O buffers.** `C` and the inputs are currently DM-core-local
   `static` bss (declared at `@@BUFFER_DECLS@@`, `harness.c.in:43`; filled at
   `@@INPUT_FILL@@`, `:49`). They are already in cluster-shared TCDM/L1 in the
   sense that the dispatched kernel reads them via `binding_ptrs`, but the
   reference compute core must see the **post-dispatch** `C`, which means the
   reference must run after the rejoin barrier.
4. **Move the verdict.** `printf` mismatch reporting (`harness.c.in:122-127`)
   must move to whichever core owns the verdict, or the error count must be
   barrier-reduced back to the DM core for the single print.
5. **`gen_harness.py`** then emits an FP reference (`fmul.d`/`fadd.d`, libm
   `expf`/`tanh` for nonlinear ops) + a `rel+abs` tolerance compare by dtype,
   and **drops** the rejections that exist only because the int gate cannot
   represent those shapes. Note those rejections live in **three** places, not
   one: the int32-overflow reject is `gen_harness.py:96-99`; the **degenerate
   `1 in shape` / matvec reject is `spec.py:93-94`**; the f32/f16 dtype reject
   is `spec.py:86-87` plus `gen_harness.py:144-145` (`C_TYPE` has only `f64`).
   All three must be relaxed for the FP gate.

**The cycle metric is unaffected:** `dispatch_cycles = t1 - t0` is bracketed by
`read_csr(mcycle)` around the dispatch fan-out (`harness.c.in:100-110`); the
reference/compare runs **after** `t1`, outside the ROI. Making it FP does not
perturb the measurement.

**Honest caveats:**

- The worker-loop capture above is the subtle, load-bearing risk: the "run FP on
  a compute core" idea is sound in principle but requires restructuring the
  dispatch rendezvous, not flipping a predicate. Budget for it accordingly.
- A **host Tier-2 may still be wanted** as an *independent* cross-check — an
  in-harness reference compiled by the same toolchain that builds the candidate
  shares failure modes (codegen bugs, libm version). The existing host path
  (`HARNESS_DUMP_OUTPUT` FNV-1a hash, `harness.c.in:129-145`, `tier2.py`) is
  cheap insurance and orthogonal. Note `tier2.py` today is an **exact** FNV-1a
  identity check (`tier2.py:32-34,94`, hash over `struct.pack("<d", ...)` of the
  integer `matmul_reference`); its own header (`tier2.py:12`) flags rel+abs
  fp-tolerance as a *later* step. So the rel+abs FP path does not exist on
  either side yet — it must be built, in-harness or host.
- **FP determinism RTL-vs-host:** an exact byte compare across RTL-FPU and host
  libm is not guaranteed (rounding, libm impl, FMA-vs-separate-mul-add,
  reduction order — note the multi-accumulator tree reductions in §1.2). The
  in-harness FP gate must be a **tolerance** compare (`rel+abs` by dtype), never
  bit-exact. This is also why the int gate was chosen originally: it is exact.
- The in-harness reference must itself obey the compute/DM split and barrier
  protocol (§1.1) — a botched re-gate deadlocks the cluster rather than failing
  cleanly.

### 2.2 KERNELS.md widening op → snitch_cluster reference kernel mapping

| KERNELS.md op | Closest reference kernel | Idiom to reuse |
|---------------|--------------------------|----------------|
| matmul / matmul_transpose_b | `gemm_fp64_opt` (`gemm_fp64.h:88-313`), `gemm_fp32_opt` (`gemm_fp32.h:209`), `sc_st_gemm` (`gemm.h:40`) | 4D/3D SSR + `unroll=8` FREP `fmadd.d`; STRIDED row split; `gemm_fp32` `beta` branch fuses `+beta*C` (`gemm_fp32.h:263-290`). |
| matvec / M=1 (GEMV) | `gemv()` / `single_core_gemv` (`gemv.h:71-96`, `:23-67`) | CONTIGUOUS row block per core; 2D SSR over `(n,m)` + stride-0 broadcast of x; `frep.o n-1` `fmadd` per row. (int32 ref overflows real K → must ride the FP gate per §2.1; the `1 in shape` reject is `spec.py:93-94`.) |
| elementwise add/mul/sub | `axpy_job` (`axpy.h:70-203`, `axpy_opt:44`); `dot` (`dot.h:92-149`) for reduction-shaped combines | 1D STRIDE-interleave; 3 SSRs (2 read / 1 write); single `frep.o frac-1` `fmadd.d`. `runtime/samples/vec_multiply` is the bare starting point KERNELS.md cites; `spec.py` already lists `elementwise` in `V1_OPS` but `gen_harness.py:148-152` raises until `_gen_elementwise` exists. |
| relu / sigmoid / tanh | `gelu`'s activation idiom: `bn_relu` (`conv2d.h:287/351`, ReLU = `vfmax.s ...,0`); `gelu_activation_fp64` (`gelu.h:29-31`, libm `tanh`); i-BERT poly `sigmoid_gelu_fp64` (`gelu.h:36-46`) if avoiding the transcendental | Per-element scalar (relu stays piecewise-linear → int-exact); sigmoid/tanh need libm + the FP gate. As `bn_relu` epilogue: swap `vfmax` for the chosen op inside one `frep`. |
| softmax | `softmax_fp32` (`softmax.h:40-72`) | 3-pass max-reduce → `expf(x-max)`+sum-reduce → divide; per-row, core-local; libm `expf`. FP gate. |
| layernorm | `layernorm_naive<T>` (`layernorm_fp32.h:12-65`) reference + `layernorm_fp32_opt` (`:67-221`) | Two sum-reductions + rsqrt (`fsqrt.s`+`fdiv.s`, `:178-180`); fp32 opt uses `vfsum.s` in `frep` (`:135-139`). For fp16/fp8 the widening `vfsumex.s.h` reduction lives in `layernorm_fp16.h`/`layernorm_fp8.h`. FP gate. The naive template is exactly the plain-C FP reference to run on a compute core. |
| fused matmul + bias + act | `bn_relu` (`conv2d.h:287/351`) **or** `gemm_fp32_opt` `beta`-init (`gemm_fp32.h:263-290`) | Cheapest matmul+bias: preload bias row into C, run `beta=1` → bias folded into accumulator init for free. matmul+bias+act: chain a `bn_relu`-style `frep` epilogue (bias=`lambda`, scale=`kappa`, act = `vfmax`/chosen op) over the in-TCDM result — no HBM round-trip. Multi-op host/FP reference required. |
| GRU / LSTM gate cell | `fused_concat_linear_optimized` (`fused_concat_linear.h:83`) + `gemv` idiom + `bn_relu` epilogue | `[x;h]@W_gate` = K-tiled GEMM over x and h with `parallelize_k=1` (concat axis = K axis, no stacked buffer); per-timestep matvec shape = `gemv`; gate sigmoid/tanh fuse as a `bn_relu`-style FREP epilogue over the gate pre-activations resident in TCDM. FP gate; depends on the nsnet2 codegen fix. |

### 2.3 Autotuner knobs (search space)

Confirmed tunables, all visible in the reference kernels:
- **Unroll / accumulator count** — `unroll=8` (≥ FMA latency), `gemm_fp64.h:101`;
  `dot` 4-acc, `dot.h:40`. Too small stalls on RAW; too large overflows the
  f-register file / exceeds `SNRT_NUM_SEQUENCER_INSNS`.
- **FREP body length / rep-count split** — `gemm_fp64.h:204,213`; stagger fields
  (`stagger_cnt,mask`) an extra structural axis for stencil-class kernels.
- **Tile counts** `m/n/k_tiles` — `gemm.h:203-205`; trade TCDM footprint vs DMA
  reload count (FA2 `B_r,B_c`, `flashattention_2_fp32.h:44-56`).
- **`double_buffer` on/off** — `gemm.h:244`; preserve the barrier-count invariant
  (§1.2).
- **STRIDED vs CONTIGUOUS row split** — `sc_st_gemm` `gemm.h:40-68`;
  **`partition_banks`** (bank-conflict avoidance) — `gemm.h:88-95`.
- **`snrt_ssr_repeat` broadcast factor** (`gemm_fp64.h:115,143`) and
  **loop-nest dimensionality** (3D vs 4D) — `gemm_fp64.h:119-143`.

Hard constraints the search must respect: `SNRT_NUM_SEQUENCER_LOOPS` /
`SNRT_NUM_SEQUENCER_INSNS` (`impl/snitch_cluster_cfg.h.tpl:55-56`) bound FREP
body size; the candidate's mcycle ROI must be read on a **compute** hart and
exclude the DM hart from FPU-utilization denominators.
