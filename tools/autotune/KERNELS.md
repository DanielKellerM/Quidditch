# Kernel library — the ops the autotuner targets

The op set we want to implement + optimise on the Snitch cluster, ranked by
relevance to the workloads we run (NsNet2 = GRU speech enhancement; general ML).
Each row states the **reference gate** it needs (how Tier-1/Tier-2 prove
correctness), its **autotuner status**, and the **concrete change** to add it.

Which of these to optimise *first* on a real model is an Amdahl question — that
is what the whole-model profiler answers (sim → per-dispatch cycles → rank).
This list is the menu; the profile picks the order for a given model.

See `SNITCH_KERNEL_NOTES.md` for the Snitch runtime/kernel reference: the
core-gating API (how to keep work off the FPU-less DM core), the SSR/FREP +
double-buffer kernel idioms, the per-op reference-kernel mapping, and the
compute-core fp-reference unblock (gating the Tier-1 reference to a compute core
to admit nonlinear/large-K ops — sound, but needs a post-dispatch rendezvous).

## Status legend
- ✅ tunable today (single-dispatch, divisor tiles, f64)
- 🧩 needs a harness/scope widening (no codegen work)
- 🔨 needs compiler/codegen work (xDSL or IREE) before it can be tuned

| Op | Shows up in | Reference gate | Status | Blocker / change needed |
|----|-------------|----------------|--------|-------------------------|
| matmul / matmul_transpose_b | FC layers, GRU/LSTM gate projections | int-exact (Tier-1) | ✅ | none |
| elementwise add/mul/sub | residuals, gate combines, bias | int-exact (Tier-1) | ✅ | `ops/ewadd` (1-D, 256); tunable (see order §1) — model as 1-D since IREE collapses N-D before tiling |
| relu / max | activations | int-exact (Tier-1) | 🧩 | as elementwise; piecewise-linear stays integer-exact |
| sigmoid / tanh | GRU/LSTM gates | **fp-tolerance (Tier-2)** | 🧩 | the fp-tolerance Tier-2 keystone (host rel+abs tol by dtype) |
| matvec (M=1 GEMV) | every RNN/GRU timestep | fp-tolerance (Tier-2) | 🧩 | M=1 currently rejected; int32 ref overflows large K → must go through the fp gate, not the int gate |
| reduction (sum/max) | softmax, pooling | int-exact / fp | 🧩 | 1D-output reference; tiling over the reduced axis |
| **fused matmul + bias + act** | the FC block | fp-tolerance (Tier-2) | 🔨 | multi-op reference + confirm IREE fuses bias+act epilogue into the matmul dispatch |
| **GRU / LSTM gate cell** | NsNet2 core | fp-tolerance (Tier-2) | 🔨 | the nsnet2 compile blocker (xDSL `zip()` crash + `util.assume.int`); the highest-value fusion |
| layernorm / softmax | transformers | fp-tolerance (Tier-2) | 🔨 | reduction + broadcast fusion; later |
| conv1d / conv2d | CNN front-ends | int-exact / fp | 🔨 | im2col vs direct; later |

## Harness scope: compute-bound tunes in-harness, memory-bound is bandwidth-bound
The harness A/B/C operand arrays are file-scope `static` `.bss` buffers, and
`base.ld` maps `.bss` (and every section) to `>L3` (2 GB at 0x80000000,
`memory.ld:7`), so they are **L3-resident, not TCDM-resident** — there is no
link-time TCDM cap on them. The generated IREE/xDSL dispatch allocates its real
TCDM working set at runtime and DMAs operand tiles L3→TCDM, computes, DMAs
results back (the harness just hands it the L3 pointers; `axpy.h` is the
reference idiom, not the executed code). So harness buffer size is independent of
TCDM capacity; the only real size gate is the int32 host-reference overflow guard
(`gen_harness.py:172-173`). Consequence:
- **Compute-bound ops** (matmul family) show a real tiling spread at the in-TCDM
  size (gemm 2×) — reuse + L1-staging make the small size representative.
  **Harness-tune these.**
- **Memory-bound ops** (pure-streaming elementwise, no reuse) are
  **bandwidth-bound**, not build-capped. Total DMA volume is a fixed 3N (read A,
  read B, write C) regardless of `l1_tiles`, so tiling cannot lower bytes moved or
  move the floor. Measured (ewadd, M=256, the committed sweep): the optimum is
  "don't tile" — at tile=256 a single tile makes `dual_buffer` moot
  (`256_dbtrue == 256_dbfalse == 767`); the 767→1423 (1.86×) spread is ENTIRELY
  small-tile overhead across {64,128,256}, where `dual_buffer` only recovers part
  of the overhead it introduces (tile=64: 1132 vs 1423) — NOT a tiling/bandwidth
  win. A throwaway run measured 1024→1201, 4096→2977, flat across tiles
  (consistent with the bandwidth story; not committed as a results file). The
  harness proves *correctness* for these ops, not a perf win — no in-harness
  tiling will produce one, because the volume is fixed.
- **The actual lever for memory-bound ops is FUSION**: keep a matmul/conv
  producer's output in TCDM and consume it by the elementwise/activation op in
  the same dispatch, killing the L3 round-trip. Fusion is an upstream IREE Flow
  dispatch-formation decision (above the Quidditch plugin), so it is a
  whole-model-compile concern, not a standalone-harness one — and whether Flow
  forms that fused dispatch for these ops today is **still to be CONFIRMED**, not
  assumed. Compute-bearing activations (expf/tanh) are the in-between case:
  `dual_buffer` overlap is a real win there, but the op must first reach a
  configured root (`ConfigureForSnitch` configures `matmul_transpose_b` only
  today) and be measured under the Tier-2 fp gate. See `MEMORY_BOUND_GAP.md`.

## Widening order (dependency-sequenced)
1. **elementwise (add/mul)** — DONE and TUNABLE (`ops/ewadd`, `autotuner-design`).
   Modeled as a **1-D** op (256 flattened elements, `l1_tiles` length 1): IREE
   collapses an N-D elementwise to one parallel loop *before* `TensorTile`, so a
   2-D `[Mt,Nt]` config is dimensionally mismatched post-collapse and silently
   dropped (that was the earlier "inert knob" — a spec-model bug, not a codegen
   gap). With the 1-D model the knob is live: sweep spread **767 (untiled, best)
   → 1423** (1.86×), and `dual_buffer=true` beats false at every tile (it overlaps
   the DMA waves). Optimum is "don't tile" here — 256 elements fit TCDM, so tiling
   is pure overhead. Tier-1 gate is live (an external `mulf`-vs-`addf`-reference
   mutation → all-elements error; in the normal sweep reference and kernel are
   co-derived and in sync, so it catches real kernel miscompiles).
2. **relu/max** — same int-exact path; first "activation".
3. **fp-tolerance Tier-2 (keystone)** — harness dumps raw output, host computes
   rel+abs tol by dtype/K. Unblocks *everything* nonlinear and the int32-overflow
   shapes in one move: sigmoid/tanh, matvec/M=1, and f32/f16.
4. **sigmoid / tanh** — fp; the GRU gate nonlinearities.
5. **matvec / M=1** — drop the `1 in shape` reject once it rides the fp gate (the
   int reference overflows for real K; rv32 has no `fcvt.l.d` for int64 widening).
6. **fused matmul+bias+act** — the first fusion; the real per-layer win (no L2
   round-trip of the intermediate). Needs a multi-op reference.
7. **GRU/LSTM gate** — depends on fixing the nsnet2 GRU codegen gap; then it is a
   real compiling model to profile end-to-end.

## Fused ops — where the wins are
Fusion avoids staging intermediates through L2/HBM. Priority targets:
- matmul → +bias → +activation (FC block): one dispatch instead of three.
- GRU/LSTM gate: the recurrent cell as one fused kernel (the NsNet2 hotspot).
- matmul → layernorm / softmax epilogue (transformers).

The autotuner tunes a fused op the same way as a single op (it is one dispatch
with one `lowering_config`); the work is (a) getting IREE+xDSL to *form* the
fused dispatch and (b) a multi-op host reference for the Tier-1/Tier-2 gates.

## Invariants every new op must keep
- A new op **rejects cleanly** at `load_spec`/`gen_harness` until its reference
  gate exists — never emit a harness that can't prove correctness (no vacuous
  gate; this is why elementwise/f32 raise today rather than mis-build).
- DM core (hart_00008) has **no FPU** — the in-harness Tier-1 reference stays
  integer (`fcvt.w.d` + int mul/add). All floating-point checking is host-side
  (Tier-2). This is why nonlinear/large-K ops need the fp-tolerance Tier-2, not
  a richer in-harness reference.
