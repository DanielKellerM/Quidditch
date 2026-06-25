# Generate the Quidditch/xDSL codegen target from the Snitch cluster cfg

*Design note — iDMA/Quidditch autotuner team. Scope: the **default** cfg (`snitch_cluster/cfg/default.json`) as the worked example, not a general multi-cfg generator.*

> **STATUS UPDATE — read the Coverage table below as the source of truth; the
> prose after it is the original PRE-implementation design and is now partly
> historical.** Implemented since the design:
> - **cost_model** `NCORE` + `DMA_BPC` are cfg-derived (`c920543`, `a0366d4`).
> - **compiler**: ALL the cfg HW params are now READ from the header into a
>   `ClusterParams` struct and THREADED into the `ExecutableTargetAttr` config dict
>   (`3a88d02` compute_cores, `664a6ee` the rest) — `compute_cores` is CONSUMED; the
>   rest (`tcdm_bytes`, `sequencer_loops/insns`, `supports_ssr/frep`) are threaded
>   PLACEHOLDERS whose consumers are `TODO(cfg-target)` (grep that tag in
>   `QuidditchTarget.cpp`). So the body's "baked literal 8 / no symbol read back /
>   nothing threaded" framing is the PRE-`3a88d02`/`664a6ee` state — NOT current.
> Still open: wiring the flag into the build so it is passed automatically, and
> wiring each placeholder's CONSUMER (the `TODO(cfg-target)` sites).
> **Line-number cites below have drifted** (`664a6ee` added the ~35-line struct):
> `compute_cores` attr is now `~:247` (not `:170`), `l1MemoryBytes` `~:150` (not
> `:87`), the attr block `~:233-258`. Treat all body cites as approximate.

## Coverage: cfg params the codegen target derives (and the TODO placeholders)

We deliberately did NOT wire every cfg variable yet — only the load-bearing
`compute_cores` (default cfg, as scoped). The rest are **flagged future work**, not
forgotten (mirrored by a `TODO(cfg-target)` block at `QuidditchTarget.cpp`'s
`compute_cores` site):

| cfg param | cfg key → symbol | status | where it must land |
|---|---|---|---|
| compute cores | `nr_cores − dm_core_num` → `CFG_CLUSTER_NR_CORES`/`SNRT_CLUSTER_DM_CORE_NUM` | **DONE** | compiler `compute_cores` attr (`3a88d02`) + cost_model `NCORE` |
| DMA bytes/cycle | `dma_data_width` (512) | **PARTIAL** | cost_model `DMA_BPC` done; compiler not (not emitted to the header — needs the 1-line tpl add in the snitch_cluster submodule) |
| TCDM size | `tcdm.size` (128 kB) → `SNRT_TCDM_SIZE` | **TODO** | compiler `l1MemoryBytes` — blocked on the `100000`-vs-`112640` DMA-stack-overflow workaround; fix that first |
| TCDM banks / width | `tcdm.banks` (32), `data_width` (64) → `SNRT_TCDM_BANK_{NUM,WIDTH}` | **TODO** | bank-partition / conflict-avoidance knob (not yet a knob) |
| ISA + FPU | `isa` (rv32imafd), `Xdiv_sqrt` (false) | **TODO** | codegen must refuse `fdiv`/`fsqrt` (today implicit) + gate FP by isa |
| SSR / FREP present | `xssr`/`xfrep` → `SNRT_SUPPORTS_{SSR,FREP}` | **TODO** | select the xDSL SSR/FREP pass pipeline (today a fixed `xDSLPasses` string) |
| sequencer bounds | `num_sequencer_{loops,insns}` (2/32) → `SNRT_NUM_SEQUENCER_*` | **TODO** | enforce the FREP body-size/nesting limit (today unchecked — see the audit) |
| FPU pipeline depth | `timing.lat_comp_fp64` (3) | **TODO** | xDSL `memref_stream_interleave` `pipeline_depth` (today a literal 4) |

Plus: **wire the `--iree-quidditch-cluster-cfg-header` flag into the build** so it
is passed automatically, and the reuse-the-header path for the params above.
⚠️ **Measured (A/B, `gemm_square` @ 16,16,16, 2026-06-23):** passing the flag on the
default cfg is a **code no-op** (`.text` disassembly byte-identical → cycles
identical) but **NOT a binary no-op**. IREE pretty-prints the full target attr into a
`"HAL device __device_0 not found"` diagnostic cstring, and the `664a6ee` placeholders
(`sequencer_*`, `supports_*`, `tcdm_bytes`) grow that `.rodata` string ~70 chars
(a `$d`/`.L0` marker shifts `0x120`→`0x122`). It survives stripping, so wiring the
flag **breaks the stripped-ELF canary** (`direct_build.py:169`) for **zero functional
gain on the default cfg** (`compute_cores` is `8` either way: `9−1` = the fallback).
So wiring is gated on a decision: either rebaseline the canary, or defer until a
placeholder CONSUMER lands / a non-default cfg is actually run (where
`compute_cores ≠ 8` is the first thing that genuinely changes codegen).

## The problem

One file describes the cluster hardware: `snitch_cluster/cfg/default.json`. `util/clustergen/clustergen.py` reads it (json5 + `jsonref` `$ref` resolution + schema-default injection; e.g. `nr_cores` is *computed* as `len(cores)` at `util/clustergen/cluster.py:308`, not present in the JSON), builds a `SnitchCluster`, and renders Mako templates into **two** artifacts that stay in lockstep with the RTL:

- the RTL package (`snitch_cluster_pkg.sv.tpl` → core count, `WideDataWidth`, `IsaCfg[]`, FPU impl);
- the runtime header (`sw/runtime/impl/snitch_cluster_cfg.h.tpl` → `CFG_CLUSTER_NR_CORES`, `SNRT_TCDM_SIZE`, `SNRT_SUPPORTS_*`, …).

The **codegen target** — the IREE Quidditch HAL plugin (`QuidditchTarget.cpp`), the xDSL lowering, and the autotuner `cost_model.py` — holds its own view of the *same* hardware (core count, TCDM size, DMA width, ISA, FPU/SSR/FREP). That view is **hand-copied and hardcoded**. `cost_model.py:27-28` documents in a comment the cfg keys it mirrors (`NCORE`, `DMA_BPC`), then bakes the literals further down (`NCORE = 8` at `:59`, `DMA_BPC = 64.0` at `:60`).

**Concrete drift risk.** Nothing wires the compiler/tuner literals to `default.json`. If the cluster is reconfigured (a different core count, a wider/narrower DMA port, a different TCDM size) the RTL and runtime header regenerate, but the compiler keeps tiling for the old hardware:

- The workgroup/thread divisor is the `compute_cores` attribute, set to the literal `8` at `QuidditchTarget.cpp:170-171` (with a `// TODO: Ideally shouldn't be hardcoded.` at `:165`). `TensorTile.cpp:65-111` reads it back via `getConfiguration().getAs<IntegerAttr>("compute_cores")` (`:68`) and does `divideCeil(largestParallelSize, <attr>)` (`:109-110`). The runtime harness, meanwhile, derives the *true* count via `snrt_cluster_compute_core_num()` — so the **compile-time tiling and the runtime team size can silently disagree**, producing under-/over-subscribed workgroups with no error.
- `l1MemoryBytes = 100000` at `QuidditchTarget.cpp:87` is *already internally inconsistent*: the comment (`:86`) and the pass default (`Passes.td:53`, `"112640"`) say `112640`, but the option default was lowered to `100000` to dodge a DMA-stack overflow (`// TODO: This should actually be 112640 but DMA stack overflows. Ooopsie!`, `:86`). If `tcdm.size` changes, neither literal follows.
- `cost_model.py` would then mis-rank tilings (wrong core-distribution divisor, wrong DMA roofline) while reporting the same Spearman ρ on the stale ground-truth set.

The fix is to make the codegen target *consume the cfg* the way the runtime header already does, rather than re-state it.

## Ground-truth cfg params (default)

Resolved via `clustergen.py` (json5 → `$ref` → schema defaults → Mako). Header line numbers are from the committed `snitch_cluster_cfg.h`; symbols/values verified.

| Param | cfg key | default value | RTL/runtime symbol it generates |
|---|---|---|---|
| Total cores | `cluster.nr_cores` (computed `len(cores)` = 8×`compute_core_template` + 1×`dma_core_template`, `default.json:68-76`; `cluster.py:308`) | 9 | `CFG_CLUSTER_NR_CORES 9` (`cfg.h:12`); `SNRT_CLUSTER_CORE_NUM` (`cfg.h:15`) |
| DM cores | (count of `dma_core_template`) | 1 | `SNRT_CLUSTER_DM_CORE_NUM 1` (`cfg.h:17`) — **literal `1` in the tpl** (`cfg.h.tpl:46`), not derived |
| Compute cores | `nr_cores − DM_CORE_NUM` | 8 | **no symbol** — runtime subtraction `snrt_cluster_compute_core_num()` = `snrt_cluster_core_num() − snrt_cluster_dm_core_num()` (`team.h:125-126`) |
| Narrow data width | `cluster.data_width` (`default.json:11`) | 64 | `SNRT_TCDM_BANK_WIDTH = data_width/8 = 8` (`cfg.h:18`; tpl `:47`); RTL `NarrowDataWidth` |
| DMA data width | `cluster.dma_data_width` (`default.json:20`) | 512 (= 64 B/cyc) | RTL `WideDataWidth` only — **never emitted to the runtime header** (no `dma_data_width`/`SNRT_DMA_*-width` line in `cfg.h` or `cfg.h.tpl`) |
| TCDM size | `cluster.tcdm.size` (kB) | 128 | `SNRT_TCDM_SIZE 0x20000` (= 128·1024) (`cfg.h:22`; tpl `:51`) |
| TCDM banks | `cluster.tcdm.banks` | 32 | `SNRT_TCDM_BANK_NUM 32` (`cfg.h:19`) |
| Sequencer loops | `hives[0].cores[0].num_sequencer_loops` (`default.json:113`) | 2 | `SNRT_NUM_SEQUENCER_LOOPS 2` (`cfg.h:26`; tpl `:55`) |
| Sequencer insns | `hives[0].cores[0].num_sequencer_instructions` (`default.json:112`) | 32 | `SNRT_NUM_SEQUENCER_INSNS 32` (`cfg.h:27`; tpl `:56`) |
| SSR layout | `compute_core_template.ssrs` (`default.json:119-123`) | 3 (2 indirection masters + 1 slave), `ssr_intersection` | per-core `IsaCfg[].Xssr`; runtime `SNRT_SUPPORTS_SSR` |
| Feature macros | OR-reduce per-core `xdma`/`xssr`/`xfrep`/`xcopift` (`cfg.h.tpl:31-35`) | all true | `SNRT_SUPPORTS_{DMA,SSR,FREP,COPIFT}` (`cfg.h:34-40`; tpl `:67-85`) |

**FPU / ISA resolution (the DM-core-FPU truth).** Both core classes carry `isa: rv32imafd` (`default.json:96,126`), so **the DM core gets a full FPU** — `IsaCfg[].RVF=1`, `RVD=1` derived straight from the parsed isa string (`snitch_cluster_pkg.sv.tpl:244-245`, `int(getattr(c['isa_parsed'], 'f'/'d'))`). There is **no** `fpu`/`Xf` enable and no per-core FPU-present gate. `fdiv.d`/`fsqrt.d` trap because the **separate** schema key `Xdiv_sqrt` defaults **false** (`snitch_cluster.schema.json:515`, "iterative FPU … disabled by default") and is omitted from *both* core templates. That makes `snitch_cluster_pkg.sv.tpl` emit the FPnew DIVSQRT unit as `fpnew_pkg::DISABLED` via the `% if c["Xdiv_sqrt"]:` gate (`pkg.sv.tpl:193-207`, MERGED branch untaken → DISABLED branch at `:201-206`) and set `IsaCfg[].XDivSqrt = int(c['Xdiv_sqrt']) = 0` (`pkg.sv.tpl:254`) on **every** core (compute and DM alike). So div/sqrt FP is illegal **cluster-wide**, not via a DM-core FPU disable — adding `Xdiv_sqrt: true` is what would change it. *(Author-verified this checkout: `schema.json:515`, `pkg.sv.tpl:193-207`/`:240-256`.)*

## Where the codegen hardcodes each (the drift surface)

| Param | hardcoded at (file:line) | cfg key it mirrors | runtime-derived (safe) vs baked literal (gap) |
|---|---|---|---|
| compute cores (tiling divisor) | `QuidditchTarget.cpp:170-171` (literal `8` into the `compute_cores` attr); read at `TensorTile.cpp:65-111`, `LowerForallOp.cpp:34-37`, `PadToTilingConfig.cpp:404-406` | 8×`compute_core_template` | **baked literal** — the load-bearing gap; disagrees with the runtime `snrt_cluster_compute_core_num()` count |
| compute cores (runtime team) | `harness.c.in:84`, `runtime/samples/gemm_square/harness.c:93` (`dispatch_state.max_concurrency = snrt_cluster_compute_core_num()`) | — | **runtime-derived (safe)** via `snrt_cluster_compute_core_num()` (`team.h:125-126`); *not* linked to the compile-time literal above |
| TCDM / L1 budget | `QuidditchTarget.cpp:87` (option default `l1MemoryBytes = 100000`); `Passes.td:53` default `"112640"`; consumed `LowerL1Allocations.cpp:52,110` | `tcdm.size = 128 kB` | **baked literal**, *double-baked and self-inconsistent* (100000 used vs 112640 documented); plumbed by value (`QuidditchTarget.cpp:266-268`), not via the attr |
| DMA bytes/cycle | `cost_model.py:60` (`DMA_BPC = 64.0`); roofline `:99-100` | `dma_data_width = 512` | **baked literal**; cfg value lives *only* in `default.json:20`, emitted to no header |
| compute cores (ranking) | `cost_model.py:59` (`NCORE = 8`); used `:93,:107` | 8×`compute_core_template` | **baked literal**; derivable from existing header (see first step) |
| f64 elem size | `cost_model.py:61` (`F64 = 8`) | f64-only V1 scope (`gemm_square.mlir` operands are `tensor<…xf64>`, `:8`) | **baked literal**; not fed from `spec.py` |
| ISA / SSR / FREP selection | implicit in `xDSLPasses` default string `QuidditchTarget.cpp:82` (`"arith-add-fastmath,test-lower-linalg-to-snitch"`), shelled to `xdsl-opt` by `ConvertToRISCV.cpp:127-130` | per-core `xssr`/`xfrep` (`SNRT_SUPPORTS_*`) | **baked into the pass name** — no capability flag on the xDSL path |
| FPU pipeline depth | `memref_stream_interleave.py:252` (`pipeline_depth: int = field(default=4)`) | `timing.lat_comp_fp64 = 3` (`default.json:39`) (+issue) | **baked literal** (xDSL); unroll bound `factor < pipeline_depth*2` (`:241-243`) *(cfg link is inferred, not proven)* |
| SSR stride-pattern rank | `test_lower_linalg_to_snitch.py:49` (`MemRefStreamTileOuterLoopsPass(target_rank=4)`) | SSR datapath dim count | **baked literal** *(medium confidence — the cfg link is inferred, not proven)* |
| sequencer body/nesting limit | `convert_riscv_scf_for_to_frep.py:57-75` (validity check is on op *type* via `is_valid_frep_body_op`, **not** on body op count or nest depth) | `num_sequencer_{insns,loops} = 32/2` | **unchecked assumption** — the only check is op-type validity; a body > 32 insns or nesting > 2 is emitted but illegal on this cfg *(medium confidence)* |
| K-outermost interchange | `ConfigureForSnitch.cpp:101` (`{2, 0, 1}`) and `spec.py:66,124` (`[2, 0, 1]`) | structural, not a cfg numeric | **baked literal** in two places; couples cost-model `KSPLIT` to codegen |
| SSR stream-register count | `snitch_allocate_registers.py:29-39` (`riscv.Registers.FT[index]`) | 3 SSRs | **derived (safe)** from op operand count — but does *not* enforce the cfg's 3-SSR limit |
| fitted ranking weights | `cost_model.py:64-69` (`TILE_SETUP=180`, `DMA_FIX/BARRIER/FIXED/KSPLIT/MSPLIT`) | — | not cfg mirrors; fitted to a 9-point gemm_square ground truth (`:116-126`), ±40%-insensitive for ranking → **low drift risk** |

## Design: cfg → codegen target

**Principle: reuse the existing cfg→header generation before adding any new generator.** `snitch_cluster_cfg.h.tpl` already Mako-reads `default.json` and emits `#define`s; it is built by the `make` rule `sn_cluster_gen_rule` (defined `make/common.mk:98-99`, whose prerequisite line `$(1): $(SN_CFG) …` is `:99`; registered for the cfg header at `make/sw.mk:39`). It is the *same artifact the C runtime consumes*, so anything readable from it cannot drift from the RTL. Two tiers:

1. **Already emitted as `SNRT_*` → just read them.** Core count (`CFG_CLUSTER_NR_CORES − SNRT_CLUSTER_DM_CORE_NUM`), TCDM size (`SNRT_TCDM_SIZE`), sequencer bounds (`SNRT_NUM_SEQUENCER_{LOOPS,INSNS}`), and the `SNRT_SUPPORTS_{DMA,SSR,FREP}` capability bits all exist today (`cfg.h:12,17,22,26,27,34,36,38`). No generator change.
2. **Not emitted → smallest new plumbing.** `dma_data_width` is emitted nowhere (verified: no DMA-width line in the header or tpl). The clean move is a **one-line tpl extension** — `#define SNRT_DMA_DATA_WIDTH ${cfg['cluster']['dma_data_width']}` — so `DMA_BPC = SNRT_DMA_DATA_WIDTH/8` becomes readable from the same regenerated header. Prefer this over reading the JSON directly in Python, because the header is the one already-regenerated source of truth.

**Where the hardware-params struct lives.** There is **no** unified HW descriptor in the compiler today; facts are scattered across the `compute_cores` attr, the `l1MemoryBytes` option, the implicit `xDSLPasses` string, and `cost_model.py` literals. The IREE-native place to unify them is the existing `IREE::HAL::ExecutableTargetAttr` *configuration* `DictionaryAttr`, populated via `NamedAttrList list` at `QuidditchTarget.cpp:163-174` — the same dict that already carries `compute_cores` (`:170-171`). Materialize a cfg-derived `HardwareParams` there (add `tcdm_bytes`, `dma_bytes_per_cycle`, `bool supports_ssr/supports_frep/supports_dma`), fed by a new `--iree-quidditch-config-json` flag (mirroring the existing `--iree-quidditch-config-table` flag pattern at `QuidditchTarget.cpp:114-119`) pointing at `default.json` or a small derived JSON, parsed once.

Then each consumer reads from one source:
- **`ConfigureForSnitch` / `TensorTile` / `LowerForallOp` / `PadToTilingConfig`** already do `ExecutableTargetAttr::lookup(op).getConfiguration().getAs<IntegerAttr>("compute_cores")` — extend the same lookup to read `tcdm_bytes` (retiring the standalone `l1MemoryBytes` option and its 100000/112640 split).
- **xDSL lowering / `ConvertToRISCV`** selects the SSR/FREP pass pipeline from `supports_ssr`/`supports_frep` in the attr instead of a fixed `xDSLPasses` string.
- **`cost_model.py`** reads `NCORE`/`DMA_BPC` from the generated header (or the same derived JSON) instead of `:59-60` literals.
- **`TensorTile` divisor** then matches `snrt_cluster_compute_core_num()` by construction, closing the compile-vs-runtime gap.

All scoped to the default cfg — `--iree-quidditch-config-json` points at `default.json`; this is *not* a general multi-cfg generator. *(The exact `HardwareParams` extension shape at `QuidditchTarget.cpp:163-174` is a recommendation, medium confidence; the dict's current use for `compute_cores` and the `lookup`/`getConfiguration` mechanism are verified.)*

## Smallest first step

**Make `NCORE` in `cost_model.py` cfg-derived by parsing the already-generated `snitch_cluster_cfg.h`.** This is the single smallest end-to-end change: it touches **one** Python file, needs **zero** generator/template/make edits, and is genuinely end-to-end because the header is the same artifact the C runtime consumes.

Concrete edit — replace the literal at `cost_model.py:59`:

```python
# before
NCORE = 8          # compute cores

# after
def _ncore_from_cfg_header(default=8):
    import os, re
    h = os.path.join(os.path.dirname(__file__),
                     "../../snitch_cluster/sw/runtime/impl/snitch_cluster_cfg.h")
    try:
        txt = open(h).read()
        nr = int(re.search(r"#define\s+CFG_CLUSTER_NR_CORES\s+(\d+)", txt).group(1))
        dm = int(re.search(r"#define\s+SNRT_CLUSTER_DM_CORE_NUM\s+(\d+)", txt).group(1))
        return nr - dm
    except Exception:
        return default      # header absent / unparseable -> documented fallback

NCORE = _ncore_from_cfg_header()   # default cfg: 9 - 1 = 8
```

Why this one: `cost_model.py` lives at `tools/autotune/`, so the `../../` path resolves to `snitch_cluster/sw/runtime/impl/snitch_cluster_cfg.h` (verified present). `CFG_CLUSTER_NR_CORES 9` (`cfg.h:12`) and `SNRT_CLUSTER_DM_CORE_NUM 1` (`cfg.h:17`) are both present today, so `NCORE = 9 − 1 = 8` — **identical** to the current literal on the default cfg, so the `_GROUND_TRUTH` fit (`cost_model.py:116-126`) and the ρ=1.0 validation are unaffected. It demonstrates the read-from-the-generated-header pattern that the rest of the design then generalizes.

**Defer to a second step** (one-line tpl addition, *not* the first step): `#define SNRT_DMA_DATA_WIDTH ${cfg['cluster']['dma_data_width']}` in `snitch_cluster_cfg.h.tpl`, then `DMA_BPC = SNRT_DMA_DATA_WIDTH / 8`. This needs the only genuinely-new plumbing (the cfg→header path carries no DMA width today) and should land after the `NCORE` reader proves out.

---

**Verified against source** (this checkout, 2026-06-23): `cost_model.py:27-28,59-61,64-69,93,99-100,107,116-126`; `QuidditchTarget.cpp:82,86-87,114-119,132,163-174,266-268`; `TensorTile.cpp:65-111`; `LowerForallOp.cpp:34-37`; `PadToTilingConfig.cpp:404-406`; `LowerL1Allocations.cpp:52,110`; `codegen/.../Snitch/Transforms/Passes.td:53`; `ConvertToRISCV.cpp:127-130`; `ConfigureForSnitch.cpp:101`; `snitch_cluster_cfg.h:12,15,17,18,19,22,26,27,34-40`; `snitch_cluster_cfg.h.tpl:31-35,41,46-58,67-85`; `default.json:11,20,39,68-76,95-96,112-113,119-123,125-126`; `snitch_cluster_pkg.sv.tpl:193-207,240-256`; `snitch_cluster.schema.json:515`; `cluster.py:308`; `team.h:125-126`; `harness.c.in:84`; `runtime/samples/gemm_square/harness.c:93`; `gemm_square.mlir:8`; `memref_stream_interleave.py:241-243,252`; `test_lower_linalg_to_snitch.py:49`; `convert_riscv_scf_for_to_frep.py:57-75`; `snitch_allocate_registers.py:29-39`; `spec.py:66,124`; `make/common.mk:98-99`; `make/sw.mk:39`; `snitch.py:155-191` (`ssr_set_dimension_*` ops). **Marked medium-confidence / inferred** (mechanism confirmed, the *cfg numeric link* is not proven by a code path): `target_rank`↔SSR-dim cfg link; `pipeline_depth`↔`lat_comp_fp64` link; the FREP body-size/nesting unchecked-assumption claim; the `QuidditchTarget.cpp:163-174` `HardwareParams` placement recommendation.