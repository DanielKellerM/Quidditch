# Snitch autotuner — what's missing for memory-bound ops (adversarially re-verified)

> **Verification status of this rewrite.** Every file:line below was opened against `/home/dankeller/Projects/Quidditch` at commit `c169f83` (`KERNELS.md:36-50` is the committed wrong text). Claims that the source confirms are marked **[verified]**; claims I could *not* confirm from a file in the repo are marked **[UNVERIFIED]** with what would be needed to confirm them. The single biggest change from the prior draft: the "8192 builds/runs" reproduction numbers and the "N=1024/4096 flat, ~0.58 cyc/elem" measurements **are not in the repo** and are demoted to UNVERIFIED.

## Correction first

The memory model committed in `c169f83` (`KERNELS.md:36-50`) is **wrong**. **[verified]**

**What is actually true:** the harness A/B/C operand arrays are file-scope `static iree_alignas(64) elem_t` `.bss` buffers (`gen_harness.py:177-179` elementwise, `:119-121` matmul). `base.ld:104-114` maps `.bss` to `>L3` (in fact `base.ld` maps *every* output section — `.text/.rodata/.data/.bss/.dram` — to `>L3`, `base.ld:16-127`), and L3 is `ORIGIN=0x80000000 LENGTH=0x80000000` = **2 GB** (`memory.ld:7`, identical in both `runtime/snitch_cluster/rtl/memory.ld` and `snitch_cluster/sw/runtime/impl/memory.ld`). So the operands are **L3-resident, not TCDM-resident**. There is no link-time TCDM cap on them. **[verified]**

The kernel's *real* TCDM working set is allocated at **runtime** and operand tiles are DMAed L3→TCDM, computed, and DMAed back. **Important caveat the prior draft got wrong:** the file it cited for this — `snitch_cluster/sw/kernels/blas/axpy/src/axpy.h` (`snrt_l1_next()` at `:82`, DMA-in at `:145-147`, DMA-out at `:169`) — is the **hand-written reference kernel**, *not* the kernel the autotuner's `ewadd` op actually runs. The autotuner harness drives the **IREE+xDSL-generated** dispatch by querying the generated library symbol directly (`harness.c.in:54-58`, `dma_fn`/`compute_fn` at `:91-104`), passing only the L3 pointers as `binding_ptrs={A,B,C}` (`gen_harness.py:185-186`). The generated dispatch does its own L3→TCDM staging via the `quidditch.dma` ops. So `axpy.h` illustrates the *pattern* (it is the documented reference for elementwise, `SNITCH_KERNEL_NOTES.md:135`) but is **not on the autotuner's path**; cite it only as the reference idiom, not as the executed code. **[verified]**

The only real size gate in the harness is the **int32 host-reference overflow guard** (`gen_harness.py:172-173` elementwise, `:97-100` matmul), which an 8192-element ewadd does not trip (max value `2L+1` of a sum stays well under 2³¹). **[verified]**

**The "8192 fails to build / ~96KB cap" claim is a red herring** — there is no static-bss TCDM cap; the buffers live in 2 GB of L3. **[UNVERIFIED reproduction numbers]**: the prior draft asserted a specific reproduction (`harness.o` 10396 B, ELF 673836 B, `.bss` at `0x8000d480` size `0x3480c`, `dispatch_cycles=8612 errors=0/8192 SUCCESS`). **None of those artifacts exist in the repo** and I did not re-run the build/sim, so I cannot stand behind those exact numbers. What *is* file-grounded: nothing in `gen_harness.py`/`base.ld`/`memory.ld` would reject an 8192-element f64 ewadd, and 192 KB of operands is ~0.009% of the 2 GB L3 window. To actually confirm "builds and runs", run `direct_build.py` on an 8192-element ewadd spec and sim it; treat the win/no-win as open until then.

**The real reason memory-bound ops are flat is BANDWIDTH-BOUND, not a build cap.** A pure f64 elementwise add has **zero data reuse**, so total DMA volume is a fixed **3N** (read A, read B, write C) regardless of `l1_tiles`. This is an **analytic argument [sound]**, not a literal line in `cost_model.py`: that file is a **GEMM ranker** (default problem `(16,16,16)`, `cost_model.py:72`), whose DMA term `dma_total` (`:97-101`) is the per-tile A+B+C tile volume of a *matmul*, and whose constants are `DMA_BPC=64.0` at **`cost_model.py:60`** and `DMA_FIX=30` at **`cost_model.py:67`** (the prior draft's `:28,:40` are the docstring mentions, not the code). The model has **no elementwise/3N path at all**; the "volume is constant under tiling" claim is the author's reasoning about a no-reuse op, correct on its face, but **not measured by this file**. **[partially verified — argument sound, file cite corrected]**

`body = max(compute_total, dma_total) if dual_buffer else compute_total + dma_total` is at **`cost_model.py:104`** **[verified]**. The inference "`dual_buffer` is inert for a pure add because `max(compute≈0, dma)=dma=sum`" is sound *in the limit of one tile*. But the **measured `dual_buffer` behavior is more nuanced than the prior draft claimed**: in `results_ewadd.tsv` `dual_buffer` is **inert only at the untiled point** (tile=256, `256_dbtrue == 256_dbfalse == 767`), and is **NOT inert at tiled points** (tile=128: `864` vs `1049`; tile=64: `1132` vs `1423`). So even for a pure add, `dual_buffer` helps once you tile — it overlaps the per-tile launch/barrier overhead, not zero compute. The clean statement is: *at the optimum (don't tile), `dual_buffer` is moot; tiling is pure overhead, and `dual_buffer` only recovers part of the overhead it introduces.* **[verified — and it corrects the prior "inert" framing]**

**Major data correction.** `results_ewadd.tsv` contains **one problem size only**, M=256, swept over tiles {64,128,256} × {dbtrue,dbfalse} (6 rows total). The spread is **767 → 1423 = 1.86×** *across tile choices at fixed N=256* (`256_dbtrue`=767 best vs `64_dbfalse`=1423 worst), **not** "a lone N=256 spread with N=1024/4096 flat." The prior draft's claims — "cycles linear in N, asymptotic slope ~0.58 cyc/elem (1024→4096)", "flat across `l1_tiles` AND `dual_buffer` at N=1024/4096" — have **no data backing in the repo**; there is no ewadd measurement at 1024 or 4096 anywhere (`grep` of all `results*.tsv` confirms). `KERNELS.md:45` itself merely *asserts* "1024 and 4096 are flat" with no result file. The `~0.58` figure coincides only with an unrelated `gemm_plain` speedup column (`results_gemm_plain.tsv:40`). **Treat the multi-size linear-bandwidth story as UNVERIFIED**; the only solid measurement is the single-size tile sweep (767→1423, dual_buffer moot at the untiled optimum). **[UNVERIFIED — corrected]**

## What's missing, ranked by leverage

### 1. FUSION — eliminate the elementwise DMA round-trip (highest leverage, biggest gap)

- **Lever:** epilogue fusion — keep a matmul/conv producer's output in TCDM and consume it by the elementwise/activation op in the *same* dispatch, killing the L3 write-back + re-read.
- **Why it helps:** the only way to actually reduce 3N volume for a memory-bound op is to *not move it*. Fused into its producer, the intermediate never leaves TCDM. Tiling/buffering cannot touch volume; fusion can.
- **Can the autotuner pull it today? No.**
  - **[verified]** The Quidditch target implements only `buildConfigurationPassPipeline` (`QuidditchTarget.cpp:177`) and `buildTranslationPassPipeline` (`:196`); a grep of that file for `DispatchRegion|FormDispatch|dispatch-region` finds **nothing**. The plugin therefore operates on **already-formed dispatches** — dispatch-region formation (and hence cross-op fusion) is decided upstream in IREE Flow, above the plugin. The *location* of the fusion decision is verified by absence.
  - **[verified]** `TensorTile.cpp` fuses **producers within a dispatch only**: `collectTiledAndFusedOps` walks producer operands (`:40-59`) and the SCF tile-and-fuse `controlFn` (`:170-187`) fuses producers of the tiled root — it cannot pull in an op Flow placed in a *separate* dispatch.
  - **[verified]** `spec.py:78-81` rejects any kernel with >1 non-yield `linalg` op as "multi-dispatch"; `V1_OPS = {matmul, matmul_transpose_b, elementwise}` (`spec.py:23`). A fused matmul+bias+act is one dispatch with 2+ linalg ops → rejected.
  - **[verified]** Single-op config is **`matmul_transpose_b`-only**, narrower than "matmul-only": `setRootConfig` (`ConfigureForSnitch.cpp:94-113`) has a `Case<linalg::MatmulTransposeBOp>` (`:97-111`) and `.Default(success())` (`:112`) — **no `Case` for plain `linalg::MatmulOp`** and none for elementwise. Any non-`matmul_transpose_b` root gets **no `LoweringConfigAttr`** from this pass, so `TensorTile`'s L1 walk (which only collects ops carrying a `LoweringConfigAttr`, `TensorTile.cpp` `case TilingLevel::L1`) skips it for L1 tiling/double-buffering. (The autotuner currently sidesteps this for `gemm_plain`/`gemm_tall`/`ewadd` by **injecting an inline `lowering_config` into the .mlir** via `direct_build._inject` (`direct_build.py:114-118`) / `sweep.py:101`, which `TensorTile` L1 then picks up — that is why those ops sweep despite no `setRootConfig` Case.)
- **[ASSERTED / UNVERIFIED — the load-bearing open question]:** *Does IREE Flow actually form a fused matmul+bias+act dispatch for these ops today?* This rewrite verifies only that *if* such a dispatch exists, the Quidditch plugin would see it whole and currently reject it. Whether Flow produces it is **not tested anywhere in the repo** — and the project's own `KERNELS.md:31` lists the change as "multi-op reference + **confirm IREE fuses bias+act epilogue into the matmul dispatch**", i.e. it is explicitly an open item, not a confirmed behavior. Do not claim IREE "forms fused dispatches today" as fact; it is unverified.
- **Concrete change:** (a) confirm/secure Flow-level epilogue fusion so the intermediate stays in TCDM; (b) relax `spec.py:78-81` so a fused dispatch is admissible; (c) add a `setRootConfig` `Case` (+ seed-table entry) for plain matmul / fused / memory-bound roots so they receive a tunable config rather than falling through `.Default`.
- **Effort:** large. (a) is an upstream-IREE concern that first needs *verifying*, not just coding; (b)/(c) are days each but inert without (a).

### 2. Tuning compute-bearing activations (expf/tanh) where `dual_buffer`/overlap pays

- **Lever:** `dual_buffer` 3-stage overlap (`body=max(compute,dma)`, `cost_model.py:104`), already a `LoweringConfigAttr` field and already swept.
- **Why it helps:** unlike a pure add, an activation (sigmoid/tanh/gelu/softmax via libm `tanh`/`expf`, `SNITCH_KERNEL_NOTES.md:137-138`) does many scalar FPU cycles per element. When `compute ≈ dma`, `max(compute,dma)` is a *genuine* overlap win.
- **Can the autotuner pull it today? Knob yes, target no.** **[verified]** `dual_buffer` is wired end to end: field at `QuidditchSnitchAttrs.td:58`; consumed by `PipelineCopyComputePass` (`QuidditchTarget.cpp:221`); swept at `sweep.py:73`; injected at `direct_build.py:118`; modeled at `cost_model.py:104`. But an activation never reaches a configured root (Lever 1: `setRootConfig` configures `matmul_transpose_b` only), so for a *standalone* activation the overlap knob has nothing to attach to — unless an inline `lowering_config` is hand-injected. It also needs the **fp-tolerance Tier-2 gate** (transcendentals aren't int-exact; `gen_harness.py` raises `SpecError` for non-f64 today, `spec.py:98-99`).
- **Concrete change:** the `setRootConfig` `Case` from 1c (give activations a config), plus enabling the Tier-2 fp gate for these ops.
- **Effort:** medium — overlap machinery exists; config-table + gate plumbing, riding on 1c.

### 3. DMA-bandwidth knobs (burst / channels / multi-DMA)

- **Lever:** wider/parallel DMA or coalesced transfers to raise effective B/cyc or amortize per-transfer launch latency.
- **Why it helps:** for a bandwidth-bound op, raising effective bandwidth or cutting per-transfer overhead is the only lever besides fusion that moves the floor.
- **Exposed? No — absent from the IR.** **[verified]** `DMAOps.td` ops (`DMA_StartTensorCopyOp` `:15`, `DMA_StartTransferOp` `:126`, `DMA_StartZeroMemTransferOp` `:158`, `DMA_WaitForTransferOp` `:182`) carry only source/dest/pad — no channel/burst/queue-width operand. The SnitchDMA dialect adds only `SnitchDMA_StatOp` (`SnitchDMAOps.td:17-29`, "id of the last completed transfer"). Bufferization lowers each copy to a bare `StartTransferOp`/`WaitForTransferOp` pair (`QuidditchTarget.cpp:237-243`). The cost model treats bandwidth as a fixed constant (`DMA_BPC=64`, `DMA_FIX=30`; `cost_model.py:60,67`), not a search axis.
- **Concrete change:** add channel/burst/transfer-count attributes to the DMA ops, lower to the multi-channel snitch DMA, surface as a tunable; add a variable-bandwidth cost term.
- **Effort:** large. Lower priority than fusion; gates Lever 6.

### 4. SSR-unroll / strided-vs-contiguous / bank-partition — not in `lowering_config`

- **Lever:** `unroll=8`, contiguous-vs-strided SSR access, TCDM bank partitioning — the knobs the hand-written reference kernels use (`SNITCH_KERNEL_NOTES.md:108` `unroll=8`; STRIDED vs CONTIGUOUS row split at `:133-135,259`).
- **Why it helps:** these set steady-state FPU/SSR throughput and bank conflicts — the *compute* side of `max(compute,dma)`, so they matter for activations (Lever 2) and any op whose compute approaches the DMA wave.
- **Exposed? No.** **[verified]** `LoweringConfigAttr` has exactly four parameters — `workgroup_tiles`, `l1_tiles`, `l1_tiles_interchange`, `dual_buffer` (`QuidditchSnitchAttrs.td:54-58`). No unroll/SSR/bank field. These are fixed inside the xDSL `test-lower-linalg-to-snitch` lowering (the cost model assumes 1 FMA/cyc/core steady, `cost_model.py:30-34`). The autotuner's only reach into the kernel lowering is the **opaque** `--iree-quidditch-xdsl-passes` pass-pipeline string (`QuidditchTarget.cpp:109`, threaded as `xdsl_passes` in `direct_build.py:148,197`): it can swap whole passes but cannot pass a numeric unroll/bank value, because the xDSL pass exposes none. **[note]** `workgroup_tiles` is a declared field but **dormant** — `setRootConfig` seeds it to zeros (`ConfigureForSnitch.cpp:99`) and distribution is auto-derived from `compute_cores=8` (`TensorTile.cpp:65-111`). **[UNVERIFIED term]** "`partition_banks`" is presented as a hand-kernel knob; the literal token `partition_banks` does not appear in `SNITCH_KERNEL_NOTES.md` — bank *alignment* idioms do (e.g. `axpy.h:13-16,98-101`). Phrase it as "bank-alignment/partition idioms" rather than a named knob.
- **Concrete change:** add parametric `unroll`/bank options to the xDSL snitch-lowering pass, surfaced as new `lowering_config` fields or typed pass options, then sweep.
- **Effort:** medium-large.

### 5. The whole-model path (profiler producer + apply-wiring) — required because fusion is a model-compile decision

- **Lever:** tune memory-bound ops *in context*, at model scale, where fusion exists and the L3 round-trip is real.
- **Why it helps:** the in-TCDM single-dispatch harness cannot represent a fused multi-dispatch model (`spec.py:78-81`), so a memory-bound win is observable only on the whole-model path.
- **What exists [verified]:** `profile_model.to_profile()` (`profile_model.py:43-53`) → `orchestrate.cmd_plan()` (`orchestrate.py:154`); and the no-rebuild override `--iree-quidditch-config-table` (`QuidditchTarget.cpp:114`, `Passes.td:32`) → `ConfigureForSnitch.applyConfigTable()` keyed on the live dispatch symbol (`ConfigureForSnitch.cpp:57-92`, called at `:104`), forwarded by `buildConfigurationPassPipeline` (`QuidditchTarget.cpp:188-191`: `opts.configTable = targetOptions.configTable`).
- **What's missing [verified]:**
  - **Profile producer** — nothing sims a real model and turns the trace into `{dispatch_name: cycles}`; `profile_model.py:16-22` says so explicitly (the cycle table is the *input*; producing it is "the remaining Phase-3 instrumentation"). `gen_trace.py` dumps per-region `tstart/tend/cycles` (`gen_trace.py:68,735`) but regions are mcycle-boundary spans with **no dispatch symbol**, so the region→`main_dispatch_<N>_<op>_<shape>_f<bits>` grouping `DISPATCH_RE` (`profile_model.py:30`) needs has no producer. The region→symbol map is the hard part.
  - **Apply-loop wiring** — `apply.py:33-66` only regex-rewrites an inline `lowering_config` in a hand-written `.mlir` (`:45-49`); it does **not** upsert the config-table JSON and explicitly flags the generated-model path as "not done" (`apply.py:10-13`). The seed table is empty (`kSeedConfigTable="{}"`, `ConfigureForSnitch.cpp:53`; the old pre-v3.11.0 keys were dropped, `:48-52`). The model build never passes the flag — `nsnet2/CMakeLists.txt` calls `quidditch_module(SRC ... DST nsnet2)` with **no `FLAGS`** (lines 3-4), though `quidditch_module.cmake:165` (`set(_COMPILER_ARGS ${_RULE_FLAGS})`, with `FLAGS` parsed at `:146`) *would* forward them.
  - **Driver glue** — `orchestrate.py` is bookkeeping; `cmd_next` (`orchestrate.py:99-120`) prints a decision but invokes no tune/build/apply. Nothing connects next → sweep → config-table upsert → model rebuild → re-profile.
- **[correction]** The prior draft flagged a "latent key-format coupling: `profile_model.DISPATCH_RE` still matches the legacy `_transpose_b` form (a hazard)." That is **not a hazard** — `DISPATCH_RE` is **deliberately** written to match both the live `main_dispatch_*` and the legacy `main$async_dispatch_*..._transpose_b` forms (`profile_model.py:9-11,29`). The real, file-grounded inconsistency is elsewhere: `ConfigureForSnitch.cpp:44` documents config-table keys as the live `main_dispatch_<N>_<op>_<MxNxK>_f<bits>` form, while `apply.py:13` still describes them as the stale `main$async_dispatch_*` form. Align `apply.py`'s comment (and any key it would emit) to the live `ConfigureForSnitch.cpp:44` form. Missing keys silently fall back to defaults (`ConfigureForSnitch.cpp:77-79`).
- **Concrete change:** (a) sim→`gen_trace`→region-to-dispatch grouper emitting the cycle table (the region→symbol map is the load-bearing piece); (b) `apply.py` upserts `best_*.json` into a config-table JSON keyed on the live symbol; (c) pass that file via `nsnet2`'s `quidditch_module(FLAGS …)`; (d) orchestrator glue.
- **Effort:** large; (a) and (b) are the two load-bearing pieces.

### 6. A bandwidth cost model vs the gemm roofline (lowest leverage, enabling)

- **Lever:** a roofline scoring memory-bound ops on bytes/cyc against the DMA floor, parallel to the compute roofline `cost_model.py` uses for gemm.
- **Why it helps:** lets the autotuner *recognize* a memory-bound op is at its bandwidth ceiling and stop sweeping inert knobs; gives a target for Levers 3/4. It scores; it doesn't move cycles.
- **Exposed? Partial.** **[verified]** `cost_model.py` has the constants (`DMA_BPC=64`, `DMA_FIX=30`; `:60,67`) and a (gemm-tile) DMA-volume term (`:97-101`), but treats bandwidth as fixed and has **no elementwise model, no roofline classifier, and no variable-bandwidth term**.
- **Concrete change:** add an arithmetic-intensity / bytes-per-cyc roofline term; classify ops compute- vs memory-bound; gate which knobs to sweep.
- **Effort:** small-medium (model-only), low payoff until 3/4 land.

## Corrected KERNELS.md "Harness scope" note

Replacement for `KERNELS.md:36-50`:

```markdown
## Harness scope: compute-bound tunes in-harness, memory-bound is bandwidth-bound

The harness A/B/C operand arrays are file-scope `static` `.bss` buffers, and
`base.ld` maps `.bss` (and every section) to `>L3` (2 GB at 0x80000000,
`memory.ld:7`), so they are **L3-resident, not TCDM-resident** — there is no
link-time TCDM cap on them. The generated IREE/xDSL dispatch allocates its real
TCDM working set at runtime and DMAs operand tiles L3->TCDM, computes, DMAs
results back (the harness just hands it the L3 pointers; `axpy.h` is the
reference idiom, not the executed code). So harness buffer size is independent of
TCDM capacity; the only real size gate is the int32 host-reference overflow guard
(`gen_harness.py:172-173`). Consequence, measured:
- **Compute-bound ops** (matmul family) show a real tiling spread at the in-TCDM
  size (gemm 2x) — reuse + L1-staging make the small size representative.
  **Harness-tune these.**
- **Memory-bound ops** (pure-streaming elementwise, no reuse) are
  **bandwidth-bound**, not build-capped. Total DMA volume is a fixed 3N (read A,
  read B, write C) regardless of `l1_tiles`, so tiling cannot lower bytes moved
  and cannot move the floor. Measured (ewadd, M=256, the only size swept): the
  optimum is "don't tile" — at tile=256 a single tile means `dual_buffer` is
  moot (`256_dbtrue == 256_dbfalse == 767`); the 767->1423 (1.86x) spread is
  ENTIRELY small-tile overhead across {64,128,256}, where `dual_buffer` then
  recovers part of the overhead it introduces (e.g. tile=64: 1132 vs 1423), NOT
  a tiling/bandwidth win. The harness proves *correctness* for these ops, not a
  perf win — and no in-harness tiling will produce one, because the volume is
  fixed. (Multi-size scaling at 1024/4096 is asserted, not yet measured — add the
  sizes to the sweep to confirm the linear-bandwidth story.)
- **The actual lever for memory-bound ops is FUSION**: keep a matmul/conv
  producer's output in TCDM and consume it by the elementwise/activation op in
  the same dispatch, killing the L3 round-trip. Fusion is an upstream IREE Flow
  dispatch-formation decision (above the Quidditch plugin), so it is a
  whole-model-compile concern, not a standalone-harness one — and whether Flow
  forms that fused dispatch for these ops today is still to be CONFIRMED, not
  assumed. Compute-bearing activations (expf/tanh) are the in-between case:
  there `dual_buffer` overlap is a real win, but the op must first reach a
  configured root (`ConfigureForSnitch` configures matmul_transpose_b only today)
  and be measured under the Tier-2 fp gate.
```

## Smallest first step

**Add a `setRootConfig` `Case` (+ one seed-table entry) for the activation op class in `ConfigureForSnitch.cpp:94-113`**, so a compute-bearing activation (tanh/expf) gets a `LoweringConfigAttr` instead of falling through `.Default(success())` (`:112`) into no-config / no-L1-tiling (`TensorTile.cpp`, `case TilingLevel::L1`).

Why this and not fusion: fusion is the bigger win but is an upstream-IREE concern that first has to be *verified* (KERNELS.md:31 lists it as unconfirmed), so there is no quick path. The activation config `Case` is a few lines, purely additive, and is the *single prerequisite* that unlocks the already-wired `dual_buffer` overlap knob (Lever 2) on a target that genuinely benefits — turning the first compute-bearing memory-bound improvement from "no knob to attach to" into a measurable sweep. It also forces the Tier-2 fp gate to be enabled for that op, the natural next small step. (All file paths are absolute under `/home/dankeller/Projects/Quidditch/`.)