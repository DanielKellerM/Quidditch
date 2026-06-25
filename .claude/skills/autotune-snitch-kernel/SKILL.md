---
name: autotune-snitch-kernel
description: >-
  Optimize a Quidditch/xDSL Snitch kernel's dispatch cycles by RTL simulation,
  across abstraction layers: L1 tiling (l1_tiles, dual_buffer, interchange), the
  xDSL pass pipeline, and agent-authored xDSL passes. Two drivers -- a
  deterministic cost-model-ranked grid sweep, and an LLM-agent-driven search
  (propose -> evaluate -> keep/revert) -- plus multi-kernel Amdahl orchestration,
  all behind sound correctness gates. Use when optimizing a single-dispatch f64
  matmul / matmul_transpose_b .mlir kernel or improving its xDSL lowering.
---

# Autotune / optimize a Snitch/Quidditch kernel

This skill drives `tools/autotune/` to lower a kernel's `dispatch_cycles` while
preserving correctness. The oracle is a standalone harness that drives the
IREE-generated Quidditch dispatch directly (no VM/HAL) on the Verilator RTL sim;
correctness is gated by Tier-1 (per-element exact, in-harness) **and** Tier-2 (an
independent off-toolchain host cross-check). You can drive it two ways:

- **Deterministic** (`sweep.py`): enumerate the knob grid, cost-model-rank it,
  build+sim in parallel, keep the best. Exhaustive on the validated knobs.
- **Agent-driven** (`agent_loop.py` + `author_pass.py`, following
  `agent/program.md`): *you* are the search policy — propose configs, pipeline
  variants, or hand-authored xDSL passes; the gates keep only correct + faster.

## Scope (v1)

A **single-dispatch, static-shape, f64 `matmul` / `matmul_transpose_b`** kernel
(one `linalg` op). Tile choices must divide M,N,K. Rejected loudly (do not work
around): multi-dispatch, dynamic shape, M/N/K = 1 (matvec), non-f64, elementwise.
Note: Tier-2 is an EXACT cross-check; an fp-*tolerance* gate for f32/f16 is not
built, so non-f64 stays out of scope.

## Prerequisites

Run from the Quidditch repo root (these are the normal autotuner environment):
`build-rt/` populated; `venv/bin/ninja`; the snitch `iree-compile`, `.venv/bin/xdsl-opt`,
and the colluca LLVM toolchain (paths hardcoded in `direct_build.py`); the sim
`snitch_cluster/target/sim/build/bin/snitch_cluster.vlt`. If a path is wrong for
this checkout, fix the constant rather than guessing around it.

## The knobs (increasing scope / risk)

1. `l1_tiles = [M,N,K]` (divisors of M,N,K) and `dual_buffer`.
2. `l1_tiles_interchange` — **leave at [2,0,1]**; other orders SILENTLY MISCOMPILE
   at multi-tile shapes (the gate catches them as FAIL, but don't chase them).
3. **xDSL pass pipeline** (`--iree-quidditch-xdsl-passes`, threaded as `--passes`)
   — Group-B pass selection/ordering.
4. **Agent-authored xDSL pass** (`author_pass.py`) — invent a novel structure by
   writing a `ModulePass`; registered + run + gated + reverted automatically.

## Workflow

### 1. Scaffold the op-spec
```bash
python3 tools/autotune/scaffold_op.py <name> <path/to/kernel.mlir>
```
Derives `entry`, `dtype`, `shape`, `transpose_b`, tile choices from the `.mlir`,
writes `ops/<name>.toml`, validates it (out-of-scope kernels rejected here —
report, don't patch). **Review `module`** (the `quidditch_module` DST, default
`<name>_mod`). `ops/gemm_tall.toml` is a worked example.

### 2a. Deterministic sweep
```bash
python3 tools/autotune/sweep.py --op <name>          # full grid (16-way parallel)
python3 tools/autotune/sweep.py --op <name> --budget K   # sim only the top-K cost-ranked
```
3-tile kernels are 3³×2 = 54 configs (~90 s); more divisors → up to 5³×2 = 250,
so use `--budget` to bound runtime (the model only sequences; the optimum is never
pruned). gemm_square also runs a dual-build canary (direct == CMake).

### 2b. Agent-driven search (you, following `agent/program.md`)
```bash
python3 tools/autotune/agent/agent_loop.py --op <name> --propose "16,16,16,true"
python3 tools/autotune/agent/agent_loop.py --op <name> --propose "16,16,16,true" --passes "<pipeline>"
python3 tools/autotune/agent/author_pass.py --op <name> --pass-file <authored_pass.py>
python3 tools/autotune/agent/agent_loop.py --op <name> --status
```
Read `agent/program.md` first — it has the hardware model (8 compute + 1 no-FPU DM
core, FREP/SSR ~0.87 ceiling, iDMA, dual_buffer), the Amdahl-ordered tiers, and
the keep/revert rules. Each proposal goes through the cost-model pre-rank + RTL
sim + Tier-1/Tier-2; a faster-but-wrong proposal is reverted, never kept.
`author_pass.py` lets you write a `ModulePass` (template: `example_passes/agent_noop.py`)
— it registers it in xdsl, runs it in the pipeline, gates it, and reverts the
submodule (broken → rejected; semantics-changing → caught).

### 3. Read the result
`best_<name>.json` (winning `tag`/`config`=`[M,N,K,dual_buffer,interchange]`/`cycles`)
and `results_<name>.tsv` (full grid); or `agent_loop.py --status` (best + history).
Only `correct=True` configs are eligible; `legal=False`/`correct=False` are excluded.

### 4. Apply the win (record -> apply)
```bash
python3 tools/autotune/apply.py --op <name> --dry-run   # preview the lowering_config change
python3 tools/autotune/apply.py --op <name>             # write best_<name>.json into the .mlir
```
This deposits `best_<name>.json`'s `lowering_config` (l1_tiles/dual_buffer/interchange)
back into the kernel `.mlir` -- the win lands in version-controlled source (commit
it + re-run the IREE build to deploy). HAND-WRITTEN kernels only. Model kernels'
tiling lives in `ConfigureForSnitch.cpp`'s per-dispatch table keyed on the
`main$async_dispatch_*` name the tuner does not yet capture -- the remaining apply
gap. Pipeline / authored-pass wins persist by editing the `Passes.td` default
pipeline / committing the xdsl pass to the submodule, then rebuilding iree-compile.

### Multi-kernel / whole-model (Amdahl)
`agent/profile_model.py` turns a per-dispatch cycle table into the orchestrator's
profile JSON; `agent/orchestrate.py plan|next|record` (vendored from AutoKernel,
MIT) schedules tuning by Amdahl cycle-share. Producing real per-dispatch cycles
from a whole-model run (sim nsnet2 + `gen_trace.py` region→dispatch grouping,
DM-core excluded) is the remaining mechanical step.

## Correctness guarantee

EVERY avenue — tile config, interchange, pass pipeline, or an authored pass —
flows through the same gates: Tier-1 per-element exact (structured asymmetric
integer inputs; the generator refuses aliasing/symmetric inputs) + Tier-2
independent host FNV cross-check (`tier2.py`). A faster-but-WRONG result is
reverted, never kept or reported. **Never modify** the harness, the `read_csr(mcycle)`
ROI, or the dispatch markers (frozen — that's gaming the metric). No workarounds:
report rejections; if the kernel is already near the roofline, say so and stop.

## Files

`tools/autotune/`: `spec.py` (op-spec + scope gate), `scaffold_op.py`,
`gen_harness.py` + `harness.c.in` (generated harness, Tier-1 + the `-DHARNESS_DUMP_OUTPUT`
hash), `direct_build.py` (ninja-free build + `compile_harness` + the `xdsl_passes`
knob), `cost_model.py`, `sweep.py`, `tier2.py` (Tier-2), `apply.py` (write a recorded
win back into the .mlir), `ops/<name>.toml`.
`tools/autotune/agent/`: `program.md` (playbook), `agent_loop.py` (search loop),
`author_pass.py` + `example_passes/` (authored xDSL passes), `orchestrate.py`
(MIT, Amdahl) + `profile_model.py`, `LICENSE.autokernel`.
