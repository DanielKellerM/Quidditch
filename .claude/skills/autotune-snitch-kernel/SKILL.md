---
name: autotune-snitch-kernel
description: >-
  Find the fastest correctness-preserving L1 tiling (l1_tiles + dual_buffer) for
  a Quidditch/xDSL Snitch matmul kernel by RTL simulation. Use when optimizing a
  single-dispatch f64 matmul / matmul_transpose_b .mlir kernel's dispatch cycles
  on the Snitch cluster. Scaffolds an op-spec from the .mlir, sweeps the tiling
  grid (cost-model-ranked, parallel, ninja-free), and reports the winning config.
---

# Autotune a Snitch/Quidditch kernel

This skill drives `tools/autotune/` to search a kernel's L1 tiling space and
return the fastest tiling that still computes the correct result. The oracle is
a standalone harness that drives the IREE-generated Quidditch dispatch directly
(no VM/HAL) on the Verilator RTL sim and reports `dispatch_cycles`; correctness
is a per-element integer-exact check against a host reference.

## Scope (v1)

Supported: a **single-dispatch, static-shape, f64 `matmul` or `matmul_transpose_b`**
kernel (one `linalg` op; explicit `indexing_maps` or a named matmul). Tile choices
must divide M, N, K. Knobs tuned: `l1_tiles = [M,N,K]` and `dual_buffer`.

Out of scope (the tools reject these loudly — do not work around it): multi-op /
multi-dispatch kernels, dynamic shapes, `M`/`N`/`K` = 1 (matvec), `dtype` other
than f64, and elementwise ops. f32/f16 + elementwise need the not-yet-built
host-side fp-tolerance gate (Tier-2).

## Prerequisites

Run from the Quidditch repo root. These must already exist (they are the normal
autotuner environment, not set up by this skill):
- `build-rt/` populated (the IREE/Quidditch runtime built once with the snitch toolchain).
- `venv/bin/ninja` (1.13), the snitch `iree-compile`, `xdsl-opt`, and the colluca LLVM toolchain — paths are hardcoded in `tools/autotune/direct_build.py`.
- The Verilator sim `snitch_cluster/target/sim/build/bin/snitch_cluster.vlt`.

If a build path is wrong for this checkout, fix the constant in `direct_build.py`
/ `sweep.py` rather than guessing around it.

## Workflow

### 1. Describe the kernel as an op-spec

If the kernel does not already have `tools/autotune/ops/<name>.toml`, scaffold it
from the `.mlir`:

```bash
python3 tools/autotune/scaffold_op.py <name> <path/to/kernel.mlir>
```

This derives `entry` (func symbol), `dtype`, `shape`, `transpose_b`, and tile
choices from the `.mlir`, writes `ops/<name>.toml`, and validates it. Most
out-of-scope kernels (multi-dispatch, dynamic shape, matvec, non-divisor tiles,
non-f64) are rejected here with a clear message; a few deeper checks (symmetric
inputs, shape↔maps disagreement) fire when you run the sweep. Either way, report
the rejection — don't patch around it. **Review `module`** in the generated TOML: it is the
`quidditch_module` DST (`#include <<module>.h>`), a build choice not present in
the `.mlir`; the default is `<name>_mod`. `ops/gemm_tall.toml` is a worked example.

### 2. Run the sweep

```bash
python3 tools/autotune/sweep.py --op <name>
```

Enumerates the `l1_tiles × dual_buffer` grid (every tile choice cubed × 2), ranks
it by the analytic cost model, then builds + sims each config through a parallel
ninja-free pipeline (16-way). The example 3-tile kernels are 3³×2 = 54 configs
(~90 s); a kernel divisible by more of {4,8,16,32,64} can reach 5³×2 = 250, so use
`--budget K` to sim only the top-K cost-ranked configs and bound runtime (the model
only sequences; the optimum is never structurally pruned).

For `gemm_square` only, a dual-build canary first asserts the ninja-free build is
stripped-identical to CMake's. Other kernels skip it (they have no CMake target —
the direct build is the only build).

### 3. Read the result

- `tools/autotune/best_<name>.json` — the winning `tag`, `config` (`[M,N,K,dual_buffer,l1_tiles_interchange]`), `cycles`, and `kernel`.
- `tools/autotune/results_<name>.tsv` — every config: `legal`, `correct`, `cycles`, `speedup_vs_base`, `cost_rank`.

Report the best tiling and its speedup over the spec's `baseline_tag`. Only
configs with `correct=True` are eligible; `legal=False` (rejected at compile) and
`correct=False` (output ≠ reference) configs are excluded and worth flagging.

### 4. Apply the winning tiling (if asked)

Edit the kernel `.mlir`'s `lowering_config` to the winning `l1_tiles` and
`dual_buffer`. Re-running the full IREE flow then picks it up (no compiler change).

## Correctness guarantee

Every sim run checks the kernel output **per element** against a host reference
computed from structured, asymmetric integer inputs (exact in int32 — the DM core
has no FPU, so the reference is integer-only). The generator refuses to emit a
harness whose inputs alias or are symmetric, so transpose / index-swap / operand-
swap miscompiles are caught — a config that miscomputes is marked `FAIL` and never
reported as best. This is the gate; do not relax or bypass it.

## Files

`tools/autotune/`: `spec.py` (op-spec loader + scope gate), `scaffold_op.py`
(spec from .mlir), `gen_harness.py` + `harness.c.in` (generated harness + Tier-1
gate), `direct_build.py` (ninja-free per-config build + `compile_harness`),
`cost_model.py` (ranker), `sweep.py` (the driver), `ops/<name>.toml` (specs).
