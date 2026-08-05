# PyTorch → StableHLO → gwaihir

Status: a real PyTorch `nn.Module` (2-layer MLP) compiles end-to-end to a gwaihir
streaming Snitch device object, and its computation is numerically validated against
the torch golden. This exercises the StableHLO front-door beyond gemm — the front-end
fuses the `nn.Linear` weight transposes + bias `broadcast_in_dim` + relu into the
matmul dispatches, and the xDSL backend streamifies each (SSR/FREP).

## The chain

```
PyTorch nn.Module
  → torch.export + torch_xla.stablehlo   (StableHLO text; see export_mlp.py)
  → iree-compile --iree-input-type=auto --iree-hal-target-backends=quidditch
        --iree-quidditch-cluster-cfg-header=<gwaihir snitch_cluster_cfg.h>   (L1 base from QUIDDITCH_L1_BASE)
        --iree-quidditch-config-table=<per-dispatch tiling>
        --compile-to=hal --iree-quidditch-static-library-output-path=mlp.o
  → mlp.o : per-layer matmul_16x16x16 dispatches, each $xdsl_kernel* (streamed)
```

## Exporting the model (toolchain)

torch_xla wheels require glibc ≥2.29; the el8 build nodes have glibc 2.28, so the
export runs in a container (apptainer, no host hacks):

```
apptainer exec --bind /scratch/<you> docker://python:3.11-slim bash -c '
  python -m venv /scratch/<you>/cvenv
  /scratch/<you>/cvenv/bin/pip install --upgrade pip
  /scratch/<you>/cvenv/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
  /scratch/<you>/cvenv/bin/pip install torch_xla==2.6.0
  PJRT_DEVICE=CPU /scratch/<you>/cvenv/bin/python export_mlp.py'
```

`export_mlp.py` (deterministic — `manual_seed`, so the reference is regenerable) emits
`torch_mlp.stablehlo.mlir`, the harness weight header `mlp_ref_bits.h`, and
`mlp_in{0..4}_*.npy` / `mlp_golden.npy` in @main binding order. It self-checks that the
emitted bit-pattern weights recompute the golden, so the committed pair can't drift.
`MLP_OUT` overrides the output dir (defaults to this sample dir).

## Numerical validation (CPU reference)

Compile the StableHLO to `llvm-cpu` and check against the torch golden with a matched
iree-compile / iree-run-module pair (the device `.o` stays byte-exact-consistent via
the front-door gate; the Snitch codegen numerics are covered by the gemm validation):

```
iree-compile --iree-input-type=auto --iree-input-demote-f64-to-f32=0 \
  --iree-hal-target-backends=llvm-cpu torch_mlp.stablehlo.mlir -o mlp_cpu.vmfb
iree-run-module --module=mlp_cpu.vmfb --function=main \
  --input=@mlp_in0_b2.npy ... --input=@mlp_in4_x.npy --expected_output=@mlp_golden.npy
# -> [SUCCESS] all function outputs matched their expected values.
```

## On-device multi-dispatch validation (single Snitch cluster)

`mlp_harness.c` runs the two dispatches back-to-back on one cluster (no IREE VM/HAL),
threading dispatch_0's output `h1` into dispatch_1's input. The compute cores run the
real f64 kernels; the DM core (no fp ARITHMETIC) seeds the actual torch weights as f64
bit patterns (`mlp_ref_bits.h`) via integer stores and dumps the fp output, which
`compare_mlp.py` tolerance-checks against the torch golden (`mlp_golden.npy`). Each build
covers one `YOFFSET` stride — the default `mlp_harness` target is offset 0 (64/256 elems);
rebuild with `-DYOFFSET=1..3` for the other strides, then pass all four logs together for
full coverage: `compare_mlp.py mlp_golden.npy <log0> <log1> <log2> <log3>` (a partial set
reports INCOMPLETE, not SUCCESS).

Verified on the Verilator single-cluster sim, all 8 compute cores identical per the
per-hart traces: FPU (520 fp-offload ops/core), SSR (enable + config CSRs), FREP
(sequencer loops), double buffering (`dual_buffer=true`) — and the device output matches
the torch golden to **1.1e-16** (f64 last-bit), 0 mismatches across all 16 rows.

## On the full gwaihir SoC (headless, no CVA6 host)

Done via nimbus: `gen_qcs_offload_image.py` derives a QCS job (2 dispatches, h1 threaded
between them) from the model's Flow IR, and the parametrized `qcs_replay` firmware replays
it on the SoC RTL under PRELMODE=5. Result: `done=144/144 fail=0`, and all 256 output `y`
elements match the torch golden to f64 last-bit (max_abs_err 1.7e-16, 0/256 mismatches).
Single-buffered on the SoC (double buffering is
blocked there by the parked Cheshire-AXI duplicate-R-last bug; clean on the standalone cluster).

## Remaining

- Per-dispatch tiling is hand-written (`mlp_config.json`); a real model wants a
  ConfigureForSnitch default.
- Broader ops (softmax/conv/layernorm) + non-16×16 shapes, each gated by xDSL coverage.
