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
        --iree-quidditch-cluster-cfg-header=<gwaihir snitch_cluster_cfg.h>   (L1_BASE=0x30000000)
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

`export_mlp.py` emits `torch_mlp.stablehlo.mlir` plus a numerical reference
(`mlp_ref.h`, and `mlp_in{0..4}_*.npy` / `mlp_golden.npy` in @main binding order).

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

## Remaining toward on-gwaihir deployment

- On-device numerics for the Snitch `.o` (single-cluster Verilator harness, like gemm).
- Multi-dispatch sequencing on the QCS host: a model issues N sequential dispatches with
  intermediate L2 buffers; gemm was one kernel.
- Per-dispatch tiling is hand-written today; a real model wants a ConfigureForSnitch default.
- Broader ops (softmax/conv/layernorm) + non-16×16 shapes, each gated by xDSL coverage.
