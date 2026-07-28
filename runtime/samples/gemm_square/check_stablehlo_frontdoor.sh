#!/usr/bin/env bash
# StableHLO front-door regression gate: prove that compiling the gemm from a bare
# StableHLO input (gemm_square.stablehlo.mlir, no baked lowering_config, tiling
# selected from gemm_square_config.json) yields a device object byte-identical to
# the hand-tuned linalg sample (gemm_square.mlir). Uses --compile-to=hal, so no
# IREE VM/EmitC host module is emitted -- the deployment path for the minimal host.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
IREE_COMPILE=${QUIDDITCH_IREE_COMPILE:?set QUIDDITCH_IREE_COMPILE to the iree-compile path}
XDSL_OPT=${QUIDDITCH_XDSL_OPT:?set QUIDDITCH_XDSL_OPT to the xdsl-opt path}
TOOLCHAIN_ROOT=${QUIDDITCH_TOOLCHAIN_ROOT:?set QUIDDITCH_TOOLCHAIN_ROOT}
CFG_HEADER=${QUIDDITCH_CFG_HEADER:?set QUIDDITCH_CFG_HEADER to the snitch_cluster_cfg.h path}
OBJDUMP=${QUIDDITCH_OBJDUMP:?set QUIDDITCH_OBJDUMP to a Snitch llvm-objdump}

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

compile() { # <in.mlir> <out.o> [extra flags...]
  "$IREE_COMPILE" \
    --iree-input-type=auto --iree-input-demote-f64-to-f32=0 \
    --iree-hal-target-backends=quidditch \
    --iree-quidditch-static-library-output-path="$2" \
    --iree-quidditch-xdsl-opt-path="$XDSL_OPT" \
    --iree-quidditch-toolchain-root="$TOOLCHAIN_ROOT" \
    --iree-quidditch-cluster-cfg-header="$CFG_HEADER" \
    "${@:3}" \
    --compile-to=hal "$1" -o /dev/null
}

# Strip addresses + the objdump file-header lines; matmul_like is the same op as matmul.
norm() { "$OBJDUMP" -d --mattr=+xdma,+xssr,+xfrep "$1" \
           | grep -vE 'file format|^/|:[[:space:]]*$' \
           | sed -E 's/^[0-9a-f ]+://; s/matmul_like/matmul/g'; }

compile "$HERE/gemm_square.mlir"            "$WORK/linalg.o"
compile "$HERE/gemm_square.stablehlo.mlir"  "$WORK/stablehlo.o" \
        --iree-quidditch-config-table="$HERE/gemm_square_config.json"

if diff <(norm "$WORK/linalg.o") <(norm "$WORK/stablehlo.o") >/dev/null; then
  echo "PASS: StableHLO front-door device object is byte-identical to the linalg sample"
else
  echo "FAIL: StableHLO front-door diverged from the linalg sample" >&2
  diff <(norm "$WORK/linalg.o") <(norm "$WORK/stablehlo.o") | head >&2
  exit 1
fi
