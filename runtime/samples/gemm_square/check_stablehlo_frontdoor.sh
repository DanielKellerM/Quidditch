#!/usr/bin/env bash
# StableHLO front-door regression gate: for each gemm shape, prove that compiling
# from a bare StableHLO input (no baked lowering_config, tiling selected from
# gemm_square_config.json keyed by the dispatch symbol) yields a device object
# byte-identical to the hand-tuned linalg sample. Uses --compile-to=hal, so no
# IREE VM/EmitC host module is emitted -- the deployment path for the minimal host.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/gate_lib.sh"
IREE_COMPILE=${QUIDDITCH_IREE_COMPILE:?set QUIDDITCH_IREE_COMPILE to the iree-compile path}
XDSL_OPT=${QUIDDITCH_XDSL_OPT:?set QUIDDITCH_XDSL_OPT to the xdsl-opt path}
TOOLCHAIN_ROOT=${QUIDDITCH_TOOLCHAIN_ROOT:?set QUIDDITCH_TOOLCHAIN_ROOT}
CFG_HEADER=${QUIDDITCH_CFG_HEADER:?set QUIDDITCH_CFG_HEADER to the snitch_cluster_cfg.h path}

# (StableHLO sample, linalg reference) pairs to check.
PAIRS=(
  "gemm_square.stablehlo.mlir gemm_square.mlir"
  "gemm_square_4x4.stablehlo.mlir gemm_square_4x4.mlir"
)

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

rc=0
for pair in "${PAIRS[@]}"; do
  read -r shlo linalg <<<"$pair"
  compile "$HERE/$linalg" "$WORK/linalg.o"
  compile "$HERE/$shlo"   "$WORK/shlo.o" --iree-quidditch-config-table="$HERE/gemm_square_config.json"
  if diff <(norm "$WORK/linalg.o") <(norm "$WORK/shlo.o") >/dev/null; then
    echo "PASS: $shlo == $linalg (byte-identical device object)"
  else
    echo "FAIL: $shlo diverged from $linalg" >&2
    diff <(norm "$WORK/linalg.o") <(norm "$WORK/shlo.o") | head >&2
    rc=1
  fi
done
exit $rc
