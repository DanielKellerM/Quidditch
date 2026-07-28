#!/usr/bin/env bash
# Flow->HAL materialization gate (S2/2b): prove that materializing a hal.executable
# directly from flow.executable (quidditch-materialize-executable-from-flow) yields a
# device object byte-identical to IREE's full Stream+MaterializeInterfaces path.
#
# The pass is device-only: it drops the host dispatch callers, so nothing references
# the export and IREE's executable-sources->hal pipeline (which still runs symbol-DCE)
# would prune it. To exercise codegen through that harness we splice the reference
# host func back as a liveness anchor -- the standalone quidditch-compile pipeline
# will instead omit executable-pruning, so no anchor is needed there.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
IREE_COMPILE=${QUIDDITCH_IREE_COMPILE:?set QUIDDITCH_IREE_COMPILE to the iree-compile path}
QUIDDITCH_OPT=${QUIDDITCH_OPT:?set QUIDDITCH_OPT to the quidditch-opt path}
XDSL_OPT=${QUIDDITCH_XDSL_OPT:?set QUIDDITCH_XDSL_OPT to the xdsl-opt path}
TOOLCHAIN_ROOT=${QUIDDITCH_TOOLCHAIN_ROOT:?set QUIDDITCH_TOOLCHAIN_ROOT}
CFG_HEADER=${QUIDDITCH_CFG_HEADER:?set QUIDDITCH_CFG_HEADER to the snitch_cluster_cfg.h path}
OBJDUMP=${QUIDDITCH_OBJDUMP:?set QUIDDITCH_OBJDUMP to a Snitch llvm-objdump}

SHAPES=(gemm_square.stablehlo.mlir gemm_square_4x4.stablehlo.mlir gemm_bias.stablehlo.mlir)

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

common=(--iree-input-type=auto --iree-input-demote-f64-to-f32=0
        --iree-hal-target-backends=quidditch
        --iree-quidditch-xdsl-opt-path="$XDSL_OPT"
        --iree-quidditch-toolchain-root="$TOOLCHAIN_ROOT"
        --iree-quidditch-cluster-cfg-header="$CFG_HEADER"
        --iree-quidditch-config-table="$HERE/gemm_square_config.json")

norm() { "$OBJDUMP" -d --mattr=+xdma,+xssr,+xfrep "$1" \
           | grep -vE 'file format|^/|:[[:space:]]*$' \
           | sed -E 's/^[0-9a-f ]+://; s/matmul_like/matmul/g'; }

rc=0
for shlo in "${SHAPES[@]}"; do
  in="$HERE/$shlo"
  "$IREE_COMPILE" "${common[@]}" \
    --iree-quidditch-static-library-output-path="$WORK/full.o" \
    --compile-to=hal "$in" -o /dev/null
  "$IREE_COMPILE" "${common[@]}" --compile-to=flow "$in" -o "$WORK/flow.mlir"
  "$QUIDDITCH_OPT" --quidditch-materialize-executable-from-flow \
    "$WORK/flow.mlir" -o "$WORK/mat.mlir"
  "$IREE_COMPILE" "${common[@]}" --compile-to=executable-sources "$in" -o "$WORK/ref.mlir"
  # Splice the reference host func onto the materialized module as a DCE liveness anchor.
  awk '/^  util.func public @/{k=1} k{print} k&&/^  }$/{exit}' "$WORK/ref.mlir" > "$WORK/host.mlir"
  grep -v '^[[:space:]]*$' "$WORK/mat.mlir" | head -n -1 > "$WORK/spliced.mlir"
  cat "$WORK/host.mlir" >> "$WORK/spliced.mlir"
  echo "}" >> "$WORK/spliced.mlir"
  "$IREE_COMPILE" "${common[@]}" \
    --iree-quidditch-static-library-output-path="$WORK/split.o" \
    --compile-from=executable-sources --compile-to=hal "$WORK/spliced.mlir" -o /dev/null
  if diff <(norm "$WORK/full.o") <(norm "$WORK/split.o") >/dev/null; then
    echo "PASS: $shlo Flow->HAL materialization byte-identical to the full path"
  else
    echo "FAIL: $shlo Flow->HAL materialization diverged" >&2
    diff <(norm "$WORK/full.o") <(norm "$WORK/split.o") | head >&2
    rc=1
  fi
done
exit $rc
