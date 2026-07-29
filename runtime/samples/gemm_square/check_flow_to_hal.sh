#!/usr/bin/env bash
# Flow->HAL materialization gate (S2/2b): the device object from the pass alone
# (quidditch-materialize-executable-from-flow) is byte-identical to the full path. The
# pass is device-only, so IREE's executable-sources->hal harness would DCE-prune the
# unreferenced executable; the reference host func is spliced back as a liveness anchor.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/gate_lib.sh"
IREE_COMPILE=${QUIDDITCH_IREE_COMPILE:?set QUIDDITCH_IREE_COMPILE to the iree-compile path}
QUIDDITCH_OPT=${QUIDDITCH_OPT:?set QUIDDITCH_OPT to the quidditch-opt path}
XDSL_OPT=${QUIDDITCH_XDSL_OPT:?set QUIDDITCH_XDSL_OPT to the xdsl-opt path}
TOOLCHAIN_ROOT=${QUIDDITCH_TOOLCHAIN_ROOT:?set QUIDDITCH_TOOLCHAIN_ROOT}
CFG_HEADER=${QUIDDITCH_CLUSTER_CFG_HEADER:?set QUIDDITCH_CLUSTER_CFG_HEADER to the snitch_cluster_cfg.h path}

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

common=(--iree-input-type=auto --iree-input-demote-f64-to-f32=0
        --iree-hal-target-backends=quidditch
        --iree-quidditch-xdsl-opt-path="$XDSL_OPT"
        --iree-quidditch-toolchain-root="$TOOLCHAIN_ROOT"
        --iree-quidditch-cluster-cfg-header="$CFG_HEADER"
        --iree-quidditch-config-table="$HERE/gemm_square_config.json")

rc=0
for shlo in "${GATE_SHAPES[@]}"; do
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
