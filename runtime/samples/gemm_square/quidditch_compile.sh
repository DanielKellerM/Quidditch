#!/usr/bin/env bash
# StableHLO -> Snitch device .o via the fused quidditch-compile binary (no Stream/
# VM/EmitC). `check` proves the object is byte-identical to the full Stream+HAL
# reference path -- the single byte-exact producer (the old shell two-stage
# --compile-to=flow | iree-opt tail was a redundant second path; retired).
set -euo pipefail

# Derive the tool paths from the runtime build's CMakeCache (the single source of
# truth cmake already resolved) when QUIDDITCH_BUILD_DIR is set, so callers can't
# hand-pick the broken venv/ xdsl-opt or a pulp-as-less toolchain. Env vars override.
if [[ -n "${QUIDDITCH_BUILD_DIR:-}" && -f "${QUIDDITCH_BUILD_DIR}/CMakeCache.txt" ]]; then
  _cache="${QUIDDITCH_BUILD_DIR}/CMakeCache.txt"
  _cache_get() { sed -n "s/^$1[^=]*=//p" "$_cache" | head -1; }
  : "${QUIDDITCH_XDSL_OPT:=$(_cache_get XDSL_OPT_PATH)}"
  : "${QUIDDITCH_TOOLCHAIN_ROOT:=$(_cache_get QUIDDITCH_TOOLCHAIN_ROOT)}"
  : "${QUIDDITCH_IREE_COMPILE:=$(_cache_get IREE_COMPILE_PATH)}"
  : "${QUIDDITCH_COMPILE:=$(find "$QUIDDITCH_BUILD_DIR" -maxdepth 4 -name quidditch-compile -type f 2>/dev/null | head -1)}"
fi

COMPILE=${QUIDDITCH_COMPILE:?set QUIDDITCH_COMPILE to the quidditch-compile binary (or QUIDDITCH_BUILD_DIR)}
IREE_COMPILE=${QUIDDITCH_IREE_COMPILE:?set QUIDDITCH_IREE_COMPILE to the iree-compile path (the byte-exact reference)}
XDSL_OPT=${QUIDDITCH_XDSL_OPT:?set QUIDDITCH_XDSL_OPT to the xdsl-opt path}
TOOLCHAIN_ROOT=${QUIDDITCH_TOOLCHAIN_ROOT:?set QUIDDITCH_TOOLCHAIN_ROOT}
CFG_HEADER=${QUIDDITCH_CFG_HEADER:?set QUIDDITCH_CFG_HEADER to the snitch_cluster_cfg.h path}

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/gate_lib.sh"
CONFIG_TABLE=${QUIDDITCH_CONFIG_TABLE:-$HERE/gemm_square_config.json}

qflags=(--iree-quidditch-xdsl-opt-path="$XDSL_OPT"
        --iree-quidditch-toolchain-root="$TOOLCHAIN_ROOT"
        --iree-quidditch-cluster-cfg-header="$CFG_HEADER"
        --iree-quidditch-config-table="$CONFIG_TABLE")

full_hal() { # <in.mlir> <out.o> : the reference object via the full Stream+HAL path
  "$IREE_COMPILE" --iree-input-type=auto --iree-input-demote-f64-to-f32=0 \
    --iree-hal-target-backends=quidditch "${qflags[@]}" \
    --iree-quidditch-static-library-output-path="$2" --compile-to=hal "$1" -o /dev/null
}

quidditch_compile() { # <in.stablehlo.mlir> <out.o> : the fused Stream-free binary
  "$COMPILE" "$1" --iree-input-type=auto --iree-input-demote-f64-to-f32=0 \
    --iree-hal-target-backends=quidditch "${qflags[@]}" \
    --iree-quidditch-static-library-output-path="$2"
}

# Compare each shape's quidditch-compile object to the full Stream+HAL reference.
run_gate() {
  local rc=0
  local work; work=$(mktemp -d); trap 'rm -rf "$work"' RETURN
  for shlo in "${GATE_SHAPES[@]}"; do
    full_hal "$HERE/$shlo" "$work/full.o"
    quidditch_compile "$HERE/$shlo" "$work/cand.o"
    if [ -s "$work/cand.o" ] && diff <(norm "$work/full.o") <(norm "$work/cand.o") >/dev/null; then
      echo "PASS: $shlo quidditch-compile byte-identical to the full Stream+HAL path"
    else
      echo "FAIL: $shlo quidditch-compile diverged" >&2
      diff <(norm "$work/full.o") <(norm "$work/cand.o") | head >&2
      rc=1
    fi
  done
  return $rc
}

usage() {
  echo "usage: $0 compile <in.stablehlo.mlir> <out.o> | check" >&2
  exit 2
}

case "${1:-}" in
  compile) [ $# -eq 3 ] || usage; quidditch_compile "$2" "$3" ;;
  check) run_gate ;;
  *) usage ;;
esac
