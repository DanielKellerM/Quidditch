#!/usr/bin/env bash
# The standalone Stream-free compile pipeline (S2/2b): StableHLO -> Snitch device .o,
# no Stream/VM/EmitC. Two reused stages -- iree-compile --compile-to=flow (front) then
# iree-opt running the device-only tail. Byte-identical to the full path (see `check`).
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
fi

IREE_COMPILE=${QUIDDITCH_IREE_COMPILE:?set QUIDDITCH_IREE_COMPILE to the iree-compile path (or QUIDDITCH_BUILD_DIR)}
IREE_OPT=${QUIDDITCH_IREE_OPT:?set QUIDDITCH_IREE_OPT to an iree-opt built with the quidditch plugin}
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

# quidditch-link-executables (module-level) merges per-dispatch executables into one
# before serialize -- a no-op for a single executable, required for more than one.
TAIL='builtin.module(quidditch-materialize-executable-from-flow, hal.executable(iree-hal-configure-executables, iree-hal-translate-all-executables), quidditch-link-executables, hal.executable(iree-hal-serialize-all-executables))'

full_hal() { # <in.mlir> <out.o> : the reference object via the full Stream+HAL path
  "$IREE_COMPILE" --iree-input-type=auto --iree-input-demote-f64-to-f32=0 \
    --iree-hal-target-backends=quidditch "${qflags[@]}" \
    --iree-quidditch-static-library-output-path="$2" --compile-to=hal "$1" -o /dev/null
}

quidditch_compile() { # <in.stablehlo.mlir> <out.o> : the two-stage Stream-free pipeline
  local in="$1" out="$2" flow
  flow="$(mktemp --suffix=.mlir)"
  "$IREE_COMPILE" --iree-input-type=auto --iree-input-demote-f64-to-f32=0 \
    --iree-hal-target-backends=quidditch "${qflags[@]}" \
    --compile-to=flow "$in" -o "$flow"
  "$IREE_OPT" "${qflags[@]}" --iree-quidditch-static-library-output-path="$out" \
    --pass-pipeline="$TAIL" "$flow" -o /dev/null
  rm -f "$flow"
}

# Compare each shape's candidate object (produced by $1 <in> <out.o>) to the full path.
run_gate() {
  local produce="$1" what="$2" rc=0
  local work; work=$(mktemp -d); trap 'rm -rf "$work"' RETURN
  for shlo in "${GATE_SHAPES[@]}"; do
    full_hal "$HERE/$shlo" "$work/full.o"
    "$produce" "$HERE/$shlo" "$work/cand.o"
    if [ -s "$work/cand.o" ] && diff <(norm "$work/full.o") <(norm "$work/cand.o") >/dev/null; then
      echo "PASS: $shlo $what byte-identical to the full Stream+HAL path"
    else
      echo "FAIL: $shlo $what diverged" >&2
      diff <(norm "$work/full.o") <(norm "$work/cand.o") | head >&2
      rc=1
    fi
  done
  return $rc
}

# The fused single binary as a candidate producer, so `check-binary` reuses run_gate.
quidditch_compile_binary() {
  "${QUIDDITCH_COMPILE:?set QUIDDITCH_COMPILE to the quidditch-compile binary}" "$1" \
    --iree-input-type=auto --iree-input-demote-f64-to-f32=0 \
    --iree-hal-target-backends=quidditch "${qflags[@]}" \
    --iree-quidditch-static-library-output-path="$2"
}

usage() {
  echo "usage: $0 compile <in.stablehlo.mlir> <out.o> | check | check-binary" >&2
  exit 2
}

case "${1:-}" in
  compile) [ $# -eq 3 ] || usage; quidditch_compile "$2" "$3" ;;
  check) run_gate quidditch_compile "standalone pipeline" ;;
  check-binary) run_gate quidditch_compile_binary "quidditch-compile binary" ;;
  *) usage ;;
esac
