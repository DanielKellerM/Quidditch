#!/usr/bin/env bash
# The standalone Quidditch compile pipeline (S2/2b): StableHLO input -> Snitch device
# object, with NO Stream / VM / EmitC and NO HAL executable-pruning.
#
# Two stages, both reusing existing tooling:
#   1. iree-compile --compile-to=flow      StableHLO -> linalg -> GlobalOpt ->
#                                          DispatchCreation -> Flow (front-end, as-is).
#   2. iree-opt --pass-pipeline=<tail>      Flow -> device .o. The tail is exactly:
#        quidditch-materialize-executable-from-flow  (replaces Stream+MaterializeInterfaces)
#        iree-hal-configure-executables              (select the Snitch tiling strategy)
#        iree-hal-translate-all-executables          (xDSL/Snitch codegen)
#        iree-hal-serialize-all-executables          (emit the static library .o)
#      ConvertToHAL (host program), LinkExecutables and the three PruneExecutables/
#      SymbolDCE passes of the full HAL pipeline are deliberately omitted: this is a
#      device-only compiler, so nothing references the executable and pruning would
#      drop it -- omitting the prune is what makes the device-only path work without a
#      host liveness anchor.
#
# Proven byte-identical to IREE's full Stream+HAL path (see check verb below).
set -euo pipefail

IREE_COMPILE=${QUIDDITCH_IREE_COMPILE:?set QUIDDITCH_IREE_COMPILE to the iree-compile path}
IREE_OPT=${QUIDDITCH_IREE_OPT:?set QUIDDITCH_IREE_OPT to an iree-opt built with the quidditch plugin}
XDSL_OPT=${QUIDDITCH_XDSL_OPT:?set QUIDDITCH_XDSL_OPT to the xdsl-opt path}
TOOLCHAIN_ROOT=${QUIDDITCH_TOOLCHAIN_ROOT:?set QUIDDITCH_TOOLCHAIN_ROOT}
CFG_HEADER=${QUIDDITCH_CFG_HEADER:?set QUIDDITCH_CFG_HEADER to the snitch_cluster_cfg.h path}

HERE="$(cd "$(dirname "$0")" && pwd)"
CONFIG_TABLE=${QUIDDITCH_CONFIG_TABLE:-$HERE/gemm_square_config.json}

qflags=(--iree-quidditch-xdsl-opt-path="$XDSL_OPT"
        --iree-quidditch-toolchain-root="$TOOLCHAIN_ROOT"
        --iree-quidditch-cluster-cfg-header="$CFG_HEADER"
        --iree-quidditch-config-table="$CONFIG_TABLE")

TAIL='builtin.module(quidditch-materialize-executable-from-flow, hal.executable(iree-hal-configure-executables, iree-hal-translate-all-executables, iree-hal-serialize-all-executables))'

# quidditch_compile <input.stablehlo.mlir> <out.o>
quidditch_compile() {
  local in="$1" out="$2" flow
  flow="$(mktemp --suffix=.mlir)"
  "$IREE_COMPILE" --iree-input-type=auto --iree-input-demote-f64-to-f32=0 \
    --iree-hal-target-backends=quidditch "${qflags[@]}" \
    --compile-to=flow "$in" -o "$flow"
  "$IREE_OPT" "${qflags[@]}" --iree-quidditch-static-library-output-path="$out" \
    --pass-pipeline="$TAIL" "$flow" -o /dev/null
  rm -f "$flow"
}

usage() {
  echo "usage: $0 compile <in.stablehlo.mlir> <out.o> | check | check-binary" >&2
  exit 2
}

case "${1:-}" in
  compile)
    [ $# -eq 3 ] || usage
    quidditch_compile "$2" "$3"
    ;;
  check)
    OBJDUMP=${QUIDDITCH_OBJDUMP:?set QUIDDITCH_OBJDUMP to a Snitch llvm-objdump}
    WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
    norm() { "$OBJDUMP" -d --mattr=+xdma,+xssr,+xfrep "$1" \
               | grep -vE 'file format|^/|:[[:space:]]*$' \
               | sed -E 's/^[0-9a-f ]+://; s/matmul_like/matmul/g'; }
    rc=0
    for shlo in gemm_square.stablehlo.mlir gemm_square_4x4.stablehlo.mlir gemm_bias.stablehlo.mlir; do
      "$IREE_COMPILE" --iree-input-type=auto --iree-input-demote-f64-to-f32=0 \
        --iree-hal-target-backends=quidditch "${qflags[@]}" \
        --iree-quidditch-static-library-output-path="$WORK/full.o" \
        --compile-to=hal "$HERE/$shlo" -o /dev/null
      quidditch_compile "$HERE/$shlo" "$WORK/out.o"
      if diff <(norm "$WORK/full.o") <(norm "$WORK/out.o") >/dev/null; then
        echo "PASS: $shlo standalone pipeline byte-identical to the full Stream+HAL path"
      else
        echo "FAIL: $shlo standalone pipeline diverged" >&2
        diff <(norm "$WORK/full.o") <(norm "$WORK/out.o") | head >&2
        rc=1
      fi
    done
    exit $rc
    ;;
  check-binary)
    # Same byte-exact check, but exercising the fused single-binary quidditch-compile
    # (not the two-stage iree-compile|iree-opt path) so the shipped tool is gated too.
    QC=${QUIDDITCH_COMPILE:?set QUIDDITCH_COMPILE to the quidditch-compile binary}
    OBJDUMP=${QUIDDITCH_OBJDUMP:?set QUIDDITCH_OBJDUMP to a Snitch llvm-objdump}
    WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
    norm() { "$OBJDUMP" -d --mattr=+xdma,+xssr,+xfrep "$1" \
               | grep -vE 'file format|^/|:[[:space:]]*$' \
               | sed -E 's/^[0-9a-f ]+://; s/matmul_like/matmul/g'; }
    rc=0
    for shlo in gemm_square.stablehlo.mlir gemm_square_4x4.stablehlo.mlir gemm_bias.stablehlo.mlir; do
      "$IREE_COMPILE" --iree-input-type=auto --iree-input-demote-f64-to-f32=0 \
        --iree-hal-target-backends=quidditch "${qflags[@]}" \
        --iree-quidditch-static-library-output-path="$WORK/full.o" \
        --compile-to=hal "$HERE/$shlo" -o /dev/null
      "$QC" "$HERE/$shlo" --iree-input-type=auto --iree-input-demote-f64-to-f32=0 \
        --iree-hal-target-backends=quidditch "${qflags[@]}" \
        --iree-quidditch-static-library-output-path="$WORK/bin.o"
      if [ -s "$WORK/bin.o" ] && diff <(norm "$WORK/full.o") <(norm "$WORK/bin.o") >/dev/null; then
        echo "PASS: $shlo quidditch-compile binary byte-identical to the full Stream+HAL path"
      else
        echo "FAIL: $shlo quidditch-compile binary diverged" >&2
        diff <(norm "$WORK/full.o") <(norm "$WORK/bin.o") | head >&2
        rc=1
      fi
    done
    exit $rc
    ;;
  *) usage ;;
esac
