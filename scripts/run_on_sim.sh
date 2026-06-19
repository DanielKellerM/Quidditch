#!/usr/bin/env bash
# Run a built sample ELF on the snitch_cluster Questa simulator, keeping every
# artifact (transcript, *.wlf, DMA/instruction traces, .rtlbinary, stdout) inside
# a dedicated runs/<name>/ directory instead of polluting the snitch_cluster root.
#
# Usage: scripts/run_on_sim.sh <elf> [run-name] [--trace]
#   <elf>       path to a built sample binary (e.g. build-rt/samples/gemm_square/gemm_square)
#   [run-name]  output subdir under runs/ (default: basename of the ELF)
#   --trace     keep the per-instruction disassembly traces (default: discard them)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER="$ROOT/snitch_cluster"
VSIM="$CLUSTER/target/sim/build/bin/snitch_cluster.vsim"

die() { echo "run_on_sim: $*" >&2; exit 1; }
[ $# -ge 1 ] || die "usage: $0 <elf> [run-name] [--trace]"

ELF="$(realpath "$1")"; shift
[ -f "$ELF" ] || die "ELF not found: $ELF"
[ -x "$VSIM" ] || die "sim wrapper not built: $VSIM (run 'make' in snitch_cluster/target/sim)"

NAME=""
TRACE=0
for arg in "$@"; do
  case "$arg" in
    --trace) TRACE=1 ;;
    *) NAME="$arg" ;;
  esac
done
[ -n "$NAME" ] || NAME="$(basename "$ELF")"

RUN_DIR="$ROOT/runs/$NAME"
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR/logs"

# The Snitch tracer is always on in sim (no plusarg gate) and writes a huge
# per-instruction logs/trace_hart_*.dasm. Discard via /dev/null unless --trace.
if [ "$TRACE" -eq 0 ]; then
  for i in $(seq 0 8); do ln -sf /dev/null "$RUN_DIR/logs/trace_hart_0000${i}.dasm"; done
fi

echo "run_on_sim: $NAME -> $RUN_DIR"
cd "$RUN_DIR"
"$VSIM" "$ELF" 2>&1 | tee "$RUN_DIR/run.log"
