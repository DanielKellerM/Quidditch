#!/usr/bin/env bash
# StableHLO front-door spike: StableHLO matmul -> linalg -> Quidditch xDSL Snitch backend -> device .o.
# Proves the StableHLO front-door produces a valid, byte-exact-to-the-linalg-sample Snitch device object.
# This still routes through iree-compile (Flow/Stream/HAL) -- see FINDINGS for the remaining coupling.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
IREE_COMPILE=${QUIDDITCH_IREE_COMPILE:-/scratch/dankeller/snitch-compiler/iree-p6-build/tools/iree-compile}
IREE_OPT=${QUIDDITCH_IREE_OPT:-/scratch/dankeller/snitch-compiler/iree-p6-build/tools/iree-opt}
XDSL_OPT=${QUIDDITCH_XDSL_OPT:-/scratch/dankeller/snitch-compiler/Quidditch/.venv/bin/xdsl-opt}
TOOLCHAIN_ROOT=${QUIDDITCH_TOOLCHAIN_ROOT:-/home/dankeller/Projects/Quidditch/toolchain}
CFG=${QUIDDITCH_CFG_HEADER:-/home/dankeller/Projects/Quidditch/build-rt/snitch_cluster/cluster_gen/snitch_cluster_cfg.h}
OBJDUMP=${QUIDDITCH_OBJDUMP:-/usr/scratch2/vulcano/colluca/tools/riscv32-snitch-llvm-almalinux8-15.0.0-snitch-0.5.0/bin/llvm-objdump}

compile() { # <input.mlir> <out.o>
  "$IREE_COMPILE" \
    --iree-input-type=auto \
    --iree-input-demote-f64-to-f32=0 \
    --iree-hal-target-backends=quidditch \
    --iree-quidditch-static-library-output-path="$2" \
    --iree-quidditch-xdsl-opt-path="$XDSL_OPT" \
    --iree-quidditch-toolchain-root="$TOOLCHAIN_ROOT" \
    --iree-quidditch-cluster-cfg-header="$CFG" \
    --output-format=vm-c --iree-vm-target-index-bits=32 \
    "$1" -o "${2%.o}.h"
}

echo "== 1. observe StableHLO -> linalg legalization =="
"$IREE_OPT" --iree-stablehlo-to-iree-input "$HERE/gemm_stablehlo.mlir" -o "$HERE/gemm_stablehlo_linalg.mlir"

echo "== 2. front-door compile: StableHLO (default config)  -> device .o =="
compile "$HERE/gemm_stablehlo.mlir" "$HERE/gemm_stablehlo_nocfg.o"

echo "== 3. front-door compile: StableHLO-linalg + injected sample lowering_config -> device .o =="
compile "$HERE/gemm_stablehlo_cfg.mlir" "$HERE/gemm_stablehlo_cfg.o"

echo "== 4. reference: the linalg sample compiled identically =="
compile /home/dankeller/Projects/Quidditch/runtime/samples/gemm_square/gemm_square.mlir "$HERE/gemm_sample.o"

echo "== 5. verify Snitch codegen (SSR/FREP/DMA) + library_query export =="
"$OBJDUMP" -t "$HERE/gemm_stablehlo_cfg.o" | grep -E 'library_query$'
"$OBJDUMP" -d --mattr=+xdma,+xssr,+xfrep "$HERE/gemm_stablehlo_cfg.o" \
  | grep -cwE 'scfgwi|frep.o|dmstati' | xargs echo "SSR/FREP/DMA insns:"

echo "== 6. byte-exact check: StableHLO+config  vs  linalg sample (modulo matmul_like naming) =="
norm() { "$OBJDUMP" -d --mattr=+xdma,+xssr,+xfrep "$1" | sed -E 's/^[0-9a-f ]+://; s/matmul_like/matmul/g; s/gemm_[a-z_]*\.o:/OBJ:/'; }
if diff <(norm "$HERE/gemm_sample.o") <(norm "$HERE/gemm_stablehlo_cfg.o") >/dev/null; then
  echo ">>> BYTE-EXACT IDENTICAL <<<"
else
  echo ">>> DIFFERS <<<"; diff <(norm "$HERE/gemm_sample.o") <(norm "$HERE/gemm_stablehlo_cfg.o") | head
fi
