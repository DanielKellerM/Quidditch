# Shared helpers for the gemm_square byte-identity gates. Source, don't execute.

# The single-dispatch StableHLO fixtures the byte-identity gates cover.
GATE_SHAPES=(gemm_square.stablehlo.mlir gemm_square_4x4.stablehlo.mlir gemm_bias.stablehlo.mlir)

# Normalize a device object to its comparable instruction stream: strip addresses, drop
# objdump framing, and treat matmul_like as matmul. The Snitch +xdma/+xssr/+xfrep
# features are what decode the custom ops -- one definition so the oracle can't drift.
norm() {
  "${QUIDDITCH_OBJDUMP:?set QUIDDITCH_OBJDUMP to a Snitch llvm-objdump}" \
    -d --mattr=+xdma,+xssr,+xfrep "$1" \
    | grep -vE 'file format|^/|:[[:space:]]*$' \
    | sed -E 's/^[0-9a-f ]+://; s/matmul_like/matmul/g'
}
