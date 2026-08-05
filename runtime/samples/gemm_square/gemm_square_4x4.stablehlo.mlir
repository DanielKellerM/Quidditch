// StableHLO front-door of gemm_square_4x4.mlir: C[64,64] = A[64,64] * B[64,64]^T, f64.
// 16-workgroup (4x4) multi-cluster gemm. As with the 16x16 sample, no baked
// lowering_config: the tiling (incl. workgroup_tiles=[16,16,0]) is selected by
// ConfigureForSnitch from gemm_square_config.json, keyed by the dispatch symbol.
module @gemm_square {
  func.func @gemm64(%a: tensor<64x64xf64>, %b: tensor<64x64xf64>) -> tensor<64x64xf64> {
    %c = "stablehlo.dot_general"(%a, %b) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [1]
      >,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<64x64xf64>, tensor<64x64xf64>) -> tensor<64x64xf64>
    return %c : tensor<64x64xf64>
  }
}
