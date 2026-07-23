// StableHLO front-door spike: C[16,16] = A[16,16] * B[16,16]^T, f64.
// dot_general with lhs contracting dim 1 and rhs contracting dim 1 expresses
// the transpose-b matmul C[i,j] = sum_k A[i,k] * B[j,k] (matches gemm_square.mlir).
module @gemm_square {
  func.func @gemm64(%a: tensor<16x16xf64>, %b: tensor<16x16xf64>) -> tensor<16x16xf64> {
    %c = "stablehlo.dot_general"(%a, %b) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [1]
      >,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<16x16xf64>, tensor<16x16xf64>) -> tensor<16x16xf64>
    return %c : tensor<16x16xf64>
  }
}
