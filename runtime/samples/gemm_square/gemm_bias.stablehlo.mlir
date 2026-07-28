func.func @bias(%a: tensor<16x16xf64>, %b: tensor<16x16xf64>, %d: tensor<16x16xf64>)
    -> tensor<16x16xf64> {
  %e = "stablehlo.dot_general"(%a, %b) {
    dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>
  } : (tensor<16x16xf64>, tensor<16x16xf64>) -> tensor<16x16xf64>
  %f = "stablehlo.add"(%e, %d) : (tensor<16x16xf64>, tensor<16x16xf64>) -> tensor<16x16xf64>
  return %f : tensor<16x16xf64>
}
