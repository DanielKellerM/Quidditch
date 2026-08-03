// A 2-layer MLP front-door sample beyond gemm: y = relu(x@W1 + b1) @ W2 + b2.
// Exercises multi-op (dot_general + add + relu-as-maximum) and multi-dispatch through
// the StableHLO front-door -- the shape of a real feedforward network layer.
func.func @mlp(%x: tensor<16x16xf64>, %w1: tensor<16x16xf64>, %b1: tensor<16x16xf64>,
               %w2: tensor<16x16xf64>, %b2: tensor<16x16xf64>) -> tensor<16x16xf64> {
  %z1 = "stablehlo.dot_general"(%x, %w1) {
    dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>
  } : (tensor<16x16xf64>, tensor<16x16xf64>) -> tensor<16x16xf64>
  %a1 = "stablehlo.add"(%z1, %b1) : (tensor<16x16xf64>, tensor<16x16xf64>) -> tensor<16x16xf64>
  %zero = stablehlo.constant dense<0.0> : tensor<16x16xf64>
  %h1 = "stablehlo.maximum"(%a1, %zero) : (tensor<16x16xf64>, tensor<16x16xf64>) -> tensor<16x16xf64>
  %z2 = "stablehlo.dot_general"(%h1, %w2) {
    dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>
  } : (tensor<16x16xf64>, tensor<16x16xf64>) -> tensor<16x16xf64>
  %y = "stablehlo.add"(%z2, %b2) : (tensor<16x16xf64>, tensor<16x16xf64>) -> tensor<16x16xf64>
  return %y : tensor<16x16xf64>
}
