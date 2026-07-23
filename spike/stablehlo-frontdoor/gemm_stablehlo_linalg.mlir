#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module @gemm_square {
  func.func @gemm64(%arg0: tensor<16x16xf64>, %arg1: tensor<16x16xf64>) -> tensor<16x16xf64> {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = tensor.empty() : tensor<16x16xf64>
    %1 = linalg.fill ins(%cst : f64) outs(%0 : tensor<16x16xf64>) -> tensor<16x16xf64>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<16x16xf64>, tensor<16x16xf64>) outs(%1 : tensor<16x16xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %3 = arith.mulf %in, %in_0 : f64
      %4 = arith.addf %out, %3 : f64
      linalg.yield %4 : f64
    } -> tensor<16x16xf64>
    return %2 : tensor<16x16xf64>
  }
}

