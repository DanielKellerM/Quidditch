module @mlp attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<16xf64>, %arg1: tensor<16x16xf64>, %arg2: tensor<16xf64>, %arg3: tensor<16x16xf64>, %arg4: tensor<16x16xf64>) -> tensor<16x16xf64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<16x16xf64>
    %0 = stablehlo.transpose %arg3, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f64[16,16]{0,1}"} : (tensor<16x16xf64>) -> tensor<16x16xf64>
    %1 = stablehlo.dot_general %arg4, %0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x16xf64>, tensor<16x16xf64>) -> tensor<16x16xf64>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<16xf64>) -> tensor<16x16xf64>
    %3 = stablehlo.add %1, %2 : tensor<16x16xf64>
    %4 = stablehlo.maximum %3, %cst : tensor<16x16xf64>
    %5 = stablehlo.transpose %arg1, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f64[16,16]{0,1}"} : (tensor<16x16xf64>) -> tensor<16x16xf64>
    %6 = stablehlo.dot_general %4, %5, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x16xf64>, tensor<16x16xf64>) -> tensor<16x16xf64>
    %7 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<16xf64>) -> tensor<16x16xf64>
    %8 = stablehlo.add %6, %7 : tensor<16x16xf64>
    return %8 : tensor<16x16xf64>
  }
}
