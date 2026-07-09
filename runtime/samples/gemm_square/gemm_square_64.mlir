builtin.module @gemm_square {
    func.func @gemm64(%arg0: tensor<64x64xf64>, %arg1: tensor<64x64xf64>) -> tensor<64x64xf64> {
      %init = tensor.empty() : tensor<64x64xf64>
      %zero = arith.constant 0.000000e+00 : f64
      %filled = linalg.fill ins(%zero : f64) outs(%init : tensor<64x64xf64>) -> tensor<64x64xf64>
      %out = linalg.matmul
        indexing_maps = [
            affine_map<(d0, d1, d2) -> (d0, d2)>,
            affine_map<(d0, d1, d2) -> (d1, d2)>,
            affine_map<(d0, d1, d2) -> (d0, d1)>
        ]
        {
            lowering_config = #quidditch_snitch.lowering_config<
                l1_tiles = [64, 64, 64],
                l1_tiles_interchange = [2, 0, 1],
                dual_buffer = false
            >
        }
        ins(%arg0, %arg1 : tensor<64x64xf64>, tensor<64x64xf64>)
        outs(%filled : tensor<64x64xf64>) -> tensor<64x64xf64>
      func.return %out : tensor<64x64xf64>
    }
}
