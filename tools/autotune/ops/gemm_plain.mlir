builtin.module @gemm_plain {
    // Plain (non-transpose) GEMM C[16,16] = A[16,16] * B[16,16], B indexed [k,j].
    // A skill-validation kernel: exercises the plain-matmul reference path and
    // the square-case transpose detection (must derive transpose_b=false from the
    // indexing_maps, since A=B=C=16x16 shapes are transpose-ambiguous).
    func.func @gemm_plain(%arg0: tensor<16x16xf64>, %arg1: tensor<16x16xf64>) -> tensor<16x16xf64> {
      %init = tensor.empty() : tensor<16x16xf64>
      %out = linalg.matmul
        indexing_maps = [
            affine_map<(d0, d1, d2) -> (d0, d2)>,
            affine_map<(d0, d1, d2) -> (d2, d1)>,
            affine_map<(d0, d1, d2) -> (d0, d1)>
        ]
        {
            lowering_config = #quidditch_snitch.lowering_config<
                l1_tiles = [8, 8, 8],
                l1_tiles_interchange = [2, 0, 1],
                dual_buffer = true
            >
        }
        ins(%arg0, %arg1 : tensor<16x16xf64>, tensor<16x16xf64>)
        outs(%init : tensor<16x16xf64>) -> tensor<16x16xf64>
      func.return %out : tensor<16x16xf64>
    }
}
