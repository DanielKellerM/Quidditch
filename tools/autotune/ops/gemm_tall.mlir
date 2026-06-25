builtin.module @gemm_tall {
    // Non-square GEMM C[32,16] = A[32,16] * B[16,16]^T (transpose-b via explicit
    // indexing_maps, as gemm_square). A second autotuner kernel with a distinct
    // shape (M!=N) and func name, to prove the op-spec / generated-harness /
    // direct-build pipeline generalizes past gemm_square.
    func.func @gemm_tall(%arg0: tensor<32x16xf64>, %arg1: tensor<16x16xf64>) -> tensor<32x16xf64> {
      %init = tensor.empty() : tensor<32x16xf64>
      %out = linalg.matmul
        indexing_maps = [
            affine_map<(d0, d1, d2) -> (d0, d2)>,
            affine_map<(d0, d1, d2) -> (d1, d2)>,
            affine_map<(d0, d1, d2) -> (d0, d1)>
        ]
        {
            lowering_config = #quidditch_snitch.lowering_config<
                l1_tiles = [8, 8, 8],
                l1_tiles_interchange = [2, 0, 1],
                dual_buffer = true
            >
        }
        ins(%arg0, %arg1 : tensor<32x16xf64>, tensor<16x16xf64>)
        outs(%init : tensor<32x16xf64>) -> tensor<32x16xf64>
      func.return %out : tensor<32x16xf64>
    }
}
