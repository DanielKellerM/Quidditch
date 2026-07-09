builtin.module @gemm_square {
    // Multi-cluster (4x4 / 16-cluster gwaihir) square GEMM: C[64,64] = A[64,64] * B[64,64]^T.
    // workgroup_tiles=[16,16,0] fans the 64x64 output into a 4x4 grid of 16x16
    // C-blocks => workgroup_count = 16, one block per cluster (16 wg % 16 clusters
    // = exact bijection). Each block is the SAME 16x16 / [8,8,8]-L1 kernel that the
    // validated single-cluster gemm_square.mlir runs (C[0][0]=22880), so the only
    // thing that scales is the dispatch grid. Function name kept as gemm64 so the
    // generated dispatch symbols referenced from the host stay valid. See
    // docs/multicluster-4x4-runbook.md.
    func.func @gemm64(%arg0: tensor<64x64xf64>, %arg1: tensor<64x64xf64>) -> tensor<64x64xf64> {
      %init = tensor.empty() : tensor<64x64xf64>
      // Zero the accumulator (matmul accumulates into outs).
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
                workgroup_tiles = [16, 16, 0],
                l1_tiles = [8, 8, 8],
                l1_tiles_interchange = [2, 0, 1],
                dual_buffer = false
            >
        }
        ins(%arg0, %arg1 : tensor<64x64xf64>, tensor<64x64xf64>)
        outs(%filled : tensor<64x64xf64>) -> tensor<64x64xf64>
      func.return %out : tensor<64x64xf64>
    }
}
