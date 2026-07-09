builtin.module @gemm_square {
    // Multi-cluster (2x2 / 4-cluster gwaihir) square GEMM: C[32,32] = A[32,32] * B[32,32]^T.
    // workgroup_tiles=[16,16,0] fans the 32x32 output into a 2x2 grid of 16x16
    // C-blocks => workgroup_count = 4, one block per cluster (4 wg % 4 clusters
    // = exact bijection). Each block is the SAME 16x16 / [8,8,8]-L1 kernel that the
    // validated single-cluster gemm_square.mlir runs (C[0][0]=22880), so the only
    // thing that scales is the dispatch grid. Function name kept as gemm64 so the
    // generated dispatch symbols referenced from the host stay valid. See
    // docs/multicluster-4x4-runbook.md.
    func.func @gemm64(%arg0: tensor<32x32xf64>, %arg1: tensor<32x32xf64>) -> tensor<32x32xf64> {
      %init = tensor.empty() : tensor<32x32xf64>
      // Zero the accumulator (matmul accumulates into outs).
      %zero = arith.constant 0.000000e+00 : f64
      %filled = linalg.fill ins(%zero : f64) outs(%init : tensor<32x32xf64>) -> tensor<32x32xf64>
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
        ins(%arg0, %arg1 : tensor<32x32xf64>, tensor<32x32xf64>)
        outs(%filled : tensor<32x32xf64>) -> tensor<32x32xf64>
      func.return %out : tensor<32x32xf64>
    }
}
