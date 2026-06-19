builtin.module @gemm_square {
    // Clean square GEMM (BLAS-3): C[64,64] = A[64,64] * B[64,64]^T, no trailing elementwise.
    // 64x64 f64 output = 32 KB, fits the ~100 KB L1 staging view (96 fits; 128 does not).
    // transpose-b expressed via explicit indexing_maps: C[i,j] = sum_k A[i,k] * B[j,k]
    // (linalg.matmul_transpose_b was removed in LLVM 23).
    func.func @gemm64(%arg0: tensor<64x64xf64>, %arg1: tensor<64x64xf64>) -> tensor<64x64xf64> {
      %init = tensor.empty() : tensor<64x64xf64>
      %out = linalg.matmul
        indexing_maps = [
            affine_map<(d0, d1, d2) -> (d0, d2)>,
            affine_map<(d0, d1, d2) -> (d1, d2)>,
            affine_map<(d0, d1, d2) -> (d0, d1)>
        ]
        {
            lowering_config = #quidditch_snitch.lowering_config<
                l1_tiles = [32, 32, 32],
                l1_tiles_interchange = [2, 0, 1],
                dual_buffer = true
            >
        }
        ins(%arg0, %arg1 : tensor<64x64xf64>, tensor<64x64xf64>)
        outs(%init : tensor<64x64xf64>) -> tensor<64x64xf64>
      func.return %out : tensor<64x64xf64>
    }
}
