builtin.module @gemm_square {
    // Small square GEMM (BLAS-3): C[16,16] = A[16,16] * B[16,16]^T.
    // Shrunk from 64x64 for a fast end-to-end correctness check on the sim;
    // keeps the 2x2x2 L1 tiling structure. transpose-b expressed via explicit
    // indexing_maps: C[i,j] = sum_k A[i,k] * B[j,k] (linalg.matmul_transpose_b
    // was removed in LLVM 23). Function name kept as gemm64 so the generated
    // dispatch symbols referenced from main.c stay valid.
    func.func @gemm64(%arg0: tensor<16x16xf64>, %arg1: tensor<16x16xf64>) -> tensor<16x16xf64> {
      %init = tensor.empty() : tensor<16x16xf64>
      // Zero the accumulator: matmul accumulates into outs, so it must start at
      // 0 (else C[i,j] += A.B reads uninitialized memory -> X on RTL).
      %zero = arith.constant 0.000000e+00 : f64
      %filled = linalg.fill ins(%zero : f64) outs(%init : tensor<16x16xf64>) -> tensor<16x16xf64>
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
        ins(%arg0, %arg1 : tensor<16x16xf64>, tensor<16x16xf64>)
        outs(%filled : tensor<16x16xf64>) -> tensor<16x16xf64>
      func.return %out : tensor<16x16xf64>
    }
}
