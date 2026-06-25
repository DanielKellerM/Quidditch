builtin.module @gemm_bigk {
    // Large-K matmul C[4,4] = A[4,1024] * B[4,1024]^T, f64 (transpose-b via
    // indexing_maps). Small M,N keep the in-harness O(M*N*K) scalar reference
    // cheap enough to sim, while K=1024 makes the structured-integer reference
    // overflow int32 (~7.9e9 > 2^31) yet stay f64-EXACT (< 2^53) -- so it
    // exercises the f64-exact reference path (the int32 gate would reject it).
    // The DM core that runs the check has an FPU and matmul needs only fmul/fadd.
    func.func @gemm_bigk(%arg0: tensor<4x1024xf64>, %arg1: tensor<4x1024xf64>) -> tensor<4x4xf64> {
      %init = tensor.empty() : tensor<4x4xf64>
      %out = linalg.matmul
        indexing_maps = [
            affine_map<(d0, d1, d2) -> (d0, d2)>,
            affine_map<(d0, d1, d2) -> (d1, d2)>,
            affine_map<(d0, d1, d2) -> (d0, d1)>
        ]
        {
            lowering_config = #quidditch_snitch.lowering_config<
                l1_tiles = [4, 4, 4],
                l1_tiles_interchange = [2, 0, 1],
                dual_buffer = true
            >
        }
        ins(%arg0, %arg1 : tensor<4x1024xf64>, tensor<4x1024xf64>)
        outs(%init : tensor<4x4xf64>) -> tensor<4x4xf64>
      func.return %out : tensor<4x4xf64>
    }
}
