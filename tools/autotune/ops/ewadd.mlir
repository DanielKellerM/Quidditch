builtin.module @ewadd {
    // 1D elementwise add C[256] = A[256] + B[256] (linalg.generic, one parallel
    // loop). IREE collapses an N-D elementwise to 1-D before TensorTile, so the
    // tunable l1_tiles is a SINGLE entry over the flattened element count (a 2-D
    // [Mt,Nt] config is dropped post-collapse). First non-matmul autotuner kernel:
    // proves the harness/spec/build/sweep generalizes to a 1-D op (l1_tiles
    // length 1, interchange [0]). Inline lowering_config so direct_build injects.
    func.func @ewadd(%arg0: tensor<256xf64>, %arg1: tensor<256xf64>) -> tensor<256xf64> {
      %init = tensor.empty() : tensor<256xf64>
      %out = linalg.generic
        {indexing_maps = [affine_map<(d0) -> (d0)>,
                          affine_map<(d0) -> (d0)>,
                          affine_map<(d0) -> (d0)>],
         iterator_types = ["parallel"],
         lowering_config = #quidditch_snitch.lowering_config<
            l1_tiles = [64],
            l1_tiles_interchange = [0],
            dual_buffer = true>}
        ins(%arg0, %arg1 : tensor<256xf64>, tensor<256xf64>)
        outs(%init : tensor<256xf64>) {
      ^bb0(%a: f64, %b: f64, %o: f64):
        %s = arith.addf %a, %b : f64
        linalg.yield %s : f64
      } -> tensor<256xf64>
      func.return %out : tensor<256xf64>
    }
}
