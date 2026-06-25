builtin.module @twomm {
    // Two-dispatch proxy MODEL (entry @main) for the whole-model loop:
    // profile -> orchestrate -> extract -> tune -> apply(ConfigureForSnitch).
    // Unlike nsnet2 (GRU, doesn't compile), this is pure matmul_transpose_b and
    // compiles through iree+xDSL. NO inline lowering_config on purpose: each
    // dispatch's tiling comes from ConfigureForSnitch (seed table / JSON file),
    // exactly the model code path the autotuner applies tunings to.
    //   dispatch 0: C0[16,16] = arg0[16,16] * arg1[16,16]^T   (M=16 N=16 K=16)
    //   dispatch 1: C1[16,24] = C0[16,16]   * arg2[24,16]^T   (M=16 N=24 K=16)
    // Distinct shapes -> distinct dispatch symbols; both M>1, divisor-tileable.
    func.func @main(%arg0: tensor<16x16xf64>, %arg1: tensor<16x16xf64>,
                    %arg2: tensor<24x16xf64>) -> tensor<16x24xf64> {
      %i0 = tensor.empty() : tensor<16x16xf64>
      %c0 = linalg.matmul
        indexing_maps = [
            affine_map<(d0, d1, d2) -> (d0, d2)>,
            affine_map<(d0, d1, d2) -> (d1, d2)>,
            affine_map<(d0, d1, d2) -> (d0, d1)>
        ]
        ins(%arg0, %arg1 : tensor<16x16xf64>, tensor<16x16xf64>)
        outs(%i0 : tensor<16x16xf64>) -> tensor<16x16xf64>
      %i1 = tensor.empty() : tensor<16x24xf64>
      %c1 = linalg.matmul
        indexing_maps = [
            affine_map<(d0, d1, d2) -> (d0, d2)>,
            affine_map<(d0, d1, d2) -> (d1, d2)>,
            affine_map<(d0, d1, d2) -> (d0, d1)>
        ]
        ins(%c0, %arg2 : tensor<16x16xf64>, tensor<24x16xf64>)
        outs(%i1 : tensor<16x24xf64>) -> tensor<16x24xf64>
      func.return %c1 : tensor<16x24xf64>
    }
}
