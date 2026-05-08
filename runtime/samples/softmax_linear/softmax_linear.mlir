builtin.module @softmax_linear {
  func.func @forward(%W: tensor<10x16xf64>,
                     %x: tensor<16xf64>,
                     %b: tensor<10xf64>) -> tensor<10xf64> {
    %zero = arith.constant 0.0 : f64

    // Wx
    %mv_init = tensor.empty() : tensor<10xf64>
    %mv_zero = linalg.fill ins(%zero : f64) outs(%mv_init : tensor<10xf64>) -> tensor<10xf64>
    %Wx = linalg.matvec
            ins(%W, %x : tensor<10x16xf64>, tensor<16xf64>)
            outs(%mv_zero : tensor<10xf64>) -> tensor<10xf64>

    // Wx + b
    %add_init = tensor.empty() : tensor<10xf64>
    %Wxb = linalg.add
             ins(%Wx, %b : tensor<10xf64>, tensor<10xf64>)
             outs(%add_init : tensor<10xf64>) -> tensor<10xf64>

    // softmax
    %sm_init = tensor.empty() : tensor<10xf64>
    %sm = linalg.softmax dimension(0)
            ins(%Wxb : tensor<10xf64>)
            outs(%sm_init : tensor<10xf64>) -> tensor<10xf64>

    func.return %sm : tensor<10xf64>
  }
}
