#ifndef QUIDDITCH_CONVERSION_PASSES
#define QUIDDITCH_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertToRISCVPass : Pass<"quidditch-convert-to-riscv", "mlir::ModuleOp"> {
  let options = [
    Option<"xDSLOptPath", "xdsl-opt-path", "std::string", [{""}],
      "Path to the 'xdsl-opt' executable to use for kernel compilation.">,
    Option<"assertCompiled", "assert-compiled", "bool", "false",
           "If true, errors if any kernel could not be compiled with xDSL."
           "Otherwise, removes the kernel from the output and emits a warning "
           "instead.">,
    Option<"xDSLPasses", "xdsl-passes", "std::string",
           [{"arith-add-fastmath,test-lower-linalg-to-snitch"}],
           "The xdsl-opt pass pipeline (-p argument) used to lower each kernel. "
           "Exposed so the autotuner can sweep Group-B pass knobs/ordering.">,
  ];

  let dependentDialects = [
    "mlir::func::FuncDialect",
  ];
}

#endif
