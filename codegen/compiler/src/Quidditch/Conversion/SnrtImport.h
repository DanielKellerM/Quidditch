// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// One place to declare an snRuntime symbol the compiler emits calls to, as a
// hal.import.bitcode LLVMFuncOp resolved from libsnRuntime at link. The canonical
// C signatures live in runtime/runtime/src/Quidditch/quidditch_snrt_abi.h; the
// MLIR types passed here mirror them. Keeps the three emission sites from each
// re-spelling the hal.import.bitcode attr.

#ifndef QUIDDITCH_CONVERSION_SNRTIMPORT_H_
#define QUIDDITCH_CONVERSION_SNRTIMPORT_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"

namespace quidditch {

inline mlir::LLVM::LLVMFuncOp
declareSnrtImport(mlir::OpBuilder &builder, mlir::StringRef name,
                  mlir::LLVM::LLVMFunctionType type) {
  auto func = builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(),
                                                     name, type);
  func->setAttr("hal.import.bitcode", builder.getUnitAttr());
  return func;
}

}  // namespace quidditch

#endif  // QUIDDITCH_CONVERSION_SNRTIMPORT_H_
