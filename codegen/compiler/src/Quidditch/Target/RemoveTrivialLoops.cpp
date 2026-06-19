#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchOps.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

// Adapted from IREE's upstream RemoveTrivialLoops pass.
//
// Since IREE moved single-iteration-loop removal to be driven by
// `ValueBoundsOpInterface`, the bounds of `quidditch_snitch.compute_core_index`
// (range `[0, compute_cores - 1]`) are now discoverable through that interface
// directly on the op. The previously hand-written `GetMinMaxExprFn` callback is
// therefore no longer required: `populateRemoveSingleIterationLoopPattern`
// queries the interface itself.

namespace quidditch {
#define GEN_PASS_DEF_REMOVETRIVIALLOOPSPASS
#include "Quidditch/Target/Passes.h.inc"
} // namespace quidditch

using namespace mlir;
using namespace iree_compiler;

static LogicalResult removeOneTripTiledLoops(FunctionOpInterface funcOp) {
  RewritePatternSet patterns(funcOp.getContext());
  populateRemoveSingleIterationLoopPattern(patterns);
  return applyPatternsGreedily(funcOp, std::move(patterns));
}

namespace {

class RemoveTrivialLoops final
    : public quidditch::impl::RemoveTrivialLoopsPassBase<RemoveTrivialLoops> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    if (failed(removeOneTripTiledLoops(funcOp))) {
      return signalPassFailure();
    }
  }
};
} // namespace
