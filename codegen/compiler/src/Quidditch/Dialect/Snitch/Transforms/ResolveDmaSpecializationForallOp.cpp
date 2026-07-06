#include "Passes.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace quidditch::Snitch {
#define GEN_PASS_DEF_RESOLVEDMASPECIALIZATIONFORALLOPPASS
#include "Quidditch/Dialect/Snitch/Transforms/Passes.h.inc"
} // namespace quidditch::Snitch

namespace {
class ResolveDmaSpecializationForallOp
    : public quidditch::Snitch::impl::ResolveDmaSpecializationForallOpPassBase<
          ResolveDmaSpecializationForallOp> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};
} // namespace

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace quidditch::Snitch;

void ResolveDmaSpecializationForallOp::runOnOperation() {
  IRRewriter rewriter(&getContext());
  // ReconcileTranslationInfo resolves the workgroup forall only in the export
  // root + call-graph-reachable funcs. The DMA-specialization companion func
  // ('<name>$dma', a clone reached via the quidditch_snitch.dma_specialization
  // attribute rather than a call) is missed, so its copy of the workgroup forall
  // survives into ConvertToLLVM ("0 or 1 blocks"). Resolve any remaining
  // workgroup forall the SAME way the main func was (deLinearizeFrom = IdX,
  // matching Reconcile's default) but WITHOUT a workgroup_count_hint -- the count
  // is owned by the export-root func. Funcs whose forall is already resolved (or
  // that never had one, e.g. the {1,1,1} path) are a no-op.
  getOperation()->walk([&](FunctionOpInterface funcOp) {
    if (funcOp.isDeclaration())
      return;
    if (failed(resolveWorkgroupForAll(rewriter, funcOp,
                                      IREE::Codegen::WorkgroupId::IdX,
                                      /*emitCountHint=*/false)))
      signalPassFailure();
  });
}
