#include "Passes.h"

#include "Quidditch/Dialect/DMA/IR/DMADialect.h"
#include "Quidditch/Dialect/DMA/IR/DMAOps.h"
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchAttrs.h"
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.h"
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace quidditch::Snitch {
#define GEN_PASS_DEF_PROMOTEOPERANDSTOL1PASS
#define GEN_PASS_DEF_PROMOTEALLOCSTOL1PASS
#define GEN_PASS_DEF_PROMOTEPADSTOL1PASS
#include "Quidditch/Dialect/Snitch/Transforms/Passes.h.inc"
} // namespace quidditch::Snitch

namespace {
class PromoteOperandsToL1
    : public quidditch::Snitch::impl::PromoteOperandsToL1PassBase<
          PromoteOperandsToL1> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};

class PromoteAllocsToL1
    : public quidditch::Snitch::impl::PromoteAllocsToL1PassBase<
          PromoteAllocsToL1> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};

class PromotePadsToL1
    : public quidditch::Snitch::impl::PromotePadsToL1PassBase<PromotePadsToL1> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};

} // namespace

using namespace mlir;
using namespace quidditch::Snitch;
using namespace quidditch::dma;

// A DPS init/accumulator value produced (transitively) by a linalg.fill is fully
// overwritten by that fill, so it must NOT be copied in from its external
// (out-of-L1) output binding: that would read the uninitialized output into L1
// and turn C = A@B into C = C_external + A@B (garbage / X on 4-state RTL, and a
// dependence on device-memory init). Such an accumulator must get a fresh,
// fill-initialized L1 tile instead. Walk through tiling (extract_slice) and
// scf.for iter_args back to the fill.
static bool isInitializedByFill(Value v) {
  while (v) {
    if (auto bbArg = dyn_cast<BlockArgument>(v)) {
      auto forOp =
          dyn_cast_or_null<scf::ForOp>(bbArg.getOwner()->getParentOp());
      if (!forOp || bbArg.getArgNumber() == 0)
        return false;
      v = forOp.getInitArgs()[bbArg.getArgNumber() - 1];
      continue;
    }
    Operation *def = v.getDefiningOp();
    if (!def)
      return false;
    if (isa<linalg::FillOp>(def))
      return true;
    if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(def)) {
      v = sliceOp.getSource();
      continue;
    }
    return false;
  }
  return false;
}

void PromoteOperandsToL1::runOnOperation() {
  // Copy all tensors used as operands to compute ops into L1 memory.
  getOperation()->walk([&](TilingInterface computeOp) {
    // A linalg.fill fully overwrites its outs, so it must never read that outs in
    // from external memory. After workgroup tiling the fill's outs is a slice of
    // the output binding (the per-workgroup C block); copying it in reads
    // uninitialized memory (X on the 4-state co-sim) even though the fill
    // overwrites it. Redirect the fill to a fresh tensor.empty (as in the untiled
    // case) so the promotion below is a bare L1 alloc, not an external copy-in.
    if (auto fillOp = dyn_cast<linalg::FillOp>(computeOp.getOperation())) {
      OpOperand *outs = fillOp.getDpsInitOperand(0);
      auto ty = dyn_cast<RankedTensorType>(outs->get().getType());
      if (ty && ty.hasStaticShape() &&
          !isa_and_nonnull<tensor::EmptyOp>(outs->get().getDefiningOp())) {
        OpBuilder b(fillOp);
        outs->set(b.create<tensor::EmptyOp>(fillOp.getLoc(), ty.getShape(),
                                            ty.getElementType()));
      }
    }

    // Note: This can create redundant copies that must be cleaned up by CSE.
    SmallVector<OpOperand *> nonL1Uses;
    auto dpsOp =
        dyn_cast<DestinationStyleOpInterface>(computeOp.getOperation());
    for (OpOperand &operand : computeOp->getOpOperands()) {
      if (!isa<RankedTensorType>(operand.get().getType()))
        continue;
      // Skip a fill-initialized DPS init operand: copying it in would read the
      // uninitialized external output binding into L1 (see isInitializedByFill).
      if (dpsOp && dpsOp.isDpsInit(&operand) &&
          isInitializedByFill(operand.get()))
        continue;
      nonL1Uses.push_back(&operand);
    }

    auto builder = OpBuilder(computeOp);
    for (OpOperand *use : nonL1Uses) {
      auto copyOp = builder.create<StartTensorCopyOp>(
          computeOp.getLoc(),
          /*copy=*/use->get(), builder.getAttr<L1EncodingAttr>());
      auto waitOp = builder.create<WaitForTensorCopyOp>(
          computeOp.getLoc(), copyOp.getResult(), copyOp.getToken(),
          /*copy=*/use->get());
      use->set(waitOp);
    }
  });
}

void PromoteAllocsToL1::runOnOperation() {
  // Change all allocas so far to be L1 allocations.
  getOperation()->walk([&](bufferization::AllocTensorOp tensorOp) {
    if (!tensorOp.getCopy()) {
      tensorOp.setMemorySpaceAttr(L1EncodingAttr::get(tensorOp.getContext()));
      return;
    }

    OpBuilder builder(tensorOp);
    auto copyOp =
        builder.create<StartTensorCopyOp>(tensorOp.getLoc(), tensorOp.getCopy(),
                                          builder.getAttr<L1EncodingAttr>());
    auto waitOp = builder.create<WaitForTensorCopyOp>(
        tensorOp.getLoc(), copyOp.getResult(), copyOp.getToken(),
        /*copy=*/tensorOp.getCopy());
    tensorOp.replaceAllUsesWith(waitOp.getResult());
    tensorOp.erase();
  });
}

void PromotePadsToL1::runOnOperation() {
  getOperation()->walk([&](tensor::PadOp padOp) {
    // 'start_tensor_copy' does not yet support lower padding.
    if (!padOp.hasZeroLowPad())
      return;

    Value constant = padOp.getConstantPaddingValue();
    if (!constant)
      return;

    // 'start_tensor_copy' supports zero-padding and undef-padding right now.
    // Poison (undef) can also be lowered to perform zero-padding.
    if (!matchPattern(constant, m_NonZero()) &&
        !matchPattern(constant, m_PosZeroFloat()) &&
        !matchPattern(constant, m_Constant<ub::PoisonAttr>(nullptr)))
      return;
    bool undefPadding =
        matchPattern(constant, m_Constant<ub::PoisonAttr>(nullptr));

    OpBuilder builder(padOp);
    auto copyOp = builder.create<StartTensorCopyOp>(
        padOp.getLoc(), padOp.getType(), builder.getType<TokenType>(),
        padOp.getSource(), builder.getAttr<L1EncodingAttr>(), padOp.getHigh(),
        padOp.getStaticHighAttr(), undefPadding);
    auto waitOp = builder.create<WaitForTensorCopyOp>(
        padOp.getLoc(), copyOp.getResult(), copyOp.getToken(),
        /*copy=*/padOp.getSource());
    padOp.replaceAllUsesWith(waitOp.getResult());
    padOp.erase();
  });
}
