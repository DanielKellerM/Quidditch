#include "Passes.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"

namespace quidditch {
#define GEN_PASS_DEF_MATERIALIZEEXECUTABLEFROMFLOWPASS
#include "Quidditch/Target/Passes.h.inc"
} // namespace quidditch

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {
class MaterializeExecutableFromFlow
    : public quidditch::impl::MaterializeExecutableFromFlowPassBase<
          MaterializeExecutableFromFlow> {
public:
  using Base::Base;

  void runOnOperation() override;
};
} // namespace

// Collect the distinct executable targets the module's device globals support.
static SmallVector<IREE::HAL::ExecutableTargetAttr>
gatherExecutableTargets(ModuleOp moduleOp) {
  SetVector<IREE::HAL::ExecutableTargetAttr> targets;
  for (auto globalOp : moduleOp.getOps<IREE::Util::GlobalOp>()) {
    auto deviceTarget = dyn_cast_or_null<IREE::HAL::DeviceTargetAttr>(
        globalOp.getInitialValueAttr());
    if (deviceTarget)
      deviceTarget.getExecutableTargets(targets);
  }
  return SmallVector<IREE::HAL::ExecutableTargetAttr>(targets.begin(),
                                                      targets.end());
}

// The storage-buffer pipeline binding for one dispatch.tensor arg; flags mirror access.
static FailureOr<IREE::HAL::PipelineBindingAttr>
bindingForArg(MLIRContext *ctx, Type argType, Location loc) {
  auto dtType = dyn_cast<IREE::TensorExt::DispatchTensorType>(argType);
  if (!dtType)
    return failure();
  if (!dtType.hasStaticShape())
    return emitError(loc, "dynamic-shape dispatch tensors are unsupported: the Snitch "
                          "codegen (pad-to-tiling-config) requires static shapes");
  auto flags = IREE::HAL::DescriptorFlags::Indirect;
  if (dtType.getAccess() == IREE::TensorExt::TensorAccess::ReadOnly)
    flags = flags | IREE::HAL::DescriptorFlags::ReadOnly;
  return IREE::HAL::PipelineBindingAttr::get(
      ctx, IREE::HAL::DescriptorType::StorageBuffer, flags);
}

// Rewrite dispatch.tensor args to binding.subspan and non-tensor args to constant.load.
static LogicalResult
rewriteFuncInterface(func::FuncOp funcOp,
                     IREE::HAL::PipelineLayoutAttr layoutAttr) {
  Block &entry = funcOp.front();
  OpBuilder builder = OpBuilder::atBlockBegin(&entry);
  Value zero =
      arith::ConstantIndexOp::create(builder, funcOp.getLoc(), 0).getResult();
  IntegerAttr align64 = builder.getIndexAttr(64);

  int64_t bindingOrdinal = 0;
  int64_t constantOrdinal = 0;
  for (BlockArgument arg : entry.getArguments()) {
    auto dtType = dyn_cast<IREE::TensorExt::DispatchTensorType>(arg.getType());
    if (dtType) {
      // Static-shape already enforced during layout derivation (bindingForArg).
      auto bindingAttr = layoutAttr.getBinding(bindingOrdinal);
      auto subspan = IREE::HAL::InterfaceBindingSubspanOp::create(
          builder, arg.getLoc(), arg.getType(), layoutAttr,
          APInt(64, bindingOrdinal), zero, /*dynamic_dims=*/ValueRange{},
          align64, bindingAttr.getFlags());
      arg.replaceAllUsesWith(subspan.getResult());
      ++bindingOrdinal;
    } else {
      auto load = IREE::HAL::InterfaceConstantLoadOp::create(
          builder, arg.getLoc(), arg.getType(), layoutAttr,
          builder.getIndexAttr(constantOrdinal), /*alignment=*/IntegerAttr{},
          /*values=*/ArrayAttr{});
      arg.replaceAllUsesWith(load);
      ++constantOrdinal;
    }
  }
  entry.eraseArguments([](BlockArgument) { return true; });
  funcOp.setType(FunctionType::get(funcOp.getContext(), {}, {}));
  return success();
}

void MaterializeExecutableFromFlow::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *ctx = &getContext();

  SmallVector<IREE::HAL::ExecutableTargetAttr> targets =
      gatherExecutableTargets(moduleOp);
  if (targets.empty()) {
    moduleOp.emitError("no #hal.executable.target found on any device global; "
                       "was a device assigned before this pass?");
    return signalPassFailure();
  }

  SmallVector<IREE::Flow::ExecutableOp> flowExecutables(
      moduleOp.getOps<IREE::Flow::ExecutableOp>());

  for (IREE::Flow::ExecutableOp flowExe : flowExecutables) {
    OpBuilder moduleBuilder(flowExe);
    auto halExe = IREE::HAL::ExecutableOp::create(moduleBuilder,
                                                  flowExe.getLoc(),
                                                  flowExe.getName());
    halExe.setVisibility(flowExe.getVisibility());
    SymbolTable halExeSymbols(halExe);

    ModuleOp innerModule = flowExe.getInnerModule();

    for (IREE::HAL::ExecutableTargetAttr target : targets) {
      OpBuilder variantBuilder(&halExe.getBlock().back());
      auto variantOp = IREE::HAL::ExecutableVariantOp::create(
          variantBuilder, flowExe.getLoc(), target.getSymbolNameFragment(),
          target);
      halExeSymbols.insert(variantOp);
      OpBuilder inVariant(&variantOp.getBlock().back());

      llvm::StringMap<IREE::HAL::PipelineLayoutAttr> funcLayouts;
      int64_t ordinal = 0;
      for (auto flowExport :
           flowExe.getBlock().getOps<IREE::Flow::ExecutableExportOp>()) {
        func::FuncOp srcFunc =
            innerModule.lookupSymbol<func::FuncOp>(flowExport.getFunctionRef());
        if (!srcFunc) {
          flowExport.emitError("export references missing function '")
              << flowExport.getFunctionRef() << "'";
          return signalPassFailure();
        }

        // Derive the pipeline layout from the dispatch func signature.
        SmallVector<IREE::HAL::PipelineBindingAttr> bindings;
        int64_t constantCount = 0;
        for (Type argType : srcFunc.getArgumentTypes()) {
          if (isa<IREE::TensorExt::DispatchTensorType>(argType)) {
            FailureOr<IREE::HAL::PipelineBindingAttr> b =
                bindingForArg(ctx, argType, srcFunc.getLoc());
            if (failed(b))
              return signalPassFailure();
            bindings.push_back(*b);
          } else {
            ++constantCount;
          }
        }
        auto layoutAttr = IREE::HAL::PipelineLayoutAttr::get(
            ctx, bindings, constantCount,
            IREE::HAL::PipelineLayoutFlags::Indirect);
        funcLayouts[flowExport.getFunctionRef()] = layoutAttr;

        auto halExport = IREE::HAL::ExecutableExportOp::create(
            inVariant, flowExport.getLoc(),
            inVariant.getStringAttr(flowExport.getFunctionRef()),
            inVariant.getIndexAttr(ordinal++), layoutAttr,
            /*workgroup_size=*/ArrayAttr{}, /*subgroup_size=*/IntegerAttr{},
            /*workgroup_local_memory=*/IntegerAttr{});

        // Port the workgroup-count region, inserting the !hal.device argument.
        if (!flowExport.getWorkgroupCount().empty()) {
          IRMapping mapper;
          flowExport.getWorkgroupCount().cloneInto(
              &halExport.getWorkgroupCount(), mapper);
          Type deviceType = inVariant.getType<IREE::HAL::DeviceType>();
          if (!llvm::is_contained(
                  flowExport.getWorkgroupCount().getArgumentTypes(), deviceType))
            halExport.getWorkgroupCount().insertArgument(0u, deviceType,
                                                         halExport.getLoc());
          // flow.return -> hal.return inside the cloned region.
          halExport.getWorkgroupCount().walk([&](IREE::Flow::ReturnOp ret) {
            OpBuilder rb(ret);
            IREE::HAL::ReturnOp::create(rb, ret.getLoc(), ret.getOperands());
            ret.erase();
          });
        }
      }

      // Clone the inner module (with the rewritten funcs) into the variant.
      OpBuilder::InsertionGuard g(inVariant);
      auto variantModule = ModuleOp::create(inVariant, variantOp.getLoc());
      OpBuilder modBuilder = OpBuilder::atBlockBegin(variantModule.getBody());
      for (auto srcFunc : innerModule.getOps<func::FuncOp>()) {
        auto cloned = cast<func::FuncOp>(modBuilder.clone(*srcFunc));
        auto it = funcLayouts.find(cloned.getName());
        if (it == funcLayouts.end())
          continue; // a helper func with no export: leave as-is
        if (failed(rewriteFuncInterface(cloned, it->second)))
          return signalPassFailure();
      }
    }

    flowExe.erase();
  }

  // Drop the host-side dispatch callers: this is a device-only compiler.
  for (auto funcOp :
       llvm::make_early_inc_range(moduleOp.getOps<IREE::Util::FuncOp>())) {
    bool hasDispatch = false;
    funcOp.walk([&](IREE::Flow::DispatchOp) { hasDispatch = true; });
    if (hasDispatch)
      funcOp.erase();
  }
}
