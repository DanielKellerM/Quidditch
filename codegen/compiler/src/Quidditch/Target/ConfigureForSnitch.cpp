#include "Passes.h"

#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

namespace quidditch {
#define GEN_PASS_DEF_CONFIGUREFORSNITCHPASS
#include "Quidditch/Target/Passes.h.inc"
} // namespace quidditch

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {
class ConfigureForSnitch
    : public quidditch::impl::ConfigureForSnitchPassBase<ConfigureForSnitch> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};
} // namespace

static LogicalResult setTranslationInfo(FunctionOpInterface funcOp) {
  return setTranslationInfo(
      funcOp,
      IREE::Codegen::TranslationInfoAttr::get(
          funcOp.getContext(),
          IREE::Codegen::DispatchLoweringPassPipeline::None, SymbolRefAttr()));
}

// Built-in per-dispatch tiling seed. The --iree-quidditch-config-table flag
// overrides this with a committed JSON file, so the autotuner can deposit tuned
// tilings with no compiler rebuild. Keys are the LIVE dispatch symbol
// (post-IREE-v3.11.0): main_dispatch_<N>_<linalgop>_<MxNxK>_f<bits>, e.g.
// main_dispatch_0_matmul_16x16x16_f64. l1_tiles[1] = rows over the 8 compute
// cores; l1_tiles[2] = reduction columns staged into L1.
//
// Empty on purpose: the former hardcoded nsnet2 tilings keyed on the
// pre-v3.11.0 main$async_dispatch_..._matmul_transpose_b_... form, which no
// longer matches any emitted dispatch (verified via the twomm proxy), so they
// silently never applied. Dropped rather than left as dead keys; re-derive with
// live names once nsnet2 compiles again (values preserved in git at 279a976).
static const char *kSeedConfigTable = R"json({})json";

// Override the tiling for `name` from the config table (a JSON file path, or the
// built-in seed when empty). Missing key or bad JSON leaves the defaults.
static void applyConfigTable(StringRef name, StringRef configTable,
                             SmallVectorImpl<int64_t> &workgroupTiles,
                             SmallVectorImpl<int64_t> &l1Tiles,
                             SmallVectorImpl<int64_t> &l1Interchange,
                             bool &dualBuffer) {
  std::string fileBuf;
  StringRef text = kSeedConfigTable;
  if (!configTable.empty())
    if (auto buf = llvm::MemoryBuffer::getFile(configTable)) {
      fileBuf = (*buf)->getBuffer().str();
      text = fileBuf;
    }
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(text);
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return;
  }
  const llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return;
  const llvm::json::Object *e = root->getObject(name);
  if (!e)
    return;
  auto getVec = [&](StringRef k, SmallVectorImpl<int64_t> &out) {
    if (const llvm::json::Array *a = e->getArray(k)) {
      out.assign(a->size(), 0);
      for (size_t i = 0; i < a->size(); ++i)
        out[i] = (*a)[i].getAsInteger().value_or(0);
    }
  };
  getVec("workgroup_tiles", workgroupTiles);
  getVec("l1_tiles", l1Tiles);
  getVec("l1_tiles_interchange", l1Interchange);
  if (std::optional<bool> b = e->getBoolean("dual_buffer"))
    dualBuffer = *b;
}

static LogicalResult setRootConfig(FunctionOpInterface funcOp, Operation *rootOp,
                                   StringRef configTable) {
  return TypeSwitch<Operation *, LogicalResult>(rootOp)
      .Case<linalg::MatmulTransposeBOp>([&](linalg::MatmulTransposeBOp op) {
        (void)op;
        SmallVector<int64_t> workgroupTiles(3, 0);
        SmallVector<int64_t> l1Tiles(3, 0);
        SmallVector<int64_t> l1Interchange = {2, 0, 1};
        bool dualBuffer = true;

        applyConfigTable(funcOp.getName(), configTable, workgroupTiles, l1Tiles,
                         l1Interchange, dualBuffer);

        setLoweringConfig(rootOp, quidditch::Snitch::LoweringConfigAttr::get(
                                      rootOp->getContext(), workgroupTiles,
                                      l1Tiles, l1Interchange, dualBuffer));
        return success();
      })
      .Default(success());
}

void ConfigureForSnitch::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  if (getTranslationInfo(funcOp))
    return;

  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  FailureOr<Operation *> rootOp = getRootOperation(computeOps);
  if (failed(rootOp))
    return signalPassFailure();
  Operation *rootOperation = rootOp.value();
  if (!rootOperation)
    return;

  // Set the same translation info for all functions right now.
  // This should move into 'setRootConfig' if we gain different pass pipelines
  // for different kernels.
  if (failed(setTranslationInfo(funcOp)))
    return signalPassFailure();

  auto loweringConfig =
      getLoweringConfig<quidditch::Snitch::LoweringConfigAttr>(rootOperation);
  if (!loweringConfig)
    if (failed(setRootConfig(funcOp, rootOperation, configTable)))
      return signalPassFailure();

  // The root configuration setting introduces `tensor.dim` operations.
  // Resolve those away.
  RewritePatternSet patterns(funcOp.getContext());
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
    signalPassFailure();
}
