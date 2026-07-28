#include "Passes.h"

#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
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

// Built-in per-dispatch tiling seed (empty by default; the autotuner-supplied
// --iree-quidditch-config-table overrides it with no compiler rebuild). Keys are
// the live dispatch symbol main_dispatch_<N>_<linalgop>_<MxNxK>_f<bits> (e.g.
// main_dispatch_0_matmul_16x16x16_f64); l1_tiles[1] = rows over the compute
// cores, l1_tiles[2] = reduction columns staged into L1.
static const char *kSeedConfigTable = R"json({})json";

// Override the tiling for funcOp from the config table (a JSON file path, or the
// built-in seed when empty); a missing key keeps the defaults, a broken explicit
// path or malformed JSON warns and keeps them.
static void applyConfigTable(FunctionOpInterface funcOp, StringRef configTable,
                             SmallVectorImpl<int64_t> &workgroupTiles,
                             SmallVectorImpl<int64_t> &l1Tiles,
                             SmallVectorImpl<int64_t> &l1Interchange,
                             bool &dualBuffer) {
  std::string fileBuf;
  StringRef text = kSeedConfigTable;
  if (!configTable.empty()) {
    auto buf = llvm::MemoryBuffer::getFile(configTable);
    if (!buf) {
      funcOp->emitWarning() << "config table '" << configTable
                            << "' could not be opened; using default tiling";
      return;
    }
    fileBuf = (*buf)->getBuffer().str();
    text = fileBuf;
  }
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(text);
  if (!parsed) {
    if (configTable.empty())
      llvm::consumeError(parsed.takeError());
    else
      funcOp->emitWarning() << "config table '" << configTable
                            << "' is not valid JSON, using default tiling: "
                            << llvm::toString(parsed.takeError());
    return;
  }
  const llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return;
  const llvm::json::Object *e = root->getObject(funcOp.getName());
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
  // Match any matmul-like contraction, not a specific named op: transpose-b is
  // carried by indexing_maps on a `linalg.matmul` (the sample) or on the
  // `linalg.generic` a StableHLO `dot_general` legalizes to -- both are
  // contractions. Per-shape tiling comes from the config table (keyed by the
  // dispatch symbol, which encodes MxNxK); absent an entry the tiles default to 0.
  auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp);
  if (!linalgOp || !linalg::isaContractionOpInterface(linalgOp))
    return success();

  SmallVector<int64_t> workgroupTiles(3, 0);
  SmallVector<int64_t> l1Tiles(3, 0);
  SmallVector<int64_t> l1Interchange = {2, 0, 1};
  bool dualBuffer = true;

  applyConfigTable(funcOp, configTable, workgroupTiles, l1Tiles,
                   l1Interchange, dualBuffer);

  setLoweringConfig(rootOp, quidditch::Snitch::LoweringConfigAttr::get(
                                rootOp->getContext(), workgroupTiles, l1Tiles,
                                l1Interchange, dualBuffer));
  return success();
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
