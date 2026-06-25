
#include "iree/compiler/PluginAPI/Client.h"

#include "compiler/plugins/target/LLVMCPU/LLVMIRPasses.h"
#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree/compiler/Codegen/Common/CPU/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "Quidditch/Conversion/Passes.h"
#include "Quidditch/Dialect/DMA/Extensions/DMACoreSpecializationOpInterfaceImpl.h"
#include "Quidditch/Dialect/DMA/IR/DMADialect.h"
#include "Quidditch/Dialect/DMA/IR/DMAOps.h"
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.h"
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchOps.h"
#include "Quidditch/Dialect/Snitch/Transforms/Passes.h"
#include "Quidditch/Dialect/SnitchDMA/IR/SnitchDMADialect.h"
#include "Quidditch/Dialect/SnitchDMA/Transforms/Passes.h"

#include "compiler/plugins/target/LLVMCPU/LinkerTool.h"
#include "compiler/plugins/target/LLVMCPU/StaticLibraryGenerator.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"

#include <optional>

#include "LibraryBuilder.h"
#include "Passes.h"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace quidditch::Snitch;

namespace {

// Find `#define <name>` in a generated C header, robust to // comments and
// whitespace: scan LINES (so a commented `// #define ...` is skipped), tolerate any
// spacing after `#define`/the name, require a word boundary after the name (so NAME
// is not matched as a prefix of NAMEX), and take the FIRST uncommented match (a
// commented duplicate cannot shadow a real define). Returns the line remainder after
// the name (the value, or "" for a bare presence flag), or nullopt if not defined.
static std::optional<StringRef> cfgDefineBody(StringRef text, StringRef name) {
  auto isWord = [](char c) {
    return c == '_' || (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
           (c >= 'A' && c <= 'Z');
  };
  SmallVector<StringRef> lines;
  text.split(lines, '\n');
  for (StringRef line : lines) {
    StringRef s = line.ltrim();
    if (!s.consume_front("#define"))
      continue; // not a #define (also skips "// #define ..." -- starts with //)
    if (s.empty() || isWord(s.front()))
      continue; // require whitespace after #define (reject "#defineX")
    s = s.ltrim();
    if (!s.consume_front(name))
      continue;
    if (!s.empty() && isWord(s.front()))
      continue; // word boundary: reject NAME matched as a prefix of NAMEX
    return s.ltrim();
  }
  return std::nullopt;
}

// Parse `#define <name> <int>` (decimal/hex) from the header.
static std::optional<int64_t> cfgHeaderDefine(StringRef text, StringRef name) {
  std::optional<StringRef> body = cfgDefineBody(text, name);
  if (!body)
    return std::nullopt;
  int64_t v;
  if (body->consumeInteger(/*Radix=*/0, v)) // consumeInteger returns true on FAILURE
    return std::nullopt;
  return v;
}

// Is `#define <name>` present (a presence flag like SNRT_SUPPORTS_SSR)?
static bool cfgHeaderDefined(StringRef text, StringRef name) {
  return cfgDefineBody(text, name).has_value();
}

// Hardware parameters read from the cfg-generated snitch_cluster_cfg.h (the same
// artifact the C runtime consumes) so the codegen TARGET matches the RTL the cfg
// built, instead of baked literals. compute_cores is CONSUMED (the tiling divisor,
// TensorTile/LowerForall); the rest are PLACEHOLDERS -- read from the cfg + threaded
// into the target attr, but their consumers are TODO (future work, not forgotten).
// Coverage table + cfg keys: CFG_DRIVEN_TARGET.md.
struct ClusterParams {
  unsigned computeCores = 8;   // CONSUMED: tiling divisor (TensorTile/LowerForall).
  int64_t tcdmBytes = 0;       // SNRT_TCDM_SIZE (0 = absent).
                               // TODO(cfg-target): -> l1MemoryBytes, BLOCKED on the
                               //   100000-vs-112640 DMA-stack overflow.
  int64_t seqLoops = 0;        // SNRT_NUM_SEQUENCER_LOOPS.
                               // TODO(cfg-target): FREP nesting bound -- no consumer
                               //   yet (>2 nesting is unreachable in the lowering).
  int64_t seqInsns = 0;        // SNRT_NUM_SEQUENCER_INSNS.
                               // CONSUMED: bounds the xDSL FREP body size.
  bool supportsSSR = true;     // SNRT_SUPPORTS_SSR. CONSUMED (xDSL SSR pass gate).
  bool supportsFREP = true;    // SNRT_SUPPORTS_FREP. CONSUMED (xDSL FREP pass gate).
  bool supportsDivSqrt = true; // SNRT_SUPPORTS_DIVSQRT (Xdiv_sqrt). CONSUMED:
                               //   refuse arith.divf when absent (xDSL).
};

static ClusterParams clusterParamsFromCfgHeader(StringRef headerPath) {
  ClusterParams p;
  if (headerPath.empty())
    return p;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buf =
      llvm::MemoryBuffer::getFile(headerPath);
  if (!buf)
    return p;
  StringRef text = (*buf)->getBuffer();
  std::optional<int64_t> nr = cfgHeaderDefine(text, "CFG_CLUSTER_NR_CORES");
  std::optional<int64_t> dm = cfgHeaderDefine(text, "SNRT_CLUSTER_DM_CORE_NUM");
  if (nr && dm && *nr > *dm)
    p.computeCores = static_cast<unsigned>(*nr - *dm);
  if (std::optional<int64_t> t = cfgHeaderDefine(text, "SNRT_TCDM_SIZE"))
    p.tcdmBytes = *t;
  if (std::optional<int64_t> l = cfgHeaderDefine(text, "SNRT_NUM_SEQUENCER_LOOPS"))
    p.seqLoops = *l;
  if (std::optional<int64_t> i = cfgHeaderDefine(text, "SNRT_NUM_SEQUENCER_INSNS"))
    p.seqInsns = *i;
  p.supportsSSR = cfgHeaderDefined(text, "SNRT_SUPPORTS_SSR");
  p.supportsFREP = cfgHeaderDefined(text, "SNRT_SUPPORTS_FREP");
  p.supportsDivSqrt = cfgHeaderDefined(text, "SNRT_SUPPORTS_DIVSQRT");
  return p;
}

class QuidditchTargetDevice final : public IREE::HAL::TargetDevice {
public:
  IREE::HAL::DeviceTargetAttr getDefaultDeviceTarget(
      MLIRContext *context,
      const IREE::HAL::TargetRegistry &targetRegistry) const override {
    Builder b(context);

    // If we had multiple target environments we would generate one target attr
    // per environment, with each setting its own environment attribute.
    SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
    targetRegistry.getTargetBackend("quidditch")
        ->getDefaultExecutableTargets(context, "quidditch",
                                      b.getDictionaryAttr({}),
                                      executableTargetAttrs);
    return IREE::HAL::DeviceTargetAttr::get(
        context, b.getStringAttr("quidditch_device"), b.getDictionaryAttr({}),
        executableTargetAttrs);
  }
};

struct QuidditchTargetOptions {
  std::string staticLibraryOutputPath;
  std::string xDSLOptPath;
  std::string xDSLPasses = "arith-add-fastmath,test-lower-linalg-to-snitch";
  std::string configTable;
  std::string clusterCfgHeader;
  std::string toolChainRoot;
  bool assertCompiled = false;
  // TODO: This should actually be 112640 but DMA stack overflows. Ooopsie!
  unsigned l1MemoryBytes = 100000;

  void bindOptions(OptionsBinder &binder) {
    LLVMInitializeRISCVTarget();
    LLVMInitializeRISCVTargetMC();
    LLVMInitializeRISCVTargetInfo();
    LLVMInitializeRISCVAsmPrinter();
    LLVMInitializeRISCVAsmParser();

    static llvm::cl::OptionCategory category("Quidditch HAL Target");

    binder.opt<std::string>(
        "iree-quidditch-static-library-output-path", staticLibraryOutputPath,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "Path to output static object (EX: '/path/to/static-library.o'). "
            "This will produce the static library at the specified path along "
            "with a similarly named '.h' file for static linking."));
    binder.opt<std::string>("iree-quidditch-xdsl-opt-path", xDSLOptPath,
                            llvm::cl::cat(category),
                            llvm::cl::desc("Path to the 'xdsl-opt' executable "
                                           "to use for kernel compilation."));
    binder.opt<std::string>("iree-quidditch-xdsl-passes", xDSLPasses,
                            llvm::cl::cat(category),
                            llvm::cl::desc("The xdsl-opt pass pipeline (-p) used "
                                           "to lower each kernel; the autotuner "
                                           "sweeps Group-B pass knobs via this."));
    binder.opt<std::string>("iree-quidditch-config-table", configTable,
                            llvm::cl::cat(category),
                            llvm::cl::desc("Path to a JSON per-dispatch tiling "
                                           "table for ConfigureForSnitch; empty "
                                           "uses the built-in seed. The autotuner "
                                           "upserts tuned tilings here."));
    binder.opt<std::string>(
        "iree-quidditch-cluster-cfg-header", clusterCfgHeader,
        llvm::cl::cat(category),
        llvm::cl::desc("Path to the cfg-generated snitch_cluster_cfg.h; the "
                       "codegen compute_cores is derived from it "
                       "(CFG_CLUSTER_NR_CORES - SNRT_CLUSTER_DM_CORE_NUM) so the "
                       "tiling divisor matches the RTL. Empty defaults to 8."));
    binder.opt<std::string>(
        "iree-quidditch-toolchain-root", toolChainRoot, llvm::cl::cat(category),
        llvm::cl::desc("Path to the root directory of the Quidditch toolchain "
                       "(containing the toolchain file)"));
    binder.opt<bool>(
        "iree-quidditch-assert-compiled", assertCompiled,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "If true, errors if any kernel could not be compiled with xDSL."
            "Otherwise, removes the kernel from the output and emits a warning "
            "instead."));
    binder.opt<unsigned>(
        "iree-quidditch-l1-memory-bytes", l1MemoryBytes,
        llvm::cl::cat(category),
        llvm::cl::desc("Size of the useable L1 memory in bytes"));
  }
};

class QuidditchTargetBackend final : public IREE::HAL::TargetBackend {
public:
  explicit QuidditchTargetBackend(QuidditchTargetOptions options)
      : targetOptions(std::move(options)) {}

  [[nodiscard]] std::string getLegacyDefaultDeviceID() const override {
    return "quidditch_device";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);
    quidditch::dma::registerDMACoreSpecializationOpInterface(registry);

    registry.insert<arm_neon::ArmNeonDialect, arm_sme::ArmSMEDialect,
                    quidditch::Snitch::QuidditchSnitchDialect,
                    quidditch::dma::DMADialect,
                    quidditch::SnitchDMA::SnitchDMADialect>();
  }

  void getDefaultExecutableTargets(
      MLIRContext *context, StringRef deviceID, DictionaryAttr deviceConfigAttr,
      SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs)
      const override {

    NamedAttrList list;
    // Target attribute info used by the LLVM lowering.
    // TODO: Ideally shouldn't be hardcoded.
    list.append("data_layout",
                StringAttr::get(context, "e-m:e-p:32:32-i64:64-n32-S128"));
    list.append("target_triple",
                StringAttr::get(context, "riscv32-unknown-elf"));
    // HW params, cfg-derived from the cluster header when the flag is given (so the
    // codegen target matches the RTL the cfg built), else defaults. INVARIANT: a cfg
    // param is emitted into the target dict ONLY once it has a CONSUMER. Emitting an
    // unconsumed attr has no codegen effect -- it only bloats IREE's pretty-printed
    // target-attr diagnostic string (.rodata) and would perturb the stripped-ELF
    // canary for nothing. compute_cores is CONSUMED (the tiling divisor,
    // TensorTile/LowerForall), so it is emitted unconditionally and tracks the cfg.
    // The other ClusterParams (tcdm_bytes, sequencer_*, supports_*) are read from the
    // cfg but NOT emitted until their consumers land; each is re-added here together
    // with its consumer (TODO(cfg-target) on ClusterParams + CFG_DRIVEN_TARGET.md).
    ClusterParams cp = clusterParamsFromCfgHeader(targetOptions.clusterCfgHeader);
    IntegerType i32 = IntegerType::get(context, 32);
    list.append("compute_cores", IntegerAttr::get(i32, cp.computeCores));
    // CONSUMED in buildTranslationPassPipeline (supports_ssr/frep gate the xDSL
    // SSR/FREP passes; sequencer_insns bounds the FREP body size). Emitted only with
    // the cfg flag; absent (no flag) reads as supported / the xDSL default.
    if (!targetOptions.clusterCfgHeader.empty()) {
      list.append("supports_ssr", BoolAttr::get(context, cp.supportsSSR));
      list.append("supports_frep", BoolAttr::get(context, cp.supportsFREP));
      list.append("supports_div_sqrt", BoolAttr::get(context, cp.supportsDivSqrt));
      if (cp.seqInsns)
        list.append("sequencer_insns", IntegerAttr::get(i32, cp.seqInsns));
    }
    executableTargetAttrs.push_back(IREE::HAL::ExecutableTargetAttr::get(
        context, StringAttr::get(context, "quidditch"),
        StringAttr::get(context, "snitch"), list.getDictionary(context)));
  }

  void
  buildConfigurationPassPipeline(IREE::HAL::ExecutableTargetAttr targetAttr,
                                 OpPassManager &passManager) override {
    OpPassManager &modulePassManager = passManager.nest<ModuleOp>();
    {
      FunctionLikeNest funcPassManager(modulePassManager);
      addCommonTargetExecutablePreprocessingPasses(
          funcPassManager,
          /*useDecomposeSoftmaxFusion=*/false);
    }
    modulePassManager.addPass(createMaterializeUserConfigsPass());
    FunctionLikeNest funcPassManager(modulePassManager);
    funcPassManager.addPass([this] {
      quidditch::ConfigureForSnitchPassOptions opts;
      opts.configTable = targetOptions.configTable;
      return quidditch::createConfigureForSnitchPass(opts);
    });
  }

  void buildTranslationPassPipeline(IREE::HAL::ExecutableTargetAttr targetAttr,
                                    OpPassManager &passManager) override {
    OpPassManager &modulePassManager = passManager.nest<ModuleOp>();

    FunctionLikeNest(modulePassManager)
        .addPass(
            [] { return createTileAndDistributeToWorkgroupsUsingForallOpPass(); })
        .addPass([] { return createBufferizeDispatchTensorLoadStorePass(); })
        .addPass(quidditch::createPadToTilingConfigPass)
        .addPass(createFoldAffineMinInDistributedLoopsPass)
        .addPass(quidditch::createRemoveTrivialLoopsPass)
        .addPass(createCanonicalizerPass)
        .addPass(createCSEPass)
        .addPass(createFuseTensorPadWithConsumerPass)
        .addPass(createConcretizePadResultShapePass)
        .addPass([] {
          return quidditch::createTensorTilePass({quidditch::TilingLevel::L1});
        })
        .addPass(createFuseTensorPadWithConsumerPass)
        .addPass(createConcretizePadResultShapePass)
        .addPass(quidditch::Snitch::createPromotePadsToL1Pass)
        .addPass(quidditch::Snitch::createPromoteOperandsToL1Pass)
        .addPass(createCanonicalizerPass)
        .addPass(createCSEPass)
        .addPass(createLoopInvariantCodeMotionPass)
        .addPass(quidditch::Snitch::createPipelineCopyComputePass)
        // TODO: Fuse scf.forall after.
        .addPass([] {
          return quidditch::createTensorTilePass(
              {quidditch::TilingLevel::Thread});
        })
        .addPass(createCanonicalizerPass)
        .addPass(createCSEPass)
        .addPass(quidditch::Snitch::createFormMicrokernelsPass);

    BufferizationOptions::AllocationFn allocationFn =
        [](OpBuilder &builder, Location loc, MemRefType memRefType,
           ValueRange dynamicSizes, unsigned alignment) -> Value {
      return builder.create<memref::AllocaOp>(
          loc, memRefType, dynamicSizes, builder.getI64IntegerAttr(alignment));
    };
    BufferizationOptions::MemCpyFn memcpyFn =
        [](OpBuilder &builder, Location loc, Value from, Value to) {
          Value token =
              builder.create<quidditch::dma::StartTransferOp>(loc, from, to);
          builder.create<quidditch::dma::WaitForTransferOp>(loc, token);
          return success();
        };

    FunctionLikeNest(modulePassManager)
        .addPass(createEliminateEmptyTensorsPass)
        .addPass(bufferization::createEmptyTensorToAllocTensorPass)
        .addPass(quidditch::Snitch::createPromoteAllocsToL1Pass)
        .addPass(createCanonicalizerPass)
        .addPass(createCSEPass)
        .addPass([&] {
          return createIREEComprehensiveBufferizePass(allocationFn, memcpyFn);
        });
    addIREEPostBufferizationPasses(modulePassManager.nest<func::FuncOp>());

    FunctionLikeNest(modulePassManager)
        .addPass(quidditch::Snitch::createLowerPipelineOpPass)
        .addPass(quidditch::Snitch::createLowerForallOpPass)
        .addPass(createSCFForLoopCanonicalizationPass)
        .addPass(createCanonicalizerPass)
        .addPass(createCSEPass)
        .addPass(createCanonicalizerPass)
        // TODO: Remove the following pass and plumb support for
        // #hal.descriptor_type memory space through the stack.
        .addPass(createEraseHALDescriptorTypeFromMemRefPass)
        .addPass([&] {
          return quidditch::Snitch::createLowerL1AllocationsPass(
              {targetOptions.l1MemoryBytes, targetOptions.assertCompiled});
        })
        .addPass(quidditch::createReluToMaxPass)
        .addPass(createCanonicalizerPass)
        .addPass(createCSEPass)
        .addPass(createLoopInvariantCodeMotionPass)
        .addPass(createLinalgGeneralizeNamedOpsPass)
        .addPass(quidditch::createRemoveTrivialLoopsPass);

    modulePassManager.addPass(quidditch::Snitch::createSpecializeDMACodePass());
    FunctionLikeNest(modulePassManager)
        .addPass(quidditch::SnitchDMA::createLegalizeDMAOperationsPass)
        .addPass(createCanonicalizerPass)
        .addPass(createCSEPass);
    // Gate the xDSL SSR/FREP passes on the cfg-derived capability attrs. Only
    // rewrite when the target lacks one, so the default cfg (both supported, or the
    // attrs absent = supported) leaves the pipeline string byte-identical. Consumes
    // supports_ssr / supports_frep (emitted by getDefaultExecutableTargets only with
    // the cfg flag).
    std::string xDSLPasses = targetOptions.xDSLPasses;
    {
      DictionaryAttr cfg = targetAttr.getConfiguration();
      auto supported = [&](StringRef key) {
        BoolAttr b = cfg ? cfg.getAs<BoolAttr>(key) : BoolAttr();
        return !b || b.getValue(); // absent -> supported (default)
      };
      SmallVector<std::string> opts;
      if (!supported("supports_ssr"))
        opts.push_back("ssr=false");
      if (!supported("supports_frep"))
        opts.push_back("frep=false");
      if (!supported("supports_div_sqrt"))
        opts.push_back("div_sqrt=false");
      // FREP body-size bound; override the xDSL default (32, the canonical sequencer
      // depth) only for a non-default cfg, so the default leaves the string untouched.
      if (IntegerAttr si =
              cfg ? cfg.getAs<IntegerAttr>("sequencer_insns") : IntegerAttr())
        if (si.getInt() != 32)
          opts.push_back("sequencer_insns=" + std::to_string(si.getInt()));
      if (!opts.empty()) {
        std::string braces = "{";
        for (size_t i = 0; i < opts.size(); ++i) {
          if (i)
            braces += " ";
          braces += opts[i];
        }
        braces += "}";
        std::string tok = "test-lower-linalg-to-snitch";
        std::string::size_type pos = xDSLPasses.find(tok);
        if (pos != std::string::npos)
          xDSLPasses.insert(pos + tok.size(), braces);
        // A custom --iree-quidditch-xdsl-passes without the aggregate owns its own
        // pass selection; the gate cannot apply and is a no-op there by design.
      }
    }
    modulePassManager.addPass(quidditch::createConvertToRISCVPass(
        {targetOptions.xDSLOptPath, targetOptions.assertCompiled, xDSLPasses}));

    FunctionLikeNest(modulePassManager)
        .addPass(IREE::LinalgExt::createLinalgExtToLoopsPass)
        .addPass(createMemrefCopyToLinalgPass)
        .addPass(createConvertLinalgToLoopsPass)
        .addPass(createCanonicalizerPass)
        .addPass(createCSEPass);

    modulePassManager.addPass(createIREEBufferizeConstantsPass());

    FunctionLikeNest(modulePassManager)
        .addPass(createFoldTensorExtractOpPass)
        // Handle complex operation conversion.
        .addPass(createConvertComplexToStandardPass)
        // math dialect elementary functions -> polynomial form.
        .addPass(createMathTransformPass)
        .addPass(createHoistStaticallyBoundAllocationsPass)
        .addPass(createIREEExpandStridedMetadataPass)
        .addPass(createCleanupBufferAllocViewPass);

    FunctionLikeNest(modulePassManager)
        .addPass(arith::createArithExpandOpsPass)
        .addPass(memref::createExpandOpsPass)
        .addPass(memref::createFoldMemRefAliasOpsPass)
        .addPass(createCanonicalizerPass)
        .addPass(createCSEPass);

    modulePassManager.addPass(quidditch::createConvertToLLVMPass());
    modulePassManager.addPass(createReconcileUnrealizedCastsPass());
    // We rely on MLIR symbol visibility being correct after this point and
    // need to mirror the LLVM linkage that was assigned during conversion.
    modulePassManager.addPass(createLLVMCPUSynchronizeSymbolVisibilityPass());

    modulePassManager.addPass(createCanonicalizerPass());
    modulePassManager.addPass(createCSEPass());
    modulePassManager.addNestedPass<LLVM::LLVMFuncOp>(
        createAddFastMathFlagsPass());

    // The forall-based workgroup distribution leaves the export op's
    // 'workgroup_count' region as an abstract 'workgroup_count_from_slice';
    // resolve it to concrete i32 counts (mirrors the LLVMCPU/VMVX variant
    // pipeline) so the host module's command_buffer.dispatch import is satisfied.
    passManager.addPass(createReconcileTranslationInfoPass());
    passManager.addPass(createResolveWorkgroupCountHintsPass());

    passManager.addPass(quidditch::createDisableQuidditchVariantPass());
  }

  void buildLinkingPassPipeline(OpPassManager &passManager) override {
    passManager.addPass(quidditch::createLinkExecutablesPass());
    // Cleanup IR duplication.
    passManager.addNestedPass<IREE::HAL::ExecutableOp>(
        mlir::createCanonicalizerPass());

    // Assign final executable constant and import ordinals.
    auto &variantPassManager = passManager.nest<IREE::HAL::ExecutableOp>()
                                   .nest<IREE::HAL::ExecutableVariantOp>();
    variantPassManager.addPass(createLLVMCPUAssignConstantOrdinalsPass());
    variantPassManager.addPass(createLLVMCPUAssignImportOrdinalsPass());
  }

  FailureOr<SmallVector<IREE::HAL::Artifact>>
  assembleXDSLOutput(ModuleOp module) {

    auto *dialect =
        module.getContext()->getLoadedDialect<QuidditchSnitchDialect>();

    SmallVector<IREE::HAL::Artifact> objectFiles;
    for (auto func :
         llvm::make_early_inc_range(module.getOps<LLVM::LLVMFuncOp>())) {
      auto assembly = dialect->getRiscvAssemblyAttrHelper().getAttr(func);
      if (!assembly)
        continue;

      SmallString<64> stdinFile;
      int stdinFd;
      if (llvm::sys::fs::createTemporaryFile("xdsl-in", "S", stdinFd,
                                             stdinFile)) {
        return failure();
      }
      llvm::FileRemover stdinFileRemove(stdinFile);
      {
        llvm::raw_fd_ostream ss(stdinFd, /*shouldClose=*/true);
        ss << assembly.getValue();
      }

      auto &objectFile = objectFiles.emplace_back(
          IREE::HAL::Artifact::createTemporary("xdsl-out", "o"));
      int ret = llvm::sys::ExecuteAndWait(
          targetOptions.toolChainRoot + "/bin/pulp-as",
          {targetOptions.toolChainRoot + "/bin/pulp-as", "--filetype=obj",
           "--target-abi=ilp32d", stdinFile.str(), "-o", objectFile.path,
           "--mcpu=snitch", "-g"});
      if (ret != 0)
        return failure();
    }
    return objectFiles;
  }

  static std::unique_ptr<llvm::Module>
  toLLVMModule(llvm::LLVMContext &context, ModuleOp module,
               const llvm::TargetMachine &machine,
               IREE::HAL::ExecutableVariantOp variantOp) {
    module->setAttr(
        LLVM::LLVMDialect::getTargetTripleAttrName(),
        StringAttr::get(module.getContext(), machine.getTargetTriple().str()));

    std::string libraryName =
        variantOp->getParentOfType<IREE::HAL::ExecutableOp>().getName().str();

    // At this moment we are leaving MLIR LLVM dialect land translating module
    // into target independent LLVMIR.
    auto llvmModule =
        mlir::translateModuleToLLVMIR(module, context, libraryName);
    if (!llvmModule) {
      module.emitError() << "failed to translate the MLIR LLVM "
                            "dialect to the native llvm::Module";
      return nullptr;
    }

    auto *dialect =
        module.getContext()->getLoadedDialect<QuidditchSnitchDialect>();

    SymbolTable symbolTable(module);
    Quidditch::LibraryBuilder libraryBuilder(
        llvmModule.get(), Quidditch::LibraryBuilder::Mode::NONE,
        Quidditch::LibraryBuilder::Version::LATEST);
    auto align16 = llvm::Attribute::getWithAlignment(context, llvm::Align(16));
    for (auto exportOp :
         variantOp.getBlock().getOps<IREE::HAL::ExecutableExportOp>()) {
      // Find the matching function in the LLVM module.
      auto *llvmFunc = llvmModule->getFunction((exportOp.getName()).str());
      if (!llvmFunc)
        continue;

      llvm::Function *dmaPointer = nullptr;
      if (Operation *mlirFunc = symbolTable.lookup(exportOp.getName())) {
        if (FlatSymbolRefAttr dmaFunc =
                dialect->getDmaSpecializationAttrHelper().getAttr(mlirFunc)) {
          dmaPointer = llvmModule->getFunction(dmaFunc.getValue());
          if (!dmaPointer) {
            module.emitError()
                << "failed to find DMA code for " << exportOp.getName();
            return nullptr;
          }
          dmaPointer->setLinkage(
              llvm::GlobalValue::LinkageTypes::InternalLinkage);
          // Name suffix recognized by tooling for xDSL generated kernels.
          llvmFunc->setName(llvmFunc->getName() + "$iree_to_xdsl");
          if (llvm::DISubprogram *subProgram = llvmFunc->getSubprogram())
            subProgram->replaceLinkageName(llvm::MDString::get(
                llvmFunc->getContext(), llvmFunc->getName()));
        }
      }

      llvmFunc->setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);
      llvmFunc->setDSOLocal(true);

      // Tag the function parameters in case they got removed during conversion.
      // (%arg0: environment, %arg1: dispatch_state, %arg2: workgroup_state)
      for (unsigned i = 0; i <= 2; ++i) {
        llvmFunc->addParamAttr(i, llvm::Attribute::NonNull);
        llvmFunc->addParamAttr(i, llvm::Attribute::NoAlias);
        llvmFunc->addParamAttr(i, align16);
      }

      // Optionally entry points may specify that they require workgroup local
      // memory. We fetch that value here and plumb it through so the runtime
      // knows how much memory to reserve and pass in.
      int64_t localMemorySize = exportOp.getWorkgroupLocalMemory()
                                    .value_or(APInt(64, 0))
                                    .getSExtValue();

      // The runtime strictly validates these against the dispatch (see
      // command_buffer.c), so they must reflect the export's pipeline layout.
      Quidditch::LibraryBuilder::DispatchAttrs dispatchAttrs;
      dispatchAttrs.localMemorySize = localMemorySize;
      if (auto layoutAttr = exportOp.getLayout()) {
        dispatchAttrs.constantCount = layoutAttr.getConstants();
        dispatchAttrs.bindingCount = layoutAttr.getBindings().size();
      }

      Quidditch::LibraryBuilder::SourceLocation sourceLocation;
      SmallVector<Quidditch::LibraryBuilder::SourceLocation> stageLocations;
      libraryBuilder.addExport(
          exportOp.getName(), std::move(sourceLocation),
          std::move(stageLocations), /*tag=*/"", dispatchAttrs, llvmFunc,
          dmaPointer);
    }
    auto *queryLibraryFunc =
        libraryBuilder.build("quidditch_" + libraryName + "_library_query");

    // The query function must be exported for dynamic libraries.
    queryLibraryFunc->setDSOLocal(false);
    queryLibraryFunc->setVisibility(
        llvm::GlobalValue::VisibilityTypes::DefaultVisibility);
    queryLibraryFunc->setLinkage(
        llvm::GlobalValue::LinkageTypes::ExternalLinkage);
    queryLibraryFunc->setDLLStorageClass(
        llvm::GlobalValue::DLLStorageClassTypes::DLLExportStorageClass);

    // Specialize the module to our target machine.
    llvmModule->setDataLayout(machine.createDataLayout());
    llvmModule->setTargetTriple(machine.getTargetTriple());
    return llvmModule;
  }

  static void optimizeLLVMModule(llvm::Module &module,
                                 llvm::TargetMachine &machine) {

    llvm::LoopAnalysisManager loopAnalysisManager;
    llvm::FunctionAnalysisManager functionAnalysisManager;
    llvm::CGSCCAnalysisManager cGSCCAnalysisManager;
    llvm::ModuleAnalysisManager moduleAnalysisManager;

    llvm::PassBuilder passBuilder(&machine);
    llvm::AAManager aa = passBuilder.buildDefaultAAPipeline();
    functionAnalysisManager.registerPass([&] { return std::move(aa); });

    passBuilder.registerModuleAnalyses(moduleAnalysisManager);
    passBuilder.registerCGSCCAnalyses(cGSCCAnalysisManager);
    passBuilder.registerFunctionAnalyses(functionAnalysisManager);
    passBuilder.registerLoopAnalyses(loopAnalysisManager);
    passBuilder.crossRegisterProxies(
        loopAnalysisManager, functionAnalysisManager, cGSCCAnalysisManager,
        moduleAnalysisManager);

    llvm::ModulePassManager modulePassManager;
    modulePassManager =
        passBuilder.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
    modulePassManager.run(module, moduleAnalysisManager);
  }

  static FailureOr<IREE::HAL::Artifact>
  compileLLVMModule(llvm::Module &module, llvm::TargetMachine &machine) {
    auto objectFile = IREE::HAL::Artifact::createTemporary("iree-out", "o");

    llvm::raw_fd_ostream &os = objectFile.outputFile->os();
    llvm::legacy::PassManager passManager;
    passManager.add(
        new llvm::TargetLibraryInfoWrapperPass(machine.getTargetTriple()));
    if (machine.addPassesToEmitFile(passManager, os,
                                    /*DwoOut=*/nullptr,
                                    llvm::CodeGenFileType::ObjectFile))
      return failure();

    passManager.run(module);
    os.flush();
    os.close();
    return objectFile;
  }

  LogicalResult serializeExecutable(const SerializationOptions &options,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {
    ModuleOp module = variantOp.getInnerModule();

    FailureOr<SmallVector<IREE::HAL::Artifact>> objectFilesOrFailure =
        assembleXDSLOutput(module);
    if (failed(objectFilesOrFailure))
      return failure();

    SmallVector<IREE::HAL::Artifact> objectFiles =
        std::move(*objectFilesOrFailure);

    std::string errorMessage;
    auto llvmTarget = llvm::TargetRegistry::lookupTarget(
        "riscv32-unknown-unknown-elf", errorMessage);
    if (!llvmTarget)
      return variantOp.emitError(errorMessage);

    std::unique_ptr<llvm::TargetMachine> machine(
        llvmTarget->createTargetMachine(
            llvm::Triple("riscv32-unknown-unknown-elf"),
            "generic-rv32" /* cpu e.g k8 */, "+m,+f,+d,+zfh", {},
            llvm::Reloc::Model::PIC_, {}, llvm::CodeGenOptLevel::Aggressive,
            /*JIT=*/false));

    llvm::LLVMContext context;
    std::unique_ptr<llvm::Module> llvmModule =
        toLLVMModule(context, module, *machine, variantOp);
    if (!llvmModule)
      return failure();

    optimizeLLVMModule(*llvmModule, *machine);

    FailureOr<IREE::HAL::Artifact> objectFileOrFailure =
        compileLLVMModule(*llvmModule, *machine);
    if (failed(objectFileOrFailure))
      return failure();

    objectFiles.push_back(std::move(*objectFileOrFailure));

    SmallVector<StringRef> arguments = {"ld.lld", "-r"};
    llvm::append_range(
        arguments,
        llvm::map_range(objectFiles,
                        [](IREE::HAL::Artifact &artifact) -> StringRef {
                          return artifact.path;
                        }));

    std::string libraryName =
        variantOp->getParentOfType<IREE::HAL::ExecutableOp>().getName().str();
    auto linkedObject = IREE::HAL::Artifact::createTemporary(libraryName, "o");
    arguments.push_back("-o");
    arguments.push_back(linkedObject.path);
    int ret = llvm::sys::ExecuteAndWait(
        targetOptions.toolChainRoot + "/bin/ld.lld", arguments);
    if (ret != 0)
      return failure();

    if (!IREE::HAL::outputStaticLibrary(
            "quidditch_" + libraryName,
            "quidditch_" + libraryName + "_library_query",
            targetOptions.staticLibraryOutputPath, linkedObject.path))
      return variantOp.emitError() << "static library generation failed";

    // Embed the library name in the executable binary op. This informs the
    // loader which static library to load for the target binary.
    std::vector<uint8_t> libraryNameVector(libraryName.begin(),
                                           libraryName.end());
    executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        variantOp.getLoc(), variantOp.getSymName(), "snitch",
        libraryNameVector);

    return success();
  }

private:
  QuidditchTargetOptions targetOptions;
};

class QuidditchSession final
    : public PluginSession<QuidditchSession, QuidditchTargetOptions,
                           PluginActivationPolicy::DefaultActivated> {
public:
  static void registerGlobalDialects(DialectRegistry &registry) {
    // Required to allow the 'quidditch_snitch' dialect to also be used in
    // input IR without just being parsed as an 'OpaqueAttr'.
    registry.insert<quidditch::Snitch::QuidditchSnitchDialect>();
  }

private:
  void populateHALTargetDevices(IREE::HAL::TargetDeviceList &targets) override {
    targets.add("quidditch_device",
                []() { return std::make_shared<QuidditchTargetDevice>(); });
  }

  void
  populateHALTargetBackends(IREE::HAL::TargetBackendList &targets) override {
    targets.add("quidditch", [&]() {
      return std::make_shared<QuidditchTargetBackend>(options);
    });
  }
};

} // namespace

IREE_DEFINE_COMPILER_OPTION_FLAGS(::QuidditchTargetOptions);

extern "C" bool iree_register_compiler_plugin_hal_target_quidditch(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<QuidditchSession>("hal_target_quidditch");
  return true;
}
