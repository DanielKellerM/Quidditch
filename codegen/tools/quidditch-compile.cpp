// quidditch-compile: single-binary Stream-free device compiler for Snitch.
//
// Compiles a StableHLO (or linalg) input straight to a Snitch device object with no
// Stream / VM / EmitC. It runs the two proven stages fused in one process:
//   1. the standard front, to the `flow` phase (input -> linalg -> GlobalOpt ->
//      DispatchCreation -> Flow), via buildIREEPrecompileTransformPassPipeline;
//   2. the device-only tail: materialize hal.executables from flow.executables, then
//      configure + translate (codegen) + serialize. ConvertToHAL, LinkExecutables and
//      the HAL pruning passes are omitted -- a device-only compiler needs none.
// The object is written by the serialize pass via
// --iree-quidditch-static-library-output-path. Setup mirrors iree-opt
// (IREEOptToolEntryPoint): plugins are loaded and their target backends merged into
// the global registry so the tail's HAL passes resolve the "quidditch" backend.

#include "iree/compiler/ConstEval/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Pipelines/Options.h"
#include "iree/compiler/Pipelines/Pipelines.h"
#include "iree/compiler/PluginAPI/PluginManager.h"
#include "iree/compiler/Tools/init_dialects.h"
#include "iree/compiler/Tools/init_llvmir_translations.h"
#include "iree/compiler/Tools/init_passes.h"
#include "iree/compiler/Utils/OptionUtils.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

#include "Quidditch/Target/Passes.h"

using namespace mlir;
using namespace mlir::iree_compiler;

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllPasses();
  registerLLVMIRTranslations(registry);

  // Load plugins (the statically linked Quidditch + StableHLO input plugins) and let
  // them register their global dialects, passes and CLI options before flag parsing.
  PluginManager pluginManager;
  if (!pluginManager.loadAvailablePlugins()) {
    llvm::errs() << "error: failed to initialize compiler plugins\n";
    return 1;
  }
  pluginManager.globalInitialize();
  pluginManager.registerPasses();
  pluginManager.registerGlobalDialects(registry);
  pluginManager.initializeCLI();

  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input .mlir>"), llvm::cl::init("-"));
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  auto &pluginManagerOptions = PluginManagerOptions::FromFlags::get();
  // Force registration of every option group's flags before parsing (the front-end
  // pipeline reads them all): each FromFlags::get() registers its flags on first call.
  GlobalPipelineOptions::FromFlags::get();
  BindingOptions::FromFlags::get();
  InputDialectOptions::FromFlags::get();
  PreprocessingOptions::FromFlags::get();
  ParameterOptions::FromFlags::get();
  GlobalOptimizationOptions::FromFlags::get();
  DispatchCreationOptions::FromFlags::get();
  SchedulingOptions::FromFlags::get();
  IREE::HAL::TargetOptions::FromFlags::get();
  IREE::VM::TargetOptions::FromFlags::get();
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "quidditch-compile: StableHLO/linalg -> Snitch device object "
      "(no Stream/VM/EmitC).\n"
      "Emit the object with --iree-quidditch-static-library-output-path=<out.o>.\n");
  OptionsBinder::global().applyOptimizationDefaults();

  // Activate plugins and merge their target devices/backends into the global
  // registry -- the tail's HAL passes resolve the backend from there.
  auto localBinder = OptionsBinder::local();
  PluginManagerSession pluginSession(pluginManager, localBinder,
                                     pluginManagerOptions);
  if (failed(pluginSession.initializePlugins()))
    return 1;
  pluginSession.registerDialects(registry);
  auto &globalRegistry =
      const_cast<IREE::HAL::TargetRegistry &>(IREE::HAL::TargetRegistry::getGlobal());
  IREE::HAL::TargetDeviceList devices;
  pluginSession.populateHALTargetDevices(devices);
  globalRegistry.mergeFrom(devices);
  IREE::HAL::TargetBackendList backends;
  pluginSession.populateHALTargetBackends(backends);
  globalRegistry.mergeFrom(backends);

  MLIRContext context(registry);
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module)
    return 1;

  PassManager pm(&context, ModuleOp::getOperationName());
  applyPassManagerCLOptions(pm);

  IREEVMPipelineHooks hooks;
  hooks.buildConstEvalPassPipelineCallback = [&](OpPassManager &p) {
    p.addPass(ConstEval::createJitGlobalsPass({&globalRegistry}));
  };
  hooks.pipelineExtensions = &pluginSession;

  // Stage 1: front-end, to the flow dialect (the precompile builder stops at
  // GlobalOptimization; DispatchCreation + Flow live in the VM builder, which we run
  // only up to the Flow phase -- Stream/HAL/VM never execute).
  buildIREEVMTransformPassPipeline(
      globalRegistry, GlobalPipelineOptions::FromFlags::get(),
      BindingOptions::FromFlags::get(), InputDialectOptions::FromFlags::get(),
      PreprocessingOptions::FromFlags::get(), ParameterOptions::FromFlags::get(),
      GlobalOptimizationOptions::FromFlags::get(),
      DispatchCreationOptions::FromFlags::get(), SchedulingOptions::FromFlags::get(),
      IREE::HAL::TargetOptions::FromFlags::get(),
      IREE::VM::TargetOptions::FromFlags::get(), hooks, pm,
      IREEVMPipelinePhase::Start, IREEVMPipelinePhase::Flow);

  // Stage 2: Flow -> device object (serialize writes the .o via the plugin flag).
  // quidditch-link-executables (module-level) merges the per-dispatch executables into
  // one before serialize -- a no-op for a single executable, required for more than one.
  pm.addPass(quidditch::createMaterializeExecutableFromFlowPass());
  auto &translatePM = pm.nest<IREE::HAL::ExecutableOp>();
  translatePM.addPass(IREE::HAL::createConfigureExecutablesPass());
  translatePM.addPass(IREE::HAL::createTranslateAllExecutablesPass());
  pm.addPass(quidditch::createLinkExecutablesPass());
  pm.nest<IREE::HAL::ExecutableOp>().addPass(
      IREE::HAL::createSerializeAllExecutablesPass());

  if (failed(pm.run(*module)))
    return 1;
  return 0;
}
