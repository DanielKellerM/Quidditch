add_executable(quidditch-opt quidditch-opt.cpp)
target_link_libraries(quidditch-opt
        PRIVATE
        MLIROptLib
        Quidditch::Conversion::ConvertSnitchToLLVM
        Quidditch::Conversion::ConvertToRISCV
        Quidditch::Dialect::DMA::Extensions::DMACoreSpecializationOpInterfaceImpl
        Quidditch::Dialect::SnitchDMA::IR::SnitchDMADialect
        Quidditch::Dialect::SnitchDMA::Transforms::Passes
        Quidditch::Dialect::Snitch::IR::QuidditchSnitchDialect
        Quidditch::Dialect::Snitch::Transforms::Passes
        Quidditch::Target::Passes
        iree::compiler::Dialect::LinalgExt::IR
        iree::compiler::Tools::init_passes_and_dialects
)

# Single-binary Stream-free device compiler: reuses IREE's front-end pipeline and HAL
# codegen/serialize passes, plus the Quidditch materialize-from-flow pass. Links the
# static compiler component libs (the shared libIREECompiler.so does not export the
# C++ pipeline symbols) and PluginManager (which force-links the plugin registrations).
add_executable(quidditch-compile quidditch-compile.cpp)
# Match LLVM/MLIR, which are built without RTTI (llvm::cl typeinfo is absent otherwise).
target_compile_options(quidditch-compile PRIVATE -fno-rtti)
target_link_libraries(quidditch-compile
        PRIVATE
        Quidditch::Target::Passes
        iree::compiler::Pipelines::Pipelines
        iree::compiler::Dialect::HAL::Transforms
        iree::compiler::Dialect::HAL::Target
        iree::compiler::ConstEval
        iree::compiler::PluginAPI::PluginManager
        iree::compiler::Tools::init_passes_and_dialects
        iree::compiler::Tools::init_llvmir_translations
        iree::compiler::Utils
        LLVMSupport
)
