# Track A: standalone xDSL-kernel measurement harness (no IREE runtime).
enable_language(ASM)

add_executable(gemm_a1 main.c gemm_kernel.S)
# The hand-emitted Snitch asm needs the Snitch ISA enabled for the assembler.
set_source_files_properties(
    gemm_kernel.S PROPERTIES
    COMPILE_OPTIONS "-mcpu=snitch;-menable-experimental-extensions;-mabi=ilp32d"
)
target_link_libraries(gemm_a1 PRIVATE snRuntime)
