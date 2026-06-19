quidditch_module(SRC gemm_square.mlir DST gemm_mod ASSERT_XDSL)

add_executable(gemm_square main.c)
target_link_libraries(
    gemm_square
    PRIVATE
    samples_util
    gemm_mod
    snRuntime
    Quidditch::dispatch::dispatch
)
