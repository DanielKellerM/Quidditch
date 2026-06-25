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

# Standalone autotuner harness: the generated kernel dispatch without the IREE
# VM/HAL (~72s -> ~17s/config). Reuses gemm_mod; links only dispatch + snRuntime.
add_executable(gemm_harness harness.c)
target_link_libraries(
    gemm_harness
    PRIVATE
    gemm_mod
    snRuntime
    Quidditch::dispatch::dispatch
)
