quidditch_module(SRC gemm_square.mlir DST gemm_mod ASSERT_XDSL)

# StableHLO front-door: the same gemm compiled from a bare StableHLO dot_general
# with no baked lowering_config -- the Snitch tiling is selected by
# ConfigureForSnitch from gemm_square_config.json (keyed by the dispatch symbol).
# Building this exercises the StableHLO frontend + config-table de-baking;
# check_stablehlo_frontdoor.sh asserts its device object is byte-identical to the
# linalg gemm_mod above.
quidditch_module(
    SRC gemm_square.stablehlo.mlir
    DST gemm_stablehlo_mod
    ASSERT_XDSL
    FLAGS --iree-quidditch-config-table=${CMAKE_CURRENT_SOURCE_DIR}/gemm_square_config.json
)

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
