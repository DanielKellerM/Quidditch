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

# 64x64 / 16-workgroup multi-cluster variant of the front-door (workgroup_tiles
# also comes from the config table).
quidditch_module(
    SRC gemm_square_4x4.stablehlo.mlir
    DST gemm_stablehlo_4x4_mod
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

# PyTorch MLP front-door (beyond gemm): 2-dispatch model for the on-device
# multi-dispatch harness. Compiled from the torch_xla-exported StableHLO.
quidditch_module(
    SRC torch_mlp.stablehlo.mlir
    DST mlp_mod
    ASSERT_XDSL
    FLAGS --iree-quidditch-config-table=${CMAKE_CURRENT_SOURCE_DIR}/mlp_config.json
)

add_executable(mlp_harness mlp_harness.c)
target_link_libraries(
    mlp_harness
    PRIVATE
    mlp_mod
    snRuntime
    Quidditch::dispatch::dispatch
)
