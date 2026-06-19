quidditch_module(SRC big_matvec.mlir DST big_matvec_mod ASSERT_XDSL)

add_executable(big_matvec main.c)
target_link_libraries(
    big_matvec
    PRIVATE
    samples_util
    big_matvec_mod
    snRuntime
    Quidditch::dispatch::dispatch
)

