quidditch_module(SRC softmax_linear.mlir ASSERT_XDSL)

add_executable(softmax_linear_sample main.c)
target_link_libraries(
    softmax_linear_sample
    PRIVATE
    samples_util
    softmax_linear
    snRuntime
    Quidditch::dispatch::dispatch
    m
)
