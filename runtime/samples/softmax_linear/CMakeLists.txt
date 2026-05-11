
iree_turbine(SRC SoftmaxLinear.py DST ${CMAKE_CURRENT_BINARY_DIR}/softmax_linear.mlirbc DTYPE "f64")
quidditch_module(SRC ${CMAKE_CURRENT_BINARY_DIR}/softmax_linear.mlirbc DST softmax_linear)

add_library(softmax_linear_util softmax_linear_util.c)
target_link_libraries(softmax_linear_util
    PRIVATE
    samples_util
    snRuntimeInterface
    Quidditch::dispatch::dispatch
)
target_include_directories(softmax_linear_util INTERFACE ${CMAKE_CURRENT_LIST_DIR})

macro(create_experiment_variant)
  cmake_parse_arguments(_RULE "" "TARGET;IREE_MODULE;QUERY_FUNC" "" ${ARGN})

  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${_RULE_TARGET}.c "\
#include <${_RULE_IREE_MODULE}.h>

#include \"softmax_linear_util.h\"

int main() {
  return run_softmax_linear_experiment(${_RULE_QUERY_FUNC});
}
")
  add_executable(${_RULE_TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${_RULE_TARGET}.c)
  target_link_libraries(
      ${_RULE_TARGET}
      PRIVATE
      softmax_linear_util
      ${_RULE_IREE_MODULE}
      snRuntime
  )
endmacro()

create_experiment_variant(
    TARGET SoftmaxLinear
    IREE_MODULE softmax_linear
    QUERY_FUNC "quidditch_compiled_softmax_linear_linked_quidditch_library_query"
)
