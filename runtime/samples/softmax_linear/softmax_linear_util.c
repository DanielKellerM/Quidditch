#include "softmax_linear_util.h"

#include <Quidditch/dispatch/dispatch.h>

#include <iree/base/alignment.h>

#include <team_decls.h>
#include <util/run_model.h>

#define IN_FEATURES 16
#define OUT_FEATURES 16

iree_status_t compiled_softmax_linear_create(iree_vm_instance_t *,
                                             iree_allocator_t,
                                             iree_vm_module_t **);

int run_softmax_linear_experiment(
    iree_hal_executable_library_query_fn_t implementation) {
  if (!snrt_is_dm_core()) return quidditch_dispatch_enter_worker_loop();

  iree_alignas(64) double input[1][IN_FEATURES];
  iree_alignas(64) double output[1][OUT_FEATURES];

  for (int i = 0; i < IN_FEATURES; i++) {
    input[0][i] = (double)(i + 1) * 0.1;
  }

  model_config_t config = {
      .libraries = (iree_hal_executable_library_query_fn_t[]){implementation},
      .num_libraries = 1,
      .module_constructor = compiled_softmax_linear_create,
      .main_function = iree_make_cstring_view("compiled_softmax_linear.main"),

      .element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_64,

      .num_inputs = 1,
      .input_data = (const void *[]){input},
      .input_sizes = (const iree_host_size_t[]){IN_FEATURES},
      .input_ranks = (const iree_host_size_t[]){3},
      .input_shapes =
          (const iree_hal_dim_t *[]){(iree_hal_dim_t[]){1, 1, IN_FEATURES}},

      .num_outputs = 1,
      .output_data = (void *[]){output},
      .output_sizes = (const iree_host_size_t[]){OUT_FEATURES},
  };

  IREE_CHECK_OK(run_model(&config));

  printf("softmax(Wx+b) =\n");
  for (int i = 0; i < OUT_FEATURES; i++) {
    printf("  [%2d] %f\n", i, output[0][i]);
  }
  return 0;
}
