#include <Quidditch/dispatch/dispatch.h>

#include <stdio.h>

#include <gemm_mod.h>
#include <gemm_mod_module.h>
#include <team_decls.h>
#include <util/run_model.h>

// Standalone square GEMM (BLAS-3): C[DIM,DIM] = A[DIM,DIM] * B[DIM,DIM]^T.
// Small DIM (shrunk from 64) for a fast end-to-end correctness check on the sim.
#define DIM 16
#define NELEM (DIM * DIM)

int main() {
  static iree_alignas(64) double A[NELEM];
  static iree_alignas(64) double B[NELEM];
  static iree_alignas(64) double C[NELEM];
  if (!snrt_is_dm_core()) return quidditch_dispatch_enter_worker_loop();

  // Known inputs: A[i,k] = k+1, B[j,k] = 1  =>  C[i,j] = sum_k (k+1) =
  // DIM*(DIM+1)/2 for every element (all sums are exact in f64).
  for (int i = 0; i < DIM; i++)
    for (int k = 0; k < DIM; k++) {
      A[i * DIM + k] = (double)(k + 1);
      B[i * DIM + k] = 1.0;
    }

  model_config_t config = {
      .libraries =
          (iree_hal_executable_library_query_fn_t[]){
              quidditch_gemm64_dispatch_0_library_query,
          },
      .num_libraries = 1,
      .module_constructor = gemm_square_create,
      .main_function = iree_make_cstring_view("gemm_square.gemm64"),

      .element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_64,

      .num_inputs = 2,
      .input_data = (const void*[]){A, B},
      .input_sizes = (const iree_host_size_t[]){NELEM, NELEM},
      .input_ranks = (const iree_host_size_t[]){2, 2},
      .input_shapes =
          (const iree_hal_dim_t*[]){(iree_hal_dim_t[]){DIM, DIM},
                                    (iree_hal_dim_t[]){DIM, DIM}},

      .num_outputs = 1,
      .output_data = (void*[]){C},
      .output_sizes = (const iree_host_size_t[]){NELEM},
  };

  IREE_CHECK_OK(run_model(&config));

  if (!snrt_is_dm_core()) return 0;

  // Verify the result: every C element must equal DIM*(DIM+1)/2.
  const int expected = DIM * (DIM + 1) / 2;
  int errors = 0;
  for (int idx = 0; idx < NELEM; idx++)
    if ((int)C[idx] != expected) errors++;
  printf("GEMM %dx%d: C[0]=%d C[last]=%d expected=%d errors=%d/%d -> %s\n", DIM,
         DIM, (int)C[0], (int)C[NELEM - 1], expected, errors, NELEM,
         errors ? "FAIL" : "SUCCESS");
  return errors ? 1 : 0;
}
