#include <Quidditch/dispatch/dispatch.h>

#include <gemm_mod.h>
#include <gemm_mod_module.h>
#include <team_decls.h>
#include <util/run_model.h>

// Standalone square GEMM (BLAS-3): C[64,64] = A[64,64] * B[64,64]^T.
// Uses the generic run_model() harness (like vec_multiply) — the working path.
#define DIM 64
#define NELEM (DIM * DIM)

int main() {
  static iree_alignas(64) double A[NELEM];
  static iree_alignas(64) double B[NELEM];
  static iree_alignas(64) double C[NELEM];
  if (!snrt_is_dm_core()) return quidditch_dispatch_enter_worker_loop();

  // Inputs left zero-init (static BSS) — values are irrelevant for the perf
  // measurement and this avoids a hot DM-core init loop that contaminated the
  // dispatch-overhead profile (§19).
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
  // No per-output printf (§19): keep the DM core off the float-formatting path.
  return 0;
}
