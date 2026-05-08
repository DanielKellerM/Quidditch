#include <math.h>

#include <Quidditch/dispatch/dispatch.h>

#include <softmax_linear.h>
#include <softmax_linear_module.h>

#include <team_decls.h>
#include <util/run_model.h>


int main() {
    // softmax(Wx + b) with W:10x16, x:16, b:10, out:10
    iree_alignas(64) double W[10 * 16];
    iree_alignas(64) double x[16];
    iree_alignas(64) double b[10];
    iree_alignas(64) double out[10];  
    
    if (!snrt_is_dm_core()) return quidditch_dispatch_enter_worker_loop();
    
    // fill with test values, so that Wx + b roughly in [-1,1]
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 16; j++) {
            W[i * 16 + j] = 0.01 * ((i + j) % 5 - 2);   // values in {-0.02, -0.01, 0, 0.01, 0.02}
        }
    }
    for (int j = 0; j < 16; j++) x[j] = 0.5 - 0.05 * j;   // linear ramp in [-0.25, 0.5]
    for (int i = 0; i < 10; i++) b[i] = 0.1 * i - 0.5;     // linear ramp in [-0.5, 0.4]

    // begin execution
    model_config_t config = {
        .libraries =
          (iree_hal_executable_library_query_fn_t[]){
              // TODO: verify this symbol after first build by inspecting
              //   build/runtime/samples/softmax_linear/softmax_linear/softmax_linear.h
              quidditch_forward_dispatch_0_library_query,
          },
        .num_libraries = 1,
        .module_constructor = softmax_linear_create,
        .main_function = iree_make_cstring_view("softmax_linear.forward"),

        .element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_64,
        
        .num_inputs = 3,
        .input_data = (const void*[]){W, x, b},
        .input_sizes = (const iree_host_size_t[]){IREE_ARRAYSIZE(W),
                                                    IREE_ARRAYSIZE(x),
                                                    IREE_ARRAYSIZE(b)},
        .input_ranks = (const iree_host_size_t[]){2, 1, 1},
        .input_shapes =
            (const iree_hal_dim_t*[]){(iree_hal_dim_t[]){10, 16},
                                        (iree_hal_dim_t[]){16},
                                        (iree_hal_dim_t[]){10}},

        .num_outputs = 1,
        .output_data = (void*[]){out},
        .output_sizes = (const iree_host_size_t[]){IREE_ARRAYSIZE(out)},
    };

    // end execution
    IREE_CHECK_OK(run_model(&config));

    if (!snrt_is_dm_core()) return 0;

    // correctness check
    double ref[10];
    // compute Wx + b
    for (int i = 0; i < 10; i++) {
        double s = b[i];
        for (int j = 0; j < 16; j++) s += W[i*16 + j] * x[j];
        ref[i] = s;
    }
    // find max
    double m = ref[0]; 
    for (int i = 1; i < 10; i++) if (ref[i] > m) m = ref[i];
    // sub and exp, sum
    double Z = 0; for (int i = 0; i < 10; i++) { ref[i] = exp(ref[i] - m); Z += ref[i]; }
    
    // normalize
    for (int i = 0; i < 10; i++) ref[i] /= Z;

    // compare ref with output 
    for (int i = 0; i < 10; i++) {
        printf("out[%d] = %f  (ref %f)\n", i, out[i], ref[i]);
        if (fabs(out[i] - ref[i]) > 1e-3) return 1;   // tolerance depends on variant
    }
    return 0;
}