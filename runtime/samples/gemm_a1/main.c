// Track A: standalone measurement of the xDSL-generated GEMM micro-kernel.
//
// This deliberately excludes the IREE VM/HAL runtime. It mimics the opencompl
// paper harness: a single compute core stages the working set in L1 TCDM and
// runs the kernel bracketed by two snrt_mcycle() reads (the ROI), then verifies.
//
// The kernel (gemm_kernel.S) is produced by OUR port's exact pipeline:
//   xdsl-opt -p arith-add-fastmath,test-lower-linalg-to-snitch -t riscv-asm
// It is a bare-pointer SSR/FREP kernel: void gemm_kernel(double*A,double*B,double*C).
//
// C[i,j] = sum_k A[i,k]*B[k,j]; with A[i,k]=k+1, B[k,j]=1 every C element is
// sum_{k=0..15}(k+1) = 16*17/2 = 136 (exact in f64).
//
// Everything runs on compute core 0 (the DMA core has xssr/xfrep disabled in
// the default cfg, and the inputs are produced and consumed by the same core,
// so no cross-core barrier is needed).

#include <riscv_decls.h>  // snrt_mcycle
#include <team_decls.h>   // snrt_cluster_core_idx

#include <stdint.h>

#define DIM 16
#define N (DIM * DIM)
#define EXPECTED (DIM * (DIM + 1) / 2)  // 136

// L1/TCDM base (snitch_cluster_raw_addrmap.h: CLUSTER_TCDM_BASE_ADDR), 128 KiB.
// This crt does not init the snRuntime allocator (no SNRT_INIT_LIBS), so we
// place the 6 KiB working set at a fixed low TCDM offset; stacks/CLS live at the
// top of the TCDM, far above this.
#define TCDM_BASE 0x10000000u
#define L1_SCRATCH (TCDM_BASE + 0x2000u)

extern void gemm_kernel(double *A, double *B, double *C);  // a0=A, a1=B, a2=C

// snrt_fpu_fence() (snitch_cluster sw/runtime/src/ssr.h) is an impl-only inline;
// replicate it locally to drain the FPU before closing the ROI.
static inline void fpu_fence(void) {
  unsigned tmp;
  asm volatile("fmv.x.w %0, fa0\n mv %0, %0\n" : "+r"(tmp)::"memory");
}

int main() {
  if (snrt_cluster_core_idx() != 0) return 0;  // only compute core 0 works

  // Place the three matrices contiguously in L1 TCDM.
  double *A = (double *)L1_SCRATCH;
  double *B = A + N;
  double *C = B + N;

  for (int i = 0; i < DIM; i++)
    for (int k = 0; k < DIM; k++) A[i * DIM + k] = (double)(k + 1);
  for (int k = 0; k < DIM; k++)
    for (int j = 0; j < DIM; j++) B[k * DIM + j] = 1.0;
  for (int idx = 0; idx < N; idx++) C[idx] = 0.0;  // kernel accumulates onto C

  snrt_mcycle();  // ---- ROI start
  gemm_kernel(A, B, C);
  fpu_fence();
  snrt_mcycle();  // ---- ROI end

  uint32_t errors = 0;
  for (int idx = 0; idx < N; idx++)
    if ((int)C[idx] != EXPECTED) errors++;
  return errors ? 1 : 0;
}
