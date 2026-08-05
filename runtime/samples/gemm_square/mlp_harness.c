// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// On-device MULTI-DISPATCH harness for the PyTorch MLP front-door. Runs the two
// generated dispatches in sequence on one Snitch cluster (no IREE VM/HAL), threading
// the intermediate h1 = dispatch_0's output into dispatch_1's input. The compute cores
// run the REAL f64 kernels (FPU + SSR/FREP streaming + double buffering, fanned out
// across the 8 compute cores). Inputs are the actual torch weights, seeded as f64 bit
// patterns via integer stores (the DM core has no fp ARITHMETIC); the fp output is
// dumped and tolerance-checked host-side against the torch golden (compare_mlp.py).
//   dispatch_0: h1 = relu(x @ W1^T + b1)   dispatch_1: y = h1 @ W2^T + b2

#include <Quidditch/dispatch/dispatch.h>
#include <Quidditch/executable/executable_library.h>
#include <iree/base/api.h>
#include <iree/hal/local/executable_library.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <encoding.h>    // read_csr(mcycle)
#include <team_decls.h>  // snrt_is_dm_core, snrt_cluster_compute_core_num

#include <mlp_mod.h>      // quidditch_mlp_linked_quidditch_library_query
#include "mlp_ref_bits.h" // x_bits/w1_bits/b1_bits/w2_bits/b2_bits (torch f64 weights)

#define N 16
typedef double elem_t;

// L1 scratch for a dispatch's tiles (double-buffered); guarded vs attrs.local_memory_pages below.
#define LOCAL_MEMORY_BYTES (32 * 1024)
static iree_alignas(64) uint8_t g_local_memory[LOCAL_MEMORY_BYTES];
static iree_hal_executable_environment_v0_t g_env;

static int run_dispatch(const quidditch_executable_library_v0_t* lib, int ordinal,
                        void** bptrs, size_t* blens, int nb) {
  iree_hal_executable_dispatch_attrs_v0_t attrs =
      ((const iree_hal_executable_dispatch_attrs_v0_t*)lib->exports.attrs)[ordinal];
  size_t local_memory_size = (size_t)attrs.local_memory_pages * 4096;
  if (local_memory_size > sizeof(g_local_memory) || attrs.binding_count != nb) {
    printf("HARNESS ERROR: ord=%d pages=%u binding_count=%u\n", ordinal,
           (unsigned)attrs.local_memory_pages, (unsigned)attrs.binding_count);
    return 2;
  }
  iree_alignas(64) iree_hal_executable_dispatch_state_v0_t st;
  memset(&st, 0, sizeof(st));
  st.workgroup_size_x = attrs.workgroup_size_x ? attrs.workgroup_size_x : 1;
  st.workgroup_size_y = attrs.workgroup_size_y ? attrs.workgroup_size_y : 1;
  st.workgroup_size_z = attrs.workgroup_size_z ? attrs.workgroup_size_z : 1;
  st.workgroup_count_x = 1;
  st.workgroup_count_y = 1;
  st.workgroup_count_z = 1;
  st.max_concurrency = snrt_cluster_compute_core_num();  // fan out across all 8 compute cores
  st.constant_count = attrs.constant_count;
  st.constants = NULL;
  st.binding_count = attrs.binding_count;
  st.binding_ptrs = bptrs;
  st.binding_lengths = blens;

  iree_hal_executable_dispatch_v0_t compute_fn = lib->exports.compute_core_ptrs[ordinal];
  iree_hal_executable_dispatch_v0_t dma_fn = lib->exports.dma_core_ptrs[ordinal];
  quidditch_dispatch_set_kernel(compute_fn, &g_env, &st);

  iree_hal_executable_workgroup_state_v0_t ws;
  memset(&ws, 0, sizeof(ws));
  ws.local_memory = g_local_memory;
  ws.local_memory_size = (uint32_t)local_memory_size;

  if (dma_fn) {
    quidditch_dispatch_queue_subgroups(&ws);
    dma_fn(&g_env, &st, &ws);
    quidditch_dispatch_wait_for_workgroup();
  } else {
    quidditch_dispatch_queue_workgroup(&ws);
    quidditch_dispatch_execute_workgroups();
  }
  return 0;
}

int main(void) {
  static iree_alignas(64) elem_t x[N * N], W1[N * N], b1[N], W2[N * N], b2[N];
  static iree_alignas(64) elem_t h1[N * N], y[N * N];

  if (!snrt_is_dm_core()) return quidditch_dispatch_enter_worker_loop();

  // Seed the real torch f64 weights via integer stores (no DM-core fp arithmetic).
  memcpy(x, x_bits, sizeof(x));
  memcpy(W1, w1_bits, sizeof(W1));
  memcpy(b1, b1_bits, sizeof(b1));
  memcpy(W2, w2_bits, sizeof(W2));
  memcpy(b2, b2_bits, sizeof(b2));
  memset(h1, 0, sizeof(h1));
  memset(y, 0, sizeof(y));

  memset(&g_env, 0, sizeof(g_env));
  const iree_hal_executable_library_header_t** header =
      quidditch_mlp_linked_quidditch_library_query(
          IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST, &g_env);
  const quidditch_executable_library_v0_t* lib =
      (const quidditch_executable_library_v0_t*)header;

  uint32_t t0 = read_csr(mcycle);
  void* b0[4] = {x, W1, b1, h1};
  size_t l0[4] = {sizeof(x), sizeof(W1), sizeof(b1), sizeof(h1)};
  int rc = run_dispatch(lib, 0, b0, l0, 4);
  void* bd1[4] = {h1, W2, b2, y};
  size_t l1[4] = {sizeof(h1), sizeof(W2), sizeof(b2), sizeof(y)};
  if (!rc) rc = run_dispatch(lib, 1, bd1, l1, 4);
  uint32_t t1 = read_csr(mcycle);
  quidditch_dispatch_quit();
  if (rc) return rc;

  // Dump y as f64 bit patterns (integer hex, no fp), strided so 4 runs (offsets 0..3)
  // cover all 256 one short line at a time (a giant printf corrupts on some tb UARTs).
#define YSTRIDE 4
#ifndef YOFFSET
#define YOFFSET 0
#endif
  // Non-hex sacrificial lines FIRST: some tb UARTs eat the first ~line at the burst
  // start, so they absorb it and the trusted YDUMP_BEGIN header + values print clean.
  printf("________________\n");
  printf("________________\n");
  printf("[MLP] dispatch_cycles=%u YDUMP_BEGIN stride=%d offset=%d\n", t1 - t0, YSTRIDE,
         YOFFSET);
  const uint64_t* u = (const uint64_t*)y;
  char line[18];
  for (int i = YOFFSET; i < N * N; i += YSTRIDE) {
    for (int nib = 15; nib >= 0; nib--) {
      int v = (int)((u[i] >> (nib * 4)) & 0xf);
      line[15 - nib] = (char)(v < 10 ? '0' + v : 'a' + v - 10);
    }
    line[16] = '\n';
    line[17] = '\0';
    printf("%s", line);
  }
  printf("[MLP] YDUMP_END\n");
  return 0;
}
