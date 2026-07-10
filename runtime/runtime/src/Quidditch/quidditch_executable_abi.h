// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// The one definition of the Quidditch executable-library export ABI: the v0.6
// IREE static-library layout plus the fork's compute_core/dma_core dispatch
// split. Included by BOTH the rv64 runtime (Quidditch/executable/executable_library.h,
// after iree/hal/local/executable_library.h) and the deps-free rv32 firmware
// (qcs_kernel_abi.h, after its vendored base types), so the compiler's emission
// and every consumer share one struct that cannot drift. This header declares
// only the fork structs; the base iree_hal_* types must already be declared by
// the includer (the real IREE header on the host, the vendored mirror on the
// firmware). Only count, compute_core_ptrs and dma_core_ptrs are dereferenced;
// the other slots are same-width pointers preserving the compiler's offsets.

#ifndef QUIDDITCH_EXECUTABLE_ABI_H_
#define QUIDDITCH_EXECUTABLE_ABI_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Mirror of IREE_HAL_EXECUTABLE_LIBRARY_VERSION_0_6; the emitted library_query
// gates on exactly this.
#define QUIDDITCH_EXECUTABLE_LIBRARY_VERSION_LATEST 6u

typedef struct quidditch_executable_export_table_v0_t {
  uint32_t count;
  const iree_hal_executable_dispatch_v0_t* compute_core_ptrs;
  const void* attrs;
  const void* params;
  const void* occupancy;
  const char* const* names;
  const char* const* tags;
  const char* const* parameter_names;
  const void* source_locations;
  const void* stage_locations;
  const iree_hal_executable_dispatch_v0_t* dma_core_ptrs;
} quidditch_executable_export_table_v0_t;

// Pin the two dispatch-pointer slots the replayer calls through: a fork field
// inserted before dma_core_ptrs fails the build instead of redirecting the DMA
// half. Pointer-width, so it holds for both the rv32 firmware and the rv64 host.
static_assert(offsetof(quidditch_executable_export_table_v0_t, compute_core_ptrs)
                  == sizeof(void*),
              "compute_core_ptrs must immediately follow count");
static_assert(offsetof(quidditch_executable_export_table_v0_t, dma_core_ptrs)
                  == offsetof(quidditch_executable_export_table_v0_t, stage_locations)
                         + sizeof(void*),
              "dma_core_ptrs must be the last struct-of-arrays slot");
static_assert(sizeof(quidditch_executable_export_table_v0_t) == 11 * sizeof(void*),
              "fork export table = count slot + 10 pointer slots");

typedef struct quidditch_executable_library_v0_t {
  const iree_hal_executable_library_header_t* header;
  iree_hal_executable_import_table_v0_t imports;
  quidditch_executable_export_table_v0_t exports;
  iree_hal_executable_constant_table_v0_t constants;
  iree_hal_executable_source_file_table_v0_t sources;
} quidditch_executable_library_v0_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // QUIDDITCH_EXECUTABLE_ABI_H_
