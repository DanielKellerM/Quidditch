// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <iree/hal/local/executable_loader.h>

#include "executable_library.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/local/executable_library.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct quidditch_executable_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  // Defines per-entry point how much workgroup local memory is required.
  // Contains entries with 0 to indicate no local memory is required or >0 in
  // units of IREE_HAL_WORKGROUP_LOCAL_MEMORY_PAGE_SIZE for the minimum amount
  // of memory required by the function.
  const iree_hal_executable_dispatch_attrs_v0_t* dispatch_attrs;

  // Execution environment.
  iree_hal_executable_environment_v0_t environment;

  // Name used for the file field in tracy and debuggers.
  iree_string_view_t identifier;

  union {
    const iree_hal_executable_library_header_t** header;
    const iree_hal_executable_library_v0_t* llvmcpu_v0;
    const quidditch_executable_library_v0_t* quidditch_v0;
  } library;

  // Transient workgroup-local-memory scratch, grown on demand and reused across
  // dispatches instead of a malloc/free per dispatch. Safe because the device
  // executes dispatches synchronously (concurrency == 1).
  iree_byte_span_t local_memory_scratch;

  bool is_llvm_cpu_executable;
} quidditch_executable_t;

iree_status_t quidditch_executable_create(
    const iree_hal_executable_params_t* executable_params,
    const iree_hal_executable_library_header_t** library_header,
    iree_hal_executable_import_provider_t import_provider,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable);

// Initializes the local executable base type.
void quidditch_executable_initialize(
    iree_allocator_t host_allocator,
    quidditch_executable_t* out_base_executable);

quidditch_executable_t* quidditch_executable_cast(
    iree_hal_executable_t* base_value);

iree_status_t quidditch_executable_issue_call(
    quidditch_executable_t* executable, iree_host_size_t ordinal,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state);

iree_status_t quidditch_executable_issue_dispatch_inline(
    quidditch_executable_t* executable, iree_host_size_t ordinal,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    uint32_t processor_id, iree_byte_span_t local_memory);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
