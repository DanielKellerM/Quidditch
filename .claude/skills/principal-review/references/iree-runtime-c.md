# IREE runtime C — worked examples

Applies to `runtime/host/**` (HAL driver, transport, firmware, Cheshire host) and
any C that links IREE. Every snippet below is copied from a real file in the tree
— when you review or write this code, open the cited source and match it.

Gold-standard neighbors to read first: `iree/runtime/src/iree/hal/drivers/null/`
(a complete minimal HAL driver), `iree/runtime/src/iree/base/api.h`,
`iree/runtime/src/iree/hal/buffer.h`.

---

## Error propagation — `iree_status_t` + `IREE_RETURN_IF_ERROR`

**Rule:** every fallible function returns `iree_status_t`; propagate with
`IREE_RETURN_IF_ERROR`, construct with `iree_make_status`, succeed with
`iree_ok_status()`. Never hand-roll status checks or return bare ints.

Idiomatic (`iree/runtime/src/iree/hal/drivers/null/buffer.c:96`):
```c
IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
    iree_hal_buffer_memory_type(base_buffer),
    IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
```
The macro unwraps, annotates, and avoids name collisions. It is the prescribed
form for *all* IREE code.

Not this:
```c
iree_status_t s = iree_hal_buffer_validate_memory_type(...);
if (!iree_status_is_ok(s)) return s;
```
Verbose, loses annotation, invites subtle propagation bugs. (A bare `int`/`bool`
return for a fallible op is a BLOCKER.)

---

## Allocation & ownership — `iree_allocator_t`, paired create/destroy

**Rule:** allocate via `iree_allocator_malloc`, store the allocator in the object,
free with the *same* allocator in `destroy`. Never bare `malloc`/`free`.

Idiomatic (`iree/runtime/src/iree/hal/drivers/null/device.c:84`):
```c
iree_host_size_t total_size = sizeof(*device) + identifier.size;
IREE_RETURN_AND_END_ZONE_IF_ERROR(
    z0, iree_allocator_malloc(host_allocator, total_size, (void**)&device));
// ... in destroy:
iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
iree_hal_allocator_release(device->device_allocator);
```
The stored allocator enables pooling, statistics, and custom arenas (exactly what
the host VM arena does). Bare `malloc`/`free` bypasses all of it and breaks the
ownership model — BLOCKER.

---

## Polymorphism — the HAL vtable pattern

**Rule:** a custom HAL object is a `static const iree_hal_<obj>_vtable_t` of
function pointers, wired with `iree_hal_resource_initialize(&vtable, &resource)`,
dispatched through the base type, recovered with `iree_hal_<obj>_cast`.

Idiomatic (`iree/runtime/src/iree/hal/drivers/null/device.c:62,87,673`):
```c
static const iree_hal_device_vtable_t iree_hal_null_device_vtable;
iree_hal_resource_initialize(&iree_hal_null_device_vtable, &device->resource);
static const iree_hal_device_vtable_t iree_hal_null_device_vtable = {
    .destroy = iree_hal_null_device_destroy,
    .id = iree_hal_null_device_id,
    .host_allocator = iree_hal_null_device_host_allocator,
    // ... the full upstream function set
};
```
`static const` = zero overhead + compile-time completeness checking. This is the
pattern `iree_hal_cluster_device` must mirror exactly.

Not this:
```c
if (device->type == NULL_DEVICE) { null_device_destroy(device); }
else if (device->type == CUDA_DEVICE) { cuda_device_destroy(device); }
```
Type-switch dispatch couples callers to every implementation and defeats the HAL
abstraction — BLOCKER for anything claiming to be a HAL driver.

---

## Reference counting — retain/release

**Rule:** `iree_hal_*_retain` when you store a reference, `iree_hal_*_release`
when done. `*_create()` transfers ownership to the caller.

Idiomatic (`iree/runtime/src/iree/hal/buffer.h:772`):
```c
// Retains the given |buffer| for the caller.
IREE_API_EXPORT void iree_hal_buffer_retain(iree_hal_buffer_t* buffer);
// Releases the given |buffer| from the caller.
IREE_API_EXPORT void iree_hal_buffer_release(iree_hal_buffer_t* buffer);
```
CoreFoundation semantics. Storing a reference without retaining is a
use-after-free waiting to happen — BLOCKER.

---

## Header hygiene — the `api.h` seam + banner

**Rule:** a module exposes one `api.h` that re-exports its submodule headers with
`IWYU pragma: export`; every header opens with the copyright/license banner and a
pointer to the API-conventions doc.

Idiomatic (`iree/runtime/src/iree/hal/api.h:1`):
```c
// Copyright 2019 The IREE Authors
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See iree/base/api.h for documentation on the API conventions used.
#ifndef IREE_HAL_API_H_
#define IREE_HAL_API_H_
#include "iree/hal/allocator.h"         // IWYU pragma: export
```
Scattered direct includes with no banner and no single export point are a TASTE
finding — match the neighbor's include structure and header banner.
