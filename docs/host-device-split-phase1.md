# Phase 1: two-process mmap transport (no RTL)

Detailed design for Phase 1 of the host-device split (see
[`host-device-split.md`](host-device-split.md) for the overall roadmap). Phase 1
de-risks the serialize → replay → IREE-integration path entirely on the dev box,
before any Cheshire/Carfield RTL (Phase 2).

## Goal

The host records an IREE command buffer into a QCS command stream
([`runtime/host/transport/cluster_command_stream.h`](../runtime/host/transport/cluster_command_stream.h))
in shared memory; a separate "cluster" process replays it onto the dispatch
fan-out; the result matches the cluster-only inline-sync oracle. No RTL, no IRQ.

## Architecture

```
┌─ host process ─────────────┐         ┌─ cluster process ──────────┐
│ IREE VM + cluster HAL      │  mmap   │ qcs_reader replay loop      │
│  • QCS-emitting cmd buffer │ ⇄shared⇄│  • DISPATCH→dispatch fan-out│
│  • shared-arena allocator  │ "DRAM"  │  • COPY/FILL/UPDATE         │
│  • job descriptor+doorbell │  file   │  • snrt_* stubbed (pthreads)│
└────────────────────────────┘         └────────────────────────────┘
```

## Key decisions

1. **Host emits QCS directly** from a custom IREE HAL command buffer (sibling of
   `runtime/runtime/src/Quidditch/command_buffer/command_buffer.c`) whose
   dispatch/copy/fill/update callbacks append QCS records — rather than walking
   IREE's internal deferred-command-buffer structs. The HAL callback shapes map
   1:1 onto the QCS records (incl. indirect dispatch via
   `iree_hal_dispatch_uses_indirect_parameters`).
2. **`device_ptr` = offset into the shared region** (device-PA 0 == region base).
   The cluster HAL allocator carves device buffers from the mmap arena; each
   process adds its own mmap base. This makes the host-VA ↔ device-PA boundary
   explicit and testable — the crux risk the ABI was built around.
3. **Stub C kernels on the dev box.** Real kernels are rv32 Snitch binaries that
   can't run on x86; Phase 1 de-risks the *transport*, so the replayer calls
   host-C stub kernels registered in a fake executable table. Real kernels arrive
   at Phase 2.
4. **snrt_\* seam stubbed with pthreads**: `snrt_cluster_hw_barrier` →
   `pthread_barrier`, `snrt_cluster_core_idx` → thread-local, `snrt_dma_start_1d`
   → `memcpy`. 8 threads = 7 workers (`enter_worker_loop`) + 1 DM core.
5. **Doorbell/completion = polling** on the shared job descriptor (Phase 0/1
   allows polling; IRQ arrives at Phase 3). Ordering follows the ABI: cluster
   writes `status` → release → `completion`; host acquire-loads `completion`.

## Deliverables (under `runtime/host/`)

- `transport/shared_region.{h,c}` — mmap arena, device-PA ↔ VA translation, bump
  allocator, doorbell handshake. **(done — slice 1)**
- `transport/test_shared_region.c` — two-process proof: host records a
  COPY+FILL+UPDATE+DISPATCH job, cluster replays against shared device buffers,
  host verifies. **(done — slice 1)**
- `hal/cluster/` — minimal IREE HAL device + shared-arena allocator + the
  QCS-emitting command buffer. *(next — spec below)*
- `firmware/replay.{h,c}` + `snrt_stubs.c` — portable replay loop + pthread snrt
  stubs feeding the dispatch fan-out. *(next)*
- `app/phase1_gemm_test.c` — records a GEMM, runs both processes, diffs against a
  reference. *(next)*

### Slice 2 implementation spec (grounded in the IREE HAL)

The shared-arena allocator is the crux: it makes a buffer's device-PA (its
offset in the shared region) recoverable, which the command buffer needs to fill
QCS bindings. IREE has no standard "buffer -> device address" API, so we follow
the **CUDA driver pattern** (`iree/runtime/src/iree/hal/drivers/cuda/cuda_buffer.{h,c}`):
a buffer subclass that stores the address alongside `iree_hal_buffer_t base`.

- **Custom buffer** `iree_hal_cluster_buffer_t { iree_hal_buffer_t base; uint64_t
  device_pa; void* host_ptr; }`. `map_range` returns `host_ptr` (region base +
  pa); a typed accessor `..._buffer_device_pa()` recovers the PA via a
  vtable-checked cast.
- **Allocator** implements the 8 mandatory `iree_hal_allocator_vtable_t` methods
  (`destroy`, `host_allocator`, `trim`, `query_statistics`,
  `query_memory_heaps`, `query_buffer_compatibility`, `allocate_buffer`,
  `deallocate_buffer`, `import_buffer`, `export_buffer`); the 10 virtual-memory
  hooks stay NULL. `allocate_buffer` calls `qcs_shared_alloc` for the PA, then
  wraps `region_base + pa` via `iree_hal_heap_buffer_wrap` (or the subclass).
  Model on `iree/runtime/src/iree/hal/allocator_heap.c`.
- **Device** reuses Quidditch's `quidditch_device_create` shape, but injected with
  the cluster allocator (it already takes `iree_hal_allocator_t*`; see
  `runtime/runtime/src/Quidditch/device/device.c`).
- **Command buffer** sibling of `command_buffer.c`: `dispatch/copy/fill/update`
  callbacks resolve each `iree_hal_buffer_ref_t` to a PA via the subclass
  accessor and append the matching QCS record (indirect dispatch keys off
  `iree_hal_dispatch_uses_indirect_parameters`).
- **Build**: a new host (x86) target linking `iree::hal` + `iree::base` — NOT the
  rv32 Snitch build. The host IREE runtime is now stood up (the fast subset, no
  compiler/tests/drivers) and produces `libiree_hal_hal.a` + `libiree_base_base.a`:
  ```
  cmake -S iree -B <hostbuild> -GNinja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DIREE_BUILD_COMPILER=OFF -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_SAMPLES=OFF \
    -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_BUILD_BINDINGS_TFLITE=OFF
  ninja -C <hostbuild> iree_hal_hal iree_base_base   # (use a ninja >= 1.8, e.g. `uvx ninja`)
  ```

## Validation

Two-process GEMM: host records → cluster replays with a C GEMM stub → output
equals the on-cluster `gemm_square` self-check. A verify-agent + `/review` gate
each slice.

## Risks

- Binding VA↔PA translation correctness (decision #2 makes it explicit/testable).
- How much IREE HAL scaffolding a minimal "cluster device" needs to satisfy the
  VM — kept to the thinnest shell that records + submits.

## Status

- **Slice 1 (shared region + two-process transport): done, verified + reviewed.**
- Slice 2 (IREE HAL shell): in progress.
  - Host IREE runtime build stood up (`libiree_hal_hal.a` etc.). **done**
  - Shared-arena HAL allocator (`hal/cluster/cluster_allocator.{h,c}`): allocates
    device buffers from the arena, recovers device-PA via `ptr - region_base`
    (no custom buffer subclass); test links real `iree::hal`, passes under ASan.
    **done, verified + reviewed**
  - QCS-emitting command buffer (`hal/cluster/cluster_command_buffer.{h,c}`):
    serializes dispatch/copy/fill/update into a QCS stream, resolving each
    binding to a device-PA by mapping it (`ptr - region_base`); test records a
    job and re-parses the stream, links real `iree::hal`, passes under ASan.
    **done, verified + reviewed**
  - HAL device shell (`hal/cluster/cluster_device.{h,c}`): exposes the cluster
    allocator, creates command buffers (stream in the arena), and a synchronous
    `queue_execute` that publishes the QCS stream into the job descriptor, rings
    the doorbell, waits completion, then signals; includes a trivial host
    semaphore. Test drives device->stream->doorbell->completion with a stub
    cluster thread. **done, verified + reviewed.** Slice 2 complete.
- Slice 3 (cluster replay loop, `runtime/host/firmware/cluster_replay.{h,c}`):
  reads the QCS stream and replays COPY/FILL/UPDATE; for DISPATCH it invokes a
  registered stub kernel over the workgroup grid via the executable-library ABI
  (bindings resolved PA->ptr). Bounds-checks every PA and the dispatch geometry
  against the untrusted stream (rejects out-of-region, oversized grid, ABI-limit
  violations). Single-threaded grid loop; the real snrt_* 8-core fan-out is a
  Phase-2 refinement. **done, verified + reviewed.**
- Slice 4 (two-process GEMM, `runtime/host/app/phase1_gemm.c`): the capstone.
  Forks; the host records one dispatch via the IREE HAL device and
  `queue_execute`s it (doorbell submit); a separate cluster process (its own
  mmap -> real cross-address-space PA translation) replays the stream via
  slice 3's `cluster_replay` and runs a stub GEMM; the host verifies `C == A@B`
  (asymmetric inputs, explicit non-assert check). **done, verified + reviewed.**

**Phase 1 is complete: the host-device split runs end-to-end on the dev box**
(IREE-style record -> QCS stream in shared memory -> doorbell -> separate process
replays -> correct kernel output), with no RTL. Phase 2 (Carfield Verilator sim,
CVA6 host running the IREE VM, real rv32 kernels via iree+xdsl) is the next and
final stage toward the goal endpoint.
