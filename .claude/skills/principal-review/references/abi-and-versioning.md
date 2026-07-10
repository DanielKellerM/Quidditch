# ABI & versioning discipline

A struct/layout that spans two independently-built components (host↔firmware,
compiler↔firmware) is where the most expensive, hardest-to-debug defects live: a
silent layout mismatch corrupts memory with no compile error. A principal holds
these to a strict bar. In this repo the QCS job descriptor, the command-stream
records, and the executable export table are all cross-component contracts.

## The four rules

1. **Single source of truth.** The layout is defined once, in a header owned by
   the **emitter** (the side that writes the bytes), and `#include`d by the
   consumer. Not hand-copied into both with a "keep in sync" comment.
2. **Versioned at the boundary.** The producer writes a magic + version; the
   consumer **checks** it before trusting any field, and refuses (clean error)
   on mismatch. The `qcs_job_descriptor_t` magic/version check is the right
   instinct — every cross-component struct gets one.
3. **Layout-asserted at the definition site.** `_Static_assert` on `sizeof` and
   `offsetof` for every field the other side dereferences, so a layout drift is a
   *compile* error, not a runtime corruption.
4. **One definition of shared offsets.** A fixed device-PA / offset
   (`QCS_JOB_DESCRIPTOR_OFFSET`) is `#define`d once in the shared header, not
   re-defined in three `.c` files.

## The repo's live violation (fix target)

`runtime/host/firmware/gwaihir/qcs_kernel_abi.h` is a **hand-vendored slice** of
IREE's `iree/hal/local/executable_library.h`, *plus* a fork divergence: it adds
`compute_core_ptrs` + `dma_core_ptrs` where upstream has a single `ptrs` field.
The compiler emits the fork layout; the firmware dereferences it; they're kept in
agreement only by `_Static_assert`s on two copies that a reader must manually
believe stay in lockstep with the compiler's emission.

Not this (two copies, sync-by-comment):
```c
// qcs_kernel_abi.h — hand-copied from IREE, "keep in sync with the compiler"
typedef struct {
  const void* compute_core_ptrs;   // fork field
  const void* dma_core_ptrs;       // fork field
  // ... fields copied from executable_library.h by hand
} quidditch_executable_export_table_v0_t;
```

This instead — one emitter-owned header, versioned, asserted:
```c
// quidditch_executable_abi.h — owned by the compiler (the emitter), included by
// firmware. The compiler's export-table emission and this struct are the SAME
// definition, so they cannot drift.
#define QUIDDITCH_EXPORT_TABLE_VERSION 2u
typedef struct {
  uint32_t version;                // written by the compiler, checked at replay
  const void* const* compute_core_ptrs;
  const void* const* dma_core_ptrs;
  // ...
} quidditch_executable_export_table_v0_t;
_Static_assert(offsetof(quidditch_executable_export_table_v0_t, dma_core_ptrs)
               == 16, "export-table layout drift vs the compiler emitter");
```
and at the boundary:
```c
if (table->version != QUIDDITCH_EXPORT_TABLE_VERSION)
  return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                          "export table v%u, firmware expects v%u",
                          table->version, QUIDDITCH_EXPORT_TABLE_VERSION);
```

## Review checklist for any cross-component struct

- [ ] Defined once, in the emitter's header; consumer includes it (no twin copy).
- [ ] Magic + version field, written by producer, checked by consumer.
- [ ] `_Static_assert`s on `sizeof`/`offsetof` for every dereferenced field.
- [ ] Shared offsets/PAs `#define`d once in the shared header.
- [ ] Endianness + address-space (host-VA vs device-PA) stated in the header
      banner (as `cluster_command_stream.h` already does).
- [ ] A roundtrip test that serializes on one side and parses on the other
      (the `test_command_stream.c` pattern).

A vendored copy "kept in sync by hand" across a repo boundary is a **BLOCKER** —
it is the exact shape that will silently corrupt after the Nimbus split.
