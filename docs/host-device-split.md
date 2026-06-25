<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 ETH Zurich and University of Bologna. -->

# Host-device split: running the IREE VM on a CVA6 host, the Snitch cluster as a pure accelerator

**Status:** design + foundation. The command-stream ABI (Phase 0a) is built and
tested (`runtime/host/transport/`, commit `3d064f9` on branch
`cheshire-host-device-split`). The payoff is *measured* (see §2). The rest is
scoped here and not yet built. Tracked as task #19.

**Audience:** anyone picking this up later. This doc is meant to be followable
end-to-end without re-deriving the analysis.

---

## 1. Why

Today the entire IREE VM/HAL runs **on the Snitch cluster's DM (data-mover)
core**. Every core boots the same ELF and forks by role
(`runtime/samples/gemm_square/main.c`): the 8 compute cores branch immediately
into a hardware-barrier worker loop
(`quidditch_dispatch_enter_worker_loop`), and only the DM core runs
`run_model()` — `iree_vm_instance_create`, the HAL module, the device, the VM
context, input marshaling, `iree_vm_invoke`.

This is IREE's **`inline/sync` HAL device** collapsed onto one core:
`quidditch_device`'s `queue_execute` replays the recorded command buffer
*inline on the calling (DM) core* rather than submitting it to a separate
device; barriers/events are no-ops and semaphores are host-only
(`runtime/runtime/src/Quidditch/device/device.c`,
`.../command_buffer/command_buffer.c`). It is the correct, minimal choice for a
**standalone, hostless** cluster, and it is how upstream Quidditch has always
worked (all samples + git history predate this port).

But it puts a general-purpose ML-runtime VM on a core whose architectural job is
data movement. Measured cost (see [project_snitch_codegen_measurement] in the
working memory and §2 below): a single 16×16 f64 GEMM inference spends
**~39,200 cycles on the DM core, ~95% of it generic VM/HAL orchestration**
(string-based function bookkeeping, malloc/free, ref-counting, command-buffer
recording, fence handling), while the 8 compute cores idle in the barrier and
the actual xDSL kernel is only **~2,100 cycles**. The dispatch is fully
synchronous, so that overhead is squarely on the wall-clock critical path.

The fix is **software placement, not micro-optimization** (an in-place
per-dispatch-malloc removal measured at 18 cycles / 0.007% — second-order, as
predicted). The architecturally-correct deployment runs the IREE VM on a host
CPU and treats the cluster as a pure IREE *device*. The natural host is
**Cheshire** (a minimal Linux-capable CVA6 SoC) or **Carfield** (Cheshire +
the 8-core double-precision Snitch PMCA + mailbox + L2 SPM + iDMA — i.e. this
exact host-device topology already exists in RTL).

> The VM/HAL overhead is **real cycles on the (simulated or real) DM core**, not
> a simulation artifact: RTL-sim cycle counts equal real-silicon cycles. So the
> split is a deployment change, not a sim trick. A dedicated hardware dispatch
> unit (PULP/Occamy build these) only helps the *residual* ~2× after the split;
> you do not build silicon to move generic VM software off the wrong core.

---

## 2. The payoff, measured (not projected)

The "device half" — the work the cluster firmware replays in the split — was
isolated by bracketing `quidditch_command_buffer_dispatch` (the function the
firmware runs) with `mcycle` readings in the *current* runtime and running the
GEMM (`C==136` preserved; probe reverted afterwards):

| Configuration | Cluster-side cycles / inference | vs. raw xDSL compute |
|---|---|---|
| Full IREE VM on cluster (today) | ~39,200 | ~19× |
| **Device-half only (= post-split cluster cost)** | **4,194** (955 binding setup + 3,239 fan-out+compute) | **~2.0×** |
| Raw xDSL kernel compute (floor) | ~2,100 | 1× |

**Moving the VM off the cluster takes cluster-side cost from ~39.2k → ~4.2k — a
9.3× reduction, landing at ~2× the raw xDSL compute.** In the real split it is
even leaner: binding addresses arrive pre-resolved in the command stream, so
the 955-cycle per-binding map step shrinks. The residual ~2× is the irreducible
8-core wake/barrier + binding setup, matching the predicted floor. The ~35k of
VM/HAL orchestration moves to the host, where it amortizes across inferences and
can pipeline behind compute.

This is the empirical justification for building the split: it is the only lever
that closes the 19× gap to "close to maximum xDSL performance."

---

## 3. What moves where

The split is already latent in the code: the dispatch fan-out + kernels are the
reusable "device half"; everything VM-shaped is host-movable.

| Component | Fate |
|---|---|
| `dispatch/dispatch.c`, `dispatch.h` (barrier wake + fan-out + worker loop) | **Reused verbatim** on the cluster — it *is* the device contract |
| Snitch kernel libraries, snRuntime, the iDMA copy path | **Reused verbatim** on the cluster |
| `executable/executable.c` `quidditch_executable_issue_dispatch_inline` | **Reused** on the cluster (the replayer's inner loop) |
| `command_buffer.c` dispatch *recording* logic (constants/bindings/wg-count extraction) | **Split**: host *serializes* to the command stream; cluster *replays* from it |
| `run_model.c` VM setup | **Moves to the host** (recompiled rv64/Linux; allocator + device swapped) |
| `device.c` (whole `iree_hal_device_t`) | **Replaced** by a new host-side `iree_hal_cluster_*` device (async submit, not inline collapse) |
| `device/semaphore.c` (host-only notify) | **Replaced** by a semaphore backed by the completion flag / IRQ |
| `device/registration/registration.c` (currently `UNIMPLEMENTED`) | **Implemented** (driver registry for the cluster driver) |
| compute-core `enter_worker_loop` | **Zero change** — contract is already "park in barrier, wake, run `configuredKernel`, signal" |

The single most valuable reuse: the compute-core side needs **no change at all**.

---

## 4. The transport (host ↔ cluster)

Grounded in the PULP/Occamy offload model ("Taming Offload Overheads", arXiv
2505.05911) and Carfield's mailbox + L2 SPM. A job descriptor lives at a fixed
shared address; the command stream and device buffers live in the AXI-shared
region (L2 SPM / DRAM the cluster iDMA can reach).

This contract is **already implemented and tested** in
`runtime/host/transport/cluster_command_stream.{h,c}` (commit `3d064f9`):

- `qcs_job_descriptor_t`: `magic`/`version`/`feature_flags`, a `doorbell`
  (host→cluster submit) and `completion`+`status` (cluster→host), an
  `executable_table_id` (the cluster rejects a job compiled against a different
  kernel set), a `record_count`, and `cmd_stream_ptr`/`len`.
- Records: `QCS_CMD_DISPATCH` (executable id, export ordinal, workgroup
  count/size, dynamic local memory, then trailing constants[] + bindings[]) and
  `QCS_CMD_COPY` (device-PA → device-PA iDMA transfer).
- Bindings are **device-physical** addresses (`buffer_base + offset` folded by
  the host) — this makes the host-VA → device-PA translation boundary explicit.
- Dependency-free (only `<stdint.h>`) so the same header compiles for the rv64
  host and the rv32 firmware. Little-endian (both sides are RISC-V LE).

`cluster_command_stream.c` provides a host **writer** (serialize) and a cluster
**reader** (parse-in-place, no allocation). `test_command_stream.c` is a
standalone roundtrip proof — host serializes a 2-dispatch + 1-copy job, the
"cluster" parses and replays it, every field round-trips, runs on the dev box
with **no IREE or RTL** (`make -C runtime/host/transport check`).

The reader was hardened against untrusted host data (an audit found and we
fixed two out-of-bounds bugs): it re-validates each record's declared counts
against its size, rejects unknown/truncated records, and the writer rejects
integer-overflowing records — all proven by adversarial probes, clean under
`-Werror` + ASan + UBSan.

**Publish/consume order** (stated in the ABI header): the cluster writes
`status` → fence → `completion`; the host polls `completion` (Phase 0/1) or
waits on the completion IRQ (Phase 2). On Carfield the doorbell maps to the
software mailbox / CLINT `msip` and completion to the PLIC `intr_ext` path back
to CVA6.

### Record types still to add (an audit flagged these for real models)

`DISPATCH` + `COPY` cover the GEMM. Before this carries real models, add:

- `QCS_CMD_FILL` — `fill_buffer` (memset a binding; zero-init / bias).
- `QCS_CMD_UPDATE` — host pushes an inline data blob into a device buffer
  (weight/parameter upload); variable-length inline payload DMA'd to `dst_ptr`.
- **Indirect dispatch** — workgroup count read from a device buffer at dispatch
  time (dynamic shapes). The ABI reserves a `flags` field for this; add an
  optional `workgroup_count_ptr`.

Correctly **omitted** (the device is synchronous): barriers, events,
`advise_buffer`. Collectives are `UNIMPLEMENTED` upstream too.

---

## 5. The new host-side HAL device

Implement `iree_hal_cluster_device` as a real `iree_hal_device_t` (model on
upstream `local-sync` + the CUDA/HIP *remote* driver pattern for the async
queue/semaphore shape):

- **Driver + registration** — implement the registry (`registration.c`) so the
  device is discoverable, or keep direct creation as today. Low effort.
- **Allocator (the crux)** — today buffers are host heap and "device memory *is*
  host memory" (`run_model.c`). The new allocator carves `DEVICE_LOCAL` buffers
  out of the AXI-shared window (L2 SPM / DRAM) so the returned pointer is a
  physical address the cluster iDMA can read. On Linux: `mmap` the SPM/DRAM
  aperture (`/dev/mem` or a UIO/dma-buf driver) + a pool allocator over it.
  `import_buffer` translates host-VA → device-PA for the binding pointers
  written into the command stream.
- **Command buffer** — keep IREE's `deferred_command_buffer` for recording. At
  `queue_execute`, instead of replaying inline, **serialize the deferred buffer
  into the command stream** (the extraction logic in `command_buffer.c` already
  exists; it writes to a buffer instead of to a dispatch state).
- **Semaphore** — replace host-only notify with one whose value is published by
  the cluster into the shared completion word and whose wait is satisfied by the
  completion IRQ (or a poll fallback).
- **`queue_execute`** becomes: wait host-side wait-semaphores → write the command
  stream + binding PAs into shared memory → ring the doorbell → block on the
  completion IRQ → signal the signal-semaphores. A near-mechanical rewrite of the
  current inline-replay `queue_execute`.

### Cluster firmware (thin job loop), reusing existing code

1. WFI on the doorbell IRQ.
2. (Optionally) iDMA-pull the descriptor + command stream into TCDM.
3. For each `DISPATCH` record: set the kernel + dispatch state (the body of
   `quidditch_executable_issue_dispatch_inline`), then
   `quidditch_dispatch_start_executing_workgroup()` (barrier-wake 8 cores) /
   `wait_for_workgroup()` — **unchanged fan-out**. Preserve the two-path
   distinction: LLVM-distributed workgroups vs. the Snitch-native
   `dma_core_ptrs[ordinal]` function on the DM core.
4. iDMA-writeback outputs.
5. Write `status` → fence → `completion`, raise the completion IRQ, clear the
   doorbell, WFI.

---

## 6. Repo scaffolding

```
Quidditch/
  codegen/                 # UNCHANGED: xDSL/MLIR → kernels (host-side compile)
  runtime/
    host/                  # host-side IREE runtime + cluster HAL driver (rv64/Linux)
      README.md            # (exists) overview + phasing
      transport/           # (exists, 3d064f9) the command-stream ABI + roundtrip test
      hal/cluster/         # NEW: iree_hal_cluster_{driver,device,command_buffer,
                           #      semaphore,allocator}.c (derived from device.c etc.)
      app/                 # NEW: run_model.c (rv64 port) + per-model host mains
    firmware/              # NEW: thin cluster job-loop ELF (rv32, Snitch LLVM)
      job_loop.c           #   doorbell WFI + descriptor parse + replay
      replay.c             #   stripped issue_dispatch_inline (no iree_hal_* objects)
                           #   <links> dispatch.c, snRuntime, kernel libs
    runtime/src/Quidditch/ # KEEP: standalone (inline-sync) mode still builds, behind
                           #       a CMake option, as the correctness oracle
```

- **Two toolchains.** Firmware: existing Snitch LLVM (`riscv32imafdzfh`,
  `-nostartfiles`, snRuntime). Host: rv64gc Linux GCC/Clang + glibc + full
  upstream IREE runtime. Top-level superbuild with
  `IREE_EXTERNAL_HAL_DRIVERS=cluster`.
- **Codegen unchanged**, but split its two outputs: the VM module
  (`--output-format=vm-bytecode`, now usable on a Linux host) → host; the Snitch
  kernel static libs → firmware. `quidditch_module.cmake` gets a flag to emit
  each side into the right tree instead of linking both into one ELF.
- **Keep the standalone mode buildable** behind a CMake option
  (`QUIDDITCH_HOST_DEVICE=OFF` default) so "everything on the DM core" stays as
  the bring-up + correctness oracle.

---

## 7. Phased bring-up

Each phase is independently testable.

- **Phase 0a — the command-stream ABI** *(DONE, `3d064f9`)*. Roundtrip proof on
  the dev box, no IREE/RTL.
- **Phase 0b — missing record types**: add `QCS_CMD_FILL`, `QCS_CMD_UPDATE`,
  indirect dispatch (§4) + extend the roundtrip test.
- **Phase 1 — two-process mmap transport (no RTL)**: host runtime + cluster
  firmware as two dev-box processes sharing an `mmap`'d file as "shared DRAM",
  doorbell = a flag + condvar. Wire the writer to an IREE deferred command
  buffer and the reader to the existing `dispatch.c` fan-out. De-risks the
  serialize→replay→IREE integration before any RTL. Validate against the current
  inline-sync output.
- **Phase 2 — one Carfield Verilator sim, shared SPM, polling**: host VM (CVA6
  ELF) + cluster firmware (Snitch ELF) co-resident, communicating through real
  shared SPM/DRAM, doorbell + completion **by polling memory** (no IRQ yet).
  Validates the crux: host-VA → device-PA binding translation + iDMA
  reachability. **This is where the ~4.2k cluster-side number gets confirmed in
  a real split.**
- **Phase 3 — real mailbox/IRQ + CVA6 under Linux**: swap polling for the
  Carfield mailbox / CLINT `msip` doorbell and the PLIC completion IRQ; semaphore
  wait becomes a UIO eventfd / WFI. The "standard IREE host/device split"
  milestone.
- **Phase 4 — async pipelining**: non-blocking `queue_execute`, multiple in-flight
  command buffers, IREE semaphores for true overlap (host prepares job N+1 while
  the cluster runs job N). Optionally multi-cluster.

---

## 8. Risks / unknowns (honest)

- **Crux — host-VA ↔ device-PA translation** for binding pointers. Today "device
  memory *is* host memory" papers over all addressing; that assumption is exactly
  what breaks in a real split. Get the binding translation wrong → the iDMA reads
  garbage. The ABI carries device-PA deliberately to make this explicit.
- **Completion-IRQ → IREE-semaphore wiring under Linux** (UIO/dma-buf driver) +
  cache coherence of the shared region (Cheshire uses self-invalidation
  coherence, so shared SPM/DRAM buffers need explicit fence/flush discipline on
  both ends).
- **The two-path dispatch** (LLVM workgroups vs. Snitch `dma_core_ptrs`) must be
  preserved in the firmware replayer.
- **VM module form** — moving from `vm-c` (linked into the ELF today) to
  `vm-bytecode` on the host is cleaner but is a codegen-flag change to validate.

---

## 9. Future: umbrella repo

The split spans several repos (Cheshire/Carfield SoC, the host IREE runtime, the
cluster firmware, Quidditch). The natural long-term home is a **dedicated
umbrella repo** (bender/git submodules) that pulls them together; not needed
until Phase 2/3. The command-stream ABI (`runtime/host/transport/`) was made
dependency-free precisely so it can move into whichever repo becomes the
integration point with no churn.
