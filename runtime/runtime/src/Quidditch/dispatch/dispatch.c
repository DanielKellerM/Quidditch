
#include "dispatch.h"

#include <iree/base/alignment.h>

#include <assert.h>
#include <encoding.h>
#include <riscv_decls.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <team_decls.h>

// snRuntime sync (sync_decls.h is C++-only since the DCA/collective refactor:
// it uses C++ default arguments). The symbol is re-exported with C linkage by
// runtime/snitch_cluster/rtl/src/quidditch_snrt_exports.cc.
extern void snrt_cluster_hw_barrier(void);

// TODO: This only works for a single cluster by using globals. Should be
// cluster local.
static iree_hal_executable_dispatch_v0_t configuredKernel;
static const iree_hal_executable_environment_v0_t* configuredEnvironment;
static const iree_hal_executable_dispatch_state_v0_t* configuredDispatchState;
// Initialize processor_id to the "no workgroup assigned" sentinel (-1) so that
// workers woken by the DM core's init-time cluster barriers (before the first
// real dispatch) skip execution instead of calling a not-yet-configured (null)
// kernel. This mirrors the state reset_workgroup_state() establishes between
// dispatches.
static iree_alignas(64)
    iree_hal_executable_workgroup_state_v0_t configuredWorkgroupState[8] = {
        [0 ... 7] = {.processor_id = (uint32_t)-1},
};
static atomic_bool error = false;
static atomic_bool exit = false;
static uint8_t nextCoreToUse = 0;

static void reset_workgroup_state() {
  nextCoreToUse = 0;
  // Sentinel value for processor has no workgroup assigned.
  for (iree_hal_executable_workgroup_state_v0_t* iter =
           configuredWorkgroupState;
       iter !=
       configuredWorkgroupState + IREE_ARRAYSIZE(configuredWorkgroupState);
       iter++)
    iter->processor_id = -1;
}

bool quidditch_dispatch_errors_occurred() { return error; }

void quidditch_dispatch_set_kernel(
    iree_hal_executable_dispatch_v0_t kernel,
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state) {
  configuredKernel = kernel;
  configuredEnvironment = environment;
  configuredDispatchState = dispatch_state;
  reset_workgroup_state();
}

int quidditch_dispatch_enter_worker_loop() {
  // Worker communication contract: Workers go into the hardware barrier
  // immediately, DMA core wakes them using a hardware barrier, works wake DMA
  // core using another hardware barrier.

  while (!exit) {
    snrt_cluster_hw_barrier();
    if (exit) break;

    // Cores not in use may spuriously wake up at any time and need to reenter
    // the barrier asap.
    iree_hal_executable_workgroup_state_v0_t* workgroupState =
        &configuredWorkgroupState[snrt_cluster_core_idx()];
    if (workgroupState->processor_id == -1) {
      continue;
    }

    read_csr(mcycle);
    if (configuredKernel(configuredEnvironment, configuredDispatchState,
                         workgroupState))
      error = true;

    read_csr(mcycle);

    // Signal being done.
    snrt_cluster_hw_barrier();
  }
  return 0;
}

void quidditch_dispatch_quit() {
  exit = true;
  snrt_cluster_hw_barrier();
}

void quidditch_dispatch_queue_workgroup(
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  configuredWorkgroupState[nextCoreToUse] = *workgroup_state;
  configuredWorkgroupState[nextCoreToUse].processor_id = nextCoreToUse;
  nextCoreToUse++;
  if (nextCoreToUse != snrt_cluster_compute_core_num()) return;

  quidditch_dispatch_execute_workgroups();
}

void quidditch_dispatch_queue_subgroups(
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  // TODO: This is unnecessarily complicated for the subgroup distribution that
  // we want to perform.
  for (iree_hal_executable_workgroup_state_v0_t* iter =
           configuredWorkgroupState;
       iter !=
       configuredWorkgroupState + IREE_ARRAYSIZE(configuredWorkgroupState);
       iter++)
    *iter = *workgroup_state;
  quidditch_dispatch_start_executing_workgroup();
}

void quidditch_dispatch_execute_workgroups() {
  // First wake workers.
  quidditch_dispatch_start_executing_workgroup();

  // Then wait for workers to be done.
  quidditch_dispatch_wait_for_workgroup();
}

void quidditch_dispatch_start_executing_workgroup() {
  snrt_cluster_hw_barrier();
}

void quidditch_dispatch_wait_for_workgroup() {
  snrt_cluster_hw_barrier();
  reset_workgroup_state();
}
