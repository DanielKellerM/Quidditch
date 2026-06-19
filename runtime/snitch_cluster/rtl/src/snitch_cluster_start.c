// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#define SNRT_INIT_TLS
#define SNRT_INIT_BSS
#define SNRT_CRT0_PRE_BARRIER
#define SNRT_INVOKE_MAIN
#define SNRT_CRT0_POST_BARRIER
#define SNRT_CRT0_EXIT

extern volatile uint32_t tohost;

static inline volatile uint32_t* snrt_exit_code_destination() {
  return (volatile uint32_t*)&tohost;
}

// Pull in the inline bodies (snrt_exit, snrt_exit_default, ...) guarded by the
// SNRT_CRT0_* macros above. The upstream impl/snitch_cluster_start.h does this;
// this Quidditch start file must too, otherwise snrt_main (in start.c) calls a
// snrt_exit whose definition is never seen and the symbol is left undefined.
#include "start.h"

#include "start.c"
