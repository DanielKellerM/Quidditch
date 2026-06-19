// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "snitch_cluster_cfg.h"
#include "snitch_cluster_raw_addrmap.h"

#undef SNRT_LOG2_STACK_SIZE
#define SNRT_LOG2_STACK_SIZE 11

#ifndef SNRT_TCDM_START_ADDR
#define SNRT_TCDM_START_ADDR SNITCH_CLUSTER_ADDRMAP_CLUSTER_TCDM_BASE_ADDR
#endif
