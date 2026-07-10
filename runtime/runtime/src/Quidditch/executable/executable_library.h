// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/hal/local/executable_library.h"

// The export-table + library structs (the compute_core/dma_core fork of the v0.6
// IREE layout) live in one shared header, so the rv64 runtime, the rv32 firmware,
// and the compiler's emission cannot drift.
#include "Quidditch/quidditch_executable_abi.h"
