// Copyright 2024 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Canonical declaration of the C-callable Snitch runtime interface used across
// the Quidditch host/runtime split: the stable snRuntime symbols the compiler
// emits calls to and the runtime calls directly. The DEFINITIONS live in
// quidditch_snrt_exports.cc, which re-exports the real snRuntime under these asm
// names into libsnRuntime. This header is the single declaration consumers
// include, so a signature cannot drift between call sites (it did: command_buffer.c
// bare-externed snrt_dma_start_1d, and the compiler synthesizes its own
// LLVMFuncOps). Keep this in agreement with quidditch_snrt_exports.cc.
//
// Freestanding: only <stddef.h>/<stdint.h>, so it compiles for the rv64 host and
// the rv32 firmware and links against libsnRuntime.

#ifndef QUIDDITCH_SNRT_ABI_H_
#define QUIDDITCH_SNRT_ABI_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// iDMA: start a 1-D transfer dst<-src of `size` bytes; returns a transfer id.
uint32_t snrt_dma_start_1d(void* dst, void* src, size_t size);

// iDMA: start a 2-D transfer of `repeat` rows of `size` bytes with per-side strides.
uint32_t snrt_dma_start_2d(void* dst, void* src, size_t size, size_t dst_stride,
                           size_t src_stride, size_t repeat);

// Block until all outstanding iDMA transfers on this core have completed.
void snrt_dma_wait_all(void);

// Index of the calling compute core within its cluster.
uint32_t snrt_cluster_core_idx(void);

// Cluster hardware barrier: returns once every core in the cluster has arrived.
void snrt_cluster_hw_barrier(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // QUIDDITCH_SNRT_ABI_H_
