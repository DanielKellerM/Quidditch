// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Snitch custom-ISA facts the codegen emits directly as inline assembly (not via
// an snRuntime call): the cluster hardware-barrier CSR and the xDMA status
// instruction. SSR (scfgw / csrsi ssr) and FREP (frep.o) are emitted on the xDSL
// side as riscv_snitch dialect ops (xdsl/xdsl/dialects/riscv_snitch.py); this
// header is the C++-side counterpart so the magic encodings live in one named
// place instead of inline string literals.

#ifndef QUIDDITCH_CONVERSION_SNITCHISA_H_
#define QUIDDITCH_CONVERSION_SNITCHISA_H_

namespace quidditch {

// Cluster hardware-barrier CSR (the snrt_cluster_hw_barrier equivalent): reading
// it blocks until every core in the cluster has arrived.
inline constexpr unsigned kSnitchClusterHwBarrierCsr = 0x7c2;

// The xDMA status instruction `dmstati rd, 0`, as R-type `.insn` fields (Snitch
// custom-1 opcode). Reads the DMA completion counter of the calling core.
inline constexpr unsigned kSnitchDmaOpcode = 0x2b;
inline constexpr unsigned kSnitchDmaStatFunct3 = 0x0;
inline constexpr unsigned kSnitchDmaStatFunct7 = 0x4;  // 0b100

}  // namespace quidditch

#endif  // QUIDDITCH_CONVERSION_SNITCHISA_H_
