#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Analytic dispatch-cycle RANKER for Snitch-cluster GEMM tilings.

Purpose: let the autotuner sim only the top-K of a large l1_tiles x dual_buffer
grid instead of all of it. The model RANKS configs; it does NOT predict cycles
(absolute predictions are ~2x low -- see "Model limits" below). Validated on 9
measured gemm_square configs: Spearman rho = 1.0, and rho >= 0.90 holds in 100%
of 2000 draws that jitter every constant by +-40%, so the ORDER comes from the
tiling structure, not a knife-edge fit of the magic numbers.

Hardware model (8 compute cores + 1 DMA core, the Quidditch dispatch in
runtime/.../dispatch/dispatch.c, kernel from xDSL test_lower_linalg_to_snitch).
A 1D dispatch tiles MxNxK into ceil(M/Mt) x ceil(N/Nt) x ceil(K/Kt) L1 tiles;
the DM core stages each tile L2->L1 over the wide iDMA while the 8 cores run an
FREP/SSR-streamed inner FMA loop. dual_buffer overlaps the two (max instead of
sum) at 2x the L1 footprint.

  dispatch ~= FIXED + BARRIER
            + ( max | sum )(compute_total, dma_total)      # max iff dual_buffer
            + KSPLIT * out_tiles * (nK - 1)                 # reduction-split tax
            + MSPLIT * nK * (nM - 1) * nN                   # parallel-frag tax

CONSTANTS (named + sourced):
  NCORE      = 8     compute cores            cfg/default.json: 8x compute_core_template
  DMA_BPC    = 64    iDMA bytes/cycle         cfg/default.json: dma_data_width=512 bit
  F64        = 8     bytes per f64 element    gemm_square.mlir: f64 operands
  FMA peak   = 1 FMA/cycle/core (FREP+SSR steady state). xDSL lowers the inner
               reduction to a frep_outer hardware loop fed by SSR streams
               (convert_riscv_scf_for_to_frep.py); steady throughput is 1
               streamed FMA/cycle/core, the documented ~0.87 RTL FPU util.
  TILE_SETUP = 180   per-L1-tile non-overlappable setup: 3x SSR reconfig
               (ssr_set_dimension_{bound,stride,source,dest}, snitch.py) + frep
               loop prologue + scalar address calc the CSE can't hide across the
               tile boundary. The DOMINANT term in this small/overhead-bound
               regime (16^3 is too small to be roofline-bound).
  DMA_FIX    = 30    per-iDMA-transfer descriptor/launch latency.
  BARRIER    = 120   dispatch fan-out + rejoin. Two snrt_cluster_hw_barrier()
               calls (queue_subgroups/start + wait_for_workgroup) plus glue;
               dispatch.c.
  FIXED      = 120   one-time dispatch glue inside the mcycle ROI (main.c/harness.c).
  KSPLIT     = 140   tax PER EXTRA reduction (K) pass per output tile. Splitting
               the K (reduction) loop forces the C tile to be re-accumulated and
               re-touched across passes -- a serial, non-overlappable dependency
               (l1_tiles_interchange=[2,0,1] in ConfigureForSnitch.cpp puts K
               outermost, so each output tile pays nK serial passes).
  MSPLIT     = 60    tax PER EXTRA parallel (M) tile per N/K trip. Splitting the
               distributed M dim fragments the 8-core work into more tile-loop
               trips, each re-incurring SSR/frep setup and a partial subgroup.

These six are fit to the 9-point ground truth but the ranking is INSENSITIVE to
them (see module docstring). Treat them as ordering weights, not physics.
"""
import math

# ---- Snitch cluster constants: cfg-DERIVED, not hand-copied ----
# The cluster .cfg (snitch_cluster/cfg/default.json) generates the RTL + the
# runtime header snitch_cluster_cfg.h. Read the same artifacts so the cost model
# cannot drift from the hardware the RTL was built for. See CFG_DRIVEN_TARGET.md.
import os as _os
import re as _re

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_CFG_HEADER = _os.path.join(_HERE, "../../snitch_cluster/sw/runtime/impl/snitch_cluster_cfg.h")
_CFG_JSON = _os.path.join(_HERE, "../../snitch_cluster/cfg/default.json")


def _grep_int(path, pattern):
    try:
        m = _re.search(pattern, open(path).read())
        return int(m.group(1)) if m else None
    except OSError:
        return None


def _ncore_from_cfg(default=8):
    nr = _grep_int(_CFG_HEADER, r"#define\s+CFG_CLUSTER_NR_CORES\s+(\d+)")
    dm = _grep_int(_CFG_HEADER, r"#define\s+SNRT_CLUSTER_DM_CORE_NUM\s+(\d+)")
    return nr - dm if (nr is not None and dm is not None) else default


def _dma_bpc_from_cfg(default=64.0):
    # Prefer the generated header (the artifact the C runtime consumes); it now
    # emits SNRT_DMA_DATA_WIDTH (the snitch_cluster_cfg.h.tpl add). Fall back to the
    # cfg's dma_data_width (bits) for an older header. Either way: cfg-derived.
    bits = _grep_int(_CFG_HEADER, r"#define\s+SNRT_DMA_DATA_WIDTH\s+(\d+)")
    if bits is None:
        bits = _grep_int(_CFG_JSON, r"dma_data_width\s*:\s*(\d+)")
    return bits / 8.0 if bits else default


NCORE = _ncore_from_cfg()      # compute cores; default cfg: 9 - 1 = 8
DMA_BPC = _dma_bpc_from_cfg()  # iDMA bytes/cycle; default cfg: 512 bit / 8 = 64
F64 = 8                        # bytes per f64

# ---- fitted ordering weights (ranking is robust to +-40% on each) ----
TILE_SETUP = 180   # per-L1-tile SSR+frep setup (overhead-bound regime: dominant)
DMA_FIX = 30       # per-iDMA-transfer launch latency
BARRIER = 120      # dispatch fan-out + rejoin (2 hw barriers)
FIXED = 120        # one-time dispatch glue in the ROI
KSPLIT = 140       # reduction-split tax, per extra K pass per output tile
MSPLIT = 60        # parallel-fragmentation tax, per extra M tile per N/K trip


def cost_model(l1_tiles, dual_buffer, problem=(16, 16, 16)):
    """Estimated dispatch cycles for ranking. NOT an absolute predictor.

    l1_tiles    : [M_t, N_t, K_t] L1 tile shape.
    dual_buffer : bool (accepts "true"/"false" too) -- overlap DMA with compute.
    problem     : [M, N, K] GEMM shape (default the 16x16x16 sample).
    Returns a float cost; rank configs by ASCENDING cost and sim the top-K.
    """
    if isinstance(dual_buffer, str):
        dual_buffer = dual_buffer.strip().lower() == "true"
    M, N, K = problem
    Mt, Nt, Kt = l1_tiles
    nM = math.ceil(M / Mt)
    nN = math.ceil(N / Nt)
    nK = math.ceil(K / Kt)
    ntiles = nM * nN * nK
    out_tiles = nM * nN

    # (a) COMPUTE roofline: MACs distributed over the 8 cores along the parallel
    # M dim (ceil(Mt/NCORE) rows/core), 1 FMA/cyc/core steady via FREP+SSR, plus
    # the per-tile SSR/frep setup that cannot hide under the stream.
    rows_per_core = math.ceil(Mt / NCORE)
    steady = rows_per_core * Nt * Kt
    compute_total = (steady + TILE_SETUP) * ntiles

    # (b) DMA roofline: stage A(Mt*Kt)+B(Nt*Kt) per tile over the wide iDMA, plus
    # the C(Mt*Nt) tile per output tile; each transfer pays a fixed launch cost.
    dma_ab = (Mt * Kt + Nt * Kt) * F64 / DMA_BPC
    dma_c = (Mt * Nt) * F64 / DMA_BPC
    dma_total = (dma_ab + DMA_FIX) * ntiles + (dma_c + DMA_FIX) * out_tiles

    # (c) dual_buffer overlaps DMA with compute -> max(); else they serialize.
    body = max(compute_total, dma_total) if dual_buffer else compute_total + dma_total

    # serialization taxes that the roofline body does not capture
    ksplit_pen = KSPLIT * out_tiles * (nK - 1)        # reduction split: serial C accumulate
    msplit_pen = MSPLIT * nK * (nM - 1) * nN          # parallel-dim fragmentation

    return body + ksplit_pen + msplit_pen + FIXED + BARRIER


# ----------------------------- validation -----------------------------
# (tag, l1_tiles, dual_buffer, measured_dispatch_cycles) -- the 9-point ground
# truth from tools/autotune/results.tsv (Verilator, gemm_square 16x16x16 f64).
_GROUND_TRUTH = [
    ("16x16x16_dbtrue",  [16, 16, 16], True,  1420),
    ("16x16x16_dbfalse", [16, 16, 16], False, 1420),
    ("16x8x16_dbtrue",   [16, 8, 16],  True,  1768),
    ("8x8x16_dbtrue",    [8, 8, 16],   True,  2102),
    ("16x8x8_dbtrue",    [16, 8, 8],   True,  2266),
    ("8x16x8_dbtrue",    [8, 16, 8],   True,  2571),
    ("8x8x8_dbtrue",     [8, 8, 8],    True,  2958),
    ("8x8x8_dbfalse",    [8, 8, 8],    False, 3275),
    ("4x8x8_dbtrue",     [4, 8, 8],    True,  5725),
]


def _spearman(a, b):
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0] * len(x)
        for rk, i in enumerate(order):
            r[i] = rk
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


def _validate():
    meas = [g[3] for g in _GROUND_TRUTH]
    pred = [cost_model(g[1], g[2]) for g in _GROUND_TRUTH]
    rho = _spearman(meas, pred)
    order_m = sorted(range(len(_GROUND_TRUTH)), key=lambda i: meas[i])
    order_p = sorted(range(len(_GROUND_TRUTH)), key=lambda i: pred[i])
    print(f"Spearman rho = {rho:.4f}\n")
    print(f"{'#':>2}  {'tag':18} {'meas':>6} {'pred':>7}  pred_rank")
    for i, gi in enumerate(order_m):
        tag = _GROUND_TRUTH[gi][0]
        pr = order_p.index(gi)
        flag = "ok" if pr == i else f"MIS -> {pr + 1}"
        print(f"{i+1:>2}  {tag:18} {meas[gi]:>6} {pred[gi]:>7.0f}  {flag}")
    return rho


if __name__ == "__main__":
    _validate()
