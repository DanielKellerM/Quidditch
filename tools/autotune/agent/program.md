<!--
Derived in structure from rightnow-ai/autokernel program.md (MIT, (c) 2026
RightNow AI; see LICENSE.autokernel). Snitch playbook content (c) 2026 ETH Zurich
and University of Bologna, Apache-2.0. SPDX-License-Identifier: MIT AND Apache-2.0
-->
# Snitch kernel optimization playbook (the agent's instructions)

You are optimizing a Quidditch/xDSL Snitch kernel's **dispatch cycles** by
RTL simulation. You are the SEARCH POLICY: you propose configs, the deterministic
oracle measures them, and you keep or revert based on the result. Correctness is
non-negotiable and enforced by the gate, not by you.

## Hardware model (the facts you reason with)

- Cluster = **8 compute cores + 1 DM (data-mover) core**. The DM core has **no FPU**.
- Compute cores stream FMAs via **FREP + SSR** at ~1 FMA/cycle/core steady; the
  measured RTL ceiling is ~0.87 FPU utilization. That is the roofline to chase.
- The **iDMA** moves ~64 bytes/cycle (512-bit). `dual_buffer` overlaps the next
  tile's DMA with the current tile's compute (cost ≈ max(compute,dma) instead of
  sum) at 2× the L1 footprint.
- L1 (TCDM) is small; a tile + its double-buffer must fit.

## The knobs you may propose (v1)

- `l1_tiles = [Mt,Nt,Kt]` — must DIVIDE M,N,K (the spec gate rejects non-divisors).
- `dual_buffer = true|false`.
- `l1_tiles_interchange` — **leave at [2,0,1] (K-outermost)**. Other orders SILENTLY
  MISCOMPILE in the current lowering (the gate catches them as FAIL); do not chase them.
- (Frontier, careful) Group-B xdsl-opt pass knobs — only once tiling is right.

## The loop (one experiment)

```
uv run agent/agent_loop.py --op <name> --propose "M,N,K,db[,ix]"
```
It returns: cost-model estimate, measured dispatch_cycles, correctness verdict,
and KEPT (new best) / REVERT. It logs every experiment and tracks the best.
Use `--status <op>` to see the best + history. Use the cost model to PRE-RANK
your candidates and sim the promising ones first — the sim is ~18 s, the cost
estimate is instant.

## Optimization tiers (Amdahl-ordered: attack the biggest term first)

1. **Maximize tile size** within the TCDM budget — kills the per-L1-tile setup
   term (`TILE_SETUP × ntiles`), which dominates the small/overhead-bound regime.
2. **Enable `dual_buffer`** once there are multiple tiles (DMA becomes visible) —
   not at a single tile, where it only wastes footprint.
3. **Avoid splitting K** — the reduction split is a serial, non-overlappable tax.
4. Interchange stays `[2,0,1]`.
5. Group-B pass-ordering / CSE — last, only after the structure is right.

Move on when FPU-util ≥ 0.90, or you plateau (consecutive reverts), or you hit
your experiment budget.

## Correctness — the hard rule

- EVERY proposal runs through the Tier-1 per-element gate; a new best must also
  pass the Tier-2 independent host cross-check before it is promoted.
- A **faster-but-wrong** config is REVERTED. Never keep it. Never report it.
- **NEVER modify** the harness, the `read_csr(mcycle)` ROI, or the dispatch
  markers — that is gaming the metric, not optimizing the kernel.
- No workarounds: if a config miscompiles, that is a finding to report, not to
  hide. If the kernel is already near the roofline, say so and stop.
