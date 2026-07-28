# GOAL

## North star
A scalable, upstreamable ML compiler for Snitch clusters: one cfg generates any
cluster/SoC config, behind a declared runtime interface, deployable to real HW via
Nimbus, competitive with hand-written kernels — with a StableHLO frontend and
minimal IREE dependence (IREE is being sunset).

## Sub-goals
- [x] interface-v0 — declared, versioned, cfg-parameterized Snitch runtime interface.
- [x] Nimbus split — gwaihir deployment carved out; Quidditch is the pure backend.
- [x] host-VM excision ("size fix") — minimal IREE-free QCS host; boots + runs cluster.
- [-] WG1 correctness — 64×64/16-wg gemm block(0,1) missing on gwaihir; deferred (a
      multi-cluster writeback/visibility race, codegen-neutral to StableHLO). See docs/.
- [x] StableHLO front-door (S1) — bare StableHLO → byte-exact device .o; tiling de-baked
      via ConfigureForSnitch + config table; vm-c dropped; byte-exact regression gate in ctest.
- [x] S1 consolidation — front-door extended to the 64×64 16-workgroup gemm; the autotuner
      best_<name>.json bridges into a ConfigureForSnitch config table.
- [x] S2 — hand-rolled Flow→HAL (Stream/VM/EmitC dropped); byte-exact single-dispatch via
      both the two-stage pipeline and the fused quidditch-compile binary. See docs/flow-to-hal.md.
- [~] upstreamability — fork changes mergeable; PR the general fixes (wellen PR open).
- [x] multi-cluster gemm end-to-end on gwaihir RTL (with the WG1 gap now known).

## Not now (explicitly deferred)
- FSDB/tsunami waveform tooling — DONE + parked; only revisit for a concurrency trace.
- Async pipelining / command-buffer caching.
- A second (non-Snitch) ISA backend.

<!-- Update statuses as work lands; per-subgoal detail lives in docs/. -->
