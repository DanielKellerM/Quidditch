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
- [x] StableHLO front-door (S1) — bare StableHLO → byte-exact device .o; tiling de-baked via
      ConfigureForSnitch + config table; vm-c dropped; byte-exact ctest gate. See docs/.
- [x] S1 consolidation — front-door extended to the 64×64 16-wg gemm; autotuner best_<name>.json
      bridges into a ConfigureForSnitch config table.
- [x] S2 — hand-rolled Flow→HAL (Stream/VM/EmitC dropped); byte-exact single-dispatch via the
      two-stage pipeline and the fused quidditch-compile binary. See docs/flow-to-hal.md.
- [x] per-workgroup codegen correctness — 64×64/16-wg gemm proven correct single-cluster for BOTH
      WG0 and WG1 (block(0,0)/(0,1), errors=0/256); the id_x delinearization + output-block addressing
      are right. (The long-suspected single-cluster "hang" was a tracing-speed artifact, not a bug.)
- [x] multi-cluster gemm correctness — DONE on the FULL 16-cluster mesh: 64x64 [16,16,0] gemm is
      BYTE-EXACT (done=144/144, fail=0, 0/4096 mismatches, all 16 blocks). Two independent bugs, both
      firmware (NOT codegen): (1) 4-cluster: the firmware linked the WRONG kernel (matmul_16x16x16
      milestone) instead of matmul_64x64x64 -> swap libgemm_square_kernel.a + rebuild qcs_replay.elf
      (cores_done=36/36 byte-exact). (2) 16-cluster ONLY: a shared-g_table RE-INIT RACE — main.c had
      EVERY cluster's DM core run gw_register_kernels(&g_table) (memset+refill a SHARED L2 .bss table
      @0x7000d028) with only a cluster-local barrier, so a cluster doing its dispatch lookup mid-memset
      saw compute_fn==NULL -> ERR_NO_KERNEL -> skipped dispatch -> its C-block stayed 0 (block(0,1),
      non-deterministic fail-set 1..2). INVISIBLE single-cluster (one DM core = no race), which is why
      all single-cluster validation passed and misdirected toward a phantom writeback race. Fix (nimbus
      main.c): gate registration to cluster 0's DM core + snrt_global_barrier to publish (matches the
      firmware's cluster-0-owns-shared-writes idiom; per-cluster g_table_percl would overflow .bss).
      Pinned via a return-code re-encode diagnostic (rc -> 11..20) that proved ERR_NO_KERNEL not a trap.
- [~] nimbus reproducibility — MUCH closed (commits 4f4f54a + 3416aa6): the host linker is cfg-derived
      (hybrid.ld.in), the build scripts are BASH_SOURCE/env-parameterized (no /home,/scratch,/usr/pack
      hardcodes; RV defaults to the checkout-matching gcc), hostio.h is in-repo, and the pre-split
      `runtime/host/...` includes are fixed so a fresh clone compiles the host+HAL. Remaining: wire a
      Quidditch->kernel-object producer (the 64x64 .a swap was manual), a co-sim run harness in-repo
      (the PRELMODE=5 QCS_DIRECT harness exists but lives in the gwaihir tree), and the DRAM path needs
      a gcc-13-built libcheshire (LTO). No hardcoded machine dirs.
- [~] upstreamability — keep fork changes mergeable; land the pushed stablehlo-frontend branch
      upstream; PR the general fixes (wellen PR open).

## Not now (explicitly deferred)
- Async pipelining / command-buffer caching.
- A second (non-Snitch) ISA backend.

<!-- Update statuses as work lands; per-subgoal detail lives in docs/ + memory. -->
