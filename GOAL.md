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
      fused quidditch-compile binary. See docs/flow-to-hal.md. The redundant SECOND byte-exact
      producer (shell two-stage `--compile-to=flow | iree-opt tail` + check_flow_to_hal.sh gate)
      was retired -> one producer + one `check` gate (quidditch_compile.sh).
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
      `runtime/host/...` includes are fixed so a fresh clone compiles the host+HAL. DONE: the kernel-
      object producer (targets/gwaihir/firmware/gwaihir/produce_kernel_lib.sh) reproducibly compiles the
      vendored gemm .mlir -> the .a the firmware links, deriving xdsl-opt/toolchain/cfg-header from the
      build CMakeCache (byte-identical to the shipped kernel; guards on $xdsl_kernel presence [scalar-
      fallback trap] + QUIDDITCH_L1_BASE [occamy-0x10000000 drift]). DONE: the harness sync — the co-sim
      mods had drifted into two separately-maintained lines (this patch file = DRAM host/PRELMODE=4;
      the gwaihir fork branch = PRELMODE=5 headless + loose vcs.mk/cluster_tile/mem_tile fixes). Merged
      both into ONE fork branch (DanielKellerM/gwaihir merge/cosim-both-final, 3-way on base 4eac10a,
      only tb_gwaihir_top.sv conflicted) and REGENERATED gwaihir-cosim.patch from it, so the patch is
      now a build artifact of a single canonical branch (retires the dual-hand-maintenance drift class);
      tb carries all six arms 0-5 (JTAG/slink/UART/hybrid/DRAM/headless); applies clean to soc/@4eac10a
      (nimbus 65a3b2b; patches/README documents the regen path). Remaining: push merge/cosim-both-final
      to the fork remote (so the patch's source branch is durable, not scratch-only); the DRAM path needs
      a gcc-13-built libcheshire (LTO). No hardcoded machine dirs.
- [~] addrmap single-source hardening — make cfg-derivation the DEFAULT for the whole SoC-address
      edge, not a per-file patch applied after each drift bug. Today gw_raw_addrmap.h is peakrdl-composed
      from THREE sources and cluster/L2/DRAM bases are hand-declared in TWO of them (gwaihir_noc.yml +
      snitch_cluster.json; only nr_clusters is auto-synced via `floogen query | sed`). Consumers are
      split: host/snitch C + hybrid.ld.in derive, but memory.ld hardcodes 0x70000000 (LENGTH already
      disagrees with GW_L2_SPM_TOTAL_SIZE), and codegen reads a PARALLEL clustergen header with a
      hardcoded occamy 0x10000000 fallback. This is the SystemRDL analog of betrusted's svd2utra/UTRA
      single-source (SystemRDL is a superset — expresses the full cluster-array map SVD can't). Steps:
      (1) generate snitch_cluster.json's cluster_base_addr + external_addr_regions from gwaihir_noc.yml
      (extend the proven nr_clusters query+sed) so apertures live in ONE file; (2) memory.ld -> memory.ld.in,
      cpp #include gw_raw_addrmap.h, derive ORIGIN/LENGTH (as hybrid.ld.in already does; also derive its
      chsspm region); (3) feed codegen from the peakrdl artifact, drop the occamy C++ fallback so a
      cfg-less compile fails loud. Retires the memory.ld/host-N/cluster-count drift bug class. NOT a Rust
      UTRA accessor port (C #define/struct + SV is enough).
      DONE + VERIFIED (all three steps):
      - (3) codegen [Quidditch, committed 27e10da on branch addrmap-single-source]: QuidditchTarget.cpp no
        longer SILENTLY falls back to occamy when a cfg header was PASSED but doesn't drive the addrmap; it
        warns loudly (unreadable / lacks QUIDDITCH_L1_BASE). No-header stays the legit occamy default
        (smoketest.mlir relies on it -> a hard-fail was WRONG). The "feed codegen from the peakrdl artifact"
        half is a no-op: snitch_cluster_cfg.h renders 100% from snitch_cluster.json (QUIDDITCH_L1_BASE <-
        alias_region_base/cluster_base_addr), so once (1) makes that json derive from the yaml, codegen
        traces to the single source automatically. Smoke-verified 4 cases.
      - (2) memory.ld [gwaihir repo cosim worktree, uncommitted]: memory.ld -> memory.ld.in (cpp #include
        gw_raw_addrmap.h; ORIGIN=GW_L2_SPM_BASE_ADDR(0), LENGTH=GW_L2_SPM_TOTAL_SIZE) + sw/sw.mk gen rule +
        .gitignore. Fixes the LENGTH drift 0x10000000->0x200000 (128x). Verified: regenerates correctly AND
        qcs_replay.elf relinks (ld.lld, 2MB region, all device ELFs fit <=0x7004a6c0).
      - (1) json<-yaml [gwaihir repo cosim worktree, uncommitted]: Makefile SN_CFG recipe extended (same
        floogen-query+sed idiom as nr_clusters; hex-only sed preserves JSON5 commas) to derive
        cluster_base_addr/offset + l2spm/dram address/length from gwaihir_noc.yml. Verified: only the
        drifted l2spm length changes (0x10000000->0x200000, consistent with (2)); json5 still parses.
      Committed (branch addrmap-single-source in each repo, NOT pushed): gwaihir steps (1)+(2) e7ddfa9
      (memory.ld git-rm-cached -> generated); Quidditch step (3) 27e10da; nimbus chsspm 14ab220.
      - chsspm DONE: hybrid.ld.in (nimbus 14ab220) derives ORIGIN/LENGTH from
        GW_CHESHIRE_INTERNAL_CHESHIRE_SPM_* like l2data (no more hardcode). Verified safe: host .text ~4 KiB.
      - cheshire RDL DONE (root fix): the SPM size was a static mementries=16384 (64 KiB) that ignored the
        LLC cfg; the true size = get_llc_size(cfg) = 128 KiB for DefaultCfg. Parameterized the cheshire
        addrmap with the LLC geometry (LlcSetAssoc/NumLines/NumBlocks/AxiDataWidth, defaults = DefaultCfg)
        and compute spm mementries from it, so the SystemRDL now DEPENDS ON the cfg (SoCs override via
        peakrdl -P). Verified: regenerated gwaihir addrmap emits CHESHIRE_SPM_SIZE=0x20000 (128 KiB), which
        flows through to hybrid.ld.in's derived chsspm. Committed on the cheshire clone (bender clone ->
        working_dir/cheshire, NOT the checkout) branch addrmap-spm-size-from-llc-cfg @8b353d7; a Bender.local
        override now points cheshire at the clone. ISSUE TO FILE: unify so the LLC geometry lives in ONE
        machine-readable cfg both cheshire_pkg (SV) and the RDL read (so -P need not be hand-set).
      Remaining:
      - propagation (outward, deliberately not done unilaterally): push the three addrmap-single-source
        branches + the cheshire branch to the forks; bump nimbus's targets/gwaihir/soc submodule to e7ddfa9;
        point gwaihir's Bender.yml cheshire dep at the fork branch (durable, vs the temp working_dir override).
      - codegen hard-fail NOT feasible at the target-definition site -- emitError from
        getDefaultExecutableTargets prints but the compile still exits 0 (no LogicalResult to propagate;
        verified). The warning is the correct loud signal; a true abort would need the check moved into a
        pass that can signalPassFailure.
- [~] upstreamability — keep fork changes mergeable; land the pushed stablehlo-frontend branch
      upstream; PR the general fixes (wellen PR open).

## Not now (explicitly deferred)
- Async pipelining / command-buffer caching.
- A second (non-Snitch) ISA backend.
- [DEFERRED, reframed] "Fully cut Flow/HAL to drop IREE." A multi-agent validation (8 agents,
  both adversarial judges agreed on the facts) found the hypothesis "drop IREE => simpler infra +
  smaller code + no DRAM" does NOT hold on its own terms: (no-DRAM) ALREADY-MET host-side by the
  host-VM excision + Nimbus split, orthogonal to the device Flow/HAL cut (device wakes+computes
  identically from SPM and DRAM); (smaller-code) mostly ALREADY banked by S1/S2 + Nimbus, and the
  device .o is byte-identical BY DESIGN so it cannot shrink -- only a modest compiler-binary shrink
  remains; (simpler-infra) FALSE -- the cut sheds only Flow::IR + HAL::Target while HAL::IR (a DIRECT
  dep of Codegen::Common/LLVMCPU) + the whole IREE Codegen backend (comprehensive bufferization,
  MaterializeUserConfigs, ReconcileTranslationInfo, HALDispatchABI, tile-to-workgroups) stay linked,
  and A-D would ADD ~3000 owned LOC (byte-exact HALDispatchABI reimpl, a Quidditch dialect for the 6
  hal.interface ops, an executable container, forked Codegen passes) at ~3-6 person-months to emit
  the identical .o. "Cut Flow/HAL" != "drop IREE". Reframe: the three stated goals are DONE; the one
  real reduction taken (collapse the dual byte-exact producer to one, above). Full A-D belongs to a
  SEPARATELY-chartered "sunset IREE decoupling" project judged on the IREE-independence north-star,
  NOT these three goals -- sequence C(container)->B(HALDispatchABI)->A/D behind the byte-exact gate.
  Counter-weight: DEFER != abandon -- if upstream IREE moves/deletes the TargetBackend/compile-to=flow
  surfaces the plugin overrides, revisit before it becomes a forced emergency migration.
- [PARKED, owner=Luca] gwaihir Cheshire-AXI duplicate-R-last bug (colluca/axi): concurrent/
  double-buffered reads through the wide→narrow DW downsizer re-accept the terminal R beat →
  AR id-counter underflow ($fatal). Blocks double-buffering + small tiles on the 16-cluster RTL
  (single-buffered runs clean). Root-caused from the full-system waveform (counter 0→0x3f, 2-pop/1-push,
  held not toggling) + candidate axi_dw_downsizer.sv accept-gate patch. UPDATE: the IP-level directed
  discriminator now RAN (branch dw-downsizer-rlast-repro, tb_axi_dw_downsizer with sustained wide-R
  backpressure + same-low-ID overlapping reads, 2:1..8:1 ratios, 10 configs × 3 seeds) and the isolated
  downsizer is EXONERATED — cnt_underflow never fires, 0 errors. So the accept-gate patch is likely the
  WRONG location and was NOT applied; evidence now favors the upstream FlooNoC wide-R (reorder/id-width)
  path with the demux as mere detector. Next: reproduce in the FlooNoC join / full gwaihir sim. Docs:
  nimbus docs/gwaihir-axi-dw-downsizer-rlast-bug.md (finding folded in). NOT our compiler north star;
  only ~10% (double-buffering) upside. Tsunami FSDB reader fixed en route.

<!-- Update statuses as work lands; per-subgoal detail lives in docs/ + memory. -->
