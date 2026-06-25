# Phase 2: host-device split on a Carfield RTL sim

The goal endpoint (see [`host-device-split.md`](host-device-split.md)): run the
split for real on RTL — the IREE VM on a CVA6 host driving the Snitch cluster as
accelerator, with iree+xdsl producing the split (a `vm-bytecode` host module +
Snitch rv32 kernel libs). Phase 1 ([`host-device-split-phase1.md`](host-device-split-phase1.md))
proved the whole flow on the dev box with stub kernels and no RTL; Phase 2
replaces the stubs/processes with real RTL, toolchains, and shared SPM.

This is a multi-week RTL + cross-toolchain campaign (~31–47 h focused), not a
dev-box increment. No architectural blockers found — Phase 1 did the hard ABI
and integration work; Phase 2 is bring-up.

## SoC: Carfield, not just Cheshire

Cheshire (a CVA6 SoC) has **no** Snitch cluster. **Carfield** = Cheshire + a
Snitch-based compute tile + mailbox + L2 SPM + iDMA — the host-device topology in
RTL. Carfield is now a Bender dep (done — step 1a): `carfield @ 0573b403` +
`cheshire` realigned to `0c95210c` (the rev carfield's frozen lock pins, so the
two roots agree). Seeded from carfield's frozen `Bender.lock` + `bender checkout`
(the cheshire pattern; `bender update` would hit the stale-axi-pin hydra).
`bender checkout` resolves 63 deps.

### ⚠ Finding: Carfield uses `spatz`, not `snitch_cluster`

Carfield's compute tile is **`spatz`** (the Snitch+vector cluster,
`rev b8fea8f2`), NOT the standalone **`snitch_cluster`** IP that Quidditch's
runtime/codegen target (snRuntime, the `snitch` executable format, the existing
`snitch_cluster` submodule). `bender path snitch_cluster` errors; `bender path
spatz` resolves. So Phase 2 has an unresolved SoC-integration decision that the
original plan glossed over:
- **Option A** — target spatz's Snitch cores from Quidditch (spatz IS Snitch-based,
  but its TCDM/runtime/packaging differ from snitch_cluster; the kernels + snRuntime
  assumptions need re-validation).
- **Option B** — use a Carfield configuration / fork that instantiates the
  `snitch_cluster` IP (if one exists), or add `snitch_cluster` as its own Bender
  dep and wire a custom integration.
- **Option C** — build a minimal Cheshire+snitch_cluster co-sim ourselves rather
  than full Carfield.
This must be settled before the sim + firmware steps; it changes which RTL is
built and which snRuntime the rv32 firmware links.

### ✅ Resolution: target Occamy, not Carfield

Scouting found the right vehicle: **Occamy** (`pulp-platform/occamy`) is a
CVA6 + `snitch_cluster` SoC ("high-efficiency FP-compute SoC"), pinning
`snitch_cluster @ rev occamy`. Quidditch's own `snitch_cluster` submodule is
forked from that same line (`occamy-diverge` = `4174fa74`, +189 commits), so it
is the **same IP base** — the kernels, snRuntime, and `snitch` executable format
already target it. Occamy is the aligned Phase-2 SoC; Carfield/spatz is the wrong
target. (The design doc's offload model — arXiv 2505.05911, "Taming Offload
Overheads" — is itself an Occamy study; Quidditch's fork already pins the
axi-multicast NoC fork that paper introduces.)

**Occamy already ships the offload flow our QCS ABI mirrors** (the strongest
signal): `target/sim/sw/host/apps/offload/src/offload.c` — a `comm_buffer` job
descriptor in shared DRAM/L2, host->device doorbell via the cluster CLINT-set /
`msip`, device->host completion via `msip` to CVA6 hart 0, and the Snitch entry
point in a SoC scratch reg. QCS maps onto `comm_buffer` + scratch + CLINT ~1:1.

Sim/host facts (from the occamy tree):
- **Sim = Questa (`occamy_top.vsim`)** or VCS; Verilator is NOT wired up (the
  README's `.vlt` targets are stale). Fine for us: Quidditch already runs the
  cluster sim on `questa-2023.4 vsim` (Phase-0/1 gemm_square). Use
  `cfg/single-cluster.hjson` (CVA6 + 1 cluster); the full 24-cluster config is
  too big for RTL sim.
- **CVA6 host is bare-metal**: `target/sim/sw/host/runtime/{start.S,host.c,host.ld}`
  (`-nostartfiles`, HTIF `tohost` exit, DRAM @ 0x80000000, no libc/malloc/FS). An
  ELF is preloaded via `fesvr`. The rv64 IREE VM must run freestanding here.
- **snitch_cluster toolchain**: the snitch LLVM (occamy's `deps/snitch_cluster`),
  same family Quidditch uses.

### Phase-2 risks (revised, Occamy)

1. **Bare-metal rv64 IREE VM** — the gating item: port the IREE runtime + a
   QCS-recording cluster HAL onto a freestanding CVA6 (allocator over DRAM, no OS
   services), linked against occamy's host runtime.
2. **snitch_cluster rev skew** — reconcile occamy's `da57b043` pin with
   Quidditch's +189-commit fork (iDMA 0.6.5, axi-multicast); bump one to the
   other. Plus shared-IP rev clashes (axi/common_cells/idma) when occamy sits
   alongside the cheshire dep — seed from occamy's frozen lock to avoid drift.
3. **Questa license/env** (IIS `questa-2023.4`) — available, but the occamy sim
   build is heavier than the cluster-only one.

### Revised first steps

1. Pivot the Bender dep: drop carfield, add occamy (frozen-lock seed); keep
   cheshire. **DONE** — deps are now `cheshire@0.3.1` + `occamy@d5d0d832`;
   `bender checkout` resolves 33 deps and `snitch_cluster` (rev `da57b043`)
   resolves transitively via occamy (it did not under carfield/spatz). Known
   non-fatal warning: occamy's path-vendored deps (cva6/opentitan_peripherals/
   axi_tlb) emit W13 (their `hw/vendor/...` paths resolve relative to the repo
   root); harmless for resolution, to be sorted when the RTL build needs cva6.
2. Smoke-test occamy's own offload demo on Questa. **DONE — RTL loop validated.**
   Built `occamy_top.vsim` (single-cluster, Questa, Errors:0) and ran the bundled
   offload demo: `[SUCCESS] Program finished successfully`, clean `$finish`, 9
   device harts traced (8 compute + 1 DM), host->CLINT-doorbell->cluster->
   completion confirmed on real RTL (reproduced independently). Caveat: the run
   reports `Errors: 8` — all the cosmetic upstream `stream_xbar.sv:186 "data_i is
   unstable"` transient assertion (does NOT abort; program completes). A CI gate
   keyed on `Errors:` must whitelist these.
3. Then swap occamy's host `main()` for an IREE-VM-driven QCS recorder mapped onto
   `comm_buffer` + CLINT, and the device side for the QCS replayer on real
   snRuntime. (This is the remaining integration; the offload demo is the template.)

### Environment facts (validated; don't re-discover)

The occamy sim builds + runs on this IIS box. Toolchains/paths:
- **bender** `/home/dankeller/.cargo/bin/bender` v0.32 — `bender checkout` works on
  occamy's frozen lock (ignore the 0.27.1 note); never `bender update`.
- **Questa** `questa-2023.4` (`/usr/sepp/bin`), Sim-64 2023.4, license available;
  Makefiles use `$(QUESTA_SEPP) vsim/vlog/vopt`.
- **CVA6 host GCC** `RISCV_GCC_BINROOT=/usr/pack/riscv-1.0-kgf/riscv64-gcc-12.2.0/bin`
  (riscv64-unknown-elf-gcc 12.2.0).
- **Snitch device LLVM** `LLVM_BINROOT=/usr/pack/riscv-1.0-kgf/pulp-llvm-0.12.0/bin`
  (clang 12.0.1, riscv32 `-mcpu=snitch`).
- **python** anaconda3-2022.05 (3.9.12) venv + `pip install -r python-requirements.txt`.
- **verible** NOT preinstalled — `make rtl` needs it; drop the v0.0-3222 tarball into
  `tools/verible/bin` and add to PATH (occamy's own recipe).
- Built workspace + `env.sh` (sources all the above) live under
  `/scratch/dankeller/snitch-compiler/occamy-phase2/` (untracked; rebuildable).
  Re-run: `source env.sh && cd target/sim && ./bin/occamy_top.vsim
  sw/host/apps/offload/build/offload-axpy.elf`.

## What exists vs. must be built

| Piece | Status |
|---|---|
| Command-stream ABI, shared-arena allocator, HAL device, replayer | ✅ Phase 1 |
| Verilator | ✅ on box (`/usr/sepp/bin/verilator`); snitch_cluster has a `.vlt` sim target |
| rv32 Snitch toolchain (snRuntime, iDMA) | ✅ `runtime/toolchain/Dockerfile` + snitch_cluster |
| rv64 cross-compile config | ✅ `iree/build_tools/cmake/linux_riscv64.cmake` |
| Carfield Bender dep + Verilator sim target | ✗ add + verify |
| Codegen split (`vm-bytecode` host + kernel libs) | ✗ `quidditch_module.cmake:124` hardcodes `--output-format=vm-c`; wire a split |
| rv64 CVA6 IREE runtime (VM + cluster HAL driver) | ✗ build (Phase-1 HAL code ports as-is) |
| rv32 firmware job-loop (real `snrt_*`, not pthread stubs) | ✗ port `runtime/host/firmware/cluster_replay.c` onto snRuntime/iDMA + dispatch.c fan-out |
| L2-SPM mmap + host-VA↔device-PA mapping on Carfield | ✗ the crux |

## Ordered checklist

1. Add Carfield to Bender (seed its frozen lock); verify a Verilator sim target.
2. Wire `--output-format=vm-bytecode` (host module) + separate Snitch kernel-lib
   emission into `quidditch_module.cmake` / `QuidditchTarget.cpp`.
3. Cross-build the IREE runtime for rv64 CVA6, linking the Phase-1 cluster HAL
   driver (`runtime/host/hal/cluster/`).
4. Port the firmware job-loop to rv32: real `snrt_cluster_hw_barrier` /
   `snrt_dma_start_1d` (replacing the pthread/memcpy stubs) feeding `dispatch.c`.
5. Map the Carfield L2-SPM aperture (host-VA ↔ device-PA) + apply the
   self-invalidation fence/flush discipline; validate PA translation in isolation
   *before* GEMM (a buffer write/read across the AXI-shared region).
6. Co-sim bring-up: host ELF on CVA6 + firmware ELF on the cluster in one Carfield
   Verilator sim; doorbell = CLINT `msip`, completion polled (Phase 3 swaps to the
   PLIC IRQ).
7. End-to-end GEMM on RTL; diff vs the on-cluster oracle.

## Risks

- **Crux — host-VA → device-PA binding translation** on real RTL (Phase 1 made it
  explicit + testable; validate the L2-SPM map first).
- **Coherence**: Cheshire self-invalidation → explicit fences on the shared region.
- **Carfield address map** (L2-SPM base, mailbox/CLINT/PLIC) must be read from RTL.
- Verilator Carfield build time (~1–2 h) + the Bender frozen-lock seed.

## Phase-2 retarget: gwaihir (the real target) + two feasibility GOs

After scouting, the SoC is **gwaihir** (`pulp-platform/gwaihir`, "Lago-Mio"):
Cheshire host + a FlooNoC mesh of snitch_cluster tiles. It supersedes Occamy as
the build target — Occamy was a validated stepping-stone (its sim ran the offload
loop on Questa, proving the IIS toolchains + the offload pattern). Why gwaihir:
it matches the goal's Cheshire framing, is the active IIS/PULP MLSys platform,
ships a CI-tested host->cluster offload example, and its `snitch_cluster@main` is
a clean 187-commit descendant of Quidditch's fork base (offload ABI unchanged).
Note: gwaihir clusters are heterogeneous (snitch_cluster **+** spatz); Phase 2
targets the **snitch_cluster** tile (per-cluster), ignoring/coexisting with spatz.

**QCS -> gwaihir mapping (from `sw/cheshire/tests/simple_offload.c`):**
- shared region -> **L2 SPM** (`l2_spm` endpoint @ `0x7000_0000`, reachable by the
  cluster wide iDMA over FlooNoC); command stream + buffers + job descriptor here.
- doorbell -> cluster `peripheral_reg.cl_clint_set` (CLINT msip); entry via
  `scratch[1]`, return-addr via `scratch[0]`.
- completion/status -> polled word in L2 SPM (the `return_code_array` pattern);
  PLIC-upgradeable later (`enable_external_interrupts: true`).
- snitch_cluster plan: pin gwaihir's `snitch_cluster@main` (2fa38482), re-apply
  Quidditch's device-half (dispatch fan-out + kernels) on top.

**Bender:** consume gwaihir as the single top dep (its frozen lock drives;
cheshire + snitch_cluster come in transitively) — do NOT co-mount a separate
cheshire dep (shared-IP rev clashes).

**Sim/build (IIS, all validated):** Questa `questa-2023.4` (also VCS); host
riscv64-gcc-12.2.0; device pulp snitch-LLVM; `iis-env.sh` (bender checkout + uv
sync). Use the mini/1-cluster NoC config (`cfg/mini_gwaihir_noc.yml`); the full
16-cluster mesh is heavy. PD repo is private but not needed for `target/sim`.

### Feasibility GO #1 — RTL vehicle runs (validated on Occamy)
occamy_top.vsim (single-cluster) built + ran the bundled offload demo on Questa:
`[SUCCESS]`, host->CLINT-doorbell->cluster->completion confirmed. gwaihir uses the
same Questa flow + offload pattern.

### Feasibility GO #2 — bare-metal rv64 IREE VM compiles freestanding
`iree_base` + `iree_vm` + `iree_vm_bytecode_module` cross-compile + link for rv64
under `-nostartfiles -ffreestanding`; a VM-using test linked to a 482 KB rv64 ELF
with zero undefined refs. IREE officially supports bare-metal
(`CMAKE_SYSTEM_NAME=Generic`). Recipe (build dir
`/scratch/dankeller/snitch-compiler/iree-rv64-baremetal/build`, toolchain file
alongside): `IREE_BUILD_COMPILER=OFF IREE_ENABLE_THREADING=OFF
IREE_SYNCHRONIZATION_DISABLE_UNSAFE=ON IREE_HAL_*_DEFAULTS=OFF IREE_USE_LINKER=bfd
IREE_ENABLE_WERROR_FLAG=OFF IREE_HOST_BIN_DIR=<host-iree>/tools`, GCC toolchain
file with `-D__intN_t_defined` shim (newlib inttypes gap). Remaining glue: newlib
syscall stubs (`_sbrk` over the DRAM/L2-SPM arena, `_write`/`_exit` -> HTIF) —
Quidditch's rv32 `syscalls.c` is the pattern; or feed the VM a custom
`iree_allocator_t` and define `IREE_ALLOCATOR_SYSTEM_CTL` to it.

### Feasibility GO #3 — gwaihir's own offload loop runs on RTL (validated)
Built gwaihir's sim on Questa (this IIS box) and ran the bundled
`sw/cheshire/tests/simple_offload.c` end-to-end: `[JTAG] SUCCESS`, `$finish`,
**Errors: 0**. (Caveat: reproducing the run needs the FULL live `iis-env.sh`
setup — a fresh shell sourcing only the captured `env.sh` hit Questa vopt/DPI
errors, an env-setup gap, not a logic failure. Re-run via `source iis-env.sh`
in the gwaihir tree.) The host (CVA6) wrote each cluster's entrypoint->`scratch[1]` +
return-code-addr->`scratch[0]`, rang `cluster[0].cl_clint_set`, and polled the
L2-SPM `return_code_array` until done — the exact host->doorbell->cluster->
completion loop QCS maps onto, confirmed on the *real* target.

Build facts (gwaihir):
- `source iis-env.sh` (bender checkout + `uv sync` + venv); exports
  `VSIM_SEPP=questa-2023.4`, `CHS_SW_GCC_BINROOT=/usr/pack/riscv-1.0-kgf/riscv64-gcc-12.2.0/bin`,
  `SN_LLVM_BINROOT=/usr/scratch2/vulcano/colluca/tools/riscv32-snitch-llvm-...-0.5.0/bin`.
- Flow: `make all` (RTL gen: floogen + clustergen + peakrdl) -> `make vsim-compile`
  (~90s, 0 err) -> `make sw` -> `make vsim-run-batch CHS_BINARY=... SN_BINARY=...
  PRELMODE=3`. TB top `tb_gwaihir_top`. PD repo (`init-pd`) NOT needed (rdl stubs).
- **Caveats:** `cfg/mini_gwaihir_noc.yml` is stale/broken (missing `NumUcie` etc.) —
  use the **default 16-cluster** config (compiles fast, small apps run ~2 min); a
  real reduced config needs an internally-consistent NoC YAML (NoC design work).
  Snitch *apps* (gemm/axpy) don't build out-of-box (`SN_BUILD_APPS=ON`); the
  working offload app is `sw/snitch/tests/build/simple.elf`. One PD-dependent
  cheshire test excluded.

### QCS -> gwaihir address map (default cfg, from `.generated/gw_addrmap.svh`)
- **L2 SPM**: base `0x7000_0000`, `0x10_0000`/tile x2 (`0x20_0000` total). The
  shared region (command stream + buffers + job descriptor). simple_offload puts
  the completion array at `L2_SPM_BASE + TOTAL_SIZE - 0x1000`.
- **Cluster i**: base `0x2000_0000 + i*0x4_0000`; peripheral_reg base
  `0x2002_1000 + i*0x4_0000` -> `scratch[4]`, then **`cl_clint_set`** (doorbell),
  `cl_clint_clear`.
- Cheshire CLINT `0x204_0000`; cluster cfg regs `0x6000_0000`; L2-SPM cfg
  `0x6001_0000`. ELF entrypoints: host `0x1000_0000`, snitch `0x7000_0000`.
- QCS doorbell -> `cl_clint_set`; entry -> `scratch[1]`; completion/status ->
  polled L2-SPM word; binding device_ptrs -> L2-SPM/DRAM addresses (cluster wide
  iDMA reachable). Mirror `simple_offload.c` for the host submit.
- Reusable build at `/scratch/dankeller/snitch-compiler/gwaihir-phase2/gwaihir`
  (compiled work lib, simple_offload.spm.elf, simple.elf, env.sh).

### End-to-end target: gwaihir `summa_gemm`, with the kernel from iree+xdsl

The concrete Phase-2 success workload is gwaihir's
`sw/snitch/apps/summa_gemm/src/summa_gemm.c` — a host-driven **distributed (SUMMA)
GEMM** across the Snitch-cluster mesh — **but with the per-tile local-GEMM kernel
produced by Quidditch's iree+xdsl chain instead of the hand-written one.**

What `summa_gemm` is (device-side): SUMMA tiling across clusters (row/col tile
distribution, B-tile multicast HW/SW, `snrt_dma_load_2d_tile`/`store_2d_tile`,
`snrt_cluster_hw_barrier`/`snrt_global_barrier`, `gw_create_mesh_comm`). The
per-tile compute is `sc_st_gemm(largs->gemm_fp, &sc_st_args)` — `gemm_fp` is the
actual single-cluster matmul.

**The iree+xdsl insertion point = `gemm_fp` / `sc_st_gemm`'s inner matmul.** Keep
SUMMA's orchestration + DMA + the offload/doorbell path; swap the hand-written
Snitch GEMM for an iree+xdsl-compiled one (the existing Quidditch gemm kernel,
emitted as a Snitch kernel lib + driven by the vm-bytecode host module). The host
side records the offload as a QCS stream (-> `cl_clint_set` + L2 SPM); the cluster
replays it and runs the iree+xdsl kernel per tile; result verified vs a host
reference.

Phasing within Phase 2:
- **2a (single cluster)**: host IREE VM -> QCS -> ONE snitch_cluster runs the
  iree+xdsl GEMM kernel on one tile -> verify. Simplest end-to-end on gwaihir.
- **2b (SUMMA)**: scale to the full multi-cluster summa_gemm topology (tile
  distribution + B multicast) with the iree+xdsl kernel as `gemm_fp`.

Note: snitch *apps* (incl. summa_gemm) didn't build out-of-box in the smoke test;
getting summa_gemm's device-side to build is a prerequisite for reusing its SUMMA
orchestration (or we lift the orchestration and supply our own kernel + offload).

### Integration progress (two prerequisites verified)

**(A) summa_gemm builds.** `make sn-apps` (NOT `make sw SN_BUILD_APPS=ON` — that
hits an upstream APP-name collision: gwaihir's sw.mk already sets SN_APPS, and
SN_BUILD_APPS=ON re-adds same-named upstream kernels -> `memory.ld region 'L3'
already defined`). Builds `sw/snitch/apps/summa_gemm/build/summa_gemm.elf` (232KB,
links clean) + gemm/axpy/... Env: `SN_LLVM_BINROOT` (the colluca riscv32-snitch
LLVM) + `CHS_SW_GCC_BINROOT` + `source .venv/bin/activate` (datagen). Config:
`sw/snitch/apps/summa_gemm/data/params.json` (default m=32 n=32 k=8, 4x4 tiles,
f64, `gemm_fp="gemm_fp64_opt"`); datagen emits `data.h`.
- **Swap target**: `sc_st_gemm(largs->gemm_fp, ...)` (summa_gemm.c:437) ->
  `gemm_fp` (a `gemm_fp_t` field of `gemm_args_t`, set in data.h). `gemm_fp_t` =
  `kernel(setup_ssr, partition_banks, transa, transb, M,N,K, A,lda, B,ldb, beta,
  C,ldc)` (snitch BLAS sig; gemm_fp64.h:88 `gemm_fp64_opt`, FREP/SSR).
- Host launch: `sw/cheshire/tests/simple_offload.c` writes entry->`scratch[1]`,
  retcode-arr->`scratch[0]`, kicks `cluster[0].cl_clint_set`, polls L2-SPM
  retcodes. Wiring it to a specific device elf is the integration step.

**(B) iree+xdsl codegen split works (no compiler rebuild).** The kernel is ALREADY
separable: `--iree-quidditch-static-library-output-path` writes the Snitch kernel
as a standalone rv32 `.o`; the HAL executable embeds only its NAME; `--output-format`
independently picks the host VM container. So `--output-format=vm-bytecode` yields
`gemm_square.vmfb` (host module) + `gemm_square_kernel.o` (genuine xDSL-lowered,
8160B). **Use `.venv/bin/xdsl-opt`** (has the riscv-asm target); the sibling
`venv/bin/xdsl-opt` is broken (no targets) and silently falls to LLVM. SPLIT=ON =
minimal `quidditch_module.cmake` edits (vm-bytecode output + drop the EmitC wrapper
+ expose the kernel `.o`); QuidditchTarget.cpp needs no change.
- venv caveat: the `venv` (not `.venv`) had `typing_extensions 4.12.2` (xdsl
  submodule needs >=4.13 for TypeForm) — bumped to 4.15 (isolated to `venv`;
  `.venv`/autotune untouched).

**KEY integration finding — ABI bridge.** summa_gemm's `gemm_fp` (snitch BLAS sig)
!= the iree+xdsl kernel's ABI (IREE executable-library dispatch:
env/dispatch_state/workgroup_state, our slice-2b/3 path). So the integration is
NOT a drop-in `gemm_fp` symbol swap — instead the **QCS replayer invokes the
iree+xdsl kernel via the IREE ABI** (as the Phase-1 replayer does), reusing
summa_gemm's SUMMA tiling/DMA/offload *concept* but our dispatch call site. (Or a
thin `gemm_fp_t` shim marshalling snitch-gemm args into an IREE dispatch_state.)

## Tight Quidditch <-> gwaihir integration (intent)

The goal is the **full Quidditch framework deployed on gwaihir as a first-class
target**, not a standalone bolt-on:
- **Host (Cheshire/CVA6)**: the IREE VM running the `vm-bytecode` host module (the
  split's host half) + Quidditch's cluster HAL driver (the QCS recorder),
  freestanding rv64 — built as a cheshire SW app in gwaihir's `chs-sw-tests` flow.
- **Device (snitch_cluster tile)**: Quidditch's existing cluster runtime
  (`runtime/runtime/src/Quidditch/dispatch/dispatch.c` + `executable.c`) + the QCS
  replayer + the iree+xdsl kernel libs, on real `snRuntime` — built as a gwaihir
  snitch app (`make sn-apps`).
- **Build/offload wiring**: pin gwaihir's `snitch_cluster` to a rev carrying
  Quidditch's device deltas (or land them upstream); drive via gwaihir's offload
  (`cl_clint_set` + `scratch` + L2-SPM), reusing `summa_gemm`'s SUMMA orchestration
  with Quidditch-compiled kernels as `gemm_fp`'s replacement (via the IREE ABI).
- **Codegen**: iree+xdsl emits both halves (SPLIT — done). So one Quidditch
  compile produces the gwaihir host module + Snitch kernel lib.
Net: gwaihir's `make`/sim flow builds + runs a Quidditch-compiled model end-to-end,
host-on-CVA6 driving the Snitch cluster.

## Programming the Spatz tiles (heterogeneous gwaihir) — design seam

gwaihir clusters are **snitch_cluster + spatz** (Snitch scalar cores vs a RVV
vector cluster). To make BOTH Quidditch-programmable (not just snitch_cluster):
- **Codegen**: add a second lowering target alongside the snitch FREP/SSR path —
  a **spatz/RVV** path (linalg -> RVV via xDSL's vector dialects, or IREE LLVMCPU
  with `--iree-llvmcpu-target-features=+v`), emitting a spatz kernel lib. Select
  per executable target (`"snitch"` vs a new `"spatz"`).
- **Device runtime**: a spatz variant of the cluster replayer (spatz `snRuntime` +
  RVV intrinsics) that invokes the spatz kernel via the same IREE executable-library
  dispatch ABI.
- **ABI/offload**: QCS is tile-type-agnostic (it carries executable_id/ordinal +
  bindings); route the doorbell to the **target tile's** `cl_clint_set` and load the
  matching kernel lib (snitch FREP kernel vs spatz RVV kernel). The host picks the
  tile by the compiled kernel's target.
- **Phasing**: snitch_cluster end-to-end first (current campaign); **spatz is the
  Phase-3 track** (needs the RVV codegen path + spatz runtime). Design the
  executable-target + tile-routing seam now so spatz drops in without a retrofit.

## Heavy-campaign results (3 workstreams landed, verified + reviewed)

1. **Device-half firmware** (`runtime/host/firmware/gwaihir/`, commit 555a00d):
   rv32 Snitch QCS replayer on real snRuntime (DM core drives; compute cores run
   the grid via hw-barrier fan-out), invoking the kernel through the IREE dispatch
   ABI. Builds in gwaihir's tree (`qcs_replay.elf`, -Werror). Reviewed: barrier
   protocol deadlock-free; fixed an untrusted-`dynamic_local_memory` OOB + an rv32
   aperture guard. Stub kernel today.
2. **SPLIT=ON codegen** (`quidditch_module.cmake`, commit 93a1ee1): one iree+xdsl
   compile emits a `.vmfb` host module + a separable xDSL Snitch kernel `.o`, no
   compiler rebuild.
3. **Bare-metal rv64 IREE VM host runner** (scratch
   `/scratch/dankeller/snitch-compiler/iree-rv64-host/`): a freestanding rv64 ELF
   links the IREE VM + bytecode loader + platform glue + the embedded
   `gemm_square.vmfb` and **runs under spike** — VM instance created, the host
   module loaded, context-create correctly `NOT_FOUND` on unresolved HAL imports.
   Recipe worth keeping: HTIF syscall-proxy stubs (`_write`/`_sbrk`/`_exit`), a
   medany crt0 + linker script (DRAM @ 0x8000_0000, `.htif`/tohost), and
   **medany libc/libm/libgcc stubs** because the newlib/libgcc `lp64d` multilib is
   medlow and its absolute `lui` relocs overflow at DRAM (R_RISCV_HI20 truncation).

### The convergence (join point) — cluster HAL on rv64
The host VM result pinpoints the seam: `gemm_square.vmfb` imports `hal.*`
(`hal.executable.create`, `hal.device.queue.execute`, command-buffer/buffer/fence).
So the host VM needs an IREE **HAL module** registered — and that HAL is exactly
Quidditch's **slice-2 cluster HAL** (`runtime/host/hal/cluster/`: cluster_device +
cluster_command_buffer + cluster_allocator) ported to rv64 freestanding: it turns
`hal.executable.create`/`queue.execute` into a **QCS stream in L2 SPM + a
`cl_clint_set` doorbell**, which the committed device-half firmware replays. Wire
the SPLIT kernel `.o` into the firmware's `gw_register_kernels`, point the host
VM's HAL at the gwaihir L2-SPM region + doorbell, and that closes the loop:
IREE VM (CVA6) -> QCS -> snitch_cluster -> iree+xdsl kernel -> completion. Then a
Cheshire host test runs `gemm_square` on the gwaihir sim, verified vs reference
(2a) -> full SUMMA (2b).

### Join PROVEN (host half end-to-end) + the verifier finding
The cluster HAL was registered as the VM's HAL module and the full host path RAN
(proven on x86 with the identical cluster-HAL stack): `hal.*` imports RESOLVED ->
context created -> `gemm_square.gemm64` invoked -> the HAL **recorded a well-formed
QCS stream** (1 DISPATCH, exec=0 ord=0, 3 bindings x 2048 B = 16x16 f64 A/B/C at
consecutive device-PAs). VA<->PA translation + binding-table resolution confirmed.
Registration flow: instance -> `iree_hal_module_register_all_types` ->
`qcs_shared_region_create` -> `iree_hal_cluster_device_create(id="quidditch_device")`
(id MUST match the vmfb's `#hal.device.target`) -> device-group -> `iree_hal_module_create`
-> bytecode module -> `context_create_with_modules([hal, bytecode])` -> resolve
`gemm_square.gemm64` -> alloc inputs via the cluster allocator -> `iree_vm_invoke` ->
`queue_execute` emits the QCS job. Real HAL fixes this surfaced (the slice-2 HAL was
never run against a full bytecode module): executable cache + stub executable in
cluster_device, `query_i64` answering `hal.executable.format`, and a DEFERRED
command buffer resolving direct/indirect binding tables in cluster_command_buffer
(being ported back into the repo + verified).

**Verifier finding (gates RTL):** under spike, rv64 builds + runs but
`context_create` does not finish in 595 s — time is in IREE's VM bytecode/flatbuffer
verifier (`iree_vm_bytecode_function_verify_bytecode_op` + flatcc `verify_table`),
pure emulation slowness, not a bug (x86 passes instantly). On cycle-accurate RTL
this is worse, so the RTL deployment must **disable VM bytecode verification** (an
IREE module-load knob) or pre-verify offline. Flag for the 2a co-sim.

### Remaining to 2a
1. DONE (dca21cc) — cluster-HAL fixes landed + x86-verified.
2. DONE (0415511) — the real iree+xdsl gemm_square kernel links into the device
   firmware (rv32, zero undef). ABI MATCH (ELF32 RISC-V ilp32d == gwaihir snitch
   LLVM). Registered via the library-query path (the dispatch entry is a LOCAL
   symbol; only `quidditch_..._library_query` is global) as harness.c:100 does;
   the kernel's static-inline snRuntime refs resolved via the quidditch_snrt_exports
   shim. lib/ = generated SPLIT output (gitignored).
3. DONE + PROVEN ON RV64 — VM bytecode verification disable: `runtime/host/patches/
   iree-verify-off.patch` guards the flatcc `BytecodeModuleDef_verify_as_root`
   (upstream gates only the per-op verifier, not this) + build the rv64 IREE libs
   with `-DIREE_VM_BYTECODE_VERIFICATION_ENABLE=0`. With it, the rv64 host VM (cluster
   HAL registered) runs the FULL path under spike — context-create -> invoke
   gemm_square.gemm64 -> QCS recorded (1 DISPATCH, 3 bindings, == x86) in **~11s**
   (was >595s, all in the flatcc verify). The host half is now proven on rv64, not
   just x86. (Repo iree submodule left clean; patch applied at build time.)
4. DONE (2b5964c) — dma-core fan-out: the firmware runs BOTH halves per dispatch
   (compute_core_ptrs on compute cores over the grid + dma_core_ptrs once on the DM
   core), mirroring dispatch.c. Reviewed deadlock-free; rebuilds rv32.
5. TODO — point freestanding `shared_region.c` at gwaihir HW: `region->base` =
   cluster L2-SPM aperture; `qcs_doorbell_ring` -> cluster `cl_clint_set` (msip) MMIO;
   drop the auto-complete; `qcs_doorbell_wait_completion` polls the real completion/
   status the DM core writes back.
5b. DONE (AOT host validated) — instead of the bytecode interpreter, emit the HOST
   as IREE **vm-c (EmitC = native compiled C)**: `--output-format=vm-c` +
   `--iree-quidditch-static-library-output-path` gives an EmitC host module
   (`<name>_create` -> `iree_vm_native_module_create`, a static C function table) +
   the SAME separable xDSL kernel `.o`. Built freestanding rv64 + the cluster HAL,
   it runs under spike: native module created, **context created INSTANTLY (no
   flatcc/parse/verify/interpreter)**, export resolved. The throughput killer is
   gone. (The rv64-spike invoke stalls on a spike-only HTIF console `_write` bug in
   syscalls.c — orthogonal to EmitC, won't apply on Cheshire HTIF; the x86 harness
   proves the recording: 1 DISPATCH, 3 bindings.) **RTL-ready host = EmitC vm-c host
   + separable kernel.o.** Deploy-config TODO: add a quidditch_module SPLIT variant
   emitting the vm-c host lib (linking iree::vm) + the kernel.o (the non-SPLIT vm-c
   branch already does both in one compile).
6. IN PROGRESS — the Questa co-sim, EmitC native host. INTEGRATED end-to-end; one
   precise RTL blocker left.
   - Stage 1 DONE: EmitC IREE host builds as a Cheshire SW binary
     (`quidditch_gemm.l2host.elf`, 900KB) against Cheshire crt0/UART/newlib (spike
     HTIF glue dropped; medany libm/libc stubs kept for reloc-safety;
     `--allow-multiple-definition` for memcpy/memset). Fills A,B; invokes
     gemm_square.gemm64; reads C; verifies C==A@B; reports via SCRATCH[2].
   - Stage 2 DONE: cluster HAL HW-wired. region.base=0x70000000 (L2 SPM, PA=offset).
     **Caught + fixed a collision**: descriptor at PA0 overlapped the firmware .text
     (0x70000000-0x7000D6F8) -> moved descriptor to offset 0x10000 (host + firmware,
     firmware rebuilt). Doorbell writes scratch[1]=entry, scratch[0]=desc PA, then
     cl_clint_set=0x1FF @0x200211A0 (disasm-verified, == simple_offload); real
     completion poll.
   - Stage 3 BLOCKER: design elaborates, both ELFs preload, the CVA6 RESUMES + fetches
     its entry word — but reads X at entry+4. Root cause: **backdoor-loaded L2 SPM is
     not coherent with the CVA6 LLC/dcache fetch path** (the Snitch reads L2 SPM
     directly, so the firmware is fine; simple_offload worked only because its host ran
     from the 64KiB Cheshire SPM @0x10000000 — too small for the 121KiB IREE host text).
     Host-placement walls: DRAM@0x80000000 = JTAG bus error; L2 SPM = CVA6 fetch-X via
     LLC; Cheshire SPM = too small. FIX direction: configure/flush the LLC so the CVA6
     sees backdoor-written L2 SPM coherently (bootrom LLC-as-SPM config or an axi_llc
     flush before resume), or a coherent load path. Data IS in SRAM (backdoor write
     succeeded; first fetch word resolved) — only the cached read path returns X.
   - Verify-ready: golden C[0,0]=43792, C[255]=1092352 for A(a[i]=i+1)/B(b[i]=2(i+1));
     jtag_wait_for_eoc reads SCRATCH[2]; l2mem.bin dumps C at PA 0x12000 for cross-check.
   - Artifacts: /scratch/.../iree-rv64-host/cheshire/ + the rebuilt qcs_replay.elf +
     a tb_gwaihir_top.sv backdoor-preload edit (scratch only).

## Status

Phase 1 complete (dev-box split end-to-end). Phase 2 SoC = **gwaihir**, validated
end-to-end on RTL (its offload loop runs on Questa here). All three feasibility
GOs in hand: RTL loop runs (gwaihir), bare-metal rv64 IREE VM builds, toolchains
work. Remaining is integration: bare-metal IREE VM on the Cheshire host (records a
QCS stream) + QCS mapped onto `cl_clint_set`/`scratch`/L2-SPM + the snitch_cluster
device-half (QCS replayer on real snRuntime) + the iree+xdsl codegen split
(vm-bytecode host + kernel libs) -> the `summa_gemm` workload with an iree+xdsl
kernel (single-cluster first, then full SUMMA) on the gwaihir sim vs a host
reference (the goal's success condition).
