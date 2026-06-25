<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 ETH Zurich and University of Bologna. -->

# RTL-sim-driven xDSL kernel autotuner

**Status:** design, adversarially reviewed, **not built**. Tracked as task #20.
This plan was produced and stress-tested by a multi-agent workflow; three
independent reviews caught four blocking defects in the first draft, all
re-verified against the repo and folded in below. Read §0 first — it is the
single most important section.

**Goal (user intent):** leverage the *methodology/recipe* of
[AutoKernel](https://github.com/rightnow-ai/autokernel) — an LLM-agent GPU
kernel autotuner — **not** the tool itself (it is Triton/CUDA/SIMT-specific and
has no Snitch backend). Reuse only the loop design: *propose a candidate →
build → measure on the cycle-accurate oracle → keep if faster AND correct, else
revert; gate correctness before perf; pick the next target by impact*. Apply it
to improving Snitch kernels via **xDSL**, scored by **RTL simulation**, then
verify the win **translates through the full IREE toolchain**.

---

## 0. Read this first: two go/no-go gates before building anything

The first draft of this plan would have **failed at its own first step**. Two
cheap experiments decide whether the entire autotuner is worth building:

- **Gate A — throughput.** A full-IREE Verilator run is **~75 s, not 1–2 s**
  (~3,400 cyc/s), because **98.4 % of every sim is IREE bring-up and only ~1.6 %
  (~4,194 cyc) is the kernel**. Tuning through the full `gemm_square` flow =
  ~47 experiments/hour = unusable for a search. → The **standalone harness**
  (drive the generated dispatch directly between the two `read_csr(mcycle)`
  markers, bypassing `iree_vm_invoke`) is the **precondition**, getting to
  ~3 s/experiment. The Track-A harness pattern already exists as
  `runtime/samples/gemm_a1` **on the `codegen-cse-improvement` branch** (not the
  current working tree — bring it over / build the equivalent for the tuned
  kernel in Phase 0b).
- **Gate B — is there anything to tune?** The kernel is already **~0.87
  FPU-utilization** and the legal Group-A tiling grid is only **~6–8 points**.
  Enumerate it; report the speedup distribution vs. baseline. **If the max
  speedup is < ~10 %, this kernel is already near-optimal — STOP and pivot to a
  kernel with a real tiling space before building any driver.** This is the
  true highest-risk assumption and it is currently untested.

Everything downstream (pinned ROI, numpy reference, search driver) is contingent
on both gates passing.

---

## 1. Architecture: a two-level loop

```
L1  FAST TUNING:   inject knob → build → sim → score(correct, cycles, fpu_util) → keep/revert   (search burns here)
L2  VALIDATION:    apply L1 winner in the full IREE flow → sim → roi_minus_l1 + SUCCESS gate     (confirms a handful)
```

- **L1** runs on the **standalone harness** (not the 75 s full-IREE run) so the
  search is affordable.
- **L2** confirms an L1 winner survives the full `iree-compile` → Quidditch
  codegen → cluster path. The `roi_minus_l1` "reproduction-gap" metric (how much
  the isolated win shrinks end-to-end) is the strongest single piece of the
  design and stays.

---

## 2. The candidate / search space

Both knob groups are tunable **without recompiling the IREE/Quidditch C++**:

- **Group A — per-kernel tiling** (`runtime/samples/gemm_square/gemm_square.mlir`,
  the inline `lowering_config` on the `linalg` op; baseline
  `l1_tiles=[8,8,8]`, `l1_tiles_interchange=[2,0,1]`, `dual_buffer=true`):
  `l1_tiles=[M,N,K]`, the interchange permutation, `dual_buffer`. `gemm64` is
  **not** in the `codegen/.../ConfigureForSnitch.cpp` name table, so it relies on
  the inline config; editing the `.mlir` + rebuild re-fires the injected values
  (verified). ~6–8 legal points (divisors of 16, N a multiple of 8, footprint
  under the TCDM budget).
- **Group B — global lowering** (`xdsl/xdsl/transforms/test_lower_linalg_to_snitch.py`):
  CSE placement (the 0.42→0.87 win this would automate), `streams` ≤ 3,
  `unroll_factor` / `iterator_index` (paired), `target_rank`, `frep` on/off,
  register allocation. Pass **ordering** is itself a degree of freedom.

**The whole combined space, after legality pruning, is low hundreds of points
and fully enumerable**, with a deterministic noise-free oracle. This is why the
driver is grid → Bayesian search, **not** an LLM agent (§4).

**Legality:** do **not** use the static "tile footprint < TCDM budget"
arithmetic to gate KEEP — `dual_buffer` halves the real budget and bufferization
adds padding the static formula can't see. Use it only to *rank*; the
authoritative overflow check is the compile-time
`--iree-quidditch-assert-compiled` error (free, build-time — promote it to an L1
pre-filter).

---

## 3. The oracle

### 3a. Throughput reality (measured)

| Quantity | First-draft assumption | Measured |
|---|---|---|
| Verilator wall / run (full IREE flow) | 1–2 s | **~75 s** |
| Sim throughput | ~10k cyc/s | **~3,400 cyc/s** |
| Compute ROI / total cycles | — | **4,194 / 257,793 ≈ 1.6 %** |
| Incremental build (`.mlir` touch) | — | **~2 s** |
| Tracer-off saving | "a speed lever" | **~2 % — not a lever; drop it** |

→ Full-IREE oracle = ~47 exp/hr (heat-death). **MEASURED (Phase 1, gemm_harness):
~1.1 s build (iree-compile codegen ~0.88 s + ninja) + ~16.5 s sim ≈ ~18 s/config.
The harness drops the IREE VM but the run is still ~58k cyc (≈55k snRuntime boot +
~3k dispatch) at ~3.5k cyc/s — so the per-config oracle is SIM-BOUND (~94 %), not
build-bound (the earlier "~1.2 s sim" projection here was wrong).** The real
throughput lever is therefore **parallel-sim fan-out**, NOT faster builds: builds
serialize (shared build-rt) but the sims fan out across the 96-core box.
VALIDATED: tools/autotune/sweep.py runs the 9-config grid in ~49 s (sims 16-way
parallel ~17 s wall), reproducing the serial spread exactly. ccache / tracer-off /
tiny-DIM / bypass-cmake are NOT levers (each <2 %). Build-side fixes
(quidditch_module.cmake:105 perpetual-dirty `_llvm.h`, in-process libIREECompiler)
are for determinism/cleanliness, not speed.

### 3b. Pinned ROI contract (the score must be unambiguous)

`read_csr(mcycle)` in `runtime/runtime/src/Quidditch/dispatch/dispatch.c` sits
**inside** the worker `while` loop, so it fires once **per workgroup dispatch
per hart** — `--dump-hart-perf` emits a *list* of segments, not one scalar.
Before any loop runs, pin:

1. **Hart selection** — aggregate over **compute harts only**. `hart_00000` is
   the **DM core**, not a compute core; the first draft read it (wrong).
2. **Segment selector** — ROI = sum of the worker-loop mcycle segments per
   compute hart (segment count varies with the tiled config), aggregated across
   compute harts by a frozen rule (sum, or max as a critical-path proxy — pick
   one and freeze it).
3. **Config-stability guard** — log the raw segment **list + count** per run; a
   change in segment count between L1 and L2 is a *flagged divergence*, not
   silently folded into `roi_minus_l1`.
4. **Identical selector in L1 and L2** — else `roi_minus_l1` measures an
   indexing artifact, not the staging gap.
5. `SimResults.get_metric(region, metric)` needs a `region` arg — define it as
   the compute-hart worker-loop region.

This is a hard prerequisite, not a refinement.

### 3c. Correctness gate — a real reference (the first draft's was vacuous)

`main.c`'s self-check is **not** a usable oracle: it tests one degenerate input
(`A[i,k]=k+1`, `B[j,k]=1`, so every output is 136) compared via `(int)C[idx]`.
The all-ones `B` makes `B == Bᵀ` (hiding orientation bugs) and the `(int)`
truncation is non-monotone (135.5→135 FAIL but 136.4→136 PASS). Replace from v0:

- Compare against an independent **per-element f64 numpy reference** computed
  from the **actual** A, B, with explicit rel+abs tolerance, over ≥ 1 randomized
  input — needs only a 20-line check on the dumped `C`, no harness.
- The kernel is **transpose-b**: `C[i,j] = Σ_k A[i,k]·B[j,k]`, i.e. **`A @ B.T`**
  (the explicit `indexing_maps` encode `Bᵀ`; it was rewritten from
  `linalg.matmul_transpose_b`). The reference must be `A @ B.T`, and v0 must
  include **one asymmetric `B`** so the transpose orientation is pinned.
- The "determinism" stage of re-running the *same* ELF on a deterministic sim is
  vacuous (bit-identical by construction). To actually catch uninit-TCDM / DMA
  races, vary the **memory init fill pattern** (0x00 / 0xFF / random) with
  identical inputs and require identical output — and run it **once per kept
  winner**, not per candidate.

### 3d. Free pre-sim filters (ahead of the 1.2 s sim)

Reject/deprioritize before building+simming: (a)
`--iree-quidditch-assert-compiled` build error = reject (≈2 s, no sim); (b) a
static roofline / FPU-util estimate to rank candidates; (c) N-multiple-of-8 /
footprint heuristics to *rank* (not gate). Instruction-count via the tracer is a
coarse pre-filter only — it cannot see stalls / SSR / DMA-overlap (the whole
0.42→0.87 story), so it never replaces the scorer.

### 3e. Build/sim invocation correctness (these would crash step 1)

- **Never call `ninja` bare** — `/usr/sepp/bin/ninja` is 1.6.0 (needs ≥ 1.8);
  hardcode `$ROOT/.venv/bin/ninja` (1.13).
- **Never run `snitch_cluster.vlt <elf> <name>`** — the wrapper forwards `<name>`
  as a Verilator **plusarg**, not a run name; the literal command errors out
  ("could not open Usage:"). Go through a `run_on_sim_vlt.sh` (the `--vlt`
  variant of `scripts/run_on_sim.sh`, preserving its `--trace` semantics), built
  **before** the throughput gate runs.
- **Confirm exit-code propagation** through the `.vlt` wrapper so the ELF's
  `return errors?1:0` surfaces as the sim exit; if not, the numpy reference (§3c)
  is the only live correctness signal.

---

## 4. The driver: grid → Bayesian, LLM agent demoted

Because the space is enumerable and the oracle is deterministic, an
LLM-agent-in-the-loop is over-engineered (nondeterminism, prompt/harness-freeze
fragility, token cost). Progression:

1. **v0 — exhaustive legal grid (no LLM).** Enumerate Group A (~6–8 points);
   report the **speedup distribution** vs. baseline. Doubles as Gate B (§0).
2. **v1 — random / Latin-hypercube → Bayesian** (e.g. a ~20-line scikit-optimize
   loop) once Group B is added and the combined space exceeds cheap exhaustion.
   The correct tool for a deterministic low-hundreds-point space.
3. **Optional follow-on — LLM agent**, gated on a *demonstrated* finding that
   grid+Bayesian plateaus below the roofline. Its only defensible role is
   cross-knob-group hypothesis generation. Keep AutoKernel's `program.md` /
   frozen-harness invariant / commit-per-experiment scaffolding *for this*, not
   the core plan. Drop the top tiers (iDMA descriptor chaining, GEMM packing) —
   they aren't injectable without a C++ rebuild.

**Invariants kept from AutoKernel regardless of search strategy:** git
commit-per-experiment; KEEP iff `correct AND cycles < best` else
`git checkout -- <edited files>`; per-experiment log files (do **not** `tee`);
`best.json` tracking; **config-hash dedup** (normalize equivalent interchange
perms before the expensive build+sim); `results.tsv` carries a `kernel` column
from day one. Single-kernel MOVE_ON criteria first (consecutive-reverts,
speedup-threshold, FPU-util ≥ ~0.90, experiment-count cap); the cross-kernel
Amdahl selection (`orchestrate.py`) is dead until multiple kernels exist → moves
to Phase 4.

---

## 5. IREE translation: applying & verifying an L1 win

- **Apply** a Group-A win by leaving the edited `.mlir` in place and rebuilding
  (`ConfigureForSnitch.cpp` no-override → `TensorTile.cpp`) — no flag, no C++
  change. Group-B wins are already global Python edits.
- **Verify (L2)** = a `verify.py`-style step: full-DIM rebuild, in-flow ROI (the
  *same* pinned compute-hart selector, §3b), `roi_minus_l1`, the `main.c` SUCCESS
  gate, assert-compiled overflow detection.
- **The dominant risk is the DIM-shrink, not the isolation.** L1 at tiny DIM
  (problem fits TCDM trivially, DMA one-shot and fully hidden, `dual_buffer`
  pointless, remainder loops never fire) can produce a config that **reverses**
  at full DIM. Mitigations:
  - Run L1 at **two DIMs** (e.g. 16 and 64) so a sign-reversal is caught before
    the expensive L2.
  - L2 runs at **full DIM**; the survival criterion is **quantitative** (L2
    speedup ≥ X % of L1 speedup AND `main.c` SUCCESS) — not "~same margin".
  - Gate search feedback on the **full-DIM L2 result as ground truth**, not the
    `roi_minus_l1` proxy, so the search can't game the proxy. `assert-compiled`
    only catches the *overflow* sub-case, not the silent "DMA no longer hides"
    reversal — that is what the two-DIM L1 check + full-DIM L2 band are for.

---

## 6. Repo layout

```
tools/autotune/
  loop.py          # the driver (commit-per-experiment, keep/revert, best.json, dedup)
  oracle.py        # build+sim+score; hardcodes venv ninja + the .vlt wrapper
  inject.py        # rewrite gemm_square.mlir lowering_config / xdsl pass list
  candidates.py    # legal candidate enumeration
  search.py        # grid → Bayesian
  reference.py     # numpy A@B.T + rel/abs tolerance
  spec.py          # roofline / FPU-util pre-filter (ranking)
  results.tsv      # (+ kernel column; gitignored)
  best.json        # committed — the regression pin
  README.md
scripts/run_on_sim_vlt.sh   # Verilator wrapper preserving run_on_sim.sh --trace semantics
```

Reused, **never edited by the loop**: `run_on_sim.sh`, `gen_trace.py`,
`SimResults.py`, `quidditch_module.cmake`, the sample. The `program.md` + agent
scaffold lands only with the optional follow-on (§4).

**Regression pin:** a committed `best.json` per kernel + a CI/ninja check that
re-sims the tuned config and asserts `cycles ≤ recorded_best*(1+tol)` AND
SUCCESS. A tuned `lowering_config` is a number baked into a `.mlir` that silently
rots on the next xDSL/IREE submodule bump (this repo's whole recent history is
such bumps). Without the pin, every win bit-rots.

---

## 7. Phased roadmap (smallest-first)

- **Phase 0 — two gates in parallel; build nothing else until both pass.**
  - 0a. Build `scripts/run_on_sim_vlt.sh` (Verilator wrapper, `--trace`
    preserved) — *ahead* of the gate so it doesn't run the broken raw-`.vlt`
    command.
  - 0b. Build the standalone harness for the tuned kernel (drive the dispatch
    between the two `read_csr(mcycle)` markers, no `iree_vm_invoke`). The
    throughput precondition.
  - 0c. **Gate A (throughput)** as experiments/hour vs. the search budget. If
    even the harness is tens of seconds, sim-in-the-loop at search scale is not
    viable → fall back: a static cost model does the search, the sim only
    *validates* a handful of winners (sim = L2 confirmer, never the L1 oracle).
    The L1/L2 skeleton survives that fallback.
  - 0d. **Gate B (is there a win)** — enumerate the legal Group-A grid, report
    the speedup distribution. Max < ~10 % → STOP, pivot kernel.
- **Phase 1 — full v0 loop.** Generalize 0d into `loop.py` (commit-per-experiment,
  keep/revert, `best.json`, config-hash dedup, per-experiment logs). Single-kernel
  MOVE_ON only. Add Group-B candidates. Pinned ROI (§3b) + numpy `A@B.T`
  reference + ≥1 randomized + 1 asymmetric input (§3c) live from the first run.
  Parallel fan-out across Verilator processes (~×16–24 realistic on the shared
  box, not ×48).
- **Phase 2 — Bayesian search + full correctness sweep.** Add `search.py` once
  Group-A×Group-B exceeds exhaustion. Add DIM/dtype sweeps (non-multiples of
  unroll/SSR factor; {f64,f32,f16}; edge dims) — now cheap via the harness.
- **Phase 3 — L2 IREE translation gate.** Port `verify.py`: full-DIM rebuild,
  pinned compute-hart ROI, `roi_minus_l1`, quantitative survival band,
  assert-compiled detection, two-DIM L1 pre-check feeding L2. Feedback gated on
  full-DIM L2 ground truth.
- **Phase 4 — multi-kernel + Amdahl + regression pins.** Widen to kernels not in
  the `ConfigureForSnitch.cpp` name table. Move the cross-kernel Amdahl selection
  here (now testable). Land the committed-`best.json` + CI re-sim check.
- **Phase 5 (optional, evidence-gated) — LLM agent.** Only if Phases 1–4 show
  grid+Bayesian plateaus below roofline.

---

## 8. Risks / unknowns (honest)

- **Hardware contention** — shared 48-physical-core box; realistic parallelism is
  ×16–24, and 3 s/exp assumes free cores.
- **Harness feasibility is itself a risk** — if the kernel genuinely cannot run
  without IREE buffer staging (the very §5 reproduction gap), the harness can't
  isolate the ROI and Gate A fails → cost-model-search fallback. Phase 0b tells
  us.
- **Near-optimal kernel** — if Gate B fails (likely for a TCDM-resident 16×16
  GEMM already at 0.87), the autotuner produces a null result on *this* kernel.
  Real, currently-untested — hence the hard Phase-0 gate.
- **ROI scalar stability** — segment count varies with config; if the pinned
  selector can't be made config-stable, `roi_minus_l1` stays noisy. Validate the
  selector on ≥2 configs in Phase 0b before Phase 1.

---

## 9. First concrete step (greenlight this)

Build `scripts/run_on_sim_vlt.sh`, then run the two Phase-0 gates back to back:

```bash
ROOT=/scratch/dankeller/snitch-compiler/Quidditch
NINJA=$ROOT/.venv/bin/ninja                      # 1.13 — bare `ninja` is 1.6.0 and FAILS

# Gate A (throughput): time ONE oracle pass on the EXISTING ELF via the proper wrapper
time $ROOT/scripts/run_on_sim_vlt.sh \
     $ROOT/build-rt/samples/gemm_square/gemm_square gemm16_vlt --trace
python3 $ROOT/snitch_cluster/util/trace/gen_trace.py \
     $ROOT/runs/gemm16_vlt/logs/trace_hart_00001.dasm \
     --dump-hart-perf $ROOT/runs/gemm16_vlt/perf1.json   # hart_00001 is a COMPUTE hart, not DMA hart_00000
#  → record wall time, ROI fraction (Σ worker-loop segments / total), exp/hr at ×1 and ×16
#  → DECIDE: harness needed (almost certainly yes, given 1.6% ROI) → build it BEFORE the loop

# Gate B (is there a win): enumerate the legal Group-A grid on the harness, report speedup distribution
#  grid e.g. l1_tiles ∈ {[8,8,8],[16,8,16],[8,16,8],[16,16,16],[8,8,16]} × dual_buffer{0,1}, footprint<budget
#  inject via rewriting gemm_square.mlir lowering_config; $NINJA -C build-rt gemm_square; oracle; score
#  → if max speedup over baseline < ~10%, STOP and pivot kernel
```

These two signals decide whether the rest is worth building. This is exactly the
place the first draft was both wrong and self-gating: it would have run the
broken raw-`.vlt` command (false negative) or measured 75 s and mislabeled it a
win. The reordered step de-risks throughput correctly **and** adds the missing
"is there anything to tune" gate.
