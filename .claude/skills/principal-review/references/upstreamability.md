# Upstreamability — we work on forks

Every repo here is a **fork**, and the goal is that our changes stay **mergeable
to the upstream they came from**. So every change carries a question a principal
asks out loud: *will this be upstreamed? If not, why not — and is that
justifiable?* Unjustified divergence is a real cost: it rots the fork, makes
every rebase onto upstream painful, and is usually the fingerprint of a hack.

## The fork map (know the upstream)

| Repo | Our fork | Upstream | Notes |
|---|---|---|---|
| Quidditch | `DanielKellerM/Quidditch` | `opencompl/Quidditch` | the compiler; base `30120ba` |
| iree | `DanielKellerM/iree` | `iree-org/iree` (v3.11.0) | pinned fork; ABI headers live here |
| xdsl | branch `quidditch-port` | `xdslproject/xdsl` (v0.66.0) | Snitch lowering passes |
| snitch_cluster | `DanielKellerM/snitch_cluster` | `pulp-platform/snitch_cluster` | Snitch runtime + RTL |

A future **Nimbus** split adds a fourth axis: deployment code that belongs in
Nimbus, not upstream anywhere.

## The three verdicts (classify every change)

1. **UPSTREAMABLE** — a general fix or improvement upstream would accept (a
   correctness fix, a portability fix, a cfg-driven generalization). *Action:*
   write it to **upstream standards** (their naming, their tests, general not
   target-specific), keep the commit atomic and labeled, and PR it — or record a
   tracked TODO to. Letting an upstreamable fix sit fork-only is itself a finding:
   it will collide on the next rebase for no reason.
2. **FORK-ONLY, JUSTIFIED** — genuinely target/deployment-specific: the gwaihir
   addrmap, the host-device-split research prototype, a Snitch-ISA-only lowering.
   Upstream wouldn't scope it. *Action:* the justification must be **explicit**
   (a one-line comment / commit / doc saying *why* it can't be general), and the
   change should be **isolated and additive** (gated behind cfg/flags, not woven
   into an upstream code path) so it rebases cleanly.
3. **FORK-ONLY, UNJUSTIFIED** — a hack/workaround/quick-and-dirty that *could and
   should* be general but wasn't. *This is a BLOCKER-class finding for fork
   health.* Either generalize it and make it upstreamable, or produce a real
   justification. "It was faster to hardcode" is not one.

## How to keep changes mergeable (write for verdict 1 or 2)

- **Additive over invasive.** Prefer a new file / a new opt-in path / a cfg flag
  to editing an upstream hot path. The `QUIDDITCH_L1_BASE` cfg-threading is the
  model: upstream behavior is the default, the fork value is an input.
- **Upstream standards even in the fork.** Match the upstream's idioms so the
  diff is PR-ready, not a reformat war (this is where the ecosystem references
  apply — a fork change still reads like upstream wrote it).
- **Isolate + label.** Keep fork-only changes in identifiable commits with a
  message that states the verdict ("fork-only: gwaihir addrmap, not upstreamable
  because …"). Atomic commits are cherry-pickable upstream later.
- **Minimize the surface.** A 3-line fix that touches one upstream function
  rebases; a 200-line reflow of an upstream file does not.

## Worked examples (from this repo's actual fork changes)

- `snitch_cluster` `cls.c` (`used,retain` on `_cls`) and `sync.h`
  (`SNRT_CLUSTER_NUM==1` barrier guard): **UPSTREAMABLE** — general runtime
  correctness, nothing gwaihir-specific. Should be PR'd to `pulp-platform`; today
  they sit fork-only with no tracked PR → a finding.
- `xdsl` `fp64_pipeline_depth` threading + SSR/FREP cfg gating: **UPSTREAMABLE**
  (general, cfg-driven). Propose upstream; keep the fast-path byte-identical so
  the diff is minimal.
- `iree` `executable_library.h` fork (`compute_core_ptrs`/`dma_core_ptrs`):
  **FORK-ONLY** — dual-core-type (compute + DM) dispatch is Snitch-specific. Is it
  *justified*? Forking an IREE ABI header is high-divergence. Prefer minimizing:
  keep the fork struct in a **Quidditch-owned** header (not a modified IREE
  header), or propose an upstream extension point. Whichever — document the
  verdict and pin the layout (see `abi-and-versioning.md`).
- Quidditch codegen cfg-target threading (`QuidditchTarget.cpp`): **UPSTREAMABLE**
  to `opencompl/Quidditch`. The host/firmware/QCS split: **fork-only, Nimbus-bound.**

## The review question (what the `upstream` lens asks per change)

For each changed file/hunk:
1. Which fork does it touch, and what is the upstream?
2. Verdict: upstreamable / fork-only-justified / unjustified-divergence?
3. If fork-only: is the justification **documented**, and is it **real**?
4. Is it written to **upstream standards** so it *could* merge?
5. Does it **minimize divergence** (additive, isolated, small surface)?

Flag: an upstreamable fix with no PR/tracking (TASTE), unjustified divergence
(BLOCKER), a fork-only change with no documented rationale (TASTE→BLOCKER), and an
invasive rewrite of an upstream file where an additive change would do (BLOCKER).
