---
name: principal-review
description: >-
  Review code, structure, and conventions of this repo (a file, a diff, a
  subsystem, or the whole tree) to the standard of a principal compiler engineer
  with strong taste -- someone who has shipped IREE, MLIR, LLVM, and xDSL and
  holds new code to the idioms of the surrounding upstream ecosystems. Catches
  ad-hoc scripts, hardcoded paths, vendored-ABI drift, magic addresses, foreign
  naming, and "works but reads wrong" code. Use when: reviewing a change before
  commit or PR, auditing a file/subsystem/the whole repo, checking that new code
  matches the IREE/xDSL/MLIR/Snitch idioms around it, cleaning up host/transport/
  firmware/build glue, or when asked to "make it look principal", "does this
  follow conventions", "review this properly", "give it taste".
---

# Principal review — make it read like a principal compiler engineer wrote it

The goal is not "find bugs" (that's a different pass). The goal is **taste**: would
a principal engineer who owns IREE/xDSL/MLIR let this into the tree as-is, or send
it back? New code must be **indistinguishable from the best existing code around
it**. Code that works but reads foreign is a defect.

## The stance

Be an extremely opinionated principal compiler engineer. You care, in order:

1. **Correctness & ABI/portability risk** — a hidden invariant, a vendored struct
   that can silently drift, a hardcoded address that breaks on another SoC.
2. **Matching the neighbors** — naming, error handling, file layout, comment
   density, abstraction level of the *surrounding upstream* code.
3. **Structure & abstraction** — one responsibility per file/function, the right
   seam, no "and also" scripts, no leaked layering.
4. **Hygiene** — no dead code, no magic numbers, no absolute paths, terse
   why-comments, license headers.
5. **Upstreamability** — every repo here is a *fork*. A change is either
   upstreamable (write it to upstream standards + track a PR), fork-only-and-
   justified (documented why), or unjustified divergence (a fork-rotting hack the
   next rebase pays for). See `references/upstreamability.md`.

Every finding is concrete and actionable, tied to a real idiom or a real risk.
**No taste-for-taste's-sake bikeshedding.** Always mark severity and say plainly
whether a principal would merge it.

## How the review runs — deploy it, don't eyeball it

A principal-grade review is **multi-lens and adversarially checked**, so the skill
runs a **multi-agent workflow** by default rather than one linear read:

1. **Scope + enumerate files.** Resolve the scope to a concrete file list:
   - uncommitted diff → `but status -fv` (the default if unspecified)
   - a commit / branch → `git diff --name-only <base>..<tip>`
   - a subsystem / the repo → glob the path(s)
2. **Run the review workflow** with that list:
   ```
   Workflow({ scriptPath: ".claude/skills/principal-review/scripts/review.js",
              args: { scope: "<label>", repo: "<abs repo root>", files: [ ... ] } })
   ```
   It fans out **one reviewer per convention dimension present** — each loads its
   reference and reads the neighbor gold standard — then **adversarially verifies
   every finding** (real violation vs style opinion; false positives dropped),
   dedups, and returns a synthesized report.
3. **Present the returned report** (verdict + blockers / taste / nits + top-3).
4. **Offer to apply** the blockers + taste fixes — on request only, idiomatic and
   minimal, matching the cited neighbor.

Scale is automatic: a one-file change spins up 1–2 reviewers + their verifiers; a
whole-repo audit spins up one reviewer (plus verifiers) per dimension. Only for a
genuinely trivial single-file glance should you skip the workflow and review
inline against the matching reference below.

## Dimensions & references (what the workflow routes to)

| Files | Dimension | Reference |
|---|---|---|
| `runtime/host/**/*.{c,h}` | IREE runtime C | `references/iree-runtime-c.md` |
| `codegen/**/*.{cpp,h,td}` | IREE / MLIR compiler C++ | `references/mlir-compiler-cpp.md` |
| xDSL passes `*.py` | xDSL | `references/xdsl-passes.md` |
| `qcs`/`command_stream`/`shared_region`/`executable` headers | cross-component ABI | `references/abi-and-versioning.md` |
| `*.md`, READMEs, header banners, doc comments | documentation | `references/documentation.md` |
| `*.sh`, `CMakeLists.txt`, `tools/**/*.py`, Snitch runtime C | build/shell + Snitch C | `references/conventions.md` §4–§7 |
| **all code files** (light, always-on) | comment discipline | `references/conventions.md` §0 |
| **all changed code** (incl. submodule diffs) | upstreamability | `references/upstreamability.md` |

A file may match several dimensions and gets reviewed through each lens on
purpose. Every reviewer reads 1–2 *neighboring upstream* files as the bar and
cites them.

**Severity:** **BLOCKER** (correctness / ABI / portability / fails review) ·
**TASTE** (works, but a principal sends it back) · **NIT** (optional polish).

## Hills to die on (this repo's recurring sins — check these every time)

These are grounded in what has actually gone wrong here. Treat a hit as at least
TASTE, usually BLOCKER:

- **No absolute paths in scripts or build.** `/scratch/...`, `/home/...`, site
  tool dirs → env-overridable **with a sane default**, or computed from
  `__file__` / the submodule root. A script that only runs on one machine is
  broken. (The `tools/autotune/*.py` + `build_cheshire*.sh` sin.)
- **One ABI, versioned — never a vendored copy that "must match".** A hand-copied
  struct mirrored in another component (e.g. `qcs_kernel_abi.h` vs IREE's
  `executable_library.h`, or the split `compute_core_ptrs`/`dma_core_ptrs`) must
  be a **single source-of-truth header owned by the emitter**, with a version
  field checked at the boundary and `_Static_assert`s on layout. Two copies with
  a comment saying "keep in sync" is a latent silent-corruption bug.
- **No magic addresses or hardware constants in code.** L1/TCDM/zero-mem
  addresses, core counts, sizes come from the **cfg header / target attrs** (the
  `QUIDDITCH_L1_BASE` pattern), never a literal in a lowering or a `.c`.
- **Runtime C discipline.** Every entrypoint returns `iree_status_t` via
  `iree_make_status` / `IREE_RETURN_IF_ERROR`; ownership through
  `iree_allocator_t`, never bare `malloc`; `iree_`-prefixed, `_t`-suffixed names;
  the `api.h` include seam; freestanding-safe (no unguarded libc).
- **Shell scripts:** `set -euo pipefail`, quoted expansions, `trap` for cleanup,
  one job per script, a `usage`, real exit codes — not a linear wall of commands.
- **Comment discipline (hard, always checked).** No multi-line comment blocks,
  doc-dumps, or debugging narrative in code; no commented-out code; no leftover
  debug prints. A needed comment is **single-line**, matches the surrounding
  style and tone, and says WHY not WHAT. If the code being modified had no
  comment, don't add one — the *why* goes in the commit message / PR. A dedicated
  light `comments` pass scans every code file for this on every run: a multi-line
  body block is TASTE, a committed debugging journal or debug spew is a BLOCKER.
- No dead branches, no `TODO` without an owner or a tracking note.
- **Fork changes stay mergeable.** We work on forks (Quidditch, IREE, xDSL,
  snitch_cluster). Of every change ask: upstreamable, fork-only-justified, or
  unjustified divergence? Write it to upstream standards, keep it additive and
  isolated, and document any fork-only rationale. An upstreamable fix left
  fork-only with no tracked PR, or unjustified divergence, is a finding. (See
  `references/upstreamability.md`.)
- **License/copyright header** matching the ecosystem on new files (IREE
  Apache-2.0-WITH-LLVM-exception header on IREE-derived C/C++).
- **Naming matches the neighbors, not personal style.** If IREE calls it
  `iree_hal_command_buffer_t`, you do not introduce `CmdBuf`.
- **CMake:** cache vars + targets + generator expressions; do not hardcode paths
  or shell out to bash for what CMake expresses natively.
- **Docs are reviewed, and rot-proof.** A doc that restates a constant living in a
  generated header (`0x30000000`, a tile size, a tool version) instead of
  referencing the source is a defect; a runbook command that no longer runs is a
  BLOCKER (it actively misleads). New/changed public surface ships or updates its
  doc, leads with Status, and sources every perf claim from a measured number.
  (See `references/documentation.md`.)

## Output

```
## Principal review — <scope>

Verdict: <MERGE AS-IS | SEND BACK> — <one line>

### Blockers
- <file>:<line> — <what> → <idiomatic fix> (why: <risk/idiom>)

### Taste
- <file>:<line> — <what> → <fix> (neighbor: <upstream file it should match>)

### Nits (optional)
- <file>:<line> — <what>

### Top 3 to fix first
1. …
```

Then offer: "Want me to apply the blockers + taste fixes?" Apply only on request;
keep edits idiomatic and minimal, matching the cited neighbor.

References:
- `references/conventions.md` — the full per-ecosystem catalog (the index).
- `references/iree-runtime-c.md`, `mlir-compiler-cpp.md`, `xdsl-passes.md` —
  worked DO/DON'T examples with **real, cited** snippets from this tree.
- `references/abi-and-versioning.md` — cross-component ABI discipline (the QCS /
  export-table contracts).
- `references/documentation.md` — the doc standard + the rot-proof Nimbus doc set.
