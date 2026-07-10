# Convention catalog

The per-ecosystem bar. Each section is the standard a principal engineer applies
to that kind of file. When reviewing, load the section(s) for the files in scope,
then read a neighboring upstream file as the concrete gold standard and cite it.

The meta-rule above all of these: **match the neighbors.** These sections encode
what the neighbors already do — when in doubt, the surrounding upstream code wins
over any rule written here.

> **Worked examples with real, cited snippets** live alongside this file — prefer
> them when reviewing that ecosystem: `iree-runtime-c.md`, `mlir-compiler-cpp.md`,
> `xdsl-passes.md`, `abi-and-versioning.md` (the deep version of §8),
> `documentation.md` (the doc standard + the rot-proof Nimbus doc set). This
> catalog is the index; those are the depth.

---

## 0. Universal (every file)

- **Indistinguishable from the best code around it.** Naming, layout, error
  handling, comment density, abstraction level all match the surrounding
  upstream. Foreign-but-working = defect.
- **One responsibility.** A file / function / script does one thing. "Builds AND
  runs AND checks AND cleans" → split. A `.c` that is five loosely-related helpers
  → split or justify.
- **No magic numbers.** Every non-obvious literal is a named constant, a cfg
  value, or a target attr, with the derivation visible. Hardware addresses/sizes
  are **never** literals in logic.
- **No dead code.** No commented-out blocks, no unreachable branches, no
  `#if 0`, no "kept just in case". Git remembers.
- **Comment discipline (hard rule).** No multi-line comment blocks in
  implementation code, no documentation-dumps, no debugging narrative, no
  commented-out code, no leftover debug prints. If a comment is genuinely needed
  it is **single-line** and matches the surrounding style and tone. If the code
  you are modifying had **no** comment, do not add one — the explanation of *why*
  belongs in the commit message or the PR, not inline. A comment says WHY (an
  invariant, a non-obvious reason), never WHAT. A multi-line explanatory block in
  a function body is a TASTE finding; a committed debugging journal or debug spew
  is a BLOCKER. (Enforced by the light `comments` pass in `scripts/review.js`.)
- **`TODO` has an owner or a tracking reference.** A bare `TODO` is a nit; a
  `TODO` hiding a correctness gap is a blocker.
- **License header** matching the ecosystem on new files.
- **Names are precise and match the domain vocabulary** already in the tree
  (workgroup, dispatch, cluster, TCDM, SSR, FREP, QCS, binding) — not invented
  synonyms.

---

## 1. IREE runtime C — `runtime/host/**` (HAL, transport, firmware, host)

Gold standard: read `iree/runtime/src/iree/hal/command_buffer.{h,c}`, a real
driver under `iree/runtime/src/iree/hal/drivers/`, and `iree/base/api.h`.

- **Error handling:** every fallible function returns `iree_status_t`. Construct
  with `iree_make_status(IREE_STATUS_..., "msg %d", x)`. Propagate with
  `IREE_RETURN_IF_ERROR(...)`. Never return a bare int/bool for an operation that
  can fail. `iree_ok_status()` for success. Clean up on the error path.
- **Ownership/allocation:** allocate through an `iree_allocator_t`
  (`iree_allocator_malloc` / `_free`), never bare `malloc`/`free`. Document who
  owns each pointer. Paired create/destroy. Retain/release for refcounted HAL
  objects (`iree_hal_*_retain/release`).
- **Naming:** `iree_`-prefixed, module-scoped
  (`iree_hal_cluster_command_buffer_...`), types `_t`-suffixed, enums
  `IREE_HAL_...`. A custom HAL object mirrors IREE's vtable pattern
  (`iree_hal_<obj>_vtable_t` with the exact upstream function set).
- **Headers:** the `api.h` seam — public surface in `*.h`, one struct/vtable per
  object, include what you use, header guards or `#pragma once` matching the
  neighbors. No leaking internal structs into public headers.
- **Freestanding (firmware / bare-metal host):** no unguarded libc, no hosted
  assumptions; match how the existing firmware handles `memcpy`/`memset`/print.
  Static buffers are sized from cfg, `_Static_assert`-bounded, and commented with
  the bound's rationale.
- **ABI boundaries (QCS, executable table, job descriptor):** see §8 — this is
  where the most expensive defects live.

DON'T:
```c
uint8_t* p = malloc(n);              // bare malloc
if (record_it(cb) != 0) return -1;   // int error, no status
cluster_cmd_buf_t                    // foreign naming (should be iree_hal_cluster_command_buffer_t)
```

---

## 2. IREE compiler C++ / MLIR — `codegen/compiler/src/Quidditch/**`

Gold standard: read an in-tree IREE conversion pass and
`iree/compiler/.../PassDetail.h`; follow LLVM/MLIR style.

- **LLVM style:** clang-format (LLVM), no `using namespace`, `PascalCase` types,
  `camelCase` functions/vars, `///` doc comments on public declarations. Match
  `.clang-format` if present.
- **Ops & passes in TableGen** where the neighbors do; passes registered the
  standard way; use `LogicalResult` / `success()` / `failure()` /
  `emitError()`, not exceptions or bare bools.
- **Rewrite patterns** over hand-rolled IR walks; `PatternRewriter` APIs;
  `getContext()`; `matchAndRewrite` shape matching the in-tree patterns.
- **No hardware constants baked in.** Addresses/sizes/core-counts come from the
  `ExecutableTargetAttr` configuration (the `l1_base` / `compute_cores` /
  `l1_zero_mem_*` pattern in `QuidditchTarget.cpp`), read with a documented
  fallback default. A raw `0x10000000` in a lowering is a blocker — thread it
  through the target attr (see §7 cfg-target pattern).
- **Pass pipeline strings** are gated/derived, not silently hardcoded; a custom
  override flag must not silently bypass a correctness gate without a comment.
- Doc comment on every pass: what it lowers, its pre/post-conditions.

---

## 3. xDSL Python passes — `xdsl/**` and xDSL-style lowering

Gold standard: read an existing pass under
`xdsl/transforms/` (e.g. `convert_riscv_scf_for_to_frep.py`) and the
`test_lower_linalg_to_snitch.py` aggregate.

- **Structure:** a `ModulePass` (or `RewritePattern` + `PatternRewriteWalker`),
  `@dataclass(frozen=True)` fields with defaults and a one-line comment on each
  tunable (as `fp64_pipeline_depth` does), `snake_case` names.
- **Type hints everywhere** (params, returns, fields). No untyped public API.
- **Fast-path invariants preserved** — if the canonical config must return a
  byte-identical pipeline (as the aggregate does), keep that property and test it.
- **User-facing errors** through xDSL verification/diagnostics, not bare
  `assert`. `assert` is for internal invariants only.
- **Tests:** filecheck under `tests/filecheck/`, mirroring the upstream layout;
  a new pass/param ships with a test.
- Match upstream xDSL's import style and pass-registration.

---

## 4. Snitch runtime C — `snitch_cluster/sw/**`, cfg-header consumers

- `snrt_` prefix; the `snitch_cluster_cfg.h` cfg-header pattern is the single
  source of topology/TCDM/DMA facts — consumers read the `CFG_*` / `SNRT_*` /
  `QUIDDITCH_*` defines, never re-hardcode.
- Section/attribute discipline (`.cbss`, `used`/`retain` where the linker
  otherwise dead-strips — as fixed in `cls.c`); comment the linker interaction.
- Single-vs-multi-cluster paths guarded on `SNRT_CLUSTER_NUM` where the primitive
  is undefined for one (as in the `snrt_inter_cluster_barrier` guard).

---

## 5. Shell scripts — `*.sh`

Gold standard: a script should be safe to run twice, on any machine, and fail
loudly.

- **Header:** `#!/usr/bin/env bash` + `set -euo pipefail`.
- **No absolute paths.** No `/scratch/...`, `/home/...`, or site tool dirs baked
  in. Every external location is an env var **with a sane default** or computed
  from the script's own location:
  ```bash
  HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  IREE_SRC="${IREE_SRC:-$HERE/../../iree}"
  LLVM_BIN="${LLVM_BIN:?set LLVM_BIN to the snitch LLVM bin dir}"
  ```
- **Quote expansions** (`"$var"`), use arrays for arg lists, `trap '...' EXIT`
  for cleanup, a `usage()` and arg parsing for anything with options.
- **One job per script.** build / run / check / clean are separate scripts or
  separate subcommands, not a linear wall.
- Real exit codes; check the results you depend on; don't `cmd || true` away
  errors silently.

DON'T (the pattern to flag on sight):
```bash
RV=/usr/pack/riscv-1.0-kgf/riscv64-gcc-13.2.0/bin      # site-hardcoded
IREE_SRC=/home/dankeller/Projects/Quidditch/iree        # machine-hardcoded
PARENT=/scratch/dankeller/snitch-compiler/iree-rv64-host # scratch-hardcoded
```

---

## 6. Python build glue — `tools/**/*.py`

- **Paths from `pathlib` + `__file__`**, or env overrides — never a module-level
  `ROOT = "/home/dankeller/Projects/Quidditch"`. Compute:
  `ROOT = Path(__file__).resolve().parents[N]`.
- Tool locations (`iree-compile`, `xdsl-opt`, LLVM bin, toolchain) via env with
  defaults resolved from the submodule/build layout, not `/scratch` literals.
- Don't `subprocess` out to bash for what Python does natively; if you must shell
  out, build arg lists (no string interpolation into a shell).
- Type hints, `argparse` for CLIs, `logging` not bare `print` for anything
  reusable, explicit work/output dirs (env-overridable).

---

## 7. CMake & the cfg-target pattern

- Cache variables + targets + generator expressions. No hardcoded paths; expose
  knobs as `CACHE` vars (the `QUIDDITCH_CODEGEN_BUILD_DIR` /
  `QUIDDITCH_CLUSTER_CFG_HEADER` pattern is correct — reuse it).
- Generated headers are produced by a custom command with declared
  outputs/deps, exposed to consumers via a cache var or an interface target —
  not read from a guessed build path.
- **cfg-target rule:** any SoC/hardware fact enters the compiler as an *input*
  (a generated cfg header → target attr), following the existing
  `snitch_cluster_cfg.h.tpl` → `QuidditchTarget.cpp` → `Convert*ToLLVM` chain.
  New hardware = a new cfg, zero code edits. Flag anything that would require a
  code edit to retarget.

---

## 8. ABI & versioning discipline (the expensive one)

A contract that spans two components (host↔firmware, compiler↔firmware) is where
silent corruption hides. The bar:

- **Single source of truth.** The struct/layout is defined once, in a header
  owned by the **emitter** (the side that writes it), and included by the
  consumer. No hand-copied "keep in sync" twin. (The `qcs_kernel_abi.h` vendored
  slice + the `compute_core_ptrs`/`dma_core_ptrs` fork is exactly the shape to
  fix.)
- **Versioned.** A version/magic field written by the producer and **checked** by
  the consumer at the boundary (the QCS job-descriptor magic/version is the right
  instinct — apply it to every cross-component struct).
- **Layout-asserted.** `_Static_assert`/`static_assert` on `sizeof` and
  `offsetof` for every field the other side dereferences, at the definition site.
- A fixed device-PA / offset contract (`QCS_JOB_DESCRIPTOR_OFFSET`) is defined
  once and shared, not re-`#define`d in three files.

---

## 9. Repo structure & hygiene

- Build artifacts, venvs, bender checkouts, sim outputs are **gitignored and
  never tracked** (`build/`, `build-rt/`, `runs/`, `venv/`, `toolchain/`,
  `.bender/`, `modelsim.ini`, `transcript`).
- A subsystem has a `README` when its usage isn't obvious from the code; docs
  live in `docs/` and are kept honest (no stale address prose contradicting the
  generated source).
- Submodule pins are deliberate; a gitlink bump is its own commit with a message
  saying what moved and why.
- No secrets, no machine-specific config, no personal paths committed.

---

## Running a large audit

For a subsystem or whole-repo review, fan out instead of eyeballing linearly:

1. **Partition** the scope by subsystem (host / transport / firmware / codegen /
   xdsl / autotune / build) × the convention dimensions above.
2. **One reviewer per partition** reads its files *and the neighboring upstream
   gold standard*, returns structured findings `{file:line, severity, rule,
   idiomatic fix, neighbor cited}`.
3. **Adversarially verify** each finding — a second pass asks "is this a real
   idiom violation / risk, or a style opinion?" Drop opinions; keep violations.
4. **Dedup + synthesize** into one report, sorted by severity, with the top fixes
   and a single merge/send-back verdict.

This is exactly what `scripts/review.js` implements, and the skill runs it by
default (see `SKILL.md`). This section documents the *shape* for when you author
or extend that workflow; scale the reviewer count to the scope and `log()` what
was and wasn't covered.
