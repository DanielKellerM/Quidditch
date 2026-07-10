# xDSL passes — worked examples

Applies to `xdsl/**` and xDSL-style lowering. Every snippet is copied from a real
pass in this repo — prefer the Snitch/RISC-V lowering passes as the model since
they match what this project actually ships.

Gold-standard neighbors: `xdsl/xdsl/transforms/convert_riscv_scf_for_to_frep.py`,
`convert_memref_stream_to_snitch_stream.py`, `test_lower_linalg_to_snitch.py`.

---

## Pass shape — `@dataclass(frozen=True)` `ModulePass` with typed fields

**Rule:** a pass is a frozen dataclass with a `name`, optional **typed** fields
with defaults (one terse comment each), and an `apply(ctx, op)` that drives a
`PatternRewriteWalker`. No hand-written `__init__`.

Idiomatic (`xdsl/xdsl/transforms/convert_riscv_scf_for_to_frep.py:104`):
```python
@dataclass(frozen=True)
class ConvertRiscvScfForToFrepPass(ModulePass):
    name = "convert-riscv-scf-for-to-frep"
    max_insns: int = 32

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([
                ScfYieldLowering(),
                ScfForLowering(self.max_insns),
            ]),
            apply_recursively=False,
        ).rewrite_module(op)
```

Not this:
```python
class ConvertRiscvScfForToFrepPass(ModulePass):
    name = "convert-riscv-scf-for-to-frep"
    def __init__(self, max_insns: int = 32):
        self.max_insns = max_insns
    def apply(self, ctx, op):
```
Manual `__init__` breaks CLI arg parsing; untyped `ctx`/`op` violate the xDSL
contract; not frozen loses immutability. BLOCKER for a shipped pass.

---

## Rewrite patterns — `@op_type_rewrite_pattern`, replace don't erase

**Rule:** a `RewritePattern` uses `@op_type_rewrite_pattern` for typed dispatch,
guards early, and emits via `rewriter.replace_op` (never `erase_op` that leaves
dangling uses).

Idiomatic (`xdsl/xdsl/transforms/convert_riscv_scf_for_to_frep.py:18`):
```python
class ScfForLowering(RewritePattern):
    def __init__(self, max_insns: int = 32):
        self.max_insns = max_insns

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: riscv_scf.ForOp,
                          rewriter: PatternRewriter) -> None:
        if indvar.uses:
            return
        rewriter.replace_op(
            op,
            (iter_count := riscv.SubOp(op.ub, op.lb),
             riscv_snitch.FrepOuterOp(...)),
        )
```
`@op_type_rewrite_pattern` gives IDE help and correct dispatch; the walrus builds
ops inline. Manual `isinstance` + `erase_op` without a replacement is a
correctness BLOCKER (dangling uses).

---

## Type hints & naming — full types, `cast` for narrowing, `<Op><Action>` names

**Rule:** fully qualified types in every signature; `cast(...)` for safe
narrowing; pattern classes named `<Thing>Lowering` / passes `<Thing>Pass`. No
one-letter names.

Idiomatic (`xdsl/xdsl/transforms/convert_memref_stream_to_snitch_stream.py:62`):
```python
class ReadOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref_stream.ReadOp, rewriter: PatternRewriter
    ) -> None:
        stream_type = op.stream.type
        value_type = cast(
            memref_stream.ReadableStreamType[Attribute], stream_type
        ).element_type
```
`class R` with untyped `op`/`rewriter` and unguarded `.element_type` is a TASTE
finding — opaque and static-analysis-blind.

---

## Registration — lazy factory closures in `get_all_passes()`

**Rule:** register a pass as a closure that imports lazily and returns the pass
**class** (not an instance), so imports stay non-circular and CLI arg binding
works.

Idiomatic (`xdsl/xdsl/transforms/__init__.py:44`):
```python
def get_all_passes() -> dict[str, Callable[[], type[ModulePass]]]:
    def get_canonicalize():
        from xdsl.transforms import canonicalize
        return canonicalize.CanonicalizePass
    return {"canonicalize": get_canonicalize, ...}
```
Eager top-level imports + instances (`CanonicalizePass()`) risk circular imports
and break CLI options binding — TASTE→BLOCKER.

---

## Tests — filecheck under `tests/filecheck/`, positive + negative

**Rule:** a new pass/param ships a filecheck test: `// RUN: xdsl-opt -p <pass> %s |
filecheck %s`, `CHECK`/`CHECK-NEXT` for structure, and a negative (no-op) case.

Idiomatic (`xdsl/tests/filecheck/backend/riscv/convert_riscv_scf_for_to_frep.mlir:1`):
```mlir
// RUN: xdsl-opt -p convert-riscv-scf-for-to-frep %s | filecheck %s
riscv_scf.for %index0 : !riscv.reg<a4> = %i0 to %i1 step %c1 {
    %f4 = riscv_snitch.read from %readable : !riscv.freg<ft0>
}
// CHECK:         riscv_snitch.frep_outer %1 {
// CHECK-NEXT:      %f4 = riscv_snitch.read from %readable : !riscv.freg<ft0>
// CHECK-NEXT:    }
// Failure case (induction var used -> not lowered):
riscv_scf.for %index_ind_var : !riscv.reg<a4> = %i0 to %i1 step %c1 { }
// CHECK:    riscv_scf.for %index_ind_var
```
A "test" with no `RUN` line and a vague `// CHECK: frep` validates nothing — a
missing/degenerate test on a new pass is a BLOCKER.
