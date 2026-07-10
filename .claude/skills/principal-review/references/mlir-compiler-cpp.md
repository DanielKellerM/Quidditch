# IREE / MLIR compiler C++ — worked examples

Applies to `codegen/compiler/src/Quidditch/**`. Every snippet is copied from a
real file in this repo — the Quidditch codegen is itself the gold standard here,
so match it. LLVM style throughout (clang-format LLVM, `PascalCase` types,
`camelCase` functions/vars, no `using namespace`, `///` on public decls).

---

## Conversion patterns — `matchAndRewrite` → `LogicalResult`

**Rule:** a lowering inherits the right pattern base (`ConvertOpToLLVMPattern<T>`
etc.), takes `OpAdaptor` + `ConversionPatternRewriter`, replaces via
`rewriter.replaceOp*`, and returns `success()`/`failure()`. Reuse the descriptor
helpers — don't hand-build IR the framework builds for you.

Idiomatic (`codegen/.../Conversion/ConvertSnitchToLLVM.cpp:31`):
```cpp
LogicalResult
matchAndRewrite(L1MemoryViewOp op, OpAdaptor adaptor,
                ConversionPatternRewriter &rewriter) const override {
  SmallVector<Value, 4> sizes, strides;
  Value size;
  this->getMemRefDescriptorSizes(op->getLoc(), op.getType(),
                                 adaptor.getOperands(), rewriter, sizes,
                                 strides, size);
  Value l1Address = rewriter.create<LLVM::ConstantOp>(
      op->getLoc(), rewriter.getI32IntegerAttr(static_cast<int32_t>(l1Base)));
  Value allocatedPtr = rewriter.create<LLVM::IntToPtrOp>(
      op->getLoc(), rewriter.getType<LLVM::LLVMPointerType>(), l1Address);
  auto desc = this->createMemRefDescriptor(op->getLoc(), op.getType(),
                                           allocatedPtr, allocatedPtr, sizes,
                                           strides, rewriter);
  rewriter.replaceOp(op, {desc});
  return success();
}
```

Not this:
```cpp
void matchAndRewrite(L1MemoryViewOp op, OpAdaptor adaptor,
                     ConversionPatternRewriter &rewriter) {
  Value ptr = rewriter.create<LLVM::IntToPtrOp>(...);
  rewriter.replaceOp(op, {ptr});
}
```
Returns `void` (no error path), skips the descriptor helpers, drops the adaptor
operands — BLOCKER.

---

## Passes — TableGen `.td` + `GEN_PASS_DEF` registration

**Rule:** define the pass in `Passes.td` (`Pass<"name", "op-type">` with a
`description`), then in the `.cpp` pull `GEN_PASS_DEF_<NAME>`, extend
`impl::<Name>PassBase<...>`, implement `runOnOperation()`. Never hand-inherit
`OperationPass`.

Idiomatic (`codegen/.../Target/Passes.td:6` + `LinkExecutables.cpp:12`):
```cpp
def LinkExecutablesPass : Pass<"quidditch-link-executables", "mlir::ModuleOp"> {
  let description = [{
    Combines all `hal.executable.variant`s of the same target into a single
    `hal.executable.variant` nested within one `hal.executable`.
  }];
}
// LinkExecutables.cpp:
namespace quidditch {
#define GEN_PASS_DEF_LINKEXECUTABLESPASS
#include "Quidditch/Target/Passes.h.inc"
}
class LinkExecutables
    : public quidditch::impl::LinkExecutablesPassBase<LinkExecutables> {
protected:
  void runOnOperation() override;
```
The `description` is help text — mandatory. Hand-rolling `OperationPass` with a
`runOp()` method loses auto-registration and options support — BLOCKER.

---

## Reading target config — `ExecutableTargetAttr` with a documented default

**Rule:** hardware facts arrive as target-attr config, not literals. Use
`IREE::HAL::ExecutableTargetAttr::lookup`, null-check each level, read with
`getAs<IntegerAttr>(key)`, and **document the fallback default**.

Idiomatic (`codegen/.../Conversion/ConvertSnitchToLLVM.cpp:207`):
```cpp
// Resolve the cfg-derived cluster L1/TCDM base from the module's target attr;
// default to the occamy/dev-box base when absent (no cfg header).
int64_t l1Base = 0x10000000;
if (auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(moduleOp))
  if (DictionaryAttr cfg = targetAttr.getConfiguration())
    if (auto attr = cfg.getAs<IntegerAttr>("l1_base"))
      l1Base = attr.getInt();
```
This is *why* the compiler retargets gwaihir (0x20000000) with no code change.

Not this:
```cpp
auto cfg = moduleOp->getAttr("hal.executable.target");
if (cfg) { auto dict = dyn_cast<DictionaryAttr>(cfg);
  if (dict && dict.contains("l1_base"))
    l1Base = dict.get("l1_base").cast<IntegerAttr>().getInt(); }
```
Bypasses the typed API, string-keys the raw attr, no documented default — TASTE
(fragile, invisible defaults).

---

## No magic constants — name it, comment its origin, read cfg first

**Rule:** every hardware address/size is a named variable, read from cfg with a
comment tying it to the cfg key + default + failure mode. A raw literal in a
lowering is a BLOCKER.

Idiomatic (`codegen/.../Conversion/ConvertDMAToLLVM.cpp:131`):
```cpp
// Cluster AXI zero-memory region, cfg-derived via the l1_zero_mem_base /
// l1_zero_mem_size target attrs ... Defaults to occamy (0x10030000 / 0x10000);
// gwaihir's is 0x20030000 / 0xF000. With the wrong base the zero-fill DMA
// targets a dead region -> the iDMA never drains -> the DM core wedges -> the
// host polls completion forever, hence this must track the cfg addrmap.
unsigned zeroMemAddress = 0x10030000;
unsigned zeroMemSize = 0x10000;
if (auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(moduleOp))
  if (DictionaryAttr cfg = targetAttr.getConfiguration()) {
    if (auto attr = cfg.getAs<IntegerAttr>("l1_zero_mem_base"))
      zeroMemAddress = attr.getValue().getZExtValue();
    if (auto attr = cfg.getAs<IntegerAttr>("l1_zero_mem_size"))
      zeroMemSize = attr.getValue().getZExtValue();
  }
```

Not this:
```cpp
Value zero = rewriter.create<LLVM::ConstantOp>(
    op->getLoc(), rewriter.getI32IntegerAttr(0x10030000));
```
Inline magic address: no name, no comment, no cfg read — breaks on any non-occamy
SoC. BLOCKER.

---

## Diagnostics — `emitError`/`emitWarning` + `attachNote`, return the diagnostic

**Rule:** emit through the op (`op->emitError`, `rewriter.notifyMatchFailure`),
attach context with `attachNote`, and return the diagnostic as `failure()`. Don't
throw away the context that helps the user fix it.

Idiomatic (`codegen/.../Conversion/ConvertToRISCV.cpp:131`):
```cpp
if (ret != 0) {
  auto diagEmit = assertCompiled ? &Operation::emitError : &Operation::emitWarning;
  InFlightDiagnostic diag = ((kernelOp)->*diagEmit)("Failed to translate kernel with xDSL");
  if (llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
          llvm::MemoryBuffer::getFile(stderrFile, /*IsText=*/true))
    diag.attachNote() << "stderr:\n" << buffer.get()->getBuffer();
  return diag;
}
```
Note the policy gate (error vs warning) and the attached `xdsl-opt` stderr — a
bare `emitError("failed"); return failure();` throws away the one thing the user
needs. TASTE→BLOCKER depending on how load-bearing the context is.
