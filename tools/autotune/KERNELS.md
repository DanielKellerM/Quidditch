# Kernel library — the ops the autotuner targets

The op set we want to implement + optimise on the Snitch cluster, ranked by
relevance to the workloads we run (NsNet2 = GRU speech enhancement; general ML).
Each row states the **reference gate** it needs (how Tier-1/Tier-2 prove
correctness), its **autotuner status**, and the **concrete change** to add it.

Which of these to optimise *first* on a real model is an Amdahl question — that
is what the whole-model profiler answers (sim → per-dispatch cycles → rank).
This list is the menu; the profile picks the order for a given model.

See `SNITCH_KERNEL_NOTES.md` for the Snitch runtime/kernel reference: the
core-gating API (how to keep work off the FPU-less DM core), the SSR/FREP +
double-buffer kernel idioms, the per-op reference-kernel mapping, and the
compute-core fp-reference unblock (gating the Tier-1 reference to a compute core
to admit nonlinear/large-K ops — sound, but needs a post-dispatch rendezvous).

## Status legend
- ✅ tunable today (single-dispatch, divisor tiles, f64)
- 🧩 needs a harness/scope widening (no codegen work)
- 🔨 needs compiler/codegen work (xDSL or IREE) before it can be tuned

| Op | Shows up in | Reference gate | Status | Blocker / change needed |
|----|-------------|----------------|--------|-------------------------|
| matmul / matmul_transpose_b | FC layers, GRU/LSTM gate projections | int-exact (Tier-1) | ✅ | none |
| elementwise add/mul/sub | residuals, gate combines, bias | int-exact (Tier-1) | 🧩 | `gen_harness` `_gen_elementwise` + 2D (M,N) shape in `spec` |
| relu / max | activations | int-exact (Tier-1) | 🧩 | as elementwise; piecewise-linear stays integer-exact |
| sigmoid / tanh | GRU/LSTM gates | **fp-tolerance (Tier-2)** | 🧩 | the fp-tolerance Tier-2 keystone (host rel+abs tol by dtype) |
| matvec (M=1 GEMV) | every RNN/GRU timestep | fp-tolerance (Tier-2) | 🧩 | M=1 currently rejected; int32 ref overflows large K → must go through the fp gate, not the int gate |
| reduction (sum/max) | softmax, pooling | int-exact / fp | 🧩 | 1D-output reference; tiling over the reduced axis |
| **fused matmul + bias + act** | the FC block | fp-tolerance (Tier-2) | 🔨 | multi-op reference + confirm IREE fuses bias+act epilogue into the matmul dispatch |
| **GRU / LSTM gate cell** | NsNet2 core | fp-tolerance (Tier-2) | 🔨 | the nsnet2 compile blocker (xDSL `zip()` crash + `util.assume.int`); the highest-value fusion |
| layernorm / softmax | transformers | fp-tolerance (Tier-2) | 🔨 | reduction + broadcast fusion; later |
| conv1d / conv2d | CNN front-ends | int-exact / fp | 🔨 | im2col vs direct; later |

## Widening order (dependency-sequenced)
1. **elementwise (add/mul)** — int-exact, smallest change; proves the non-matmul
   harness path. Candidates already present: `runtime/samples/vec_multiply`.
2. **relu/max** — same int-exact path; first "activation".
3. **fp-tolerance Tier-2 (keystone)** — harness dumps raw output, host computes
   rel+abs tol by dtype/K. Unblocks *everything* nonlinear and the int32-overflow
   shapes in one move: sigmoid/tanh, matvec/M=1, and f32/f16.
4. **sigmoid / tanh** — fp; the GRU gate nonlinearities.
5. **matvec / M=1** — drop the `1 in shape` reject once it rides the fp gate (the
   int reference overflows for real K; rv32 has no `fcvt.l.d` for int64 widening).
6. **fused matmul+bias+act** — the first fusion; the real per-layer win (no L2
   round-trip of the intermediate). Needs a multi-op reference.
7. **GRU/LSTM gate** — depends on fixing the nsnet2 GRU codegen gap; then it is a
   real compiling model to profile end-to-end.

## Fused ops — where the wins are
Fusion avoids staging intermediates through L2/HBM. Priority targets:
- matmul → +bias → +activation (FC block): one dispatch instead of three.
- GRU/LSTM gate: the recurrent cell as one fused kernel (the NsNet2 hotspot).
- matmul → layernorm / softmax epilogue (transformers).

The autotuner tunes a fused op the same way as a single op (it is one dispatch
with one `lowering_config`); the work is (a) getting IREE+xDSL to *form* the
fused dispatch and (b) a multi-op host reference for the Tier-1/Tier-2 gates.

## Invariants every new op must keep
- A new op **rejects cleanly** at `load_spec`/`gen_harness` until its reference
  gate exists — never emit a harness that can't prove correctness (no vacuous
  gate; this is why elementwise/f32 raise today rather than mis-build).
- DM core (hart_00008) has **no FPU** — the in-harness Tier-1 reference stays
  integer (`fcvt.w.d` + int mul/add). All floating-point checking is host-side
  (Tier-2). This is why nonlinear/large-K ops need the fp-tolerance Tier-2, not
  a richer in-harness reference.
