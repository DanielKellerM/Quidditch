#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Generate a standalone autotuner harness.c for an op-spec.

Fills harness.c.in with the kernel's query symbol, dims, dtype, bindings, a
structured-integer input fill and a per-element host reference (the Tier-1
correctness gate). Before generating, the spec's declared shape/dtype are
cross-checked against the kernel .mlir func signature, so a spec that lies about
the kernel is rejected here rather than silently producing a wrong harness.

v1 scope: matmul / matmul_transpose_b / elementwise in f64. f32/f16 references
need the host-side fp-tolerance gate (they raise SpecError, never emit a wrong
harness).
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spec import load_spec, SpecError

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "harness.c.in")
C_TYPE = {"f64": "double"}          # f32/f16 harness reference: step 3


def _mlir_arg_shapes(spec):
    """Static shapes of the entry func's args + result, in order."""
    text = re.sub(r"//[^\n]*", "", open(spec.mlir).read())
    m = re.search(r"func\.func\s+@" + re.escape(spec.entry) + r"\s*\((.*?)\)\s*->\s*([^\{]+)",
                  text, re.DOTALL)
    if not m:
        raise SpecError(f"[{spec.name}] entry @{spec.entry} not found in {os.path.basename(spec.mlir)}")
    sig = m.group(1) + " -> " + m.group(2)
    return [[int(x) for x in t.split("x")]
            for t in re.findall(r"tensor<([0-9x]+)x" + spec.dtype + r">", sig)]


def _mlir_transpose_b(spec):
    """Derive transpose-b from the linalg indexing_maps (authoritative); the B map
    has the contraction dim last for B[j,k] (transpose) vs first for B[k,j] (plain).
    Returns True/False, or None if no explicit maps (fall back to op name)."""
    text = re.sub(r"//[^\n]*", "", open(spec.mlir).read())
    maps = re.findall(r"affine_map<\([^)]*\)\s*->\s*\(([^)]*)\)>", text)
    if len(maps) < 3:
        return None
    a, b, c = (tuple(s.strip() for s in m.split(",")) for m in maps[:3])
    contraction = (set(a) & set(b)) - set(c)        # dim in A and B but not C
    if len(contraction) != 1:
        return None
    kd = contraction.pop()
    if b[-1] == kd and b[0] != kd:
        return True
    if b[0] == kd and b[-1] != kd:
        return False
    return None


def _crosscheck_matmul(spec):
    """Cross-check the spec against the .mlir (audit 2a) and return the authoritative
    transpose_b. Rejects a spec whose shape/dtype/transpose disagrees with the kernel."""
    M, N, K = spec.shape
    shapes = _mlir_arg_shapes(spec)
    if len(shapes) < 3:
        raise SpecError(f"[{spec.name}] expected 3 {spec.dtype} tensor operands in @{spec.entry}, "
                        f"found {len(shapes)} -- spec dtype/shape disagrees with the .mlir")
    a, b, c = shapes[0], shapes[1], shapes[2]
    spec_tb = spec.op == "matmul_transpose_b" or spec.transpose_b
    mlir_tb = _mlir_transpose_b(spec)
    transpose_b = spec_tb if mlir_tb is None else mlir_tb
    if mlir_tb is not None and mlir_tb != spec_tb:
        raise SpecError(f"[{spec.name}] spec transpose_b={spec_tb} disagrees with the .mlir "
                        f"indexing_maps (transpose_b={mlir_tb})")
    want_b = [N, K] if transpose_b else [K, N]
    if a != [M, K] or b != want_b or c != [M, N]:
        raise SpecError(f"[{spec.name}] spec shape M={M} N={N} K={K} (transpose_b={transpose_b}) "
                        f"disagrees with .mlir operands A={a} B={b} C={c}")
    return transpose_b


def _matmul_reference(M, N, K):
    """Host reference C[i,j]=sum_k A[i,k]*B[j,k] for the harness's structured inputs.
    A[i,k]=i*K+k+1, B value=k*N+j+1 (same value-map for both layouts)."""
    return [[sum((i * K + k + 1) * (k * N + j + 1) for k in range(K))
             for j in range(N)] for i in range(M)]


def _gen_matmul(spec, transpose_b):
    if len(spec.bindings) != 3:
        raise SpecError(f"[{spec.name}] matmul harness needs 3 bindings, got {len(spec.bindings)}")
    M, N, K = spec.shape
    # Inputs are distinct, asymmetric integers; assert the reference discriminates
    # row/column/transpose permutations. The reference uses INT32 (exact, simplest)
    # while the result fits, else an f64-EXACT reference: every core (incl. the DM
    # core that runs the check) has an FPU and matmul needs only fmul/fadd (NO div),
    # so f64 raises the exact-compare ceiling from 2^31 to 2^53 -- inputs are exact
    # integers and every partial sum stays < 2^53, so == is order-independent/exact.
    C = _matmul_reference(M, N, K)
    mx = max(max(r) for r in C)
    if mx >= 2 ** 53:
        raise SpecError(f"[{spec.name}] reference exceeds f64-exact (max {mx} >= 2^53) for "
                        f"M={M} N={N} K={K}; needs a rel+abs tolerance gate (not built)")
    fp_ref = mx >= 2 ** 31      # past int32 -> use the f64-exact reference + compare
    if len({tuple(r) for r in C}) != M:
        raise SpecError(f"[{spec.name}] reference has duplicate rows -- inputs miss row permutations")
    if len({tuple(C[i][j] for i in range(M)) for j in range(N)}) != N:
        raise SpecError(f"[{spec.name}] reference has duplicate columns -- inputs miss col permutations")
    if M == N and all(C[i][j] == C[j][i] for i in range(M) for j in range(N)):
        raise SpecError(f"[{spec.name}] reference is symmetric -- inputs miss a transpose miscompile")
    if transpose_b:
        b_decl = "static iree_alignas(64) elem_t B[NN * KK];"
        b_fill = ("  for (int j = 0; j < NN; j++)\n"
                  "    for (int k = 0; k < KK; k++)\n"
                  "      B[j * KK + k] = (elem_t)(k * NN + j + 1);")
        b_idx = "B[j * KK + k]"
    else:
        b_decl = "static iree_alignas(64) elem_t B[KK * NN];"
        b_fill = ("  for (int k = 0; k < KK; k++)\n"
                  "    for (int j = 0; j < NN; j++)\n"
                  "      B[k * NN + j] = (elem_t)(k * NN + j + 1);")
        b_idx = "B[k * NN + j]"
    buffer_decls = ("  static iree_alignas(64) elem_t A[MM * KK];\n"
                    f"  {b_decl}\n"
                    "  static iree_alignas(64) elem_t C[MM * NN];")
    input_fill = ("  for (int i = 0; i < MM; i++)\n"
                  "    for (int k = 0; k < KK; k++)\n"
                  "      A[i * KK + k] = (elem_t)(i * KK + k + 1);\n"
                  f"{b_fill}\n"
                  "  for (int idx = 0; idx < MM * NN; idx++) C[idx] = (elem_t)0;")
    binding_arrays = ("  void* binding_ptrs[3] = {A, B, C};\n"
                      "  size_t binding_lengths[3] = {sizeof(A), sizeof(B), sizeof(C)};")
    if fp_ref:   # f64-exact reference (mul/add only, no div -> runs on the DM core)
        check = ("  for (int i = 0; i < MM; i++)\n"
                 "    for (int j = 0; j < NN; j++) {\n"
                 "      elem_t want = (elem_t)0;\n"
                 "      for (int k = 0; k < KK; k++)\n"
                 f"        want += A[i * KK + k] * {b_idx};\n"
                 "      elem_t got = C[i * NN + j];\n"
                 "      if (got != want) {\n"
                 "        if (first_i < 0) { first_i = i; first_j = j;\n"
                 "          first_got = (int)got; first_want = (int)want; }\n"
                 "        errors++;\n"
                 "      }\n"
                 "    }")
    else:
        check = ("  for (int i = 0; i < MM; i++)\n"
                 "    for (int j = 0; j < NN; j++) {\n"
                 "      int want = 0;\n"
                 "      for (int k = 0; k < KK; k++)\n"
                 f"        want += (int)A[i * KK + k] * (int){b_idx};\n"
                 "      int got = (int)C[i * NN + j];\n"
                 "      if (got != want) {\n"
                 "        if (first_i < 0) { first_i = i; first_j = j; first_got = got; first_want = want; }\n"
                 "        errors++;\n"
                 "      }\n"
                 "    }")
    return {"BUFFER_DECLS": buffer_decls, "INPUT_FILL": input_fill,
            "BINDING_ARRAYS": binding_arrays, "CHECK": check, "NBIND": "3"}


# elementwise binary op (the linalg.generic body) -> the host-reference C operator.
_ELEM_OPS = {"arith.addf": "+", "arith.mulf": "*", "arith.subf": "-"}


def _mlir_elementwise_op(spec):
    """The single arith binary op in the linalg.generic body (authoritative)."""
    text = re.sub(r"//[^\n]*", "", open(spec.mlir).read())
    found = [op for op in _ELEM_OPS if op in text]
    if len(found) != 1:
        raise SpecError(f"[{spec.name}] elementwise: expected exactly one of "
                        f"{sorted(_ELEM_OPS)} in the kernel body, found {found}")
    return _ELEM_OPS[found[0]]


def _gen_elementwise(spec):
    if len(spec.bindings) != 3:
        raise SpecError(f"[{spec.name}] elementwise needs 3 bindings, got {len(spec.bindings)}")
    M = spec.shape[0]                          # flattened element count (1-D op)
    shapes = _mlir_arg_shapes(spec)            # cross-check: A,B,C all [M]
    if len(shapes) < 3 or any(s != [M] for s in shapes[:3]):
        raise SpecError(f"[{spec.name}] elementwise expects 3 [{M}] {spec.dtype} 1-D operands; "
                        f".mlir has {shapes}")
    c_op = _mlir_elementwise_op(spec)
    # Structured integer inputs A[L]=L+1, B[L]=2L+1: the per-element result is
    # strictly monotonic in L for +,*,- -> all values DISTINCT, so ANY index
    # permutation (wrong stride / tile-boundary bug) is caught.
    py = {"+": lambda a, b: a + b, "*": lambda a, b: a * b, "-": lambda a, b: a - b}[c_op]
    flat = [py(L + 1, 2 * L + 1) for L in range(M)]
    if max(abs(v) for v in flat) >= 2 ** 31:
        raise SpecError(f"[{spec.name}] int32 elementwise reference overflows for M={M}")
    if len(set(flat)) != M:
        raise SpecError(f"[{spec.name}] elementwise reference not all-distinct -- "
                        "inputs miss index permutations")
    buffer_decls = ("  static iree_alignas(64) elem_t A[MM];\n"
                    "  static iree_alignas(64) elem_t B[MM];\n"
                    "  static iree_alignas(64) elem_t C[MM];")
    input_fill = ("  for (int L = 0; L < MM; L++) {\n"
                  "    A[L] = (elem_t)(L + 1);\n"
                  "    B[L] = (elem_t)(2 * L + 1);\n"
                  "    C[L] = (elem_t)0;\n"
                  "  }")
    binding_arrays = ("  void* binding_ptrs[3] = {A, B, C};\n"
                      "  size_t binding_lengths[3] = {sizeof(A), sizeof(B), sizeof(C)};")
    check = ("  for (int L = 0; L < MM; L++) {\n"
             f"    int want = (int)A[L] {c_op} (int)B[L];\n"
             "    int got = (int)C[L];\n"
             "    if (got != want) {\n"
             "      if (first_i < 0) { first_i = L; first_j = 0; first_got = got; first_want = want; }\n"
             "      errors++;\n"
             "    }\n"
             "  }")
    return {"BUFFER_DECLS": buffer_decls, "INPUT_FILL": input_fill,
            "BINDING_ARRAYS": binding_arrays, "CHECK": check, "NBIND": "3"}


def generate(spec):
    if spec.dtype not in C_TYPE:
        raise SpecError(f"[{spec.name}] harness gen for dtype {spec.dtype}: step 3")
    if not spec.module:
        raise SpecError(f"[{spec.name}] spec needs `module` (the quidditch_module DST) to gen a harness")
    if spec.op in ("matmul", "matmul_transpose_b"):
        transpose_b = _crosscheck_matmul(spec)
        blocks = _gen_matmul(spec, transpose_b)
    elif spec.op == "elementwise":
        blocks = _gen_elementwise(spec)
    else:
        raise SpecError(f"[{spec.name}] harness gen for op {spec.op}: step 3")

    M, N, K = spec.shape
    subs = {"NAME": spec.name, "ENTRY": spec.entry, "QUERY": spec.query_symbol,
            "HEADER": f"{spec.module}.h", "M": str(M),
            "N": str(N if N is not None else 1),   # NN=1 for a 1-D elementwise op
            "K": str(K if K is not None else 1),   # KK unused; keep the #define valid
            "ELEM": C_TYPE[spec.dtype], **blocks}
    out = open(TEMPLATE).read()
    for k, v in subs.items():
        out = out.replace(f"@@{k}@@", v)
    leftover = sorted(set(re.findall(r"@@\w+@@", out)))
    if leftover:
        raise SpecError(f"[{spec.name}] unfilled template tokens: {leftover}")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("op", help="op-spec name under ops/")
    ap.add_argument("--out", help="output path (default: <sample dir>/harness.c)")
    args = ap.parse_args()
    spec = load_spec(args.op)
    out = args.out or os.path.join(os.path.dirname(spec.mlir), "harness.c")
    src = generate(spec)
    with open(out, "w") as f:
        f.write(src)
    print(f"generated {out} ({len(src)} bytes) for {spec.name} "
          f"[{spec.op} {spec.dtype} M={spec.shape[0]} N={spec.shape[1]} K={spec.shape[2]}]")


if __name__ == "__main__":
    try:
        main()
    except SpecError as e:
        sys.exit(str(e))
