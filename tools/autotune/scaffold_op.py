#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Scaffold a starter op-spec (ops/<name>.toml) from a kernel .mlir.

Derives entry / dtype / shape / transpose-b from the func signature and the
linalg indexing_maps, picks tile choices that divide M,N,K, and writes a TOML
the autotuner can load. Validates it via load_spec (so an out-of-v1-scope kernel
is rejected here with a clear message). Review the result -- especially `module`
(the iree quidditch_module DST, which is a build choice, defaulted to <name>_mod).

Usage: python3 scaffold_op.py <name> <kernel.mlir> [--module <dst>]
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spec import ROOT, OPS_DIR, load_spec, SpecError

TILE_CANDIDATES = (4, 8, 16, 32, 64)


def _transpose_b(text):
    maps = re.findall(r"affine_map<\([^)]*\)\s*->\s*\(([^)]*)\)>", text)
    if len(maps) < 3:
        return None
    a, b, c = (tuple(s.strip() for s in m.split(",")) for m in maps[:3])
    contraction = (set(a) & set(b)) - set(c)
    if len(contraction) != 1:
        return None
    kd = contraction.pop()
    return b[-1] == kd and b[0] != kd


def scaffold(name, mlir_path, module=None):
    mlir_abs = mlir_path if os.path.isabs(mlir_path) else os.path.join(ROOT, mlir_path)
    if not os.path.isfile(mlir_abs):
        sys.exit(f"no .mlir at {mlir_abs}")
    text = re.sub(r"//[^\n]*", "", open(mlir_abs).read())
    m = re.search(r"func\.func\s+@(\w+)\s*\((.*?)\)\s*->\s*([^\{]+)", text, re.DOTALL)
    if not m:
        sys.exit("no func.func found in the .mlir")
    entry, args, ret = m.group(1), m.group(2), m.group(3)
    dt = re.search(r"tensor<[0-9x]+x(f\d+)>", text)
    if not dt:
        sys.exit("no static f<N> tensor operand found (dynamic shape / non-float?)")
    dtype = dt.group(1)
    shapes = [[int(x) for x in t.split("x")]
              for t in re.findall(r"tensor<([0-9x]+)x" + dtype + r">", args + " -> " + ret)]
    if len(shapes) < 3:
        sys.exit(f"expected 3 {dtype} tensor operands, found {len(shapes)} -- "
                 "not a single matmul (multi-dispatch / unsupported)")
    tb = _transpose_b(text)
    transpose_b = bool(tb)
    A, B = shapes[0], shapes[1]
    M, K = A
    N = B[0] if transpose_b else B[1]
    tiles = [t for t in TILE_CANDIDATES if M % t == 0 and N % t == 0 and K % t == 0]
    if not tiles:
        sys.exit(f"no tile choice in {TILE_CANDIDATES} divides M={M} N={N} K={K}")
    base = tiles[len(tiles) // 2]
    op = "matmul_transpose_b" if transpose_b else "matmul"
    rel = os.path.relpath(mlir_abs, ROOT)

    toml = f"""# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Autotuner op-spec scaffolded from {rel}. REVIEW `module` (the quidditch_module
# DST -> #include <<module>.h>) -- it is a build choice, not in the .mlir.
name        = "{name}"
mlir        = "{rel}"
entry       = "{entry}"
module      = "{module or name + '_mod'}"
op          = "{op}"
dtype       = "{dtype}"
transpose_b = {str(transpose_b).lower()}
shape       = {{ M = {M}, N = {N}, K = {K} }}
bindings    = ["A", "B", "C"]

[knobs]
l1_tiles    = {{ type = "triple", choices = {tiles} }}
dual_buffer = {{ type = "bool" }}

[validation]
baseline_tag = "{base}x{base}x{base}_dbtrue"
"""
    os.makedirs(OPS_DIR, exist_ok=True)
    out = os.path.join(OPS_DIR, f"{name}.toml")
    with open(out, "w") as f:
        f.write(toml)
    try:
        s = load_spec(name)
    except SpecError as e:
        os.remove(out)
        sys.exit(f"scaffolded spec is out of v1 scope, not written:\n{e}")
    print(f"wrote {out}\n  {s.op} {s.dtype} M={s.shape[0]} N={s.shape[1]} K={s.shape[2]} "
          f"transpose_b={s.transpose_b} tiles={s.tile_choices}")
    print(f"  next: python3 {os.path.relpath(__file__, ROOT).replace('scaffold_op.py','sweep.py')} --op {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("mlir", help="kernel .mlir path (relative to repo root or absolute)")
    ap.add_argument("--module", help="quidditch_module DST name (default <name>_mod)")
    args = ap.parse_args()
    scaffold(args.name, args.mlir, args.module)


if __name__ == "__main__":
    main()
