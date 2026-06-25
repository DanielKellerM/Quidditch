#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Op-spec loader for the Snitch kernel autotuner.

One TOML per tunable kernel (ops/<name>.toml) is the single source of truth for
the harness generator, direct_build, and the correctness reference -- replacing
the gemm_square constants once baked into harness.c / sweep.py / direct_build.py.

Loading enforces the v1 scope (single-dispatch, static shape, known op + dtype,
tile choices that divide the problem). Anything outside it is REJECTED here with
a named error rather than silently mis-built.
"""
import os
import re
import tomllib
from dataclasses import dataclass

ROOT = "/home/dankeller/Projects/Quidditch"
OPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ops")

V1_OPS = {"matmul", "matmul_transpose_b", "elementwise"}
V1_DTYPES = {"f64"}             # f32/f16 need the host-side fp-tolerance gate (Tier-2)
DTYPE_BYTES = {"f64": 8, "f32": 4, "f16": 2}


class SpecError(ValueError):
    """A kernel outside the autotuner's v1 scope (rejected at load, not built)."""


@dataclass
class Spec:
    name: str
    mlir: str               # absolute path to the kernel .mlir
    entry: str              # func.func symbol (dispatch ordinal 0)
    module: str             # iree quidditch_module DST (header is <module>.h)
    op: str                 # matmul | matmul_transpose_b | elementwise
    dtype: str              # f64 | f32 | f16
    shape: tuple            # (M, N, K)
    transpose_b: bool
    bindings: list          # ordered binding names, matches binding_ptrs
    tile_choices: list      # l1_tiles knob choices
    dual_buffer: list       # ["true", "false"] or ["false"]
    interchange_choices: list  # l1_tiles_interchange perms; default [[2,0,1]]
    baseline_tag: str       # PERF anchor only, never a correctness reference

    @property
    def query_symbol(self):
        return f"quidditch_{self.entry}_dispatch_0_library_query"

    @property
    def elem_bytes(self):
        return DTYPE_BYTES[self.dtype]

    @property
    def ndims(self):
        """Loop/tile dimensionality: matmul = 3 (M,N,K); elementwise = 1. IREE
        collapses an N-D elementwise to a single flattened parallel loop before
        TensorTile, so the tunable l1_tiles is 1 entry over the total elements."""
        return 1 if self.op == "elementwise" else 3

    @property
    def default_interchange(self):
        """The canonical (no-tag-suffix) l1_tiles_interchange for this op shape."""
        return [2, 0, 1] if self.ndims == 3 else list(range(self.ndims))


def _reject(name, msg):
    raise SpecError(f"[{name}] REJECT: {msg}")


def _check_mlir_scope(name, mlir):
    raw = open(mlir).read()
    text = re.sub(r"//[^\n]*", "", raw)          # strip comments: prose mentions ops too
    if re.search(r"tensor<[^>]*\?", text):
        _reject(name, "dynamic tensor shape (v1 is static-shape only)")
    payload = [o for o in re.findall(r"\blinalg\.\w+", text) if o != "linalg.yield"]
    if len(payload) > 1:
        _reject(name, f"multi-dispatch ({len(payload)} linalg ops {payload}); "
                      "v1 drives ordinal 0 only -- single-dispatch kernels only")


def load_spec(name):
    """Load and validate ops/<name>.toml (or a direct .toml path). Raises
    SpecError for any kernel outside the v1 scope."""
    path = name if name.endswith(".toml") else os.path.join(OPS_DIR, f"{name}.toml")
    if not os.path.isfile(path):
        raise SpecError(f"no op-spec at {path}")
    with open(path, "rb") as f:
        d = tomllib.load(f)
    nm = d.get("name") or os.path.splitext(os.path.basename(path))[0]

    op = d.get("op")
    if op not in V1_OPS:
        _reject(nm, f"unknown op {op!r} (v1: {sorted(V1_OPS)})")
    dtype = d.get("dtype")
    if dtype not in V1_DTYPES:
        _reject(nm, f"dtype {dtype!r} has no defined tolerance (v1: {sorted(V1_DTYPES)})")

    # matmul tiles 3 loops (M,N,K); elementwise collapses to 1 flattened loop.
    ndims = 1 if op == "elementwise" else 3
    dim_names = ("M", "N", "K")[:ndims]
    sh = d.get("shape", {})
    dims = [sh.get(dn) for dn in dim_names]
    if any(v is None for v in dims):
        _reject(nm, f"shape must give {', '.join(dim_names)}")
    if 1 in dims:
        _reject(nm, "degenerate dim =1 (matvec / zero-tile regime) unsupported in v1")
    shape = tuple(dims) + (None,) * (3 - ndims)   # pad to (M, N, K); K=None for 2D ops

    knobs = d.get("knobs", {})
    tile_choices = knobs.get("l1_tiles", {}).get("choices", [])
    if not tile_choices:
        _reject(nm, "knobs.l1_tiles.choices required")
    if 0 in tile_choices:
        _reject(nm, "zero tile choice (untiled-dim convention) unsupported in v1")
    for dim, v in zip(dim_names, dims):
        if any(v % t for t in tile_choices):
            _reject(nm, f"shape {dim}={v} not divisible by every tile choice "
                        f"{tile_choices} -- cost model unvalidated for remainders")
    dual_buffer = ["true", "false"] if knobs.get("dual_buffer") else ["false"]
    perm = list(range(ndims))
    default_ix = [2, 0, 1] if ndims == 3 else perm
    interchange_choices = knobs.get("l1_tiles_interchange", {}).get("choices") or [default_ix]
    for ix in interchange_choices:
        if sorted(ix) != perm:
            _reject(nm, f"l1_tiles_interchange {ix} is not a permutation of {perm}")

    mlir = d.get("mlir") or ""
    mlir = mlir if os.path.isabs(mlir) else os.path.join(ROOT, mlir)
    if not os.path.isfile(mlir):
        _reject(nm, f"mlir not found: {mlir}")
    _check_mlir_scope(nm, mlir)

    entry = d.get("entry")
    if not entry:
        _reject(nm, "entry (func.func symbol) required")

    return Spec(
        name=nm, mlir=mlir, entry=entry, module=d.get("module", ""),
        op=op, dtype=dtype, shape=shape,
        transpose_b=bool(d.get("transpose_b", False)),
        bindings=d.get("bindings", []), tile_choices=tile_choices,
        dual_buffer=dual_buffer, interchange_choices=interchange_choices,
        baseline_tag=d.get("validation", {}).get("baseline_tag", ""))


if __name__ == "__main__":
    import sys
    s = load_spec(sys.argv[1] if len(sys.argv) > 1 else "gemm_square")
    print(f"name        {s.name}")
    print(f"mlir        {s.mlir}")
    print(f"entry       {s.entry}  -> {s.query_symbol}")
    print(f"op/dtype    {s.op} / {s.dtype} ({s.elem_bytes} B)")
    print(f"shape       M={s.shape[0]} N={s.shape[1]} K={s.shape[2]} transpose_b={s.transpose_b}")
    print(f"bindings    {s.bindings}")
    print(f"tile choices {s.tile_choices}  dual_buffer {s.dual_buffer}")
    print(f"interchange  {s.interchange_choices}")
    print(f"baseline    {s.baseline_tag}")
