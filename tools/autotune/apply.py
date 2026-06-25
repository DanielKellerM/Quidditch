#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Apply a recorded autotuner win back into the kernel source -- the 'apply' half
of record -> apply -> track (AutoKernel keeps the agent's kernel.py edit in place;
we write best_<name>.json's config into the .mlir's lowering_config, so the win
lands in version-controlled source and the next IREE build runs it).

Scope: HAND-WRITTEN kernels (the lowering_config is an attribute in the .mlir).
GENERATED model kernels carry no inline config -- their tiling lives in the
ConfigureForSnitch.cpp per-dispatch table keyed on the `main$async_dispatch_*`
symbol name, which the tuner does not yet capture; that path is flagged, not done.

The recorded best_<name>.json files ARE the system of record (git-tracked, like
AutoKernel's results.tsv) -- this just deposits the win into the kernel.

  uv run apply.py --op gemm_square --dry-run    # show what would change
  uv run apply.py --op gemm_square              # write it into the .mlir
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spec import load_spec, SpecError

HERE = os.path.dirname(os.path.abspath(__file__))


def apply(op, dry_run=False):
    spec = load_spec(op)
    best_path = os.path.join(HERE, f"best_{op}.json")
    if not os.path.isfile(best_path):
        sys.exit(f"no best_{op}.json -- run sweep.py or agent_loop.py first")
    best = json.load(open(best_path))
    if not best.get("ok", True):
        sys.exit(f"best_{op}.json is not a correct config -- refusing to apply")
    cfg = best["config"]                       # [M, N, K, dual_buffer, [interchange]]
    mt, nt, kt, db = cfg[0], cfg[1], cfg[2], cfg[3]
    ix = cfg[4] if len(cfg) > 4 else [2, 0, 1]

    src = open(spec.mlir).read()
    new = re.sub(r"l1_tiles = \[[0-9, ]+\]", f"l1_tiles = [{mt}, {nt}, {kt}]", src)
    new = re.sub(r"dual_buffer = (?:true|false)", f"dual_buffer = {db}", new)
    new = re.sub(r"l1_tiles_interchange = \[[0-9, ]+\]",
                 f"l1_tiles_interchange = [{ix[0]}, {ix[1]}, {ix[2]}]", new)

    if new == src:
        print(f"{op}: lowering_config already matches best ({best['tag']}, {best['cycles']} cyc)")
        return 0
    rel = os.path.relpath(spec.mlir, os.path.dirname(os.path.dirname(HERE)))
    if dry_run:
        print(f"{op}: would set lowering_config to {best['tag']} "
              f"({best['cycles']} cyc) in {rel}")
        for ln_old, ln_new in zip(src.splitlines(), new.splitlines()):
            if ln_old != ln_new:
                print(f"  - {ln_old.strip()}\n  + {ln_new.strip()}")
        return 0
    with open(spec.mlir, "w") as f:
        f.write(new)
    print(f"{op}: applied best {best['tag']} ({best['cycles']} cyc) to {rel}")
    print("  re-run the IREE build to pick it up; commit the .mlir to deploy the win.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="apply a recorded autotuner win to the kernel .mlir")
    ap.add_argument("--op", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="show the lowering_config change without writing it")
    a = ap.parse_args()
    return apply(a.op, a.dry_run)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SpecError as e:
        sys.exit(str(e))
