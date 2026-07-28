#!/usr/bin/env python3
"""Convert an autotuner result (best_<name>.json) into a ConfigureForSnitch config
table entry, keyed by the dispatch symbol.

The autotuner tunes a baked `lowering_config` by rewriting the kernel MLIR
(sweep.py). The StableHLO front-door carries no such attribute -- its tiling is
selected by ConfigureForSnitch from a config table (--iree-quidditch-config-table),
keyed by the dispatch symbol. This bridges the two: it reads a tuned result and
emits/merges the equivalent table entry, so a tuned tiling feeds the StableHLO
path too. The dispatch key is derived by compiling the kernel to
executable-sources and reading the emitted symbol (robust to IREE's naming).

Usage:
  emit_config_table.py --best best_gemm_square.json --mlir <kernel.mlir> \
      --out gemm_square_config.json
Env (required): QUIDDITCH_IREE_COMPILE, QUIDDITCH_TOOLCHAIN_ROOT, QUIDDITCH_CFG_HEADER.
"""
import argparse
import json
import os
import re
import subprocess
import sys


def _env(name):
    v = os.environ.get(name)
    if not v:
        sys.exit(f"error: set {name}")
    return v


def dispatch_symbol(mlir_path):
    """The dispatch func symbol IREE emits for the kernel (the config-table key)."""
    out = subprocess.run(
        [
            _env("QUIDDITCH_IREE_COMPILE"),
            "--iree-input-type=auto",
            "--iree-input-demote-f64-to-f32=0",
            "--iree-hal-target-backends=quidditch",
            f"--iree-quidditch-toolchain-root={_env('QUIDDITCH_TOOLCHAIN_ROOT')}",
            f"--iree-quidditch-cluster-cfg-header={_env('QUIDDITCH_CFG_HEADER')}",
            "--compile-to=executable-sources",
            mlir_path,
        ],
        capture_output=True,
        text=True,
    )
    names = set(re.findall(r"@(\w+_dispatch_\d+_\w+)", out.stdout + out.stderr))
    if not names:
        sys.exit(f"error: no dispatch symbol found for {mlir_path}\n{out.stderr[:400]}")
    # The inner dispatch func carries the op+shape suffix; it is the longest.
    return max(names, key=len)


def entry_from_best(best):
    """best['config'] is [l1_tiles, dual_buffer, interchange] (dual_buffer a bool
    or its string). ConfigureForSnitch reads l1_tiles / l1_tiles_interchange /
    dual_buffer (and workgroup_tiles, which the v1 autotuner does not tune)."""
    l1, db, ix = best["config"]
    return {
        "l1_tiles": list(l1),
        "l1_tiles_interchange": list(ix),
        "dual_buffer": str(db).lower() == "true",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", required=True, help="autotuner best_<name>.json")
    ap.add_argument("--mlir", required=True, help="the kernel .mlir (for the dispatch key)")
    ap.add_argument("--out", required=True, help="config table to write or merge into")
    args = ap.parse_args()

    with open(args.best) as f:
        best = json.load(f)
    key = dispatch_symbol(args.mlir)
    entry = entry_from_best(best)

    table = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            table = json.load(f)
    table[key] = entry
    with open(args.out, "w") as f:
        json.dump(table, f, indent=2)
        f.write("\n")
    print(f"wrote {key} -> {entry} into {args.out}")


if __name__ == "__main__":
    main()
