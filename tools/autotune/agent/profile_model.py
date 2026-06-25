#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Phase-3 profiler glue: a per-dispatch cycle table -> the Amdahl orchestrator's
`dispatches` JSON (orchestrate.py plan).

Parses op_type + shape from IREE dispatch names
(main$async_dispatch_<N>_<op>_<MxNx...>_f<bits>), computes each dispatch's
percent of total COMPUTE cycles, and emits {dispatches:[{file,op_type,shape,
cycles,pct_total}]} descending. This is the deterministic, tested conversion the
orchestrator consumes.

The per-dispatch cycle counts are the INPUT here (a JSON {dispatch_name: cycles}).
Producing them from a whole-model run is the remaining Phase-3 instrumentation:
sim the model, run snitch_cluster/util/trace/gen_trace.py for per-region cycles,
EXCLUDE hart_00000 (the DM core), and group regions by dispatch. That trace
correlation + the model sim are not in this file; this defines + validates the
interface they must produce.
"""
import argparse
import json
import re
import sys

# main$async_dispatch_0_matmul_transpose_b_1x600x600_f64$xdsl_kernel1
DISPATCH_RE = re.compile(r"dispatch_\d+_([a-z_]+?)_(\d+(?:x\d+)+)_f(\d+)")


def parse_dispatch(name):
    """Recover (op_type, shape, dtype) from an IREE dispatch name."""
    m = DISPATCH_RE.search(name)
    if not m:
        return {"op_type": "unknown", "shape": None, "dtype": None}
    return {"op_type": m.group(1),
            "shape": [int(x) for x in m.group(2).split("x")],
            "dtype": "f" + m.group(3)}


def to_profile(cycles_by_dispatch):
    """{dispatch_name: cycles} -> the orchestrator's dispatches JSON, descending."""
    total = sum(cycles_by_dispatch.values()) or 1
    out = []
    for name, cyc in cycles_by_dispatch.items():
        d = parse_dispatch(name)
        out.append({"file": name, "op_type": d["op_type"], "shape": d["shape"],
                    "dtype": d["dtype"], "cycles": cyc,
                    "pct_total": 100.0 * cyc / total})
    out.sort(key=lambda x: x["pct_total"], reverse=True)
    return {"dispatches": out}


def main():
    ap = argparse.ArgumentParser(description="per-dispatch cycle table -> orchestrator profile")
    ap.add_argument("cycle_table", help='JSON {dispatch_name: cycles}')
    ap.add_argument("-o", "--out", default="-", help="output path (default stdout)")
    args = ap.parse_args()
    with open(args.cycle_table) as f:
        cyc = json.load(f)
    prof = to_profile(cyc)
    text = json.dumps(prof, indent=2)
    if args.out == "-":
        print(text)
    else:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {len(prof['dispatches'])} dispatches to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
