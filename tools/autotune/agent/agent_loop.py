#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Loop STRUCTURE (propose -> evaluate -> keep/revert + results log) follows
# AutoKernel's bench.py contract (MIT, see LICENSE.autokernel); this code is ours.
"""LLM-agent search loop for the Snitch autotuner.

The agent (an LLM following agent/program.md) is the search POLICY: it proposes
a config; THIS harness evaluates it through the trusted oracle + correctness
gates and keeps it iff it is correct AND faster than the best so far, else
reverts. The cost model pre-ranks (instant), the RTL sim measures (~18 s), and
the Tier-1 + Tier-2 gates guarantee a faster-but-WRONG config is reverted, never
kept. State (best + full history) persists in workspace/agent_<op>.json.

  uv run agent/agent_loop.py --op gemm_tall --propose "16,16,16,true"
  uv run agent/agent_loop.py --op gemm_tall --status
"""
import argparse
import json
import os
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import direct_build
import gen_harness
import sweep
import tier2
from cost_model import cost_model
from spec import load_spec, SpecError


def _state_path(op):
    return os.path.join(sweep.WORK, f"agent_{op}.json")


def _load(op):
    p = _state_path(op)
    return json.load(open(p)) if os.path.isfile(p) else {"op": op, "best": None, "history": []}


def _save(op, s):
    os.makedirs(sweep.WORK, exist_ok=True)
    with open(_state_path(op), "w") as f:
        json.dump(s, f, indent=2)


def _harness_obj(spec):
    """gemm_square reuses the prebuilt CMake object; others compile once + cache."""
    if spec.name == "gemm_square":
        return None
    obj = os.path.join(sweep.WORK, f"{spec.name}_agent.o")
    if os.path.isfile(obj):
        return obj
    os.makedirs(sweep.WORK, exist_ok=True)
    hc = os.path.join(sweep.WORK, f"{spec.name}_agent_harness.c")
    with open(hc, "w") as f:
        f.write(gen_harness.generate(spec))
    out, err = direct_build.compile_harness(hc, spec.module, spec.query_symbol, obj)
    if err:
        sys.exit(f"harness compile failed:\n{err}")
    return out


def _parse(cfg):
    p = cfg.split(",")
    c = [int(p[0]), int(p[1]), int(p[2]), p[3]]
    c.append(tuple(int(x) for x in p[4]) if len(p) > 4 else (2, 0, 1))
    return tuple(c)


def propose(op, cfg_str, passes=None):
    spec = load_spec(op)
    c = _parse(cfg_str)
    tag = sweep.tag_of(c) + ("" if not passes else f"+p{zlib.crc32(passes.encode()) % 10000:04d}")
    s = _load(op)
    est = round(cost_model([c[0], c[1], c[2]], c[3], spec.shape))
    r = sweep.pipeline(c, spec, _harness_obj(spec), xdsl_passes=passes)
    legal, correct, cycles = r.get("legal"), r.get("ok", False), r.get("cycles")
    kept, note = False, ""
    if not legal:
        note = "ILLEGAL (rejected at compile)"
    elif not correct:
        note = "INCORRECT (Tier-1 FAIL) -> revert"
    elif s["best"] is not None and cycles >= s["best"]["cycles"]:
        note = f"slower than best {s['best']['cycles']} -> revert"
    elif tier2.check(op, c, passes) != 0:         # independent promotion gate (same pipeline)
        note = "Tier-2 cross-check FAIL -> revert (not promoted)"
    else:
        s["best"] = {"tag": tag, "config": list(c[:4]) + [list(c[4])], "cycles": cycles}
        kept, note = True, f"KEPT (new best {cycles} cyc)"
    s["history"].append({"tag": tag, "cost_est": est, "cycles": cycles,
                         "correct": correct, "legal": legal, "kept": kept})
    _save(op, s)
    print(f"{tag}: cost_est={est} cycles={cycles} correct={correct} -> {note}")
    if s["best"]:
        print(f"  best so far: {s['best']['tag']} = {s['best']['cycles']} cyc")
    return 0


def status(op):
    s = _load(op)
    print(f"op={op}  best={s['best']}  ({len(s['history'])} experiments)")
    for h in s["history"]:
        print(f"  {h['tag']:24} est={h['cost_est']:<6} cyc={h['cycles']} "
              f"correct={h['correct']} kept={h['kept']}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="LLM-agent propose/evaluate/keep-revert loop")
    ap.add_argument("--op", required=True)
    ap.add_argument("--propose", help="M,N,K,dual_buffer[,i0i1i2]")
    ap.add_argument("--passes", help="xdsl-opt pass pipeline override (Group-B knob)")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()
    if a.reset:
        if os.path.exists(_state_path(a.op)):
            os.remove(_state_path(a.op))
        print(f"reset {a.op}")
        return 0
    if a.status:
        return status(a.op)
    if a.propose:
        return propose(a.op, a.propose, a.passes)
    ap.error("need --propose, --status, or --reset")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SpecError as e:
        sys.exit(str(e))
