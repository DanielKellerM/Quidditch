#!/usr/bin/env python3
# Derived from rightnow-ai/autokernel orchestrate.py (MIT, (c) 2026 RightNow AI;
# full text in LICENSE.autokernel). Snitch/RTL-sim modifications (c) 2026 ETH
# Zurich and University of Bologna, licensed under the Apache License v2.0.
# SPDX-License-Identifier: MIT AND Apache-2.0
"""Amdahl multi-kernel scheduler for the Snitch autotuner.

Tracks per-dispatch optimization across a whole model and decides, by Amdahl's
law + move-on criteria, which dispatch to tune next. Adapted from AutoKernel:
GPU throughput (TFLOPS, higher=better) becomes Snitch dispatch CYCLES (lower=
better), percent-of-peak becomes FPU-utilization FRACTION, and speedup is
baseline_cycles/best_cycles. State lives in workspace/orchestration_state.json;
the deterministic per-dispatch grid (sweep.py) is the tuner this schedules.

CLI: status | next | record | report | plan   (e.g. uv run orchestrate.py next)
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

MOVE_ON = {
    "consecutive_reverts": 5,
    "fpu_util_threshold": 0.90,     # FRACTION (AutoKernel used 90.0 percent)
    "max_minutes_per_kernel": 120,
    "speedup_threshold": 2.0,
}
STATUS_PENDING = "pending"
STATUS_OPTIMIZING = "optimizing"
STATUS_DONE = "done"
STATUS_SKIPPED = "skipped"

STATE = os.environ.get("ORCH_STATE", "workspace/orchestration_state.json")


def _now():
    return datetime.now(timezone.utc).isoformat()


def load_state(path=None):
    with open(path or STATE) as f:
        return json.load(f)


def save_state(state, path=None):
    p = path or STATE
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w") as f:
        json.dump(state, f, indent=2)


def estimate_aggregate_speedup(kernels):
    """Amdahl: S = 1 / (1 - sum_i frac_i * (1 - 1/speedup_i)), frac = pct_total/100."""
    saved = 0.0
    for k in kernels:
        sp = k.get("speedup") or 1.0
        frac = (k.get("pct_total") or 0.0) / 100.0
        saved += frac * (1.0 - 1.0 / sp)
    rem = 1.0 - saved
    return 1.0 / rem if rem > 0 else float("inf")


def _should_move_on(k):
    """Returns (should_move, reason) — Snitch move-on criteria."""
    if k.get("consecutive_reverts", 0) >= MOVE_ON["consecutive_reverts"]:
        return True, (f"Plateau detected: {k['consecutive_reverts']} consecutive reverts "
                      f"(threshold: {MOVE_ON['consecutive_reverts']})")
    fu = k.get("pct_peak_fpu")
    if fu is not None and fu >= MOVE_ON["fpu_util_threshold"]:
        return True, (f"Near FPU-util ceiling: {fu:.2f} "
                      f"(threshold: {MOVE_ON['fpu_util_threshold']:.2f})")
    if k.get("time_spent_minutes", 0) >= MOVE_ON["max_minutes_per_kernel"]:
        return True, (f"Time budget exhausted: {k['time_spent_minutes']:.0f} min "
                      f"(max: {MOVE_ON['max_minutes_per_kernel']} min)")
    sp = k.get("speedup")
    if sp is not None and sp >= MOVE_ON["speedup_threshold"]:
        return True, (f"Strong speedup achieved: {sp:.2f}x "
                      f"(threshold: {MOVE_ON['speedup_threshold']:.1f}x)")
    return False, "Current dispatch still has optimization headroom"


def _find_next_pending(kernels, current_idx):
    for i in range(current_idx + 1, len(kernels)):
        if kernels[i]["status"] == STATUS_PENDING:
            return i
    for i, k in enumerate(kernels):
        if k["status"] == STATUS_PENDING:
            return i
    return None


def _transition_to(state, next_idx):
    state["current_kernel_idx"] = next_idx
    state["current_kernel_file"] = state["kernels"][next_idx]["file"]
    state["kernels"][next_idx]["status"] = STATUS_OPTIMIZING


def cmd_next(state):
    kernels = state["kernels"]
    idx = state.get("current_kernel_idx", -1)
    cur = kernels[idx] if 0 <= idx < len(kernels) else None
    if cur and cur["status"] == STATUS_OPTIMIZING:
        move, reason = _should_move_on(cur)
        if not move:
            print(f"DECISION: Continue optimizing {cur['file']}")
            print(f"  Reason: {reason}")
            return 0
        cur["status"] = STATUS_DONE
    nxt = _find_next_pending(kernels, idx)
    if nxt is None:
        print("DECISION: All dispatches done or skipped")
        print(f"  Aggregate Amdahl speedup: {estimate_aggregate_speedup(kernels):.3f}x")
        return 0
    _transition_to(state, nxt)
    k = kernels[nxt]
    print(f"DECISION: Move to {k['file']} (rank {k['rank']}, {k['op_type']})")
    print(f"  Reason: highest remaining cycle share ({k.get('pct_total', 0):.1f}% of total)")
    print(f"  File: {k['file']}")
    return 0


def cmd_record(state, kernel_file, best_cycles, status, description):
    k = next((x for x in state["kernels"] if x["file"] == kernel_file), None)
    if k is None:
        sys.exit(f"no dispatch with file={kernel_file}")
    k["experiments_run"] = k.get("experiments_run", 0) + 1
    norm = status.lower()
    if norm in ("kept", "keep", "improved"):
        k["experiments_kept"] = k.get("experiments_kept", 0) + 1
        k["consecutive_reverts"] = 0
        if k.get("baseline_cycles") is None:
            k["baseline_cycles"] = best_cycles
        if k.get("best_cycles") is None or best_cycles < k["best_cycles"]:
            k["best_cycles"] = best_cycles
            if k.get("baseline_cycles"):
                k["speedup"] = k["baseline_cycles"] / best_cycles
    elif norm in ("revert", "reverted", "slower", "same"):
        k["consecutive_reverts"] = k.get("consecutive_reverts", 0) + 1
    # failed/fail/crash/error/timeout: counted in experiments_run, no metric change
    print(f"recorded {status} for {kernel_file}: best_cycles={k.get('best_cycles')} "
          f"speedup={k.get('speedup')} reverts={k.get('consecutive_reverts', 0)}")


def cmd_status(state):
    print(f"current_kernel_idx={state.get('current_kernel_idx')}  started={state.get('started_at')}")
    for k in state["kernels"]:
        print(f"  rank {k['rank']:>2} {k['file']:24} {k['op_type']:18} "
              f"{k.get('pct_total', 0):5.1f}%  {k['status']:11} "
              f"best={k.get('best_cycles')} speedup={k.get('speedup')}")
    print(f"aggregate Amdahl speedup: {estimate_aggregate_speedup(state['kernels']):.3f}x")


def cmd_plan(profile_path):
    """Seed orchestration_state.json from a per-dispatch profile (list of
    {file, op_type, pct_total}, descending pct_total)."""
    with open(profile_path) as f:
        prof = json.load(f)
    dispatches = prof["dispatches"] if isinstance(prof, dict) else prof
    dispatches = sorted(dispatches, key=lambda d: d.get("pct_total", 0), reverse=True)
    kernels = [{
        "rank": i, "file": d["file"], "op_type": d.get("op_type", "unknown"),
        "pct_total": d.get("pct_total", 0.0), "status": STATUS_PENDING,
        "baseline_cycles": None, "best_cycles": None, "speedup": None,
        "pct_peak_fpu": None, "experiments_run": 0, "experiments_kept": 0,
        "consecutive_reverts": 0, "time_spent_minutes": 0,
    } for i, d in enumerate(dispatches)]
    state = {"current_kernel_idx": -1, "current_kernel_file": None,
             "started_at": _now(), "kernels": kernels}
    save_state(state)
    print(f"planned {len(kernels)} dispatches into {STATE}")


def build_parser():
    ap = argparse.ArgumentParser(description="Snitch autotuner Amdahl orchestrator")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("next")
    sub.add_parser("report")
    rec = sub.add_parser("record")
    rec.add_argument("kernel_file")
    rec.add_argument("best_cycles", type=int)
    rec.add_argument("status")
    rec.add_argument("description")
    pl = sub.add_parser("plan")
    pl.add_argument("profile", help="per-dispatch profile JSON")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "plan":
        cmd_plan(args.profile)
        return 0
    state = load_state()
    if args.cmd in ("status", "report"):
        cmd_status(state)
        return 0
    if args.cmd == "next":
        rc = cmd_next(state)
        save_state(state)
        return rc
    if args.cmd == "record":
        cmd_record(state, args.kernel_file, args.best_cycles, args.status, args.description)
        save_state(state)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
