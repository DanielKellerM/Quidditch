#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Snitch kernel autotuner (v1): cost-model-ranked, parallel, ninja-free.

Pipeline: load the op-spec (spec.py) -> enumerate its l1_tiles x dual_buffer
grid -> rank by the analytic cost model (cost_model.py) -> build+sim each config
through a parallel pipeline (direct_build.py, no CMake/ninja, isolated tmpdirs
so builds parallelize too) -> keep the best by measured dispatch_cycles ->
best_<name>.json. The kernel is selected with --op <name>; every gemm_square
constant that used to live here now comes from ops/<name>.toml.

Two facts from the build-speed analysis shape this:
 - It is SIM-bound (~16.5s sim vs ~1.1s build), so the pipeline fans sims out
   N-way; the direct build removes the shared-build-rt serialization.
 - The cost model is a validated RANKER (Spearman 1.0 on 9 configs) but is blind
   to Group-B knobs, so it only SEQUENCES (sim promising configs first); it never
   structurally prunes the optimum. Default sims everything; --budget K sims the
   top-K legal configs in rank order. v1 specs are divisor-only by load contract
   (spec.py rejects non-divisor tiles the model cannot rank), so ranked-order is
   always safe.

A per-run dual-build CANARY (direct vs ninja, stripped-md5) makes the N=1 build
equivalence a live invariant and catches direct_build recipe drift. The canary
only exists for gemm_square (the kernel with a committed CMake harness target);
for other ops it is skipped with a warning.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cost_model import cost_model
from direct_build import build as direct_build, compile_harness
from gen_harness import generate as gen_harness_src
from spec import load_spec, SpecError

ROOT = "/home/dankeller/Projects/Quidditch"
NINJA = f"{ROOT}/venv/bin/ninja"          # 1.13 (the non-dot venv has it)
BUILD = f"{ROOT}/build-rt"
SIM = f"{ROOT}/snitch_cluster/target/sim"
VLT = f"{SIM}/build/bin/snitch_cluster.vlt"
LLVM_STRIP = "/usr/scratch2/vulcano/colluca/tools/riscv32-snitch-llvm-almalinux8-15.0.0-snitch-0.5.0/bin/llvm-strip"
CCACHE = "/scratch/dankeller/snitch-compiler/.ccache"
WORK = "/scratch/dankeller/snitch-compiler/autotune-work"

SIM_WORKERS = 16
HARNESS_RE = re.compile(r"dispatch_cycles=(\d+).*?errors=(\d+)/\d+\s*->\s*(\w+)")


def tag_of(c):
    base = f"{c[0]}x{c[1]}x{c[2]}_db{c[3]}"
    ix = tuple(c[4]) if len(c) > 4 else (2, 0, 1)
    return base if ix == (2, 0, 1) else f"{base}_i{ix[0]}{ix[1]}{ix[2]}"


def enumerate_grid(spec):
    return [(mt, nt, kt, db, tuple(ix))
            for mt in spec.tile_choices for nt in spec.tile_choices
            for kt in spec.tile_choices for db in spec.dual_buffer
            for ix in spec.interchange_choices]


def rank_grid(grid, spec):
    # cost_model is blind to interchange (KSPLIT/MSPLIT use tile counts only), so
    # interchange variants of a tile tuple rank together and are all simmed.
    return sorted(grid, key=lambda c: cost_model([c[0], c[1], c[2]], c[3], spec.shape))


def _strip_md5(elf):
    s = elf + ".stripped"
    shutil.copy(elf, s)
    subprocess.run([LLVM_STRIP, s], check=True)
    return hashlib.md5(open(s, "rb").read()).hexdigest()


def ninja_build(c, outdir, spec, ninja_elf):
    """Build one config via the real CMake/ninja path (only for the canary)."""
    orig = open(spec.mlir).read()
    try:
        s = re.sub(r"l1_tiles = \[[0-9, ]+\]", f"l1_tiles = [{c[0]}, {c[1]}, {c[2]}]", orig)
        s = re.sub(r"dual_buffer = (?:true|false)", f"dual_buffer = {c[3]}", s)
        open(spec.mlir, "w").write(s)
        env = dict(os.environ, CCACHE_DIR=CCACHE)
        r = subprocess.run([NINJA, "-C", BUILD, "gemm_harness"], env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True)
        if r.returncode != 0 or "error:" in r.stderr:
            return None
        dst = os.path.join(outdir, "ninja_elf")
        shutil.copy(ninja_elf, dst)
        return dst
    finally:
        open(spec.mlir, "w").write(orig)


def canary_check(spec, canary, ninja_elf):
    """Build CANARY via direct AND ninja; assert stripped-md5 equal. Abort on drift.
    Only gemm_square has a CMake harness target to diff against."""
    if spec.name != "gemm_square":
        print(f"canary SKIP: {spec.name} has no committed CMake harness target; "
              "direct-vs-ninja equivalence is only checkable for gemm_square", flush=True)
        return
    d = tempfile.mkdtemp(dir=WORK, prefix="canary_")
    elf_direct, err = direct_build({"l1_tiles": list(canary[:3]), "dual_buffer": canary[3],
                                    "interchange": list(canary[4])},
                                   d, mlir_template=spec.mlir, module=spec.module)
    if err:
        sys.exit(f"CANARY ABORT: direct build of {tag_of(canary)} failed:\n{err}")
    elf_ninja = ninja_build(canary, d, spec, ninja_elf)
    if elf_ninja is None:
        sys.exit("CANARY ABORT: ninja build of the canary config failed")
    if _strip_md5(elf_direct) != _strip_md5(elf_ninja):
        sys.exit("CANARY ABORT: direct vs ninja ELF diverged after strip -- direct_build "
                 "recipe drift; re-extract via `ninja -t commands gemm_harness`.")
    print("canary OK: direct build is stripped-identical to ninja", flush=True)


def run_sim(tag, elf):
    rd = f"{SIM}/runs/autotune_{tag}"
    shutil.rmtree(rd, ignore_errors=True)
    os.makedirs(f"{rd}/logs", exist_ok=True)
    for i in range(9):
        try:
            os.symlink("/dev/null", f"{rd}/logs/trace_hart_0000{i}.dasm")
        except OSError:
            pass
    try:
        r = subprocess.run(["timeout", "180", VLT, elf], cwd=rd,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, timeout=240)
    except subprocess.TimeoutExpired:
        return None
    m = HARNESS_RE.search(r.stdout)
    return (int(m.group(1)), int(m.group(2)), m.group(3)) if m else None


def pipeline(c, spec, harness_obj, xdsl_passes=None):
    """Full build->sim for one config in an isolated dir (parallel-safe).
    xdsl_passes overrides the xdsl-opt pass pipeline (Group-B knob); None = default."""
    tag = tag_of(c)
    outdir = tempfile.mkdtemp(dir=WORK, prefix=f"{tag}_")
    cfg = {"l1_tiles": list(c[:3]), "dual_buffer": c[3], "interchange": list(c[4])}
    elf, err = direct_build(cfg, outdir, mlir_template=spec.mlir, module=spec.module,
                            harness_obj=harness_obj, xdsl_passes=xdsl_passes)
    if err is not None:          # stderr 'error:' legality gate fired
        return {"tag": tag, "config": c, "legal": False}
    res = run_sim(tag, elf)
    if res is None:
        return {"tag": tag, "config": c, "legal": True, "ok": False}
    cyc, errn, status = res
    return {"tag": tag, "config": c, "legal": True,
            "ok": (status == "SUCCESS" and errn == 0), "cycles": cyc, "errors": errn}


def run_budget(ranked, budget, spec, harness_obj):
    """Submit pipelines in cost-rank order, N-way parallel, until `budget` legal
    results complete. The full ranked list is the queue, so the optimum is never
    excluded; budget>=len(ranked) sims everything."""
    queue = list(ranked)
    results, legal_done = [], 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=SIM_WORKERS) as ex:
        futs = {}
        while queue and len(futs) < min(SIM_WORKERS, budget):
            futs[ex.submit(pipeline, queue.pop(0), spec, harness_obj)] = 1
        while futs:
            done = next(concurrent.futures.as_completed(list(futs)))
            del futs[done]
            r = done.result()
            results.append(r)
            if r["legal"]:
                legal_done += 1
                msg = f"cycles={r['cycles']:<6} SUCCESS" if r.get("ok") else "RUN-FAIL"
                print(f"  {r['tag']:22} {msg}", flush=True)
            else:
                print(f"  {r['tag']:22} ILLEGAL (rejected)", flush=True)
            if queue and (legal_done + len(futs)) < budget:
                futs[ex.submit(pipeline, queue.pop(0), spec, harness_obj)] = 1
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="gemm_square",
                    help="op-spec name under ops/ (default gemm_square)")
    ap.add_argument("--budget", type=int, default=-1,
                    help="sim this many LEGAL configs in cost-rank order; -1 = all (default)")
    args = ap.parse_args()

    try:
        spec = load_spec(args.op)
    except SpecError as e:
        sys.exit(str(e))
    db_canary = "true" if "true" in spec.dual_buffer else "false"
    canary = (max(spec.tile_choices),) * 3 + (db_canary, (2, 0, 1))
    ninja_elf = f"{BUILD}/samples/{spec.name}/gemm_harness"
    baseline_tag = spec.baseline_tag
    out_tsv = f"{ROOT}/tools/autotune/results_{spec.name}.tsv"
    out_best = f"{ROOT}/tools/autotune/best_{spec.name}.json"

    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK, exist_ok=True)
    canary_check(spec, canary, ninja_elf)

    # Harness object: gemm_square reuses the prebuilt CMake object so its canary
    # stays byte-identical; any other op compiles its harness.o once (config-
    # invariant) and shares it across the parallel per-config builds.
    harness_obj = None
    if spec.name != "gemm_square":
        try:
            src = gen_harness_src(spec)
        except SpecError as e:
            sys.exit(str(e))
        hc = os.path.join(WORK, f"{spec.name}_harness.c")
        with open(hc, "w") as f:
            f.write(src)
        harness_obj, herr = compile_harness(hc, spec.module, spec.query_symbol,
                                            os.path.join(WORK, f"{spec.name}_harness.o"))
        if herr:
            sys.exit(f"harness compile failed for {spec.name}:\n{herr}")

    grid = enumerate_grid(spec)
    ranked = rank_grid(grid, spec)
    budget = len(ranked) if args.budget < 0 else min(args.budget, len(ranked))
    print(f"op={spec.name} grid={len(grid)} configs, ranked by cost model, "
          f"sim budget={budget} ({SIM_WORKERS}-way parallel)\n", flush=True)

    results = run_budget(ranked, budget, spec, harness_obj)

    base = next((r["cycles"] for r in results if r["tag"] == baseline_tag and r.get("ok")), None)
    ok = [r for r in results if r.get("ok")]
    ok.sort(key=lambda r: r["cycles"])
    print(f"\n=== {len(ok)} correct configs, best-first"
          + (f" (vs {baseline_tag}={base})" if base else "") + " ===")
    with open(out_tsv, "w") as f:
        f.write("kernel\ttag\tlegal\tcorrect\tcycles\tspeedup_vs_base\tcost_rank\n")
        rank_idx = {tag_of(c): i for i, c in enumerate(ranked)}
        for r in sorted(results, key=lambda r: r.get("cycles") or 1 << 30):
            sp = f"{base / r['cycles']:.3f}" if (r.get("ok") and base) else ""
            if r.get("ok") and base:
                print(f"  {r['tag']:22} {r['cycles']:<6} {base / r['cycles']:.2f}x")
            f.write(f"{spec.name}\t{r['tag']}\t{r.get('legal')}\t{r.get('ok')}\t"
                    f"{r.get('cycles','')}\t{sp}\t{rank_idx.get(r['tag'],'')}\n")
    if ok:
        best = dict(ok[0], kernel=spec.name)
        json.dump(best, open(out_best, "w"), indent=2)
        print(f"\nBEST: {best['tag']} = {best['cycles']} cyc"
              + (f" ({base / best['cycles']:.2f}x vs baseline)" if base else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
