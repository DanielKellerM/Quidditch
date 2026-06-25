#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate an agent-AUTHORED xDSL pass -- the capability for the agent to invent
a novel kernel structure by adding a transform to the lowering pipeline, SAFELY.

Registers the authored ModulePass in xdsl's pass registry (copies the file into
xdsl/xdsl/transforms/ + injects one lazy entry into get_all_passes), inserts it
into the --xdsl-passes pipeline, builds + sims + gates the kernel, then ALWAYS
reverts the registry and removes the file (the pinned xdsl submodule is never
left mutated -- mirrors the .mlir inject/restore in the grid).

The same gates that guard config tuning guard authored passes: a pass that breaks
the IR fails to compile (legality gate); a pass that changes semantics is caught
(Tier-1 per-element + Tier-2 independent cross-check). So only a correct authored
structure is reported, never a broken or silently-wrong one. Producing a WINNING
structure is the agent's open exploration; this makes that exploration sound.

  uv run agent/author_pass.py --op gemm_square --pass-file agent/example_passes/agent_noop.py
"""
import argparse
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                       # agent_loop helpers
sys.path.insert(0, os.path.dirname(HERE))      # sweep / direct_build / tier2 / spec
import direct_build
import sweep
import tier2
from agent_loop import _harness_obj, _parse
from spec import load_spec, SpecError

XDSL_TRANSFORMS = os.path.join(sweep.ROOT, "xdsl", "xdsl", "transforms")
INIT = os.path.join(XDSL_TRANSFORMS, "__init__.py")
BASE_PIPELINE = "arith-add-fastmath,test-lower-linalg-to-snitch"


def _scan(src, pat, what):
    m = re.search(pat, src)
    if not m:
        sys.exit(f"authored pass {what}")
    return m.group(1)


def evaluate(op, pass_file, config_str):
    spec = load_spec(op)
    src = open(pass_file).read()
    pname = _scan(src, r'name\s*=\s*"([^"]+)"', 'has no `name = "..."`')
    cname = _scan(src, r"class\s+(\w+)\s*\(\s*ModulePass\s*\)", "has no `class X(ModulePass)`")
    mod = "agent_authored_" + re.sub(r"\W", "_", pname)
    # Insert the authored pass after fastmath, before the big lowering.
    pipeline = BASE_PIPELINE.replace("arith-add-fastmath,",
                                     f"arith-add-fastmath,{pname},")
    init_bak = open(INIT).read()
    dst = os.path.join(XDSL_TRANSFORMS, mod + ".py")
    try:
        shutil.copy(pass_file, dst)
        entry = (f'        "{pname}": lambda: __import__('
                 f'"xdsl.transforms.{mod}", fromlist=["x"]).{cname},\n')
        open(INIT, "w").write(init_bak.replace("    return {\n", "    return {\n" + entry, 1))

        c = _parse(config_str)
        d = tempfile.mkdtemp(dir=sweep.WORK, prefix="authp_")
        elf, err = direct_build.build(
            {"l1_tiles": list(c[:3]), "dual_buffer": c[3], "interchange": list(c[4])},
            d, mlir_template=spec.mlir, module=spec.module,
            harness_obj=_harness_obj(spec), xdsl_passes=pipeline)
        if err:
            print(f"authored pass '{pname}': BUILD REJECTED (legality gate) -- not a valid "
                  f"transform, correctly not kept.\n  {err.strip()[-240:]}")
            return 1
        res = sweep.run_sim(f"authp_{pname}", elf)
        if res is None:
            print(f"authored pass '{pname}': sim produced no metric -> rejected")
            return 1
        cyc, errn, status = res
        if status != "SUCCESS" or errn:
            print(f"authored pass '{pname}': INCORRECT ({errn} Tier-1 errors) -> rejected "
                  "(the pass changed the result)")
            return 1
        t2 = "PASS" if tier2.check(op, c, pipeline) == 0 else "FAIL"
        print(f"authored pass '{pname}': compiles + CORRECT, {cyc} cyc, Tier-2 {t2}. "
              "The pass ran in the pipeline and was evaluated by the sound gates.")
        return 0 if t2 == "PASS" else 1
    finally:
        open(INIT, "w").write(init_bak)
        for p in (dst, dst + "c"):
            if os.path.exists(p):
                os.remove(p)
        shutil.rmtree(os.path.join(XDSL_TRANSFORMS, "__pycache__"), ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="evaluate an agent-authored xDSL pass")
    ap.add_argument("--op", default="gemm_square")
    ap.add_argument("--pass-file", required=True)
    ap.add_argument("--config", default="16,16,16,false")
    a = ap.parse_args()
    return evaluate(a.op, a.pass_file, a.config)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SpecError as e:
        sys.exit(str(e))
