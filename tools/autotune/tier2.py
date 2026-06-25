#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Tier-2 correctness: independent host-side cross-check of a kernel's output.

Builds the harness with -DHARNESS_DUMP_OUTPUT (the DM core emits each output
element's raw bytes -- no fp on the data-mover), sims one config, reconstructs
the f64 output on the host, and compares it against a host reference computed
OFF the Snitch toolchain. For the integer Tier-1 inputs the match is EXACT; this
is the independent cross-check the on-device gate cannot be, and the foundation
for fp-tolerance (real-fp inputs + a measured drift band) in a later step.
"""
import argparse
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import direct_build
import gen_harness
import sweep
from spec import load_spec, SpecError

OUTHASH_RE = re.compile(r"\[OUTHASH\] (\d+) ([0-9a-f]{8})")


def fnv32(data):
    """FNV-1a 32-bit, matching the harness's bytewise hash of the output."""
    h = 2166136261
    for x in data:
        h ^= x
        h = (h * 16777619) & 0xffffffff
    return h


def matmul_reference(spec):
    """Host C[i,j] = sum_k A[i,k]*B[j,k] for the harness's structured integer
    inputs (A[i,k]=i*K+k+1, B value=k*N+j+1). Flat row-major, exact integers."""
    M, N, K = spec.shape
    A = [[i * K + k + 1 for k in range(K)] for i in range(M)]
    B = [[k * N + j + 1 for k in range(K)] for j in range(N)]
    return [float(sum(A[i][k] * B[j][k] for k in range(K)))
            for i in range(M) for j in range(N)]


def _sim_stdout(tag, elf):
    rd = f"{sweep.SIM}/runs/tier2_{tag}"
    shutil.rmtree(rd, ignore_errors=True)
    os.makedirs(f"{rd}/logs", exist_ok=True)
    for i in range(9):
        try:
            os.symlink("/dev/null", f"{rd}/logs/trace_hart_0000{i}.dasm")
        except OSError:
            pass
    r = subprocess.run(["timeout", "180", sweep.VLT, elf], cwd=rd,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True, timeout=240)
    return r.stdout


def check(op, config, xdsl_passes=None):
    spec = load_spec(op)
    if spec.op not in ("matmul", "matmul_transpose_b"):
        raise SpecError(f"[{spec.name}] Tier-2 reference only defined for matmul (op={spec.op})")
    os.makedirs(sweep.WORK, exist_ok=True)
    hc = os.path.join(sweep.WORK, f"{spec.name}_t2_harness.c")
    with open(hc, "w") as f:
        f.write(gen_harness.generate(spec))
    obj, err = direct_build.compile_harness(
        hc, spec.module, spec.query_symbol,
        os.path.join(sweep.WORK, f"{spec.name}_t2.o"), defines=("-DHARNESS_DUMP_OUTPUT",))
    if err:
        sys.exit(f"harness compile failed:\n{err}")
    ix = list(config[4]) if len(config) > 4 else [2, 0, 1]
    # tier2 is matmul-only (3D); build the nested (tiles, db, ix) repr sweep.tag_of expects.
    tag = sweep.tag_of((tuple(config[:3]), config[3], tuple(ix)), spec)
    d = tempfile.mkdtemp(dir=sweep.WORK, prefix="t2_")
    elf, err = direct_build.build(
        {"l1_tiles": list(config[:3]), "dual_buffer": config[3], "interchange": ix},
        d, mlir_template=spec.mlir, module=spec.module, harness_obj=obj,
        xdsl_passes=xdsl_passes)
    if err:
        sys.exit(f"build failed:\n{err}")
    M, N, _ = spec.shape
    m = OUTHASH_RE.search(_sim_stdout(tag, elf))
    if not m:
        sys.exit("Tier-2 FAIL: harness emitted no [OUTHASH] line")
    got_count, got_hash = int(m.group(1)), int(m.group(2), 16)
    if got_count != M * N:
        sys.exit(f"Tier-2 FAIL: hash over {got_count} elements, expected {M * N}")
    expect = fnv32(b"".join(struct.pack("<d", v) for v in matmul_reference(spec)))
    tag = sweep.tag_of(config)
    if got_hash != expect:
        print(f"Tier-2 {spec.name} {tag}: FAIL -- output hash {got_hash:08x} "
              f"!= independent reference {expect:08x}")
        return 1
    print(f"Tier-2 {spec.name} {tag}: PASS -- output hash {got_hash:08x} matches the "
          f"independent host reference ({M * N} elements)")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Tier-2 independent host cross-check")
    ap.add_argument("--op", default="gemm_square")
    ap.add_argument("--config", default="16,16,16,false",
                    help="M,N,K,dual_buffer[,i0i1i2] e.g. 16,16,16,false or 8,8,8,true,201")
    args = ap.parse_args()
    p = args.config.split(",")
    cfg = [int(p[0]), int(p[1]), int(p[2]), p[3]]
    if len(p) > 4:
        cfg.append(tuple(int(c) for c in p[4]))
    return check(args.op, tuple(cfg))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SpecError as e:
        sys.exit(str(e))
