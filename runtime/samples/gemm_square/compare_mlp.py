# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Tolerance-check the mlp_harness device output (f64 bit-pattern YDUMP) against the
# torch golden. Each sim log carries its own stride/offset in the YDUMP_BEGIN header;
# several logs (one per offset) combine into full coverage. The offset is TRUSTED from
# the header, never inferred from the data being validated (an oracle that relabels its
# input to fit the golden could mask a systematic miscompile).
# Usage: compare_mlp.py <golden.npy> <sim.log> [more sim.logs ...]
import sys
import re
import numpy as np

args = sys.argv[1:]
golden = None
logs = []
for a in args:
    if a.endswith(".npy"):
        golden = np.load(a).astype("<f8").ravel()
    else:
        logs.append(a)
if golden is None or not logs:
    print("usage: compare_mlp.py <golden.npy> <sim.log> [more...]")
    sys.exit(2)

filled = {}  # index -> device f64 value
for path in logs:
    with open(path) as f:
        log = f.read()
    m = re.search(r"YDUMP_BEGIN stride=(\d+) offset=(\d+)", log)
    if not m:
        print(f"ERROR: no parseable YDUMP_BEGIN header in {path}")
        sys.exit(2)
    stride, offset = int(m.group(1)), int(m.group(2))
    tail = log[m.end():]
    e = tail.find("YDUMP_END")
    if e >= 0:
        tail = tail[:e]
    hexes = [l.strip() for l in tail.splitlines()
             if len(l.strip()) == 16 and all(c in "0123456789abcdef" for c in l.strip())]
    vals = np.array([int(h, 16) for h in hexes], dtype=np.uint64).view("<f8")
    for k, v in enumerate(vals):
        idx = offset + k * stride
        if idx < golden.size:
            filled[idx] = v

idx = sorted(filled)
dev = np.array([filled[i] for i in idx])
gold = golden[idx]
diff = np.abs(dev - gold)
rel = diff / np.maximum(1.0, np.abs(gold))
n_bad = int((rel > 1e-9).sum())
cov = len(idx)
print(f"coverage={cov}/{golden.size} ({100 * cov // golden.size}%) max_abs_err={diff.max():.3e} "
      f"max_rel_err={rel.max():.3e} mismatches(>1e-9 rel)={n_bad} -> {'FAIL' if n_bad else 'SUCCESS'}")
sys.exit(1 if n_bad else 0)
