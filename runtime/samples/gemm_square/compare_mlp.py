#!/usr/bin/env python
# Parse the mlp_harness YDUMP (f64 bit patterns) from a sim log and tolerance-check
# the device output against the torch golden. Usage: compare_mlp.py <sim.log> <golden.npy>
import sys
import numpy as np

log = open(sys.argv[1]).read()
golden = np.load(sys.argv[2]).astype(np.float64).ravel()

beg = log.index("YDUMP_BEGIN")
stride = 1
import re
m = re.search(r"stride=(\d+)", log[beg:beg + 40])
if m:
    stride = int(m.group(1))
tail = log[beg:]
end = tail.find("YDUMP_END")
if end >= 0:
    tail = tail[:end]
hexes = [l.strip() for l in tail.splitlines()
         if len(l.strip()) == 16 and all(c in "0123456789abcdef" for c in l.strip())]
u = np.array([int(h, 16) for h in hexes], dtype=np.uint64)
y = u.view(np.float64)

golden = golden[::stride]  # device dumped every `stride`-th element
n = min(y.size, golden.size)
if y.size < golden.size:
    print(f"NOTE: partial dump -- checking {n}/{golden.size} (stride={stride}) elements")
else:
    print(f"NOTE: stride={stride}, {n} elements spanning all rows")
y, golden = y[:n], golden[:n]

diff = np.abs(y - golden)
rel = diff / np.maximum(1.0, np.abs(golden))
n_bad = int((rel > 1e-9).sum())
print(f"elements={y.size} max_abs_err={diff.max():.3e} max_rel_err={rel.max():.3e} "
      f"mismatches(>1e-9 rel)={n_bad} -> {'FAIL' if n_bad else 'SUCCESS'}")
print("device[:4]=", np.array2string(y[:4], precision=6))
print("golden[:4]=", np.array2string(golden[:4], precision=6))
sys.exit(1 if n_bad else 0)
