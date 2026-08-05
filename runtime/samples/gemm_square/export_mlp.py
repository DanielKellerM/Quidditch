# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Export a tiny PyTorch MLP to StableHLO for the Quidditch front-door, and emit the
# numerical reference the single-cluster harness consumes: the torch weights as f64
# bit patterns (mlp_ref_bits.h, seeded by integer stores since the DM core has no fp),
# the @main-binding-order inputs (mlp_in{0..4}_*.npy, for the iree-run-module CPU
# check), and the torch golden (mlp_golden.npy). Deterministic (manual_seed) so the
# committed reference is regenerable byte-for-byte. Runs in the apptainer
# python:3.11-slim container (modern glibc for torch_xla); OUT defaults to this dir.
import os
import struct
import torch
import numpy as np
from torch_xla.stablehlo import exported_program_to_stablehlo

torch.manual_seed(0)
N = 16
OUT = os.environ.get("MLP_OUT", os.path.dirname(os.path.abspath(__file__)))


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(N, N)
        self.l2 = torch.nn.Linear(N, N)

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))


m = MLP().eval().double()
x = torch.randn(N, N, dtype=torch.float64)
with torch.no_grad():
    golden = m(x)

ep = torch.export.export(m, (x,))
shlo = exported_program_to_stablehlo(ep)
with open(f"{OUT}/torch_mlp.stablehlo.mlir", "w") as f:
    f.write(shlo.get_stablehlo_text())

# @main binding order (from the exported signature): (l2.bias, l2.weight, l1.bias, l1.weight, x).
bindings = [("b2", m.l2.bias), ("W2", m.l2.weight), ("b1", m.l1.bias), ("W1", m.l1.weight), ("x", x)]
# The harness seeds by role name (x, w1, b1, w2, b2), matching mlp_harness.c's memcpy order.
weights = {"x": x, "w1": m.l1.weight, "b1": m.l1.bias, "w2": m.l2.weight, "b2": m.l2.bias}


def bits_arr(name, t):
    a = t.detach().numpy().astype("<f8").ravel()
    body = ", ".join("0x%016xull" % struct.unpack("<Q", struct.pack("<d", float(v)))[0] for v in a)
    return f"static const uint64_t {name}_bits[{a.size}] = {{{body}}};"


with open(f"{OUT}/mlp_ref_bits.h", "w") as f:
    f.write("// Real torch MLP weights as f64 bit patterns (integer-store seeding, no DM-core fp).\n")
    for nm, t in weights.items():
        f.write(bits_arr(nm, t) + "\n")

for i, (nm, t) in enumerate(bindings):
    np.save(f"{OUT}/mlp_in{i}_{nm}.npy", t.detach().numpy().astype("<f8"))
np.save(f"{OUT}/mlp_golden.npy", golden.detach().numpy().astype("<f8"))

# Self-consistency gate: the emitted bit-pattern weights must recompute the golden the
# device is checked against, so the committed reference can never drift into an unmatched pair.
w = {k: v.detach().numpy().astype("<f8") for k, v in weights.items()}
ref = np.maximum(w["x"] @ w["w1"].T + w["b1"], 0.0) @ w["w2"].T + w["b2"]
assert np.allclose(ref, golden.detach().numpy(), atol=1e-12), "bits do not recompute the golden"

print(f"wrote {OUT}/torch_mlp.stablehlo.mlir + mlp_ref_bits.h + mlp_in{{0..4}}_*.npy + mlp_golden.npy")
print("golden[0,:4] =", golden.detach().numpy().ravel()[:4])
