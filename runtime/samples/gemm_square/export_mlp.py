#!/usr/bin/env python
# Export a tiny PyTorch MLP to StableHLO for the Quidditch front-door, and dump a
# numerical reference (fixed inputs in @main binding order + torch golden output) so
# the compiled gwaihir object can be validated on the single-cluster harness.
# Runs inside the apptainer python:3.11-slim container (modern glibc for torch_xla).
import torch, numpy as np
from torch_xla.stablehlo import exported_program_to_stablehlo

torch.manual_seed(0)
N = 16
OUT = "/scratch/dankeller"


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
text = shlo.get_stablehlo_text()
open(f"{OUT}/torch_mlp.stablehlo.mlir", "w").write(text)

# @main binding order (from the exported signature): (l2.bias, l2.weight, l1.bias, l1.weight, x)
bindings = [("b2", m.l2.bias), ("W2", m.l2.weight), ("b1", m.l1.bias), ("W1", m.l1.weight), ("x", x)]


def c_arr(name, t):
    a = t.detach().numpy().astype(np.float64).ravel()
    body = ", ".join(repr(float(v)) for v in a)
    return f"static const double {name}[{a.size}] = {{{body}}};"


with open(f"{OUT}/mlp_ref.h", "w") as f:
    f.write("// Generated: fixed MLP inputs (@main binding order) + torch golden output.\n")
    for nm, t in bindings:
        f.write(c_arr(nm, t) + "\n")
    f.write(c_arr("y_golden", golden) + "\n")

# .npy in @main binding order for iree-run-module correctness check
for i, (nm, t) in enumerate(bindings):
    np.save(f"{OUT}/mlp_in{i}_{nm}.npy", t.detach().numpy().astype(np.float64))
np.save(f"{OUT}/mlp_golden.npy", golden.detach().numpy().astype(np.float64))

print("wrote torch_mlp.stablehlo.mlir + mlp_ref.h + mlp_in{0..4}.npy + mlp_golden.npy")
print("golden[0,:4] =", golden.detach().numpy().ravel()[:4])
