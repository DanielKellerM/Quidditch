import argparse
import os
import random
import torch
import torch.nn as nn
from iree.turbine import aot

seed = 1234
random.seed(seed)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

IN_FEATURES = 16
OUT_FEATURES = 16

parser = argparse.ArgumentParser(prog='iree-turbine')
parser.add_argument('output', nargs='?')
parser.add_argument('--frames', dest='frames', metavar='N', type=int, default=1, nargs='?')
parser.add_argument('--dtype', dest='dtype', metavar='F', choices=['f32', 'f64'], default='f32')
parser.add_argument('-dump', dest='dump', action='store_true', default=False)
args = parser.parse_args()

name_to_dtype = {
    'f32': torch.float32,
    'f64': torch.float64,
}
dtype = name_to_dtype[args.dtype]


class SoftmaxLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, dtype=dtype)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)


model = SoftmaxLinear(IN_FEATURES, OUT_FEATURES)
model.train(False)


def with_frames(n_frames):
    size = 1, n_frames, model.in_features

    class CompiledSoftmaxLinear(aot.CompiledModule):
        def main(self, x=aot.AbstractTensor(*size, dtype=dtype)):
            y = aot.jittable(model.forward)(
                x,
                constraints=[]
            )
            return y

    return CompiledSoftmaxLinear


exported = aot.export(with_frames(n_frames=args.frames))
if args.dump:
    exported.print_readable()
else:
    exported.save_mlir(args.output)
