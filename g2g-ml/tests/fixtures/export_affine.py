#!/usr/bin/env python3
# Generate the ExecuTorch fixture used by `g2g-ml/tests/executorch_inference.rs`:
# `affine_2x_plus_1.pte`, a program computing `x * SCALE + OFFSET` on an f32 SHAPE
# input. Committed for provenance; the .pte it writes is the checked-in fixture
# (the build does not run this).
#
# Run:
#   uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python executorch==1.5.1 "torch==2.14.0" --index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
#   .venv/bin/python g2g-ml/tests/fixtures/export_affine.py g2g-ml/tests/fixtures/affine_2x_plus_1.pte
import sys
import torch
from torch.export import export
from executorch.exir import to_edge_transform_and_lower

SCALE = 2.0
OFFSET = 1.0
SHAPE = (1, 3, 4, 4)

class Affine(torch.nn.Module):
    def forward(self, x):
        return x * SCALE + OFFSET

program = to_edge_transform_and_lower(export(Affine(), (torch.ones(SHAPE),))).to_executorch()
with open(sys.argv[1], "wb") as f:
    f.write(program.buffer)
print(len(program.buffer))
