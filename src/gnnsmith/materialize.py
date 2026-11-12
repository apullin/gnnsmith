"""Materialize a Z3-solved GraphIR into an executable PyTorch Module."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from gnnsmith.ir import GraphIR
from gnnsmith.ops import OP_REGISTRY
from gnnsmith.ops.base import StepModule
from gnnsmith.topology import GraphBundle


class MaterializedGNN(nn.Module):
    """Executable multi-branch GNN module built from a concrete GraphIR."""

    def __init__(self, ir: GraphIR) -> None:
        super().__init__()
        self.ir = ir
        steps: List[StepModule] = []
        for inst in ir.insts:
            op_cls = OP_REGISTRY[inst.op_type]
            steps.append(op_cls.materialize(inst.params))
        self.steps = nn.ModuleList(steps)

    def forward(self, bundle: GraphBundle) -> torch.Tensor:
        env: Dict[str, torch.Tensor] = {self.ir.input_value: bundle.x}
        scope_bundles: Dict[int, GraphBundle] = {0: bundle}

        for inst, step in zip(self.ir.insts, self.steps):
            in_bundle = scope_bundles[inst.in_scope]
            xs = [env[name] for name in inst.inputs]
            out_tensor, out_bundle = step.forward_step(xs, in_bundle)
            env[inst.output] = out_tensor
            scope_bundles[inst.out_scope] = out_bundle

        return env[self.ir.output_value]


def materialize_ir(ir: GraphIR, seed: int = 0) -> MaterializedGNN:
    torch.manual_seed(seed)
    model = MaterializedGNN(ir)
    model.eval()
    return model
