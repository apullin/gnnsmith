"""Abstract GNN operator specification with Z3 constraints and PyTorch materialization."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import z3

from gnnsmith.constraints import Z3GraphContext
from gnnsmith.ir import TensorRole
from gnnsmith.topology import GraphBundle


class StepModule(nn.Module):
    """Uniform runtime wrapper: `(inputs: List[Tensor], bundle: GraphBundle) -> (Tensor, GraphBundle)`."""

    @abstractmethod
    def forward_step(self, inputs: List[torch.Tensor], bundle: GraphBundle) -> Tuple[torch.Tensor, GraphBundle]:
        raise NotImplementedError


@dataclass(frozen=True)
class OpCapabilities:
    supports_empty_edges: bool = True
    supports_edge_index_subclass: bool = True
    supports_decomposed_layers: bool = True
    supports_to_hetero: bool = False
    mutates_topology: bool = False
    is_permutation_equivariant: bool = True
    is_batch_invariant: bool = True


class AbsGNNOp(ABC):
    """Base declarative operator spec for Z3 DAG synthesis and PyG instantiation."""

    op_type: str = "BaseOp"
    arity: int = 1
    in_role: TensorRole = TensorRole.NODE_FEAT
    out_role: TensorRole = TensorRole.NODE_FEAT
    capabilities: OpCapabilities = OpCapabilities()

    @abstractmethod
    def sample_and_constrain(
        self,
        ctx: Z3GraphContext,
        in_channels: List[z3.ArithRef],
        out_channels: z3.ArithRef,
        scope_info: Dict[str, int],
        rng: random.Random,
    ) -> Tuple[List[z3.BoolRef | bool], Dict[str, Any]]:
        """Return (z3_constraints, symbolic_or_concrete_params)."""
        raise NotImplementedError

    @classmethod
    def resolve_params(cls, model: z3.ModelRef, raw_params: Dict[str, Any]) -> Dict[str, Any]:
        resolved: Dict[str, Any] = {}
        for k, v in raw_params.items():
            if isinstance(v, z3.ArithRef):
                resolved[k] = Z3GraphContext.eval_int(model, v)
            else:
                resolved[k] = v
        return resolved

    @classmethod
    @abstractmethod
    def materialize(cls, params: Dict[str, Any]) -> StepModule:
        """Instantiate an executable PyTorch StepModule from solved concrete parameters."""
        raise NotImplementedError
