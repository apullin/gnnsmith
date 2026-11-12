"""PyG Normalization, Pooling (topology-mutating + global readout), and DAG combinators."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn.aggr as pyg_aggr
import torch_geometric.nn.norm as pyg_norm
import torch_geometric.nn.pool as pyg_pool
import z3

from gnnsmith.ir import TensorRole
from gnnsmith.ops.base import AbsGNNOp, OpCapabilities, StepModule
from gnnsmith.ops.pyg_conv import _LambdaStep
from gnnsmith.topology import GraphBundle


class GraphNormOp(AbsGNNOp):
    op_type = "GraphNorm"

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [out_channels == in_channels[0]], {"c": in_channels[0]}

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_norm.GraphNorm(p["c"])
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], batch=b.batch, batch_size=b.num_graphs))


class LayerNormOp(AbsGNNOp):
    op_type = "LayerNorm"

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [out_channels == in_channels[0]], {
            "c": in_channels[0],
            "mode": rng.choice(["graph", "node"]),
            "affine": rng.choice([True, False]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_norm.LayerNorm(p["c"], affine=p["affine"], mode=p["mode"])
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], batch=b.batch, batch_size=b.num_graphs))


class InstanceNormOp(AbsGNNOp):
    op_type = "InstanceNorm"

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [out_channels == in_channels[0]], {
            "c": in_channels[0],
            "affine": rng.choice([True, False]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_norm.InstanceNorm(p["c"], affine=p["affine"])
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], batch=b.batch, batch_size=b.num_graphs))


class PairNormOp(AbsGNNOp):
    op_type = "PairNorm"

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [out_channels == in_channels[0]], {
            "scale_individually": rng.choice([False, True]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_norm.PairNorm(scale_individually=p["scale_individually"])
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], batch=b.batch, batch_size=b.num_graphs))


class GraphSizeNormOp(AbsGNNOp):
    op_type = "GraphSizeNorm"

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [out_channels == in_channels[0]], {}

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_norm.GraphSizeNorm()
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], batch=b.batch, batch_size=b.num_graphs))


class MeanSubtractionNormOp(AbsGNNOp):
    op_type = "MeanSubtractionNorm"

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [out_channels == in_channels[0]], {}

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_norm.MeanSubtractionNorm()
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], batch=b.batch, dim_size=b.num_graphs))


class DiffGroupNormOp(AbsGNNOp):
    op_type = "DiffGroupNorm"
    capabilities = OpCapabilities(is_batch_invariant=False)  # normalizes across full batch N

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [out_channels == in_channels[0]], {"c": in_channels[0], "groups": rng.choice([2, 4])}

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_norm.DiffGroupNorm(p["c"], groups=p["groups"])
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0]))


# --- Topology-Mutating Graph Pooling Ops (transition Scope_k -> Scope_{k+1}) ---


class _TopKPoolStep(StepModule):
    def __init__(self, pool: nn.Module, use_edge_attr: bool):
        super().__init__()
        self.pool = pool
        self.use_edge_attr = use_edge_attr

    def forward_step(self, inputs: List[torch.Tensor], b: GraphBundle) -> Tuple[torch.Tensor, GraphBundle]:
        x_out, ei_out, ea_out, batch_out, perm, _ = self.pool(
            inputs[0], b.edge_index, b.edge_attr if self.use_edge_attr else None, batch=b.batch
        )
        n_new = x_out.size(0)
        e_new = ei_out.size(1)
        new_bundle = GraphBundle(
            x=x_out,
            x_0=x_out,
            pos=b.pos[perm],
            edge_index=ei_out,
            edge_weight=torch.ones((e_new,), dtype=x_out.dtype, device=x_out.device),
            edge_attr=(
                ea_out
                if ea_out is not None
                else torch.zeros((e_new, b.edge_attr.size(-1)), dtype=x_out.dtype, device=x_out.device)
            ),
            edge_type=torch.zeros((e_new,), dtype=torch.long, device=x_out.device),
            batch=batch_out,
            num_nodes=n_new,
            num_graphs=b.num_graphs,
            num_relations=b.num_relations,
            topology_kind=b.topology_kind,
        )
        return x_out, new_bundle


class TopKPoolingOp(AbsGNNOp):
    op_type = "TopKPooling"
    capabilities = OpCapabilities(mutates_topology=True, supports_edge_index_subclass=False)

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [out_channels == in_channels[0]], {
            "c": in_channels[0],
            "ratio": rng.choice([0.5, 0.8]),
            "use_edge": rng.choice([False, True]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        pool = pyg_pool.TopKPooling(p["c"], ratio=p["ratio"])
        return _TopKPoolStep(pool, use_edge_attr=p["use_edge"])


class SAGPoolingOp(AbsGNNOp):
    op_type = "SAGPooling"
    capabilities = OpCapabilities(mutates_topology=True, supports_edge_index_subclass=False)

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [out_channels == in_channels[0]], {
            "c": in_channels[0],
            "ratio": rng.choice([0.5, 0.8]),
            "use_edge": rng.choice([False, True]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        pool = pyg_pool.SAGPooling(p["c"], ratio=p["ratio"])
        return _TopKPoolStep(pool, use_edge_attr=p["use_edge"])


# --- Global Graph Readout Aggregations (NODE_FEAT [N, C] -> GRAPH_FEAT [B, C_out]) ---


class GlobalPoolReadoutOp(AbsGNNOp):
    op_type = "GlobalReadout"
    in_role = TensorRole.NODE_FEAT
    out_role = TensorRole.GRAPH_FEAT

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        kind = rng.choice(["sum", "mean", "max", "min", "var", "std", "softmax", "powermean", "set2set", "attn"])
        if kind == "set2set":
            c = [out_channels == 2 * in_channels[0]]
        else:
            c = [out_channels == in_channels[0]]
        return c, {"in_c": in_channels[0], "kind": kind}

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        kind = p["kind"]
        in_c = p["in_c"]
        if kind == "sum":
            ag = pyg_aggr.SumAggregation()
        elif kind == "mean":
            ag = pyg_aggr.MeanAggregation()
        elif kind == "max":
            ag = pyg_aggr.MaxAggregation()
        elif kind == "min":
            ag = pyg_aggr.MinAggregation()
        elif kind == "var":
            ag = pyg_aggr.VarAggregation()
        elif kind == "std":
            ag = pyg_aggr.StdAggregation()
        elif kind == "softmax":
            ag = pyg_aggr.SoftmaxAggregation(learn=True)
        elif kind == "powermean":
            ag = pyg_aggr.PowerMeanAggregation(learn=True)
        elif kind == "set2set":
            ag = pyg_aggr.Set2Set(in_c, processing_steps=2)
        else:
            ag = pyg_aggr.AttentionalAggregation(gate_nn=nn.Linear(in_c, 1))
        return _LambdaStep(ag, lambda mod, xs, b: mod(xs[0], index=b.batch, dim_size=b.num_graphs))


# --- Structural DAG Combinators (Residuals, Concatenations, Projections, Activations) ---


class ResidualAddOp(AbsGNNOp):
    op_type = "ResidualAdd"
    arity = 2
    capabilities = OpCapabilities(supports_to_hetero=True)

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [in_channels[0] == in_channels[1], out_channels == in_channels[0]], {
            "act": rng.choice(["relu", "gelu", "tanh", "identity"])
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        act = p["act"]

        def fn(_m, xs, _b):
            s = xs[0] + xs[1]
            if act == "relu":
                return F.relu(s)
            if act == "gelu":
                return F.gelu(s)
            if act == "tanh":
                return torch.tanh(s)
            return s

        return _LambdaStep(nn.Identity(), fn)


class ChannelConcatOp(AbsGNNOp):
    op_type = "ChannelConcat"
    arity = 2
    capabilities = OpCapabilities(supports_to_hetero=True)

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [out_channels == in_channels[0] + in_channels[1]], {}

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        return _LambdaStep(nn.Identity(), lambda _m, xs, _b: torch.cat([xs[0], xs[1]], dim=-1))


class LinearActOp(AbsGNNOp):
    op_type = "LinearAct"
    capabilities = OpCapabilities(supports_to_hetero=True)

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [], {
            "in_c": in_channels[0],
            "out_c": out_channels,
            "act": rng.choice(["relu", "elu", "silu", "leaky_relu"]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        lin = nn.Linear(p["in_c"], p["out_c"])
        act = p["act"]

        def fn(mod, xs, _b):
            y = mod(xs[0])
            if act == "relu":
                return F.relu(y)
            if act == "elu":
                return F.elu(y)
            if act == "silu":
                return F.silu(y)
            return F.leaky_relu(y, 0.2)

        return _LambdaStep(lin, fn)


NORM_OPS: List[Type[AbsGNNOp]] = [
    GraphNormOp,
    LayerNormOp,
    InstanceNormOp,
    PairNormOp,
    GraphSizeNormOp,
    MeanSubtractionNormOp,
    DiffGroupNormOp,
]

POOL_OPS: List[Type[AbsGNNOp]] = [
    TopKPoolingOp,
    SAGPoolingOp,
]

COMBINATOR_OPS: List[Type[AbsGNNOp]] = [
    ResidualAddOp,
    ChannelConcatOp,
    LinearActOp,
]
