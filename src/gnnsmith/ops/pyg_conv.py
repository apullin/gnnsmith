"""PyTorch Geometric message-passing convolution operator specifications (42+ layers)."""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Tuple, Type

import torch
import torch.nn as nn
import torch_geometric.nn.conv as pyg_conv
import z3

from gnnsmith.constraints import Z3GraphContext
from gnnsmith.ops.base import AbsGNNOp, OpCapabilities, StepModule
from gnnsmith.topology import GraphBundle


class _LambdaStep(StepModule):
    def __init__(self, mod: nn.Module, call_fn: Callable[[nn.Module, List[torch.Tensor], GraphBundle], torch.Tensor]):
        super().__init__()
        self.mod = mod
        self._call_fn = call_fn

    def forward_step(self, inputs: List[torch.Tensor], bundle: GraphBundle) -> Tuple[torch.Tensor, GraphBundle]:
        out = self._call_fn(self.mod, inputs, bundle)
        return out, bundle


class GCNConvOp(AbsGNNOp):
    op_type = "GCNConv"
    capabilities = OpCapabilities(supports_to_hetero=False)

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        params = {
            "in_c": in_channels[0],
            "out_c": out_channels,
            "improved": rng.choice([False, True]),
            "add_self_loops": rng.choice([False, True]),
            "normalize": rng.choice([False, True]),
            "use_weight": rng.choice([False, True]),
        }
        if not params["normalize"]:
            params["add_self_loops"] = False
        return [], params

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.GCNConv(
            p["in_c"],
            p["out_c"],
            improved=p["improved"],
            add_self_loops=p["add_self_loops"],
            normalize=p["normalize"],
        )
        return _LambdaStep(
            m,
            lambda mod, xs, b: mod(xs[0], b.edge_index, b.edge_weight if p["use_weight"] else None),
        )


class SAGEConvOp(AbsGNNOp):
    op_type = "SAGEConv"
    capabilities = OpCapabilities(supports_to_hetero=True)

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        params = {
            "in_c": in_channels[0],
            "out_c": out_channels,
            "aggr": rng.choice(["mean", "max", "sum"]),
            "normalize": rng.choice([False, True]),
            "root_weight": rng.choice([False, True]),
            "project": rng.choice([False, True]),
        }
        return [], params

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.SAGEConv(
            p["in_c"],
            p["out_c"],
            aggr=p["aggr"],
            normalize=p["normalize"],
            root_weight=p["root_weight"],
            project=p["project"],
        )
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index))


class GraphConvOp(AbsGNNOp):
    op_type = "GraphConv"
    capabilities = OpCapabilities(supports_to_hetero=True)

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        params = {
            "in_c": in_channels[0],
            "out_c": out_channels,
            "aggr": rng.choice(["add", "mean", "max"]),
            "use_weight": rng.choice([False, True]),
        }
        return [], params

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.GraphConv(p["in_c"], p["out_c"], aggr=p["aggr"])
        return _LambdaStep(
            m,
            lambda mod, xs, b: mod(xs[0], b.edge_index, b.edge_weight if p["use_weight"] else None),
        )


class GATConvOp(AbsGNNOp):
    op_type = "GATConv"
    capabilities = OpCapabilities(supports_to_hetero=True)

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        heads = rng.choice([1, 2, 4])
        concat = rng.choice([True, False])
        head_c = ctx.new_int("gat_hc", 2, 32)
        constraints: List[z3.BoolRef | bool] = [
            out_channels == (head_c * heads if concat else head_c)
        ]
        use_edge = rng.choice([False, True])
        params = {
            "in_c": in_channels[0],
            "head_c": head_c,
            "heads": heads,
            "concat": concat,
            "add_self_loops": rng.choice([False, True]),
            "residual": rng.choice([False, True]),
            "fill_value": rng.choice(["mean", "add", "max", 0.5]),
            "edge_dim": scope_info["edge_dim"] if use_edge else None,
        }
        return constraints, params

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.GATConv(
            p["in_c"],
            p["head_c"],
            heads=p["heads"],
            concat=p["concat"],
            dropout=0.0,
            add_self_loops=p["add_self_loops"],
            edge_dim=p["edge_dim"],
            fill_value=p["fill_value"],
            residual=p["residual"],
        )
        return _LambdaStep(
            m,
            lambda mod, xs, b: mod(xs[0], b.edge_index, b.edge_attr if p["edge_dim"] is not None else None),
        )


class GATv2ConvOp(AbsGNNOp):
    op_type = "GATv2Conv"
    capabilities = OpCapabilities(supports_to_hetero=True)

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        heads = rng.choice([1, 2, 4])
        concat = rng.choice([True, False])
        head_c = ctx.new_int("gatv2_hc", 2, 32)
        use_edge = rng.choice([False, True])
        params = {
            "in_c": in_channels[0],
            "head_c": head_c,
            "heads": heads,
            "concat": concat,
            "add_self_loops": rng.choice([False, True]),
            "share_weights": rng.choice([False, True]),
            "residual": rng.choice([False, True]),
            "edge_dim": scope_info["edge_dim"] if use_edge else None,
        }
        return [out_channels == (head_c * heads if concat else head_c)], params

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.GATv2Conv(
            p["in_c"],
            p["head_c"],
            heads=p["heads"],
            concat=p["concat"],
            dropout=0.0,
            add_self_loops=p["add_self_loops"],
            share_weights=p["share_weights"],
            residual=p["residual"],
            edge_dim=p["edge_dim"],
        )
        return _LambdaStep(
            m,
            lambda mod, xs, b: mod(xs[0], b.edge_index, b.edge_attr if p["edge_dim"] is not None else None),
        )


class TransformerConvOp(AbsGNNOp):
    op_type = "TransformerConv"
    capabilities = OpCapabilities(supports_to_hetero=True)

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        heads = rng.choice([1, 2, 4])
        concat = rng.choice([True, False])
        head_c = ctx.new_int("tf_hc", 2, 32)
        use_edge = rng.choice([False, True])
        params = {
            "in_c": in_channels[0],
            "head_c": head_c,
            "heads": heads,
            "concat": concat,
            "beta": rng.choice([False, True]),
            "root_weight": rng.choice([False, True]),
            "edge_dim": scope_info["edge_dim"] if use_edge else None,
        }
        return [out_channels == (head_c * heads if concat else head_c)], params

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.TransformerConv(
            p["in_c"],
            p["head_c"],
            heads=p["heads"],
            concat=p["concat"],
            beta=p["beta"],
            dropout=0.0,
            edge_dim=p["edge_dim"],
            root_weight=p["root_weight"],
        )
        return _LambdaStep(
            m,
            lambda mod, xs, b: mod(xs[0], b.edge_index, b.edge_attr if p["edge_dim"] is not None else None),
        )


class GINConvOp(AbsGNNOp):
    op_type = "GINConv"
    capabilities = OpCapabilities(supports_to_hetero=True)

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        hid = ctx.new_int("gin_hid", 4, 32)
        return [], {
            "in_c": in_channels[0],
            "hid": hid,
            "out_c": out_channels,
            "train_eps": rng.choice([False, True]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        mlp = nn.Sequential(nn.Linear(p["in_c"], p["hid"]), nn.ReLU(), nn.Linear(p["hid"], p["out_c"]))
        m = pyg_conv.GINConv(mlp, train_eps=p["train_eps"])
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index))


class GINEConvOp(AbsGNNOp):
    op_type = "GINEConv"
    capabilities = OpCapabilities(supports_to_hetero=True)

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        hid = ctx.new_int("gine_hid", 4, 32)
        return [], {
            "in_c": in_channels[0],
            "hid": hid,
            "out_c": out_channels,
            "edge_dim": scope_info["edge_dim"],
            "train_eps": rng.choice([False, True]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        mlp = nn.Sequential(nn.Linear(p["in_c"], p["hid"]), nn.ReLU(), nn.Linear(p["hid"], p["out_c"]))
        m = pyg_conv.GINEConv(mlp, train_eps=p["train_eps"], edge_dim=p["edge_dim"])
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index, b.edge_attr))


class ChebConvOp(AbsGNNOp):
    op_type = "ChebConv"

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [], {
            "in_c": in_channels[0],
            "out_c": out_channels,
            "K": rng.choice([1, 2, 3]),
            "normalization": rng.choice([None, "sym", "rw"]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.ChebConv(p["in_c"], p["out_c"], K=p["K"], normalization=p["normalization"])
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index, b.edge_weight, batch=b.batch))


class TAGConvOp(AbsGNNOp):
    op_type = "TAGConv"

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [], {
            "in_c": in_channels[0],
            "out_c": out_channels,
            "K": rng.choice([1, 2, 3]),
            "normalize": rng.choice([False, True]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.TAGConv(p["in_c"], p["out_c"], K=p["K"], normalize=p["normalize"])
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index, b.edge_weight))


class SGConvOp(AbsGNNOp):
    op_type = "SGConv"

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [], {
            "in_c": in_channels[0],
            "out_c": out_channels,
            "K": rng.choice([1, 2, 3]),
            "add_self_loops": rng.choice([False, True]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.SGConv(p["in_c"], p["out_c"], K=p["K"], add_self_loops=p["add_self_loops"])
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index, b.edge_weight))


class SSGConvOp(AbsGNNOp):
    op_type = "SSGConv"

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [], {
            "in_c": in_channels[0],
            "out_c": out_channels,
            "alpha": rng.choice([0.1, 0.5]),
            "K": rng.choice([1, 2, 3]),
            "add_self_loops": rng.choice([False, True]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.SSGConv(p["in_c"], p["out_c"], alpha=p["alpha"], K=p["K"], add_self_loops=p["add_self_loops"])
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index, b.edge_weight))


class APPNPOp(AbsGNNOp):
    op_type = "APPNP"

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        add_sl = rng.choice([False, True])
        norm = rng.choice([False, True])
        if not norm:
            add_sl = False
        return [out_channels == in_channels[0]], {
            "K": rng.choice([1, 2, 4]),
            "alpha": rng.choice([0.1, 0.5]),
            "add_self_loops": add_sl,
            "normalize": norm,
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.APPNP(
            K=p["K"],
            alpha=p["alpha"],
            dropout=0.0,
            add_self_loops=p["add_self_loops"],
            normalize=p["normalize"],
        )
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index, b.edge_weight))


class ARMAConvOp(AbsGNNOp):
    op_type = "ARMAConv"

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [], {
            "in_c": in_channels[0],
            "out_c": out_channels,
            "num_stacks": rng.choice([1, 2]),
            "num_layers": rng.choice([1, 2]),
            "shared_weights": rng.choice([False, True]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.ARMAConv(
            p["in_c"],
            p["out_c"],
            num_stacks=p["num_stacks"],
            num_layers=p["num_layers"],
            shared_weights=p["shared_weights"],
            dropout=0.0,
        )
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index, b.edge_weight))


class AGNNConvOp(AbsGNNOp):
    op_type = "AGNNConv"

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [out_channels == in_channels[0]], {
            "requires_grad": rng.choice([False, True]),
            "add_self_loops": rng.choice([False, True]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.AGNNConv(requires_grad=p["requires_grad"], add_self_loops=p["add_self_loops"])
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index))


class ResGatedGraphConvOp(AbsGNNOp):
    op_type = "ResGatedGraphConv"
    capabilities = OpCapabilities(supports_to_hetero=True)

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        use_edge = rng.choice([False, True])
        return [], {
            "in_c": in_channels[0],
            "out_c": out_channels,
            "root_weight": rng.choice([False, True]),
            "edge_dim": scope_info["edge_dim"] if use_edge else None,
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.ResGatedGraphConv(
            p["in_c"], p["out_c"], root_weight=p["root_weight"], edge_dim=p["edge_dim"]
        )
        return _LambdaStep(
            m,
            lambda mod, xs, b: mod(xs[0], b.edge_index, b.edge_attr if p["edge_dim"] is not None else None),
        )


class GatedGraphConvOp(AbsGNNOp):
    op_type = "GatedGraphConv"

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [out_channels >= in_channels[0]], {
            "out_c": out_channels,
            "num_layers": rng.choice([1, 2, 3]),
            "aggr": rng.choice(["add", "mean", "max"]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.GatedGraphConv(p["out_c"], num_layers=p["num_layers"], aggr=p["aggr"])
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index, b.edge_weight))


class MFConvOp(AbsGNNOp):
    op_type = "MFConv"
    capabilities = OpCapabilities(supports_to_hetero=True)

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [], {
            "in_c": in_channels[0],
            "out_c": out_channels,
            "max_degree": rng.choice([3, 8]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.MFConv(p["in_c"], p["out_c"], max_degree=p["max_degree"])
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index))


class RGCNConvOp(AbsGNNOp):
    op_type = "RGCNConv"

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        mode = rng.choice(["plain", "bases", "blocks"])
        constraints: List[z3.BoolRef | bool] = []
        num_bases = None
        num_blocks = None
        if mode == "bases":
            num_bases = 2
        elif mode == "blocks":
            num_blocks = 2
            in_k = ctx.new_int("rgcn_ik", 1, 16)
            out_k = ctx.new_int("rgcn_ok", 1, 16)
            constraints.extend([in_channels[0] == in_k * 2, out_channels == out_k * 2])
        return constraints, {
            "in_c": in_channels[0],
            "out_c": out_channels,
            "num_relations": scope_info["num_relations"],
            "num_bases": num_bases,
            "num_blocks": num_blocks,
            "aggr": rng.choice(["mean", "add", "max"]),
            "root_weight": rng.choice([False, True]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.RGCNConv(
            p["in_c"],
            p["out_c"],
            num_relations=p["num_relations"],
            num_bases=p["num_bases"],
            num_blocks=p["num_blocks"],
            aggr=p["aggr"],
            root_weight=p["root_weight"],
        )
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index, b.edge_type))


class FastRGCNConvOp(RGCNConvOp):
    op_type = "FastRGCNConv"

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        constraints, params = super().sample_and_constrain(ctx, in_channels, out_channels, scope_info, rng)
        params["aggr"] = rng.choice(["mean", "add"])
        return constraints, params

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.FastRGCNConv(
            p["in_c"],
            p["out_c"],
            num_relations=p["num_relations"],
            num_bases=p["num_bases"],
            num_blocks=p["num_blocks"],
            aggr=p["aggr"],
            root_weight=p["root_weight"],
        )
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index, b.edge_type))


class RGATConvOp(AbsGNNOp):
    op_type = "RGATConv"

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        heads = rng.choice([1, 2])
        concat = rng.choice([True, False])
        head_c = ctx.new_int("rgat_hc", 2, 16)
        use_edge = rng.choice([False, True])
        return [out_channels == (head_c * heads if concat else head_c)], {
            "in_c": in_channels[0],
            "head_c": head_c,
            "num_relations": scope_info["num_relations"],
            "heads": heads,
            "concat": concat,
            "attention_mechanism": rng.choice(["across-relation", "within-relation"]),
            "attention_mode": rng.choice(
                [
                    "additive-self-attention",
                    "multiplicative-self-attention",
                ]
            ),
            "mod": rng.choice([None, "additive", "scaled"]),
            "edge_dim": scope_info["edge_dim"] if use_edge else None,
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.RGATConv(
            p["in_c"],
            p["head_c"],
            num_relations=p["num_relations"],
            heads=p["heads"],
            concat=p["concat"],
            attention_mechanism=p["attention_mechanism"],
            attention_mode=p["attention_mode"],
            mod=p["mod"],
            dropout=0.0,
            edge_dim=p["edge_dim"],
        )
        return _LambdaStep(
            m,
            lambda mod, xs, b: mod(
                xs[0],
                b.edge_index,
                b.edge_type,
                edge_attr=b.edge_attr if p["edge_dim"] is not None else None,
            ),
        )


class FiLMConvOp(AbsGNNOp):
    op_type = "FiLMConv"
    capabilities = OpCapabilities(supports_to_hetero=True)

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [], {
            "in_c": in_channels[0],
            "out_c": out_channels,
            "num_relations": scope_info["num_relations"],
            "aggr": rng.choice(["mean", "max", "add"]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.FiLMConv(p["in_c"], p["out_c"], num_relations=p["num_relations"], aggr=p["aggr"])
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index, b.edge_type))


class GMMConvOp(AbsGNNOp):
    op_type = "GMMConv"
    capabilities = OpCapabilities(supports_to_hetero=True)

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [], {
            "in_c": in_channels[0],
            "out_c": out_channels,
            "dim": scope_info["edge_dim"],
            "kernel_size": rng.choice([2, 4]),
            "separate_gaussians": rng.choice([False, True]),
            "aggr": rng.choice(["mean", "add", "max"]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.GMMConv(
            p["in_c"],
            p["out_c"],
            dim=p["dim"],
            kernel_size=p["kernel_size"],
            separate_gaussians=p["separate_gaussians"],
            aggr=p["aggr"],
        )
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index, b.edge_attr))


class NNConvOp(AbsGNNOp):
    op_type = "NNConv"
    capabilities = OpCapabilities(supports_to_hetero=True)

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        # Keep in_c * out_c bounded for edge MLP
        return [in_channels[0] <= 16, out_channels <= 16], {
            "in_c": in_channels[0],
            "out_c": out_channels,
            "edge_dim": scope_info["edge_dim"],
            "aggr": rng.choice(["add", "mean", "max"]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        edge_mlp = nn.Linear(p["edge_dim"], p["in_c"] * p["out_c"])
        m = pyg_conv.NNConv(p["in_c"], p["out_c"], nn=edge_mlp, aggr=p["aggr"])
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index, b.edge_attr))


class CGConvOp(AbsGNNOp):
    op_type = "CGConv"
    capabilities = OpCapabilities(supports_to_hetero=True)

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        use_edge = rng.choice([False, True])
        return [out_channels == in_channels[0]], {
            "c": in_channels[0],
            "dim": scope_info["edge_dim"] if use_edge else 0,
            "aggr": rng.choice(["add", "mean", "max"]),
            "batch_norm": rng.choice([False, True]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.CGConv(p["c"], dim=p["dim"], aggr=p["aggr"], batch_norm=p["batch_norm"])
        return _LambdaStep(
            m,
            lambda mod, xs, b: mod(xs[0], b.edge_index, b.edge_attr if p["dim"] > 0 else None),
        )


class EdgeConvOp(AbsGNNOp):
    op_type = "EdgeConv"
    capabilities = OpCapabilities(supports_to_hetero=False)

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [], {
            "in_c": in_channels[0],
            "out_c": out_channels,
            "aggr": rng.choice(["max", "mean", "add"]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        mlp = nn.Sequential(nn.Linear(2 * p["in_c"], p["out_c"]), nn.ReLU(), nn.Linear(p["out_c"], p["out_c"]))
        m = pyg_conv.EdgeConv(mlp, aggr=p["aggr"])
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index))


class FeaStConvOp(AbsGNNOp):
    op_type = "FeaStConv"
    capabilities = OpCapabilities(supports_to_hetero=True)

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [], {
            "in_c": in_channels[0],
            "out_c": out_channels,
            "heads": rng.choice([1, 2, 4]),
            "add_self_loops": rng.choice([False, True]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.FeaStConv(p["in_c"], p["out_c"], heads=p["heads"], add_self_loops=p["add_self_loops"])
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index))


class PointNetConvOp(AbsGNNOp):
    op_type = "PointNetConv"

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [], {
            "in_c": in_channels[0],
            "out_c": out_channels,
            "add_self_loops": rng.choice([False, True]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        local_nn = nn.Linear(p["in_c"] + 3, p["out_c"])
        global_nn = nn.Linear(p["out_c"], p["out_c"])
        m = pyg_conv.PointNetConv(local_nn=local_nn, global_nn=global_nn, add_self_loops=p["add_self_loops"])
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.pos, b.edge_index))


class PointTransformerConvOp(AbsGNNOp):
    op_type = "PointTransformerConv"

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [], {
            "in_c": in_channels[0],
            "out_c": out_channels,
            "add_self_loops": rng.choice([False, True]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.PointTransformerConv(p["in_c"], p["out_c"], add_self_loops=p["add_self_loops"])
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.pos, b.edge_index))


class LEConvOp(AbsGNNOp):
    op_type = "LEConv"
    capabilities = OpCapabilities(supports_to_hetero=True)

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [], {"in_c": in_channels[0], "out_c": out_channels}

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.LEConv(p["in_c"], p["out_c"])
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index, b.edge_weight))


class ClusterGCNConvOp(AbsGNNOp):
    op_type = "ClusterGCNConv"

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [], {
            "in_c": in_channels[0],
            "out_c": out_channels,
            "diag_lambda": rng.choice([0.0, 0.5]),
            "add_self_loops": rng.choice([False, True]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.ClusterGCNConv(
            p["in_c"], p["out_c"], diag_lambda=p["diag_lambda"], add_self_loops=p["add_self_loops"]
        )
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index))


class GENConvOp(AbsGNNOp):
    op_type = "GENConv"
    capabilities = OpCapabilities(supports_to_hetero=True)

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        use_edge = rng.choice([False, True])
        return [], {
            "in_c": in_channels[0],
            "out_c": out_channels,
            "aggr": rng.choice(["softmax", "powermean", "add", "mean", "max"]),
            "msg_norm": rng.choice([False, True]),
            "norm": rng.choice(["layer", "batch"]),
            "edge_dim": scope_info["edge_dim"] if use_edge else None,
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.GENConv(
            p["in_c"],
            p["out_c"],
            aggr=p["aggr"],
            msg_norm=p["msg_norm"],
            norm=p["norm"],
            edge_dim=p["edge_dim"],
        )
        return _LambdaStep(
            m,
            lambda mod, xs, b: mod(xs[0], b.edge_index, b.edge_attr if p["edge_dim"] is not None else None),
        )


class PNAConvOp(AbsGNNOp):
    op_type = "PNAConv"

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        towers = rng.choice([1, 2])
        in_k = ctx.new_int("pna_ik", 2, 16)
        out_k = ctx.new_int("pna_ok", 2, 16)
        use_edge = rng.choice([False, True])
        return [in_channels[0] == in_k * towers, out_channels == out_k * towers], {
            "in_c": in_channels[0],
            "out_c": out_channels,
            "towers": towers,
            "aggregators": rng.sample(["mean", "min", "max", "std", "var", "sum"], k=2),
            "scalers": rng.sample(["identity", "amplification", "attenuation", "linear"], k=2),
            "edge_dim": scope_info["edge_dim"] if use_edge else None,
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        deg = torch.tensor([1, 4, 8, 4, 2, 1], dtype=torch.long)
        m = pyg_conv.PNAConv(
            p["in_c"],
            p["out_c"],
            aggregators=p["aggregators"],
            scalers=p["scalers"],
            deg=deg,
            towers=p["towers"],
            edge_dim=p["edge_dim"],
        )
        return _LambdaStep(
            m,
            lambda mod, xs, b: mod(xs[0], b.edge_index, b.edge_attr if p["edge_dim"] is not None else None),
        )


class EGConvOp(AbsGNNOp):
    op_type = "EGConv"

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        heads = rng.choice([2, 4])
        ok = ctx.new_int("eg_ok", 2, 16)
        return [out_channels == ok * heads], {
            "in_c": in_channels[0],
            "out_c": out_channels,
            "num_heads": heads,
            "num_bases": rng.choice([2, 4]),
            "aggregators": rng.sample(["symnorm", "sum", "mean", "max", "min", "std"], k=2),
            "add_self_loops": rng.choice([False, True]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.EGConv(
            p["in_c"],
            p["out_c"],
            aggregators=p["aggregators"],
            num_heads=p["num_heads"],
            num_bases=p["num_bases"],
            add_self_loops=p["add_self_loops"],
        )
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index))


class PDNConvOp(AbsGNNOp):
    op_type = "PDNConv"

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        add_sl = rng.choice([False, True])
        norm = rng.choice([False, True])
        if not norm:
            add_sl = False
        return [], {
            "in_c": in_channels[0],
            "out_c": out_channels,
            "edge_dim": scope_info["edge_dim"],
            "hid": rng.choice([8, 16]),
            "add_self_loops": add_sl,
            "normalize": norm,
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.PDNConv(
            p["in_c"],
            p["out_c"],
            edge_dim=p["edge_dim"],
            hidden_channels=p["hid"],
            add_self_loops=p["add_self_loops"],
            normalize=p["normalize"],
        )
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index, b.edge_attr))


class GeneralConvOp(AbsGNNOp):
    op_type = "GeneralConv"
    capabilities = OpCapabilities(supports_to_hetero=True)

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        heads = rng.choice([1, 2])
        ok = ctx.new_int("genconv_ok", 2, 16)
        attention = rng.choice([False, True])
        use_edge = rng.choice([False, True])
        return [out_channels == ok * heads], {
            "in_c": in_channels[0],
            "out_c": out_channels,
            "in_edge_channels": scope_info["edge_dim"] if use_edge else None,
            "aggr": rng.choice(["add", "mean", "max"]),
            "heads": heads,
            "attention": attention,
            "attention_type": rng.choice(["additive", "dot_product"]),
            "l2_normalize": rng.choice([False, True]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.GeneralConv(
            p["in_c"],
            p["out_c"],
            in_edge_channels=p["in_edge_channels"],
            aggr=p["aggr"],
            heads=p["heads"],
            attention=p["attention"],
            attention_type=p["attention_type"],
            l2_normalize=p["l2_normalize"],
        )
        return _LambdaStep(
            m,
            lambda mod, xs, b: mod(
                xs[0], b.edge_index, b.edge_attr if p["in_edge_channels"] is not None else None
            ),
        )


class SimpleConvOp(AbsGNNOp):
    op_type = "SimpleConv"
    capabilities = OpCapabilities(supports_to_hetero=True)

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        combine = rng.choice([None, "sum", "cat", "self_loop"])
        if combine == "cat":
            c = [out_channels == 2 * in_channels[0]]
        else:
            c = [out_channels == in_channels[0]]
        return c, {"aggr": rng.choice(["sum", "mean", "max", "min"]), "combine_root": combine}

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.SimpleConv(aggr=p["aggr"], combine_root=p["combine_root"])
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index))


class LGConvOp(AbsGNNOp):
    op_type = "LGConv"

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [out_channels == in_channels[0]], {"normalize": rng.choice([False, True])}

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.LGConv(normalize=p["normalize"])
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index, b.edge_weight))


class AntiSymmetricConvOp(AbsGNNOp):
    op_type = "AntiSymmetricConv"

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [out_channels == in_channels[0]], {
            "in_c": in_channels[0],
            "num_iters": rng.choice([1, 2]),
            "act": rng.choice(["tanh", "relu"]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.AntiSymmetricConv(p["in_c"], num_iters=p["num_iters"], act=p["act"])
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index))


class DirGNNConvOp(AbsGNNOp):
    op_type = "DirGNNConv"
    capabilities = OpCapabilities(supports_edge_index_subclass=False)

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        return [], {
            "in_c": in_channels[0],
            "out_c": out_channels,
            "alpha": rng.choice([0.0, 0.5, 1.0]),
            "root_weight": rng.choice([False, True]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        inner = pyg_conv.SAGEConv(p["in_c"], p["out_c"])
        m = pyg_conv.DirGNNConv(inner, alpha=p["alpha"], root_weight=p["root_weight"])
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index))


class MixHopConvOp(AbsGNNOp):
    op_type = "MixHopConv"

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        powers = rng.choice([[0, 1], [0, 1, 2], [1, 2]])
        branch_c = ctx.new_int("mixhop_c", 2, 16)
        return [out_channels == branch_c * len(powers)], {
            "in_c": in_channels[0],
            "branch_c": branch_c,
            "powers": powers,
            "add_self_loops": rng.choice([False, True]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.MixHopConv(
            p["in_c"], p["branch_c"], powers=p["powers"], add_self_loops=p["add_self_loops"]
        )
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index, b.edge_weight))


class FAConvOp(AbsGNNOp):
    op_type = "FAConv"
    arity = 2  # (x, x_0)

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        add_sl = rng.choice([False, True])
        norm = rng.choice([False, True])
        if not norm:
            add_sl = False
        return [in_channels[0] == in_channels[1], out_channels == in_channels[0]], {
            "c": in_channels[0],
            "eps": rng.choice([0.1, 0.5]),
            "add_self_loops": add_sl,
            "normalize": norm,
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.FAConv(
            p["c"], eps=p["eps"], dropout=0.0, add_self_loops=p["add_self_loops"], normalize=p["normalize"]
        )
        return _LambdaStep(
            m,
            lambda mod, xs, b: mod(xs[0], xs[1], b.edge_index, None if p["normalize"] else b.edge_weight),
        )


class GCN2ConvOp(AbsGNNOp):
    op_type = "GCN2Conv"
    arity = 2  # (x, x_0)

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        add_sl = rng.choice([False, True])
        norm = rng.choice([False, True])
        if not norm:
            add_sl = False
        return [in_channels[0] == in_channels[1], out_channels == in_channels[0]], {
            "c": in_channels[0],
            "alpha": rng.choice([0.1, 0.5]),
            "theta": rng.choice([0.5, 1.0]),
            "layer": rng.choice([1, 2]),
            "shared_weights": rng.choice([False, True]),
            "add_self_loops": add_sl,
            "normalize": norm,
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.GCN2Conv(
            p["c"],
            alpha=p["alpha"],
            theta=p["theta"],
            layer=p["layer"],
            shared_weights=p["shared_weights"],
            add_self_loops=p["add_self_loops"],
            normalize=p["normalize"],
        )
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], xs[1], b.edge_index, b.edge_weight))


class GPSConvOp(AbsGNNOp):
    op_type = "GPSConv"

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        heads = rng.choice([1, 2, 4])
        hk = ctx.new_int("gps_hk", 2, 16)
        return [in_channels[0] == hk * heads, out_channels == in_channels[0]], {
            "c": in_channels[0],
            "heads": heads,
            "norm": rng.choice([None, "batch_norm", "layer_norm"]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        inner = pyg_conv.GINConv(nn.Sequential(nn.Linear(p["c"], p["c"]), nn.ReLU(), nn.Linear(p["c"], p["c"])))
        m = pyg_conv.GPSConv(p["c"], conv=inner, heads=p["heads"], dropout=0.0, norm=p["norm"])
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index, batch=b.batch))


class SuperGATConvOp(AbsGNNOp):
    op_type = "SuperGATConv"

    def sample_and_constrain(self, ctx, in_channels, out_channels, scope_info, rng):
        heads = rng.choice([1, 2])
        concat = rng.choice([True, False])
        hc = ctx.new_int("sgat_hc", 2, 16)
        return [out_channels == (hc * heads if concat else hc)], {
            "in_c": in_channels[0],
            "hc": hc,
            "heads": heads,
            "concat": concat,
            "attention_type": rng.choice(["MX", "SD"]),
        }

    @classmethod
    def materialize(cls, p: Dict[str, Any]) -> StepModule:
        m = pyg_conv.SuperGATConv(
            p["in_c"],
            p["hc"],
            heads=p["heads"],
            concat=p["concat"],
            dropout=0.0,
            attention_type=p["attention_type"],
        )
        return _LambdaStep(m, lambda mod, xs, b: mod(xs[0], b.edge_index, batch=b.batch))


CONV_OPS: List[Type[AbsGNNOp]] = [
    GCNConvOp,
    SAGEConvOp,
    GraphConvOp,
    GATConvOp,
    GATv2ConvOp,
    TransformerConvOp,
    GINConvOp,
    GINEConvOp,
    ChebConvOp,
    TAGConvOp,
    SGConvOp,
    SSGConvOp,
    APPNPOp,
    ARMAConvOp,
    AGNNConvOp,
    ResGatedGraphConvOp,
    GatedGraphConvOp,
    MFConvOp,
    RGCNConvOp,
    FastRGCNConvOp,
    RGATConvOp,
    FiLMConvOp,
    GMMConvOp,
    NNConvOp,
    CGConvOp,
    EdgeConvOp,
    FeaStConvOp,
    PointNetConvOp,
    PointTransformerConvOp,
    LEConvOp,
    ClusterGCNConvOp,
    GENConvOp,
    PNAConvOp,
    EGConvOp,
    PDNConvOp,
    GeneralConvOp,
    SimpleConvOp,
    LGConvOp,
    AntiSymmetricConvOp,
    DirGNNConvOp,
    MixHopConvOp,
    FAConvOp,
    GCN2ConvOp,
    GPSConvOp,
    SuperGATConvOp,
]
