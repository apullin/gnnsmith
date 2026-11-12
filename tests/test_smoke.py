"""Smoke tests for GNNSmith Z3 DAG synthesis, materialization, and WhiteFox inspector."""

import torch_geometric.edge_index as ei_mod

from gnnsmith.graph_gen import synthesize_gnn_dag
from gnnsmith.materialize import materialize_ir
from gnnsmith.ops.pyg_conv import GATv2ConvOp, GCNConvOp, SAGEConvOp
from gnnsmith.ops.pyg_norm_pool import ChannelConcatOp, ResidualAddOp
from gnnsmith.topology import generate_graph_bundle
from gnnsmith.whitefox.inspector import extract_branch_targets


def test_z3_dag_synthesis_and_execution() -> None:
    ops = [GCNConvOp, SAGEConvOp, GATv2ConvOp, ResidualAddOp, ChannelConcatOp]
    for seed in range(5):
        ir = synthesize_gnn_dag(seed=seed, min_ops=2, max_ops=4, allowed_ops=ops)
        s0 = ir.scopes[0]
        bundle = generate_graph_bundle(
            num_nodes=int(s0.num_nodes),
            num_edges=int(s0.num_edges),
            in_channels=int(ir.values[ir.input_value].channels),
            num_graphs=int(s0.num_graphs),
            edge_dim=s0.edge_dim,
            num_relations=s0.num_relations,
            topology_kind="erdos_renyi",
            seed=seed,
        )
        model = materialize_ir(ir, seed=seed)
        out = model(bundle)
        assert out.ndim == 2


def test_whitefox_ast_branch_extractor() -> None:
    targets = extract_branch_targets(ei_mod)
    assert len(targets) > 50
    assert any(t.has_ptr_or_sort_branch for t in targets)
