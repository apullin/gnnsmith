"""Smoke tests for GNNSmith v0.2.0: Z3 synthesis, coverage monitor, reducer, and WhiteFox AST inspector."""

import torch_geometric.edge_index as ei_mod

from gnnsmith.coverage_loop import PyGCoverageMonitor
from gnnsmith.graph_gen import synthesize_gnn_dag
from gnnsmith.materialize import materialize_ir
from gnnsmith.ops.pyg_conv import GATv2ConvOp, GCNConvOp, SAGEConvOp
from gnnsmith.ops.pyg_norm_pool import ChannelConcatOp, ResidualAddOp
from gnnsmith.reducer import reduce_failing_case
from gnnsmith.topology import generate_graph_bundle
from gnnsmith.whitefox.inspector import extract_branch_targets


def test_z3_dag_synthesis_and_coverage_monitor() -> None:
    ops = [GCNConvOp, SAGEConvOp, GATv2ConvOp, ResidualAddOp, ChannelConcatOp]
    with PyGCoverageMonitor() as mon:
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

    report = mon.find_uncovered_branches([ei_mod])
    assert report.files_executed >= 3
    assert report.lines_executed >= 20


def test_delta_debugging_reducer() -> None:
    ops = [GCNConvOp, SAGEConvOp, GATv2ConvOp]
    ir = synthesize_gnn_dag(seed=12, min_ops=3, max_ops=3, allowed_ops=ops, topology_kind="erdos_renyi")
    s0 = ir.scopes[0]
    bundle = generate_graph_bundle(
        num_nodes=18,
        num_edges=40,
        in_channels=int(ir.values["x0"].channels),
        num_graphs=3,
        edge_dim=s0.edge_dim,
        num_relations=s0.num_relations,
        seed=12,
    )
    target_op = ir.insts[1].op_type
    reduced = reduce_failing_case(
        ir,
        bundle,
        predicate=lambda cand_ir, cand_b: any(i.op_type == target_op for i in cand_ir.insts) and cand_b.num_nodes >= 2,
    )
    assert len(reduced.ir.insts) == 1
    assert reduced.bundle.num_nodes == 2
    assert "edge_index = torch.tensor(" in reduced.repro_code


def test_whitefox_ast_branch_extractor() -> None:
    targets = extract_branch_targets(ei_mod)
    assert len(targets) > 50
    assert any(t.has_ptr_or_sort_branch for t in targets)
