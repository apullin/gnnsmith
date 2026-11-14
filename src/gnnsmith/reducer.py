"""Automatic Delta-Debugging Reducer (`gnnsmith reduce`) for failing GNN DAGs and topologies."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, List

import torch

from gnnsmith.ir import GraphIR, OpInst, SymValue, TensorRole
from gnnsmith.materialize import materialize_ir
from gnnsmith.topology import GraphBundle, generate_graph_bundle


@dataclass
class ReducedCase:
    ir: GraphIR
    bundle: GraphBundle
    repro_code: str


def _slice_single_op_ir(ir: GraphIR, inst_idx: int) -> GraphIR | None:
    """Extract a 1-op GraphIR from instruction `inst_idx` if it operates on a single NODE_FEAT input."""
    inst = ir.insts[inst_idx]
    if len(inst.inputs) != 1:
        return None
    in_val = ir.values[inst.inputs[0]]
    out_val = ir.values[inst.output]
    if in_val.role != TensorRole.NODE_FEAT:
        return None

    scope0 = copy.deepcopy(ir.scopes[inst.in_scope])
    scope0.scope_id = 0
    scope0.parent_scope = None
    scopes = {0: scope0}
    out_scope_id = 0
    if inst.out_scope != inst.in_scope:
        scope1 = copy.deepcopy(ir.scopes[inst.out_scope])
        scope1.scope_id = 1
        scope1.parent_scope = 0
        scopes[1] = scope1
        out_scope_id = 1

    new_x0 = SymValue(name="x0", role=TensorRole.NODE_FEAT, scope_id=0, channels=int(in_val.channels))
    new_v1 = SymValue(name="v1", role=out_val.role, scope_id=out_scope_id, channels=int(out_val.channels))
    new_inst = OpInst(
        inst_id=0,
        op_type=inst.op_type,
        inputs=["x0"],
        output="v1",
        in_scope=0,
        out_scope=out_scope_id,
        params=copy.deepcopy(inst.params),
    )
    return GraphIR(
        scopes=scopes,
        values={"x0": new_x0, "v1": new_v1},
        insts=[new_inst],
        input_value="x0",
        output_value="v1",
        topology_kind=ir.topology_kind,
        seed=ir.seed,
    )


def reduce_failing_case(
    ir: GraphIR,
    bundle: GraphBundle,
    predicate: Callable[[GraphIR, GraphBundle], bool],
) -> ReducedCase:
    """Delta-debug a failing `(GraphIR, GraphBundle)` down to minimal ops, nodes, and edges."""
    cur_ir = copy.deepcopy(ir)
    cur_bundle = bundle

    # Pass 1: Try isolating a single responsible operator in the DAG
    if len(cur_ir.insts) > 1:
        for idx in range(len(cur_ir.insts)):
            cand_ir = _slice_single_op_ir(cur_ir, idx)
            if cand_ir is None:
                continue
            in_c = int(cand_ir.values["x0"].channels)
            cand_b = generate_graph_bundle(
                num_nodes=cur_bundle.num_nodes,
                num_edges=cur_bundle.edge_index.size(1),
                in_channels=in_c,
                num_graphs=cur_bundle.num_graphs,
                edge_dim=cand_ir.scopes[0].edge_dim,
                num_relations=cand_ir.scopes[0].num_relations,
                topology_kind=cand_ir.topology_kind,
                seed=cand_ir.seed,
            )
            if predicate(cand_ir, cand_b):
                cur_ir = cand_ir
                cur_bundle = cand_b
                break

    # Pass 2: Shrink graph node count `N`, edge count `E`, and batch count `B`
    in_c = int(cur_ir.values[cur_ir.input_value].channels)
    s0 = cur_ir.scopes[0]
    for cand_b_cnt in sorted({1, cur_bundle.num_graphs}):
        for cand_n in sorted({1, 2, 3, 4, 6, cur_bundle.num_nodes}):
            if cand_n < cand_b_cnt or cand_n > cur_bundle.num_nodes:
                continue
            for cand_e in sorted({0, 1, 2, 4, cur_bundle.edge_index.size(1)}):
                if cand_e > cur_bundle.edge_index.size(1):
                    continue
                cand_bundle = generate_graph_bundle(
                    num_nodes=cand_n,
                    num_edges=cand_e,
                    in_channels=in_c,
                    num_graphs=cand_b_cnt,
                    edge_dim=s0.edge_dim,
                    num_relations=s0.num_relations,
                    topology_kind=cur_ir.topology_kind,
                    seed=cur_ir.seed,
                )
                if predicate(cur_ir, cand_bundle):
                    cur_bundle = cand_bundle
                    s0.num_nodes = cand_bundle.num_nodes
                    s0.num_edges = cand_bundle.edge_index.size(1)
                    s0.num_graphs = cand_bundle.num_graphs
                    break
            else:
                continue
            break

    repro_code = render_standalone_repro(cur_ir, cur_bundle)
    return ReducedCase(ir=cur_ir, bundle=cur_bundle, repro_code=repro_code)


def render_standalone_repro(ir: GraphIR, bundle: GraphBundle) -> str:
    """Emit a self-contained Python reproduction script for a reduced `(GraphIR, GraphBundle)`."""
    ops_desc = ", ".join(f"{i.op_type}({i.params})" for i in ir.insts)
    return (
        "import torch\n"
        "from gnnsmith.ir import GraphIR\n"
        f"# Reduced DAG ({len(ir.insts)} op(s)): {ops_desc}\n"
        f"# Reduced Graph: N={bundle.num_nodes}, E={bundle.edge_index.size(1)}, "
        f"B={bundle.num_graphs}, topo={bundle.topology_kind!r}\n"
        f"edge_index = torch.tensor({bundle.edge_index.tolist()}, dtype=torch.long)\n"
        f"batch = torch.tensor({bundle.batch.tolist()}, dtype=torch.long)\n"
        f"x = torch.randn({bundle.num_nodes}, {bundle.x.size(1)})\n"
    )
