"""Z3-backed random GNN computation DAG generator."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple, Type

import z3

from gnnsmith.constraints import Z3GraphContext
from gnnsmith.ir import GraphIR, GraphScope, OpInst, SymValue, TensorRole
from gnnsmith.ops import ALL_OPS, OP_REGISTRY
from gnnsmith.ops.base import AbsGNNOp
from gnnsmith.ops.pyg_norm_pool import GlobalPoolReadoutOp
from gnnsmith.topology import TOPOLOGY_KINDS


def synthesize_gnn_dag(
    *,
    seed: int = 0,
    min_ops: int = 2,
    max_ops: int = 7,
    allowed_ops: Optional[List[Type[AbsGNNOp]]] = None,
    topology_kind: Optional[str] = None,
    include_readout_prob: float = 0.25,
    hetero_only: bool = False,
) -> GraphIR:
    """Synthesize a valid, Z3-solved multi-branch GNN DAG."""
    rng = random.Random(seed)
    ctx = Z3GraphContext(rng=rng, timeout_ms=2000)

    candidates = list(allowed_ops) if allowed_ops is not None else [
        op for op in ALL_OPS if op is not GlobalPoolReadoutOp
    ]
    if hetero_only:
        candidates = [op for op in candidates if op.capabilities.supports_to_hetero]

    topo = topology_kind or rng.choice(TOPOLOGY_KINDS)
    if hetero_only and topo in ("single_node",):
        topo = "erdos_renyi"

    # Initial symbolic GraphScope 0
    num_graphs = 1 if topo == "single_node" else rng.choice([1, 2, 3])
    num_nodes = 1 if topo == "single_node" else rng.randint(max(3, num_graphs * 2), 28)
    num_edges = 0 if topo == "empty_edges" else rng.randint(0, num_nodes * 4)
    num_relations = rng.choice([2, 4, 6])
    edge_dim = rng.choice([2, 4, 8])

    scope0 = GraphScope(
        scope_id=0,
        num_nodes=num_nodes,
        num_edges=num_edges,
        num_graphs=num_graphs,
        num_relations=num_relations,
        edge_dim=edge_dim,
    )
    scopes: Dict[int, GraphScope] = {0: scope0}
    current_scope_id = 0

    c_in = ctx.new_int("c_in", 4, 32)
    x0 = SymValue(name="x0", role=TensorRole.NODE_FEAT, scope_id=0, channels=c_in)
    values: Dict[str, SymValue] = {"x0": x0}
    scope_node_vars: Dict[int, List[str]] = {0: ["x0"]}
    insts: List[OpInst] = []

    target_ops = rng.randint(min_ops, max_ops)
    hint_pairs: List[Tuple[z3.ArithRef, int]] = [(c_in, rng.choice([4, 8, 12, 16, 24]))]

    attempts = 0
    while len(insts) < target_ops and attempts < target_ops * 8:
        attempts += 1
        op_cls = rng.choice(candidates)
        if op_cls.capabilities.mutates_topology and (hetero_only or current_scope_id >= 1 or num_nodes < 4):
            continue

        avail = scope_node_vars[current_scope_id]
        if op_cls.arity > len(avail):
            # Allow reusing the same tensor twice (e.g., ResidualAdd(x, x) or FAConv(x, x0))
            if len(avail) == 0:
                continue
            chosen_inputs = [rng.choice(avail) for _ in range(op_cls.arity)]
        else:
            # Prefer recent + skip connection mix
            if op_cls.arity == 2 and len(avail) >= 2:
                chosen_inputs = [avail[-1], rng.choice(avail[:-1])]
            else:
                chosen_inputs = [avail[-1]] if rng.random() < 0.7 else [rng.choice(avail)]

        in_syms = [values[v].channels for v in chosen_inputs]
        c_out = ctx.new_int(f"c_{len(insts) + 1}", 2, 64)

        op_spec = op_cls()
        scope_info = {
            "num_nodes": scopes[current_scope_id].num_nodes,
            "num_edges": scopes[current_scope_id].num_edges,
            "num_graphs": scopes[current_scope_id].num_graphs,
            "num_relations": scopes[current_scope_id].num_relations,
            "edge_dim": scopes[current_scope_id].edge_dim,
        }
        constraints, raw_params = op_spec.sample_and_constrain(
            ctx, in_syms, c_out, scope_info, rng
        )

        if not ctx.try_commit(*constraints):
            continue

        hint_pairs.append((c_out, rng.choice([4, 8, 12, 16, 24, 32])))

        out_scope_id = current_scope_id
        if op_cls.capabilities.mutates_topology:
            out_scope_id = current_scope_id + 1
            scopes[out_scope_id] = GraphScope(
                scope_id=out_scope_id,
                num_nodes=max(1, num_nodes // 2),
                num_edges=max(0, num_edges // 2),
                num_graphs=num_graphs,
                num_relations=num_relations,
                edge_dim=edge_dim,
                parent_scope=current_scope_id,
            )
            current_scope_id = out_scope_id
            scope_node_vars[current_scope_id] = []

        out_name = f"v{len(insts) + 1}"
        values[out_name] = SymValue(
            name=out_name,
            role=op_cls.out_role,
            scope_id=out_scope_id,
            channels=c_out,
        )
        if op_cls.out_role == TensorRole.NODE_FEAT:
            scope_node_vars[out_scope_id].append(out_name)

        insts.append(
            OpInst(
                inst_id=len(insts),
                op_type=op_cls.op_type,
                inputs=chosen_inputs,
                output=out_name,
                in_scope=scope_info and (out_scope_id if not op_cls.capabilities.mutates_topology else out_scope_id - 1),
                out_scope=out_scope_id,
                params=raw_params,
            )
        )

    # Optionally append a GlobalPoolReadoutOp at the end
    last_var = scope_node_vars[current_scope_id][-1]
    if not hetero_only and rng.random() < include_readout_prob and len(insts) >= 1:
        ro_cls = GlobalPoolReadoutOp
        ro_out = ctx.new_int("c_readout", 2, 128)
        ro_spec = ro_cls()
        ro_c, ro_p = ro_spec.sample_and_constrain(
            ctx,
            [values[last_var].channels],
            ro_out,
            {"num_graphs": num_graphs, "edge_dim": edge_dim, "num_relations": num_relations},
            rng,
        )
        if ctx.try_commit(*ro_c):
            out_name = f"v{len(insts) + 1}"
            values[out_name] = SymValue(
                name=out_name,
                role=TensorRole.GRAPH_FEAT,
                scope_id=current_scope_id,
                channels=ro_out,
            )
            insts.append(
                OpInst(
                    inst_id=len(insts),
                    op_type=ro_cls.op_type,
                    inputs=[last_var],
                    output=out_name,
                    in_scope=current_scope_id,
                    out_scope=current_scope_id,
                    params=ro_p,
                )
            )
            last_var = out_name

    model = ctx.solve(randomize_hints=hint_pairs)
    if model is None:
        raise RuntimeError(f"Z3 failed to solve GNNSmith DAG (seed={seed})")

    # Concretize all symbolic channels & op parameters
    for val in values.values():
        val.channels = ctx.eval_int(model, val.channels)

    for inst in insts:
        op_cls = OP_REGISTRY[inst.op_type]
        inst.params = op_cls.resolve_params(model, inst.params)

    return GraphIR(
        scopes=scopes,
        values=values,
        insts=insts,
        input_value="x0",
        output_value=last_var,
        topology_kind=topo,
        seed=seed,
    )
