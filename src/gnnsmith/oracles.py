"""Differential and Metamorphic Oracles for Graph Neural Networks.

Implements 6 automatic correctness checks requiring zero ground-truth labels:
1. Finite / Crash Oracle: Forward + backward autograd must not crash or emit NaN/Inf on finite inputs.
2. Edge-Order Permutation Invariance: Shuffling columns of `edge_index` (+ `edge_attr`/`edge_weight`/`edge_type`)
   must leave outputs and input gradients invariant.
3. Within-Batch Node Permutation Equivariance: Permuting node IDs within each batch graph `P`
   must satisfy `f(P X, P E P^T) == P f(X, E)` (or `== f(X, E)` for graph-level readout).
4. Disjoint Batching Invariance: Running `[f(G_0), ..., f(G_{B-1})]` separately vs `f(Batch)` must match.
5. EdgeIndex Subclass Parity: Plain `LongTensor` `edge_index` vs PyG `EdgeIndex` tensor subclass.
6. Device Parity (CPU vs MPS): Forward outputs and `x.grad` on `mps` vs `cpu`.
"""

from __future__ import annotations

import copy
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch_geometric import EdgeIndex
from torch_geometric.nn.conv import MessagePassing

from gnnsmith.ir import GraphIR, TensorRole
from gnnsmith.materialize import MaterializedGNN
from gnnsmith.ops import OP_REGISTRY
from gnnsmith.topology import GraphBundle


@dataclass
class BugFinding:
    oracle: str
    category: str          # "crash" | "nan_inf" | "shape_mismatch" | "numerical_divergence" | "grad_divergence"
    summary: str
    detail: str
    op_types: List[str]
    topology_kind: str
    seed: int
    max_diff: Optional[float] = None
    traceback_str: str = ""
    signature_key: str = ""
    extra: Dict[str, object] = field(default_factory=dict)


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 and b.numel() == 0:
        return 0.0
    return float((a.detach().cpu().float() - b.detach().cpu().float()).abs().max().item())


def _is_close(a: torch.Tensor, b: torch.Tensor, atol: float = 2e-3, rtol: float = 5e-3) -> bool:
    if a.shape != b.shape:
        return False
    if a.numel() == 0:
        return True
    return bool(torch.allclose(a.detach().cpu().float(), b.detach().cpu().float(), atol=atol, rtol=rtol))


def _permute_nodes_within_batch(bundle: GraphBundle, seed: int) -> tuple[GraphBundle, torch.Tensor]:
    """Generate a permutation `perm` that shuffles node IDs within each batch segment (preserving sorted `batch`)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 991)
    perm = torch.arange(bundle.num_nodes, dtype=torch.long)
    for g in range(bundle.num_graphs):
        idx = torch.nonzero(bundle.batch == g, as_tuple=False).view(-1)
        if idx.numel() > 1:
            local_order = idx[torch.randperm(idx.numel(), generator=gen)]
            perm[idx] = local_order

    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(bundle.num_nodes, dtype=torch.long)

    ei_perm = inv_perm[bundle.edge_index] if bundle.edge_index.numel() > 0 else bundle.edge_index.clone()
    return (
        GraphBundle(
            x=bundle.x[perm],
            x_0=bundle.x_0[perm],
            pos=bundle.pos[perm],
            edge_index=ei_perm,
            edge_weight=bundle.edge_weight.clone(),
            edge_attr=bundle.edge_attr.clone(),
            edge_type=bundle.edge_type.clone(),
            batch=bundle.batch[perm],
            num_nodes=bundle.num_nodes,
            num_graphs=bundle.num_graphs,
            num_relations=bundle.num_relations,
            topology_kind=bundle.topology_kind,
        ),
        perm,
    )


def _permute_edges(bundle: GraphBundle, seed: int) -> GraphBundle:
    e = bundle.edge_index.size(1)
    if e <= 1:
        return bundle
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 773)
    ep = torch.randperm(e, generator=gen)
    return GraphBundle(
        x=bundle.x.clone(),
        x_0=bundle.x_0.clone(),
        pos=bundle.pos.clone(),
        edge_index=bundle.edge_index[:, ep],
        edge_weight=bundle.edge_weight[ep],
        edge_attr=bundle.edge_attr[ep],
        edge_type=bundle.edge_type[ep],
        batch=bundle.batch.clone(),
        num_nodes=bundle.num_nodes,
        num_graphs=bundle.num_graphs,
        num_relations=bundle.num_relations,
        topology_kind=bundle.topology_kind,
    )


def _make_sig(oracle: str, category: str, op_types: List[str], tb_or_msg: str) -> str:
    last_line = tb_or_msg.strip().splitlines()[-1] if tb_or_msg.strip() else category
    # Normalize memory addresses / exact numbers
    last_line = last_line[:120]
    return f"{oracle}|{category}|{'+'.join(sorted(set(op_types)))}|{last_line}"


def run_all_oracles(
    ir: GraphIR,
    model: MaterializedGNN,
    bundle: GraphBundle,
    *,
    check_mps: bool = True,
    check_decomposed: bool = True,
    atol: float = 2e-3,
    rtol: float = 5e-3,
) -> List[BugFinding]:
    """Run all applicable differential & metamorphic oracles on a materialized GNN + GraphBundle."""
    findings: List[BugFinding] = []
    op_types = [inst.op_type for inst in ir.insts]
    caps = [OP_REGISTRY[t].capabilities for t in op_types]
    out_role = ir.values[ir.output_value].role
    mutates_topo = any(c.mutates_topology for c in caps)

    model.eval()

    # --- Oracle 1: Baseline CPU Forward + Backward + Finite Check ---
    b_ref = bundle.clone_requires_grad()
    try:
        model.zero_grad(set_to_none=True)
        out_ref = model(b_ref)
        loss_ref = (out_ref * torch.linspace(0.5, 1.5, out_ref.numel(), device=out_ref.device).view_as(out_ref)).sum()
        loss_ref.backward()
    except Exception as exc:
        tb = traceback.format_exc()
        findings.append(
            BugFinding(
                oracle="cpu_fwd_bwd",
                category="crash",
                summary=f"{type(exc).__name__}: {exc}",
                detail=ir.summary(),
                op_types=op_types,
                topology_kind=bundle.topology_kind,
                seed=ir.seed,
                traceback_str=tb,
                signature_key=_make_sig("cpu_fwd_bwd", "crash", op_types, f"{type(exc).__name__}: {exc}"),
            )
        )
        return findings

    expected_rows = bundle.num_graphs if out_role == TensorRole.GRAPH_FEAT else None
    if expected_rows is not None and out_ref.size(0) != expected_rows:
        msg = f"Expected {expected_rows} graph rows, got shape {tuple(out_ref.shape)}"
        findings.append(
            BugFinding(
                oracle="output_shape",
                category="shape_mismatch",
                summary=msg,
                detail=ir.summary(),
                op_types=op_types,
                topology_kind=bundle.topology_kind,
                seed=ir.seed,
                signature_key=_make_sig("output_shape", "shape_mismatch", op_types, msg),
            )
        )
        return findings

    if not mutates_topo and out_role == TensorRole.NODE_FEAT and out_ref.size(0) != bundle.num_nodes:
        msg = f"Expected {bundle.num_nodes} node rows, got shape {tuple(out_ref.shape)} (likely missing dim_size!)"
        findings.append(
            BugFinding(
                oracle="output_shape",
                category="shape_mismatch",
                summary=msg,
                detail=ir.summary(),
                op_types=op_types,
                topology_kind=bundle.topology_kind,
                seed=ir.seed,
                signature_key=_make_sig("output_shape", "shape_mismatch", op_types, "node_row_count_mismatch"),
            )
        )
        return findings

    if not torch.isfinite(out_ref).all():
        msg = f"Non-finite forward output (nan={torch.isnan(out_ref).any().item()}, inf={torch.isinf(out_ref).any().item()})"
        findings.append(
            BugFinding(
                oracle="finite_check",
                category="nan_inf",
                summary=msg,
                detail=ir.summary(),
                op_types=op_types,
                topology_kind=bundle.topology_kind,
                seed=ir.seed,
                signature_key=_make_sig("finite_check", "nan_inf", op_types, msg),
            )
        )
        return findings

    gx_ref = b_ref.x.grad.detach().clone() if b_ref.x.grad is not None else None
    if gx_ref is not None and not torch.isfinite(gx_ref).all():
        msg = f"Non-finite input gradient x.grad (nan={torch.isnan(gx_ref).any().item()}, inf={torch.isinf(gx_ref).any().item()})"
        findings.append(
            BugFinding(
                oracle="finite_grad_check",
                category="nan_inf",
                summary=msg,
                detail=ir.summary(),
                op_types=op_types,
                topology_kind=bundle.topology_kind,
                seed=ir.seed,
                signature_key=_make_sig("finite_grad_check", "nan_inf", op_types, msg),
            )
        )
        return findings

    # --- Oracle 2: Edge-Order Permutation Invariance ---
    if bundle.edge_index.size(1) > 1 and not mutates_topo:
        b_ep = _permute_edges(bundle, ir.seed).clone_requires_grad()
        try:
            model.zero_grad(set_to_none=True)
            out_ep = model(b_ep)
            if not _is_close(out_ref, out_ep, atol=atol, rtol=rtol):
                diff = _max_abs_diff(out_ref, out_ep)
                if diff > 1e-2:
                    msg = f"Edge-order permutation changed output by max_diff={diff:.4e}"
                    findings.append(
                        BugFinding(
                            oracle="edge_permutation",
                            category="numerical_divergence",
                            summary=msg,
                            detail=ir.summary(),
                            op_types=op_types,
                            topology_kind=bundle.topology_kind,
                            seed=ir.seed,
                            max_diff=diff,
                            signature_key=_make_sig("edge_permutation", "numerical_divergence", op_types, "edge_order"),
                        )
                    )
        except Exception as exc:
            tb = traceback.format_exc()
            findings.append(
                BugFinding(
                    oracle="edge_permutation",
                    category="crash",
                    summary=f"{type(exc).__name__}: {exc}",
                    detail=ir.summary(),
                    op_types=op_types,
                    topology_kind=bundle.topology_kind,
                    seed=ir.seed,
                    traceback_str=tb,
                    signature_key=_make_sig("edge_permutation", "crash", op_types, f"{type(exc).__name__}: {exc}"),
                )
            )

    # --- Oracle 3: Within-Batch Node Permutation Equivariance ---
    # Skip SuperGATConv (samples random negative edges internally on forward!) and pooling tie-breaks
    has_stochastic_neg = "SuperGATConv" in op_types
    if bundle.num_nodes > 1 and not mutates_topo and not has_stochastic_neg and all(c.is_permutation_equivariant for c in caps):
        b_np_raw, perm = _permute_nodes_within_batch(bundle, ir.seed)
        b_np = b_np_raw.clone_requires_grad()
        try:
            model.zero_grad(set_to_none=True)
            out_np = model(b_np)
            expected = out_ref[perm] if out_role == TensorRole.NODE_FEAT else out_ref
            if not _is_close(expected, out_np, atol=atol, rtol=rtol):
                diff = _max_abs_diff(expected, out_np)
                if diff > 1e-2:
                    msg = f"Node permutation equivariance violated: max_diff={diff:.4e}"
                    findings.append(
                        BugFinding(
                            oracle="node_equivariance",
                            category="numerical_divergence",
                            summary=msg,
                            detail=ir.summary(),
                            op_types=op_types,
                            topology_kind=bundle.topology_kind,
                            seed=ir.seed,
                            max_diff=diff,
                            signature_key=_make_sig("node_equivariance", "numerical_divergence", op_types, "perm_equiv"),
                        )
                    )
        except Exception as exc:
            tb = traceback.format_exc()
            findings.append(
                BugFinding(
                    oracle="node_equivariance",
                    category="crash",
                    summary=f"{type(exc).__name__}: {exc}",
                    detail=ir.summary(),
                    op_types=op_types,
                    topology_kind=bundle.topology_kind,
                    seed=ir.seed,
                    traceback_str=tb,
                    signature_key=_make_sig("node_equivariance", "crash", op_types, f"{type(exc).__name__}: {exc}"),
                )
            )

    # --- Oracle 4: Disjoint Batching Invariance ([f(G_0), f(G_1)] vs f(Batch)) ---
    # Note: ChebConv without explicit lambda_max normalizes by max degree across whole edge_index if batch not handled,
    # and CGConv(batch_norm=True) / GENConv(norm='batch') / GPSConv(norm='batch_norm') use global BatchNorm1d stats in train,
    # though in eval() mode running_mean=0, running_var=1 so eval() IS batch-invariant!
    if (
        bundle.num_graphs >= 2
        and not mutates_topo
        and not has_stochastic_neg
        and all(c.is_batch_invariant for c in caps)
    ):
        sub_list = bundle.to_pyg_data_list()
        try:
            sub_outs: List[torch.Tensor] = []
            for d in sub_list:
                sub_b = GraphBundle(
                    x=d.x,
                    x_0=d.x_0,
                    pos=d.pos,
                    edge_index=d.edge_index,
                    edge_weight=d.edge_weight,
                    edge_attr=d.edge_attr,
                    edge_type=d.edge_type,
                    batch=torch.zeros(d.num_nodes, dtype=torch.long),
                    num_nodes=d.num_nodes,
                    num_graphs=1,
                    num_relations=bundle.num_relations,
                    topology_kind=bundle.topology_kind,
                )
                sub_outs.append(model(sub_b))
            cat_sub = torch.cat(sub_outs, dim=0)
            if cat_sub.shape != out_ref.shape:
                msg = f"Disjoint batch shape mismatch: per-graph cat {tuple(cat_sub.shape)} vs batched {tuple(out_ref.shape)}"
                findings.append(
                    BugFinding(
                        oracle="batch_invariance",
                        category="shape_mismatch",
                        summary=msg,
                        detail=ir.summary(),
                        op_types=op_types,
                        topology_kind=bundle.topology_kind,
                        seed=ir.seed,
                        signature_key=_make_sig("batch_invariance", "shape_mismatch", op_types, msg),
                    )
                )
            elif not _is_close(cat_sub, out_ref, atol=atol, rtol=rtol):
                diff = _max_abs_diff(cat_sub, out_ref)
                if diff > 1e-2:
                    msg = f"Cross-graph batch leakage! Per-graph vs Batched max_diff={diff:.4e}"
                    findings.append(
                        BugFinding(
                            oracle="batch_invariance",
                            category="numerical_divergence",
                            summary=msg,
                            detail=ir.summary(),
                            op_types=op_types,
                            topology_kind=bundle.topology_kind,
                            seed=ir.seed,
                            max_diff=diff,
                            signature_key=_make_sig("batch_invariance", "numerical_divergence", op_types, "batch_leak"),
                        )
                    )
        except Exception as exc:
            tb = traceback.format_exc()
            findings.append(
                BugFinding(
                    oracle="batch_invariance",
                    category="crash",
                    summary=f"{type(exc).__name__}: {exc}",
                    detail=ir.summary(),
                    op_types=op_types,
                    topology_kind=bundle.topology_kind,
                    seed=ir.seed,
                    traceback_str=tb,
                    signature_key=_make_sig("batch_invariance", "crash", op_types, f"{type(exc).__name__}: {exc}"),
                )
            )

    # --- Oracle 5: PyG `EdgeIndex` Subclass vs Plain `LongTensor` ---
    if not mutates_topo and not has_stochastic_neg and all(c.supports_edge_index_subclass for c in caps):
        try:
            ei_sub = EdgeIndex(
                bundle.edge_index.clone(),
                sparse_size=(bundle.num_nodes, bundle.num_nodes),
            )
            b_ei = bundle.clone_requires_grad()
            b_ei.edge_index = ei_sub
            model.zero_grad(set_to_none=True)
            out_ei = model(b_ei)
            if not _is_close(out_ref, out_ei, atol=atol, rtol=rtol):
                diff = _max_abs_diff(out_ref, out_ei)
                if diff > 1e-2:
                    msg = f"EdgeIndex subclass diverged from LongTensor: max_diff={diff:.4e}"
                    findings.append(
                        BugFinding(
                            oracle="edge_index_subclass",
                            category="numerical_divergence",
                            summary=msg,
                            detail=ir.summary(),
                            op_types=op_types,
                            topology_kind=bundle.topology_kind,
                            seed=ir.seed,
                            max_diff=diff,
                            signature_key=_make_sig("edge_index_subclass", "numerical_divergence", op_types, "ei_sub"),
                        )
                    )
        except Exception as exc:
            tb = traceback.format_exc()
            findings.append(
                BugFinding(
                    oracle="edge_index_subclass",
                    category="crash",
                    summary=f"{type(exc).__name__}: {exc}",
                    detail=ir.summary(),
                    op_types=op_types,
                    topology_kind=bundle.topology_kind,
                    seed=ir.seed,
                    traceback_str=tb,
                    signature_key=_make_sig("edge_index_subclass", "crash", op_types, f"{type(exc).__name__}: {exc}"),
                )
            )

    # --- Oracle 6: Decomposed Layers (decomposed_layers=2 vs 1) ---
    if check_decomposed and bundle.edge_index.size(1) >= 2 and not mutates_topo and not has_stochastic_neg:
        try:
            model_dec = copy.deepcopy(model)
            changed_any = False
            for m in model_dec.modules():
                if isinstance(m, MessagePassing):
                    try:
                        m.decomposed_layers = 2
                        changed_any = True
                    except Exception:
                        pass
            if changed_any:
                b_dec = bundle.clone_requires_grad()
                out_dec = model_dec(b_dec)
                if not _is_close(out_ref, out_dec, atol=atol, rtol=rtol):
                    diff = _max_abs_diff(out_ref, out_dec)
                    if diff > 1e-2:
                        msg = f"decomposed_layers=2 diverged from decomposed_layers=1: max_diff={diff:.4e}"
                        findings.append(
                            BugFinding(
                                oracle="decomposed_layers",
                                category="numerical_divergence",
                                summary=msg,
                                detail=ir.summary(),
                                op_types=op_types,
                                topology_kind=bundle.topology_kind,
                                seed=ir.seed,
                                max_diff=diff,
                                signature_key=_make_sig("decomposed_layers", "numerical_divergence", op_types, "decomp"),
                            )
                        )
        except Exception as exc:
            tb = traceback.format_exc()
            findings.append(
                BugFinding(
                    oracle="decomposed_layers",
                    category="crash",
                    summary=f"{type(exc).__name__}: {exc}",
                    detail=ir.summary(),
                    op_types=op_types,
                    topology_kind=bundle.topology_kind,
                    seed=ir.seed,
                    traceback_str=tb,
                    signature_key=_make_sig("decomposed_layers", "crash", op_types, f"{type(exc).__name__}: {exc}"),
                )
            )

    # --- Oracle 7: Device Parity (CPU vs MPS Forward + Backward) ---
    if check_mps and torch.backends.mps.is_available() and not has_stochastic_neg:
        try:
            model_mps = copy.deepcopy(model).to("mps")
            model_mps.eval()
            b_mps = bundle.to("mps").clone_requires_grad()
            out_mps = model_mps(b_mps)
            w_mps = torch.linspace(0.5, 1.5, out_mps.numel(), device="mps").view_as(out_mps)
            (out_mps * w_mps).sum().backward()

            if not torch.isfinite(out_mps).all():
                msg = f"MPS produced NaN/Inf whereas CPU output was finite!"
                findings.append(
                    BugFinding(
                        oracle="cpu_vs_mps",
                        category="nan_inf",
                        summary=msg,
                        detail=ir.summary(),
                        op_types=op_types,
                        topology_kind=bundle.topology_kind,
                        seed=ir.seed,
                        signature_key=_make_sig("cpu_vs_mps", "nan_inf", op_types, msg),
                    )
                )
            elif not _is_close(out_ref, out_mps, atol=5e-3, rtol=1e-2):
                diff = _max_abs_diff(out_ref, out_mps)
                if diff > 2e-2:
                    msg = f"CPU vs MPS forward divergence: max_diff={diff:.4e}"
                    findings.append(
                        BugFinding(
                            oracle="cpu_vs_mps",
                            category="numerical_divergence",
                            summary=msg,
                            detail=ir.summary(),
                            op_types=op_types,
                            topology_kind=bundle.topology_kind,
                            seed=ir.seed,
                            max_diff=diff,
                            signature_key=_make_sig("cpu_vs_mps", "numerical_divergence", op_types, "fwd_diff"),
                        )
                    )
            elif gx_ref is not None and b_mps.x.grad is not None:
                if not torch.isfinite(b_mps.x.grad).all():
                    msg = "MPS produced NaN/Inf in x.grad whereas CPU x.grad was finite!"
                    findings.append(
                        BugFinding(
                            oracle="cpu_vs_mps_grad",
                            category="nan_inf",
                            summary=msg,
                            detail=ir.summary(),
                            op_types=op_types,
                            topology_kind=bundle.topology_kind,
                            seed=ir.seed,
                            signature_key=_make_sig("cpu_vs_mps_grad", "nan_inf", op_types, msg),
                        )
                    )
                elif not _is_close(gx_ref, b_mps.x.grad, atol=1e-2, rtol=2e-2):
                    gdiff = _max_abs_diff(gx_ref, b_mps.x.grad)
                    if gdiff > 5e-2:
                        msg = f"CPU vs MPS x.grad divergence: max_diff={gdiff:.4e}"
                        findings.append(
                            BugFinding(
                                oracle="cpu_vs_mps_grad",
                                category="grad_divergence",
                                summary=msg,
                                detail=ir.summary(),
                                op_types=op_types,
                                topology_kind=bundle.topology_kind,
                                seed=ir.seed,
                                max_diff=gdiff,
                                signature_key=_make_sig("cpu_vs_mps_grad", "grad_divergence", op_types, "bwd_diff"),
                            )
                        )
        except Exception as exc:
            tb = traceback.format_exc()
            findings.append(
                BugFinding(
                    oracle="cpu_vs_mps",
                    category="crash",
                    summary=f"{type(exc).__name__}: {exc}",
                    detail=ir.summary(),
                    op_types=op_types,
                    topology_kind=bundle.topology_kind,
                    seed=ir.seed,
                    traceback_str=tb,
                    signature_key=_make_sig("cpu_vs_mps", "crash", op_types, f"{type(exc).__name__}: {exc}"),
                )
            )

    return findings
