"""Epoch 3 Deep PyG Fuzzing Engine: torch.compile, AMP/bf16/fp16, gradcheck, Transforms, Models, Explain, & Serialization."""

from __future__ import annotations

import copy
import io
import traceback
from typing import Any, Callable, List

import torch
import torch.nn as nn
from torch_geometric import EdgeIndex, Index
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.explain import AttentionExplainer, DummyExplainer, Explainer, GNNExplainer, PGExplainer
import torch_geometric.nn as pyg_nn
import torch_geometric.nn.aggr as pyg_aggr
import torch_geometric.nn.models as pyg_models
import torch_geometric.transforms as T
import torch_geometric.utils as pyg_utils

from gnnsmith.graph_gen import synthesize_gnn_dag
from gnnsmith.materialize import materialize_ir
from gnnsmith.ops import ALL_OPS
from gnnsmith.ops.pyg_norm_pool import GlobalPoolReadoutOp
from gnnsmith.oracles import BugFinding, _is_close, _max_abs_diff
from gnnsmith.topology import generate_graph_bundle


def _rec(
    findings: List[BugFinding],
    oracle: str,
    category: str,
    target: str,
    summary: str,
    detail: str = "",
    tb: str = "",
    max_diff: float | None = None,
) -> None:
    sig = f"deep|{oracle}|{category}|{target}|{summary[:110]}"
    findings.append(
        BugFinding(
            oracle=oracle,
            category=category,
            summary=summary,
            detail=detail,
            op_types=[target],
            topology_kind="deep_fuzz",
            seed=0,
            max_diff=max_diff,
            traceback_str=tb,
            signature_key=sig,
        )
    )


# ---------------------------------------------------------------------------
# D1: torch.compile (dynamic=True, forward + backward, Tensor & EdgeIndex)
# ---------------------------------------------------------------------------
def battery_d1_torch_compile() -> List[BugFinding]:
    findings: List[BugFinding] = []
    ops = [op for op in ALL_OPS if op is not GlobalPoolReadoutOp and op.op_type != "SuperGATConv"]

    for idx, op_cls in enumerate(ops):
        for topo in ("erdos_renyi", "trailing_isolated", "empty_edges"):
            if op_cls.capabilities.mutates_topology and topo == "empty_edges":
                continue
            torch._dynamo.reset()
            seed = 5000 + idx * 10 + (0 if topo == "erdos_renyi" else (1 if topo == "trailing_isolated" else 2))
            try:
                ir = synthesize_gnn_dag(
                    seed=seed, min_ops=1, max_ops=1, allowed_ops=[op_cls],
                    topology_kind=topo, include_readout_prob=0.0,
                )
                s0 = ir.scopes[0]
                b = generate_graph_bundle(
                    num_nodes=int(s0.num_nodes), num_edges=int(s0.num_edges),
                    in_channels=int(ir.values["x0"].channels), num_graphs=int(s0.num_graphs),
                    edge_dim=s0.edge_dim, num_relations=s0.num_relations,
                    topology_kind=topo, seed=seed,
                )
                m_eager = materialize_ir(ir, seed=seed).eval()
                b_ref = b.clone_requires_grad()
                out_ref = m_eager(b_ref)
                out_ref.sum().backward()
                gx_ref = b_ref.x.grad.detach().clone()

                m_comp = torch.compile(copy.deepcopy(m_eager), dynamic=True)
                b_cmp = b.clone_requires_grad()
                out_cmp = m_comp(b_cmp)
                out_cmp.sum().backward()
                gx_cmp = b_cmp.x.grad.detach().clone()

                if not torch.isfinite(out_cmp).all() or not torch.isfinite(gx_cmp).all():
                    _rec(findings, "d1_compile", "nan_inf", f"{op_cls.op_type}[{topo}]", "torch.compile produced NaN/Inf whereas eager was finite")
                elif not _is_close(out_ref, out_cmp, atol=3e-3, rtol=1e-2):
                    diff = _max_abs_diff(out_ref, out_cmp)
                    _rec(findings, "d1_compile", "numerical_divergence", f"{op_cls.op_type}[{topo}]", f"torch.compile fwd diverged: max_diff={diff:.4e}", max_diff=diff)
                elif not _is_close(gx_ref, gx_cmp, atol=5e-3, rtol=1e-2):
                    gdiff = _max_abs_diff(gx_ref, gx_cmp)
                    if gdiff > 2e-2:
                        _rec(findings, "d1_compile_grad", "grad_divergence", f"{op_cls.op_type}[{topo}]", f"torch.compile x.grad diverged: max_diff={gdiff:.4e}", max_diff=gdiff)
            except Exception as exc:
                _rec(
                    findings,
                    "d1_compile",
                    "crash",
                    f"{op_cls.op_type}[{topo}]",
                    f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}",
                    detail=f"op={op_cls.op_type}, topo={topo}",
                    tb=traceback.format_exc(),
                )
    return findings


# ---------------------------------------------------------------------------
# D2: Mixed Precision (bfloat16 native, float16 native, & torch.autocast)
# ---------------------------------------------------------------------------
def battery_d2_mixed_precision() -> List[BugFinding]:
    findings: List[BugFinding] = []
    ops = [op for op in ALL_OPS if op is not GlobalPoolReadoutOp and op.op_type != "SuperGATConv"]

    for idx, op_cls in enumerate(ops):
        seed = 7000 + idx
        ir = synthesize_gnn_dag(
            seed=seed, min_ops=1, max_ops=1, allowed_ops=[op_cls],
            topology_kind="trailing_isolated", include_readout_prob=0.0,
        )
        s0 = ir.scopes[0]
        b_fp32 = generate_graph_bundle(
            num_nodes=12, num_edges=24, in_channels=int(ir.values["x0"].channels),
            num_graphs=2, edge_dim=s0.edge_dim, num_relations=s0.num_relations,
            topology_kind="trailing_isolated", seed=seed,
        )

        # (a) Native bfloat16 on CPU
        bf16_crashed = False
        try:
            m_bf16 = materialize_ir(ir, seed=seed).to(torch.bfloat16).eval()
            b_bf16 = b_fp32.to("cpu", dtype=torch.bfloat16).clone_requires_grad()
            y_bf16 = m_bf16(b_bf16)
            y_bf16.sum().backward()
            if not torch.isfinite(y_bf16).all():
                _rec(findings, "d2_bfloat16", "nan_inf", op_cls.op_type, "Native bfloat16 produced NaN/Inf")
        except Exception as exc:
            bf16_crashed = True
            _rec(
                findings,
                "d2_bfloat16",
                "crash",
                op_cls.op_type,
                f"bfloat16 forward/backward crash: {type(exc).__name__}: {exc}",
                tb=traceback.format_exc(),
            )

        # (b) CPU `torch.autocast("cpu", dtype=torch.bfloat16)` with fp32 model & inputs!
        try:
            m_amp = materialize_ir(ir, seed=seed).eval()
            b_amp = b_fp32.clone_requires_grad()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                y_amp = m_amp(b_amp)
                loss = y_amp.float().sum()
            loss.backward()
            if not torch.isfinite(y_amp).all():
                _rec(findings, "d2_autocast_cpu", "nan_inf", op_cls.op_type, "CPU autocast(bf16) produced NaN/Inf")
        except Exception as exc:
            _rec(
                findings,
                "d2_autocast_cpu",
                "crash",
                op_cls.op_type,
                f"torch.autocast('cpu', bfloat16) crash: {type(exc).__name__}: {exc}",
                tb=traceback.format_exc(),
            )

        # (c) Native float16 on MPS (skip if CPU low-precision already flagged mixed-dtype linear bug to avoid MPSNDArray SIGABRT)
        if torch.backends.mps.is_available() and not bf16_crashed:
            try:
                m_fp16 = materialize_ir(ir, seed=seed).to(device="mps", dtype=torch.float16).eval()
                b_fp16 = b_fp32.to(device="mps", dtype=torch.float16).clone_requires_grad()
                y_fp16 = m_fp16(b_fp16)
                y_fp16.sum().backward()
                if not torch.isfinite(y_fp16).all():
                    _rec(findings, "d2_mps_fp16", "nan_inf", op_cls.op_type, "MPS float16 produced NaN/Inf")
            except Exception as exc:
                _rec(
                    findings,
                    "d2_mps_fp16",
                    "crash",
                    op_cls.op_type,
                    f"MPS float16 crash: {type(exc).__name__}: {exc}",
                    tb=traceback.format_exc(),
                )

    return findings


# ---------------------------------------------------------------------------
# D3: torch.autograd.gradcheck (float64 analytical vs numerical Jacobians!)
# ---------------------------------------------------------------------------
def battery_d3_gradcheck() -> List[BugFinding]:
    findings: List[BugFinding] = []
    torch.manual_seed(0)

    # 1. EdgeIndex.matmul across reduce=('sum','mean') and transpose=(False, True), weighted & unweighted!
    for red in ("sum", "mean"):
        for trans in (False, True):
            for weighted in (False, True):
                name = f"EdgeIndex.matmul(reduce={red}, T={trans}, weighted={weighted})"
                try:
                    ei = EdgeIndex(
                        torch.tensor([[0, 0, 2, 3], [1, 3, 0, 4]], dtype=torch.long),
                        sparse_size=(4, 5),
                    ).sort_by("col" if trans else "row").values
                    n_in = 4 if trans else 5
                    x = torch.randn(n_in, 3, dtype=torch.float64, requires_grad=True)
                    if weighted:
                        val = (torch.rand(4, dtype=torch.float64) + 0.2).requires_grad_(True)
                        ok = torch.autograd.gradcheck(
                            lambda x_in, v_in: ei.matmul(x_in, input_value=v_in, reduce=red, transpose=trans),
                            (x, val),
                            eps=1e-6,
                            atol=1e-4,
                        )
                    else:
                        ok = torch.autograd.gradcheck(
                            lambda x_in: ei.matmul(x_in, reduce=red, transpose=trans),
                            (x,),
                            eps=1e-6,
                            atol=1e-4,
                        )
                    if not ok:
                        _rec(findings, "d3_gradcheck", "grad_divergence", name, "gradcheck returned False")
                except Exception as exc:
                    _rec(findings, "d3_gradcheck", "crash", name, f"{type(exc).__name__}: {exc}", tb=traceback.format_exc())

    # 2. utils.softmax (index & ptr) in float64
    x_sm = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
    idx_sm = torch.tensor([0, 0, 2, 2, 2, 4], dtype=torch.long)
    ptr_sm = torch.tensor([0, 2, 2, 5, 5, 6], dtype=torch.long)
    for mode, fn in [
        ("softmax(index)", lambda t: pyg_utils.softmax(t, index=idx_sm, num_nodes=5)),
        ("softmax(ptr)", lambda t: pyg_utils.softmax(t, ptr=ptr_sm, num_nodes=5)),
        ("softmax(ptr, dim=1)", lambda t: pyg_utils.softmax(t.t().contiguous(), ptr=ptr_sm, dim=1)),
    ]:
        try:
            torch.autograd.gradcheck(fn, (x_sm,), eps=1e-6, atol=1e-4)
        except Exception as exc:
            _rec(findings, "d3_gradcheck", "grad_divergence", mode, f"{type(exc).__name__}: {str(exc)[:180]}", tb=traceback.format_exc())

    # 3. Differentiable Aggregations in float64
    x_ag = (torch.rand(6, 3, dtype=torch.float64) + 0.5).requires_grad_(True)
    for ag_name, ag_mod in [
        ("VarAggregation", pyg_aggr.VarAggregation().double()),
        ("StdAggregation", pyg_aggr.StdAggregation().double()),
        ("PowerMeanAggregation", pyg_aggr.PowerMeanAggregation(learn=True).double()),
        ("SoftmaxAggregation", pyg_aggr.SoftmaxAggregation(learn=True).double()),
    ]:
        try:
            torch.autograd.gradcheck(lambda t: ag_mod(t, index=idx_sm, dim_size=5), (x_ag,), eps=1e-6, atol=1e-4)
        except Exception as exc:
            _rec(findings, "d3_gradcheck", "grad_divergence", ag_name, f"{type(exc).__name__}: {str(exc)[:180]}", tb=traceback.format_exc())

    return findings


# ---------------------------------------------------------------------------
# D4: Graph Transforms (`torch_geometric.transforms`) across edge cases!
# ---------------------------------------------------------------------------
def battery_d4_transforms() -> List[BugFinding]:
    findings: List[BugFinding] = []

    def make_cases() -> list[tuple[str, Data]]:
        return [
            (
                "normal",
                Data(
                    x=torch.randn(6, 4),
                    pos=torch.randn(6, 3),
                    norm=torch.nn.functional.normalize(torch.randn(6, 3), dim=-1),
                    edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 0, 4]], dtype=torch.long),
                    edge_attr=torch.randn(4, 2),
                    edge_weight=torch.rand(4) + 0.1,
                    y=torch.tensor([0, 1, 0, 1, 0, 1]),
                ),
            ),
            (
                "empty_edges",
                Data(
                    x=torch.randn(4, 4),
                    pos=torch.randn(4, 3),
                    norm=torch.nn.functional.normalize(torch.randn(4, 3), dim=-1),
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                    edge_attr=torch.empty((0, 2)),
                    edge_weight=torch.empty((0,)),
                    y=torch.tensor([0, 1, 0, 1]),
                ),
            ),
            (
                "single_node",
                Data(
                    x=torch.randn(1, 4),
                    pos=torch.randn(1, 3),
                    norm=torch.tensor([[1.0, 0.0, 0.0]]),
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                    edge_attr=torch.empty((0, 2)),
                    edge_weight=torch.empty((0,)),
                    y=torch.tensor([0]),
                ),
            ),
            (
                "edge_index_subclass",
                Data(
                    x=torch.randn(5, 4),
                    pos=torch.randn(5, 3),
                    norm=torch.nn.functional.normalize(torch.randn(5, 3), dim=-1),
                    edge_index=EdgeIndex([[0, 1, 2], [1, 2, 0]], sparse_size=(5, 5), sort_order="row"),
                    edge_attr=torch.randn(3, 2),
                    edge_weight=torch.rand(3) + 0.1,
                    y=torch.tensor([0, 1, 0, 1, 0]),
                ),
            ),
        ]

    transforms_list: list[tuple[str, Callable[[], Any]]] = [
        ("NormalizeFeatures", lambda: T.NormalizeFeatures()),
        ("NormalizeScale", lambda: T.NormalizeScale()),
        ("Center", lambda: T.Center()),
        ("Cartesian", lambda: T.Cartesian()),
        ("Distance", lambda: T.Distance()),
        ("Polar", lambda: T.Polar()),
        ("Spherical", lambda: T.Spherical()),
        ("PointPairFeatures", lambda: T.PointPairFeatures()),
        ("OneHotDegree", lambda: T.OneHotDegree(max_degree=5)),
        ("TargetIndegree", lambda: T.TargetIndegree()),
        ("LocalDegreeProfile", lambda: T.LocalDegreeProfile()),
        ("ToUndirected", lambda: T.ToUndirected()),
        ("AddSelfLoops", lambda: T.AddSelfLoops()),
        ("RemoveSelfLoops", lambda: T.RemoveSelfLoops()),
        ("RemoveIsolatedNodes", lambda: T.RemoveIsolatedNodes()),
        ("RemoveDuplicatedEdges", lambda: T.RemoveDuplicatedEdges()),
        ("GCNNorm", lambda: T.GCNNorm()),
        ("LineGraph", lambda: T.LineGraph()),
        ("TwoHop", lambda: T.TwoHop()),
        ("VirtualNode", lambda: T.VirtualNode()),
        ("AddRandomWalkPE", lambda: T.AddRandomWalkPE(walk_length=3)),
        ("AddLaplacianEigenvectorPE", lambda: T.AddLaplacianEigenvectorPE(k=1, is_undirected=True)),
        ("SIGN", lambda: T.SIGN(K=2)),
        ("RootedEgoNets", lambda: T.RootedEgoNets(num_hops=1)),
        ("GDC", lambda: T.GDC()),
        ("RandomNodeSplit", lambda: T.RandomNodeSplit(num_val=1, num_test=1)),
        ("RandomLinkSplit", lambda: T.RandomLinkSplit(num_val=0.2, num_test=0.2, is_undirected=False)),
    ]

    for t_name, t_fn in transforms_list:
        for c_name, data in make_cases():
            if t_name in ("Polar",) and c_name != "empty_edges":
                # Polar expects 2D pos!
                data = copy.deepcopy(data)
                data.pos = data.pos[:, :2]
            elif t_name == "Polar":
                data = copy.deepcopy(data)
                data.pos = data.pos[:, :2]
            else:
                data = copy.deepcopy(data)

            if t_name == "AddLaplacianEigenvectorPE" and data.num_nodes <= 2:
                continue
            if t_name == "RandomNodeSplit" and data.num_nodes < 3:
                continue
            if t_name == "RandomLinkSplit" and data.num_edges < 2:
                continue
            if t_name == "GDC":
                data.edge_attr = None

            try:
                tr = t_fn()
                out = tr(data)
                if isinstance(out, Data):
                    for k, v in out.to_dict().items():
                        if isinstance(v, torch.Tensor) and v.is_floating_point() and not torch.isfinite(v).all():
                            _rec(
                                findings,
                                "d4_transforms",
                                "nan_inf",
                                f"{t_name}[{c_name}]",
                                f"Transform {t_name} on `{c_name}` produced non-finite tensor in `{k}`",
                            )
            except Exception as exc:
                _rec(
                    findings,
                    "d4_transforms",
                    "crash",
                    f"{t_name}[{c_name}]",
                    f"{type(exc).__name__}: {exc}",
                    tb=traceback.format_exc(),
                )

    return findings


# ---------------------------------------------------------------------------
# D5: High-Level `torch_geometric.nn.models` (GCN, GraphSAGE, GIN, GAT, PNA, EdgeCNN, GraphUNet, LightGCN, SignedGCN, C&S)
# ---------------------------------------------------------------------------
def battery_d5_models() -> List[BugFinding]:
    findings: List[BugFinding] = []
    deg = torch.tensor([1, 4, 4, 2])

    model_builders: list[tuple[str, Callable[[], nn.Module], str]] = [
        ("GCN(jk=cat)", lambda: pyg_models.GCN(8, 16, num_layers=2, out_channels=4, jk="cat"), "std"),
        ("GCN(jk=lstm)", lambda: pyg_models.GCN(8, 16, num_layers=2, out_channels=4, jk="lstm"), "std"),
        ("GraphSAGE(jk=max)", lambda: pyg_models.GraphSAGE(8, 16, num_layers=2, out_channels=4, jk="max"), "std"),
        ("GIN(jk=cat)", lambda: pyg_models.GIN(8, 16, num_layers=2, out_channels=4, jk="cat"), "std"),
        ("GAT(v2=True,jk=cat)", lambda: pyg_models.GAT(8, 16, num_layers=2, out_channels=4, v2=True, heads=2, jk="cat"), "std"),
        ("PNA", lambda: pyg_models.PNA(8, 16, num_layers=2, out_channels=4, aggregators=["mean", "max"], scalers=["identity", "amplification"], deg=deg), "std"),
        ("EdgeCNN", lambda: pyg_models.EdgeCNN(8, 16, num_layers=2, out_channels=4), "std"),
        ("GraphUNet", lambda: pyg_models.GraphUNet(8, 16, 4, depth=2, pool_ratios=0.5), "unet"),
        ("LightGCN", lambda: pyg_models.LightGCN(num_nodes=10, embedding_dim=8, num_layers=2), "lightgcn"),
    ]

    for m_name, m_fn, kind in model_builders:
        for sc_name, x, ei, batch in [
            ("trailing_isolated", torch.randn(10, 8), torch.tensor([[0, 1, 2, 5], [1, 2, 0, 6]]), torch.tensor([0]*5 + [1]*5)),
            ("empty_edges", torch.randn(10, 8), torch.empty((2, 0), dtype=torch.long), torch.tensor([0]*5 + [1]*5)),
            ("single_node", torch.randn(1, 8), torch.empty((2, 0), dtype=torch.long), torch.tensor([0])),
        ]:
            if kind == "lightgcn" and sc_name == "single_node":
                continue
            try:
                torch.manual_seed(0)
                m = m_fn().eval()
                if kind == "unet":
                    out = m(x, ei, batch=batch)
                elif kind == "lightgcn":
                    out = m.get_embedding(ei)
                else:
                    out = m(x, ei)
                if out.size(0) != x.size(0) or not torch.isfinite(out).all():
                    _rec(
                        findings,
                        "d5_models",
                        "shape_mismatch" if out.size(0) != x.size(0) else "nan_inf",
                        f"{m_name}[{sc_name}]",
                        f"Output shape={tuple(out.shape)} (expected {x.size(0)} rows), finite={torch.isfinite(out).all().item()}",
                    )
            except Exception as exc:
                _rec(
                    findings,
                    "d5_models",
                    "crash",
                    f"{m_name}[{sc_name}]",
                    f"{type(exc).__name__}: {exc}",
                    tb=traceback.format_exc(),
                )

    return findings


# ---------------------------------------------------------------------------
# D6: Explainability (`Explainer` + `GNNExplainer` / `PGExplainer` / `AttentionExplainer` on CPU & MPS & Hetero)
# ---------------------------------------------------------------------------
def battery_d6_explain() -> List[BugFinding]:
    findings: List[BugFinding] = []

    class ExplainGAT(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = pyg_nn.GATConv(8, 8, heads=2, concat=False)
            self.c2 = pyg_nn.GATConv(8, 3, heads=1, concat=False)

        def forward(self, x, edge_index):
            return self.c2(self.c1(x, edge_index).relu(), edge_index)

    x = torch.randn(6, 8)
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 0, 4]], dtype=torch.long)
    ei_empty = torch.empty((2, 0), dtype=torch.long)

    # 1. AttentionExplainer on E=0 & MPS!
    for sc_name, e_in, dev in [
        ("empty_edges_cpu", ei_empty, "cpu"),
        ("normal_mps", ei, "mps" if torch.backends.mps.is_available() else "cpu"),
    ]:
        try:
            m = ExplainGAT().to(dev).eval()
            exp = Explainer(
                model=m,
                algorithm=AttentionExplainer(),
                explanation_type="model",
                edge_mask_type="object",
                model_config=dict(mode="multiclass_classification", task_level="node", return_type="raw"),
            )
            res = exp(x.to(dev), e_in.to(dev), index=0)
            if res.edge_mask is not None and not torch.isfinite(res.edge_mask).all():
                _rec(findings, "d6_explain", "nan_inf", f"AttentionExplainer[{sc_name}]", "Non-finite edge_mask")
        except Exception as exc:
            _rec(findings, "d6_explain", "crash", f"AttentionExplainer[{sc_name}]", f"{type(exc).__name__}: {exc}", tb=traceback.format_exc())

    # 2. GNNExplainer on E=0, EdgeIndex subclass, & MPS!
    for sc_name, e_in, dev in [
        ("empty_edges_cpu", ei_empty, "cpu"),
        ("edge_index_subclass", EdgeIndex(ei.clone(), sparse_size=(6, 6), sort_order="row"), "cpu"),
        ("normal_mps", ei, "mps" if torch.backends.mps.is_available() else "cpu"),
    ]:
        try:
            m = ExplainGAT().to(dev).eval()
            exp = Explainer(
                model=m,
                algorithm=GNNExplainer(epochs=2),
                explanation_type="model",
                node_mask_type="attributes",
                edge_mask_type="object",
                model_config=dict(mode="multiclass_classification", task_level="node", return_type="raw"),
            )
            res = exp(x.to(dev), e_in.to(dev), index=0)
            if res.edge_mask is not None and not torch.isfinite(res.edge_mask).all():
                _rec(findings, "d6_explain", "nan_inf", f"GNNExplainer[{sc_name}]", "Non-finite edge_mask")
        except Exception as exc:
            _rec(findings, "d6_explain", "crash", f"GNNExplainer[{sc_name}]", f"{type(exc).__name__}: {exc}", tb=traceback.format_exc())

    return findings


# ---------------------------------------------------------------------------
# D7: `Data` / `Batch` / `HeteroData` / `EdgeIndex` Collation, Slicing & `torch.load(weights_only=True)`!
# ---------------------------------------------------------------------------
def battery_d7_serialization_and_batching() -> List[BugFinding]:
    findings: List[BugFinding] = []

    ei = EdgeIndex([[0, 1], [1, 0]], sparse_size=(3, 3), sort_order="row", is_undirected=True)
    ei.fill_cache_()
    idx = Index([0, 1, 1, 2], dim_size=4, is_sorted=True)
    idx.fill_cache_()

    d1 = Data(x=torch.randn(3, 4), edge_index=ei)
    d2 = Data(x=torch.randn(2, 4), edge_index=EdgeIndex(torch.empty((2, 0), dtype=torch.long), sparse_size=(2, 2), sort_order="row"))
    batch = Batch.from_data_list([d1, d2])

    hd = HeteroData()
    hd["paper"].x = torch.randn(3, 4)
    hd["author"].x = torch.randn(2, 4)
    hd["author", "writes", "paper"].edge_index = torch.tensor([[0, 1], [1, 2]])

    # 1. PyTorch 2.6+ default `torch.load(..., weights_only=True)` serialization check!
    for obj_name, obj in [
        ("Index(cached)", idx),
        ("EdgeIndex(cached)", ei),
        ("EdgeIndex(cat)", torch.cat([ei, ei], dim=1)),
        ("Data(with_EdgeIndex)", d1),
        ("Batch(with_EdgeIndex)", batch),
        ("HeteroData", hd),
    ]:
        try:
            buf = io.BytesIO()
            torch.save(obj, buf)
            buf.seek(0)
            _ = torch.load(buf, weights_only=True)
        except Exception as exc:
            _rec(
                findings,
                "d7_weights_only_load",
                "crash",
                obj_name,
                f"torch.load(..., weights_only=True) fails on `{obj_name}`: {type(exc).__name__}: {str(exc).splitlines()[0][:180]}",
                tb=traceback.format_exc(),
            )

    # 2. `Batch.from_data_list` -> `to_data_list` round-trip when one graph has empty `EdgeIndex`!
    try:
        restored = batch.to_data_list()
        if restored[0].x.shape != (3, 4) or restored[1].x.shape != (2, 4) or restored[1].edge_index.shape != (2, 0):
            _rec(findings, "d7_batch_roundtrip", "shape_mismatch", "Batch.to_data_list", f"Mismatch on round-trip: {restored}")
    except Exception as exc:
        _rec(findings, "d7_batch_roundtrip", "crash", "Batch.to_data_list", f"{type(exc).__name__}: {exc}", tb=traceback.format_exc())

    return findings


def run_all_deep_batteries() -> List[BugFinding]:
    all_findings: List[BugFinding] = []
    for label, fn in [
        ("D1: torch.compile (dynamic=True, fwd+bwd)", battery_d1_torch_compile),
        ("D2: Mixed Precision (bf16, autocast, MPS fp16)", battery_d2_mixed_precision),
        ("D3: torch.autograd.gradcheck (float64)", battery_d3_gradcheck),
        ("D4: Graph Transforms (27 transforms x 4 cases)", battery_d4_transforms),
        ("D5: High-Level nn.models (GCN/SAGE/GIN/GAT/PNA/GraphUNet/LightGCN)", battery_d5_models),
        ("D6: Explainability (AttentionExplainer / GNNExplainer)", battery_d6_explain),
        ("D7: Serialization (weights_only=True) & Batch Round-Trip", battery_d7_serialization_and_batching),
    ]:
        print(f">>> Running {label} ...", flush=True)
        hits = fn()
        print(f"    └─ {len(hits)} raw hits", flush=True)
        all_findings.extend(hits)
    return all_findings
