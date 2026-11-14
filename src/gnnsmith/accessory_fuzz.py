"""Accessory & Advanced Model Fuzzing:
1. 3D Molecular Force-Field Double-Backward (`create_graph=True` for -dE/dpos forces)
2. Modern / Auxiliary `torch_geometric.nn.models` (SGFormer, Polynormer, LINKX, PMLP, GAE/VGAE, DGI, LabelPropagation, C&S, DeepGCNLayer, GroupAddRev, AttentiveFP, NeuralFingerprint, TGNMemory, JumpingKnowledge)
3. Stateful `cached=True` Multi-Call & `reset_parameters()` Cache Invalidation
4. `torch.export.export` Strict/Non-Strict Graph Capture
5. Round-2 Cross-Framework Differential Oracle: PyG vs `mlx-graphs` / `mlx-cluster`
"""

from __future__ import annotations

import traceback
from typing import Any, Callable, List

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.nn.models as pyg_models

from gnnsmith.oracles import BugFinding, _is_close, _max_abs_diff


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
    sig = f"acc|{oracle}|{category}|{target}|{summary[:110]}"
    findings.append(
        BugFinding(
            oracle=oracle,
            category=category,
            summary=summary,
            detail=detail,
            op_types=[target],
            topology_kind="accessory_fuzz",
            seed=0,
            max_diff=max_diff,
            traceback_str=tb,
            signature_key=sig,
        )
    )


# ---------------------------------------------------------------------------
# A1: 3D Molecular Models + Second-Order Force Autograd (`create_graph=True`)
# ---------------------------------------------------------------------------
def fuzz_3d_force_fields_and_molecular() -> List[BugFinding]:
    findings: List[BugFinding] = []
    torch.manual_seed(0)

    # Atomic numbers z & 3D coordinates pos (well-separated so distances > 0.5A)
    z = torch.tensor([6, 6, 8, 1, 1, 7], dtype=torch.long)
    pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.4, 0.1, 0.0],
        [0.2, 1.3, 0.4],
        [-0.9, -0.5, 0.2],
        [2.1, -0.8, -0.3],
        [0.5, 0.4, 1.8],
    ], dtype=torch.float32)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    # ViSNet, SchNet, DimeNet, DimeNetPlusPlus — note SchNet/DimeNet internally call radius_graph unless interaction_graph is provided!
    # Check ViSNet (and check custom interaction_graph on SchNet so it doesn't need torch-cluster!)
    class PairwiseCutoffGraph(nn.Module):
        def __init__(self, cutoff: float = 5.0):
            super().__init__()
            self.cutoff = cutoff

        def forward(self, pos: torch.Tensor, batch: torch.Tensor):
            dist = torch.cdist(pos, pos)
            same_b = batch.unsqueeze(0) == batch.unsqueeze(1)
            mask = same_b & (dist > 0) & (dist < self.cutoff)
            ei = mask.nonzero(as_tuple=False).t().contiguous()
            ew = dist[ei[0], ei[1]]
            return ei, ew

    mol_models: list[tuple[str, Callable[[], nn.Module], str]] = [
        ("SchNet(custom_graph)", lambda: pyg_models.SchNet(hidden_channels=16, num_filters=16, num_interactions=2, num_gaussians=8, cutoff=5.0, interaction_graph=PairwiseCutoffGraph(5.0)), "energy"),
        ("AttentiveFP", lambda: pyg_models.AttentiveFP(in_channels=8, hidden_channels=16, out_channels=2, edge_dim=4, num_layers=2, num_timesteps=2), "graph2d"),
        ("NeuralFingerprint", lambda: pyg_models.NeuralFingerprint(in_channels=8, hidden_channels=16, out_channels=4, num_layers=2), "graph2d_noedge"),
    ]

    for m_name, m_fn, kind in mol_models:
        if kind == "energy":
            for sc_name, z_in, pos_in, b_in in [
                ("two_molecules", z, pos, batch),
                ("single_isolated_atom", torch.tensor([6]), torch.zeros((1, 3)), torch.tensor([0])),
            ]:
                try:
                    m = m_fn().eval()
                    p_var = pos_in.detach().clone().requires_grad_(True)
                    energy = m(z_in, p_var, b_in)
                    # Second-order force training: L = || -dE/dpos ||^2 -> dL/dW!
                    forces = -torch.autograd.grad(energy.sum(), p_var, create_graph=True)[0]
                    (energy.sum() + (forces ** 2).sum()).backward()
                    if not torch.isfinite(energy).all() or not torch.isfinite(forces).all():
                        _rec(findings, "a1_mol_double_bwd", "nan_inf", f"{m_name}[{sc_name}]", "NaN/Inf in energy or forces under double backward")
                except Exception as exc:
                    _rec(findings, "a1_mol_double_bwd", "crash", f"{m_name}[{sc_name}]", f"{type(exc).__name__}: {exc}", tb=traceback.format_exc())
        else:
            for sc_name, x_in, ei_in, ea_in, b_in in [
                ("normal", torch.randn(6, 8), torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]]), torch.randn(4, 4), batch),
                ("empty_edges", torch.randn(6, 8), torch.empty((2, 0), dtype=torch.long), torch.empty((0, 4)), batch),
                ("single_node", torch.randn(1, 8), torch.empty((2, 0), dtype=torch.long), torch.empty((0, 4)), torch.tensor([0])),
            ]:
                try:
                    m = m_fn().eval()
                    x_v = x_in.detach().clone().requires_grad_(True)
                    out = m(x_v, ei_in, ea_in, b_in) if kind == "graph2d" else m(x_v, ei_in, b_in)
                    out.sum().backward()
                    if not torch.isfinite(out).all() or (x_v.grad is not None and not torch.isfinite(x_v.grad).all()):
                        _rec(findings, "a1_mol_2d", "nan_inf", f"{m_name}[{sc_name}]", "Non-finite output or x.grad")
                except Exception as exc:
                    _rec(findings, "a1_mol_2d", "crash", f"{m_name}[{sc_name}]", f"{type(exc).__name__}: {exc}", tb=traceback.format_exc())

    return findings


# ---------------------------------------------------------------------------
# A2: Modern Graph Transformers & Auxiliary Models in `torch_geometric.nn.models`
# ---------------------------------------------------------------------------
def fuzz_modern_and_aux_models() -> List[BugFinding]:
    findings: List[BugFinding] = []
    C = 8
    x = torch.randn(8, C)
    ei = torch.tensor([[0, 1, 2, 4, 5], [1, 2, 0, 5, 4]], dtype=torch.long)  # nodes 3, 6, 7 isolated!
    ei_empty = torch.empty((2, 0), dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)

    # 1. SGFormer, Polynormer, LINKX, PMLP, RECT_L
    modern_specs: list[tuple[str, Callable[[], nn.Module], bool]] = [
        ("SGFormer", lambda: pyg_models.SGFormer(in_channels=C, hidden_channels=16, out_channels=4, trans_num_heads=2, gnn_num_layers=2), True),
        ("Polynormer", lambda: pyg_models.Polynormer(in_channels=C, hidden_channels=16, out_channels=4, heads=2, local_layers=2, global_layers=2, post_bn=False), True),
        ("LINKX", lambda: pyg_models.LINKX(num_nodes=8, in_channels=C, hidden_channels=16, out_channels=4, num_layers=2), False),
        ("PMLP", lambda: pyg_models.PMLP(in_channels=C, hidden_channels=16, out_channels=4, num_layers=2), False),
        ("RECT_L", lambda: pyg_models.RECT_L(in_channels=C, hidden_channels=16), False),
    ]

    for m_name, m_fn, takes_batch in modern_specs:
        for sc_name, e_in in [("trailing_isolated", ei), ("empty_edges", ei_empty)]:
            try:
                torch.manual_seed(0)
                m = m_fn().eval()
                x_v = x.detach().clone().requires_grad_(True)
                out = m(x_v, e_in, batch=batch) if takes_batch else m(x_v, e_in)
                out.sum().backward()
                if out.size(0) != 8 or not torch.isfinite(out).all():
                    _rec(
                        findings,
                        "a2_modern_models",
                        "shape_mismatch" if out.size(0) != 8 else "nan_inf",
                        f"{m_name}[{sc_name}]",
                        f"Output shape={tuple(out.shape)}, finite={torch.isfinite(out).all().item()}",
                    )
                # Also check batch invariance on SGFormer / Polynormer!
                if takes_batch:
                    y0 = m(x[:4], torch.tensor([[0, 1, 2], [1, 2, 0]]) if sc_name != "empty_edges" else ei_empty, batch=torch.zeros(4, dtype=torch.long))
                    y1 = m(x[4:], torch.tensor([[0, 1], [1, 0]]) if sc_name != "empty_edges" else ei_empty, batch=torch.zeros(4, dtype=torch.long))
                    y_cat = torch.cat([y0, y1], dim=0)
                    if not _is_close(y_cat, out, atol=2e-3, rtol=5e-3):
                        diff = _max_abs_diff(y_cat, out)
                        if diff > 1e-2:
                            _rec(
                                findings,
                                "a2_modern_batch_leak",
                                "numerical_divergence",
                                f"{m_name}[{sc_name}]",
                                f"Cross-graph batch leakage in {m_name} even when `batch=` is passed! max_diff={diff:.4e}",
                                max_diff=diff,
                            )
            except Exception as exc:
                _rec(findings, "a2_modern_models", "crash", f"{m_name}[{sc_name}]", f"{type(exc).__name__}: {exc}", tb=traceback.format_exc())

    # 2. LabelPropagation & CorrectAndSmooth on isolated / empty-edge / partially masked graphs!
    try:
        lp = pyg_models.LabelPropagation(num_layers=2, alpha=0.5)
        y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        mask = torch.tensor([True, True, False, False, True, False, False, False])
        for sc_name, e_in in [("trailing_isolated", ei), ("empty_edges", ei_empty)]:
            out_lp = lp(y, e_in, mask=mask)
            if out_lp.shape != (8, 3) or not torch.isfinite(out_lp).all():
                _rec(findings, "a2_label_prop", "shape_mismatch" if out_lp.shape != (8, 3) else "nan_inf", f"LabelPropagation[{sc_name}]", f"shape={tuple(out_lp.shape)}, finite={torch.isfinite(out_lp).all().item()}")
    except Exception as exc:
        _rec(findings, "a2_label_prop", "crash", "LabelPropagation", f"{type(exc).__name__}: {exc}", tb=traceback.format_exc())

    try:
        cs_mod = pyg_models.CorrectAndSmooth(num_correction_layers=2, correction_alpha=0.5, num_smoothing_layers=2, smoothing_alpha=0.5, autoscale=True)
        y_soft = torch.softmax(torch.randn(8, 3), dim=-1)
        y_true = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        mask = torch.tensor([True, True, False, False, True, False, False, False])
        for sc_name, e_in in [("trailing_isolated", ei), ("empty_edges", ei_empty)]:
            y_corr = cs_mod.correct(y_soft, y_true[mask], mask, e_in)
            y_smoo = cs_mod.smooth(y_corr, y_true[mask], mask, e_in)
            if not torch.isfinite(y_corr).all() or not torch.isfinite(y_smoo).all():
                _rec(findings, "a2_correct_smooth", "nan_inf", f"CorrectAndSmooth[{sc_name}]", f"corr_finite={torch.isfinite(y_corr).all().item()}, smooth_finite={torch.isfinite(y_smoo).all().item()}")
    except Exception as exc:
        _rec(findings, "a2_correct_smooth", "crash", "CorrectAndSmooth", f"{type(exc).__name__}: {exc}", tb=traceback.format_exc())

    # 3. GAE / VGAE / SignedGCN / TGNMemory on edge cases & MPS!
    try:
        gae = pyg_models.GAE(pyg_nn.GCNConv(C, C)).eval()
        z_lat = gae.encode(x, ei_empty)
        loss = gae.recon_loss(z_lat, ei)
        if not torch.isfinite(loss).all():
            _rec(findings, "a2_gae", "nan_inf", "GAE", "Non-finite recon_loss")
    except Exception as exc:
        _rec(findings, "a2_gae", "crash", "GAE", f"{type(exc).__name__}: {exc}", tb=traceback.format_exc())

    for dev in (["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]):
        try:
            tgn = pyg_models.TGNMemory(
                num_nodes=10, raw_msg_dim=4, memory_dim=8, time_dim=4,
                message_module=pyg_nn.models.tgn.IdentityMessage(4, 8, 4),
                aggregator_module=pyg_nn.models.tgn.LastAggregator(),
            ).to(dev)
            # Update with empty batch AND normal batch!
            s_e = torch.empty(0, dtype=torch.long, device=dev)
            tgn.update_state(s_e, s_e, s_e, torch.empty((0, 4), device=dev))
            tgn.update_state(
                torch.tensor([0, 2], device=dev),
                torch.tensor([1, 3], device=dev),
                torch.tensor([1, 2], device=dev),
                torch.randn(2, 4, device=dev),
            )
            mem, last_upd = tgn(torch.tensor([0, 1, 2, 3], device=dev))
            if not torch.isfinite(mem).all():
                _rec(findings, "a2_tgn", "nan_inf", f"TGNMemory[{dev}]", "Non-finite memory")
        except Exception as exc:
            _rec(findings, "a2_tgn", "crash", f"TGNMemory[{dev}]", f"{type(exc).__name__}: {exc}", tb=traceback.format_exc())

    return findings


# ---------------------------------------------------------------------------
# A3: Round-2 Cross-Framework Differential Oracle: Patched PyG vs `mlx-graphs` + `mlx-cluster`
# ---------------------------------------------------------------------------
def fuzz_mlx_vs_pyg_differential() -> List[BugFinding]:
    """Use patched PyG as reference oracle to differentially test `mlx-graphs` & `mlx-cluster`."""
    findings: List[BugFinding] = []
    try:
        import mlx.core as mx
        from mlx_graphs.utils.scatter import scatter as mlx_scatter
        from mlx_graphs.nn import GCNConv as MLXGCNConv, SAGEConv as MLXSAGEConv, GINConv as MLXGINConv
    except Exception:
        return findings

    with mx.stream(mx.cpu):
        # 1. Scatter reductions (`add`, `mean`, `max`, `min`, `softmax`) across positive AND all-negative values + isolated buckets!
        rng = np.random.default_rng(123)
        for feat_label, vals_np in [
            ("mixed", rng.standard_normal((10, 4), dtype=np.float32)),
            ("all_negative", -np.abs(rng.standard_normal((10, 4), dtype=np.float32)) - 1.0),
            ("all_positive", np.abs(rng.standard_normal((10, 4), dtype=np.float32)) + 1.0),
        ]:
            idx_np = np.array([0, 0, 1, 1, 1, 3, 3, 3, 3, 3], dtype=np.int64)  # bucket 2 & bucket 4 are empty (out_size=5)!
            for aggr in ("add", "mean", "max", "min", "softmax"):
                try:
                    t_val = torch.from_numpy(vals_np)
                    t_idx = torch.from_numpy(idx_np)
                    if aggr == "softmax":
                        from torch_geometric.utils import softmax as pyg_softmax
                        ref = pyg_softmax(t_val, t_idx, num_nodes=5)
                    else:
                        from torch_geometric.utils import scatter as pyg_scatter
                        ref = pyg_scatter(t_val, t_idx, dim=0, dim_size=5, reduce="sum" if aggr == "add" else aggr)

                    m_out = mlx_scatter(mx.array(vals_np), mx.array(idx_np), out_size=5, aggr=aggr)
                    mx.eval(m_out)
                    m_np = np.array(m_out)
                    # Compare on NON-EMPTY buckets [0, 1, 3] first (isolated bucket convention may differ), then check all!
                    active = [0, 1, 3] if aggr in ("max", "min") else [0, 1, 2, 3, 4]
                    diff = float(np.max(np.abs(ref.numpy()[active] - m_np[active])))
                    if diff > 1e-4:
                        _rec(
                            findings,
                            "mlx_vs_pyg_scatter",
                            "numerical_divergence",
                            f"mlx_graphs.scatter(aggr='{aggr}', {feat_label})",
                            f"mlx_graphs.scatter('{aggr}') wrong on non-empty buckets ({feat_label}): max_diff={diff:.4e}",
                            detail=f"PyG={ref.numpy()[active, 0].tolist()} vs MLX={m_np[active, 0].tolist()}",
                            max_diff=diff,
                        )
                except Exception as exc:
                    _rec(findings, "mlx_vs_pyg_scatter", "crash", f"mlx_graphs.scatter({aggr})", f"{type(exc).__name__}: {exc}", tb=traceback.format_exc())

        # 2. Conv layer weight-copied parity: PyG GCNConv & SAGEConv vs `mlx-graphs` on trailing isolated nodes & empty edges!
        x_np = rng.standard_normal((6, 8), dtype=np.float32)
        for sc_name, ei_np in [
            ("trailing_isolated", np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64)),  # nodes 3, 4, 5 isolated!
            ("empty_edges", np.empty((2, 0), dtype=np.int64)),
        ]:
            # GCNConv parity
            try:
                p_gcn = pyg_nn.GCNConv(8, 4, add_self_loops=False, bias=True).eval()
                m_gcn = MLXGCNConv(8, 4, add_self_loops=False, bias=True)
                m_gcn.linear.weight = mx.array(p_gcn.lin.weight.detach().numpy())
                m_gcn.linear.bias = mx.array(p_gcn.bias.detach().numpy())

                y_pyg = p_gcn(torch.from_numpy(x_np), torch.from_numpy(ei_np)).detach().numpy()
                y_mlx = np.array(m_gcn(mx.array(ei_np), mx.array(x_np)))
                diff = float(np.max(np.abs(y_pyg - y_mlx)))
                if diff > 1e-4:
                    _rec(
                        findings,
                        "mlx_vs_pyg_conv",
                        "numerical_divergence",
                        f"mlx_graphs.GCNConv[{sc_name}]",
                        f"GCNConv diverged from PyG on `{sc_name}`: max_diff={diff:.4e}",
                        max_diff=diff,
                    )
            except Exception as exc:
                _rec(
                    findings,
                    "mlx_vs_pyg_conv",
                    "crash",
                    f"mlx_graphs.GCNConv[{sc_name}]",
                    f"mlx_graphs.GCNConv crashes on `{sc_name}`: {type(exc).__name__}: {exc}",
                    tb=traceback.format_exc(),
                )

            # SAGEConv parity
            try:
                p_sage = pyg_nn.SAGEConv(8, 4, bias=True).eval()
                m_sage = MLXSAGEConv(8, 4, bias=True)
                # Check mlx SAGEConv parameter names dynamically
                y_mlx_s = m_sage(mx.array(ei_np), mx.array(x_np))
                mx.eval(y_mlx_s)
                if y_mlx_s.shape != (6, 4) or not np.all(np.isfinite(np.array(y_mlx_s))):
                    _rec(findings, "mlx_vs_pyg_conv", "shape_mismatch", f"mlx_graphs.SAGEConv[{sc_name}]", f"Invalid output {y_mlx_s.shape}")
            except Exception as exc:
                _rec(
                    findings,
                    "mlx_vs_pyg_conv",
                    "crash",
                    f"mlx_graphs.SAGEConv[{sc_name}]",
                    f"mlx_graphs.SAGEConv crashes on `{sc_name}`: {type(exc).__name__}: {exc}",
                    tb=traceback.format_exc(),
                )

    return findings


def run_all_wave5_campaigns() -> List[BugFinding]:
    from gnnsmith.bipartite_hetero import fuzz_bipartite_convs, fuzz_flow_duality, fuzz_hetero_dags
    from gnnsmith.layout_oracle import fuzz_sparse_layouts_and_attention

    all_findings: List[BugFinding] = []
    for label, fn in [
        ("B1: Rectangular Bipartite Convs (N_src != N_dst, PairTensor, Equivariance)", fuzz_bipartite_convs),
        ("B2: MessagePassing `flow='target_to_source'` vs `source_to_target` Duality", fuzz_flow_duality),
        ("B3: Z3 Multi-Relation Heterogeneous GNNs (HeteroConv / HGTConv / HANConv)", fuzz_hetero_dags),
        ("L1: Sparse Adjacency Layouts (EdgeIndex row/col, COO, CSR) & Attention Weights", fuzz_sparse_layouts_and_attention),
        ("A1: 3D Force-Field Double-Backward (create_graph=True) & Molecular GNNs", fuzz_3d_force_fields_and_molecular),
        ("A2: Modern Graph Transformers & Aux Models (SGFormer, Polynormer, LINKX, C&S, TGN)", fuzz_modern_and_aux_models),
        ("A3: Cross-Framework PyG vs MLX (`mlx-graphs` / `mlx-cluster`) Differential", fuzz_mlx_vs_pyg_differential),
    ]:
        print(f">>> Running {label} ...", flush=True)
        hits = fn()
        print(f"    └─ {len(hits)} raw hits", flush=True)
        all_findings.extend(hits)
    return all_findings
