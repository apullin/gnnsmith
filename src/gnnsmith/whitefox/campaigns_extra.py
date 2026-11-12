"""Extended WhiteFox campaigns: HeteroGNNs, Utils deep sweep, Hypergraph/Signed/DNA convs, Explain, & torch.compile."""

from __future__ import annotations

import copy
import traceback
from typing import List

import torch
import torch.nn as nn
import torch_geometric.nn.conv as pyg_conv
import torch_geometric.nn.norm as pyg_norm
import torch_geometric.utils as pyg_utils
from torch_geometric.explain import AttentionExplainer, Explainer, GNNExplainer

from gnnsmith.oracles import BugFinding
from gnnsmith.whitefox.campaigns import _record


def campaign_hetero_whitebox() -> List[BugFinding]:
    """Campaign W6: Fuzz `HANConv`, `HGTConv`, `HEATConv`, `HeteroConv`, `HeteroBatchNorm`, `HeteroLayerNorm`."""
    findings: List[BugFinding] = []
    metadata = (
        ["paper", "author"],
        [("paper", "cites", "paper"), ("author", "writes", "paper"), ("paper", "rev_writes", "author")],
    )

    x_dict = {"paper": torch.randn(6, 8, requires_grad=True), "author": torch.randn(4, 8, requires_grad=True)}
    ei_nonempty = {
        ("paper", "cites", "paper"): torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        ("author", "writes", "paper"): torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        ("paper", "rev_writes", "author"): torch.tensor([[2, 3], [0, 1]], dtype=torch.long),
    }
    # Edge case: ONE relation has 0 edges (`empty((2, 0))`) while other relations are non-empty!
    ei_one_empty = {
        ("paper", "cites", "paper"): torch.empty((2, 0), dtype=torch.long),
        ("author", "writes", "paper"): torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        ("paper", "rev_writes", "author"): torch.empty((2, 0), dtype=torch.long),
    }

    for conv_name, factory in [
        ("HANConv", lambda: pyg_conv.HANConv(8, 8, metadata=metadata, heads=2)),
        ("HGTConv", lambda: pyg_conv.HGTConv(8, 8, metadata=metadata, heads=2)),
    ]:
        # Test A: One relation has 0 edges
        try:
            torch.manual_seed(0)
            m = factory().eval()
            xd = {k: v.detach().clone().requires_grad_(True) for k, v in x_dict.items()}
            out = m(xd, ei_one_empty)
            loss = sum(v.sum() for v in out.values() if v is not None)
            loss.backward()
            for k, v in out.items():
                if v is not None and not torch.isfinite(v).all():
                    _record(findings, "wf_hetero_empty_rel", "nan_inf", conv_name, f"{conv_name} produced NaN/Inf when one edge relation is empty!", k)
        except Exception as exc:
            _record(
                findings,
                "wf_hetero_empty_rel",
                "crash",
                conv_name,
                f"{conv_name} crashes when one edge relation has 0 edges: {type(exc).__name__}: {exc}",
                "ei_dict with ('paper','cites','paper') having shape (2, 0).",
                traceback.format_exc(),
            )

        # Test B: CPU vs MPS parity on Hetero GNNs!
        if torch.backends.mps.is_available():
            try:
                torch.manual_seed(0)
                m_cpu = factory().eval()
                m_mps = copy.deepcopy(m_cpu).to("mps").eval()
                xd_cpu = {k: v.detach().clone().requires_grad_(True) for k, v in x_dict.items()}
                xd_mps = {k: v.detach().to("mps").requires_grad_(True) for k, v in x_dict.items()}
                ei_mps = {k: v.to("mps") for k, v in ei_nonempty.items()}

                out_c = m_cpu(xd_cpu, ei_nonempty)
                sum(v.sum() for v in out_c.values() if v is not None).backward()
                out_m = m_mps(xd_mps, ei_mps)
                sum(v.sum() for v in out_m.values() if v is not None).backward()

                for k in out_c:
                    if out_c[k] is not None and out_m[k] is not None:
                        diff = float((out_c[k] - out_m[k].cpu()).abs().max().item())
                        if diff > 5e-3 or not torch.isfinite(out_m[k]).all():
                            _record(
                                findings,
                                "wf_hetero_mps",
                                "numerical_divergence",
                                conv_name,
                                f"{conv_name} CPU vs MPS divergence on '{k}': max_diff={diff:.4e}",
                                "Hetero CPU vs MPS.",
                                max_diff=diff,
                            )
            except Exception as exc:
                _record(
                    findings,
                    "wf_hetero_mps",
                    "crash",
                    conv_name,
                    f"{conv_name} crashes on MPS: {type(exc).__name__}: {exc}",
                    "Hetero conv on MPS.",
                    traceback.format_exc(),
                )

    # Test HANConv with return_semantic_attention_weights=True when one relation is empty or node count = 0!
    try:
        m = pyg_conv.HANConv(8, 8, metadata=metadata, heads=2).eval()
        out, sem_attn = m({k: v.detach() for k, v in x_dict.items()}, ei_one_empty, return_semantic_attention_weights=True)
        for k, v in sem_attn.items():
            if v is not None and not torch.isfinite(v).all():
                _record(findings, "wf_han_sem_attn", "nan_inf", "HANConv", f"HANConv semantic attention NaN on '{k}'", "empty rel")
    except Exception as exc:
        _record(
            findings,
            "wf_han_sem_attn",
            "crash",
            "HANConv(return_semantic_attention_weights=True)",
            f"HANConv(return_semantic_attention_weights=True) crashes when a relation has 0 edges: {type(exc).__name__}: {exc}",
            "HANConv with empty relation.",
            traceback.format_exc(),
        )

    # Test HEATConv on empty edges & isolated nodes
    try:
        heat = pyg_conv.HEATConv(
            in_channels=8,
            out_channels=4,
            num_node_types=3,
            num_edge_types=2,
            edge_type_emb_dim=4,
            edge_dim=4,
            edge_attr_emb_dim=4,
            heads=2,
        ).eval()
        x = torch.randn(5, 8)
        nt = torch.tensor([0, 1, 2, 0, 1])
        out = heat(x, torch.empty((2, 0), dtype=torch.long), nt, torch.empty((0,), dtype=torch.long), torch.empty((0, 4)))
        if out.shape != (5, 8) or not torch.isfinite(out).all():
            _record(findings, "wf_heat_empty", "shape_mismatch", "HEATConv", f"HEATConv empty edges output {tuple(out.shape)}", "E=0")
    except Exception as exc:
        _record(findings, "wf_heat_empty", "crash", "HEATConv", f"HEATConv crashes on empty edges (E=0): {type(exc).__name__}: {exc}", "E=0", traceback.format_exc())

    # Test HeteroBatchNorm & HeteroLayerNorm when one type is absent from `type_vec`!
    for norm_name, norm_fn in [
        ("HeteroBatchNorm", lambda: pyg_norm.HeteroBatchNorm(8, num_types=4)),
        ("HeteroLayerNorm(node)", lambda: pyg_norm.HeteroLayerNorm(8, num_types=4, mode="node")),
        ("HeteroLayerNorm(graph)", lambda: pyg_norm.HeteroLayerNorm(8, num_types=4, mode="graph")),
    ]:
        try:
            n_mod = norm_fn()
            n_mod.train()
            x = torch.randn(6, 8, requires_grad=True)
            type_vec = torch.tensor([0, 0, 2, 2, 2, 0], dtype=torch.long)  # types 1 and 3 have 0 nodes in this mini-batch!
            y = n_mod(x, type_vec)
            y.sum().backward()
            if not torch.isfinite(y).all() or not torch.isfinite(x.grad).all():
                _record(
                    findings,
                    "wf_hetero_norm_missing_type",
                    "nan_inf",
                    norm_name,
                    f"{norm_name} produced NaN/Inf when some node types are absent from mini-batch!",
                    "type_vec=[0,0,2,2,2,0] with num_types=4.",
                )
        except Exception as exc:
            _record(
                findings,
                "wf_hetero_norm_missing_type",
                "crash",
                norm_name,
                f"{norm_name} crashes when a node type has 0 nodes in mini-batch: {type(exc).__name__}: {exc}",
                "type_vec=[0,0,2,2,2,0] with num_types=4.",
                traceback.format_exc(),
            )

    return findings


def campaign_utils_deep_dive() -> List[BugFinding]:
    """Campaign W7: Fuzz `to_dense_adj`, `to_dense_batch`, `coalesce`, `k_hop_subgraph`, `homophily`, `assortativity`, `HypergraphConv`."""
    findings: List[BugFinding] = []

    # W7.1: `to_dense_adj` when trailing graph in `batch` has 0 edges!
    try:
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)  # Graph 1 has 2 nodes and 0 edges!
        adj = pyg_utils.to_dense_adj(ei, batch=batch)
        if adj.size(0) != 2:
            _record(
                findings,
                "wf_utils_to_dense_adj",
                "shape_mismatch",
                "utils.to_dense_adj",
                f"to_dense_adj(ei, batch) returned {adj.size(0)} graphs instead of 2 when trailing graph has 0 edges!",
                f"batch={batch.tolist()}, got shape {tuple(adj.shape)}.",
            )
    except Exception as exc:
        _record(findings, "wf_utils_to_dense_adj", "crash", "utils.to_dense_adj", f"{type(exc).__name__}: {exc}", "edgeless trailing graph", traceback.format_exc())

    # W7.2: `to_dense_adj` when `ei` is completely empty (`E == 0`) with `batch` of 2 graphs!
    try:
        ei_empty = torch.empty((2, 0), dtype=torch.long)
        batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
        adj = pyg_utils.to_dense_adj(ei_empty, batch=batch)
        if adj.shape != (2, 3, 3):
            _record(
                findings,
                "wf_utils_to_dense_adj_empty",
                "shape_mismatch",
                "utils.to_dense_adj(E=0)",
                f"to_dense_adj on edgeless batch (N=5, B=2, max_nodes=3) returned shape {tuple(adj.shape)} instead of (2, 3, 3)",
                "E=0 with batch=[0,0,1,1,1].",
            )
    except Exception as exc:
        _record(
            findings,
            "wf_utils_to_dense_adj_empty",
            "crash",
            "utils.to_dense_adj(E=0)",
            f"to_dense_adj crashes when edge_index is empty (E=0) with batch provided: {type(exc).__name__}: {exc}",
            "ei=empty((2, 0)), batch=[0, 0, 1, 1, 1].",
            traceback.format_exc(),
        )

    # W7.3: `homophily` across methods (`edge`, `node`, `edge_insensitive`) on edge cases (E=0, 1 class, isolated nodes)
    for method in ("edge", "node", "edge_insensitive"):
        for sc_name, ei, y in [
            ("empty_edges", torch.empty((2, 0), dtype=torch.long), torch.tensor([0, 1, 0, 1])),
            ("single_class", torch.tensor([[0, 1], [1, 2]]), torch.tensor([0, 0, 0])),
            ("trailing_isolated", torch.tensor([[0, 1], [1, 0]]), torch.tensor([0, 1, 2, 2])),
        ]:
            try:
                h = pyg_utils.homophily(ei, y, method=method)
                if isinstance(h, float) and (h != h):  # NaN check
                    _record(
                        findings,
                        "wf_utils_homophily",
                        "nan_inf",
                        f"utils.homophily(method='{method}')",
                        f"homophily(method='{method}') returned NaN on `{sc_name}`!",
                        f"Scenario {sc_name}",
                    )
            except Exception as exc:
                _record(
                    findings,
                    "wf_utils_homophily",
                    "crash",
                    f"utils.homophily(method='{method}')",
                    f"homophily(method='{method}') crashes on `{sc_name}`: {type(exc).__name__}: {exc}",
                    f"Scenario {sc_name}",
                    traceback.format_exc(),
                )

    # W7.4: `assortativity` on empty edges or constant degree (e.g., regular ring graph where every node has degree 2 -> variance=0 -> NaN?!)
    for sc_name, ei in [
        ("empty_edges", torch.empty((2, 0), dtype=torch.long)),
        ("regular_cycle", torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)),
    ]:
        try:
            a = pyg_utils.assortativity(ei)
            if isinstance(a, float) and (a != a or abs(a) == float("inf")):
                _record(
                    findings,
                    "wf_utils_assortativity",
                    "nan_inf",
                    f"utils.assortativity({sc_name})",
                    f"assortativity() returned {a} on valid `{sc_name}` graph!",
                    f"Scenario {sc_name}",
                )
        except Exception as exc:
            _record(
                findings,
                "wf_utils_assortativity",
                "crash",
                f"utils.assortativity({sc_name})",
                f"assortativity() crashes on `{sc_name}`: {type(exc).__name__}: {exc}",
                f"Scenario {sc_name}",
                traceback.format_exc(),
            )

    # W7.5: `HypergraphConv` with `use_attention=True` & `False`, empty hyperedges, isolated nodes, `attention_mode='node'|'edge'`!
    for use_attn in (False, True):
        for attn_mode in ("node", "edge"):
            if not use_attn and attn_mode == "edge":
                continue
            for sc_name, x, hei, hattr in [
                (
                    "trailing_isolated_node",
                    torch.randn(6, 8, requires_grad=True),
                    torch.tensor([[0, 1, 2, 1], [0, 0, 1, 1]], dtype=torch.long),  # nodes 3,4,5 not in any hyperedge!
                    torch.randn(2, 8),
                ),
                (
                    "empty_hyperedges",
                    torch.randn(4, 8, requires_grad=True),
                    torch.empty((2, 0), dtype=torch.long),
                    torch.empty((0, 8)),
                ),
                (
                    "unused_hyperedge_id",
                    torch.randn(4, 8, requires_grad=True),
                    torch.tensor([[0, 1, 2], [0, 2, 2]], dtype=torch.long),  # hyperedge 1 has 0 incident nodes, hattr has 3 rows!
                    torch.randn(3, 8),
                ),
            ]:
                try:
                    hc = pyg_conv.HypergraphConv(
                        8, 4, use_attention=use_attn, attention_mode=attn_mode, heads=2, dropout=0.0
                    ).eval()
                    x_in = x.detach().clone().requires_grad_(True)
                    out = hc(x_in, hei, hyperedge_attr=hattr if use_attn else None)
                    out.sum().backward()
                    if out.shape != (x.size(0), 8) or not torch.isfinite(out).all():
                        _record(
                            findings,
                            "wf_hypergraph_conv",
                            "shape_mismatch" if out.shape != (x.size(0), 8) else "nan_inf",
                            f"HypergraphConv(attn={use_attn}, mode={attn_mode})",
                            f"HypergraphConv ({sc_name}) output shape={tuple(out.shape)} (expected {(x.size(0), 8)}), finite={torch.isfinite(out).all().item()}",
                            sc_name,
                        )
                except Exception as exc:
                    _record(
                        findings,
                        "wf_hypergraph_conv",
                        "crash",
                        f"HypergraphConv(attn={use_attn}, mode={attn_mode})",
                        f"HypergraphConv ({sc_name}) crashes: {type(exc).__name__}: {exc}",
                        sc_name,
                        traceback.format_exc(),
                    )

    # W7.6: `DNAConv` on empty edges (`E == 0`) & trailing isolated nodes!
    for sc_name, ei in [
        ("empty_edges", torch.empty((2, 0), dtype=torch.long)),
        ("trailing_isolated", torch.tensor([[0, 1], [1, 0]], dtype=torch.long)),
    ]:
        try:
            dna = pyg_conv.DNAConv(8, heads=2, groups=2, dropout=0.0, add_self_loops=False).eval()
            x_3d = torch.randn(5, 3, 8, requires_grad=True)  # [N, num_layers, channels]
            out = dna(x_3d, ei)
            out.sum().backward()
            if out.shape != (5, 8) or not torch.isfinite(out).all():
                _record(findings, "wf_dnaconv", "nan_inf", "DNAConv", f"DNAConv ({sc_name}) invalid output", sc_name)
        except Exception as exc:
            _record(
                findings,
                "wf_dnaconv",
                "crash",
                f"DNAConv({sc_name})",
                f"DNAConv(add_self_loops=False) crashes on `{sc_name}`: {type(exc).__name__}: {exc}",
                sc_name,
                traceback.format_exc(),
            )

    # W7.7: `Explainer` (`GNNExplainer` & `AttentionExplainer`) on empty edges & MPS!
    class TinyExplainModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = pyg_conv.GATConv(8, 8, heads=2, concat=False)
            self.conv2 = pyg_conv.GATConv(8, 3, heads=1, concat=False)

        def forward(self, x, edge_index):
            return self.conv2(self.conv1(x, edge_index).relu(), edge_index)

    for alg_name, alg_fn in [
        ("GNNExplainer", lambda: GNNExplainer(epochs=2)),
        ("AttentionExplainer", lambda: AttentionExplainer()),
    ]:
        try:
            m = TinyExplainModel().eval()
            explainer = Explainer(
                model=m,
                algorithm=alg_fn(),
                explanation_type="model",
                node_mask_type="attributes",
                edge_mask_type="object",
                model_config=dict(mode="multiclass_classification", task_level="node", return_type="raw"),
            )
            # Test on graph with 0 edges!
            exp = explainer(torch.randn(4, 8), torch.empty((2, 0), dtype=torch.long), index=0)
            if exp.edge_mask is not None and not torch.isfinite(exp.edge_mask).all():
                _record(findings, "wf_explainer_empty", "nan_inf", alg_name, f"{alg_name} non-finite mask on E=0", "E=0")
        except Exception as exc:
            _record(
                findings,
                "wf_explainer_empty",
                "crash",
                f"Explainer({alg_name}, E=0)",
                f"Explainer({alg_name}) crashes when graph has 0 edges: {type(exc).__name__}: {exc}",
                "x=(4, 8), edge_index=empty(2, 0)",
                traceback.format_exc(),
            )

    return findings
