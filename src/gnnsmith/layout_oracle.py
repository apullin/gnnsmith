"""Sparse Adjacency Layout & `return_attention_weights=True` Differential Oracle."""

from __future__ import annotations

import traceback
from typing import Callable, List

import torch
import torch.nn as nn
from torch_geometric import EdgeIndex
import torch_geometric.nn.conv as pyg_conv
from torch_geometric.utils import to_torch_coo_tensor, to_torch_csc_tensor, to_torch_csr_tensor

from gnnsmith.oracles import BugFinding, _is_close, _max_abs_diff


def _rec(
    findings: List[BugFinding],
    oracle: str,
    category: str,
    target: str,
    summary: str,
    tb: str = "",
    max_diff: float | None = None,
) -> None:
    sig = f"layout|{oracle}|{category}|{target}|{summary[:110]}"
    findings.append(
        BugFinding(
            oracle=oracle,
            category=category,
            summary=summary,
            detail=target,
            op_types=[target],
            topology_kind="sparse_layout",
            seed=0,
            max_diff=max_diff,
            traceback_str=tb,
            signature_key=sig,
        )
    )


def fuzz_sparse_layouts_and_attention() -> List[BugFinding]:
    """Compare LongTensor vs EdgeIndex(row/col) vs torch.sparse COO/CSR/CSC.t() across PyG convs."""
    findings: List[BugFinding] = []
    N, C = 8, 8
    gen = torch.Generator().manual_seed(99)
    x = torch.randn(N, C, generator=gen)
    x_0 = torch.randn(N, C, generator=gen)
    pos = torch.randn(N, 3, generator=gen)
    # Directed graph with trailing isolated node 7 (NO duplicate edges so COO/CSR/CSC are 1-to-1)
    ei = torch.tensor([[0, 0, 1, 2, 3, 5], [1, 3, 2, 4, 0, 6]], dtype=torch.long)

    ei_row = EdgeIndex(ei.clone(), sparse_size=(N, N)).sort_by("row").values
    ei_col = EdgeIndex(ei.clone(), sparse_size=(N, N)).sort_by("col").values
    # PyG convention for torch.sparse adjacency: transposed A^T (i.e. `to_torch_csc_tensor(ei).t()` = CSR of dst<-src!)
    adj_coo = to_torch_coo_tensor(ei.flip(0), size=(N, N))
    adj_csr = to_torch_csc_tensor(ei, size=(N, N)).t()  # layout == sparse_csr

    layouts = [
        ("EdgeIndex(row)", ei_row),
        ("EdgeIndex(col)", ei_col),
        ("torch.sparse_coo", adj_coo),
        ("torch.sparse_csr", adj_csr),
    ]

    conv_specs: list[tuple[str, Callable[[], nn.Module], Callable[[nn.Module, torch.Tensor, object], torch.Tensor]]] = [
        ("GCNConv", lambda: pyg_conv.GCNConv(C, C), lambda m, x_, a_: m(x_, a_)),
        ("SAGEConv(mean)", lambda: pyg_conv.SAGEConv(C, C, aggr="mean"), lambda m, x_, a_: m(x_, a_)),
        ("SAGEConv(max)", lambda: pyg_conv.SAGEConv(C, C, aggr="max"), lambda m, x_, a_: m(x_, a_)),
        ("GraphConv", lambda: pyg_conv.GraphConv(C, C), lambda m, x_, a_: m(x_, a_)),
        ("GATConv", lambda: pyg_conv.GATConv(C, C // 2, heads=2), lambda m, x_, a_: m(x_, a_)),
        ("GATv2Conv", lambda: pyg_conv.GATv2Conv(C, C // 2, heads=2), lambda m, x_, a_: m(x_, a_)),
        ("TransformerConv", lambda: pyg_conv.TransformerConv(C, C // 2, heads=2), lambda m, x_, a_: m(x_, a_)),
        ("GINConv", lambda: pyg_conv.GINConv(nn.Linear(C, C)), lambda m, x_, a_: m(x_, a_)),
        ("TAGConv", lambda: pyg_conv.TAGConv(C, C, K=2), lambda m, x_, a_: m(x_, a_)),
        ("SGConv", lambda: pyg_conv.SGConv(C, C, K=2), lambda m, x_, a_: m(x_, a_)),
        ("SSGConv", lambda: pyg_conv.SSGConv(C, C, alpha=0.2, K=2), lambda m, x_, a_: m(x_, a_)),
        ("APPNP", lambda: pyg_conv.APPNP(K=2, alpha=0.2), lambda m, x_, a_: m(x_, a_)),
        ("ARMAConv", lambda: pyg_conv.ARMAConv(C, C), lambda m, x_, a_: m(x_, a_)),
        ("AGNNConv", lambda: pyg_conv.AGNNConv(), lambda m, x_, a_: m(x_, a_)),
        ("ResGatedGraphConv", lambda: pyg_conv.ResGatedGraphConv(C, C), lambda m, x_, a_: m(x_, a_)),
        ("GatedGraphConv", lambda: pyg_conv.GatedGraphConv(C, num_layers=2), lambda m, x_, a_: m(x_, a_)),
        ("MFConv", lambda: pyg_conv.MFConv(C, C), lambda m, x_, a_: m(x_, a_)),
        ("FeaStConv", lambda: pyg_conv.FeaStConv(C, C, heads=2), lambda m, x_, a_: m(x_, a_)),
        ("LEConv", lambda: pyg_conv.LEConv(C, C), lambda m, x_, a_: m(x_, a_)),
        ("ClusterGCNConv", lambda: pyg_conv.ClusterGCNConv(C, C, diag_lambda=0.5), lambda m, x_, a_: m(x_, a_)),
        ("GENConv", lambda: pyg_conv.GENConv(C, C), lambda m, x_, a_: m(x_, a_)),
        ("EGConv", lambda: pyg_conv.EGConv(C, C, num_heads=2, num_bases=2), lambda m, x_, a_: m(x_, a_)),
        ("GeneralConv", lambda: pyg_conv.GeneralConv(C, C), lambda m, x_, a_: m(x_, a_)),
        ("SimpleConv", lambda: pyg_conv.SimpleConv(aggr="mean"), lambda m, x_, a_: m(x_, a_)),
        ("LGConv", lambda: pyg_conv.LGConv(), lambda m, x_, a_: m(x_, a_)),
        ("FAConv", lambda: pyg_conv.FAConv(C), lambda m, x_, a_: m(x_, x_0, a_)),
        ("GCN2Conv", lambda: pyg_conv.GCN2Conv(C, alpha=0.2, theta=0.5, layer=1), lambda m, x_, a_: m(x_, x_0, a_)),
        ("PointNetConv", lambda: pyg_conv.PointNetConv(nn.Linear(C + 3, C), nn.Linear(C, C)), lambda m, x_, a_: m(x_, pos, a_)),
        ("PointTransformerConv", lambda: pyg_conv.PointTransformerConv(C, C), lambda m, x_, a_: m(x_, pos, a_)),
    ]

    for c_name, factory, caller in conv_specs:
        torch.manual_seed(7)
        m = factory().eval()
        x_ref = x.detach().clone().requires_grad_(True)
        out_ref = caller(m, x_ref, ei)
        out_ref.sum().backward()
        gx_ref = x_ref.grad.detach().clone()

        for l_name, adj in layouts:
            try:
                m.zero_grad(set_to_none=True)
                x_in = x.detach().clone().requires_grad_(True)
                out_l = caller(m, x_in, adj)
                out_l.sum().backward()

                if out_l.shape != out_ref.shape:
                    _rec(
                        findings,
                        "layout_shape",
                        "shape_mismatch",
                        f"{c_name}[{l_name}]",
                        f"Shape {tuple(out_l.shape)} != LongTensor shape {tuple(out_ref.shape)}",
                    )
                elif not torch.isfinite(out_l).all():
                    _rec(findings, "layout_finite", "nan_inf", f"{c_name}[{l_name}]", "Non-finite output on sparse layout")
                elif not _is_close(out_ref, out_l, atol=2e-3, rtol=5e-3):
                    diff = _max_abs_diff(out_ref, out_l)
                    if diff > 1e-2:
                        _rec(
                            findings,
                            "layout_parity",
                            "numerical_divergence",
                            f"{c_name}[{l_name}]",
                            f"{l_name} forward diverged from LongTensor: max_diff={diff:.4e}",
                            max_diff=diff,
                        )
                elif x_in.grad is not None and not _is_close(gx_ref, x_in.grad, atol=5e-3, rtol=1e-2):
                    gdiff = _max_abs_diff(gx_ref, x_in.grad)
                    if gdiff > 2e-2:
                        _rec(
                            findings,
                            "layout_grad_parity",
                            "grad_divergence",
                            f"{c_name}[{l_name}]",
                            f"{l_name} x.grad diverged from LongTensor: max_diff={gdiff:.4e}",
                            max_diff=gdiff,
                        )
            except NotImplementedError:
                pass
            except Exception as exc:
                _rec(
                    findings,
                    "layout_parity",
                    "crash",
                    f"{c_name}[{l_name}]",
                    f"{type(exc).__name__}: {exc}",
                    tb=traceback.format_exc(),
                )

    # Also test `return_attention_weights=True` across layouts & empty edges (`E=0`)!
    attn_specs: list[tuple[str, Callable[[], nn.Module], bool]] = [
        ("GATConv", lambda: pyg_conv.GATConv(C, C // 2, heads=2, add_self_loops=False), False),
        ("GATConv(loops)", lambda: pyg_conv.GATConv(C, C // 2, heads=2, add_self_loops=True), False),
        ("GATv2Conv", lambda: pyg_conv.GATv2Conv(C, C // 2, heads=2, add_self_loops=False), False),
        ("GATv2Conv(loops)", lambda: pyg_conv.GATv2Conv(C, C // 2, heads=2, add_self_loops=True), False),
        ("TransformerConv", lambda: pyg_conv.TransformerConv(C, C // 2, heads=2), False),
        ("FAConv", lambda: pyg_conv.FAConv(C, add_self_loops=False), True),
    ]
    for a_name, a_fn, needs_x0 in attn_specs:
        for sc_name, adj in [
            ("LongTensor", ei),
            ("empty_edges", torch.empty((2, 0), dtype=torch.long)),
            ("EdgeIndex(row)", ei_row),
            ("torch.sparse_coo", adj_coo),
            ("torch.sparse_csr", adj_csr),
        ]:
            try:
                torch.manual_seed(3)
                m = a_fn().eval()
                res = m(x, x_0, adj, return_attention_weights=True) if needs_x0 else m(x, adj, return_attention_weights=True)
                out_a = res[0]
                if out_a.shape != (N, C) or not torch.isfinite(out_a).all():
                    _rec(findings, "return_attn", "shape_mismatch", f"{a_name}[{sc_name}]", f"Invalid output {tuple(out_a.shape)}")
            except NotImplementedError:
                pass
            except Exception as exc:
                _rec(
                    findings,
                    "return_attn",
                    "crash",
                    f"{a_name}[{sc_name}]",
                    f"return_attention_weights=True on `{sc_name}` crashes: {type(exc).__name__}: {exc}",
                    tb=traceback.format_exc(),
                )

    return findings
