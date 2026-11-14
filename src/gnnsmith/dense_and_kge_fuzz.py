"""Wave 7 Fuzzing: Dense GNNs vs Sparse Parity, Masked-Padding Leakage, Dense Pooling, KGE, Encoding, & Attention."""

from __future__ import annotations

import copy
import traceback
from typing import Callable, List

import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.nn.attention as pyg_attn
import torch_geometric.nn.dense as pyg_dense
import torch_geometric.nn.encoding as pyg_enc
import torch_geometric.nn.kge as pyg_kge
from torch_geometric.utils import to_dense_adj, to_dense_batch

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
    sig = f"w7|{oracle}|{category}|{target}|{summary[:110]}"
    findings.append(
        BugFinding(
            oracle=oracle,
            category=category,
            summary=summary,
            detail=detail,
            op_types=[target],
            topology_kind="dense_kge_attn",
            seed=0,
            max_diff=max_diff,
            traceback_str=tb,
            signature_key=sig,
        )
    )


# ---------------------------------------------------------------------------
# W7.1: Dense Conv vs Sparse Conv Parity + Masked-Padding Non-Leakage Oracle!
# ---------------------------------------------------------------------------
def fuzz_dense_convs_and_padding() -> List[BugFinding]:
    findings: List[BugFinding] = []
    C_in, C_out = 8, 8
    torch.manual_seed(42)

    # Graph 0: 4 nodes, 4 edges; Graph 1: 2 nodes, 0 edges (padded to N_max=4)
    x_sparse = torch.randn(6, C_in)
    batch = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.long)
    ei_sparse = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)  # undirected so adj == adj.T

    x_dense, mask = to_dense_batch(x_sparse, batch)  # [2, 4, 8], mask=[2, 4]
    adj_dense = to_dense_adj(ei_sparse, batch)       # [2, 4, 4]

    # Create a version where padded node positions (~mask) in `x_dense` hold non-zero values (`999.0`)!
    # A proper `mask`-aware DenseConv MUST zero/ignore `x[~mask]` so valid nodes `out[mask]` are unaffected!
    x_dirty = x_dense.clone()
    x_dirty[~mask] = 50.0

    # Test 1: Masked-padding invariance + Edgeless graph finiteness across all 5 Dense*Conv layers!
    for add_loop in (True, False):
        dense_convs: list[tuple[str, nn.Module]] = [
            (f"DenseGCNConv(loop={add_loop})", pyg_dense.DenseGCNConv(C_in, C_out, improved=False)),
            (f"DenseSAGEConv", pyg_dense.DenseSAGEConv(C_in, C_out)),
            (f"DenseGraphConv", pyg_dense.DenseGraphConv(C_in, C_out)),
            (f"DenseGINConv(loop={add_loop})", pyg_dense.DenseGINConv(nn.Linear(C_in, C_out))),
            (f"DenseGATConv(loop={add_loop})", pyg_dense.DenseGATConv(C_in, C_out // 2, heads=2, dropout=0.0)),
        ]
        for name, mod in dense_convs:
            mod.eval()
            try:
                if "DenseGCNConv" in name or "DenseGINConv" in name or "DenseGATConv" in name:
                    y_clean = mod(x_dense, adj_dense, mask=mask, add_loop=add_loop)
                    y_dirty = mod(x_dirty, adj_dense, mask=mask, add_loop=add_loop)
                else:
                    if not add_loop:
                        continue
                    y_clean = mod(x_dense, adj_dense, mask=mask)
                    y_dirty = mod(x_dirty, adj_dense, mask=mask)

                if not torch.isfinite(y_clean).all():
                    _rec(findings, "dense_conv_finite", "nan_inf", name, f"{name} produced NaN/Inf on valid batch with edgeless graph!")
                elif bool((y_clean[~mask] != 0).any().item()):
                    _rec(findings, "dense_conv_mask_out", "numerical_divergence", name, f"{name} left non-zero values at masked output positions `~mask`!")
                elif not _is_close(y_clean[mask], y_dirty[mask], atol=1e-4, rtol=1e-4):
                    diff = _max_abs_diff(y_clean[mask], y_dirty[mask])
                    _rec(
                        findings,
                        "dense_conv_pad_leak",
                        "numerical_divergence",
                        name,
                        f"{name} leaks padded `x[~mask]` values into valid node outputs! max_diff={diff:.4e}",
                        max_diff=diff,
                    )
            except Exception as exc:
                _rec(findings, "dense_conv_crash", "crash", name, f"{type(exc).__name__}: {exc}", tb=traceback.format_exc())

    # Test 2: Exact Sparse-vs-Dense mathematical parity on `DenseGCNConv`, `DenseSAGEConv`, `DenseGraphConv`, `DenseGINConv`, `DenseGATConv`!
    # Check DenseGATConv vs GATConv on a 1-graph input (no padding) AND 2-graph padded input!
    try:
        torch.manual_seed(7)
        s_gat = pyg_nn.GATConv(C_in, C_out // 2, heads=2, add_self_loops=True, bias=True).eval()
        d_gat = pyg_dense.DenseGATConv(C_in, C_out // 2, heads=2, bias=True).eval()
        d_gat.lin.load_state_dict(s_gat.lin.state_dict())
        d_gat.att_src.data.copy_(s_gat.att_src.data)
        d_gat.att_dst.data.copy_(s_gat.att_dst.data)
        d_gat.bias.data.copy_(s_gat.bias.data)

        y_sparse = s_gat(x_sparse, ei_sparse)
        y_dense = d_gat(x_dense, adj_dense, mask=mask, add_loop=True)[mask]
        diff = _max_abs_diff(y_sparse, y_dense)
        if not torch.isfinite(y_dense).all() or diff > 1e-3:
            _rec(
                findings,
                "dense_vs_sparse_gat",
                "nan_inf" if not torch.isfinite(y_dense).all() else "numerical_divergence",
                "DenseGATConv",
                f"DenseGATConv diverged from GATConv with identical weights (graph 1 has 0 edges)! max_diff={diff:.4e}",
                max_diff=diff,
            )
    except Exception as exc:
        _rec(findings, "dense_vs_sparse_gat", "crash", "DenseGATConv", f"{type(exc).__name__}: {exc}", tb=traceback.format_exc())

    return findings


# ---------------------------------------------------------------------------
# W7.2: Dense Pooling (`dense_diff_pool`, `dense_mincut_pool`, `DMoNPooling`)
# ---------------------------------------------------------------------------
def fuzz_dense_pooling() -> List[BugFinding]:
    findings: List[BugFinding] = []
    torch.manual_seed(0)

    x_1 = torch.randn(1, 3, 8, requires_grad=True)
    s_1 = torch.randn(1, 3, 2, requires_grad=True)
    adj_zero = torch.zeros(1, 3, 3)  # Edgeless graph!
    adj_norm = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]])

    # Check padded vs unpadded invariance AND edgeless (`adj == 0`) finiteness!
    for sc_name, adj_in in [("normal", adj_norm), ("empty_edges", adj_zero)]:
        # 1. dense_diff_pool
        try:
            xv = x_1.detach().clone().requires_grad_(True)
            sv = s_1.detach().clone().requires_grad_(True)
            xo, ao, l1, l2 = pyg_dense.dense_diff_pool(xv, adj_in, sv)
            (xo.sum() + l1 + l2).backward()
            if not (torch.isfinite(xo).all() and torch.isfinite(l1).all() and torch.isfinite(l2).all()):
                _rec(findings, "dense_pool_finite", "nan_inf", f"dense_diff_pool[{sc_name}]", f"NaN/Inf in diff_pool ({sc_name})")
        except Exception as exc:
            _rec(findings, "dense_pool_crash", "crash", f"dense_diff_pool[{sc_name}]", f"{type(exc).__name__}: {exc}", tb=traceback.format_exc())

        # 2. dense_mincut_pool
        try:
            xv = x_1.detach().clone().requires_grad_(True)
            sv = s_1.detach().clone().requires_grad_(True)
            xo, ao, mc_loss, o_loss = pyg_dense.dense_mincut_pool(xv, adj_in, sv)
            (xo.sum() + mc_loss + o_loss).backward()
            if not (torch.isfinite(xo).all() and torch.isfinite(mc_loss).all() and torch.isfinite(o_loss).all()):
                _rec(
                    findings,
                    "dense_pool_finite",
                    "nan_inf",
                    f"dense_mincut_pool[{sc_name}]",
                    f"NaN/Inf in dense_mincut_pool ({sc_name}): mc_loss={mc_loss.item()}, o_loss={o_loss.item()}",
                )
            elif xv.grad is not None and not torch.isfinite(xv.grad).all():
                _rec(findings, "dense_pool_grad", "nan_inf", f"dense_mincut_pool[{sc_name}]", "NaN in x.grad")
        except Exception as exc:
            _rec(findings, "dense_pool_crash", "crash", f"dense_mincut_pool[{sc_name}]", f"{type(exc).__name__}: {exc}", tb=traceback.format_exc())

        # 3. DMoNPooling
        try:
            dmon = pyg_dense.DMoNPooling([8, 8], k=2).eval()
            xv = x_1.detach().clone().requires_grad_(True)
            s_out, xo, ao, sp_loss, o_loss, c_loss = dmon(xv, adj_in)
            (xo.sum() + sp_loss + o_loss + c_loss).backward()
            if not (torch.isfinite(xo).all() and torch.isfinite(sp_loss).all() and torch.isfinite(o_loss).all() and torch.isfinite(c_loss).all()):
                _rec(
                    findings,
                    "dense_pool_finite",
                    "nan_inf",
                    f"DMoNPooling[{sc_name}]",
                    f"NaN/Inf in DMoNPooling ({sc_name}): sp={sp_loss.item()}, o={o_loss.item()}, c={c_loss.item()}",
                )
            elif xv.grad is not None and not torch.isfinite(xv.grad).all():
                _rec(findings, "dense_pool_grad", "nan_inf", f"DMoNPooling[{sc_name}]", "NaN in DMoNPooling backward grad")
        except Exception as exc:
            _rec(findings, "dense_pool_crash", "crash", f"DMoNPooling[{sc_name}]", f"{type(exc).__name__}: {exc}", tb=traceback.format_exc())

    # Test 4: DMoNPooling & dense_mincut_pool masked-padding invariance!
    # Graph of 3 nodes unpadded `[1, 3, 8]` vs padded to 5 nodes `[1, 5, 8]` with `mask=[[T, T, T, F, F]]` and dirty pad `50.0`!
    x_pad = torch.cat([x_1.detach(), torch.full((1, 2, 8), 50.0)], dim=1)
    s_pad = torch.cat([s_1.detach(), torch.full((1, 2, 2), 50.0)], dim=1)
    adj_pad = torch.zeros(1, 5, 5)
    adj_pad[:, :3, :3] = adj_norm
    mask_pad = torch.tensor([[True, True, True, False, False]])

    try:
        xo_ref, ao_ref, mc_ref, o_ref = pyg_dense.dense_mincut_pool(x_1.detach(), adj_norm, s_1.detach())
        xo_pad, ao_pad, mc_pad, o_pad = pyg_dense.dense_mincut_pool(x_pad, adj_pad, s_pad, mask=mask_pad)
        diff_x = _max_abs_diff(xo_ref, xo_pad)
        diff_mc = abs(float(mc_ref.item() - mc_pad.item()))
        if diff_x > 1e-4 or diff_mc > 1e-4:
            _rec(
                findings,
                "dense_pool_pad_leak",
                "numerical_divergence",
                "dense_mincut_pool(mask)",
                f"dense_mincut_pool leaks padded nodes despite mask: x_diff={diff_x:.4e}, mc_loss_diff={diff_mc:.4e}",
                max_diff=max(diff_x, diff_mc),
            )
    except Exception as exc:
        _rec(findings, "dense_pool_pad_leak", "crash", "dense_mincut_pool(mask)", f"{type(exc).__name__}: {exc}", tb=traceback.format_exc())

    try:
        dmon = pyg_dense.DMoNPooling([8, 8], k=2).eval()
        _, xo_r, _, sp_r, o_r, c_r = dmon(x_1.detach(), adj_norm)
        _, xo_p, _, sp_p, o_p, c_p = dmon(x_pad, adj_pad, mask=mask_pad)
        diff_x = _max_abs_diff(xo_r, xo_p)
        diff_l = max(abs(float(sp_r - sp_p)), abs(float(o_r - o_p)), abs(float(c_r - c_p)))
        if diff_x > 1e-4 or diff_l > 1e-4:
            _rec(
                findings,
                "dense_pool_pad_leak",
                "numerical_divergence",
                "DMoNPooling(mask)",
                f"DMoNPooling leaks padded nodes despite mask! x_diff={diff_x:.4e}, loss_diff={diff_l:.4e} (c_ref={c_r.item():.4f} vs c_pad={c_p.item():.4f})",
                max_diff=max(diff_x, diff_l),
            )
    except Exception as exc:
        _rec(findings, "dense_pool_pad_leak", "crash", "DMoNPooling(mask)", f"{type(exc).__name__}: {exc}", tb=traceback.format_exc())

    return findings


# ---------------------------------------------------------------------------
# W7.3: KGE (`TransE`, `ComplEx`, `DistMult`, `RotatE`), `Encoding`, `Attention`, & `HeteroLinear`
# ---------------------------------------------------------------------------
def fuzz_kge_encoding_attention_heterolinear() -> List[BugFinding]:
    findings: List[BugFinding] = []
    torch.manual_seed(0)

    # 1. KGE models on normal & empty batches + MPS!
    for k_name, k_fn in [
        ("TransE", lambda: pyg_kge.TransE(num_nodes=10, num_relations=3, hidden_channels=8)),
        ("ComplEx", lambda: pyg_kge.ComplEx(num_nodes=10, num_relations=3, hidden_channels=8)),
        ("DistMult", lambda: pyg_kge.DistMult(num_nodes=10, num_relations=3, hidden_channels=8)),
        ("RotatE", lambda: pyg_kge.RotatE(num_nodes=10, num_relations=3, hidden_channels=8, margin=1.0)),
    ]:
        for sc_name, h, r, t in [
            ("normal", torch.tensor([0, 2, 4]), torch.tensor([0, 1, 2]), torch.tensor([1, 3, 5])),
            ("empty_batch", torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
        ]:
            try:
                m = k_fn()
                score = m(h, r, t)
                loss = m.loss(h, r, t)
                if not torch.isfinite(score).all() or not torch.isfinite(loss).all():
                    _rec(findings, "kge_finite", "nan_inf", f"{k_name}[{sc_name}]", f"score_finite={torch.isfinite(score).all().item()}, loss={loss.item()}")
            except Exception as exc:
                _rec(findings, "kge_crash", "crash", f"{k_name}[{sc_name}]", f"{type(exc).__name__}: {exc}", tb=traceback.format_exc())

    # 2. PositionalEncoding & TemporalEncoding across empty, bfloat16, MPS!
    for enc_name, enc_mod in [
        ("PositionalEncoding", pyg_enc.PositionalEncoding(8)),
        ("TemporalEncoding", pyg_enc.TemporalEncoding(8)),
    ]:
        for sc_name, dev, dt, inp in [
            ("empty_cpu", "cpu", torch.float32, torch.empty(0)),
            ("bf16_cpu", "cpu", torch.bfloat16, torch.tensor([0.0, 1.5, 10.0], dtype=torch.bfloat16)),
            ("mps_fp32", "mps" if torch.backends.mps.is_available() else "cpu", torch.float32, torch.tensor([0.0, 1.5, 10.0])),
        ]:
            try:
                em = copy.deepcopy(enc_mod).to(device=dev, dtype=dt)
                out = em(inp.to(device=dev, dtype=dt))
                if out.shape != (inp.numel(), 8) or not torch.isfinite(out).all():
                    _rec(findings, "encoding_check", "shape_mismatch" if out.shape != (inp.numel(), 8) else "nan_inf", f"{enc_name}[{sc_name}]", f"shape={tuple(out.shape)}, finite={torch.isfinite(out).all().item()}")
            except Exception as exc:
                _rec(findings, "encoding_check", "crash", f"{enc_name}[{sc_name}]", f"{type(exc).__name__}: {exc}", tb=traceback.format_exc())

    # 3. PerformerAttention, SGFormerAttention, PolynormerAttention on masked padding & empty/1-token!
    x_seq = torch.randn(2, 5, 8)
    mask_seq = torch.tensor([[True, True, True, True, True], [True, True, False, False, False]])
    x_seq_dirty = x_seq.clone()
    x_seq_dirty[~mask_seq] = 100.0

    for a_name, a_fn in [
        ("PerformerAttention", lambda: pyg_attn.PerformerAttention(channels=8, heads=2)),
        ("SGFormerAttention", lambda: pyg_attn.SGFormerAttention(channels=8, heads=2, head_channels=4)),
        ("PolynormerAttention", lambda: pyg_attn.PolynormerAttention(channels=8, heads=2, head_channels=4)),
    ]:
        try:
            am = a_fn().eval()
            y_clean = am(x_seq, mask=mask_seq)
            y_dirty = am(x_seq_dirty, mask=mask_seq)
            if not torch.isfinite(y_clean).all():
                _rec(findings, "attn_finite", "nan_inf", a_name, "Non-finite output with padding mask")
            elif not _is_close(y_clean[mask_seq], y_dirty[mask_seq], atol=1e-4, rtol=1e-4):
                diff = _max_abs_diff(y_clean[mask_seq], y_dirty[mask_seq])
                _rec(
                    findings,
                    "attn_pad_leak",
                    "numerical_divergence",
                    a_name,
                    f"{a_name} leaks masked padding tokens `x[~mask]` into valid token representations! max_diff={diff:.4e}",
                    max_diff=diff,
                )
        except Exception as exc:
            _rec(findings, "attn_crash", "crash", a_name, f"{type(exc).__name__}: {exc}", tb=traceback.format_exc())

    # 4. HeteroLinear & HeteroDictLinear when a type has 0 rows or absent type!
    try:
        hl = pyg_dense.HeteroLinear(8, 4, num_types=4).eval()
        x_h = torch.randn(5, 8)
        tv = torch.tensor([0, 0, 2, 2, 2], dtype=torch.long)  # types 1 and 3 absent!
        out_hl = hl(x_h, tv)
        if out_hl.shape != (5, 4) or not torch.isfinite(out_hl).all():
            _rec(findings, "hetero_linear", "nan_inf", "HeteroLinear", f"shape={tuple(out_hl.shape)}")

        out_empty = hl(torch.empty(0, 8), torch.empty(0, dtype=torch.long))
        if out_empty.shape != (0, 4):
            _rec(findings, "hetero_linear", "shape_mismatch", "HeteroLinear(N=0)", f"shape={tuple(out_empty.shape)}")
    except Exception as exc:
        _rec(findings, "hetero_linear", "crash", "HeteroLinear", f"{type(exc).__name__}: {exc}", tb=traceback.format_exc())

    try:
        hdl = pyg_dense.HeteroDictLinear({"a": 8, "b": 6}, 4).eval()
        out_hdl = hdl({"a": torch.randn(3, 8), "b": torch.empty(0, 6)})
        if out_hdl["b"].shape != (0, 4):
            _rec(findings, "hetero_dict_linear", "shape_mismatch", "HeteroDictLinear(empty)", f"shape={tuple(out_hdl['b'].shape)}")
    except Exception as exc:
        _rec(findings, "hetero_dict_linear", "crash", "HeteroDictLinear", f"{type(exc).__name__}: {exc}", tb=traceback.format_exc())

    return findings


def run_all_wave7_campaigns() -> List[BugFinding]:
    all_findings: List[BugFinding] = []
    for label, fn in [
        ("W7.1: Dense Convs vs Sparse Parity & Masked-Padding Leakage", fuzz_dense_convs_and_padding),
        ("W7.2: Dense Pooling (diff_pool, mincut_pool, DMoNPooling) Edgeless & Mask Leakage", fuzz_dense_pooling),
        ("W7.3: KGE (TransE/ComplEx/DistMult/RotatE), Encoding, Attention, & HeteroLinear", fuzz_kge_encoding_attention_heterolinear),
    ]:
        print(f">>> Running {label} ...", flush=True)
        hits = fn()
        print(f"    └─ {len(hits)} raw hits", flush=True)
        all_findings.extend(hits)
    return all_findings
