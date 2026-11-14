"""First-class Bipartite (N_src != N_dst), `flow` Duality, and Z3 Heterogeneous GNN Fuzzing."""

from __future__ import annotations

import copy
import random
import traceback
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.nn.conv as pyg_conv

from gnnsmith.constraints import Z3GraphContext
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
    sig = f"bip|{oracle}|{category}|{target}|{summary[:110]}"
    findings.append(
        BugFinding(
            oracle=oracle,
            category=category,
            summary=summary,
            detail=detail,
            op_types=[target],
            topology_kind="bipartite_hetero",
            seed=0,
            max_diff=max_diff,
            traceback_str=tb,
            signature_key=sig,
        )
    )


def fuzz_bipartite_convs(num_trials_per_op: int = 4) -> List[BugFinding]:
    """Fuzz all PairTensor-capable PyG convs on rectangular (N_src != N_dst, C_src != C_dst) bipartite graphs."""
    findings: List[BugFinding] = []
    rng = random.Random(2026)

    # (name, builder(c_src, c_dst, c_out, edge_dim), supports_none_dst, call_kind)
    bipartite_specs: list[tuple[str, Callable[[int, int, int, int], nn.Module], bool, str]] = [
        ("SAGEConv", lambda cs, cd, co, ed: pyg_conv.SAGEConv((cs, cd), co, aggr=rng.choice(["mean", "max", "sum"]), project=rng.choice([False, True])), True, "std_size"),
        ("GraphConv", lambda cs, cd, co, ed: pyg_conv.GraphConv((cs, cd), co, aggr=rng.choice(["add", "mean", "max"])), True, "weight_size"),
        ("GATConv", lambda cs, cd, co, ed: pyg_conv.GATConv((cs, cd), co // 2, heads=2, add_self_loops=False, edge_dim=ed), True, "edge_size"),
        ("GATv2Conv", lambda cs, cd, co, ed: pyg_conv.GATv2Conv((cs, cd), co // 2, heads=2, add_self_loops=False, edge_dim=ed), False, "edge_nosize"),
        ("TransformerConv", lambda cs, cd, co, ed: pyg_conv.TransformerConv((cs, cd), co // 2, heads=2, edge_dim=ed), False, "edge_nosize"),
        ("GINConv", lambda cs, cd, co, ed: pyg_conv.GINConv(nn.Linear(cs, co)), False, "same_c_size"),
        ("GINEConv", lambda cs, cd, co, ed: pyg_conv.GINEConv(nn.Linear(cs, co), edge_dim=ed), False, "same_c_edge_size"),
        ("ResGatedGraphConv", lambda cs, cd, co, ed: pyg_conv.ResGatedGraphConv((cs, cd), co, edge_dim=ed), False, "edge_nosize"),
        ("MFConv", lambda cs, cd, co, ed: pyg_conv.MFConv((cs, cd), co, max_degree=4), True, "std_size"),
        ("RGCNConv", lambda cs, cd, co, ed: pyg_conv.RGCNConv((cs, cd), co, num_relations=3, aggr=rng.choice(["mean", "add", "max"])), False, "rel_nosize"),
        ("FastRGCNConv", lambda cs, cd, co, ed: pyg_conv.FastRGCNConv((cs, cd), co, num_relations=3, aggr=rng.choice(["mean", "add"])), False, "rel_nosize"),
        ("FiLMConv", lambda cs, cd, co, ed: pyg_conv.FiLMConv((cs, cd), co, num_relations=3, aggr=rng.choice(["mean", "max", "add"])), False, "rel_nosize"),
        ("GMMConv", lambda cs, cd, co, ed: pyg_conv.GMMConv((cs, cd), co, dim=ed, kernel_size=2), True, "edge_size"),
        ("NNConv", lambda cs, cd, co, ed: pyg_conv.NNConv((cs, cd), co, nn=nn.Linear(ed, cs * co)), True, "edge_size"),
        ("CGConv", lambda cs, cd, co, ed: pyg_conv.CGConv((cs, cd), dim=ed), False, "cg_nosize"),
        ("EdgeConv", lambda cs, cd, co, ed: pyg_conv.EdgeConv(nn.Linear(cs + cd, co)), False, "same_c_nosize"),
        ("FeaStConv", lambda cs, cd, co, ed: pyg_conv.FeaStConv(cs, co, heads=2, add_self_loops=False), False, "same_c_nosize"),
        ("LEConv", lambda cs, cd, co, ed: pyg_conv.LEConv((cs, cd), co), False, "weight_nosize"),
        ("GENConv", lambda cs, cd, co, ed: pyg_conv.GENConv((cs, cd), co, aggr=rng.choice(["softmax", "add", "mean"]), edge_dim=ed), True, "same_c_edge_size"),
        ("GeneralConv", lambda cs, cd, co, ed: pyg_conv.GeneralConv((cs, cd), co, in_edge_channels=ed, attention=rng.choice([False, True])), True, "edge_size"),
        ("SimpleConv", lambda cs, cd, co, ed: pyg_conv.SimpleConv(aggr="mean", combine_root="sum"), True, "same_c_size"),
        ("WLConvContinuous", lambda cs, cd, co, ed: pyg_conv.WLConvContinuous(), True, "same_c_weight_size"),
    ]

    topos = ["rectangular_lt", "rectangular_gt", "trailing_isolated_dst", "empty_edges"]

    for op_name, builder, supports_none_dst, call_kind in bipartite_specs:
        for t_idx in range(num_trials_per_op):
            topo = topos[t_idx % len(topos)]
            ctx = Z3GraphContext(rng)
            cs_sym = ctx.new_int("cs", 4, 12)
            cd_sym = ctx.new_int("cd", 4, 12)
            co_sym = ctx.new_int("co", 4, 16)
            hk = ctx.new_int("hk", 2, 8)
            ctx.require(co_sym == hk * 2)
            if "same_c" in call_kind or call_kind == "cg_nosize":
                ctx.require(cs_sym == cd_sym, co_sym == cs_sym)
            else:
                ctx.require(cs_sym != cd_sym)
            model_z3 = ctx.solve([(cs_sym, 6), (cd_sym, 10), (co_sym, 8)])
            cs = ctx.eval_int(model_z3, cs_sym)
            cd = ctx.eval_int(model_z3, cd_sym)
            co = ctx.eval_int(model_z3, co_sym)
            ed = 4

            if topo == "rectangular_lt":
                n_src, n_dst, n_e = 4, 9, 14
            elif topo == "rectangular_gt":
                n_src, n_dst, n_e = 9, 4, 14
            elif topo == "trailing_isolated_dst":
                n_src, n_dst, n_e = 5, 8, 10
            else:
                n_src, n_dst, n_e = 4, 7, 0

            gen = torch.Generator().manual_seed(300 + t_idx)
            x_s = torch.randn(n_src, cs, generator=gen, requires_grad=True)
            x_d = torch.randn(n_dst, cd, generator=gen, requires_grad=True)
            if n_e == 0:
                ei = torch.empty((2, 0), dtype=torch.long)
            elif topo == "trailing_isolated_dst":
                ei = torch.stack([
                    torch.randint(0, n_src, (n_e,), generator=gen),
                    torch.randint(0, n_dst - 3, (n_e,), generator=gen),
                ])
            else:
                ei = torch.stack([
                    torch.randint(0, n_src, (n_e,), generator=gen),
                    torch.randint(0, n_dst, (n_e,), generator=gen),
                ])
            ew = torch.rand(n_e, generator=gen) + 0.1
            ea = torch.randn(n_e, ed, generator=gen)
            et = torch.randint(0, 3, (n_e,), generator=gen) if n_e > 0 else torch.empty((0,), dtype=torch.long)

            def invoke(mod: nn.Module, xs: torch.Tensor, xd: torch.Tensor | None, e_idx: torch.Tensor) -> torch.Tensor:
                pair = (xs, xd)
                sz = (xs.size(0), n_dst) if xd is None else None
                if "size" in call_kind and "nosize" not in call_kind:
                    if "weight" in call_kind:
                        return mod(pair, e_idx, ew, size=sz)
                    if "edge" in call_kind:
                        return mod(pair, e_idx, ea, size=sz)
                    return mod(pair, e_idx, size=sz)
                else:
                    if "rel" in call_kind:
                        return mod(pair, e_idx, et)
                    if "weight" in call_kind:
                        return mod(pair, e_idx, ew)
                    if "edge" in call_kind or call_kind == "cg_nosize":
                        return mod(pair, e_idx, ea)
                    return mod(pair, e_idx)

            try:
                torch.manual_seed(t_idx)
                m = builder(cs, cd, co, ed).eval()
                out = invoke(m, x_s, x_d, ei)
                out.sum().backward()

                if out.shape != (n_dst, co):
                    _rec(
                        findings,
                        "bipartite_shape",
                        "shape_mismatch",
                        f"{op_name}[{topo}]",
                        f"Expected shape {(n_dst, co)}, got {tuple(out.shape)} on bipartite ({n_src}->{n_dst})",
                    )
                    continue
                if not torch.isfinite(out).all():
                    _rec(findings, "bipartite_finite", "nan_inf", f"{op_name}[{topo}]", "Non-finite bipartite output")
                    continue

                # Bipartite Independent Src/Dst Node Permutation Equivariance!
                ps = torch.randperm(n_src, generator=gen)
                pd = torch.randperm(n_dst, generator=gen)
                inv_ps = torch.argsort(ps)
                inv_pd = torch.argsort(pd)
                ei_p = torch.stack([inv_ps[ei[0]], inv_pd[ei[1]]]) if n_e > 0 else ei.clone()
                out_p = invoke(m, x_s.detach()[ps], x_d.detach()[pd], ei_p)
                if not _is_close(out[pd], out_p, atol=2e-3, rtol=5e-3):
                    diff = _max_abs_diff(out[pd], out_p)
                    if diff > 1e-2:
                        _rec(
                            findings,
                            "bipartite_equivariance",
                            "numerical_divergence",
                            f"{op_name}[{topo}]",
                            f"Bipartite (P_src, P_dst) equivariance violated! max_diff={diff:.4e}",
                            max_diff=diff,
                        )

                # Test `(x_src, None)` with explicit `size=(n_src, n_dst)` when supported!
                if supports_none_dst and op_name not in ("GATConv", "GENConv"):
                    # Only when root_weight doesn't require x_dst or handles None
                    out_none = invoke(m, x_s.detach(), None, ei)
                    if out_none.size(0) != n_dst or not torch.isfinite(out_none).all():
                        _rec(
                            findings,
                            "bipartite_none_dst",
                            "shape_mismatch" if out_none.size(0) != n_dst else "nan_inf",
                            f"{op_name}[{topo}]",
                            f"(x_src, None) returned {tuple(out_none.shape)}, expected {n_dst} rows",
                        )
            except Exception as exc:
                _rec(
                    findings,
                    "bipartite_fwd_bwd",
                    "crash",
                    f"{op_name}[{topo}]",
                    f"{type(exc).__name__}: {exc}",
                    tb=traceback.format_exc(),
                )

    return findings


def fuzz_flow_duality() -> List[BugFinding]:
    """Mathematical Duality Oracle: `conv(flow='source_to_target')(x, ei) == conv(flow='target_to_source')(x, ei.flip(0))`."""
    findings: List[BugFinding] = []
    N, C = 8, 8
    gen = torch.Generator().manual_seed(77)
    x = torch.randn(N, C, generator=gen)
    x_0 = torch.randn(N, C, generator=gen)
    pos = torch.randn(N, 3, generator=gen)
    # Asymmetric directed graph with trailing isolated node 7!
    ei = torch.tensor([[0, 0, 1, 2, 3, 5, 2], [1, 3, 2, 4, 0, 6, 6]], dtype=torch.long)
    ei_rev = ei.flip(0).contiguous()
    ew = torch.rand(ei.size(1), generator=gen) + 0.2
    ea = torch.randn(ei.size(1), 4, generator=gen)
    et = torch.randint(0, 3, (ei.size(1),), generator=gen)
    deg = torch.tensor([2, 3, 2, 1])

    flow_specs: list[tuple[str, Callable[[str], nn.Module], Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]]] = [
        ("GCNConv", lambda fl: pyg_conv.GCNConv(C, C, flow=fl), lambda m, x_, e_: m(x_, e_, ew)),
        ("SAGEConv", lambda fl: pyg_conv.SAGEConv(C, C, flow=fl), lambda m, x_, e_: m(x_, e_)),
        ("GraphConv", lambda fl: pyg_conv.GraphConv(C, C, flow=fl), lambda m, x_, e_: m(x_, e_, ew)),
        ("GATConv", lambda fl: pyg_conv.GATConv(C, C // 2, heads=2, flow=fl, edge_dim=4), lambda m, x_, e_: m(x_, e_, ea)),
        ("GATv2Conv", lambda fl: pyg_conv.GATv2Conv(C, C // 2, heads=2, flow=fl, edge_dim=4), lambda m, x_, e_: m(x_, e_, ea)),
        ("TransformerConv", lambda fl: pyg_conv.TransformerConv(C, C // 2, heads=2, flow=fl, edge_dim=4), lambda m, x_, e_: m(x_, e_, ea)),
        ("GINConv", lambda fl: pyg_conv.GINConv(nn.Linear(C, C), flow=fl), lambda m, x_, e_: m(x_, e_)),
        ("GINEConv", lambda fl: pyg_conv.GINEConv(nn.Linear(C, C), edge_dim=4, flow=fl), lambda m, x_, e_: m(x_, e_, ea)),
        ("ChebConv", lambda fl: pyg_conv.ChebConv(C, C, K=2, flow=fl), lambda m, x_, e_: m(x_, e_, ew)),
        ("TAGConv", lambda fl: pyg_conv.TAGConv(C, C, K=2, flow=fl), lambda m, x_, e_: m(x_, e_, ew)),
        ("SGConv", lambda fl: pyg_conv.SGConv(C, C, K=2, flow=fl), lambda m, x_, e_: m(x_, e_, ew)),
        ("SSGConv", lambda fl: pyg_conv.SSGConv(C, C, alpha=0.2, K=2, flow=fl), lambda m, x_, e_: m(x_, e_, ew)),
        ("APPNP", lambda fl: pyg_conv.APPNP(K=2, alpha=0.2, flow=fl), lambda m, x_, e_: m(x_, e_, ew)),
        ("ARMAConv", lambda fl: pyg_conv.ARMAConv(C, C, flow=fl), lambda m, x_, e_: m(x_, e_, ew)),
        ("AGNNConv", lambda fl: pyg_conv.AGNNConv(flow=fl), lambda m, x_, e_: m(x_, e_)),
        ("ResGatedGraphConv", lambda fl: pyg_conv.ResGatedGraphConv(C, C, edge_dim=4, flow=fl), lambda m, x_, e_: m(x_, e_, ea)),
        ("GatedGraphConv", lambda fl: pyg_conv.GatedGraphConv(C, num_layers=2, flow=fl), lambda m, x_, e_: m(x_, e_, ew)),
        ("MFConv", lambda fl: pyg_conv.MFConv(C, C, flow=fl), lambda m, x_, e_: m(x_, e_)),
        ("RGCNConv", lambda fl: pyg_conv.RGCNConv(C, C, num_relations=3, flow=fl), lambda m, x_, e_: m(x_, e_, et)),
        ("FastRGCNConv", lambda fl: pyg_conv.FastRGCNConv(C, C, num_relations=3, flow=fl), lambda m, x_, e_: m(x_, e_, et)),
        ("RGATConv", lambda fl: pyg_conv.RGATConv(C, C // 2, num_relations=3, heads=2, flow=fl), lambda m, x_, e_: m(x_, e_, et)),
        ("FiLMConv", lambda fl: pyg_conv.FiLMConv(C, C, num_relations=3, flow=fl), lambda m, x_, e_: m(x_, e_, et)),
        ("GMMConv", lambda fl: pyg_conv.GMMConv(C, C, dim=4, kernel_size=2, flow=fl), lambda m, x_, e_: m(x_, e_, ea)),
        ("NNConv", lambda fl: pyg_conv.NNConv(C, C, nn=nn.Linear(4, C * C), flow=fl), lambda m, x_, e_: m(x_, e_, ea)),
        ("CGConv", lambda fl: pyg_conv.CGConv(C, dim=4, flow=fl), lambda m, x_, e_: m(x_, e_, ea)),
        ("EdgeConv", lambda fl: pyg_conv.EdgeConv(nn.Linear(2 * C, C), flow=fl), lambda m, x_, e_: m(x_, e_)),
        ("FeaStConv", lambda fl: pyg_conv.FeaStConv(C, C, heads=2, flow=fl), lambda m, x_, e_: m(x_, e_)),
        ("PointNetConv", lambda fl: pyg_conv.PointNetConv(nn.Linear(C + 3, C), nn.Linear(C, C), flow=fl), lambda m, x_, e_: m(x_, pos, e_)),
        ("PointTransformerConv", lambda fl: pyg_conv.PointTransformerConv(C, C, flow=fl), lambda m, x_, e_: m(x_, pos, e_)),
        ("LEConv", lambda fl: pyg_conv.LEConv(C, C, flow=fl), lambda m, x_, e_: m(x_, e_, ew)),
        ("ClusterGCNConv", lambda fl: pyg_conv.ClusterGCNConv(C, C, diag_lambda=0.5, flow=fl), lambda m, x_, e_: m(x_, e_)),
        ("GENConv", lambda fl: pyg_conv.GENConv(C, C, edge_dim=4, flow=fl), lambda m, x_, e_: m(x_, e_, ea)),
        ("PNAConv", lambda fl: pyg_conv.PNAConv(C, C, ["mean", "max"], ["identity", "amplification"], deg=deg, flow=fl), lambda m, x_, e_: m(x_, e_)),
        ("EGConv", lambda fl: pyg_conv.EGConv(C, C, num_heads=2, num_bases=2, flow=fl), lambda m, x_, e_: m(x_, e_)),
        ("PDNConv", lambda fl: pyg_conv.PDNConv(C, C, edge_dim=4, hidden_channels=8, flow=fl), lambda m, x_, e_: m(x_, e_, ea)),
        ("GeneralConv", lambda fl: pyg_conv.GeneralConv(C, C, in_edge_channels=4, attention=True, flow=fl), lambda m, x_, e_: m(x_, e_, ea)),
        ("SimpleConv", lambda fl: pyg_conv.SimpleConv(aggr="mean", flow=fl), lambda m, x_, e_: m(x_, e_)),
        ("LGConv", lambda fl: pyg_conv.LGConv(flow=fl), lambda m, x_, e_: m(x_, e_, ew)),
        ("MixHopConv", lambda fl: pyg_conv.MixHopConv(C, C // 2, powers=[0, 1], flow=fl), lambda m, x_, e_: m(x_, e_, ew)),
        ("FAConv", lambda fl: pyg_conv.FAConv(C, flow=fl), lambda m, x_, e_: m(x_, x_0, e_)),
        ("GCN2Conv", lambda fl: pyg_conv.GCN2Conv(C, alpha=0.2, theta=0.5, layer=1, flow=fl), lambda m, x_, e_: m(x_, x_0, e_, ew)),
    ]

    for name, factory, caller in flow_specs:
        try:
            torch.manual_seed(11)
            m_s2t = factory("source_to_target").eval()
            m_t2s = factory("target_to_source").eval()
            m_t2s.load_state_dict(m_s2t.state_dict())

            x1 = x.detach().clone().requires_grad_(True)
            y_s2t = caller(m_s2t, x1, ei)
            y_s2t.sum().backward()

            x2 = x.detach().clone().requires_grad_(True)
            y_t2s = caller(m_t2s, x2, ei_rev)
            y_t2s.sum().backward()

            if not _is_close(y_s2t, y_t2s, atol=2e-3, rtol=5e-3):
                diff = _max_abs_diff(y_s2t, y_t2s)
                if diff > 1e-2:
                    _rec(
                        findings,
                        "flow_duality",
                        "numerical_divergence",
                        name,
                        f"`flow='target_to_source'` on reversed edges diverged from `flow='source_to_target'`! max_diff={diff:.4e}",
                        detail=f"{name} likely hardcodes `edge_index[0]`/`edge_index[1]` in `forward()` instead of respecting `self.flow`.",
                        max_diff=diff,
                    )
            elif x1.grad is not None and x2.grad is not None and not _is_close(x1.grad, x2.grad, atol=5e-3, rtol=1e-2):
                gdiff = _max_abs_diff(x1.grad, x2.grad)
                if gdiff > 2e-2:
                    _rec(
                        findings,
                        "flow_duality_grad",
                        "grad_divergence",
                        name,
                        f"`flow='target_to_source'` x.grad diverged: max_diff={gdiff:.4e}",
                        max_diff=gdiff,
                    )
        except Exception as exc:
            _rec(
                findings,
                "flow_duality",
                "crash",
                name,
                f"`flow='target_to_source'` crashes: {type(exc).__name__}: {exc}",
                tb=traceback.format_exc(),
            )

    return findings


def fuzz_hetero_dags(num_dags: int = 25) -> List[BugFinding]:
    """Synthesize Z3-constrained multi-type Heterogeneous GNNs and check per-type equivariance & MPS parity."""
    findings: List[BugFinding] = []
    rng = random.Random(888)
    node_types = ["paper", "author", "venue"]
    edge_types = [
        ("paper", "cites", "paper"),
        ("author", "writes", "paper"),
        ("paper", "rev_writes", "author"),
        ("paper", "published_in", "venue"),
        ("venue", "hosts", "paper"),
    ]
    metadata = (node_types, edge_types)

    for idx in range(num_dags):
        ctx = Z3GraphContext(rng)
        c_p = ctx.new_int("cp", 4, 16)
        c_a = ctx.new_int("ca", 4, 16)
        c_v = ctx.new_int("cv", 4, 16)
        c_hid = ctx.new_int("ch", 4, 16)
        hk = ctx.new_int("hk", 2, 8)
        ctx.require(c_hid == hk * 2, c_p != c_a)
        z3m = ctx.solve([(c_p, 6), (c_a, 10), (c_v, 8), (c_hid, 8)])
        dims = {"paper": ctx.eval_int(z3m, c_p), "author": ctx.eval_int(z3m, c_a), "venue": ctx.eval_int(z3m, c_v)}
        ch = ctx.eval_int(z3m, c_hid)

        n_counts = {"paper": rng.randint(4, 10), "author": rng.randint(3, 8), "venue": rng.randint(2, 6)}
        gen = torch.Generator().manual_seed(4000 + idx)
        x_dict = {nt: torch.randn(n_counts[nt], dims[nt], generator=gen) for nt in node_types}

        ei_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
        for src_t, rel, dst_t in edge_types:
            if idx % 5 == 0 and rel == "cites":
                # Inject empty relation!
                ei_dict[(src_t, rel, dst_t)] = torch.empty((2, 0), dtype=torch.long)
            else:
                ne = rng.randint(2, 12)
                # Leave last dst node isolated on odd seeds!
                dst_hi = max(1, n_counts[dst_t] - 1) if idx % 2 == 1 else n_counts[dst_t]
                ei_dict[(src_t, rel, dst_t)] = torch.stack([
                    torch.randint(0, n_counts[src_t], (ne,), generator=gen),
                    torch.randint(0, dst_hi, (ne,), generator=gen),
                ])

        arch = ("HeteroConv", "HGTConv", "HANConv")[idx % 3]
        try:
            torch.manual_seed(idx)
            if arch == "HeteroConv":
                convs1 = {}
                for et_key in edge_types:
                    cs, cd = dims[et_key[0]], dims[et_key[2]]
                    choice = rng.choice(["sage", "gatv2", "graph"])
                    if choice == "sage":
                        convs1[et_key] = pyg_conv.SAGEConv((cs, cd), ch)
                    elif choice == "gatv2":
                        convs1[et_key] = pyg_conv.GATv2Conv((cs, cd), ch // 2, heads=2, add_self_loops=False)
                    else:
                        convs1[et_key] = pyg_conv.GraphConv((cs, cd), ch)
                model = pyg_conv.HeteroConv(convs1, aggr=rng.choice(["sum", "mean", "max", "min"])).eval()
            elif arch == "HGTConv":
                model = pyg_conv.HGTConv(dims, ch, metadata=metadata, heads=2).eval()
            else:
                model = pyg_conv.HANConv(dims, ch, metadata=metadata, heads=2, dropout=0.0).eval()

            xd_ref = {k: v.clone().requires_grad_(True) for k, v in x_dict.items()}
            out_ref = model(xd_ref, ei_dict)
            sum(v.sum() for v in out_ref.values() if v is not None).backward()

            # Check per-node-type independent permutation equivariance!
            perms = {nt: torch.randperm(n_counts[nt], generator=gen) for nt in node_types}
            inv_p = {nt: torch.argsort(perms[nt]) for nt in node_types}
            xd_p = {nt: x_dict[nt][perms[nt]] for nt in node_types}
            ei_p = {
                (s, r, d): (torch.stack([inv_p[s][e[0]], inv_p[d][e[1]]]) if e.numel() > 0 else e.clone())
                for (s, r, d), e in ei_dict.items()
            }
            out_p = model(xd_p, ei_p)
            for nt in node_types:
                if out_ref[nt] is not None and out_p[nt] is not None:
                    if not torch.isfinite(out_ref[nt]).all():
                        _rec(findings, "hetero_finite", "nan_inf", f"{arch}[{nt}]", f"Non-finite output for {nt}")
                    elif not _is_close(out_ref[nt][perms[nt]], out_p[nt], atol=2e-3, rtol=5e-3):
                        diff = _max_abs_diff(out_ref[nt][perms[nt]], out_p[nt])
                        if diff > 1e-2:
                            _rec(
                                findings,
                                "hetero_equivariance",
                                "numerical_divergence",
                                f"{arch}[{nt}]",
                                f"Hetero per-type equivariance violated on `{nt}`: max_diff={diff:.4e}",
                                max_diff=diff,
                            )
        except Exception as exc:
            _rec(
                findings,
                "hetero_dag",
                "crash",
                arch,
                f"{type(exc).__name__}: {exc}",
                tb=traceback.format_exc(),
            )

    return findings
