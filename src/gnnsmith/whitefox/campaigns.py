"""WhiteFox white-box targeted fuzzing campaigns derived from PyG source branch analysis."""

from __future__ import annotations

import traceback
from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import EdgeIndex
from torch_geometric.data import Batch, Data
import torch_geometric.nn as pyg_nn
import torch_geometric.nn.aggr as pyg_aggr
import torch_geometric.nn.pool as pyg_pool
from torch_geometric.nn import SAGEConv, GATv2Conv, GraphConv, to_hetero, to_hetero_with_bases
import torch_geometric.utils as pyg_utils

from gnnsmith.oracles import BugFinding


def _record(
    findings: List[BugFinding],
    oracle: str,
    category: str,
    op_name: str,
    summary: str,
    detail: str,
    tb: str = "",
    max_diff: float | None = None,
) -> None:
    sig = f"wf|{oracle}|{category}|{op_name}|{summary[:100]}"
    findings.append(
        BugFinding(
            oracle=oracle,
            category=category,
            summary=summary,
            detail=detail,
            op_types=[op_name],
            topology_kind="whitefox_targeted",
            seed=0,
            max_diff=max_diff,
            traceback_str=tb,
            signature_key=sig,
        )
    )


def campaign_edge_index_whitebox() -> List[BugFinding]:
    """Campaign W1: Target `torch_geometric/edge_index.py` metadata, slicing, arithmetic, and SpMM branches."""
    findings: List[BugFinding] = []

    # W1.1: Negative-step slicing along dim=1 (`edge_index[:, ::-1]` / `aten.slice.Tensor(..., step=-1)`)
    # Source branch: edge_index.py:_slice lines 1495-1504
    try:
        raw = torch.tensor([[0, 0, 1, 2], [3, 1, 2, 0]], dtype=torch.long)
        ei = EdgeIndex(raw, sparse_size=(4, 4), sort_order="row")
        sliced = torch.ops.aten.slice.Tensor(ei, 1, None, None, -1)
        if isinstance(sliced, EdgeIndex) and sliced.is_sorted:
            # Verify whether claimed sort_order actually matches tensor data!
            if sliced.is_sorted_by_col:
                cols = sliced[1]
                actually_sorted_col = bool((cols[1:] >= cols[:-1]).all().item())
                if not actually_sorted_col:
                    _record(
                        findings,
                        "wf_edge_index_metadata",
                        "corrupt_metadata",
                        "EdgeIndex.slice(step<0)",
                        f"aten.slice(dim=1, step=-1) on row-sorted EdgeIndex falsely sets sort_order='col' on unsorted data {sliced.tolist()}",
                        "edge_index.py:_slice swaps ROW<->COL when step<0 along dim=1 (confusing reversing edge order with flipping rows/cols).",
                    )
            if sliced.is_sorted_by_row:
                rows = sliced[0]
                actually_sorted_row = bool((rows[1:] >= rows[:-1]).all().item())
                if not actually_sorted_row:
                    _record(
                        findings,
                        "wf_edge_index_metadata",
                        "corrupt_metadata",
                        "EdgeIndex.slice(step<0)",
                        f"aten.slice(dim=1, step=-1) falsely sets sort_order='row' on unsorted data {sliced.tolist()}",
                        "edge_index.py:_slice metadata corruption.",
                    )
    except Exception as exc:
        _record(findings, "wf_edge_index_slice", "crash", "EdgeIndex.slice", f"{type(exc).__name__}: {exc}", "slice step=-1", traceback.format_exc())

    # W1.2: Scalar / Tensor left-hand subtraction (`10 - edge_index` / `torch.sub(scalar_tensor, edge_index)`)
    # Source branch: edge_index.py:_sub assumes `input` is always `EdgeIndex`!
    try:
        ei = EdgeIndex(torch.tensor([[0, 1], [1, 2]]), sparse_size=(4, 4))
        _ = 3 - ei
    except Exception as exc:
        _record(
            findings,
            "wf_edge_index_arith",
            "crash",
            "EdgeIndex.__rsub__",
            f"`int - EdgeIndex` crashes in aten.sub.Tensor handler: {type(exc).__name__}: {exc}",
            "edge_index.py:_sub accesses `input._data` unconditionally without checking `isinstance(input, EdgeIndex)` when EdgeIndex is the RHS operand.",
            traceback.format_exc(),
        )

    # W1.3: `EdgeIndex` subtraction or negative-alpha addition invalidating `sort_order` / `sparse_size`
    # Source branch: edge_index.py:_add / _sub with negative alpha or non-monotonic tensor offset
    try:
        ei = EdgeIndex(torch.tensor([[1, 2], [2, 1]]), sparse_size=(4, 4), sort_order="row", is_undirected=True)
        out = torch.add(ei, 5, alpha=-1)  # produces [[-4, -3], [-3, -4]] or similar, wait: ei + (-5)!
        if isinstance(out, EdgeIndex) and (out < 0).any():
            _record(
                findings,
                "wf_edge_index_arith",
                "corrupt_metadata",
                "EdgeIndex.add(alpha=-1)",
                f"EdgeIndex.add with negative offset produces negative indices {out.tolist()} with sparse_size={out.sparse_size()} while remaining an EdgeIndex",
                "edge_index.py:_add preserves EdgeIndex wrapper and sort_order even when result contains negative node indices and negative sparse_size.",
            )
    except Exception as exc:
        _record(findings, "wf_edge_index_arith", "crash", "EdgeIndex.add(alpha<0)", f"{type(exc).__name__}: {exc}", "negative alpha add", traceback.format_exc())

    # W1.4: Rectangular Bipartite `EdgeIndex.matmul(..., transpose=True)` Backward Pass!
    # Source branch: edge_index.py:_TorchSPMM.backward lines 1877-1893 (`size=input.get_sparse_size()[::-1]`)
    try:
        ei_bip = EdgeIndex(
            torch.tensor([[0, 2, 3, 1], [0, 1, 3, 5]], dtype=torch.long),
            sparse_size=(4, 7),
        ).sort_by("col").values
        x_src = torch.randn(4, 8, requires_grad=True)
        out = ei_bip.matmul(x_src, transpose=True)  # [7, 8] = A^T @ x_src
        out.sum().backward()
        # Compare against dense ground truth
        adj_dense = torch.zeros(4, 7)
        adj_dense[ei_bip[0], ei_bip[1]] = 1.0
        x_ref = x_src.detach().clone().requires_grad_(True)
        (adj_dense.t() @ x_ref).sum().backward()
        diff = float((x_src.grad - x_ref.grad).abs().max().item())
        if diff > 1e-4:
            _record(
                findings,
                "wf_edge_index_spmm_bwd",
                "grad_divergence",
                "EdgeIndex.matmul(transpose=True)",
                f"Bipartite EdgeIndex.matmul(transpose=True) backward wrong gradient: max_diff={diff:.4e}",
                "Bipartite (4, 7) col-sorted SpMM backward.",
                max_diff=diff,
            )
    except Exception as exc:
        _record(
            findings,
            "wf_edge_index_spmm_bwd",
            "crash",
            "EdgeIndex.matmul(transpose=True).backward",
            f"Bipartite EdgeIndex.matmul(..., transpose=True).backward() crashes: {type(exc).__name__}: {exc}",
            "edge_index.py:_TorchSPMM.backward builds CSR tensor with crow_indices of length M+1 but passes reversed size=(N, M) = input.get_sparse_size()[::-1]!",
            traceback.format_exc(),
        )

    # W1.5: `EdgeIndex.matmul(..., reduce='max'|'min')` with `requires_grad=False` vs `requires_grad=True` when isolated rows exist!
    # Source branch: edge_index.py:_spmm lines 1955-1960 (`_TorchSPMM` uses `torch.sparse.mm` when `not requires_grad`, `_scatter_spmm` when `requires_grad`!)
    for red in ("max", "min"):
        try:
            # Row 2 has NO edges; all features are strictly negative (for max) or positive (for min)
            ei = EdgeIndex(torch.tensor([[0, 0, 1], [1, 2, 0]], dtype=torch.long), sparse_size=(4, 4), sort_order="row")
            x_neg = torch.full((4, 3), -5.0 if red == "max" else 5.0)
            out_nograd = ei.matmul(x_neg, reduce=red)
            x_grad = x_neg.detach().clone().requires_grad_(True)
            out_grad = ei.matmul(x_grad, reduce=red)
            diff = float((out_nograd - out_grad).abs().max().item())
            if not torch.allclose(out_nograd, out_grad, atol=1e-5) or not torch.isfinite(out_nograd).all():
                _record(
                    findings,
                    "wf_edge_index_spmm_grad_parity",
                    "numerical_divergence",
                    f"EdgeIndex.matmul(reduce='{red}')",
                    f"EdgeIndex.matmul(reduce='{red}') changes forward output depending on x.requires_grad! nograd={out_nograd[:,0].tolist()} vs grad={out_grad[:,0].detach().tolist()}",
                    "edge_index.py:_spmm dispatches `requires_grad=False` to `_TorchSPMM` (`torch.sparse.mm`) and `requires_grad=True` to `_scatter_spmm`, which disagree on empty rows / negative values.",
                    max_diff=diff,
                )
        except Exception as exc:
            _record(findings, "wf_edge_index_spmm_grad_parity", "crash", f"EdgeIndex.matmul(reduce='{red}')", f"{type(exc).__name__}: {exc}", "spmm max/min parity", traceback.format_exc())

    # W1.6: `EdgeIndex.matmul(..., reduce='mean', transpose=True)`!
    # Source branch: edge_index.py:_spmm lines 1951-1953 (`count = input.get_indptr().diff()`)
    try:
        ei = EdgeIndex(
            torch.tensor([[0, 0, 0, 2], [0, 1, 2, 2]], dtype=torch.long),
            sparse_size=(4, 4),
        ).sort_by("col").values
        x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        out = ei.matmul(x, reduce="mean", transpose=True)
        adj = torch.zeros(4, 4)
        adj[ei[0], ei[1]] = 1.0
        deg_col = adj.t().sum(dim=1, keepdim=True).clamp(min=1.0)
        ref = (adj.t() @ x) / deg_col
        diff = float((out - ref).abs().max().item())
        if diff > 1e-4:
            _record(
                findings,
                "wf_edge_index_spmm_mean_T",
                "numerical_divergence",
                "EdgeIndex.matmul(reduce='mean', transpose=True)",
                f"EdgeIndex.matmul(reduce='mean', transpose=True) wrong result: max_diff={diff:.4e}",
                "Check get_indptr() under col sort order.",
                max_diff=diff,
            )
    except Exception as exc:
        _record(findings, "wf_edge_index_spmm_mean_T", "crash", "EdgeIndex.matmul(mean, T)", f"{type(exc).__name__}: {exc}", "mean transpose", traceback.format_exc())

    # W1.7: `Index` / `EdgeIndex` RHS in-place `add_` / `sub_` crashes `sort_edge_index`, `coalesce`, and `is_undirected`!
    for fn_name, fn in [
        ("utils.sort_edge_index(EdgeIndex)", lambda e: pyg_utils.sort_edge_index(e)),
        ("utils.coalesce(EdgeIndex)", lambda e: pyg_utils.coalesce(e)),
        ("utils.is_undirected(EdgeIndex)", lambda e: pyg_utils.is_undirected(e)),
    ]:
        try:
            ei = EdgeIndex(torch.tensor([[0, 1], [1, 0]]), sparse_size=(3, 3))
            _ = fn(ei)
        except Exception as exc:
            _record(
                findings,
                "wf_index_rhs_inplace",
                "crash",
                fn_name,
                f"{fn_name} crashes in index.py:add_: {type(exc).__name__}: {exc}",
                "index.py:add_ assumes LHS `input` is an `Index` and accesses `input.dim_size` when `plain_tensor += Index` executes.",
                traceback.format_exc(),
            )

    # W1.8: `torch.compile(GCNConv, dynamic=True)` with `EdgeIndex` crashes TorchInductor SymInt proxy tracking!
    try:
        gcn_c = torch.compile(pyg_nn.GCNConv(8, 8).eval(), dynamic=True)
        ei = EdgeIndex(torch.tensor([[0, 1, 2], [1, 2, 0]]), sparse_size=(4, 4), sort_order="row")
        _ = gcn_c(torch.randn(4, 8), ei)
    except Exception as exc:
        _record(
            findings,
            "wf_compile_edge_index",
            "crash",
            "torch.compile(GCNConv, EdgeIndex)",
            f"torch.compile(GCNConv, dynamic=True)(x, EdgeIndex) crashes Inductor: {type(exc).__name__}: {str(exc)[:140]}",
            "SymInt not tracked with proxy when compiling GCNConv with EdgeIndex tensor subclass.",
            traceback.format_exc(),
        )

    return findings


def campaign_utils_whitebox() -> List[BugFinding]:
    """Campaign W2: Target `torch_geometric/utils/` (`softmax`, `scatter`, `coalesce`, `subgraph`, `unbatch`, `to_dense_batch`)."""
    findings: List[BugFinding] = []

    # W2.1: `softmax` `index` vs `ptr` parity across `dim=0, 1, -1` and empty groups in `ptr`!
    # Source branch: _softmax.py lines 59-79 (`ptr = ptr.view(size)` + `segment`)
    for dim in (0, 1, -1):
        try:
            if dim == 0:
                src = torch.randn(6, 4, requires_grad=True)
            else:
                src = torch.randn(4, 6, requires_grad=True)
            index = torch.tensor([0, 0, 2, 2, 2, 4], dtype=torch.long)  # Note: groups 1 and 3 are EMPTY!
            ptr = torch.tensor([0, 2, 2, 5, 5, 6], dtype=torch.long)

            s_idx = pyg_utils.softmax(src, index=index, num_nodes=5, dim=dim)
            s_ptr = pyg_utils.softmax(src, ptr=ptr, num_nodes=5, dim=dim)
            diff = float((s_idx - s_ptr).abs().max().item())
            if s_idx.shape != s_ptr.shape or diff > 1e-4 or not torch.isfinite(s_ptr).all():
                _record(
                    findings,
                    "wf_utils_softmax",
                    "numerical_divergence",
                    f"utils.softmax(dim={dim})",
                    f"softmax(index=...) vs softmax(ptr=...) mismatch at dim={dim}: shape {tuple(s_idx.shape)} vs {tuple(s_ptr.shape)}, max_diff={diff:.4e}",
                    "Compare index vs CSR ptr path with empty segments.",
                    max_diff=diff,
                )
        except Exception as exc:
            _record(
                findings,
                "wf_utils_softmax",
                "crash",
                f"utils.softmax(ptr, dim={dim})",
                f"utils.softmax(src, ptr=ptr, dim={dim}) crashes while index path succeeds: {type(exc).__name__}: {exc}",
                f"In _softmax.py, `softmax(src, index=..., dim={dim})` works for non-zero dim, whereas `softmax(src, ptr=..., dim={dim})` fails.",
                traceback.format_exc(),
            )

    # W2.2: `softmax` in float16 / bfloat16 with moderate logits (>12.0, where exp(12) > 65504 overflows fp16 if max-subtraction fails or sum overflows!)
    for dt_name, dt in [("float16", torch.float16), ("bfloat16", torch.bfloat16)]:
        try:
            # 200 elements in group 0 all equal to 0.0 -> exp(0.0) = 1.0, sum = 200; now try 70000 elements equal to 0.0 in fp16!
            # Wait: max(fp16) = 65504! If a hub node has >65504 incoming edges with equal attention score, sum(exp(0)) = 70000 -> Inf in fp16 -> 1/Inf = 0.0 (all-zero softmax)!
            src = torch.zeros(70_000, 2, dtype=dt)
            index = torch.zeros(70_000, dtype=torch.long)
            out = pyg_utils.softmax(src, index=index, num_nodes=1)
            col_sums = out.float().sum(dim=0)
            if not torch.isfinite(out).all() or (col_sums - 1.0).abs().max().item() > 0.1:
                _record(
                    findings,
                    "wf_utils_softmax_lowprec",
                    "numerical_divergence",
                    f"utils.softmax({dt_name})",
                    f"utils.softmax in {dt_name} overflows denominator on high-degree hub (70k edges), yielding sum={col_sums.tolist()} instead of 1.0",
                    "_softmax.py accumulates `scatter(out, index, reduce='sum')` in fp16 without upcasting to fp32; degrees > 65504 overflow fp16 max (65504) to Inf, collapsing all attention weights to 0.",
                    max_diff=float((col_sums - 1.0).abs().max().item()),
                )
        except Exception as exc:
            _record(findings, "wf_utils_softmax_lowprec", "crash", f"utils.softmax({dt_name})", f"{type(exc).__name__}: {exc}", "lowprec softmax", traceback.format_exc())

    # W2.3: `scatter` with `reduce='min'|'max'` on all-negative (for max) or all-positive (for min) values WHEN `dim_size > index.max() + 1` (unseen buckets!)
    # Source branch: _scatter.py line 106 (`src.new_zeros(size).scatter_reduce_(..., include_self=False)`) vs `mul` (`new_ones(..., include_self=True)`)
    # Also test `scatter` backward on MPS vs CPU for `reduce in ('sum', 'mean', 'max', 'min', 'mul', 'any')`!
    if torch.backends.mps.is_available():
        for red in ("sum", "mean", "max", "min", "mul"):
            try:
                src_cpu = (torch.randn(12, 4) - 2.0).requires_grad_(True)
                idx_cpu = torch.tensor([0, 0, 1, 3, 3, 3, 0, 1, 3, 1, 0, 3], dtype=torch.long)
                out_cpu = pyg_utils.scatter(src_cpu, idx_cpu, dim=0, dim_size=5, reduce=red)
                (out_cpu * torch.arange(1, 21, dtype=torch.float32).view(5, 4)).sum().backward()

                src_mps = src_cpu.detach().to("mps").requires_grad_(True)
                idx_mps = idx_cpu.to("mps")
                out_mps = pyg_utils.scatter(src_mps, idx_mps, dim=0, dim_size=5, reduce=red)
                (out_mps * torch.arange(1, 21, dtype=torch.float32, device="mps").view(5, 4)).sum().backward()

                f_diff = float((out_cpu - out_mps.cpu()).abs().max().item())
                g_diff = float((src_cpu.grad - src_mps.grad.cpu()).abs().max().item())
                if not torch.isfinite(out_mps).all() or not torch.isfinite(src_mps.grad).all():
                    _record(
                        findings,
                        "wf_utils_scatter_mps",
                        "nan_inf",
                        f"utils.scatter(reduce='{red}')",
                        f"scatter(reduce='{red}') on MPS produced NaN/Inf (out finite={torch.isfinite(out_mps).all().item()}, grad finite={torch.isfinite(src_mps.grad).all().item()})",
                        "CPU vs MPS scatter forward/backward.",
                    )
                elif f_diff > 1e-3 or g_diff > 1e-3:
                    _record(
                        findings,
                        "wf_utils_scatter_mps",
                        "grad_divergence" if g_diff > f_diff else "numerical_divergence",
                        f"utils.scatter(reduce='{red}')",
                        f"scatter(reduce='{red}') CPU vs MPS mismatch: fwd_diff={f_diff:.4e}, grad_diff={g_diff:.4e}",
                        "CPU vs MPS scatter parity.",
                        max_diff=max(f_diff, g_diff),
                    )
            except Exception as exc:
                _record(
                    findings,
                    "wf_utils_scatter_mps",
                    "crash",
                    f"utils.scatter(reduce='{red}')",
                    f"scatter(reduce='{red}') on MPS crashes: {type(exc).__name__}: {exc}",
                    "CPU vs MPS scatter.",
                    traceback.format_exc(),
                )

    # W2.4: `unbatch` / `unbatch_edge_index` when intermediate graph in `batch` is empty (0 nodes)!
    try:
        x = torch.randn(5, 3)
        batch = torch.tensor([0, 0, 2, 2, 2], dtype=torch.long)  # Graph 1 has 0 nodes!
        ei = torch.tensor([[0, 2, 3], [1, 3, 4]], dtype=torch.long)
        xs = pyg_utils.unbatch(x, batch)
        eis = pyg_utils.unbatch_edge_index(ei, batch)
        if len(xs) != len(eis):
            _record(
                findings,
                "wf_utils_unbatch",
                "shape_mismatch",
                "utils.unbatch_edge_index",
                f"unbatch(x, batch) returned {len(xs)} graphs while unbatch_edge_index(ei, batch) returned {len(eis)} graphs when graph 1 is empty!",
                f"batch={batch.tolist()}: len(unbatch)={len(xs)} vs len(unbatch_edge_index)={len(eis)}.",
            )
    except Exception as exc:
        _record(
            findings,
            "wf_utils_unbatch",
            "crash",
            "utils.unbatch_edge_index",
            f"unbatch / unbatch_edge_index with empty intermediate batch graph crashes: {type(exc).__name__}: {exc}",
            "batch=[0, 0, 2, 2, 2] where graph 1 has 0 nodes.",
            traceback.format_exc(),
        )

    return findings


def campaign_aggregations_whitebox() -> List[BugFinding]:
    """Campaign W3: Fuzz all 25 `torch_geometric.nn.aggr` modules across `index` vs `ptr`, empty groups, trailing groups, and MPS."""
    findings: List[BugFinding] = []
    C = 8

    aggr_builders: list[tuple[str, Callable[[], nn.Module], bool]] = [
        ("SumAggregation", lambda: pyg_aggr.SumAggregation(), True),
        ("MeanAggregation", lambda: pyg_aggr.MeanAggregation(), True),
        ("MaxAggregation", lambda: pyg_aggr.MaxAggregation(), True),
        ("MinAggregation", lambda: pyg_aggr.MinAggregation(), True),
        ("MulAggregation", lambda: pyg_aggr.MulAggregation(), True),
        ("VarAggregation", lambda: pyg_aggr.VarAggregation(), True),
        ("StdAggregation", lambda: pyg_aggr.StdAggregation(), True),
        ("SoftmaxAggregation", lambda: pyg_aggr.SoftmaxAggregation(learn=True), True),
        ("PowerMeanAggregation", lambda: pyg_aggr.PowerMeanAggregation(learn=True), True),
        ("MedianAggregation", lambda: pyg_aggr.MedianAggregation(), True),
        ("QuantileAggregation", lambda: pyg_aggr.QuantileAggregation(q=0.5), True),
        ("QuantileAggregation(list)", lambda: pyg_aggr.QuantileAggregation(q=[0.25, 0.75]), True),
        ("LSTMAggregation", lambda: pyg_aggr.LSTMAggregation(C, C), False),
        ("GRUAggregation", lambda: pyg_aggr.GRUAggregation(C, C), False),
        ("Set2Set", lambda: pyg_aggr.Set2Set(C, processing_steps=2), True),
        ("SortAggregation", lambda: pyg_aggr.SortAggregation(k=3), True),
        ("GraphMultisetTransformer", lambda: pyg_aggr.GraphMultisetTransformer(C, k=2, heads=2), True),
        ("AttentionalAggregation", lambda: pyg_aggr.AttentionalAggregation(gate_nn=nn.Linear(C, 1)), True),
        ("EquilibriumAggregation", lambda: pyg_aggr.EquilibriumAggregation(C, C, num_layers=[C]), True),
        ("MLPAggregation", lambda: pyg_aggr.MLPAggregation(C, C, max_num_elements=6, num_layers=1), False),
        ("DeepSetsAggregation", lambda: pyg_aggr.DeepSetsAggregation(nn.Linear(C, C), nn.Linear(C, C)), True),
        ("SetTransformerAggregation", lambda: pyg_aggr.SetTransformerAggregation(C, num_seed_points=2, heads=2), True),
        ("LCMAggregation", lambda: pyg_aggr.LCMAggregation(C, C), False),
        ("VariancePreservingAggregation", lambda: pyg_aggr.VariancePreservingAggregation(), True),
        ("PatchTransformerAggregation", lambda: pyg_aggr.PatchTransformerAggregation(C, C, patch_size=2, hidden_channels=C, heads=2), False),
    ]

    for name, builder, order_invariant in aggr_builders:
        # Test A: `index` vs `ptr` parity (including an empty middle group & trailing empty group!)
        try:
            torch.manual_seed(0)
            ag = builder().eval()
            x = torch.abs(torch.randn(7, C)) + 0.2  # positive so PowerMean is defined
            x.requires_grad_(True)
            index = torch.tensor([0, 0, 0, 2, 2, 3, 3], dtype=torch.long)  # group 1 & group 4 are empty (dim_size=5)!
            ptr = torch.tensor([0, 3, 3, 5, 7, 7], dtype=torch.long)

            out_idx = ag(x, index=index, dim_size=5)
            out_idx.sum().backward()
            gx_idx = x.grad.detach().clone() if x.grad is not None else None

            x_p = x.detach().clone().requires_grad_(True)
            out_ptr = ag(x_p, ptr=ptr, dim_size=5)
            out_ptr.sum().backward()
            gx_ptr = x_p.grad.detach().clone() if x_p.grad is not None else None

            if not torch.isfinite(out_idx).all():
                _record(findings, "wf_aggr_finite", "nan_inf", name, f"{name}(index=...) produced NaN/Inf on valid input with empty groups!", f"index={index.tolist()}, dim_size=5")
            elif out_idx.shape != out_ptr.shape:
                _record(
                    findings,
                    "wf_aggr_idx_vs_ptr",
                    "shape_mismatch",
                    name,
                    f"{name}: index vs ptr output shape mismatch: {tuple(out_idx.shape)} vs {tuple(out_ptr.shape)}",
                    f"index={index.tolist()}, ptr={ptr.tolist()}, dim_size=5",
                )
            else:
                diff = float((out_idx - out_ptr).abs().max().item())
                if not torch.isfinite(out_ptr).all() or diff > 1e-3:
                    _record(
                        findings,
                        "wf_aggr_idx_vs_ptr",
                        "nan_inf" if not torch.isfinite(out_ptr).all() else "numerical_divergence",
                        name,
                        f"{name}: index vs ptr forward mismatch (max_diff={diff:.4e}, ptr_finite={torch.isfinite(out_ptr).all().item()})",
                        f"index={index.tolist()}, ptr={ptr.tolist()}, dim_size=5",
                        max_diff=diff,
                    )
                elif gx_idx is not None and gx_ptr is not None:
                    if not torch.isfinite(gx_idx).all() or not torch.isfinite(gx_ptr).all():
                        _record(
                            findings,
                            "wf_aggr_grad_finite",
                            "nan_inf",
                            name,
                            f"{name}: backward x.grad is NaN/Inf when empty groups exist (idx_grad_finite={torch.isfinite(gx_idx).all().item()}, ptr_grad_finite={torch.isfinite(gx_ptr).all().item()})",
                            f"index={index.tolist()}, dim_size=5",
                        )
        except NotImplementedError:
            pass
        except Exception as exc:
            _record(
                findings,
                "wf_aggr_idx_vs_ptr",
                "crash",
                name,
                f"{name} crashed on valid grouped input with empty bucket: {type(exc).__name__}: {exc}",
                "index=[0,0,0,2,2,3,3], ptr=[0,3,3,5,7,7], dim_size=5",
                traceback.format_exc(),
            )

        # Test B: Empty input tensor (`N == 0`, `dim_size = 2`)!
        try:
            torch.manual_seed(0)
            ag = builder().eval()
            x_empty = torch.empty((0, C), dtype=torch.float32, requires_grad=True)
            idx_empty = torch.empty((0,), dtype=torch.long)
            out_e = ag(x_empty, index=idx_empty, dim_size=2)
            if out_e.size(0) != 2 or not torch.isfinite(out_e).all():
                _record(
                    findings,
                    "wf_aggr_empty_input",
                    "shape_mismatch" if out_e.size(0) != 2 else "nan_inf",
                    name,
                    f"{name} on empty input (N=0, dim_size=2) returned shape {tuple(out_e.shape)}, finite={torch.isfinite(out_e).all().item()}",
                    "Empty edge/node input with dim_size=2.",
                )
        except Exception as exc:
            _record(
                findings,
                "wf_aggr_empty_input",
                "crash",
                name,
                f"{name} crashes on empty input (N=0, dim_size=2): {type(exc).__name__}: {exc}",
                "x=empty(0, C), index=empty(0), dim_size=2",
                traceback.format_exc(),
            )

        # Test C: Unsorted `index` permutation invariance (for order-invariant aggregations!)
        if order_invariant:
            try:
                torch.manual_seed(0)
                ag = builder().eval()
                x_s = torch.abs(torch.randn(8, C)) + 0.2
                idx_s = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], dtype=torch.long)
                perm = torch.tensor([5, 0, 3, 7, 1, 4, 6, 2], dtype=torch.long)
                out_sorted = ag(x_s, index=idx_s, dim_size=3)
                out_unsorted = ag(x_s[perm], index=idx_s[perm], dim_size=3)
                diff = float((out_sorted - out_unsorted).abs().max().item())
                if diff > 2e-3 or not torch.isfinite(out_unsorted).all():
                    _record(
                        findings,
                        "wf_aggr_unsorted_index",
                        "numerical_divergence",
                        name,
                        f"{name} silently produces wrong result when `index` is unsorted! max_diff={diff:.4e}",
                        "Order-invariant aggregation failed when index was permuted from [0,0,0,1,1,2,2,2] to [2,0,1,2,0,1,2,0].",
                        max_diff=diff,
                    )
            except Exception as exc:
                _record(
                    findings,
                    "wf_aggr_unsorted_index",
                    "crash",
                    name,
                    f"{name} crashes when `index` is unsorted: {type(exc).__name__}: {exc}",
                    "Permuted unsorted index.",
                    traceback.format_exc(),
                )

    return findings


def campaign_to_hetero_whitebox() -> List[BugFinding]:
    """Campaign W4: Fuzz `to_hetero` & `to_hetero_with_bases` `torch.fx` AST/IR graph surgery."""
    findings: List[BugFinding] = []

    class SimpleHomoGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = SAGEConv((-1, -1), 16)
            self.conv2 = SAGEConv((-1, -1), 8)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            return self.conv2(x, edge_index)

    # W4.1: Equivalence check! A 1-node-type, 1-edge-type `to_hetero(model, (['n'], [('n','r','n')]), aggr=...)`
    # MUST produce the EXACT same output as the homogeneous `model`!
    for aggr in ("sum", "mean", "max", "min", "mul"):
        try:
            torch.manual_seed(42)
            homo = SimpleHomoGNN().eval()
            x = torch.randn(10, 8)
            ei = torch.randint(0, 10, (2, 24))
            # Force concrete weights first
            ref = homo(x, ei)
            hetero = to_hetero(homo, (["n"], [("n", "r", "n")]), aggr=aggr).eval()
            # Copy initialized weights from homo into hetero
            hetero.conv1["n__r__n"].load_state_dict(homo.conv1.state_dict())
            hetero.conv2["n__r__n"].load_state_dict(homo.conv2.state_dict())
            out_h = hetero({"n": x}, {("n", "r", "n"): ei})["n"]
            diff = float((ref - out_h).abs().max().item())
            if diff > 1e-4:
                _record(
                    findings,
                    "wf_to_hetero_1type_parity",
                    "numerical_divergence",
                    f"to_hetero(aggr='{aggr}')",
                    f"Single-relation to_hetero(aggr='{aggr}') diverged from homogeneous model: max_diff={diff:.4e}",
                    "1-type hetero equivalence check.",
                    max_diff=diff,
                )
        except Exception as exc:
            _record(
                findings,
                "wf_to_hetero_1type_parity",
                "crash",
                f"to_hetero(aggr='{aggr}')",
                f"to_hetero(aggr='{aggr}') crashed: {type(exc).__name__}: {exc}",
                "Single-relation to_hetero.",
                traceback.format_exc(),
            )

    # W4.2: `to_hetero` with 3 incoming edge types per destination across ALL `aggr` modes (`sum`, `mean`, `max`, `min`, `mul`)!
    # Remember our WhiteFox source read of `to_hetero_transformer.py:234-245` (`torch.max`/`torch.min` queue naming & `mean` divisor)!
    metadata_3 = (
        ["paper", "author", "inst"],
        [
            ("paper", "cites", "paper"),
            ("author", "writes", "paper"),
            ("inst", "funds", "paper"),
            ("paper", "written_by", "author"),
            ("author", "affiliated", "inst"),
        ],
    )
    x_dict = {
        "paper": torch.randn(6, 8),
        "author": torch.randn(4, 8),
        "inst": torch.randn(3, 8),
    }
    ei_dict = {
        ("paper", "cites", "paper"): torch.randint(0, 6, (2, 10)),
        ("author", "writes", "paper"): torch.stack([torch.randint(0, 4, (8,)), torch.randint(0, 6, (8,))]),
        ("inst", "funds", "paper"): torch.stack([torch.randint(0, 3, (5,)), torch.randint(0, 6, (5,))]),
        ("paper", "written_by", "author"): torch.stack([torch.randint(0, 6, (8,)), torch.randint(0, 4, (8,))]),
        ("author", "affiliated", "inst"): torch.stack([torch.randint(0, 4, (4,)), torch.randint(0, 3, (4,))]),
    }
    for aggr in ("sum", "mean", "max", "min", "mul"):
        try:
            homo = SimpleHomoGNN().eval()
            hetero = to_hetero(homo, metadata_3, aggr=aggr).eval()
            out = hetero(x_dict, ei_dict)
            # Verify destination aggregation manually for conv2!
            h1 = {}
            # Check output finiteness & shapes
            if out["paper"].shape != (6, 8) or not torch.isfinite(out["paper"]).all():
                _record(
                    findings,
                    "wf_to_hetero_multi_rel",
                    "shape_mismatch" if out["paper"].shape != (6, 8) else "nan_inf",
                    f"to_hetero(aggr='{aggr}', 3-rel)",
                    f"to_hetero(aggr='{aggr}') output invalid: shape={tuple(out['paper'].shape)}",
                    "3-relation destination aggregation in ToHeteroTransformer.",
                )
        except Exception as exc:
            _record(
                findings,
                "wf_to_hetero_multi_rel",
                "crash",
                f"to_hetero(aggr='{aggr}', 3-rel)",
                f"to_hetero(aggr='{aggr}') with 3 incoming relations crashes: {type(exc).__name__}: {exc}",
                "ToHeteroTransformer queue reduction over 3 incoming relations.",
                traceback.format_exc(),
            )

    # W4.3: `to_hetero` when a node type is a pure source (0 incoming edge types — `validate()` only emits a UserWarning instead of raising, then what happens in `forward` / `transform`?!)
    try:
        metadata_src_only = (["author", "paper"], [("author", "writes", "paper")])
        homo = SimpleHomoGNN().eval()
        hetero = to_hetero(homo, metadata_src_only).eval()
        _ = hetero(
            {"author": torch.randn(4, 8), "paper": torch.randn(6, 8)},
            {("author", "writes", "paper"): torch.stack([torch.randint(0, 4, (5,)), torch.randint(0, 6, (5,))])},
        )
    except Exception as exc:
        _record(
            findings,
            "wf_to_hetero_source_only_node",
            "crash",
            "to_hetero(source-only node type)",
            f"to_hetero emits a UserWarning for source-only node types ('author') claiming transform proceeds, then crashes with {type(exc).__name__}: {exc}",
            "to_hetero_transformer.py:validate() warns about unused destination node types instead of handling them or raising a descriptive ValueError, then FX graph construction crashes with internal ValueError/KeyError.",
            traceback.format_exc(),
        )

    # W4.4: `to_hetero` with GlobalPooling (`SumAggregation` / `GlobalPooling` module) across `aggr='sum'|'mean'|'max'`!
    class HomoGraphClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = SAGEConv((-1, -1), 8)
            self.pool = pyg_aggr.SumAggregation()
            self.lin = nn.Linear(8, 3)

        def forward(self, x, edge_index, batch):
            x = self.conv(x, edge_index).relu()
            g = self.pool(x, batch)
            return self.lin(g)

    for aggr in ("sum", "mean", "max"):
        try:
            m = HomoGraphClassifier().eval()
            hm = to_hetero(m, metadata_3, aggr=aggr).eval()
            batch_dict = {
                "paper": torch.tensor([0, 0, 0, 1, 1, 1]),
                "author": torch.tensor([0, 0, 1, 1]),
                "inst": torch.tensor([0, 1, 1]),
            }
            out = hm(x_dict, ei_dict, batch_dict)
            if out.shape != (2, 3):
                _record(
                    findings,
                    "wf_to_hetero_global_pool",
                    "shape_mismatch",
                    f"to_hetero+GlobalPool(aggr='{aggr}')",
                    f"Expected shape (2, 3), got {tuple(out.shape)}",
                    "ToHeteroTransformer.call_global_pooling_module",
                )
        except Exception as exc:
            _record(
                findings,
                "wf_to_hetero_global_pool",
                "crash",
                f"to_hetero+GlobalPool(aggr='{aggr}')",
                f"to_hetero with GlobalPooling (aggr='{aggr}') crashes: {type(exc).__name__}: {exc}",
                "ToHeteroTransformer.call_global_pooling_module queue naming bug!",
                traceback.format_exc(),
            )

    # W4.5: `to_hetero_with_bases` on multi-layer / residual models & bipartite graphs!
    try:
        class HomoBasisGNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GraphConv((-1, -1), 8)
                self.conv2 = GraphConv((8, 8), 8)

            def forward(self, x, edge_index):
                return self.conv2(self.conv1(x, edge_index).relu(), edge_index)

        hb = to_hetero_with_bases(HomoBasisGNN(), metadata_3, num_bases=2, in_channels={"x": 8}).eval()
        x_diff_dims = {
            "paper": torch.randn(6, 8),
            "author": torch.randn(4, 8),
            "inst": torch.randn(3, 8),
        }
        out_b = hb(x_diff_dims, ei_dict)
        if not torch.isfinite(out_b["paper"]).all():
            _record(findings, "wf_to_hetero_bases", "nan_inf", "to_hetero_with_bases", "Non-finite output", "basis hetero")
    except Exception as exc:
        _record(
            findings,
            "wf_to_hetero_bases",
            "crash",
            "to_hetero_with_bases",
            f"to_hetero_with_bases crashes: {type(exc).__name__}: {exc}",
            "ToHeteroWithBasesTransformer",
            traceback.format_exc(),
        )

    return findings


def campaign_pooling_whitebox() -> List[BugFinding]:
    """Campaign W5: Fuzz hierarchical graph pooling (`EdgePooling`, `ClusterPooling`, `ASAPooling`, `TopKPooling`, `SAGPooling`, `MemPooling`)."""
    findings: List[BugFinding] = []
    C = 8

    # Test pooling layers on: (a) empty edges E=0, (b) multi-graph batch where graph 0 has 0 edges, (c) single node N=1, (d) batch invariance!
    pool_factories = [
        ("TopKPooling", lambda: pyg_pool.TopKPooling(C, ratio=0.5), "topk"),
        ("SAGPooling", lambda: pyg_pool.SAGPooling(C, ratio=0.5), "topk"),
        ("ASAPooling", lambda: pyg_pool.ASAPooling(C, ratio=0.5), "asa"),
        ("EdgePooling", lambda: pyg_pool.EdgePooling(C, dropout=0.0), "edge"),
        ("ClusterPooling", lambda: pyg_pool.ClusterPooling(C, dropout=0.0), "cluster"),
    ]

    scenarios = [
        ("empty_edges", torch.randn(6, C), torch.empty((2, 0), dtype=torch.long), torch.tensor([0, 0, 0, 1, 1, 1])),
        ("single_node_no_edge", torch.randn(1, C), torch.empty((2, 0), dtype=torch.long), torch.tensor([0])),
        (
            "batch_with_edgeless_first_graph",
            torch.randn(6, C),
            torch.tensor([[3, 4], [4, 5]], dtype=torch.long),  # Graph 0 (nodes 0,1,2) has 0 edges; Graph 1 (nodes 3,4,5) has 2 edges!
            torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
        ),
    ]

    for p_name, factory, kind in pool_factories:
        for sc_name, x_raw, ei, batch in scenarios:
            try:
                torch.manual_seed(0)
                pool = factory().eval()
                x = x_raw.detach().clone().requires_grad_(True)
                if kind == "topk":
                    res = pool(x, ei, batch=batch)
                    x_out, ei_out, _, batch_out = res[0], res[1], res[2], res[3]
                elif kind == "asa":
                    res = pool(x, ei, batch=batch)
                    x_out, ei_out, _, batch_out = res[0], res[1], res[2], res[3]
                else:
                    res = pool(x, ei, batch=batch)
                    x_out, ei_out, batch_out = res[0], res[1], res[2]

                x_out.sum().backward()
                # Check relational validity of pooled graph!
                if ei_out.numel() > 0 and (ei_out.min().item() < 0 or ei_out.max().item() >= x_out.size(0)):
                    _record(
                        findings,
                        f"wf_pool_{sc_name}",
                        "corrupt_metadata",
                        p_name,
                        f"{p_name} ({sc_name}) emitted out-of-bounds pooled edge_index! max={ei_out.max().item()} >= num_pooled_nodes={x_out.size(0)}",
                        f"Scenario {sc_name}: pooled x={tuple(x_out.shape)}, ei={ei_out.tolist()}",
                    )
                elif batch_out.size(0) != x_out.size(0):
                    _record(
                        findings,
                        f"wf_pool_{sc_name}",
                        "shape_mismatch",
                        p_name,
                        f"{p_name} ({sc_name}) batch_out length {batch_out.size(0)} != x_out rows {x_out.size(0)}",
                        f"Scenario {sc_name}",
                    )
                elif not torch.isfinite(x_out).all() or (x.grad is not None and not torch.isfinite(x.grad).all()):
                    _record(
                        findings,
                        f"wf_pool_{sc_name}",
                        "nan_inf",
                        p_name,
                        f"{p_name} ({sc_name}) produced NaN/Inf in output or gradient!",
                        f"Scenario {sc_name}",
                    )
            except Exception as exc:
                _record(
                    findings,
                    f"wf_pool_{sc_name}",
                    "crash",
                    p_name,
                    f"{p_name} crashes on valid `{sc_name}` graph: {type(exc).__name__}: {exc}",
                    f"Scenario {sc_name}: N={x_raw.size(0)}, E={ei.size(1)}, batch={batch.tolist()}",
                    traceback.format_exc(),
                )

    # Also test MemPooling batch invariance & empty mask / skewed sizes
    try:
        torch.manual_seed(0)
        mp = pyg_pool.MemPooling(C, C, heads=2, num_clusters=3).eval()
        x1 = torch.randn(4, C)
        x2 = torch.randn(2, C)
        out1, _ = mp(x1, batch=torch.zeros(4, dtype=torch.long))
        out2, _ = mp(x2, batch=torch.zeros(2, dtype=torch.long))
        out_b, _ = mp(torch.cat([x1, x2], dim=0), batch=torch.tensor([0, 0, 0, 0, 1, 1]))
        diff = float((torch.cat([out1, out2], dim=0) - out_b).abs().max().item())
        if diff > 1e-3 or not torch.isfinite(out_b).all():
            _record(
                findings,
                "wf_pool_mempool",
                "numerical_divergence",
                "MemPooling",
                f"MemPooling batch invariance violated: max_diff={diff:.4e}",
                "Compare [MemPool(G1), MemPool(G2)] vs MemPool(Batch([G1, G2])).",
                max_diff=diff,
            )
    except Exception as exc:
        _record(findings, "wf_pool_mempool", "crash", "MemPooling", f"{type(exc).__name__}: {exc}", "MemPooling batch", traceback.format_exc())

    return findings


def run_all_whitefox_campaigns() -> List[BugFinding]:
    from gnnsmith.whitefox.campaigns_extra import campaign_hetero_whitebox, campaign_utils_deep_dive

    all_findings: List[BugFinding] = []
    for fn in (
        campaign_edge_index_whitebox,
        campaign_utils_whitebox,
        campaign_aggregations_whitebox,
        campaign_to_hetero_whitebox,
        campaign_pooling_whitebox,
        campaign_hetero_whitebox,
        campaign_utils_deep_dive,
    ):
        all_findings.extend(fn())
    return all_findings
