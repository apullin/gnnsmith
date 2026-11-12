"""Deterministic minimal reproducers for all 22 PyTorch Geometric 2.8.1 bugs found by GNNSmith + WhiteFox."""

from __future__ import annotations

import argparse
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch_geometric
from torch_geometric import EdgeIndex
import torch_geometric.nn as pyg_nn
import torch_geometric.nn.aggr as pyg_aggr
import torch_geometric.nn.norm as pyg_norm
import torch_geometric.nn.pool as pyg_pool
import torch_geometric.utils as pyg_utils


def bug_001_index_rhs_inplace() -> Tuple[bool, str]:
    """BUG-001: `coalesce(EdgeIndex)`, `sort_edge_index(EdgeIndex)`, `is_undirected(EdgeIndex)` crash in `index.py:add_`."""
    ei = EdgeIndex(torch.tensor([[0, 1], [1, 0]]), sparse_size=(3, 3))
    try:
        pyg_utils.coalesce(ei)
        return False, "Expected AttributeError did not occur"
    except AttributeError as e:
        return True, f"AttributeError: {e}"


def bug_002_edge_index_rsub() -> Tuple[bool, str]:
    """BUG-002: `Tensor - EdgeIndex` and `Tensor += EdgeIndex` crash in `edge_index.py` (`_data` / `_sparse_size`)."""
    ei = EdgeIndex(torch.tensor([[0, 1], [1, 2]]), sparse_size=(4, 4))
    c1 = c2 = False
    try:
        _ = torch.tensor([[3, 3], [3, 3]]) - ei
    except AttributeError:
        c1 = True
    try:
        t = torch.zeros(2, 2, dtype=torch.long)
        t += ei
    except AttributeError:
        c2 = True
    return (c1 and c2), f"Tensor_sub_EdgeIndex_crash={c1}, Tensor_iadd_EdgeIndex_crash={c2}"


def bug_003_edge_index_negative_indices() -> Tuple[bool, str]:
    """BUG-003: `EdgeIndex - int` / `torch.add(ei, 5, alpha=-1)` creates negative indices & negative `sparse_size`."""
    ei = EdgeIndex(torch.tensor([[1, 2], [2, 1]]), sparse_size=(4, 4), sort_order="row", is_undirected=True)
    out = ei - 5
    bad = isinstance(out, EdgeIndex) and bool((out < 0).any().item()) and out.sparse_size()[0] < 0
    return bad, f"type={type(out).__name__}, data={out.tolist()}, sparse_size={out.sparse_size()}, sort_order={out.sort_order}"


def bug_004_bipartite_spmm_transpose_backward() -> Tuple[bool, str]:
    """BUG-004: Rectangular bipartite `EdgeIndex.matmul(..., transpose=True).backward()` swaps CSR size to `(N, M)`."""
    ei = EdgeIndex(torch.tensor([[0, 2, 3, 1], [0, 1, 3, 5]]), sparse_size=(4, 7)).sort_by("col").values
    x = torch.randn(4, 8, requires_grad=True)
    try:
        ei.matmul(x, transpose=True).sum().backward()
        return False, "Expected RuntimeError did not occur"
    except RuntimeError as e:
        return True, f"RuntimeError: {e}"


def bug_005_no_mkl_spgemm_asapooling() -> Tuple[bool, str]:
    """BUG-005: `typing.NO_MKL` is False on Apple Silicon, crashing `EdgeIndex @ EdgeIndex` and `ASAPooling`."""
    torch.manual_seed(0)
    pool = pyg_pool.ASAPooling(8, ratio=0.5).eval()
    try:
        pool(torch.randn(6, 8), torch.tensor([[0, 1, 2], [1, 2, 0]]))
        return False, "Did not crash (platform may have MKL)"
    except RuntimeError as e:
        return True, f"RuntimeError: {e}"


def bug_006_torch_compile_gcn_edge_index() -> Tuple[bool, str]:
    """BUG-006: `torch.compile(GCNConv, dynamic=True)(x, EdgeIndex)` crashes Inductor with untracked SymInt proxy."""
    gcn = torch.compile(pyg_nn.GCNConv(8, 8).eval(), dynamic=True)
    ei = EdgeIndex(torch.tensor([[0, 1, 2], [1, 2, 0]]), sparse_size=(4, 4), sort_order="row")
    try:
        gcn(torch.randn(4, 8), ei)
        return False, "Did not crash"
    except Exception as e:
        return True, f"{type(e).__name__}: {str(e).splitlines()[0]}"


def bug_007a_decomposed_layers_tuple_crash() -> Tuple[bool, str]:
    """BUG-007a: `decomposed_layers=2` crashes 18+ conv layers (`SAGEConv`, `GATConv`, `GINConv`, ...) on tuple `x`."""
    conv = pyg_nn.SAGEConv(8, 8, decomposed_layers=2).eval()
    try:
        conv(torch.randn(6, 8), torch.tensor([[0, 1, 2], [1, 2, 0]]))
        return False, "Did not crash"
    except AttributeError as e:
        return True, f"AttributeError: {e}"


def bug_007b_decomposed_layers_channel_mismatch() -> Tuple[bool, str]:
    """BUG-007b: `decomposed_layers=2` crashes layers projecting inside `message()` (`TransformerConv`, `RGATConv`, `PNAConv`, ...)."""
    conv = pyg_nn.TransformerConv(8, 4, heads=2, decomposed_layers=2).eval()
    try:
        conv(torch.randn(6, 8), torch.tensor([[0, 1, 2], [1, 2, 0]]))
        return False, "Did not crash"
    except RuntimeError as e:
        return True, f"RuntimeError: {e}"


def bug_007c_decomposed_layers_agnn_silent_wrong() -> Tuple[bool, str]:
    """BUG-007c: `AGNNConv(decomposed_layers=2)` silently computes wrong cosine/softmax output vs `decomposed_layers=1`."""
    torch.manual_seed(0)
    x = torch.randn(10, 8)
    ei = torch.randint(0, 10, (2, 24))
    m1 = pyg_nn.AGNNConv(decomposed_layers=1).eval()
    m2 = pyg_nn.AGNNConv(decomposed_layers=2).eval()
    m2.load_state_dict(m1.state_dict())
    diff = float((m1(x, ei) - m2(x, ei)).abs().max().item())
    return diff > 0.05, f"max_diff={diff:.4e}"


def bug_008_egconv_isolated_equivariance() -> Tuple[bool, str]:
    """BUG-008: `EGConv` omits `num_nodes` in `add_remaining_self_loops`, breaking equivariance, batching & EdgeIndex parity."""
    torch.manual_seed(0)
    conv = pyg_nn.EGConv(8, 8, aggregators=["sum", "max"], num_heads=2, num_bases=2, add_self_loops=True).eval()
    x = torch.randn(5, 8)
    ei = torch.tensor([[0, 1], [1, 0]])  # nodes 2, 3, 4 are trailing isolated
    perm = torch.tensor([4, 1, 0, 2, 3])
    inv = torch.argsort(perm)
    y_orig = conv(x, ei)[perm]
    y_perm = conv(x[perm], inv[ei])
    diff = float((y_orig - y_perm).abs().max().item())
    return diff > 1e-2, f"node permutation equivariance max_diff={diff:.4e}"


def bug_009_clustergcn_diag_bug() -> Tuple[bool, str]:
    """BUG-009: `ClusterGCNConv` crashes with `add_self_loops=False` AND silently mis-indexes degrees on reordered self-loops."""
    torch.manual_seed(0)
    conv = pyg_nn.ClusterGCNConv(4, 4, diag_lambda=1.0, add_self_loops=False).eval()
    x = torch.randn(2, 4)
    # Part 1: Silent wrong math on identical graph with reordered self-loops
    ei_a = torch.tensor([[0, 1, 1], [0, 0, 1]])
    ei_b = torch.tensor([[1, 1, 0], [1, 0, 0]])
    diff = float((conv(x, ei_a) - conv(x, ei_b)).abs().max().item())
    # Part 2: Crash when no self-loops
    crashed = False
    try:
        conv(x, torch.tensor([[0], [1]]))
    except RuntimeError:
        crashed = True
    return (diff > 0.1 and crashed), f"silent edge-reorder max_diff={diff:.4e} AND crash_without_loops={crashed}"


def bug_010_chebconv_nan_and_batch_leak() -> Tuple[bool, str]:
    """BUG-010: `ChebConv(normalization=None)` emits silent `NaN` on edgeless/1-node graphs (`0/0`) AND leaks `lambda_max` across `batch`."""
    torch.manual_seed(0)
    conv = pyg_nn.ChebConv(4, 4, K=2, normalization=None).eval()
    y_single = conv(torch.randn(1, 4), torch.empty((2, 0), dtype=torch.long))
    has_nan = bool(torch.isnan(y_single).any().item())

    x1, ei1 = torch.randn(3, 4), torch.tensor([[0, 1], [1, 0]])
    x2, ei2 = torch.randn(4, 4), torch.tensor([[0, 1, 2, 0], [1, 2, 3, 3]])
    y_sep = torch.cat([conv(x1, ei1), conv(x2, ei2)], dim=0)
    y_bat = conv(
        torch.cat([x1, x2], dim=0),
        torch.cat([ei1, ei2 + 3], dim=1),
        batch=torch.tensor([0, 0, 0, 1, 1, 1, 1]),
    )
    leak = float((y_sep - y_bat).abs().max().item())
    return (has_nan and leak > 1e-2), f"edgeless_nan={has_nan}, batch_leak_max_diff={leak:.4e}"


def bug_011_feastconv_empty_edges() -> Tuple[bool, str]:
    """BUG-011: `FeaStConv(add_self_loops=False)` crashes on empty edge_index `(2, 0)` due to ambiguous `.view(0, H, -1)`."""
    conv = pyg_nn.FeaStConv(8, 4, heads=2, add_self_loops=False).eval()
    try:
        conv(torch.randn(4, 8), torch.empty((2, 0), dtype=torch.long))
        return False, "Did not crash"
    except RuntimeError as e:
        return True, f"RuntimeError: {e}"


def bug_012_hypergraph_conv_heads_typo() -> Tuple[bool, str]:
    """BUG-012: `HypergraphConv(use_attention=False, heads=2)` crashes because `bias` uses `heads` instead of `self.heads`."""
    conv = pyg_nn.HypergraphConv(8, 4, use_attention=False, heads=2).eval()
    try:
        conv(torch.randn(4, 8), torch.tensor([[0, 1, 2], [0, 0, 1]]))
        return False, "Did not crash"
    except RuntimeError as e:
        return True, f"RuntimeError: {e}"


def bug_013_add_remaining_self_loops_race() -> Tuple[bool, str]:
    """BUG-013: `add_remaining_self_loops` duplicate-index write breaks edge-order invariance & CPU-vs-MPS determinism."""
    ei_1 = torch.tensor([[0, 0, 0], [0, 0, 1]])
    ew_1 = torch.tensor([0.2, 5.0, 1.0])
    ei_2 = torch.tensor([[0, 0, 0], [0, 1, 0]])
    ew_2 = torch.tensor([5.0, 1.0, 0.2])
    _, out_w1 = pyg_utils.add_remaining_self_loops(ei_1, ew_1)
    _, out_w2 = pyg_utils.add_remaining_self_loops(ei_2, ew_2)
    diff = float((out_w1 - out_w2).abs().max().item())
    return diff > 1.0, f"reordering duplicate self-loop changed weight by {diff:.2f} ({out_w1.tolist()} vs {out_w2.tolist()})"


def bug_014_faconv_fastrgcn_bare_asserts() -> Tuple[bool, str]:
    """BUG-014: `FAConv(normalize=True)` with documented `edge_weight` and `FastRGCNConv(aggr='max')` crash with bare `AssertionError`."""
    fa = pyg_nn.FAConv(4, normalize=True).eval()
    rg = pyg_nn.FastRGCNConv(4, 4, num_relations=2, aggr="max").eval()
    x = torch.randn(3, 4)
    ei = torch.tensor([[0, 1], [1, 2]])
    c1 = c2 = False
    try:
        fa(x, x, ei, edge_weight=torch.ones(2))
    except AssertionError:
        c1 = True
    try:
        rg(x, ei, torch.tensor([0, 1]))
    except AssertionError:
        c2 = True
    return (c1 and c2), f"FAConv_AssertionError={c1}, FastRGCNConv_AssertionError={c2}"


def bug_015_softmax_fp16_hub_overflow() -> Tuple[bool, str]:
    """BUG-015: `utils.softmax` in `float16` overflows denominator to `Inf` on hubs (>65504 edges), returning all zeros."""
    src = torch.zeros(70_000, 1, dtype=torch.float16)
    idx = torch.zeros(70_000, dtype=torch.long)
    out = pyg_utils.softmax(src, index=idx, num_nodes=1)
    total = float(out.float().sum().item())
    return total == 0.0, f"softmax probability sum={total} (expected 1.0)"


def bug_016_softmax_ptr_nonzero_dim() -> Tuple[bool, str]:
    """BUG-016: `utils.softmax(src, ptr=ptr, dim=-1)` crashes with `ImportError` whereas `index` path succeeds."""
    src = torch.randn(2, 4)
    ptr = torch.tensor([0, 2, 4])
    try:
        pyg_utils.softmax(src, ptr=ptr, dim=-1)
        return False, "Did not crash"
    except ImportError as e:
        return True, f"ImportError: {e}"


def bug_017_median_quantile_empty_crash() -> Tuple[bool, str]:
    """BUG-017: `MedianAggregation` & `QuantileAggregation` crash on empty inputs `(0, C)` with `dim_size > 0`."""
    med = pyg_aggr.MedianAggregation()
    qnt = pyg_aggr.QuantileAggregation(q=0.5)
    c1 = c2 = False
    try:
        med(torch.empty(0, 4), torch.empty(0, dtype=torch.long), dim_size=2)
    except RuntimeError:
        c1 = True
    try:
        qnt(torch.empty(0, 4), torch.empty(0, dtype=torch.long), dim_size=2)
    except RuntimeError:
        c2 = True
    return (c1 and c2), f"Median_crash={c1}, Quantile_crash={c2}"


def bug_018_learned_aggrs_empty_crash_or_nan() -> Tuple[bool, str]:
    """BUG-018: `SortAggregation`, `GMT`, `SetTransformer`, `LCM` crash and `PatchTransformer` emits `NaN` on empty input."""
    x0 = torch.empty(0, 8)
    i0 = torch.empty(0, dtype=torch.long)
    results = []
    for name, mod in [
        ("Sort", pyg_aggr.SortAggregation(k=2)),
        ("GMT", pyg_aggr.GraphMultisetTransformer(8, k=2, heads=2)),
        ("SetTransformer", pyg_aggr.SetTransformerAggregation(8, num_seed_points=2, heads=2)),
        ("LCM", pyg_aggr.LCMAggregation(8, 8)),
    ]:
        try:
            mod.eval()(x0, i0, dim_size=2)
            results.append((name, False))
        except Exception:
            results.append((name, True))
    pt_out = pyg_aggr.PatchTransformerAggregation(8, 8, patch_size=2, hidden_channels=8, heads=2).eval()(x0, i0, dim_size=2)
    pt_nan = bool(torch.isnan(pt_out).any().item())
    ok = all(flag for _, flag in results) and pt_nan
    return ok, f"crashes={results}, PatchTransformer_NaN={pt_nan}"


def bug_019_hetero_layer_norm_mode_assert() -> Tuple[bool, str]:
    """BUG-019: `HeteroLayerNorm` exposes `mode: str = 'node'` parameter in public API but asserts `mode == 'node'`."""
    try:
        pyg_norm.HeteroLayerNorm(8, num_types=2, mode="graph")
        return False, "Did not fail"
    except AssertionError:
        return True, "Bare AssertionError on mode='graph'"


def bug_020_to_hetero_source_only_node_crash() -> Tuple[bool, str]:
    """BUG-020: `to_hetero` emits a warning for source-only node types then crashes during FX graph generation."""
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = pyg_nn.SAGEConv((-1, -1), 8)

        def forward(self, x, edge_index):
            return self.c(x, edge_index).relu()

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            pyg_nn.to_hetero(M(), (["author", "paper"], [("author", "writes", "paper")]))
            return False, "Did not crash"
        except ValueError as e:
            return True, f"ValueError after non-fatal warning: {e}"


def bug_021_homophily_nan_and_single_class_assert() -> Tuple[bool, str]:
    """BUG-021: `homophily` returns `NaN` on edgeless graphs (`edge`) and hits bare `AssertionError` on 1-class graphs (`edge_insensitive`)."""
    h_nan = pyg_utils.homophily(torch.empty((2, 0), dtype=torch.long), torch.tensor([0, 1]), method="edge")
    is_nan = h_nan != h_nan
    crashed = False
    try:
        pyg_utils.homophily(torch.tensor([[0, 1], [1, 2]]), torch.tensor([0, 0, 0]), method="edge_insensitive")
    except AssertionError:
        crashed = True
    return (is_nan and crashed), f"edge_empty_NaN={is_nan}, edge_insensitive_1class_AssertionError={crashed}"


def bug_022_assortativity_empty_crash_and_regular_nan() -> Tuple[bool, str]:
    """BUG-022: `assortativity` crashes on `E=0` and returns `NaN` on any regular graph (zero degree variance)."""
    crashed = False
    try:
        pyg_utils.assortativity(torch.empty((2, 0), dtype=torch.long))
    except RuntimeError:
        crashed = True
    a_reg = pyg_utils.assortativity(torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]))
    is_nan = a_reg != a_reg
    return (crashed and is_nan), f"empty_RuntimeError={crashed}, regular_cycle_NaN={is_nan}"


ALL_REPROS: Dict[str, Callable[[], Tuple[bool, str]]] = {
    "BUG-001": bug_001_index_rhs_inplace,
    "BUG-002": bug_002_edge_index_rsub,
    "BUG-003": bug_003_edge_index_negative_indices,
    "BUG-004": bug_004_bipartite_spmm_transpose_backward,
    "BUG-005": bug_005_no_mkl_spgemm_asapooling,
    "BUG-006": bug_006_torch_compile_gcn_edge_index,
    "BUG-007a": bug_007a_decomposed_layers_tuple_crash,
    "BUG-007b": bug_007b_decomposed_layers_channel_mismatch,
    "BUG-007c": bug_007c_decomposed_layers_agnn_silent_wrong,
    "BUG-008": bug_008_egconv_isolated_equivariance,
    "BUG-009": bug_009_clustergcn_diag_bug,
    "BUG-010": bug_010_chebconv_nan_and_batch_leak,
    "BUG-011": bug_011_feastconv_empty_edges,
    "BUG-012": bug_012_hypergraph_conv_heads_typo,
    "BUG-013": bug_013_add_remaining_self_loops_race,
    "BUG-014": bug_014_faconv_fastrgcn_bare_asserts,
    "BUG-015": bug_015_softmax_fp16_hub_overflow,
    "BUG-016": bug_016_softmax_ptr_nonzero_dim,
    "BUG-017": bug_017_median_quantile_empty_crash,
    "BUG-018": bug_018_learned_aggrs_empty_crash_or_nan,
    "BUG-019": bug_019_hetero_layer_norm_mode_assert,
    "BUG-020": bug_020_to_hetero_source_only_node_crash,
    "BUG-021": bug_021_homophily_nan_and_single_class_assert,
    "BUG-022": bug_022_assortativity_empty_crash_and_regular_nan,
}


def run_repros(selected: str | None = None) -> int:
    print(f"Verifying PyG {torch_geometric.__version__} (torch {torch.__version__}) bug reproducers:")
    items = [(selected, ALL_REPROS[selected])] if selected else list(ALL_REPROS.items())
    reproduced = 0
    for bug_id, fn in items:
        ok, msg = fn()
        status = "REPRODUCED" if ok else "NOT_REPRODUCED"
        if ok:
            reproduced += 1
        print(f"  [{status:14s}] {bug_id:8s} | {fn.__doc__.strip().splitlines()[0]}")
        print(f"                   └─ {msg}")
    print(f"\nSummary: {reproduced}/{len(items)} bug reproducers confirmed.")
    return 0 if reproduced == len(items) else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bug", choices=list(ALL_REPROS.keys()), default=None)
    args = p.parse_args()
    raise SystemExit(run_repros(args.bug))
