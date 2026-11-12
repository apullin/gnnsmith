# PyTorch Geometric (`2.8.1`) Bug Ledger — Epoch 1 (`GNNSmith` + `WhiteFox`)

**Target Environment:** `torch-geometric==2.8.1`, `torch==2.15.0`, Python `3.13.14`, Apple M2 Ultra (`arm64` CPU + Metal `MPS`)  
**Verification:** Run `python repros/repro_all.py` (confirms **24/24** deterministic reproducers across **22** distinct root-cause bugs).

---

## Summary Table

| ID | Subsystem | Severity | Bug Class | Affected APIs | Discovery Method |
|---|---|---|---|---|---|
| **BUG-001** | `Index` Tensor Subclass | **High** | Crash (`AttributeError`) | `coalesce(EdgeIndex)`, `sort_edge_index(EdgeIndex)`, `is_undirected(EdgeIndex)`, `Tensor += Index` | WhiteFox (`index.py` dispatch) |
| **BUG-002** | `EdgeIndex` Subclass | **Medium** | Crash (`AttributeError`) | `Tensor - EdgeIndex`, `Tensor += EdgeIndex`, `Tensor -= EdgeIndex` | WhiteFox (`edge_index.py` dispatch) |
| **BUG-003** | `EdgeIndex` Subclass | **Medium** | Metadata / Invariant Corruption | `EdgeIndex - int`, `torch.add(EdgeIndex, k, alpha=-1)` | WhiteFox (`edge_index.py` arith) |
| **BUG-004** | `EdgeIndex` SpMM Autograd | **Critical** | Corrupted CSR Shape + Backward Crash | `EdgeIndex.matmul(x, transpose=True).backward()` on bipartite `(M != N)` graphs | WhiteFox (`_TorchSPMM.backward`) |
| **BUG-005** | `EdgeIndex` SpGEMM / Pool | **High** | Crash (`RuntimeError`) | `EdgeIndex @ EdgeIndex`, `ASAPooling` on Apple Silicon / non-MKL CPUs | WhiteFox (`NO_MKL` guard) |
| **BUG-006** | `torch.compile` + `EdgeIndex` | **High** | Compiler Crash (`SymInt` proxy + cache pickle) | `torch.compile(GCNConv, dynamic=True)(x, EdgeIndex)` + AOTAutograd cache | WhiteFox (PT2 compiler probe) |
| **BUG-007** | `MessagePassing` Core | **Critical** | Crash across 25+ layers & Silent Wrong Math | `decomposed_layers > 1` on `SAGEConv`, `GATConv`, `TransformerConv`, `PNAConv`, `AGNNConv`, etc. | GNNSmith (`decomposed_layers` oracle) |
| **BUG-008** | `EGConv` | **High** | Silent Wrong Math (Equivariance & Batch Leak) | `EGConv(..., add_self_loops=True)` without `'symnorm'` on isolated trailing nodes | GNNSmith (`node_equivariance`, `batch_invariance`, `edge_index_subclass`) |
| **BUG-009** | `ClusterGCNConv` | **High** | Silent Wrong Math + Crash | `ClusterGCNConv(..., add_self_loops=False)` (wrong degree mapping on reordered loops + crash) | GNNSmith (`fuzz-ops` + `fuzz-dag`) |
| **BUG-010** | `ChebConv` | **High** | Silent `NaN` + Cross-Graph Batch Leakage | `ChebConv(..., normalization=None, K>=2)` (`0/0` NaN on edgeless + global `lambda_max` leak) | GNNSmith (`finite_check` + `batch_invariance`) |
| **BUG-011** | `FeaStConv` | **Medium** | Crash (`RuntimeError`) | `FeaStConv(heads>1, add_self_loops=False)` on `E == 0` | GNNSmith (`empty_edges` topology) |
| **BUG-012** | `HypergraphConv` | **Medium** | Unconditional Shape Crash | `HypergraphConv(in, out, use_attention=False, heads>1)` | WhiteFox (`hypergraph_conv.py`) |
| **BUG-013** | `utils.loop` / Spectral Convs | **High** | Non-Determinism & CPU-vs-MPS Divergence | `add_remaining_self_loops` -> `GCNConv`, `SGConv`, `SSGConv`, `APPNP`, `MixHopConv`, `PDNConv` | GNNSmith (`edge_permutation` + `cpu_vs_mps`) |
| **BUG-014** | `FAConv` & `FastRGCNConv` | **Low-Med** | Bare `AssertionError` on Documented Args | `FAConv(normalize=True)(..., edge_weight=...)`, `FastRGCNConv(aggr='max')` | GNNSmith (`fuzz-ops`) |
| **BUG-015** | `utils.softmax` | **High** | Silent `fp16` Overflow to All-Zeros | `softmax(src_fp16, index)` on high-degree hubs (`deg > 65504`) | WhiteFox (`_softmax.py`) |
| **BUG-016** | `utils.softmax` | **Medium** | `ImportError` Asymmetry (`ptr` vs `index`) | `softmax(src, ptr=ptr, dim=1 \| -1)` without `torch-scatter` | WhiteFox (`_softmax.py`) |
| **BUG-017** | `nn.aggr` (`Median`/`Quantile`) | **Medium** | Crash (`RuntimeError`) | `MedianAggregation`, `QuantileAggregation` on empty inputs (`E=0`, `dim_size>0`) | WhiteFox (`aggr` empty-input sweep) |
| **BUG-018** | `nn.aggr` (5 Set/Learned Aggrs) | **High** | Crashes & Silent `NaN`s on Empty Inputs | `SortAggregation`, `GraphMultisetTransformer`, `SetTransformerAggregation`, `LCMAggregation`, `PatchTransformerAggregation` | WhiteFox (`aggr` empty-input sweep) |
| **BUG-019** | `HeteroLayerNorm` | **Low** | Public Param Hits Bare `AssertionError` | `HeteroLayerNorm(..., mode='graph')` | WhiteFox (`hetero_layer_norm.py`) |
| **BUG-020** | `to_hetero` FX Transformer | **Medium** | Misleading Warning Followed by FX Crash | `to_hetero` on graphs with source-only node types (e.g. `author -> paper`) | WhiteFox (`to_hetero_transformer.py`) |
| **BUG-021** | `utils.homophily` | **Medium** | Silent `NaN` & Bare `AssertionError` | `homophily(method='edge')` on `E=0`; `homophily(method='edge_insensitive')` on 1-class graphs | WhiteFox (`_homophily.py`) |
| **BUG-022** | `utils.assortativity` | **Medium** | Crash on `E=0` & Silent `NaN` on Regular Graphs | `assortativity(empty_ei)` and `assortativity(cycle_graph)` | WhiteFox (`_assortativity.py`) |

---

## Detailed Root-Cause Notes

### BUG-001: `Index` RHS In-Place Dispatch Breaks `coalesce`, `sort_edge_index`, and `is_undirected` on `EdgeIndex`
- **Location:** `torch_geometric/index.py:749` (`add_`, `sub_`) & `torch_geometric/utils/_sort_edge_index.py:116`
- **Mechanism:** Indexing a row of an `EdgeIndex` (`ei[0]` or `ei[1]`) returns a `torch_geometric.Index` tensor subclass. In `_sort_edge_index.py:116`, `idx = edge_index[1 - int(sort_by_row)] * num_nodes` produces a plain `torch.Tensor`, and the next line `idx += edge_index[int(sort_by_row)]` invokes `Index.__torch_dispatch__` for `aten.add_.Tensor(plain_tensor, Index)`. `index.py:add_` assumes the LHS `input` is always an `Index` and reads `input.dim_size`, crashing immediately with `AttributeError: 'Tensor' object has no attribute 'dim_size'`.
- **Impact:** Calling `coalesce(ei)`, `sort_edge_index(ei)`, or `is_undirected(ei)` on PyG's own `EdgeIndex` crashes 100% of the time.

### BUG-002: `EdgeIndex` RHS Binary/In-Place Arithmetic Crashes (`Tensor - EdgeIndex`, `Tensor += EdgeIndex`)
- **Location:** `torch_geometric/edge_index.py:1645` (`add_`), `1697` (`_sub`), `1741` (`sub_`)
- **Mechanism:** `_sub`, `add_`, and `sub_` access `input._data` / `input._sparse_size` without checking `isinstance(input, EdgeIndex)` when `EdgeIndex` appears on the RHS.

### BUG-003: `EdgeIndex` Subtraction / Negative Offset Produces Negative Node IDs & Negative `sparse_size`
- **Location:** `torch_geometric/edge_index.py:1600` (`_add`), `1697` (`_sub`)
- **Mechanism:** `ei - 5` or `torch.add(ei, 5, alpha=-1)` wraps negative node indices (e.g. `[[-4, -3], [-3, -4]]`) inside a valid-looking `EdgeIndex` with `sparse_size=(-1, -1)`, `sort_order='row'`, `is_undirected=True`, violating core `EdgeIndex` invariants and risking downstream out-of-bounds indexing in C++/sparse kernels.

### BUG-004: Bipartite `EdgeIndex.matmul(x, transpose=True).backward()` Builds Corrupted CSR Tensor & Crashes
- **Location:** `torch_geometric/edge_index.py:1881-1893` (`_TorchSPMM.backward`)
- **Mechanism:** When `ctx.transpose=True` and `not input.is_undirected` on a rectangular `(M, N)` `EdgeIndex`, `_TorchSPMM.backward` computes `(rowptr, col), perm = input.get_csr()` (`rowptr` has length `M + 1`, `col` values in `[0, N)`) and passes `size=input.get_sparse_size()[::-1]` = `(N, M)` instead of `(M, N)` to `torch.sparse_csr_tensor`. Because PyTorch disables sparse invariant checks by default, this silently instantiates a malformed CSR tensor in C++ memory (`crow_indices` length `M+1 != N+1`, `col_indices >= M`) before crashing in `adj @ grad_out`.

### BUG-005: `typing.NO_MKL` Misdetection Crashes `EdgeIndex @ EdgeIndex` and `ASAPooling` on Apple Silicon
- **Location:** `torch_geometric/typing.py` & `torch_geometric/edge_index.py:1983`
- **Mechanism:** `NO_MKL` evaluates to `False` on macOS `arm64` (where MKL does not exist), routing `EdgeIndex.matmul(EdgeIndex)` and `ASAPooling` to `SparseCsr @ SparseCsr` instead of the `to_sparse_coo()` fallback, crashing unconditionally with `RuntimeError: addmm: computation on CPU is not implemented for SparseCsr + SparseCsr @ SparseCsr without MKL`.

### BUG-006: `torch.compile(GCNConv, dynamic=True)(x, EdgeIndex)` Crashes TorchInductor (`SymInt` Proxy + Unpicklable Cache)
- **Location:** `torch_geometric/edge_index.py` + `torch_geometric/nn/conv/gcn_conv.py`
- **Mechanism:** Compiling `GCNConv` with `dynamic=True` and an `EdgeIndex` input crashes Inductor lowering (`RuntimeError: s92 (<class 'torch.SymInt'>) is not tracked with proxy`), and `EdgeIndex` fails AOTAutograd cache serialization (`Can't get local object 'WeakValueDictionary.__init__.<locals>.remove'`) because `_stable_hash_for_caching` is unimplemented.

### BUG-007: `MessagePassing(decomposed_layers > 1)` Is Broken Across 25+ Layers (and Silently Wrong in `AGNNConv`)
- **Location:** `torch_geometric/nn/conv/message_passing.py:553`
- **Mechanism:** `propagate()` chunks `_j` arguments along feature channel `dim=-1` (`kwargs[a].chunk(decomposed_layers, -1)`):
  1. **Tuple crash (`BUG-007a`):** 18+ layers (`SAGEConv`, `GATConv`, `GATv2Conv`, `GINConv`, `GINEConv`, `GraphConv`, `MFConv`, `GMMConv`, `NNConv`, `CGConv`, `EdgeConv`, `FeaStConv`, `PointNetConv`, `PointTransformerConv`, `GENConv`, `GeneralConv`, `SimpleConv`, `DirGNNConv`) pass `x=(x_src, x_dst)` tuples into `propagate`, crashing immediately with `AttributeError: 'tuple' object has no attribute 'chunk'`.
  2. **Channel mismatch crash (`BUG-007b`):** Layers that apply `Linear`/attention weights inside `message()` (`TransformerConv`, `ResGatedGraphConv`, `FastRGCNConv`, `RGATConv`, `FiLMConv`, `LEConv`, `PNAConv`) receive `C / k`-channel slices against `C`-channel weights and crash.
  3. **Silent mathematical corruption (`BUG-007c`):** `AGNNConv(decomposed_layers=2)` computes cosine similarity + `softmax` independently on `C / 2`-dimensional feature halves and concatenates them (`max_diff = 0.38` vs `decomposed_layers=1`).

### BUG-008: `EGConv` Drops Self-Loops on Trailing Isolated Nodes (Violates Equivariance, Batching & `EdgeIndex` Parity)
- **Location:** `torch_geometric/nn/conv/eg_conv.py:164`
- **Mechanism:** When `'symnorm'` is not in `self.aggregators` and `self.add_self_loops=True`, `EGConv.forward` calls `add_remaining_self_loops(edge_index)` **without `num_nodes=x.size(self.node_dim)`**. Nodes with index `> edge_index.max()` fail to receive self-loops, violating node permutation equivariance (`max_diff = 1.70–7.32`), disjoint batching invariance (`max_diff = 5.90`), and `LongTensor` vs `EdgeIndex` parity.

### BUG-009: `ClusterGCNConv` Crashes on `add_self_loops=False` and Silently Mis-Indexes Node Degrees
- **Location:** `torch_geometric/nn/conv/cluster_gcn_conv.py:121`
- **Mechanism:** `edge_weight[row == col] += self.diag_lambda * deg_inv` adds a `[num_nodes]` vector (`deg_inv`) directly into `[num_self_loops]` (`edge_weight[row == col]`) instead of indexing `deg_inv[row[row == col]]`. Crashes whenever `num_self_loops != num_nodes` (`add_self_loops=False` or duplicate self-loops), and **silently adds the wrong node's degree (`max_diff = 0.48`)** if self-loops appear in any order other than `0, 1, ..., N-1`.

### BUG-010: `ChebConv(normalization=None)` Produces Silent `NaN`s on Edgeless Nodes & Leaks `lambda_max` Across Batches
- **Location:** `torch_geometric/nn/conv/cheb_conv.py:146-158` (`__norm__`)
- **Mechanism:** (1) On edgeless/1-node graphs with `normalization=None`, `edge_weight` is `0.0` and `lambda_max = 0.0`, producing `0.0 / 0.0 = NaN`; `masked_fill_(edge_weight == float('inf'), 0)` misses `NaN`. (2) When `batch` is passed with default `lambda_max=None`, `lambda_max = 2.0 * edge_weight.max()` reduces globally across all graphs in the batch rather than per-graph (`max_diff = 0.40–0.58` cross-graph leakage).

### BUG-011: `FeaStConv(add_self_loops=False)` Crashes on Empty Edge Sets (`E == 0`)
- **Location:** `torch_geometric/nn/conv/feast_conv.py:121`
- **Mechanism:** Reshapes 0-element message tensors via `.view(0, self.heads, -1)` where `-1` is ambiguous (`0 / 0`), raising `RuntimeError`.

### BUG-012: `HypergraphConv(use_attention=False, heads > 1)` Unconditional Shape Crash (`heads` vs `self.heads`)
- **Location:** `torch_geometric/nn/conv/hypergraph_conv.py:92`
- **Mechanism:** In the `use_attention=False` branch, `__init__` sets `self.heads = 1; self.concat = True; self.lin = Linear(in_channels, out_channels)`, then allocates `self.bias = Parameter(torch.empty(heads * out_channels))` using the raw parameter `heads` instead of `self.heads`.

### BUG-013: `add_remaining_self_loops` Duplicate-Index Assignment Race Breaks Edge-Order Invariance & MPS Parity
- **Location:** `torch_geometric/utils/loop.py:296`
- **Mechanism:** `loop_attr[edge_index[0][inv_mask]] = edge_attr[inv_mask]` performs an unreduced indexed write when duplicate self-loops exist on a node. Reordering edges changes outputs by up to `7.39` across `GCNConv`, `SGConv`, `SSGConv`, `APPNP`, `MixHopConv`, and `PDNConv`, and parallel GPU threads race on `MPS` (`max_diff = 0.56–6.09` vs `CPU`).

### BUG-014: Bare `AssertionError` Crashes on Public Args in `FAConv` & `FastRGCNConv`
- **Location:** `torch_geometric/nn/conv/fa_conv.py:157` & `rgcn_conv.py:423`
- **Mechanism:** `FAConv.forward` documents optional `edge_weight` alongside default `normalize=True` yet hits bare `assert edge_weight is None`; `FastRGCNConv` accepts `aggr='max'` in `__init__` then hits bare `assert self.aggr in ['add', 'sum', 'mean']` in `forward`.

### BUG-015: `utils.softmax` `float16` Sum Overflow Collapses Hub Attention Weights to `0.0`
- **Location:** `torch_geometric/utils/_softmax.py:86`
- **Mechanism:** Accumulates `exp(src - src_max)` in `float16` without `float32` upcasting; for hub nodes with `> 65,504` incoming edges, `out_sum` overflows `fp16` max (`65504`) to `Inf`, silently producing `0.0` across all edges (`sum = 0.0` instead of `1.0`).

### BUG-016: `utils.softmax(src, ptr=ptr, dim != 0)` Raises `ImportError` While `index` Path Works
- **Location:** `torch_geometric/utils/_softmax.py:59-79`
- **Mechanism:** Non-zero `dim` with CSR `ptr` calls `segment(..., ptr)` which raises `ImportError: 'segment' in an arbitrary dimension requires 'torch-scatter'` instead of falling back to native PyTorch scatter.

### BUG-017: `MedianAggregation` & `QuantileAggregation` Crash on Empty Inputs (`N=0, dim_size > 0`)
- **Location:** `torch_geometric/nn/aggr/median.py` & `quantile.py`
- **Mechanism:** Fails on `E=0` / empty input tensors with `RuntimeError: index_select(): self indexing axis dim should be positive`.

### BUG-018: 5 Learned/Set Aggregations Crash or Emit `NaN`s on Empty Inputs (`N=0, dim_size > 0`)
- **Location:** `sort.py`, `gmt.py`, `set_transformer.py`, `lcm.py`, `patch_transformer.py`
- **Mechanism:** On `x=empty(0, C), index=empty(0), dim_size=2`: `SortAggregation`, `GraphMultisetTransformer`, `SetTransformerAggregation`, and `LCMAggregation` (`math.log2(0)`) crash with unhandled exceptions, while `PatchTransformerAggregation` silently returns `NaN`s.

### BUG-019: `HeteroLayerNorm` Accepts `mode` Parameter But Hits Bare `assert mode == 'node'`
- **Location:** `torch_geometric/nn/norm/hetero_layer_norm.py:41`

### BUG-020: `to_hetero` Warns That Source-Only Node Types Proceed, Then Crashes Mid-FX Transform
- **Location:** `torch_geometric/nn/to_hetero_transformer.py:151` & `call_method`
- **Mechanism:** `validate()` emits a `UserWarning` when a node type has no incoming edges (e.g., `('author', 'writes', 'paper')`), then downstream FX node creation (`relu`, `Linear`, `output`) crashes with `ValueError: Cannot generate a graph node 'relu' for type 'author'`.

### BUG-021: `utils.homophily` Emits `NaN` on Empty Edges and Bare `AssertionError` on 1-Class Graphs
- **Location:** `torch_geometric/utils/_homophily.py`

### BUG-022: `utils.assortativity` Crashes on `E=0` and Emits `NaN` on Any Regular Graph
- **Location:** `torch_geometric/utils/_assortativity.py`
