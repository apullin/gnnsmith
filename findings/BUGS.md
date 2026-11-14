# PyTorch Geometric (`2.8.1` / `2.9.0-dev`) Master Bug Ledger — Waves 1–8 (`GNNSmith` + `WhiteFox`)

**Target Environment:** `torch-geometric` (`upstream/master` `2.9.0`), `torch==2.15.0`, Python `3.13.14`, Apple M2 Ultra (`arm64` CPU + Metal `MPS`)  
**Fix Branch:** [`apullin/pytorch_geometric` → `mega-bugfix`](https://github.com/apullin/pytorch_geometric/tree/mega-bugfix) (8 stacked subsystem commits)  
**Local Verification:** `uv run python repros_pyg.py` (**`47/47`** bug reproducer checks across **44 distinct PyG bugs**; **`0/47`** remaining on `mega-bugfix`).

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
| **BUG-023** | `MFConv` + PyTorch AMP | **High** | Crash under `torch.autocast` (`RuntimeError`) | `MFConv` under CPU/CUDA `bfloat16`/`float16` `autocast` (`index_copy_` dtype mismatch) | Deep Fuzz D2 (AMP) |
| **BUG-024** | `ClusterGCNConv` + MPS `fp16` | **Critical** | `bf16` Crash & Hard C++ `SIGABRT` on MPS `fp16` | `ClusterGCNConv` in `bfloat16` / `float16` (`degree()` defaults to `fp32`, aborting `MPSNDArrayMatrixMultiplication`) | Deep Fuzz D2 (MPS `fp16`) |
| **BUG-025** | `EdgeIndex` SpMM Dtypes & Grad | **Critical** | Crash on `fp64`/`fp16`/`bf16` + Silent Wrong Backward Grad | `EdgeIndex.matmul(x)` (`_get_value()` defaults to `fp32`; backward overwrites `value` with ones when `perm is None`) | Deep Fuzz D3 (`gradcheck` + WhiteFox) |
| **BUG-026** | `transforms.NormalizeScale` | **Medium** | Silent `NaN` Corruption (`0.0 * inf`) | `NormalizeScale()` on single-node (`N=1`) or colocated point clouds | Deep Fuzz D4 (Transforms) |
| **BUG-027** | `transforms.Polar` & `Spherical` | **Medium** | Crash (`RuntimeError`) on Edgeless Graphs | `Polar()`, `Spherical()` when `E == 0` (`rho.max()` on empty tensor) | Deep Fuzz D4 (Transforms) |
| **BUG-028** | `transforms.Spherical` (+ `Cartesian`/`Distance`) | **High** | Silent `NaN` on Self-Loops / Colocated Edges | `Spherical()` (`0.0 / 0.0` inside `acos` on any self-loop `(i, i)`); `Cartesian`/`Distance` when `max_val == 0` | Deep Fuzz D4 + WhiteFox |
| **BUG-029** | `transforms.GDC` | **Medium** | Crash (`linalg.inv` 0D Tensor & Trailing Isolated Size) | `GDC()` on single-node (`N=1`) graphs (`.squeeze()` -> 0D) & isolated trailing nodes | Deep Fuzz D4 (Transforms) |
| **BUG-030** | `nn.models.GraphUNet` | **High** | Crash (`RuntimeError`) on Apple Silicon / Windows | `GraphUNet` (`SparseCsr @ SparseCsr` without `NO_MKL` fallback) | Deep Fuzz D5 (Models) |
| **BUG-031** | `data` Serialization (PT 2.6+) | **High** | `UnpicklingError` on Default `torch.load(weights_only=True)` | `torch.save` + `torch.load` on `Data`, `Batch`, `HeteroData`, `TemporalData` | Deep Fuzz D7 (Serialization) |
| **BUG-032** | `Index`/`EdgeIndex` + TorchDynamo | **High** | `InternalTorchDynamoError` (`AttributeError: _data`) | `Index.__new__` / `EdgeIndex.__new__` when Dynamo traces subclass init directly | Epoch 2 `pytest` + PT2 Compile |
| **BUG-033** | `utils.spmm` & `ARMAConv` | **High** | Crash (`RuntimeError`) on 3D Feature Tensors | `spmm` & `ARMAConv` (`[K, N, F]`) on `torch.sparse` COO/CSR/CSC (`mat2 must be a matrix, got 3-D tensor`) | Wave 5 L1 (Sparse Layouts) |
| **BUG-034** | `GeneralConv` Bipartite | **Medium** | Crash (`TypeError`) on `(x_src, None)` | `GeneralConv` (`OptPairTensor` signature crashes calling `lin_self(None)`) | Wave 5 B1 (Bipartite Z3) |
| **BUG-035** | `ClusterGCNConv` Flow | **High** | Silent Wrong Math (`max_diff = 1.76`) | `ClusterGCNConv(flow='target_to_source')` hardcodes `col = edge_index[1]` | Wave 5 B2 (`flow` Duality) |
| **BUG-036** | `ChebConv` Flow | **High** | Silent Wrong Math (`max_diff = 1.04`) | `ChebConv(flow='target_to_source')` calls `get_laplacian` along unflipped `edge_index[0]` | Wave 5 B2 (`flow` Duality) |
| **BUG-037** | `GATConv` & `GATv2Conv` Flow | **High** | Silent Wrong Math (`max_diff = 2.79`) | `GATConv` / `GATv2Conv` (`flow='target_to_source'`) aggregates self-loop `edge_attr` onto source axis | Wave 5 B2 (`flow` Duality) |
| **BUG-038** | `MessagePassing` Core / Jinja | **Critical** | Jinja vs Python `_collect` Mismatch + Lost `size` | `edge_updater` hardcodes `size=None`; `_collect` swaps `(src, dst)` tuples under `target_to_source` vs `collect.jinja`, and `collect.jinja` passes hardcoded `0`/`1` to `_set_size` | Wave 5 B2 + WhiteFox |
| **BUG-039** | `nn.models.TGNMemory` | **High** | Crash (`ValueError`) on `.eval()` / Empty Batch | `TGNMemory.eval()` before updates or `update_state` with 0 events crashes unpacking `zip(*[])` | Wave 5 A2 (Temporal Models) |
| **BUG-040** | `utils.loop` + `torch.export` | **Medium** | `GuardOnDataDependentSymNode` Export Failure | `MixHopConv` / weighted `add_remaining_self_loops` branching on unbacked masked self-loop count | Wave 5 `torch.export` Sweep |
| **BUG-041** | `DenseGATConv` | **High** | Silent `NaN` Forward & Backward | `DenseGATConv(add_loop=False)` runs `softmax([-inf, ..., -inf]) -> NaN` on isolated/edgeless nodes | Wave 7 W7.1 (Dense Convs) |
| **BUG-042** | `dense_mincut_pool` & `DMoNPooling` | **High** | Silent `NaN` Regularization Loss (`0 / 0`) | `dense_mincut_pool` & `DMoNPooling` divide `0 / 0` whenever a graph in a batch has `0` edges | Wave 7 W7.2 (Dense Pooling) |
| **BUG-043** | Graph Transformer Attention | **Critical** | Silent Padding Token Leakage Across Batched Graphs | `PerformerAttention`, `SGFormerAttention`, and `PolynormerAttention` mask `v` but forget to mask `k` (and `SGFormer` uses padded `N` instead of valid node count), corrupting variable-size batches (`max_diff = 0.12–0.39`) | Wave 7 W7.3 (Masked Padding Oracle) |
| **BUG-044** | `nn.kge` (`TransE`/`ComplEx`/`DistMult`/`RotatE`) | **Medium** | Silent `NaN` Loss on Empty Triplet Batches | All 4 KGE models return `NaN` from `.loss()` when `head_index` has `0` triplets | Wave 7 W7.3 (KGE) |
| *(MLX-001)* | `mlx-graphs` (`GCNConv`) | **Medium** | Crash (`AssertionError`) on Empty Edges | `mlx_graphs.nn.GCNConv` asserts `edge_index[1].size > 0`, crashing on `E == 0` | Wave 5 A3 (PyG-vs-MLX Oracle) |

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

---

## Epoch 3 Deep Fuzz Additions (`torch.compile`, AMP/`bf16`/`fp16`, `gradcheck`, Transforms, Models, Serialization)

### BUG-023: `MFConv` Crashes Under `torch.autocast` (`bfloat16` / `float16` AMP)
- **Location:** `torch_geometric/nn/conv/mf_conv.py:107-115`
- **Mechanism:** `out = h.new_empty(...)` allocates `float32`, while `lin_l(...)` under `torch.autocast` returns `bfloat16`/`float16`; `out.index_copy_(self.node_dim, idx, r)` crashes with `RuntimeError: index_copy_(): self and source expected to have the same dtype`.

### BUG-024: `ClusterGCNConv` Crashes in `bfloat16` and Aborts Process (`SIGABRT`) on MPS `float16`
- **Location:** `torch_geometric/nn/conv/cluster_gcn_conv.py:90,103,118`
- **Mechanism:** `degree(col, num_nodes=...)` defaults to `torch.float32` because `dtype=x.dtype` was omitted, promoting message-passing outputs to `float32` before `self.lin_out(out)` (`float16`/`bfloat16`). On CPU `bfloat16` it raises `RuntimeError`; on Apple Silicon MPS `float16` it hits a fatal Objective-C++ assertion in `MPSNDArrayMatrixMultiplication.mm:6941` (`SIGABRT`, exit code 134).

### BUG-025: `EdgeIndex.matmul` Crashes on `float64`/`float16`/`bfloat16` (`025a`) & Drops Weights in Backward When `perm is None` (`025b`)
- **Location:** `torch_geometric/edge_index.py:1875-1945` (`_TorchSPMM`)
- **Mechanism:** (a) Unweighted `_TorchSPMM.forward`/`backward` called `self._get_value()` with `dtype=None` (`float32`), crashing `torch.sparse.mm` whenever `other` is `float64`, `float16`, or `bfloat16`. (b) In `_TorchSPMM.backward`, `if value is not None and perm is not None: value = value[perm] else: value = input._get_value()` entered `else:` whenever `value` was provided on an already bi-sorted graph (`perm is None`, e.g. diagonal adjacency), silently replacing edge weights with `1.0` during backpropagation!

### BUG-026: `transforms.NormalizeScale` Corrupts Single-Node / Colocated `pos` to `NaN`
- **Location:** `torch_geometric/transforms/normalize_scale.py:21`
- **Mechanism:** `Center()` zeroes out a single node's position, making `scale = 1.0 / 0.0 = inf` and `data.pos * scale = 0.0 * inf = NaN`.

### BUG-027: `transforms.Polar` and `Spherical` Crash on Edgeless Graphs (`E == 0`)
- **Location:** `torch_geometric/transforms/polar.py:48` & `spherical.py:51`
- **Mechanism:** Calls `rho.max()` on an empty `(0, 1)` tensor when `E == 0` and `max_value is None`.

### BUG-028: `transforms.Spherical` Emits `NaN` on Every Self-Loop / Colocated Edge (+ `Cartesian`/`Distance` Zero-Range `NaN`)
- **Location:** `torch_geometric/transforms/spherical.py:48`, `cartesian.py:52`, `distance.py:50`
- **Mechanism:** `torch.acos(cart[..., 2] / rho.view(-1))` evaluates `0.0 / 0.0 = NaN` on any self-loop `(i, i)` where `pos[col] == pos[row]`.

### BUG-029: `transforms.GDC` Crashes on Single-Node Graphs (`N == 1`) & Trailing Isolated Nodes
- **Location:** `torch_geometric/transforms/gdc.py:245,255,260`
- **Mechanism:** `to_dense_adj(edge_index, edge_attr=edge_weight).squeeze()` omits `max_num_nodes=num_nodes` and squeezes `(1, 1, 1)` to a 0D scalar `()`, crashing `torch.inverse(mat)`.

### BUG-030: `nn.models.GraphUNet` Crashes on Apple Silicon & Windows Non-MKL CPUs
- **Location:** `torch_geometric/nn/models/graph_unet.py:143` (`augment_adj`)
- **Mechanism:** Unconditionally multiplies `SparseCsr @ SparseCsr` on CPU without a `NO_MKL` `to_torch_coo_tensor` + `torch.sparse.mm` fallback.

### BUG-031: `torch.load(..., weights_only=True)` (PyTorch 2.6+ Default) Fails on `Data`, `Batch`, and `HeteroData`
- **Location:** `torch_geometric/data/__init__.py:77` & `torch_geometric/data/batch.py:53,215`
- **Mechanism:** Missing `add_safe_globals` entries (`DataEdgeAttr`, `DataTensorAttr`, `BaseStorage`, `NodeStorage`, `EdgeStorage`, `GlobalStorage`, `Batch`) plus `Batch.__reduce__` using a `DynamicInheritanceGetter()` callable instance (rejected by `WeightsUnpickler` `REDUCE`) and dynamic `DataBatch`/`HeteroDataBatch` metaclass subclasses not registered at import time.

### BUG-032: `Index.__new__` / `EdgeIndex.__new__` Crashes TorchDynamo `is_fake()` When Traced Directly
- **Location:** `torch_geometric/index.py:185` & `torch_geometric/edge_index.py:329`
- **Mechanism:** Dynamo graph-breaks immediately after `Tensor._make_wrapper_subclass(...)` before `out._data = data` is attached; resuming calls `is_fake(out)` -> `__tensor_flatten__` -> `getattr(out, '_data')`, raising `InternalTorchDynamoError: AttributeError: 'Index' object has no attribute '_data'`.

---

## Stacked Fix Commits on `apullin/pytorch_geometric:mega-bugfix`

1. `a1c08ea8` — Fix `Index`/`EdgeIndex` RHS dispatch, bipartite SpMM backward, MKL check (`ASAPooling`), and dynamic `torch.compile` (`BUG-001`..`BUG-006`, `BUG-032`)
2. `492d4b96` — Fix `MessagePassing` `decomposed_layers` for tuple inputs and guard non-decomposable layers (`BUG-007a/b/c`)
3. `33c1dc0e` — Fix isolated-node, empty-edge, batch-leak, and parameter bugs across 7 `Conv` layers (`BUG-008`..`BUG-012`, `BUG-014`)
4. `ee590437` — Fix crashes and `NaN`s on empty inputs across 8 `Aggregation` modules (`BUG-017`, `BUG-018`)
5. `d9911ced` — Fix self-loop multigraph race, `fp16`/arbitrary-dim `softmax`, `homophily`, `assortativity`, `HeteroLayerNorm`, `to_hetero` source-only nodes, and Python 3.13+ `inspector.py` (`BUG-013`, `BUG-015`, `BUG-016`, `BUG-019`..`BUG-022`)
6. `c24032ba` — Fix AMP/low-prec convs (`MFConv`, `ClusterGCNConv` MPS `SIGABRT`), `float64`/weighted SpMM, geometric/`GDC` transforms, `GraphUNet` non-MKL, and `weights_only=True` serialization (`BUG-023`..`BUG-031`)
7. `e7bca7f3` — Enable 3D `ptr` assertion in `test_attentional_aggregation` on PyTorch >= 2.0 (`BUG-016`)
8. `238be457` — Fix 3D `spmm` (`ARMAConv`), bipartite `GeneralConv`, `flow='target_to_source'` duality (`ClusterGCNConv`, `ChebConv`, `GATConv`, `GATv2Conv`, `MessagePassing._collect`/`collect.jinja`), `TGNMemory` empty batch/eval, `torch.export` branchless self-loops, `DenseGATConv`/`mincut_pool`/`DMoNPooling` edgeless `NaN`s, linear attention (`Performer`/`SGFormer`/`Polynormer`) padding leakage, and empty KGE loss (`BUG-033`..`BUG-044`)

---

## Waves 5–8 Additions (Bipartite, Flow Duality, Sparse Layouts, `torch.export`, Dense GNNs, & Graph Transformer Attention)

### BUG-033: `utils.spmm` & `ARMAConv` Crash on 3D Feature Tensors (`[K, N, F]`)
- **Location:** `torch_geometric/utils/_spmm.py` & `nn/conv/arma_conv.py`
- **Mechanism:** `ARMAConv` stacks `num_stacks` into `[K, N, F]` and delegates `torch.sparse` COO/CSR inputs to `spmm(adj_t, x)`, which called 2D-only `torch.sparse.mm` directly (`RuntimeError: addmm: mat2 must be a matrix, got 3-D tensor`). PyG even had a `with pytest.raises(RuntimeError): # No 3D feature tensor support` workaround in `test_arma_conv.py`, now resolved and enabled.

### BUG-034: `GeneralConv` Crashes on `(x_src, None)` Bipartite Inputs
- **Location:** `torch_geometric/nn/conv/general_conv.py:158`
- **Mechanism:** Annotated `x: Union[Tensor, OptPairTensor]`, but called `self.lin_self(x[1])` unconditionally when `x[1]` was `None`.

### BUG-035, BUG-036, BUG-037: `ClusterGCNConv`, `ChebConv`, `GATConv`, and `GATv2Conv` Violate `flow='target_to_source'` Mathematical Duality
- **Locations:** `cluster_gcn_conv.py:118`, `cheb_conv.py:118`, `gat_conv.py:345`, `gatv2_conv.py:310`
- **Mechanism:** `conv(flow='source_to_target')(x, ei)` must identically equal `conv(flow='target_to_source')(x, ei.flip(0))`, yet `ClusterGCNConv` hardcoded `col = edge_index[1]` (`diff = 1.76`), `ChebConv` passed unflipped `edge_index` to `get_laplacian` (`diff = 1.04`), and `GATConv`/`GATv2Conv` computed self-loop `fill_value='mean'` edge attributes along `edge_index[1]` (`diff = 0.69–2.79`).

### BUG-038: `MessagePassing._collect` vs `collect.jinja` Tuple Disagreement Under `target_to_source` + Ignored `size` in `edge_updater`
- **Locations:** `torch_geometric/nn/conv/message_passing.py:358,665` & `collect.jinja:122,128`
- **Mechanism:** Python `_collect` indexed 2-tuples `(x_0, x_1)` by `dim` (`swapping` source/target under `flow='target_to_source'`), whereas `collect.jinja` always treated `[0]` as `_j` and `[1]` as `_i` (while passing hardcoded `0`/`1` instead of `j`/`i` to `_set_size`), causing `decomposed_layers > 1` / `explain=True` to disagree with Jinja propagation! Meanwhile `edge_updater` hardcoded `self._check_input(edge_index, size=None)` instead of passing `size`.

### BUG-039: `TGNMemory` Crashes on `.eval()` Before Updates or on Empty Event Batches
- **Location:** `torch_geometric/nn/models/tgn.py:131,165,183`
- **Mechanism:** Calling `tgn.eval()` flushes `msg_store` via `_update_memory`, and when `msg_store` or `n_id` is empty, `src, dst, t, raw_msg = list(zip(*data))` crashes with `ValueError: not enough values to unpack (expected 4, got 0)`.

### BUG-040: Data-Dependent Self-Loop Count Guard Breaks `torch.export.export`
- **Location:** `torch_geometric/utils/loop.py:644`
- **Mechanism:** Replaced Python `if loop_nodes.numel() > 0:` branch on boolean-masked `loop_nodes` with branchless `scatter_` + `scatter` + `torch.where`, allowing strict `torch.export.export` on `MixHopConv` and weighted spectral layers.

### BUG-041: `DenseGATConv(add_loop=False)` Emits `NaN` Forward & Backward on Isolated / Edgeless Nodes
- **Location:** `torch_geometric/nn/dense/dense_gat_conv.py:94-97`
- **Mechanism:** Filling an entire row of `alpha` with `-inf` when a node has `0` neighbors causes `.softmax(dim=2)` to evaluate `0 / 0 = NaN` (and `SoftmaxBackward0` propagates `0 * NaN = NaN` into gradients even if masked afterward).

### BUG-042: `dense_mincut_pool` & `DMoNPooling` Emit `NaN` Loss on Edgeless Graphs (`adj == 0`)
- **Locations:** `torch_geometric/nn/dense/mincut_pool.py:86` & `dmon_pool.py:136`
- **Mechanism:** Zero total degree (`mincut_den == 0` / `2m == 0`) divides `0 / 0 = NaN` in `mincut_loss` and `spectral_loss`.

### BUG-043: `PerformerAttention`, `SGFormerAttention`, and `PolynormerAttention` Leak Masked Padding Tokens Into Valid Node Embeddings
- **Locations:** `nn/attention/performer.py:92,165`, `sgformer.py:65-88`, `polynormer.py:76-88`
- **Mechanism:** All three linear-attention Graph Transformer modules zeroed `v` at `~mask` **but forgot to zero `k` after positive kernel/activation maps (`ReLU + eps`, `L2-norm`, `sigmoid(0) = 0.5`)**, injecting phantom key mass from padded positions into the attention denominator `q @ k.sum()` (and `SGFormerAttention` additionally added the padded batch width `N` instead of per-graph valid node counts `mask.sum(dim=1)`!). Corrupted variable-size batched inference/training across `GPSConv(attn_type='performer')`, `SGFormer`, and `PolynormerAttention`.

### BUG-044: Knowledge Graph Embedding Models (`TransE`, `ComplEx`, `DistMult`, `RotatE`) Return `NaN` Loss on Empty Batches
- **Locations:** `nn/kge/{transe,complex,distmult,rotate}.py`


