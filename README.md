# GNNSmith (v0.2.0)

Z3-constrained Graph Neural Network DAG synthesis, metamorphic/differential oracles, `sys.monitoring` coverage guidance, automatic delta-debugging reduction, and WhiteFox white-box testing for PyTorch Geometric and graph ML libraries.

## Core Capabilities

1. **Z3 Relational & Bipartite/Hetero GraphIR Synthesis (`gnnsmith.graph_gen`, `gnnsmith.bipartite_hetero`)**:
   - Symbolic Z3 solving over channel widths, multi-head/concat constraints, PNA/MixHop multipliers, skip/residual merges, rectangular bipartite `(N_src != N_dst, C_src != C_dst)` PairTensors, multi-relation `HeteroData` schemas, and topology-mutating pooling scopes across 57+ PyG operators and 10 adversarial topologies.
2. **Zero-Label Algebraic, Layout & Differential Oracles (`gnnsmith.oracles`, `gnnsmith.layout_oracle`, `gnnsmith.dense_and_kge_fuzz`)**:
   - Finite forward & backward autograd + second-order `create_graph=True` force derivatives (`-dE/dpos`) + `float64` `gradcheck`
   - Edge-order permutation invariance & within-batch node permutation equivariance ($f(PX, PEP^\top) = Pf(X, E)$)
   - `MessagePassing(flow='source_to_target')` vs. `flow='target_to_source'` mathematical duality ($f_{\text{s2t}}(X, E) = f_{\text{t2s}}(X, E^\top)$)
   - Disjoint graph batching invariance & dense masked-padding non-leakage (`x[~mask]` isolation)
   - Adjacency layout parity (`LongTensor` vs. `EdgeIndex(row/col)` vs. `torch.sparse` COO/CSR) + `Dense*Conv` vs. sparse parity
   - Execution parity (`decomposed_layers=2` vs. `1`, eager vs. `torch.compile(dynamic=True)` vs. `torch.export`, `bfloat16`/`autocast`, `CPU` vs. `MPS` `fp32`/`fp16`, and PyG vs. `mlx-graphs` cross-framework parity)
3. **Python 3.12+ `sys.monitoring` Coverage Loop & Delta-Debugging Reducer (`gnnsmith.coverage_loop`, `gnnsmith.reducer`)**:
   - One-shot `sys.monitoring` line/AST-branch coverage tracking inside `torch_geometric` paired with `whitefox.inspector` AST predicate extraction, plus automated DAG/node/edge delta-debugging reduction.

## CLI Suites

```bash
gnnsmith whitefox
gnnsmith fuzz-ops --seeds 2
gnnsmith fuzz-dag --num-dags 300
gnnsmith fuzz-deep
gnnsmith fuzz-wave5
gnnsmith fuzz-wave7
```
