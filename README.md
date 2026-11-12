# GNNSmith

Z3-constrained Graph Neural Network DAG synthesis, metamorphic/differential oracles, and WhiteFox white-box testing for PyTorch Geometric and graph ML libraries.

## Overview

Dense DNN fuzzers like `NNSmith` model tensors purely by `(shape, dtype)` with unconstrained values—an assumption broken by Graph Neural Networks where index values (`edge_index`, `batch`, `ptr`), sparse metadata (`EdgeIndex` sort order, `sparse_size`), and dynamic topology-mutating pooling layers (`TopKPooling`, `SAGPooling`, `EdgePooling`) couple tensor values directly to relational graph domains.

**GNNSmith** combines:
1. **Z3 Relational GraphIR Synthesis (`gnnsmith.graph_gen`)**: Symbolic solving over channel widths, multi-head/concat constraints, PNA/MixHop multipliers, skip/residual merges, and multi-scope topology transitions across 55+ PyG operators and 10 adversarial graph topologies.
2. **7 Zero-Label Metamorphic & Differential Oracles (`gnnsmith.oracles`)**:
   - Finite forward & backward autograd (`x.grad`, parameter grads)
   - Edge-order permutation invariance
   - Within-batch node permutation equivariance ($f(PX, PEP^\top) = Pf(X, E)$)
   - Disjoint graph batching invariance ($[f(G_0), f(G_1)] = f(\text{Batch}([G_0, G_1]))$)
   - `EdgeIndex` tensor subclass vs. plain `LongTensor` parity
   - `MessagePassing(decomposed_layers=2)` vs. `decomposed_layers=1` parity
   - Device parity (`CPU` vs. Apple Silicon `MPS` forward + backward)
3. **WhiteFox Source-Guided Campaigns (`gnnsmith.whitefox`)**: AST branch analysis + surgical trigger synthesis for `EdgeIndex` dispatch, `to_hetero` / `to_hetero_with_bases` `torch.fx` transforms, `nn.aggr` modules, hierarchical `nn.pool` layers, and `torch_geometric.utils`.

## Quickstart

```bash
gnnsmith whitefox
gnnsmith fuzz-ops --seeds 2
gnnsmith fuzz-dag --num-dags 300
```

