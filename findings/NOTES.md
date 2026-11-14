# Continuation & Handoff Notes

## 1. Repositories & Branches

- **Fuzzer Framework:** [`https://github.com/apullin/gnnsmith`](https://github.com/apullin/gnnsmith)
  - `master`: Clean fuzzer package only (`whitefox`, `fuzz-ops`, `fuzz-dag`, `fuzz-deep`).
  - `agent_work`: Everything on `master` **plus** `findings/BUGS.md` (44 bugs), `findings/NOTES.md` (this file), `findings/pyg_vs_mlx_bench.py`, and `repros/repro_all.py` (47 deterministic reproducers).
- **PyTorch Geometric Fixes:** [`https://github.com/apullin/pytorch_geometric`](https://github.com/apullin/pytorch_geometric)
  - `mega-bugfix`: 7 stacked commits atop `pyg-team/pytorch_geometric:master` (`2.9.0-dev`) fixing all **44 distinct bugs** (`47/47` reproducers resolved) with upstream-style `pytest` unit tests and zero scratch files.

## 2. Reconstructing Workspace on Another Machine

```bash
mkdir gnn && cd gnn
git clone https://github.com/apullin/gnnsmith.git
git -C gnnsmith fetch origin agent_work

git clone --filter=blob:none https://github.com/apullin/pytorch_geometric.git -b mega-bugfix
git -C pytorch_geometric remote add upstream https://github.com/pyg-team/pytorch_geometric.git

uv venv --python 3.12  # or 3.13
uv pip install --python .venv torch ogb networkx yacs lightning z3-solver pytest
uv pip install --python .venv -e ./pytorch_geometric -e ./gnnsmith

# Run fuzzer suites:
.venv/bin/gnnsmith whitefox
.venv/bin/gnnsmith fuzz-ops --seeds 2
.venv/bin/gnnsmith fuzz-dag --num-dags 300
.venv/bin/gnnsmith fuzz-deep

# Verify 47/47 reproducers against upstream/master vs mega-bugfix:
git -C gnnsmith show origin/agent_work:repros/repro_all.py > /tmp/repro_all.py
.venv/bin/python /tmp/repro_all.py   # -> 0/35 reproduced on mega-bugfix (47/47 on upstream/master)
```

*(On Apple Silicon macOS, optionally also install `mlx`, `mlx-cluster`, `mlxvs`, and `mlx-graphs @ git+https://github.com/mlx-graphs/mlx-graphs@main`.)*

---

## 3. Bug Count Breakdown (32 Distinct Root-Cause Bugs / 35 Reproducer Sub-Cases)

- **Epoch 1 (WhiteFox + Single-Op Matrix + Z3 Multi-Op DAGs):** **22 root-cause bugs** (`BUG-001`..`BUG-022`, with `BUG-007` split into `007a` tuple crash, `007b` weight mismatch crash, `007c` `AGNNConv` silent wrong math = 24 sub-cases).
- **Epoch 2 (During `pytest` + TorchDynamo User-Code Tracing):** **1 additional bug** (`BUG-032`: `Index.__new__` / `EdgeIndex.__new__` crashing TorchDynamo `is_fake()` when traced directly before `_data` is attached).
- **Epoch 3 (Deep Pass: `torch.compile`, AMP/`bf16`/MPS `fp16`, `gradcheck`, Transforms, Models, Explain, PT 2.6+ `weights_only=True`):** **9 additional root-cause bugs** (`BUG-023`..`BUG-031`, with `BUG-025` split into `025a` non-`fp32` crash and `025b` silent dropped weights in backward when `perm is None` = 10 sub-cases).
- **Total:** **44 distinct root-cause bugs** (**35** executable checks in `repros/repro_all.py`), all **fixed** on `apullin/pytorch_geometric:mega-bugfix`.

---

## 4. Apple Silicon Performance Summary (M2 Ultra, 128 GB Unified RAM)

Measured on `SAGEConv` (`100,000` nodes, `1,000,000` edges, `H=256`, full fwd+bwd including `dx`):
- **PyG (`cpu`):** `108.6 ms`
- **PyG (`mps`):** `127.4 ms` (bottlenecked by unfused `[E, H]` message materialization + PyTorch MPS `scatter_reduce_(mean)` expanding 2D `[E, H]` `int64` indices at `132.8 ms` vs `8.3 ms` CPU)
- **MLX (`mlx_cluster.spmm_csr` + MLX matmul on `gpu`):** **`10.4 ms`** (**12.3× faster** than PyG MPS; `mean` aggregation kernel alone is **`1.76 ms`** vs `138.3 ms` on MPS = **78.6× faster**).

---

## 5. Next High-Value Directions

1. **Splitting `mega-bugfix` into Upstream PRs (`pyg-team/pytorch_geometric`):**
   - The 7 commits on `mega-bugfix` are already grouped cleanly by subsystem (`index`/`edge_index`, `message_passing`, `nn/conv`, `nn/aggr`, `utils`/`to_hetero`, AMP/transforms/`GraphUNet`/`weights_only` serialization) and each includes `pytest` unit tests—easy to cherry-pick into focused upstream PRs.
2. **Linux + CUDA Fuzz Pass (if moving to a CUDA machine):**
   - With `pyg-lib`, `torch-scatter`, `torch-sparse`, and `torch-cluster` installed on CUDA, run `gnnsmith` comparing **CPU fallback vs. CUDA C++ extension kernels** (`WITH_PYG_LIB=True/False`, `WITH_TORCH_SCATTER=True/False`, `CuGraph*` layers, `FusedGATConv`, `NeighborLoader`, `LinkNeighborLoader`, `HGTLoader`, `torch.compile` with Triton Inductor kernels, and `fp16`/`bf16` CUDA AMP).
3. **Round 2 — Fuzzing `mlx-graphs` + `mlx-cluster` (using patched PyG as reference oracle):**
   - Two real bugs already spotted in `mlx-graphs` (`main`) during source inspection:
     - `mlx_graphs/utils/scatter.py`: `scatter(..., aggr="max" | "min")` initializes output buffers with `mx.zeros(...)` instead of `-inf` / `+inf`, silently clamping all-negative max aggregations (and all-positive min aggregations) to `0.0`.
     - `mlx_graphs/nn/message_passing.py`: `MessagePassing.propagate` mutates `self.num_nodes` on `self`, breaking pure functional `@mx.compile` tracing.
