import time
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv as PyGSAGEConv

import mlx.core as mx
import mlx.nn as mnn
import mlx.optimizers as moptim
from mlx_graphs.data import GraphData
from mlx_graphs.loaders import NeighborLoader
from mlx_graphs.nn import GATv2Conv as MLXGATv2Conv, SAGEConv as MLXSAGEConv
from mlx_cluster import CSRTranspose, build_long_rows, coo_to_csr, spmm_csr


def benchmark_pyg_vs_mlx(num_nodes: int = 100_000, num_edges: int = 1_000_000, channels: int = 256) -> None:
    print(f"\n=== SAGEConv Fwd+Bwd Benchmark ({num_nodes:,} nodes, {num_edges:,} edges, H={channels}) ===")
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((num_nodes, channels), dtype=np.float32)
    src_np = rng.integers(0, num_nodes, size=(num_edges,), dtype=np.int64)
    dst_np = rng.integers(0, num_nodes, size=(num_edges,), dtype=np.int64)

    for dev in ("cpu", "mps"):
        if dev == "mps" and not torch.backends.mps.is_available():
            continue
        x_pt = torch.from_numpy(x_np).to(dev).requires_grad_(True)
        ei_pt = torch.from_numpy(np.stack([src_np, dst_np])).to(dev)
        conv = PyGSAGEConv(channels, channels).to(dev)
        for _ in range(2):
            conv(x_pt, ei_pt).sum().backward()
            conv.zero_grad()
            x_pt.grad = None
        if dev == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            conv(x_pt, ei_pt).sum().backward()
            conv.zero_grad()
            x_pt.grad = None
        if dev == "mps":
            torch.mps.synchronize()
        print(f"  PyG ({dev:3s})            : {(time.perf_counter() - t0) / 5 * 1000:6.2f} ms")

    x_mx = mx.array(x_np)
    w_self = mx.array(rng.standard_normal((channels, channels), dtype=np.float32) * 0.05)
    w_neigh = mx.array(rng.standard_normal((channels, channels), dtype=np.float32) * 0.05)
    try:
        csr = coo_to_csr(mx.array(dst_np), mx.array(src_np), num_nodes)
        rowptr, col_csr, long_rows, csr_t = csr.rowptr, csr.col, csr.long_rows, csr.transpose
    except RuntimeError:
        # Fallback if macOS Metal JIT cache is sandboxed
        perm = np.argsort(dst_np, kind="stable")
        rowptr = mx.array(np.concatenate([[0], np.cumsum(np.bincount(dst_np, minlength=num_nodes))]).astype(np.int64))
        col_csr_np = src_np[perm]
        col_csr = mx.array(col_csr_np)
        perm_t = np.argsort(col_csr_np, kind="stable")
        rowptr_t = mx.array(
            np.concatenate([[0], np.cumsum(np.bincount(col_csr_np, minlength=num_nodes))]).astype(np.int64)
        )
        csr_t = CSRTranspose(
            rowptr=rowptr_t,
            col=mx.array(dst_np[perm][perm_t]),
            perm=mx.array(perm_t),
            long_rows=build_long_rows(rowptr_t),
        )
        long_rows = build_long_rows(rowptr)

    mx.eval(x_mx, w_self, w_neigh, rowptr, col_csr, csr_t.rowptr, csr_t.col)

    def mlx_sage_step(x_in, ws, wn):
        agg = spmm_csr(rowptr, col_csr, x_in, reduce="mean", long_rows=long_rows, transpose=csr_t)
        return (x_in @ ws + agg @ wn).sum()

    grad_fn = mx.value_and_grad(mlx_sage_step, argnums=(0, 1, 2))
    loss, grads = grad_fn(x_mx, w_self, w_neigh)
    mx.eval(loss, *grads)
    t0 = time.perf_counter()
    for _ in range(10):
        loss, grads = grad_fn(x_mx, w_self, w_neigh)
        mx.eval(loss, *grads)
    print(f"  MLX (spmm_csr, gpu)  : {(time.perf_counter() - t0) / 10 * 1000:6.2f} ms")


def demo_mlx_graphs() -> None:
    print("\n=== mlx-graphs NeighborLoader + SAGEConv/GATv2Conv Training Step ===")
    rng = np.random.default_rng(42)
    data = GraphData(
        edge_index=mx.array(rng.integers(0, 2000, size=(2, 10_000), dtype=np.int64)),
        node_features=mx.array(rng.standard_normal((2000, 32), dtype=np.float32)),
        node_labels=mx.array(rng.integers(0, 4, size=(2000,), dtype=np.int32)),
    )

    class MLXGNN(mnn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = MLXSAGEConv(32, 64)
            self.conv2 = MLXGATv2Conv(64, 4, heads=2, concat=False)

        def __call__(self, edge_index: mx.array, x: mx.array) -> mx.array:
            x = mnn.relu(self.conv1(edge_index, x))
            return self.conv2(edge_index, x)

    def run():
        loader = NeighborLoader(data, num_neighbors=[10, 5], batch_size=64, shuffle=True)
        batch = next(iter(loader))
        model = MLXGNN()
        mx.eval(model.parameters())
        opt = moptim.Adam(learning_rate=1e-2)

        def loss_fn(m, b):
            logits = m(b.edge_index, b.node_features)
            return mnn.losses.cross_entropy(
                logits[: b.batch_size], b.node_labels[: b.batch_size], reduction="mean"
            )

        loss, grads = mnn.value_and_grad(model, loss_fn)(model, batch)
        opt.update(model, grads)
        mx.eval(loss, model.parameters())
        print(
            f"  Sampled subgraph: {batch.num_nodes} nodes, {batch.edge_index.shape[1]} edges "
            f"(seeds={batch.batch_size}) | loss={loss.item():.4f}"
        )

    try:
        run()
    except RuntimeError:
        with mx.stream(mx.cpu):
            run()


def main() -> None:
    print(f"torch {torch.__version__} | pyg {torch_geometric.__version__} | mlx {mx.__version__}")
    benchmark_pyg_vs_mlx()
    demo_mlx_graphs()


if __name__ == "__main__":
    main()
