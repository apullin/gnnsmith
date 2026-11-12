"""Adversarial relational graph materializer for GNN fuzzing.

Generates valid PyTorch Geometric `Data` / `HeteroData` batches spanning both
well-behaved graphs and nasty boundary topologies that routinely break GNN layers:
- Trailing isolated nodes (`edge_index.max() < num_nodes - 1`) -> exposes missing `dim_size`
- Leading isolated nodes (node 0 has 0 in/out degree)
- Empty edge sets (`E == 0`, shape `[2, 0]`)
- Single-node graphs (`N == 1`, with and without self-loops)
- Extreme hub / star graphs (all edges funneled into 1 node, or out of 1 node)
- Multigraphs with duplicate parallel edges + self-loops
- Disjoint batches where an intermediate graph in `batch` has 0 edges or 1 node
- Negative / extreme / near-zero node & edge features
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch_geometric.data import Batch, Data, HeteroData

TOPOLOGY_KINDS: Tuple[str, ...] = (
    "erdos_renyi",
    "trailing_isolated",
    "leading_isolated",
    "empty_edges",
    "single_node",
    "in_star_hub",
    "out_star_hub",
    "multigraph_selfloops",
    "disconnected_clusters",
    "skewed_batch",
)


@dataclass
class GraphBundle:
    """Materialized homogeneous graph inputs with full auxiliary tensors."""

    x: torch.Tensor                  # [N, C]
    x_0: torch.Tensor                # [N, C] initial features (for GCN2Conv, etc.)
    pos: torch.Tensor                # [N, 3] 3D coordinates (for PointNetConv, GMMConv, etc.)
    edge_index: torch.Tensor         # [2, E] int64
    edge_weight: torch.Tensor        # [E] positive float (for GCN/Cheb/GraphConv)
    edge_attr: torch.Tensor          # [E, edge_dim] float
    edge_type: torch.Tensor          # [E] int64 in [0, num_relations)
    batch: torch.Tensor              # [N] int64 sorted in [0, B)
    num_nodes: int
    num_graphs: int
    num_relations: int
    topology_kind: str

    def to(self, device: torch.device | str, dtype: Optional[torch.dtype] = None) -> "GraphBundle":
        f_dtype = dtype or self.x.dtype
        return GraphBundle(
            x=self.x.to(device=device, dtype=f_dtype),
            x_0=self.x_0.to(device=device, dtype=f_dtype),
            pos=self.pos.to(device=device, dtype=f_dtype),
            edge_index=self.edge_index.to(device=device),
            edge_weight=self.edge_weight.to(device=device, dtype=f_dtype),
            edge_attr=self.edge_attr.to(device=device, dtype=f_dtype),
            edge_type=self.edge_type.to(device=device),
            batch=self.batch.to(device=device),
            num_nodes=self.num_nodes,
            num_graphs=self.num_graphs,
            num_relations=self.num_relations,
            topology_kind=self.topology_kind,
        )

    def clone_requires_grad(self, grad_edge_attr: bool = True) -> "GraphBundle":
        x = self.x.detach().clone().requires_grad_(True)
        x_0 = self.x_0.detach().clone().requires_grad_(True)
        pos = self.pos.detach().clone().requires_grad_(True)
        ew = self.edge_weight.detach().clone().requires_grad_(True)
        ea = self.edge_attr.detach().clone().requires_grad_(grad_edge_attr)
        return GraphBundle(
            x=x,
            x_0=x_0,
            pos=pos,
            edge_index=self.edge_index.clone(),
            edge_weight=ew,
            edge_attr=ea,
            edge_type=self.edge_type.clone(),
            batch=self.batch.clone(),
            num_nodes=self.num_nodes,
            num_graphs=self.num_graphs,
            num_relations=self.num_relations,
            topology_kind=self.topology_kind,
        )

    def to_pyg_data_list(self) -> List[Data]:
        """Split a batched GraphBundle into per-graph `Data` objects (when edges respect batch)."""
        out: List[Data] = []
        for b in range(self.num_graphs):
            node_mask = self.batch == b
            n_b = int(node_mask.sum().item())
            if n_b == 0:
                continue
            global_idx = torch.nonzero(node_mask, as_tuple=False).view(-1)
            remap = torch.full((self.num_nodes,), -1, dtype=torch.long, device=self.x.device)
            remap[global_idx] = torch.arange(n_b, dtype=torch.long, device=self.x.device)
            if self.edge_index.numel() > 0:
                e_mask = node_mask[self.edge_index[0]] & node_mask[self.edge_index[1]]
                ei_b = remap[self.edge_index[:, e_mask]]
                ew_b = self.edge_weight[e_mask]
                ea_b = self.edge_attr[e_mask]
                et_b = self.edge_type[e_mask]
            else:
                ei_b = torch.empty((2, 0), dtype=torch.long, device=self.x.device)
                ew_b = torch.empty((0,), dtype=self.x.dtype, device=self.x.device)
                ea_b = torch.empty((0, self.edge_attr.size(-1)), dtype=self.x.dtype, device=self.x.device)
                et_b = torch.empty((0,), dtype=torch.long, device=self.x.device)
            out.append(
                Data(
                    x=self.x[node_mask],
                    x_0=self.x_0[node_mask],
                    pos=self.pos[node_mask],
                    edge_index=ei_b,
                    edge_weight=ew_b,
                    edge_attr=ea_b,
                    edge_type=et_b,
                    num_nodes=n_b,
                )
            )
        return out


def generate_graph_bundle(
    *,
    num_nodes: int,
    num_edges: int,
    in_channels: int,
    num_graphs: int = 2,
    edge_dim: int = 4,
    num_relations: int = 4,
    topology_kind: str = "erdos_renyi",
    seed: int = 0,
    feature_mode: str = "standard",
    dtype: torch.dtype = torch.float32,
) -> GraphBundle:
    """Deterministically materialize a GraphBundle for a specific topology class."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    if topology_kind == "single_node":
        num_nodes = 1
        num_graphs = 1

    num_nodes = max(1, num_nodes)
    num_graphs = max(1, min(num_graphs, num_nodes))

    # Build sorted batch assignment [N] covering [0, num_graphs)
    if num_graphs == 1:
        batch = torch.zeros(num_nodes, dtype=torch.long)
    elif topology_kind == "skewed_batch" and num_nodes >= 4 and num_graphs >= 2:
        # Graph 0 gets 1 node, last graph gets most nodes
        batch = torch.full((num_nodes,), num_graphs - 1, dtype=torch.long)
        for g in range(num_graphs - 1):
            batch[g] = g
    else:
        # Guarantee at least 1 node per graph, rest random sorted
        base = torch.arange(num_graphs, dtype=torch.long)
        extra = torch.randint(0, num_graphs, (num_nodes - num_graphs,), generator=gen, dtype=torch.long)
        batch = torch.cat([base, extra]).sort().values

    # Build edge_index within each batch component so disjoint batching holds!
    src_parts: List[torch.Tensor] = []
    dst_parts: List[torch.Tensor] = []

    if topology_kind == "empty_edges" or num_edges == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    elif topology_kind == "single_node":
        if num_edges > 0 and seed % 2 == 0:
            edge_index = torch.zeros((2, 1), dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edges_per_graph = max(1, num_edges // num_graphs)
        for g in range(num_graphs):
            nodes_g = torch.nonzero(batch == g, as_tuple=False).view(-1)
            n_g = nodes_g.numel()
            if n_g == 0:
                continue
            if topology_kind == "skewed_batch" and g == 0:
                # Leave graph 0 with 0 edges (1 isolated node in a multi-graph batch!)
                continue
            if topology_kind == "trailing_isolated" and n_g >= 3:
                active = nodes_g[:-2]
            elif topology_kind == "leading_isolated" and n_g >= 3:
                active = nodes_g[1:]
            else:
                active = nodes_g

            n_act = active.numel()
            if n_act == 0:
                continue

            if topology_kind == "in_star_hub":
                hub = active[0]
                src_parts.append(active)
                dst_parts.append(hub.expand(n_act))
            elif topology_kind == "out_star_hub":
                hub = active[0]
                src_parts.append(hub.expand(n_act))
                dst_parts.append(active)
            elif topology_kind == "multigraph_selfloops":
                s = active[torch.randint(0, n_act, (edges_per_graph,), generator=gen)]
                d = active[torch.randint(0, n_act, (edges_per_graph,), generator=gen)]
                # Duplicate edges + add self-loops
                src_parts.extend([s, s, active])
                dst_parts.extend([d, d, active])
            elif topology_kind == "disconnected_clusters" and n_act >= 4:
                half = n_act // 2
                c1, c2 = active[:half], active[half:]
                s1 = c1[torch.randint(0, c1.numel(), (max(1, edges_per_graph // 2),), generator=gen)]
                d1 = c1[torch.randint(0, c1.numel(), (max(1, edges_per_graph // 2),), generator=gen)]
                src_parts.append(s1)
                dst_parts.append(d1)
                # Leave c2 completely edge-free!
            else:
                s = active[torch.randint(0, n_act, (edges_per_graph,), generator=gen)]
                d = active[torch.randint(0, n_act, (edges_per_graph,), generator=gen)]
                src_parts.append(s)
                dst_parts.append(d)

        if src_parts:
            edge_index = torch.stack([torch.cat(src_parts), torch.cat(dst_parts)], dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

    actual_e = edge_index.size(1)

    if feature_mode == "all_negative":
        x = -torch.rand((num_nodes, in_channels), generator=gen, dtype=dtype) - 0.5
        edge_attr = -torch.rand((actual_e, edge_dim), generator=gen, dtype=dtype) - 0.5
    elif feature_mode == "near_zero":
        x = torch.randn((num_nodes, in_channels), generator=gen, dtype=dtype) * 1e-6
        edge_attr = torch.randn((actual_e, edge_dim), generator=gen, dtype=dtype) * 1e-6
    elif feature_mode == "large_magnitude":
        x = torch.randn((num_nodes, in_channels), generator=gen, dtype=dtype) * 50.0
        edge_attr = torch.randn((actual_e, edge_dim), generator=gen, dtype=dtype) * 50.0
    else:
        x = torch.randn((num_nodes, in_channels), generator=gen, dtype=dtype)
        edge_attr = torch.randn((actual_e, edge_dim), generator=gen, dtype=dtype)

    x_0 = torch.randn((num_nodes, in_channels), generator=gen, dtype=dtype)
    pos = torch.randn((num_nodes, 3), generator=gen, dtype=dtype)
    # Keep edge_weight strictly positive in (0.1, 2.1) so spectral/GCN norm layers stay mathematically valid
    edge_weight = torch.rand((actual_e,), generator=gen, dtype=dtype) * 2.0 + 0.1
    edge_type = (
        torch.randint(0, max(1, num_relations), (actual_e,), generator=gen, dtype=torch.long)
        if actual_e > 0
        else torch.empty((0,), dtype=torch.long)
    )

    return GraphBundle(
        x=x,
        x_0=x_0,
        pos=pos,
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_attr=edge_attr,
        edge_type=edge_type,
        batch=batch,
        num_nodes=num_nodes,
        num_graphs=num_graphs,
        num_relations=max(1, num_relations),
        topology_kind=topology_kind,
    )
