"""Symbolic Graph Intermediate Representation (GraphIR) for GNN computation DAGs.

Unlike dense DNN fuzzers (where AbsTensor only tracks [shape, dtype]), GNNSmith
tracks relational *GraphScope* domains (N_nodes, E_edges, B_graphs, R_relations,
C_edge) alongside node/graph feature tensors. Topology-mutating operators
(e.g., TopKPooling, SAGPooling, EdgePooling, subgraph) transition to a fresh
child GraphScope so downstream operators stay relationally sound.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TensorRole(str, Enum):
    NODE_FEAT = "node_feat"      # [N, C]
    GRAPH_FEAT = "graph_feat"    # [B, C]
    EDGE_INDEX = "edge_index"    # [2, E] in [0, N)
    EDGE_ATTR = "edge_attr"      # [E, C_e]
    EDGE_WEIGHT = "edge_weight"  # [E]
    EDGE_TYPE = "edge_type"      # [E] in [0, R)
    BATCH = "batch"              # [N] monotonic in [0, B)


@dataclass
class GraphScope:
    """Represents a symbolic or concrete relational graph topology domain."""

    scope_id: int
    num_nodes: int | Any        # Concrete int or z3.ArithRef
    num_edges: int | Any        # Concrete int or z3.ArithRef
    num_graphs: int | Any       # Concrete int or z3.ArithRef (batch size B)
    num_relations: int = 4      # R for relational GNNs (RGCN, RGAT, FiLM)
    edge_dim: int = 8           # Feature dim of edge_attr [E, edge_dim]
    parent_scope: Optional[int] = None


@dataclass
class SymValue:
    """A typed tensor value flowing through the GNN computation DAG."""

    name: str
    role: TensorRole
    scope_id: int
    channels: int | Any         # Concrete int or z3.ArithRef (for NODE_FEAT / GRAPH_FEAT)
    dtype: str = "float32"


@dataclass
class OpInst:
    """A single operator node in the GNN DAG."""

    inst_id: int
    op_type: str
    inputs: List[str]           # Names of input SymValues
    output: str                 # Name of output SymValue
    in_scope: int
    out_scope: int
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphIR:
    """Complete GNN computation DAG over one or more relational GraphScopes."""

    scopes: Dict[int, GraphScope] = field(default_factory=dict)
    values: Dict[str, SymValue] = field(default_factory=dict)
    insts: List[OpInst] = field(default_factory=list)
    input_value: str = "x0"
    output_value: str = "x0"
    topology_kind: str = "erdos_renyi"
    seed: int = 0

    def summary(self) -> str:
        ops = " -> ".join(f"{i.op_type}({','.join(i.inputs)}->{i.output})" for i in self.insts)
        return f"GraphIR[topo={self.topology_kind}, scopes={len(self.scopes)}, ops={len(self.insts)}]: {ops}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "topology_kind": self.topology_kind,
            "input_value": self.input_value,
            "output_value": self.output_value,
            "scopes": {
                str(k): {
                    "scope_id": v.scope_id,
                    "num_nodes": int(v.num_nodes),
                    "num_edges": int(v.num_edges),
                    "num_graphs": int(v.num_graphs),
                    "num_relations": int(v.num_relations),
                    "edge_dim": int(v.edge_dim),
                    "parent_scope": v.parent_scope,
                }
                for k, v in self.scopes.items()
            },
            "values": {
                k: {
                    "name": v.name,
                    "role": v.role.value,
                    "scope_id": v.scope_id,
                    "channels": int(v.channels),
                    "dtype": v.dtype,
                }
                for k, v in self.values.items()
            },
            "insts": [
                {
                    "inst_id": i.inst_id,
                    "op_type": i.op_type,
                    "inputs": i.inputs,
                    "output": i.output,
                    "in_scope": i.in_scope,
                    "out_scope": i.out_scope,
                    "params": i.params,
                }
                for i in self.insts
            ],
        }
