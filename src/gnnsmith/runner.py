"""Fuzzing campaign orchestrator with signature deduplication and JSONL reporting."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Type

from gnnsmith.graph_gen import synthesize_gnn_dag
from gnnsmith.materialize import materialize_ir
from gnnsmith.ops import ALL_OPS
from gnnsmith.ops.base import AbsGNNOp
from gnnsmith.ops.pyg_norm_pool import GlobalPoolReadoutOp
from gnnsmith.oracles import BugFinding, run_all_oracles
from gnnsmith.topology import TOPOLOGY_KINDS, generate_graph_bundle


class CampaignRunner:
    """Runs Z3 DAG and single-op fuzzing campaigns while deduplicating findings."""

    def __init__(self, out_jsonl: Optional[Path] = None, check_mps: bool = True) -> None:
        self.out_jsonl = out_jsonl
        self.check_mps = check_mps
        self.unique_findings: Dict[str, BugFinding] = {}
        self.total_cases = 0
        self.total_raw_hits = 0

    def record(self, findings: List[BugFinding]) -> int:
        new_count = 0
        for f in findings:
            self.total_raw_hits += 1
            key = f.signature_key or f"{f.oracle}|{f.category}|{','.join(f.op_types)}|{f.summary}"
            if key not in self.unique_findings:
                self.unique_findings[key] = f
                new_count += 1
                if self.out_jsonl is not None:
                    self.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
                    with self.out_jsonl.open("a", encoding="utf-8") as fp:
                        fp.write(json.dumps(asdict(f)) + "\n")
        return new_count

    def run_single_op_sweep(
        self,
        *,
        ops: Optional[List[Type[AbsGNNOp]]] = None,
        topologies: Optional[List[str]] = None,
        seeds_per_pair: int = 2,
    ) -> List[BugFinding]:
        """Sweep individual operators across adversarial topologies to isolate root causes cleanly."""
        target_ops = ops or [op for op in ALL_OPS if op is not GlobalPoolReadoutOp]
        target_topos = topologies or list(TOPOLOGY_KINDS)

        seed_counter = 10_000
        for op_cls in target_ops:
            for topo in target_topos:
                if op_cls.capabilities.mutates_topology and topo == "single_node":
                    continue
                for s in range(seeds_per_pair):
                    seed_counter += 1
                    try:
                        ir = synthesize_gnn_dag(
                            seed=seed_counter,
                            min_ops=1,
                            max_ops=1,
                            allowed_ops=[op_cls],
                            topology_kind=topo,
                            include_readout_prob=0.0,
                        )
                    except Exception:
                        continue

                    scope0 = ir.scopes[0]
                    in_c = int(ir.values[ir.input_value].channels)
                    feat_mode = "all_negative" if s == 1 else "standard"
                    bundle = generate_graph_bundle(
                        num_nodes=int(scope0.num_nodes),
                        num_edges=int(scope0.num_edges),
                        in_channels=in_c,
                        num_graphs=int(scope0.num_graphs),
                        edge_dim=scope0.edge_dim,
                        num_relations=scope0.num_relations,
                        topology_kind=topo,
                        seed=seed_counter,
                        feature_mode=feat_mode,
                    )
                    model = materialize_ir(ir, seed=seed_counter)
                    self.total_cases += 1
                    hits = run_all_oracles(ir, model, bundle, check_mps=self.check_mps)
                    self.record(hits)

        return list(self.unique_findings.values())

    def run_dag_campaign(
        self,
        *,
        num_dags: int = 200,
        start_seed: int = 1,
        min_ops: int = 2,
        max_ops: int = 6,
    ) -> List[BugFinding]:
        """Synthesize and differentially test random Z3-solved multi-branch GNN DAGs."""
        t0 = time.perf_counter()
        for idx in range(num_dags):
            seed = start_seed + idx
            try:
                ir = synthesize_gnn_dag(seed=seed, min_ops=min_ops, max_ops=max_ops)
            except Exception:
                continue

            scope0 = ir.scopes[0]
            in_c = int(ir.values[ir.input_value].channels)
            feat_mode = ("standard", "all_negative", "near_zero", "large_magnitude")[idx % 4]
            bundle = generate_graph_bundle(
                num_nodes=int(scope0.num_nodes),
                num_edges=int(scope0.num_edges),
                in_channels=in_c,
                num_graphs=int(scope0.num_graphs),
                edge_dim=scope0.edge_dim,
                num_relations=scope0.num_relations,
                topology_kind=ir.topology_kind,
                seed=seed,
                feature_mode=feat_mode,
            )
            model = materialize_ir(ir, seed=seed)
            self.total_cases += 1
            hits = run_all_oracles(ir, model, bundle, check_mps=self.check_mps, check_decomposed=False)
            self.record(hits)

            if (idx + 1) % 50 == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"[DAG {idx + 1}/{num_dags}] elapsed={elapsed:.1f}s | "
                    f"unique_findings={len(self.unique_findings)} (raw={self.total_raw_hits})"
                )

        return list(self.unique_findings.values())
