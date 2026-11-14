"""Command-line entrypoint for GNNSmith & WhiteFox-GNN fuzzing campaigns."""

from __future__ import annotations

import argparse
from pathlib import Path

from gnnsmith.runner import CampaignRunner
from gnnsmith.whitefox.campaigns import run_all_whitefox_campaigns


def main() -> None:
    parser = argparse.ArgumentParser(description="GNNSmith + WhiteFox GNN library fuzzer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_wf = sub.add_parser("whitefox", help="Run WhiteFox white-box targeted PyG campaigns")
    p_wf.add_argument("--out", type=Path, default=Path("artifacts/whitefox.jsonl"))

    p_ops = sub.add_parser("fuzz-ops", help="Sweep single PyG operators across adversarial topologies")
    p_ops.add_argument("--seeds", type=int, default=2)
    p_ops.add_argument("--no-mps", action="store_true")
    p_ops.add_argument("--out", type=Path, default=Path("artifacts/fuzz_ops.jsonl"))

    p_dag = sub.add_parser("fuzz-dag", help="Run Z3 multi-operator GNN DAG fuzzing campaign")
    p_dag.add_argument("--num-dags", type=int, default=200)
    p_dag.add_argument("--seed", type=int, default=1)
    p_dag.add_argument("--no-mps", action="store_true")
    p_dag.add_argument("--out", type=Path, default=Path("artifacts/fuzz_dag.jsonl"))

    p_deep = sub.add_parser("fuzz-deep", help="Run deep PyTorch 2 compile/AMP/gradcheck/transforms/models/explain/serialization batteries")
    p_deep.add_argument("--out", type=Path, default=Path("artifacts/fuzz_deep.jsonl"))

    p_w5 = sub.add_parser("fuzz-wave5", help="Run bipartite, flow-duality, hetero-Z3, sparse-layout, 3D force-field, modern models, & MLX differential batteries under sys.monitoring coverage")
    p_w5.add_argument("--out", type=Path, default=Path("artifacts/fuzz_wave5.jsonl"))

    p_w7 = sub.add_parser("fuzz-wave7", help="Run Dense GNNs vs Sparse parity, masked-padding leakage, Dense Pooling, KGE, Encoding, & Attention batteries")
    p_w7.add_argument("--out", type=Path, default=Path("artifacts/fuzz_wave7.jsonl"))

    args = parser.parse_args()

    if args.cmd == "whitefox":
        runner = CampaignRunner(out_jsonl=args.out)
        hits = run_all_whitefox_campaigns()
        runner.record(hits)
        print(f"WhiteFox finished: {len(runner.unique_findings)} unique findings.")
        for i, f in enumerate(runner.unique_findings.values(), 1):
            print(f"  [{i:02d}] ({f.oracle} / {f.category}) {f.op_types[0]}: {f.summary}")

    elif args.cmd == "fuzz-ops":
        runner = CampaignRunner(out_jsonl=args.out, check_mps=not args.no_mps)
        runner.run_single_op_sweep(seeds_per_pair=args.seeds)
        print(
            f"Single-op sweep finished: {runner.total_cases} cases | "
            f"{len(runner.unique_findings)} unique findings (raw={runner.total_raw_hits})."
        )
        for i, f in enumerate(runner.unique_findings.values(), 1):
            print(f"  [{i:02d}] ({f.oracle} / {f.category}) {','.join(f.op_types)} [{f.topology_kind}]: {f.summary}")

    elif args.cmd == "fuzz-dag":
        runner = CampaignRunner(out_jsonl=args.out, check_mps=not args.no_mps)
        runner.run_dag_campaign(num_dags=args.num_dags, start_seed=args.seed)
        print(
            f"DAG campaign finished: {runner.total_cases} DAGs | "
            f"{len(runner.unique_findings)} unique findings (raw={runner.total_raw_hits})."
        )
        for i, f in enumerate(runner.unique_findings.values(), 1):
            print(f"  [{i:02d}] ({f.oracle} / {f.category}) {'->'.join(f.op_types)} [{f.topology_kind}]: {f.summary}")

    elif args.cmd == "fuzz-deep":
        from gnnsmith.deep_fuzz import run_all_deep_batteries

        runner = CampaignRunner(out_jsonl=args.out)
        hits = run_all_deep_batteries()
        runner.record(hits)
        print(f"\nDeep Fuzz finished: {len(runner.unique_findings)} unique findings (raw={runner.total_raw_hits}).")
        for i, f in enumerate(runner.unique_findings.values(), 1):
            print(f"  [{i:02d}] ({f.oracle} / {f.category}) {f.op_types[0]}: {f.summary}")

    elif args.cmd == "fuzz-wave5":
        import torch_geometric.edge_index as ei_mod
        import torch_geometric.nn.conv.message_passing as mp_mod
        from gnnsmith.accessory_fuzz import run_all_wave5_campaigns
        from gnnsmith.coverage_loop import PyGCoverageMonitor

        runner = CampaignRunner(out_jsonl=args.out)
        with PyGCoverageMonitor() as mon:
            hits = run_all_wave5_campaigns()
        runner.record(hits)
        cov = mon.find_uncovered_branches([ei_mod, mp_mod])
        print(
            f"\n[Coverage] Executed {cov.lines_executed:,} lines across {cov.files_executed} PyG files | "
            f"Uncovered AST branches in (edge_index, message_passing): {len(cov.uncovered_branches)}/{cov.total_ast_branches_scanned}"
        )
        print(f"Wave 5 finished: {len(runner.unique_findings)} unique findings (raw={runner.total_raw_hits}).")
        for i, f in enumerate(runner.unique_findings.values(), 1):
            print(f"  [{i:02d}] ({f.oracle} / {f.category}) {f.op_types[0]}: {f.summary}")

    elif args.cmd == "fuzz-wave7":
        from gnnsmith.dense_and_kge_fuzz import run_all_wave7_campaigns

        runner = CampaignRunner(out_jsonl=args.out)
        hits = run_all_wave7_campaigns()
        runner.record(hits)
        print(f"\nWave 7 finished: {len(runner.unique_findings)} unique findings (raw={runner.total_raw_hits}).")
        for i, f in enumerate(runner.unique_findings.values(), 1):
            print(f"  [{i:02d}] ({f.oracle} / {f.category}) {f.op_types[0]}: {f.summary}")


if __name__ == "__main__":
    main()
