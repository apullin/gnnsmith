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


if __name__ == "__main__":
    main()
