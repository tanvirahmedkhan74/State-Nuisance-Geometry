from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from state_geometry.data.schema import read_table  # noqa: E402
from state_geometry.data.splits import assign_group_splits, build_dependency_groups  # noqa: E402
from state_geometry.utils.hashing import atomic_write_json, sha256_file  # noqa: E402


def csv_values(text: str | None) -> list[str]:
    return [part.strip() for part in (text or "").split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic connected-group Phase 1 splits.")
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--build-connected-groups", required=True)
    parser.add_argument("--null-group-values-no-edge", action="store_true")
    parser.add_argument("--group", default="dependency_group_id")
    parser.add_argument("--ratios", default="0.70,0.15,0.15")
    parser.add_argument("--stratify", default="")
    parser.add_argument("--stratify-multilabel", default="")
    parser.add_argument("--stratification-objective", default="deterministic_iterative_group_balance")
    parser.add_argument("--report-realized-stratum-mass", action="store_true")
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.null_group_values_no_edge:
        parser.error("--null-group-values-no-edge is mandatory; nulls may never form graph edges")
    if args.stratification_objective != "deterministic_iterative_group_balance":
        parser.error("unsupported stratification objective")

    frame = read_table(args.observations)
    frame[args.group] = build_dependency_groups(
        frame, csv_values(args.build_connected_groups)
    )
    ratios = tuple(float(value) for value in csv_values(args.ratios))
    result = assign_group_splits(
        frame,
        ratios=ratios,
        stratify=csv_values(args.stratify),
        multilabel=csv_values(args.stratify_multilabel),
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.mapping.to_parquet(args.output, index=False)
    report = {
        "source_sha256": sha256_file(args.observations),
        "output_sha256": sha256_file(args.output),
        "seed": args.seed,
        "ratios": ratios,
        "group_fields": csv_values(args.build_connected_groups),
        "null_values_form_edges": False,
        "stratify": csv_values(args.stratify),
        "stratify_multilabel": csv_values(args.stratify_multilabel),
        "realized": result.realized_counts,
    }
    atomic_write_json(args.output.with_suffix(".report.json"), report)
    print(f"Wrote {len(result.mapping)} split rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

