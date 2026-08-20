from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from state_geometry.data.release_views import build_release_views  # noqa: E402
from state_geometry.data.schema import read_table  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build train/validation, redacted, and sealed test views.")
    parser.add_argument("--internal-observations", type=Path, required=True)
    parser.add_argument("--internal-triplets", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output-trainval-observations", type=Path)
    parser.add_argument("--output-feature-inputs", type=Path)
    parser.add_argument("--output-trainval-triplets", type=Path)
    parser.add_argument("--output-sealed-test-targets", type=Path)
    parser.add_argument("--output-sealed-test-triplets", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--feature-input-column-allowlist", type=Path, required=True)
    parser.add_argument("--reject-extra-feature-input-columns", action="store_true")
    parser.add_argument("--label-redact-feature-inputs", action="store_true")
    parser.add_argument("--suppress-test-summaries-until-locked", action="store_true")
    parser.add_argument("--hash-lock-outputs", action="store_true")
    args = parser.parse_args()
    required_flags = (
        args.reject_extra_feature_input_columns,
        args.label_redact_feature_inputs,
        args.suppress_test_summaries_until_locked,
        args.hash_lock_outputs,
    )
    if not all(required_flags):
        parser.error("all fail-closed redaction/sealing/hash flags are mandatory")
    explicit = {
        "analysis_trainval.parquet": args.output_trainval_observations,
        "feature_inputs.parquet": args.output_feature_inputs,
        "triplets_trainval.parquet": args.output_trainval_triplets,
        "sealed_test_targets.parquet": args.output_sealed_test_targets,
        "sealed_test_triplets.parquet": args.output_sealed_test_triplets,
    }
    if args.output_root is None:
        if any(path is None for path in explicit.values()) or args.output_manifest is None:
            parser.error("provide --output-root or every explicit output path plus --output-manifest")
        parents = {path.parent.resolve() for path in explicit.values() if path is not None}
        parents.add(args.output_manifest.parent.resolve())
        if len(parents) != 1:
            parser.error("release views and manifest must share one output directory")
        output_root = parents.pop()
        for expected_name, supplied in explicit.items():
            if supplied is None or supplied.name != expected_name:
                parser.error(f"expected {expected_name}, received {supplied}")
        if args.output_manifest.name != "release_views.json":
            parser.error("output manifest must be named release_views.json")
    else:
        output_root = args.output_root
        if any(path is not None for path in explicit.values()) or args.output_manifest is not None:
            parser.error("do not mix --output-root with explicit output paths")

    allowed = [
        line.strip()
        for line in args.feature_input_column_allowlist.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    manifest = build_release_views(
        read_table(args.internal_observations),
        read_table(args.internal_triplets),
        output_root,
        allowed,
    )
    print(f"Wrote {len(manifest['artifacts'])} hash-locked release views")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
