from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from state_geometry.data.schema import (  # noqa: E402
    FEATURE_INPUT_ALLOWED_COLUMNS,
    read_table,
    validate_analysis_manifest,
    validate_curated_manifest,
    validate_feature_inputs,
)
from state_geometry.utils.hashing import atomic_write_json, sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed staged Phase 1 manifest validator.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=("curated", "analysis", "feature-input"), default="curated")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--require-state-labels", action="store_true")
    parser.add_argument("--require-physical-object-ids", action="store_true")
    parser.add_argument("--require-observability", action="store_true")
    parser.add_argument("--require-aligned-regions", action="store_true")
    parser.add_argument("--require-aligned-boxes", action="store_true")
    parser.add_argument("--require-split-and-dependency-groups", action="store_true")
    parser.add_argument("--require-complete-motion-provenance", action="store_true")
    args = parser.parse_args()

    frame = read_table(args.manifest)
    if args.stage == "analysis":
        errors = validate_analysis_manifest(frame)
    elif args.stage == "feature-input":
        errors = validate_feature_inputs(frame, FEATURE_INPUT_ALLOWED_COLUMNS)
    else:
        errors = validate_curated_manifest(
            frame, require_boxes=args.require_aligned_regions or args.require_aligned_boxes
        )
    report = {
        "stage": args.stage,
        "manifest": args.manifest.as_posix(),
        "manifest_sha256": sha256_file(args.manifest),
        "rows": len(frame),
        "status": "fail" if errors else "pass",
        "errors": errors,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_root / "validation_report.json", report)
    if errors:
        print(f"Manifest validation failed closed with {len(errors)} error(s):")
        for error in errors:
            print(f"  - {error}")
        return 2
    print(f"Manifest validation passed: {len(frame)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

