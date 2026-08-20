from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from state_geometry.data.inventory import (  # noqa: E402
    coco_summary,
    filesystem_inventory,
    metadata_revisions,
    video_tas_alignment,
)
from state_geometry.utils.hashing import atomic_write_json  # noqa: E402


FOCUSED_COCO = (
    "data/annotations/hoi/coco_annotations_egointeract.json",
    "data/annotations/hoi/coco_annotations_hand_egointeract.json",
    "data/annotations/nao/coco_annotations_egointeract.json",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the local Phase 1 dataset without modifying it.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--skip-video-probe", action="store_true")
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    if not root.is_dir():
        parser.error(f"dataset root does not exist: {root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    revisions = metadata_revisions(root / ".cache")
    inventory = filesystem_inventory(root)
    report: dict[str, object] = {
        "dataset_root": root.as_posix(),
        "requested_revision": args.dataset_revision,
        "metadata_revisions": revisions,
        "filesystem": inventory,
        "revision_match": (
            args.dataset_revision is None
            or revisions == {args.dataset_revision: sum(revisions.values())}
        ),
        "focused_coco": [],
    }
    for relative in FOCUSED_COCO:
        path = root / relative
        report["focused_coco"].append(coco_summary(path) if path.exists() else {"path": relative, "missing": True})

    if not args.skip_video_probe:
        video_rows, alignment = video_tas_alignment(root)
        pd.DataFrame(video_rows).to_parquet(args.output_root / "codec_report.parquet", index=False)
        report["video_tas_alignment"] = alignment

    atomic_write_json(args.output_root / "inventory.json", report)
    blockers = []
    if inventory["partial_or_lock_files"]:
        blockers.append("cache contains partial/lock files")
    if not report["revision_match"]:
        blockers.append("dataset revision mismatch")
    alignment = report.get("video_tas_alignment", {})
    if isinstance(alignment, dict) and alignment.get("alignment_errors"):
        blockers.append("video/TAS frame-count mismatch")
    if blockers:
        print("Dataset audit failed closed:")
        for blocker in blockers:
            print(f"  - {blocker}")
        return 2
    print(json.dumps({"status": "pass", **inventory}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

