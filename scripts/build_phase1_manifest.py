from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def runs(labels: list[str]) -> list[tuple[int, int, str]]:
    if not labels:
        return []
    output: list[tuple[int, int, str]] = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            output.append((start, index - 1, labels[start]))
            start = index
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sample_16(start: int) -> list[int]:
    # Deterministic 16-of-30 endpoints-inclusive sampling.
    return [start + round(index * 29 / 15) for index in range(16)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create feature-blind manual-curation candidates.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-revision", default="313d1ef6586571d6ce1fe85581f690c507110fea")
    parser.add_argument("--window-frames", type=int, default=30)
    parser.add_argument("--limit-videos", type=int, help="Smoke-test only; omit for the full candidate manifest.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    tas_root = root / "data" / "annotations" / "tas"
    video_root = root / "data" / "videos" / "0"
    records: list[dict[str, object]] = []
    video_hashes: dict[Path, str] = {}
    tas_files = sorted(tas_root.glob("sequence.*.txt"), key=lambda p: int(p.name.split(".")[1]))
    if args.limit_videos is not None:
        if args.limit_videos <= 0:
            parser.error("--limit-videos must be positive")
        tas_files = tas_files[: args.limit_videos]
    for tas in tas_files:
        sequence = tas.name.split(".")[1]
        video = video_root / f"sequence_{sequence}.mp4"
        if not video.exists():
            continue
        if video not in video_hashes:
            video_hashes[video] = sha256(video)
        labels = tas.read_text(encoding="utf-8").splitlines()
        for run_index, (run_start, run_end, weak_label) in enumerate(runs(labels)):
            length = run_end - run_start + 1
            if length < args.window_frames:
                continue
            window_start = run_start + (length - args.window_frames) // 2
            observation_id = f"egointeract_v0_{sequence}_run{run_index}"
            records.append(
                {
                    "observation_id": observation_id,
                    "dataset_revision": args.dataset_revision,
                    "source_video_id": f"sequence.{sequence}",
                    "media_relpath": video.relative_to(root).as_posix(),
                    "media_type": "video",
                    "physical_object_id": None,
                    "verified_asset_group_id": None,
                    "object_category_manual": None,
                    "state_family": None,
                    "state_label": None,
                    "stable_segment_id": f"sequence.{sequence}.run{run_index}",
                    "transition_id": None,
                    "start_frame": window_start,
                    "end_frame": window_start + args.window_frames - 1,
                    "sampled_frame_indices": sample_16(window_start),
                    "fps": 30.0,
                    "state_observable": False,
                    "identity_verified": False,
                    "box_annotation_relpath": None,
                    "mask_annotation_relpath": None,
                    "mask_available": False,
                    "coordinate_space": "video_1408x1408_source",
                    "nuisance_tags": [],
                    "hand_present": None,
                    "stationary_background_verified": False,
                    "motion_review_status": "pending",
                    "curator_id": None,
                    "reviewer_id": None,
                    "review_status": "pending",
                    "media_sha256": video_hashes[video],
                    "perceptual_hash": None,
                    "duplicate_group_id": None,
                    "tas_weak_action_label": weak_label,
                    "candidate_only": True,
                }
            )
    frame = pd.DataFrame.from_records(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(
        f"Wrote {len(frame)} feature-blind candidates. State/identity/region fields are intentionally blank; curated validation must fail until manual review."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
