from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from state_geometry.utils.hashing import atomic_write_json, sha256_file  # noqa: E402
from state_geometry.features.preprocess import (  # noqa: E402
    eval_resize_crop_geometry,
    rasterized_box_token_mass,
    transform_xywh_box,
)


def load_coco(path: Path) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    images = {str(image["id"]): image for image in data["images"]}
    return images, data["annotations"]


def sequence_frame(filename: str) -> tuple[str, int]:
    stem = Path(filename).stem
    sequence, frame = stem.rsplit("_", 1)
    return sequence, int(frame)


def dataset_root(path: Path) -> Path:
    resolved = path.resolve()
    for candidate in (resolved, *resolved.parents):
        if candidate.name == "EgoInteract":
            return candidate
    raise ValueError(f"cannot locate EgoInteract root above {path}")


def valid_polygons(segmentation: object) -> list[list[float]]:
    if not isinstance(segmentation, list):
        return []
    return [
        component
        for component in segmentation
        if isinstance(component, list) and len(component) >= 6 and len(component) % 2 == 0
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the explicitly confounded EgoInteract interaction-phase proxy.")
    parser.add_argument("--nao-json", type=Path, required=True)
    parser.add_argument("--nao-frame-root", type=Path, required=True)
    parser.add_argument("--hoi-json", type=Path, required=True)
    parser.add_argument("--hoi-frame-root", type=Path, required=True)
    parser.add_argument("--phase-labels", default="pre_contact,contact")
    parser.add_argument("--min-observations-per-phase", type=int, default=2)
    parser.add_argument("--require-contacting-hand", action="store_true")
    parser.add_argument("--deduplicate-by-sha256", action="store_true")
    parser.add_argument("--reject-cross-label-hash-conflicts", action="store_true")
    parser.add_argument("--encoder-crop-size", type=int, default=384)
    parser.add_argument("--minimum-box-token-mass", type=float, default=1.0)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    phases = [part.strip() for part in args.phase_labels.split(",")]
    if phases != ["pre_contact", "contact"]:
        parser.error("this proxy only supports pre_contact,contact")
    if not (args.require_contacting_hand and args.deduplicate_by_sha256 and args.reject_cross_label_hash_conflicts):
        parser.error("contact-hand, SHA deduplication, and cross-label conflict rejection are mandatory")
    if args.encoder_crop_size <= 0 or args.minimum_box_token_mass <= 0:
        parser.error("encoder crop size and minimum box-token mass must be positive")
    root = dataset_root(args.nao_frame_root)
    if dataset_root(args.hoi_frame_root) != root:
        parser.error("NAO and HOI assets must share one EgoInteract root")
    common_filenames = sorted(
        {path.name for path in args.nao_frame_root.glob("*.jpg")}
        & {path.name for path in args.hoi_frame_root.glob("*.jpg")}
    )
    exact_cross_directory_pairs = sum(
        sha256_file(args.nao_frame_root / name) == sha256_file(args.hoi_frame_root / name)
        for name in common_filenames
    )

    nao_images, nao_annotations = load_coco(args.nao_json)
    hoi_images, hoi_annotations = load_coco(args.hoi_json)
    contact_images = {
        str(annotation["image_id"])
        for annotation in hoi_annotations
        if int(annotation.get("category_id", -1)) == 1
        and int(annotation.get("isincontact", -1)) == 1
    }

    rows: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    hash_cache: dict[Path, str] = {}

    def add(annotation: dict[str, object], image: dict[str, object], phase: str, frame_root: Path) -> None:
        filename = str(image["file_name"])
        sequence, frame = sequence_frame(filename)
        media = (frame_root / filename).resolve()
        if not media.is_file():
            exclusions.append({"phase": phase, "sequence_id": sequence, "frame_id": frame, "reason": "missing_media"})
            return
        source_box = tuple(float(value) for value in annotation["bbox"])
        geometry = eval_resize_crop_geometry(
            int(float(image["height"])),
            int(float(image["width"])),
            args.encoder_crop_size,
        )
        try:
            transformed_box = transform_xywh_box(source_box, geometry)
        except ValueError:
            exclusions.append(
                {
                    "phase": phase,
                    "sequence_id": sequence,
                    "frame_id": frame,
                    "reason": "box_outside_encoder_crop",
                }
            )
            return
        token_mass = rasterized_box_token_mass(transformed_box, patch_size=16)
        if token_mass < args.minimum_box_token_mass:
            exclusions.append(
                {
                    "phase": phase,
                    "sequence_id": sequence,
                    "frame_id": frame,
                    "reason": "box_below_minimum_token_mass",
                    "encoder_box_token_mass": token_mass,
                }
            )
            return
        if media not in hash_cache:
            hash_cache[media] = sha256_file(media)
        instance = str(annotation.get("solo_instance_id"))
        polygons = valid_polygons(annotation.get("segmentation")) if phase == "contact" else []
        rows.append(
            {
                "observation_id": f"proxy_{phase}_{sequence}_{frame}_{instance}",
                "source_family": "egointeract_enigma",
                "sequence_id": sequence,
                "frame_id": frame,
                "media_relpath": media.relative_to(root).as_posix(),
                "media_type": "image",
                "image_width": int(float(image["width"])),
                "image_height": int(float(image["height"])),
                "category_id": "object",
                "solo_instance_id": instance,
                "asset_proxy_id": f"egointeract_enigma:object:{instance}",
                "interaction_phase": phase,
                "bbox_xywh": list(source_box),
                "encoder_crop_xywh": list(transformed_box),
                "encoder_box_token_mass": token_mass,
                "encoder_crop_size": args.encoder_crop_size,
                "encoder_geometry_version": "short_side_256_over_224_bilinear_center_crop_v1",
                "valid_segmentation_json": json.dumps(polygons, separators=(",", ":")) if polygons else None,
                "mask_available": bool(polygons),
                "contacting_hand_required": True,
                "media_sha256": hash_cache[media],
                "duplicate_group_id": f"sha256:{hash_cache[media]}",
                "normalized_time_proxy": frame,
            }
        )

    for annotation in nao_annotations:
        image = nao_images[str(annotation["image_id"])]
        add(annotation, image, "pre_contact", args.nao_frame_root)
    for annotation in hoi_annotations:
        if int(annotation.get("category_id", -1)) != 2:
            continue
        image_id = str(annotation["image_id"])
        if image_id not in contact_images:
            image = hoi_images[image_id]
            sequence, frame = sequence_frame(str(image["file_name"]))
            exclusions.append({"phase": "contact", "sequence_id": sequence, "frame_id": frame, "reason": "no_positive_contacting_hand"})
            continue
        add(annotation, hoi_images[image_id], "contact", args.hoi_frame_root)

    frame = pd.DataFrame.from_records(rows)
    conflict_hashes = set(
        frame.groupby("media_sha256")["interaction_phase"].nunique().loc[lambda values: values > 1].index
    )
    if conflict_hashes:
        conflict_rows = frame[frame["media_sha256"].isin(conflict_hashes)]
        exclusions.extend(
            {"phase": row.interaction_phase, "sequence_id": row.sequence_id, "frame_id": int(row.frame_id), "reason": "cross_phase_exact_duplicate"}
            for row in conflict_rows.itertuples()
        )
        frame = frame[~frame["media_sha256"].isin(conflict_hashes)].copy()

    key_columns = ["sequence_id", "asset_proxy_id"]
    counts = frame.groupby(key_columns + ["interaction_phase"]).size().unstack(fill_value=0)
    eligible = counts[
        (counts.get("pre_contact", 0) >= args.min_observations_per_phase)
        & (counts.get("contact", 0) >= args.min_observations_per_phase)
    ].index
    eligible_keys = set(eligible.tolist())
    keep = frame.apply(lambda row: (row["sequence_id"], row["asset_proxy_id"]) in eligible_keys, axis=1)
    for row in frame.loc[~keep].itertuples():
        exclusions.append({"phase": row.interaction_phase, "sequence_id": row.sequence_id, "frame_id": int(row.frame_id), "reason": "insufficient_phase_support"})
    frame = frame.loc[keep].sort_values(key_columns + ["interaction_phase", "frame_id"]).reset_index(drop=True)

    triplets: list[dict[str, object]] = []
    for (sequence, asset), group in frame.groupby(key_columns, sort=True):
        pre = group[group["interaction_phase"] == "pre_contact"].sort_values("frame_id")
        contact = group[group["interaction_phase"] == "contact"].sort_values("frame_id")
        pre_ids = pre["observation_id"].tolist()
        contact_ids = contact["observation_id"].tolist()
        triplets.extend(
            [
                {
                    "triplet_id": f"proxy_{sequence}_{asset}_pre_anchor",
                    "anchor_observation_id": pre_ids[0],
                    "nuisance_observation_id": pre_ids[1],
                    "state_observation_id": contact_ids[0],
                    "asset_proxy_id": asset,
                    "sequence_id": sequence,
                    "anchor_phase": "pre_contact",
                    "state_target": "contact",
                    "interpretation": "interaction_phase_proxy_not_physical_state",
                },
                {
                    "triplet_id": f"proxy_{sequence}_{asset}_contact_anchor",
                    "anchor_observation_id": contact_ids[0],
                    "nuisance_observation_id": contact_ids[1],
                    "state_observation_id": pre_ids[0],
                    "asset_proxy_id": asset,
                    "sequence_id": sequence,
                    "anchor_phase": "contact",
                    "state_target": "pre_contact",
                    "interpretation": "interaction_phase_proxy_not_physical_state",
                },
            ]
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_root / "observations.parquet", index=False)
    pd.DataFrame.from_records(triplets).to_parquet(args.output_root / "proxy_triplets.parquet", index=False)
    pd.DataFrame.from_records(exclusions).to_parquet(args.output_root / "exclusions.parquet", index=False)
    summary = {
        "label_namespace": "interaction_phase",
        "physical_state_claim_allowed": False,
        "observations": len(frame),
        "triplets": len(triplets),
        "eligible_sequence_asset_keys": len(eligible_keys),
        "unique_asset_proxy_ids": int(frame["asset_proxy_id"].nunique()),
        "phase_counts": dict(Counter(frame["interaction_phase"])),
        "mask_available_contact_observations": int(
            frame.loc[frame["interaction_phase"] == "contact", "mask_available"].sum()
        ),
        "cross_directory_same_filename_pairs": len(common_filenames),
        "cross_directory_exact_duplicate_pairs": exact_cross_directory_pairs,
        "cross_phase_duplicate_hashes_excluded": len(conflict_hashes),
        "encoder_crop_size": args.encoder_crop_size,
        "minimum_box_token_mass": args.minimum_box_token_mass,
        "exclusion_counts": dict(Counter(row["reason"] for row in exclusions)),
    }
    atomic_write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
