from __future__ import annotations

import math
import re
from pathlib import Path, PurePosixPath
from typing import Iterable

import pandas as pd


CURATED_REQUIRED_COLUMNS = (
    "observation_id",
    "dataset_revision",
    "source_video_id",
    "media_relpath",
    "media_type",
    "physical_object_id",
    "object_category_manual",
    "state_family",
    "state_label",
    "stable_segment_id",
    "transition_id",
    "start_frame",
    "end_frame",
    "sampled_frame_indices",
    "fps",
    "state_observable",
    "identity_verified",
    "box_annotation_relpath",
    "mask_annotation_relpath",
    "mask_available",
    "coordinate_space",
    "nuisance_tags",
    "review_status",
    "media_sha256",
    "perceptual_hash",
    "duplicate_group_id",
    "verified_asset_group_id",
)

ANALYSIS_REQUIRED_COLUMNS = CURATED_REQUIRED_COLUMNS + (
    "dependency_group_id",
    "split",
    "motion_backend_id",
    "motion_feature_schema_version",
    "motion_feature_vector_scaled",
    "motion_quality_pass",
)

FEATURE_INPUT_ALLOWED_COLUMNS = (
    "observation_id",
    "dataset_revision",
    "media_relpath",
    "media_type",
    "media_sha256",
    "start_frame",
    "end_frame",
    "sampled_frame_indices",
    "fps",
    "box_annotation_relpath",
    "mask_annotation_relpath",
    "mask_available",
    "coordinate_space",
    "motion_backend_id",
    "motion_feature_schema_version",
    "motion_quality_pass",
)

FEATURE_INPUT_FORBIDDEN_COLUMNS = frozenset(
    {
        "state_family",
        "state_label",
        "anchor_state",
        "state_target",
        "triplet_id",
        "anchor_observation_id",
        "nuisance_observation_id",
        "state_observation_id",
        "physical_object_id",
        "verified_asset_group_id",
        "object_category_manual",
        "source_video_id",
        "transition_id",
        "stable_segment_id",
        "nuisance_tags",
        "nuisance_type",
        "nuisance_severity",
        "hand_present",
        "dependency_group_id",
        "split",
        "confirmatory_eligible",
    }
)

GENERIC_STATE = re.compile(r"^(before|after|pre|post|state[_ -]?\d+|unknown|n/?a)?$", re.I)


class ManifestValidationError(ValueError):
    """Raised when a staged manifest violates the fail-closed contract."""


def is_null(value: object) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
        return bool(result) if not hasattr(result, "__len__") else False
    except (TypeError, ValueError):
        return False


def _valid_relative_posix(value: object) -> bool:
    if is_null(value) or not isinstance(value, str) or not value:
        return False
    if "\\" in value or re.match(r"^[A-Za-z]:", value):
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    missing = sorted(set(columns).difference(frame.columns))
    return [f"missing required column: {name}" for name in missing]


def _duplicate_errors(frame: pd.DataFrame, key: str) -> list[str]:
    if key not in frame:
        return []
    duplicate_count = int(frame[key].duplicated(keep=False).sum())
    return [f"{duplicate_count} rows have duplicate {key}"] if duplicate_count else []


def validate_curated_manifest(frame: pd.DataFrame, require_boxes: bool = True) -> list[str]:
    errors = _require_columns(frame, CURATED_REQUIRED_COLUMNS)
    if errors:
        return errors
    errors.extend(_duplicate_errors(frame, "observation_id"))
    for column in (
        "observation_id",
        "source_video_id",
        "physical_object_id",
        "object_category_manual",
        "state_family",
        "state_label",
        "stable_segment_id",
        "transition_id",
        "perceptual_hash",
        "duplicate_group_id",
    ):
        count = int(frame[column].map(is_null).sum())
        if count:
            errors.append(f"{count} rows have null {column}")
    generic = frame["state_label"].astype(str).str.fullmatch(GENERIC_STATE, na=True)
    if bool(generic.any()):
        errors.append(f"{int(generic.sum())} rows have generic/empty state labels")
    for column in ("state_observable", "identity_verified", "mask_available"):
        non_boolean = ~frame[column].map(lambda value: isinstance(value, bool))
        if bool(non_boolean.any()):
            errors.append(f"{int(non_boolean.sum())} rows have non-boolean {column}")
    for column in ("state_observable", "identity_verified"):
        bad = ~frame[column].eq(True)
        if bool(bad.any()):
            errors.append(f"{int(bad.sum())} rows fail {column}=true")
    approved = frame["review_status"].isin(("approved", "adjudicated"))
    if not bool(approved.all()):
        errors.append(f"{int((~approved).sum())} rows are not approved/adjudicated")
    valid_media = frame["media_type"].isin(("image", "video"))
    if not bool(valid_media.all()):
        errors.append(f"{int((~valid_media).sum())} rows have invalid media_type")
    nuisance_missing = frame["nuisance_tags"].map(is_null)
    if bool(nuisance_missing.any()):
        errors.append(f"{int(nuisance_missing.sum())} rows lack nuisance_tags")
    if require_boxes:
        bad = frame["box_annotation_relpath"].map(is_null)
        if bool(bad.any()):
            errors.append(f"{int(bad.sum())} rows lack aligned boxes")
    for column in ("media_relpath", "box_annotation_relpath"):
        present = ~frame[column].map(is_null)
        invalid = present & ~frame[column].map(_valid_relative_posix)
        if bool(invalid.any()):
            errors.append(f"{int(invalid.sum())} rows have unsafe/non-POSIX {column}")
    present = ~frame["mask_annotation_relpath"].map(is_null)
    invalid = present & ~frame["mask_annotation_relpath"].map(_valid_relative_posix)
    if bool(invalid.any()):
        errors.append(f"{int(invalid.sum())} rows have unsafe mask paths")
    declared = frame["mask_available"].eq(True)
    absent = frame["mask_annotation_relpath"].map(is_null)
    if bool((declared & absent).any()):
        errors.append(f"{int((declared & absent).sum())} rows declare a mask without a path")
    coordinate_missing = frame["coordinate_space"].map(is_null)
    if bool(coordinate_missing.any()):
        errors.append(f"{int(coordinate_missing.sum())} rows lack coordinate_space")
    bad_frames = (frame["start_frame"] < 0) | (frame["end_frame"] < frame["start_frame"])
    if bool(bad_frames.any()):
        errors.append(f"{int(bad_frames.sum())} rows have invalid frame intervals")
    fps = pd.to_numeric(frame["fps"], errors="coerce")
    bad_fps = ~fps.map(lambda x: math.isfinite(x) and x > 0)
    if bool(bad_fps.any()):
        errors.append(f"{int(bad_fps.sum())} rows have invalid fps")
    bad_sha = ~frame["media_sha256"].astype(str).str.fullmatch(r"[0-9a-fA-F]{64}", na=False)
    if bool(bad_sha.any()):
        errors.append(f"{int(bad_sha.sum())} rows have invalid media_sha256")

    def valid_samples(row: pd.Series) -> bool:
        values = row["sampled_frame_indices"]
        if not isinstance(values, (list, tuple)) and not hasattr(values, "tolist"):
            return False
        samples = list(values.tolist() if hasattr(values, "tolist") else values)
        return (
            len(samples) == 16
            and samples == sorted(set(samples))
            and all(int(row["start_frame"]) <= int(value) <= int(row["end_frame"]) for value in samples)
        )

    sample_ok = frame.apply(valid_samples, axis=1)
    if not bool(sample_ok.all()):
        errors.append(f"{int((~sample_ok).sum())} rows have invalid 16-frame samples")
    return errors


def validate_analysis_manifest(frame: pd.DataFrame) -> list[str]:
    errors = _require_columns(frame, ANALYSIS_REQUIRED_COLUMNS)
    if errors:
        return errors
    errors.extend(validate_curated_manifest(frame))
    if not frame["split"].isin(("train", "validation", "test")).all():
        errors.append("split must be train, validation, or test")
    for column in ("dependency_group_id", "motion_backend_id", "motion_feature_schema_version"):
        count = int(frame[column].map(is_null).sum())
        if count:
            errors.append(f"{count} rows have null {column}")
    return errors


def validate_feature_inputs(frame: pd.DataFrame, allowed: Iterable[str] = FEATURE_INPUT_ALLOWED_COLUMNS) -> list[str]:
    allowed_set = set(allowed)
    columns = set(frame.columns)
    errors = [f"forbidden feature-input column: {c}" for c in sorted(columns & FEATURE_INPUT_FORBIDDEN_COLUMNS)]
    errors.extend(f"unexpected feature-input column: {c}" for c in sorted(columns - allowed_set))
    errors.extend(f"missing feature-input column: {c}" for c in sorted(allowed_set - columns))
    errors.extend(_duplicate_errors(frame, "observation_id"))
    return errors


def raise_on_errors(errors: Iterable[str], context: str) -> None:
    errors = list(errors)
    if errors:
        detail = "\n".join(f"  - {item}" for item in errors)
        raise ManifestValidationError(f"{context} failed closed:\n{detail}")


def read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    raise ValueError(f"Unsupported table type: {source}")
