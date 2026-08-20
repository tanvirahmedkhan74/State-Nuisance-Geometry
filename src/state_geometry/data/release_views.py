from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from state_geometry.utils.hashing import atomic_write_json, sha256_file

from .schema import FEATURE_INPUT_ALLOWED_COLUMNS, raise_on_errors, validate_feature_inputs


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def build_release_views(
    observations: pd.DataFrame,
    triplets: pd.DataFrame,
    output_dir: str | Path,
    feature_columns: Sequence[str] = FEATURE_INPUT_ALLOWED_COLUMNS,
) -> dict[str, object]:
    required_observation = {"observation_id", "split", *feature_columns}
    missing_observation = required_observation.difference(observations.columns)
    if missing_observation:
        raise ValueError(f"observations missing release columns: {sorted(missing_observation)}")
    required_triplet = {"triplet_id", "split"}
    missing_triplet = required_triplet.difference(triplets.columns)
    if missing_triplet:
        raise ValueError(f"triplets missing release columns: {sorted(missing_triplet)}")
    if not observations["split"].isin(("train", "validation", "test")).all():
        raise ValueError("invalid observation split")
    if not triplets["split"].isin(("train", "validation", "test")).all():
        raise ValueError("invalid triplet split")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    views = {
        "analysis_trainval.parquet": observations.loc[observations["split"] != "test"].copy(),
        "feature_inputs.parquet": observations.loc[:, list(feature_columns)].copy(),
        "triplets_trainval.parquet": triplets.loc[triplets["split"] != "test"].copy(),
        "sealed_test_targets.parquet": observations.loc[observations["split"] == "test"].copy(),
        "sealed_test_triplets.parquet": triplets.loc[triplets["split"] == "test"].copy(),
    }
    raise_on_errors(validate_feature_inputs(views["feature_inputs.parquet"], feature_columns), "feature-input view")

    # Intentionally do not compute or expose any label/role distribution summaries.
    for name, frame in views.items():
        _write_parquet(frame, destination / name)
    artifacts = {
        name: {"sha256": sha256_file(destination / name), "rows": len(frame) if "test" not in name else None}
        for name, frame in views.items()
    }
    manifest = {
        "schema_version": 1,
        "feature_input_columns": list(feature_columns),
        "test_summaries_suppressed": True,
        "artifacts": artifacts,
    }
    atomic_write_json(destination / "release_views.json", manifest)
    return manifest

