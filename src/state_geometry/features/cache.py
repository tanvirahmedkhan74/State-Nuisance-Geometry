from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from state_geometry.utils.hashing import atomic_write_json, sha256_file


CATALOG_COLUMNS = (
    "feature_key",
    "run_id",
    "pool",
    "layer",
    "input_control",
    "subset_name",
    "features_path",
    "features_sha256",
    "index_path",
    "index_sha256",
    "metadata_path",
    "rows",
    "dimension",
    "dtype",
    "status",
)


@dataclass(frozen=True)
class CachedFeature:
    feature_key: str
    array: np.ndarray


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(destination)


def _atomic_npy(values: np.ndarray, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
    temporary.replace(destination)


def write_immutable_feature_run(
    output_root: str | Path,
    run_id: str,
    observation_ids: Sequence[str],
    features_by_pool: Mapping[str, np.ndarray],
    metadata: Mapping[str, object],
    *,
    layer: int,
    input_control: str,
    subset_name: str,
    feature_key_prefix: str,
) -> list[dict[str, object]]:
    destination = Path(output_root)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"immutable feature run already exists: {destination}")
    ids = [str(value) for value in observation_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("observation_ids must be unique")
    if not ids:
        raise ValueError("cannot write an empty feature run")
    validated_features: dict[str, np.ndarray] = {}
    for pool, raw_values in features_by_pool.items():
        values = np.asarray(raw_values)
        if values.ndim != 2 or values.shape[0] != len(ids):
            raise ValueError(f"{pool} features must have shape [N,D] aligned to the index")
        if values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError(f"{pool} features must be finite FP32")
        validated_features[pool] = values
    if not validated_features:
        raise ValueError("at least one feature pool is required")
    destination.mkdir(parents=True, exist_ok=False)
    index_path = destination / "index.parquet"
    index = pd.DataFrame({"row_index": np.arange(len(ids), dtype=np.int64), "observation_id": ids})
    _atomic_parquet(index, index_path)
    metadata_path = destination / "metadata.json"
    run_metadata = {
        **dict(metadata),
        "run_id": run_id,
        "rows": len(ids),
        "observation_order_sha256": __import__("hashlib").sha256(
            json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    atomic_write_json(metadata_path, run_metadata)
    rows: list[dict[str, object]] = []
    for pool, values in validated_features.items():
        path = destination / f"features_{pool}.npy"
        _atomic_npy(values, path)
        rows.append(
            {
                "feature_key": f"{feature_key_prefix}/layer{layer}/{pool}/{input_control}/{subset_name}",
                "run_id": run_id,
                "pool": pool,
                "layer": int(layer),
                "input_control": input_control,
                "subset_name": subset_name,
                "features_path": path.resolve().as_posix(),
                "features_sha256": sha256_file(path),
                "index_path": index_path.resolve().as_posix(),
                "index_sha256": sha256_file(index_path),
                "metadata_path": metadata_path.resolve().as_posix(),
                "rows": len(ids),
                "dimension": int(values.shape[1]),
                "dtype": str(values.dtype),
                "status": "complete",
            }
        )
    atomic_write_json(destination / "complete.json", {"catalog_rows": rows})
    return rows


def append_feature_catalog(catalog_path: str | Path, rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    path = Path(catalog_path)
    incoming = pd.DataFrame(rows, columns=CATALOG_COLUMNS)
    if incoming.empty or incoming["feature_key"].duplicated().any():
        raise ValueError("incoming catalog rows must have unique feature keys")
    if path.exists():
        existing = pd.read_parquet(path)
        duplicate_keys = set(existing["feature_key"]) & set(incoming["feature_key"])
        duplicate_runs = set(existing["run_id"]) & set(incoming["run_id"])
        if duplicate_keys or duplicate_runs:
            raise FileExistsError(
                f"catalog is append-only; duplicate keys={sorted(duplicate_keys)}, runs={sorted(duplicate_runs)}"
            )
        combined = pd.concat([existing, incoming], ignore_index=True)
    else:
        combined = incoming
    _atomic_parquet(combined, path)
    return combined


def load_cached_feature(catalog_path: str | Path, feature_key: str) -> tuple[np.ndarray, pd.DataFrame]:
    catalog = pd.read_parquet(catalog_path)
    selected = catalog.loc[catalog["feature_key"] == feature_key]
    if len(selected) != 1:
        raise KeyError(f"feature key must resolve exactly once: {feature_key}")
    row = selected.iloc[0]
    features_path = Path(row["features_path"])
    index_path = Path(row["index_path"])
    if sha256_file(features_path) != row["features_sha256"]:
        raise RuntimeError("feature cache hash mismatch")
    if sha256_file(index_path) != row["index_sha256"]:
        raise RuntimeError("feature index hash mismatch")
    values = np.load(features_path, mmap_mode="r", allow_pickle=False)
    index = pd.read_parquet(index_path)
    if values.shape != (len(index), int(row["dimension"])):
        raise RuntimeError("feature cache shape/index mismatch")
    return values, index
