from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from state_geometry.data.schema import (  # noqa: E402
    FEATURE_INPUT_ALLOWED_COLUMNS,
    raise_on_errors,
    read_table,
    validate_feature_inputs,
)
from state_geometry.features.cache import append_feature_catalog, write_immutable_feature_run  # noqa: E402
from state_geometry.features.extraction import extract_observation_pools  # noqa: E402
from state_geometry.features.vjepa import load_frozen_vjepa21_vitb  # noqa: E402
from state_geometry.utils.hashing import sha256_file  # noqa: E402


def _filter_subset(frame: pd.DataFrame, query: str | None) -> pd.DataFrame:
    if query is None:
        return frame
    normalized = query.strip().lower()
    if normalized not in ("mask_available=true", "mask_available=false"):
        raise ValueError("only feature-blind mask_available=true/false subset queries are supported")
    expected = normalized.endswith("true")
    return frame.loc[frame["mask_available"].eq(expected)].copy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen V-JEPA 2.1 ViT-B feature extraction.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--require-label-redacted-input", action="store_true")
    parser.add_argument("--require-motion-provenance", action="store_true")
    parser.add_argument("--no-motion-quality-filter", action="store_true")
    parser.add_argument("--subset-query")
    parser.add_argument(
        "--limit-observations",
        type=int,
        help="Deterministic feature-blind prefix for a smoke run; omit for experiments.",
    )
    parser.add_argument("--layers", required=True)
    parser.add_argument("--pools", required=True)
    parser.add_argument("--input-control", required=True)
    parser.add_argument("--subset-name", required=True)
    parser.add_argument("--feature-key-prefix", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--component-resolution", type=Path)
    args = parser.parse_args()
    if args.layers != "11":
        parser.error("the Phase 1 primary extraction is frozen to final layer 11")
    if args.batch_size != 1 or args.workers != 0:
        parser.error("the audited 6 GiB local protocol requires batch-size=1 and workers=0")
    pools = [value.strip() for value in args.pools.split(",") if value.strip()]
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_config = config["model"]
    checkpoint_sha = model_config.get("checkpoint_sha256")
    checkpoint_path = resolve_config_path(args.config, model_config["checkpoint_path"])
    source_root = resolve_config_path(args.config, model_config["source_root"])
    media_root = resolve_config_path(
        args.config,
        config.get("data", {}).get("media_root", "."),
    )
    asset_errors: list[str] = []
    if not checkpoint_path.is_file():
        asset_errors.append(f"primary ViT-B checkpoint is absent: {checkpoint_path}")
    if not checkpoint_sha:
        asset_errors.append("model.checkpoint_sha256 lacks an independently recorded fingerprint")
    if asset_errors:
        raise RuntimeError("model asset preflight failed:\n  - " + "\n  - ".join(asset_errors))

    observations = read_table(args.observations)
    if args.require_label_redacted_input:
        raise_on_errors(
            validate_feature_inputs(observations, FEATURE_INPUT_ALLOWED_COLUMNS),
            "label-redacted extraction input",
        )
    if args.require_motion_provenance:
        required = {"motion_backend_id", "motion_feature_schema_version", "motion_quality_pass"}
        missing = sorted(required.difference(observations.columns))
        if missing or observations[list(required)].isna().any().any():
            raise ValueError(f"motion provenance is incomplete; missing={missing}")
    observations = _filter_subset(observations, args.subset_query)
    if args.limit_observations is not None:
        if args.limit_observations <= 0:
            parser.error("--limit-observations must be positive")
        observations = observations.iloc[: args.limit_observations].copy()
    if observations.empty:
        raise ValueError("feature-blind subset contains no observations")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder, model_metadata = load_frozen_vjepa21_vitb(
        source_root,
        checkpoint_path,
        checkpoint_sha,
        device=device,
        expected_commit=model_config["source_commit"],
    )
    collected = {pool: [] for pool in pools}
    observation_ids: list[str] = []
    for position, (_, row) in enumerate(observations.iterrows(), start=1):
        outputs = extract_observation_pools(
            encoder,
            row,
            media_root,
            pools,
            args.input_control,
            use_bfloat16=device.type == "cuda",
        )
        for pool in pools:
            collected[pool].append(outputs[pool])
        observation_ids.append(str(row["observation_id"]))
        if position % 100 == 0 or position == len(observations):
            print(f"extracted {position}/{len(observations)} observations", flush=True)
    arrays = {pool: np.stack(values).astype(np.float32) for pool, values in collected.items()}
    metadata = {
        **model_metadata,
        "config_path": args.config.resolve().as_posix(),
        "config_sha256": sha256_file(args.config),
        "observations_path": args.observations.resolve().as_posix(),
        "observations_sha256": sha256_file(args.observations),
        "media_root": media_root.as_posix(),
        "device": str(device),
        "batch_size": 1,
        "workers": 0,
        "forward_precision": "bf16" if device.type == "cuda" else "fp32",
        "pooled_precision": "fp32",
        "modality_counts": observations["media_type"].value_counts().to_dict(),
        "smoke_limit_observations": args.limit_observations,
        "context_tokens_are_globally_contextualized": True,
    }
    rows = write_immutable_feature_run(
        args.output_root,
        args.run_id,
        observation_ids,
        arrays,
        metadata,
        layer=11,
        input_control=args.input_control,
        subset_name=args.subset_name,
        feature_key_prefix=args.feature_key_prefix,
    )
    append_feature_catalog(args.catalog, rows)
    print(f"wrote {len(rows)} immutable feature keys to {args.catalog}")
    return 0


def resolve_config_path(config_path: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    # Config paths are workspace-relative, not dependent on the caller's CWD.
    return (PROJECT_ROOT / candidate).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
