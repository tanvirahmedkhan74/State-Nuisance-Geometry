from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RobustMotionScaler:
    median: np.ndarray
    denominator: np.ndarray
    active: np.ndarray
    native_floor: np.ndarray

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        native_floor: float | Sequence[float] = 1e-6,
        constant_tolerance: float = 0.0,
    ) -> "RobustMotionScaler":
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] < 2:
            raise ValueError("motion scaler requires [N,K] with at least two rows")
        if not np.isfinite(array).all():
            raise ValueError("motion values must be complete and finite")
        median = np.median(array, axis=0)
        q25, q75 = np.quantile(array, (0.25, 0.75), axis=0)
        iqr = q75 - q25
        floor = np.broadcast_to(np.asarray(native_floor, dtype=np.float64), iqr.shape).copy()
        if (floor <= 0).any() or not np.isfinite(floor).all():
            raise ValueError("native floors must be finite and positive")
        span = np.ptp(array, axis=0)
        active = span > constant_tolerance
        if not active.any():
            raise ValueError("all train motion dimensions are constant")
        denominator = np.maximum(iqr, floor)
        return cls(median=median, denominator=denominator, active=active, native_floor=floor)

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1] != self.median.size:
            raise ValueError("motion schema width mismatch")
        if not np.isfinite(array).all():
            raise ValueError("null/nonfinite motion coordinates cannot be imputed")
        return ((array - self.median) / self.denominator)[..., self.active].astype(np.float32)


def motion_pair_change(anchor: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchor = np.asarray(anchor, dtype=np.float32)
    other = np.asarray(other, dtype=np.float32)
    if anchor.shape != other.shape:
        raise ValueError("pair descriptors must have identical schemas")
    signed = other - anchor
    severity = np.abs(signed)
    level = np.linalg.norm(severity, axis=-1)
    return signed, severity, level


def motion_triplet_mismatch(
    anchor: np.ndarray, nuisance: np.ndarray, state: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signed_n, severity_n, level_n = motion_pair_change(anchor, nuisance)
    signed_s, severity_s, level_s = motion_pair_change(anchor, state)
    signed_mismatch = np.linalg.norm(signed_n - signed_s, axis=-1)
    severity_mismatch = np.linalg.norm(severity_n - severity_s, axis=-1)
    common_level = 0.5 * (level_n + level_s)
    return signed_mismatch, severity_mismatch, common_level


def motion_matched_mask(
    signed_mismatch: np.ndarray,
    severity_mismatch: np.ndarray,
    quality_pass: np.ndarray,
    signed_caliper: float,
    severity_caliper: float,
) -> np.ndarray:
    quality = np.asarray(quality_pass, dtype=bool)
    if quality.ndim != 2 or quality.shape[1] != 3:
        raise ValueError("quality_pass must be [N,3] for anchor/nuisance/state")
    return (
        (np.asarray(signed_mismatch) <= signed_caliper)
        & (np.asarray(severity_mismatch) <= severity_caliper)
        & quality.all(axis=1)
    )


def paired_group_role_permutation(
    labels: np.ndarray,
    pair_ids: np.ndarray,
    dependency_groups: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int8)
    pair_ids = np.asarray(pair_ids)
    groups = np.asarray(dependency_groups)
    if not (labels.ndim == pair_ids.ndim == groups.ndim == 1):
        raise ValueError("labels, pair_ids, and groups must be vectors")
    if not (len(labels) == len(pair_ids) == len(groups)):
        raise ValueError("permutation inputs must have equal length")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("role labels must be 0/1")
    for pair in np.unique(pair_ids):
        index = np.flatnonzero(pair_ids == pair)
        if len(index) != 2 or set(labels[index]) != {0, 1}:
            raise ValueError(f"pair {pair!r} must contain exactly one role 0 and one role 1")
        if len(np.unique(groups[index])) != 1:
            raise ValueError(f"pair {pair!r} crosses dependency groups")
    result = labels.copy()
    for group in np.unique(groups):
        if bool(rng.integers(0, 2)):
            index = groups == group
            result[index] = 1 - result[index]
    return result

