from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def l2_normalize(values: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    if (norms <= epsilon).any():
        raise ValueError("zero-norm feature cannot be normalized")
    return array / np.maximum(norms, epsilon)


def cosine_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = l2_normalize(left)
    b = l2_normalize(right)
    if a.shape != b.shape:
        raise ValueError("cosine inputs must have identical shapes")
    return (1.0 - np.sum(a * b, axis=-1, dtype=np.float32)).astype(np.float32)


def triplet_distances(anchor: np.ndarray, nuisance: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return cosine_distance(anchor, nuisance), cosine_distance(anchor, state)


def normalized_margin(d_n: np.ndarray, d_s: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    nuisance = np.asarray(d_n, dtype=np.float32)
    state = np.asarray(d_s, dtype=np.float32)
    return (state - nuisance) / (state + nuisance + epsilon)


@dataclass(frozen=True)
class NestedEstimate:
    estimate: float
    groups: int
    transitions: int
    anchors: int
    rows: int


def nested_group_mean(
    frame: pd.DataFrame,
    value_column: str,
    group_column: str = "dependency_group_id",
    transition_column: str = "transition_id",
    anchor_column: str = "anchor_observation_id",
) -> NestedEstimate:
    required = {value_column, group_column, transition_column, anchor_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing estimator columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("cannot estimate from zero rows")
    anchor = (
        frame.groupby([group_column, transition_column, anchor_column], sort=False, dropna=False)[value_column]
        .mean()
        .rename("anchor_mean")
        .reset_index()
    )
    transition = (
        anchor.groupby([group_column, transition_column], sort=False, dropna=False)["anchor_mean"]
        .mean()
        .rename("transition_mean")
        .reset_index()
    )
    group = transition.groupby(group_column, sort=False, dropna=False)["transition_mean"].mean()
    return NestedEstimate(
        estimate=float(group.mean()),
        groups=int(len(group)),
        transitions=int(len(transition)),
        anchors=int(len(anchor)),
        rows=int(len(frame)),
    )


def state_nuisance_score(
    frame: pd.DataFrame,
    tolerance: float = 1e-6,
    d_n_column: str = "d_n",
    d_s_column: str = "d_s",
) -> NestedEstimate:
    if tolerance < 0:
        raise ValueError("tie tolerance must be nonnegative")
    work = frame.copy()
    delta = work[d_s_column].to_numpy(np.float32) - work[d_n_column].to_numpy(np.float32)
    work["_sns_row"] = (delta > tolerance).astype(np.float32) + 0.5 * (
        np.abs(delta) <= tolerance
    ).astype(np.float32)
    return nested_group_mean(work, "_sns_row")


def effective_rank(values: np.ndarray, epsilon: float = 1e-12) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or len(array) < 2:
        raise ValueError("effective rank requires [N,D] with N>=2")
    covariance = np.cov(array, rowvar=False, ddof=0)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = eigenvalues.sum()
    if total <= epsilon:
        return 0.0
    probabilities = eigenvalues[eigenvalues > epsilon] / total
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))
