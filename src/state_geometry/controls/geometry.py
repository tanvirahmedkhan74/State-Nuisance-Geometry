from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _normalize(values: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms <= epsilon).any():
        raise ValueError("geometry transform produced zero-norm rows")
    return (values / np.maximum(norms, epsilon)).astype(np.float32)


@dataclass(frozen=True)
class PCAGeometryControl:
    mean: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray
    whiten: bool
    shrinkage: float
    eigenvalue_floor: float

    @classmethod
    def fit(
        cls,
        train: np.ndarray,
        dimensions: int,
        *,
        whiten: bool = False,
        shrinkage_relative: float = 0.0,
        eigenvalue_floor_relative: float = 1e-6,
    ) -> "PCAGeometryControl":
        values = np.asarray(train, dtype=np.float64)
        if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("train features must be finite [N,D] with N>=2")
        maximum = min(values.shape[1], values.shape[0] - 1)
        if dimensions <= 0 or dimensions > maximum:
            raise ValueError(f"dimensions must be in [1,{maximum}]")
        if shrinkage_relative < 0 or eigenvalue_floor_relative <= 0:
            raise ValueError("shrinkage must be nonnegative and floor positive")
        mean = values.mean(axis=0)
        centered = values - mean
        covariance = centered.T @ centered / len(values)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1][:dimensions]
        selected_values = np.clip(eigenvalues[order], 0.0, None)
        positive = eigenvalues[eigenvalues > 0]
        scale = float(positive.mean()) if len(positive) else 1.0
        return cls(
            mean=mean.astype(np.float32),
            components=eigenvectors[:, order].T.astype(np.float32),
            eigenvalues=selected_values.astype(np.float32),
            whiten=whiten,
            shrinkage=float(shrinkage_relative * scale),
            eigenvalue_floor=float(eigenvalue_floor_relative * scale),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.mean.size:
            raise ValueError("feature width mismatch")
        projected = (array - self.mean) @ self.components.T
        if self.whiten:
            denominator = np.sqrt(
                np.maximum(self.eigenvalues + self.shrinkage, self.eigenvalue_floor)
            )
            projected = projected / denominator
        return _normalize(projected)


def seeded_random_projection(input_dimension: int, output_dimension: int, seed: int) -> np.ndarray:
    if output_dimension <= 0 or output_dimension > input_dimension:
        raise ValueError("random projection output must be in [1,input_dimension]")
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(input_dimension, output_dimension))
    orthogonal, _ = np.linalg.qr(matrix)
    return orthogonal.T.astype(np.float32)

