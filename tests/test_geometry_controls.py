import unittest

import numpy as np

import _bootstrap  # noqa: F401
from state_geometry.controls.geometry import PCAGeometryControl, seeded_random_projection
from state_geometry.evaluation.geometry import cosine_distance


class GeometryControlTests(unittest.TestCase):
    def test_full_pca_is_centered_cosine_invariant(self) -> None:
        rng = np.random.default_rng(2)
        values = rng.normal(size=(20, 8)).astype(np.float32)
        control = PCAGeometryControl.fit(values, dimensions=8)
        transformed = control.transform(values)
        centered = values - values.mean(axis=0, keepdims=True)
        np.testing.assert_allclose(
            cosine_distance(transformed[:-1], transformed[1:]),
            cosine_distance(centered[:-1], centered[1:]),
            atol=2e-6,
        )

    def test_rank_limit_and_random_projection(self) -> None:
        values = np.eye(4, dtype=np.float32)
        with self.assertRaises(ValueError):
            PCAGeometryControl.fit(values, dimensions=4)
        projection = seeded_random_projection(8, 3, seed=4)
        np.testing.assert_allclose(projection @ projection.T, np.eye(3), atol=1e-6)


if __name__ == "__main__":
    unittest.main()

