import unittest

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from state_geometry.evaluation.geometry import (
    cosine_distance,
    effective_rank,
    state_nuisance_score,
)


class GeometryTests(unittest.TestCase):
    def test_cosine_distance(self) -> None:
        left = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        right = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(cosine_distance(left, right), [0.0, 2.0])

    def test_tie_aware_nested_weighting(self) -> None:
        rows = []
        for index in range(100):
            rows.append({"dependency_group_id": "g", "transition_id": "t1", "anchor_observation_id": "a1", "d_n": 1.0, "d_s": 0.0})
        rows.extend(
            [
                {"dependency_group_id": "g", "transition_id": "t1", "anchor_observation_id": "a2", "d_n": 0.0, "d_s": 1.0},
                {"dependency_group_id": "g", "transition_id": "t2", "anchor_observation_id": "a3", "d_n": 0.0, "d_s": 1.0},
            ]
        )
        estimate = state_nuisance_score(pd.DataFrame(rows))
        self.assertAlmostEqual(estimate.estimate, 0.75)
        self.assertEqual(estimate.transitions, 2)

    def test_tie_credit_and_effective_rank(self) -> None:
        frame = pd.DataFrame(
            [{"dependency_group_id": "g", "transition_id": "t", "anchor_observation_id": "a", "d_n": 0.5, "d_s": 0.5}]
        )
        self.assertEqual(state_nuisance_score(frame).estimate, 0.5)
        self.assertEqual(effective_rank(np.ones((4, 3))), 0.0)


if __name__ == "__main__":
    unittest.main()

