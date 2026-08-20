import unittest

import numpy as np

import _bootstrap  # noqa: F401
from state_geometry.controls.motion import (
    RobustMotionScaler,
    motion_triplet_mismatch,
    paired_group_role_permutation,
)


class MotionTests(unittest.TestCase):
    def test_scaler_drops_constant_and_uses_floor(self) -> None:
        values = np.array([[0.0, 1.0], [1e-9, 1.0], [2e-9, 1.0]])
        scaler = RobustMotionScaler.fit(values, native_floor=[1e-3, 1e-3])
        self.assertEqual(scaler.active.tolist(), [True, False])
        self.assertGreaterEqual(scaler.denominator[0], 1e-3)
        self.assertEqual(scaler.transform(values).shape, (3, 1))

    def test_signed_change_catches_opposite_motion(self) -> None:
        anchor = np.array([[0.0, 0.0]])
        nuisance = np.array([[1.0, 0.0]])
        state = np.array([[-1.0, 0.0]])
        signed, severity, _ = motion_triplet_mismatch(anchor, nuisance, state)
        self.assertEqual(float(signed[0]), 2.0)
        self.assertEqual(float(severity[0]), 0.0)

    def test_permutation_swaps_whole_components(self) -> None:
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        pairs = np.array(["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"])
        groups = np.array(["g1", "g1", "g1", "g1", "g2", "g2", "g2", "g2"])
        result = paired_group_role_permutation(labels, pairs, groups, np.random.default_rng(3))
        for group in ("g1", "g2"):
            index = groups == group
            self.assertTrue(np.array_equal(result[index], labels[index]) or np.array_equal(result[index], 1 - labels[index]))


if __name__ == "__main__":
    unittest.main()

