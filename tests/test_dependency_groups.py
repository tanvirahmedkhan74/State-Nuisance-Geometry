import unittest

import pandas as pd

import _bootstrap  # noqa: F401
from state_geometry.data.splits import assign_group_splits, build_dependency_groups


class DependencyGroupTests(unittest.TestCase):
    def test_nulls_create_no_edges(self) -> None:
        frame = pd.DataFrame(
            {
                "observation_id": ["a", "b", "c"],
                "video": ["v1", "v2", "v3"],
                "asset": [None, None, None],
            }
        )
        groups = build_dependency_groups(frame, ["video", "asset"])
        self.assertEqual(groups.nunique(), 3)

    def test_transitive_edges_form_one_component(self) -> None:
        frame = pd.DataFrame(
            {
                "observation_id": ["a", "b", "c", "d"],
                "video": ["v1", "v1", "v2", "v3"],
                "asset": [None, "x", "x", None],
            }
        )
        groups = build_dependency_groups(frame, ["video", "asset"])
        self.assertEqual(groups.iloc[0], groups.iloc[2])
        self.assertNotEqual(groups.iloc[0], groups.iloc[3])

    def test_split_keeps_components_intact(self) -> None:
        frame = pd.DataFrame(
            {
                "observation_id": [f"o{i}" for i in range(12)],
                "dependency_group_id": [f"g{i // 2}" for i in range(12)],
                "state": ["open", "closed"] * 6,
                "nuisance": [["view"]] * 12,
            }
        )
        result = assign_group_splits(
            frame, stratify=["state"], multilabel=["nuisance"], seed=7
        )
        self.assertEqual(
            result.mapping.groupby("dependency_group_id")["split"].nunique().max(), 1
        )
        counts = result.mapping["split"].value_counts().to_dict()
        self.assertEqual(counts, {"train": 8, "validation": 2, "test": 2})


if __name__ == "__main__":
    unittest.main()
