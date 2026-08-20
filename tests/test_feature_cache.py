import gc
import tempfile
import unittest
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from state_geometry.features.cache import (
    append_feature_catalog,
    load_cached_feature,
    write_immutable_feature_run,
)


class FeatureCacheTests(unittest.TestCase):
    def test_immutable_hash_checked_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = write_immutable_feature_run(
                root / "run",
                "run",
                ["a", "b"],
                {"box": np.ones((2, 4), dtype=np.float32)},
                {"checkpoint_sha256": "0" * 64},
                layer=11,
                input_control="original",
                subset_name="all",
                feature_key_prefix="vjepa21b",
            )
            catalog = root / "catalog.parquet"
            append_feature_catalog(catalog, rows)
            values, index = load_cached_feature(catalog, "vjepa21b/layer11/box/original/all")
            self.assertEqual(values.shape, (2, 4))
            self.assertEqual(index["observation_id"].tolist(), ["a", "b"])
            with self.assertRaises(FileExistsError):
                write_immutable_feature_run(
                    root / "run",
                    "run",
                    ["a"],
                    {"box": np.ones((1, 4), dtype=np.float32)},
                    {},
                    layer=11,
                    input_control="original",
                    subset_name="all",
                    feature_key_prefix="vjepa21b",
                )
            del values
            gc.collect()


if __name__ == "__main__":
    unittest.main()
