import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

import _bootstrap  # noqa: F401
from state_geometry.features.extraction import (
    decode_observation,
    load_boxes,
    rasterize_source_polygons,
    resolve_workspace_path,
)


class ExtractionHelperTests(unittest.TestCase):
    def test_path_escape_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                resolve_workspace_path(temporary, "../escape.jpg")

    def test_proxy_image_and_regions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (32, 24), "white").save(root / "image.jpg")
            row = pd.Series(
                {
                    "observation_id": "x",
                    "media_relpath": "image.jpg",
                    "media_type": "image",
                    "bbox_xywh": [1, 2, 10, 8],
                }
            )
            decoded = decode_observation(row, root)
            self.assertEqual(tuple(decoded.shape), (1, 24, 32, 3))
            self.assertEqual(load_boxes(row, root, 1), [(1.0, 2.0, 10.0, 8.0)])
            masks = rasterize_source_polygons([[[1, 1, 10, 1, 10, 10]]], 24, 32)
            self.assertEqual(tuple(masks.shape), (1, 24, 32))
            self.assertTrue(masks.any())


if __name__ == "__main__":
    unittest.main()
