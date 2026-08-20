import unittest

import torch

import _bootstrap  # noqa: F401
from state_geometry.features.preprocess import (
    eval_resize_crop_geometry,
    preprocess_rgb_frames,
    rasterized_box_token_mass,
    rasterize_xywh_boxes,
    transform_binary_masks,
    transform_xywh_box,
)


class PreprocessTests(unittest.TestCase):
    def test_official_geometry_and_rgb_shape(self) -> None:
        geometry = eval_resize_crop_geometry(720, 1280)
        self.assertEqual((geometry.resized_height, geometry.resized_width), (438, 778))
        self.assertEqual((geometry.crop_top, geometry.crop_left), (27, 197))
        video, realized = preprocess_rgb_frames(torch.zeros(16, 720, 1280, 3, dtype=torch.uint8))
        self.assertEqual(realized, geometry)
        self.assertEqual(tuple(video.shape), (3, 16, 384, 384))

    def test_masks_and_boxes_share_field_of_view(self) -> None:
        geometry = eval_resize_crop_geometry(720, 1280)
        masks = torch.zeros(16, 720, 1280, dtype=torch.bool)
        masks[:, 100:300, 400:700] = True
        transformed = transform_binary_masks(masks, geometry)
        self.assertEqual(tuple(transformed.shape), (16, 384, 384))
        mapped = transform_xywh_box((400, 100, 300, 200), geometry)
        boxes = rasterize_xywh_boxes([(400, 100, 300, 200)] * 16, geometry)
        self.assertEqual(tuple(boxes.shape), (16, 384, 384))
        self.assertTrue(boxes.any())
        self.assertTrue(all(value >= 0 for value in mapped))
        self.assertGreaterEqual(rasterized_box_token_mass(mapped), 1.0)

    def test_box_outside_crop_fails(self) -> None:
        geometry = eval_resize_crop_geometry(720, 1280)
        with self.assertRaises(ValueError):
            transform_xywh_box((0, 0, 1, 1), geometry)


if __name__ == "__main__":
    unittest.main()
