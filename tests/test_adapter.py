import unittest

import torch

import _bootstrap  # noqa: F401
from state_geometry.models.residual_adapter import (
    ResidualOrderingAdapter,
    build_metric_adapter,
    margin_triplet_loss,
)


class AdapterTests(unittest.TestCase):
    def test_parameter_count_and_first_step_gradients(self) -> None:
        model = ResidualOrderingAdapter(768, 256)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 395777)
        features = torch.randn(48, 768)
        adapted = model(features).reshape(3, 16, 768)
        loss = margin_triplet_loss(adapted[0], adapted[1], adapted[2], margin=2.0)
        loss.backward()
        self.assertIsNotNone(model.down.weight.grad)
        self.assertIsNotNone(model.up.weight.grad)
        self.assertIsNotNone(model.alpha.grad)
        self.assertGreater(float(model.down.weight.grad.norm()), 0.0)
        self.assertGreater(float(model.up.weight.grad.norm()), 0.0)

    def test_matched_metric_baselines(self) -> None:
        features = torch.randn(4, 768)
        for architecture in (
            "identity",
            "positive_diagonal",
            "linear_residual",
            "mlp_nonresidual",
            "residual_bottleneck",
        ):
            transformed = build_metric_adapter(architecture)(features)
            self.assertEqual(tuple(transformed.shape), (4, 768))
            torch.testing.assert_close(transformed.norm(dim=-1), torch.ones(4))


if __name__ == "__main__":
    unittest.main()
