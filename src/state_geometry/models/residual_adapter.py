from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class IdentityMetric(nn.Module):
    def forward(self, features: Tensor) -> Tensor:
        return F.normalize(features, dim=-1)


class PositiveDiagonalMetric(nn.Module):
    """Learn a diagonal PSD metric under the same ordering supervision."""

    def __init__(self, dimension: int = 768) -> None:
        super().__init__()
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.raw_scale = nn.Parameter(torch.zeros(dimension))

    def forward(self, features: Tensor) -> Tensor:
        scale = F.softplus(self.raw_scale) + 1e-6
        return F.normalize(features * scale, dim=-1)


class LinearResidualMetric(nn.Module):
    def __init__(self, dimension: int = 768, alpha: float = 1e-3) -> None:
        super().__init__()
        if dimension <= 0 or alpha <= 0:
            raise ValueError("dimension and alpha must be positive")
        self.norm = nn.LayerNorm(dimension)
        self.linear = nn.Linear(dimension, dimension)
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features: Tensor) -> Tensor:
        return F.normalize(features + self.alpha * self.linear(self.norm(features)), dim=-1)


class NonResidualMLPMetric(nn.Module):
    def __init__(self, dimension: int = 768, bottleneck: int = 256) -> None:
        super().__init__()
        if dimension <= 0 or bottleneck <= 0:
            raise ValueError("dimension and bottleneck must be positive")
        self.network = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, dimension),
        )
        nn.init.xavier_uniform_(self.network[1].weight)
        nn.init.zeros_(self.network[1].bias)
        nn.init.xavier_uniform_(self.network[3].weight)
        nn.init.zeros_(self.network[3].bias)

    def forward(self, features: Tensor) -> Tensor:
        return F.normalize(self.network(features), dim=-1)


class ResidualOrderingAdapter(nn.Module):
    def __init__(self, dimension: int = 768, bottleneck: int = 256, alpha: float = 1e-3) -> None:
        super().__init__()
        if dimension <= 0 or bottleneck <= 0 or alpha <= 0:
            raise ValueError("dimension, bottleneck, and alpha must be positive")
        self.norm = nn.LayerNorm(dimension)
        self.down = nn.Linear(dimension, bottleneck)
        self.up = nn.Linear(bottleneck, dimension)
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.xavier_uniform_(self.up.weight, gain=0.1)
        nn.init.zeros_(self.up.bias)

    def forward(self, features: Tensor) -> Tensor:
        residual = self.up(F.gelu(self.down(self.norm(features))))
        return F.normalize(features + self.alpha * residual, dim=-1)


def cosine_distance(left: Tensor, right: Tensor) -> Tensor:
    return 1.0 - F.cosine_similarity(left.float(), right.float(), dim=-1)


def margin_triplet_loss(anchor: Tensor, nuisance: Tensor, state: Tensor, margin: float = 0.1) -> Tensor:
    if margin < 0 or margin > 2:
        raise ValueError("cosine-distance margin must be in [0,2]")
    d_n = cosine_distance(anchor, nuisance)
    d_s = cosine_distance(anchor, state)
    return F.relu(float(margin) + d_n - d_s).mean()


def build_metric_adapter(
    architecture: str,
    dimension: int = 768,
    bottleneck: int = 256,
) -> nn.Module:
    factories = {
        "identity": lambda: IdentityMetric(),
        "positive_diagonal": lambda: PositiveDiagonalMetric(dimension),
        "linear_residual": lambda: LinearResidualMetric(dimension),
        "mlp_nonresidual": lambda: NonResidualMLPMetric(dimension, bottleneck),
        "residual_bottleneck": lambda: ResidualOrderingAdapter(dimension, bottleneck),
    }
    if architecture not in factories:
        raise ValueError(f"unsupported metric architecture: {architecture}")
    return factories[architecture]()
