import torch
import torch.nn as nn
from pydantic import BaseModel, PositiveFloat, PositiveInt
from torch import Tensor
from torch.distributions import Pareto
from torch.utils.data import Dataset

from superposition.util import normalize


def sign(size: tuple[int, ...]) -> Tensor:
    return torch.bernoulli(torch.full(size, 0.5)) * 2 - 1


def pareto(size: tuple[int, ...], scale: float, alpha: float) -> Tensor:
    return Pareto(torch.tensor(scale).float(), torch.tensor(alpha).float()).sample(size)  # type: ignore


def lomax(size: tuple[int, ...], scale: float, alpha: float) -> Tensor:
    return pareto(size, scale, alpha) - 1


def generate_lomax(
    num_samples: int, num_inputs: int, num_features: int, scale: float, alpha: float
) -> tuple[Tensor, Tensor]:
    # Kaiming uniform initialization
    features = nn.Linear(num_features, num_inputs, bias=False).weight.detach()

    coef = lomax((num_samples, num_features), scale, alpha)
    coef *= sign((num_samples, num_features))

    samples = coef @ features.T

    return samples, normalize(features)


class LomaxDatasetConfig(BaseModel):
    num_features: PositiveInt
    scale: PositiveFloat
    alpha: PositiveFloat


class LomaxDataset(Dataset):
    def __init__(
        self, num_samples: int, num_inputs: int, config: LomaxDatasetConfig
    ) -> None:
        self.config = config
        self.samples, self.features = generate_lomax(
            num_samples, num_inputs, config.num_features, config.scale, config.alpha
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        return self.samples[idx], self.samples[idx]
