import torch
from pydantic import PositiveFloat
from torch import Tensor
from torch.utils.data import Dataset

from .base import DatasetConfig


class GaussianDatasetConfig(DatasetConfig):
    mean: float
    std: PositiveFloat


class GaussianDataset(Dataset):
    def __init__(self, config: GaussianDatasetConfig) -> None:
        self.config = config
        self.samples = (
            torch.randn(config.num_samples, config.num_inputs) * config.std
            + config.mean
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        return self.samples[idx], self.samples[idx]
