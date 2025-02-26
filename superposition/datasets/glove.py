# https://nlp.stanford.edu/projects/glove/
# https://nlp.stanford.edu/data/glove.6B.zip

import os

import numpy as np
import torch
from numpy.typing import ArrayLike
from pydantic import PositiveInt
from torch import Tensor
from torch.utils.data import Dataset

from .base import DatasetConfig


class GloveDatasetConfig(DatasetConfig):
    num_inputs: PositiveInt = 50


class GloveDataset(Dataset):
    def __init__(self, config: GloveDatasetConfig) -> None:
        self.config = config
        self.samples = load(config.num_inputs)[1]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        return self.samples[idx], self.samples[idx]


def load(dim: int) -> tuple[ArrayLike, Tensor]:
    dir, file, name = "data/glove.6B", f"glove.6B.{dim}d.txt", f"glove.6B.{dim}d"
    if not os.path.exists(os.path.join(dir, f"{name}.features.npy")):
        features, samples = load_txt(os.path.join(dir, file))
        save_arr(features, samples, dir, name)
    else:
        features, samples = load_arr(dir, name)
    return features, samples


def load_txt(file: str) -> tuple[ArrayLike, Tensor]:
    features, samples = [], []
    with open(file, encoding="utf-8") as f:
        for line in f:
            features.append(line.strip().split()[0])
            samples.append([float(x) for x in line.strip().split()[1:]])
    return np.array(features), torch.tensor(samples, dtype=torch.float)


def load_arr(save_dir: str, name: str) -> tuple[ArrayLike, Tensor]:
    return np.load(os.path.join(save_dir, f"{name}.features.npy")), torch.load(
        os.path.join(save_dir, f"{name}.samples.pt"), weights_only=True
    )


def save_arr(features: ArrayLike, samples: Tensor, save_dir: str, name: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"{name}.features.npy"), features)
    torch.save(samples, os.path.join(save_dir, f"{name}.samples.pt"))


if __name__ == "__main__":
    for d in [50, 100, 200, 300]:
        load(d)
