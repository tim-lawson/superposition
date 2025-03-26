# https://github.com/EleutherAI/pythia

import os

import numpy as np
import torch
from numpy.typing import ArrayLike
from pydantic import BaseModel
from torch import Tensor
from torch.utils.data import Dataset
from transformers import AutoTokenizer, GPTNeoXModel


class PythiaDatasetConfig(BaseModel):
    model_name: str = "EleutherAI/pythia-14m"
    revision: str = "main"


class PythiaDataset(Dataset):
    def __init__(self, config: PythiaDatasetConfig) -> None:
        self.config = config
        self.samples = load(config.model_name, config.revision)[1]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        return self.samples[idx], self.samples[idx]


def load(model_name: str, revision: str) -> tuple[ArrayLike, Tensor]:
    dir, name = "data/pythia", model_name.replace("/", "_") + f"_{revision}"
    if not os.path.exists(os.path.join(dir, f"{name}.features.npy")):
        features, samples = load_model(model_name, revision)
        save_arr(features, samples, dir, name)
    else:
        features, samples = load_arr(dir, name)
    return features, samples


@torch.no_grad()
def load_model(model_name: str, revision: str) -> tuple[ArrayLike, Tensor]:
    model = GPTNeoXModel.from_pretrained(model_name, revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    features = np.array(list(v for k, v in tokenizer.get_vocab().items()))
    samples = model.embed_in.weight.data[: len(features), :]
    return features, samples


def load_arr(save_dir: str, name: str) -> tuple[ArrayLike, Tensor]:
    return np.load(os.path.join(save_dir, f"{name}.features.npy")), torch.load(
        os.path.join(save_dir, f"{name}.samples.pt"), weights_only=True
    )


def save_arr(features: ArrayLike, samples: Tensor, save_dir: str, name: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"{name}.features.npy"), features)
    torch.save(samples, os.path.join(save_dir, f"{name}.samples.pt"))


if __name__ == "__main__":
    for model_name in ["EleutherAI/pythia-14m"]:
        load(model_name, "main")
