import json
import os
from typing import Self

import torch
import torch.nn as nn
from pydantic import BaseModel
from torch import Tensor

from superposition.util import hash_obj


class BaseAutoencoder(nn.Module):
    encoder: nn.Linear
    decoder: nn.Linear

    def __init__(self, config: BaseModel) -> None:
        super().__init__()
        self.config = config

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def save(self, dataset_config: BaseModel, train_config: BaseModel) -> None:
        obj = {
            "autoencoder": self.config.model_dump(),
            "dataset": dataset_config.model_dump(),
            "train": train_config.model_dump(),
        }
        dir = os.path.join("models", hash_obj(obj))
        os.makedirs(dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(dir, "autoencoder.pt"))
        with open(os.path.join(dir, "config.json"), "w") as f:
            f.write(json.dumps(obj, indent=2))

    @classmethod
    def load(
        cls,
        config: BaseModel,
        dataset_config: BaseModel,
        train_config: BaseModel,
        device: torch.device,
    ) -> Self:
        obj = {
            "autoencoder": config.model_dump(),
            "dataset": dataset_config.model_dump(),
            "train": train_config.model_dump(),
        }
        dir = os.path.join("models", hash_obj(obj))
        with open(os.path.join(dir, "config.json")) as f:
            assert obj == json.load(f), "config mismatch"
        model = cls(config).to(device)
        model.load_state_dict(
            torch.load(
                os.path.join(dir, "autoencoder.pt"),
                map_location=device,
                weights_only=True,
            )
        )
        return model

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
