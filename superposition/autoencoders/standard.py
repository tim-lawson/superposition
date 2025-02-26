import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, PositiveInt
from torch import Tensor

from superposition.util import standardize

from .base import BaseAutoencoder


class AutoencoderConfig(BaseModel):
    num_inputs: PositiveInt
    num_latents: PositiveInt


class Autoencoder(BaseAutoencoder):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__(config)
        self.encoder = nn.Linear(config.num_inputs, config.num_latents)
        self.decoder = nn.Linear(config.num_latents, config.num_inputs, bias=False)
        nn.init.orthogonal_(self.decoder.weight)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x, mean, std = standardize(x)
        encoded = F.relu(self.encoder.forward(x))
        self.unit_norm_decoder()
        decoded = self.decoder.forward(encoded)
        decoded = decoded * std + mean
        return decoded, encoded

    def unit_norm_decoder(self) -> None:
        with torch.no_grad():
            norm = torch.norm(self.decoder.weight, dim=0, keepdim=True)
            self.decoder.weight.data = self.decoder.weight.data / norm
