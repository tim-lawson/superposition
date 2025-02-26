import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, PositiveInt
from torch import Tensor

from superposition.util import standardize

from .base import BaseAutoencoder


class TopKAutoencoderConfig(BaseModel):
    num_inputs: PositiveInt
    num_latents: PositiveInt
    k: PositiveInt


def topk(x: Tensor, k: int) -> Tensor:
    values, indices = torch.topk(x, k=k, sorted=False)
    return values.new_zeros(x.shape).scatter_(-1, indices, values)


class TopKAutoencoder(BaseAutoencoder):
    def __init__(self, config: TopKAutoencoderConfig) -> None:
        super().__init__(config)
        self.k = config.k
        self.pre_encoder_bias = nn.Parameter(torch.zeros(config.num_inputs))
        self.encoder = nn.Linear(config.num_inputs, config.num_latents, bias=False)
        self.decoder = nn.Linear(config.num_latents, config.num_inputs, bias=False)
        self.decoder.weight.data = self.encoder.weight.data.T.clone().T.contiguous().T
        self.unit_norm_decoder()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x, mean, std = standardize(x)
        encoded = F.relu(topk(self.encoder.forward(x - self.pre_encoder_bias), self.k))
        self.unit_norm_decoder()
        self.unit_norm_decoder_gradient()
        decoded = self.decoder.forward(encoded) * std + mean
        return decoded, encoded

    def unit_norm_decoder(self) -> None:
        with torch.no_grad():
            norm = torch.norm(self.decoder.weight, dim=0, keepdim=True)
            self.decoder.weight.data = self.decoder.weight.data / norm

    def unit_norm_decoder_gradient(self) -> None:
        if self.decoder.weight.grad is not None:
            self.decoder.weight.grad -= einops.einsum(
                einops.einsum(
                    self.decoder.weight.grad,
                    self.decoder.weight,
                    "... n d, ... n d -> ... d",
                ),
                self.decoder.weight,
                "... d, ... n d -> ... n d",
            )
