import torch.nn as nn
from torch import Tensor


class FFN(nn.Module):
    def __init__(
        self, in_size: int, out_size: int, hidden_size: int, depth: int
    ) -> None:
        super().__init__()
        layers = []
        current_size = in_size

        for _ in range(depth - 1):
            layers.extend([nn.Linear(current_size, hidden_size), nn.ReLU()])
            current_size = hidden_size

        layers.append(nn.Linear(current_size, out_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers.forward(x)
