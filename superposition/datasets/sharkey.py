import torch
from pydantic import PositiveFloat, PositiveInt
from torch import Tensor
from torch.distributions import MultivariateNormal, Normal
from torch.utils.data import Dataset

from .base import DatasetConfig

STANDARD_NORMAL = Normal(0, 1)


@torch.no_grad()
def generate_sharkey(
    num_samples: int,
    num_inputs: int,
    num_features: int,
    avg_active_features: float,
    lambda_decay: float,
) -> tuple[Tensor, Tensor]:
    """
    Lee Sharkey, Dan Braun, and Beren Millidge. Taking features out of
    superposition with sparse autoencoders, December 2022. URL
    https://www.lesswrong.com/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition#Toy_dataset_generation

    Args:
        num_samples (int): The number of samples to generate.

        num_inputs (int): The number of input dimensions (h = 256 in the paper).

        num_features (int): The number of ground truth features (G = 512 in the paper).

        avg_active_features (float): The average number of ground truth features that
            are active at a time (5 in the paper).

        lambda_decay (float): The exponential decay factor for feature probabilities
            (λ = 0.99 in the paper).
    """

    features = torch.randn(num_inputs, num_features)
    features /= torch.norm(features, dim=0, keepdim=True)

    covariance = torch.randn(num_features, num_features)
    covariance = covariance @ covariance.T
    correlated_normal = MultivariateNormal(
        torch.zeros(num_features), covariance_matrix=covariance
    )

    samples = []
    for _ in range(num_samples):
        p = STANDARD_NORMAL.cdf(correlated_normal.sample())
        p = p ** (lambda_decay * torch.arange(num_features))
        p = p * (avg_active_features / (num_features * torch.mean(p)))

        coef = torch.bernoulli(p.clamp(0, 1)) * torch.rand(num_features)

        sample = coef @ features.T
        samples.append(sample)

    return torch.stack(samples), features


class SharkeyDatasetConfig(DatasetConfig):
    num_features: PositiveInt = 512
    avg_active_features: PositiveFloat = 5
    lambda_decay: PositiveFloat = 0.99


class SharkeyDataset(Dataset):
    def __init__(self, config: SharkeyDatasetConfig) -> None:
        self.config = config
        self.samples, self.features = generate_sharkey(
            config.num_samples,
            config.num_inputs,
            config.num_features,
            config.avg_active_features,
            config.lambda_decay,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        return self.samples[idx], self.samples[idx]
