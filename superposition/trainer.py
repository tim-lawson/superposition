import torch
import torch.nn.functional as F
import torch.optim as optim
from pydantic import BaseModel, PositiveFloat, PositiveInt
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from superposition.autoencoders import BaseAutoencoder
from superposition.util import normalize


class TrainConfig(BaseModel):
    seed: int = 0
    num_epochs: PositiveInt = 100
    batch_size: PositiveInt = 256
    lr: PositiveFloat = 1e-3
    l1_coef: PositiveFloat = 0.1


def train(model: BaseAutoencoder, dataset: Dataset, config: TrainConfig) -> None:
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    with tqdm(
        desc="train", total=config.num_epochs, unit="epochs", leave=False
    ) as pbar:
        for _epoch in range(config.num_epochs):
            metrics = {
                "loss": [],
            }

            input: Tensor
            for input, _label in dataloader:
                input = input.to(model.device)
                decoded, encoded = model.forward(input)

                mse = F.mse_loss(decoded, input)
                l1_loss = config.l1_coef * torch.mean(torch.abs(encoded))
                loss = mse + l1_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                metrics["loss"].append(loss.item())

            pbar.set_postfix({k: sum(v) / len(v) for k, v in metrics.items()})
            pbar.update(1)

    pbar.close()


@torch.no_grad()
def test(
    model: BaseAutoencoder,
    dataset: Dataset,
    config: TrainConfig,
    features: Tensor | None = None,
) -> dict[str, float]:
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    metrics = {
        metric: []
        for metric in [
            "mse",
            "fvu",
            "l0_norm",
            "l1_norm",
            "l2_norm",
            "l1_over_sqrt_l2",
            "hoyer",
        ]
    }

    input: Tensor
    for input, _label in dataloader:
        input = input.to(model.device)
        decoded, encoded = model.forward(input)

        mse = F.mse_loss(decoded, input)
        fvu = mse / torch.var(input)

        l0_norm = torch.gt(encoded, 0.0).float().sum(dim=1).mean()
        l1_norm = torch.abs(encoded).sum(dim=1).mean()
        l2_norm = torch.norm(encoded, p=2, dim=1).mean()
        l1_over_sqrt_l2 = l1_norm / torch.sqrt(l2_norm)

        sqrt_n = torch.sqrt(torch.tensor(encoded.size(1), device=encoded.device))
        hoyer = ((sqrt_n - l1_norm / l2_norm) / (sqrt_n - 1)).mean()

        metrics["mse"].append(mse.item())
        metrics["fvu"].append(fvu.item())
        metrics["l0_norm"].append(l0_norm.item())
        metrics["l1_norm"].append(l1_norm.item())
        metrics["l2_norm"].append(l2_norm.item())
        metrics["l1_over_sqrt_l2"].append(l1_over_sqrt_l2.item())
        metrics["hoyer"].append(hoyer.item())

    metrics = {k: sum(v) / len(v) for k, v in metrics.items()}

    if features is not None:
        cosine_similarity = features.T @ normalize(model.decoder.weight.data)
        max_cosine_similarity = torch.max(cosine_similarity, dim=1).values
        metrics["mmcs"] = float(max_cosine_similarity.mean().item())

    return metrics
