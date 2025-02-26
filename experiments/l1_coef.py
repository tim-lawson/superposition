import os
from dataclasses import dataclass
from pprint import pprint

import numpy as np
import pandas as pd
import torch
from simple_parsing import Serializable, parse
from tqdm import tqdm

from superposition.autoencoders import Autoencoder, AutoencoderConfig
from superposition.datasets import (
    GaussianDataset,
    GaussianDatasetConfig,
    SharkeyDataset,
    SharkeyDatasetConfig,
)
from superposition.ffn import FFN
from superposition.trainer import TrainConfig, test, train
from superposition.util import get_device, seed_everything


@dataclass
class Args(Serializable):
    # autoencoder
    num_latents: int

    # experiment
    train: bool = True
    test: bool = True
    ffn: bool = False
    gaussian: bool = False

    # sweep
    num_seeds: int = 10
    num_l1_coefs: int = 21

    # dataset
    num_samples: int = 10000
    num_inputs: int = 256
    num_features: int = 512
    avg_active_features: float = 5
    lambda_decay: float = 0.99

    test_split: float = 0.1

    @property
    def filename(self) -> str:
        filename = "sharkey_"
        if self.gaussian:
            filename += "gaussian_"
        if self.ffn:
            filename += "ffn_"
        filename += "l1_coef_"
        filename += str(self.num_latents)
        return filename


def main(args: Args) -> None:
    pprint(args)
    device = get_device()

    config = AutoencoderConfig(
        num_inputs=args.num_inputs,
        num_latents=args.num_latents,
    )

    results = []
    metric_keys = []

    for seed in tqdm(range(args.num_seeds), total=args.num_seeds, unit="seeds"):
        l1_coefs = np.logspace(start=-3, stop=2, num=args.num_l1_coefs)
        for l1_coef in tqdm(l1_coefs, total=len(l1_coefs), unit="coefs", leave=False):
            train_config = TrainConfig(seed=seed, l1_coef=l1_coef)
            seed_everything(train_config.seed)

            dataset = SharkeyDataset(
                config=SharkeyDatasetConfig(
                    num_samples=args.num_samples,
                    num_inputs=args.num_inputs,
                    num_features=args.num_features,
                    avg_active_features=args.avg_active_features,
                    lambda_decay=args.lambda_decay,
                ),
            )

            dataset.samples = dataset.samples.to(device)
            features = dataset.features.to(device)

            if args.gaussian:
                mean = dataset.samples.mean().item()
                std = dataset.samples.std().item()

                dataset = GaussianDataset(
                    config=GaussianDatasetConfig(
                        num_samples=args.num_samples,
                        num_inputs=args.num_inputs,
                        mean=mean,
                        std=std,
                    ),
                )

                features = None

            if args.ffn:
                ffn = FFN(
                    in_size=args.num_inputs,
                    out_size=args.num_inputs,
                    hidden_size=args.num_inputs * 4,
                    depth=2,
                ).to(device)

                with torch.no_grad():
                    dataset.samples = ffn.forward(dataset.samples)

                    if features is not None:
                        features = ffn.forward(features.T).T

            train_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [1 - args.test_split, args.test_split]
            )

            if args.train:
                model = Autoencoder(config).to(device)
                train(model, train_dataset, train_config)
                model.save(dataset.config, train_config)

            if args.test:
                model = Autoencoder.load(config, dataset.config, train_config, device)

                metrics = test(model, test_dataset, train_config, features)
                # pprint(metrics)

                results.append({**train_config.__dict__, **metrics})
                metric_keys = list(metrics.keys())

    os.makedirs("results/l1_coef", exist_ok=True)
    dataframe = pd.DataFrame(results)
    dataframe.to_csv(f"results/l1_coef/{args.filename}.csv", index=False)

    dataframe = dataframe.groupby("l1_coef").agg(
        {metric_key: ["mean", "std", "sem"] for metric_key in metric_keys}
    )
    dataframe.columns = dataframe.columns.map("_".join)
    dataframe.to_csv(f"results/l1_coef/{args.filename}_summary.csv")


if __name__ == "__main__":
    main(parse(Args))
