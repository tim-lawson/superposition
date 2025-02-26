import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor

from superposition.datasets.lomax import lomax, sign
from superposition.ffn import FFN
from superposition.util import seed_everything

figsize = (1.5, 1.5)


def plot_inputs(sparse_inputs: Tensor, dense_inputs: Tensor, lim: float) -> None:
    plt.rcParams.update({"axes.linewidth": 0})

    fig = plt.figure(figsize=figsize, dpi=600)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.scatter(
        sparse_inputs[:, 0].detach(),
        sparse_inputs[:, 1].detach(),
        zs=sparse_inputs[:, 2].detach(),
        s=0.5,
        alpha=0.5,
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)  # type: ignore
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])  # type: ignore

    plt.savefig(
        "results/2d/sparse_inputs.png", format="png", bbox_inches="tight", pad_inches=0
    )
    plt.close(fig)

    fig = plt.figure(figsize=figsize, dpi=600)
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.scatter(
        dense_inputs[:, 0].detach(), dense_inputs[:, 1].detach(), s=0.1, alpha=0.5
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_axis_off()

    plt.savefig(
        "results/2d/dense_inputs.png", format="png", bbox_inches="tight", pad_inches=0
    )
    plt.close(fig)


def plot_outputs(sparse_outputs: Tensor, dense_outputs: Tensor, lim: float) -> None:
    fig = plt.figure(figsize=figsize, dpi=600)
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.scatter(
        dense_outputs[:, 0].detach(), dense_outputs[:, 1].detach(), s=0.1, alpha=0.5
    )
    ax.set_xlim(-lim / 2, lim)
    ax.set_ylim(-lim / 2, lim)
    ax.set_axis_off()

    fig.savefig(
        "results/2d/dense_outputs.png", format="png", bbox_inches="tight", pad_inches=0
    )
    plt.close(fig)

    fig = plt.figure(figsize=figsize, dpi=600)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.scatter(
        sparse_outputs[:, 0].detach(),
        sparse_outputs[:, 1].detach(),
        zs=sparse_outputs[:, 2].detach(),
        s=0.5,
        alpha=0.5,
    )
    ax.set_xlim(-lim / 2, lim)
    ax.set_ylim(-lim / 2, lim)
    ax.set_zlim(-lim / 2, lim)  # type: ignore
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])  # type: ignore

    plt.savefig(
        "results/2d/sparse_outputs.png", format="png", bbox_inches="tight", pad_inches=0
    )
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    seed = 0
    num_features = 3
    num_inputs = 2
    num_samples = 10000
    scale = 1.0
    alpha = 1.0

    seed_everything(seed)

    sparse_to_dense = nn.Linear(num_features, num_inputs, bias=False)
    dense_to_sparse = torch.pinverse(sparse_to_dense.weight.data)

    sparse_inputs = lomax((num_samples, num_features), scale, alpha)
    sparse_inputs *= sign((num_samples, num_features))

    dense_inputs = sparse_to_dense.forward(sparse_inputs)

    ffn = FFN(num_inputs, num_inputs, num_inputs * 4, depth=2)

    def linear_sparse(x: Tensor) -> Tensor:
        return torch.mm(
            dense_to_sparse,
            ffn.forward(torch.mm(sparse_to_dense.weight, x.T).T).T,
        ).T

    dense_outputs = ffn.forward(dense_inputs)
    sparse_outputs = linear_sparse(sparse_inputs)

    os.makedirs("results/2d", exist_ok=True)

    plot_inputs(sparse_inputs, dense_inputs, lim=25)
    plot_outputs(sparse_outputs, dense_outputs, lim=25)

    pd.DataFrame(sparse_inputs.numpy()).to_csv(
        "results/2d/sparse_inputs.csv", header=["x", "y", "z"], index=False
    )
    pd.DataFrame(dense_inputs.numpy()).to_csv(
        "results/2d/dense_inputs.csv", header=["x", "y"], index=False
    )
    pd.DataFrame(sparse_outputs.numpy()).to_csv(
        "results/2d/sparse_outputs.csv", header=["x", "y", "z"], index=False
    )
    pd.DataFrame(dense_outputs.numpy()).to_csv(
        "results/2d/dense_outputs.csv", header=["x", "y"], index=False
    )


if __name__ == "__main__":
    main()
