# A toy model of superposition

This repository accompanies Section 4 of the preprint "Sparse Autoencoders Can Interpret Randomly Initialized Transformers" (<https://arxiv.org/abs/2501.17727>).
For the code that accompanies the rest of the paper, see <https://github.com/ThomasHeap/random_sae>.

We speculated in Section 1 of the paper that the activations of randomized transformers could appear 'interpretable' because the input data exhibits superposition preserved by (even randomized) neural networks, or because neural networks amplify or introduce superposition into the input data.
In Section 4, we trained SAEs on toy data designed to exhibit superposition ([Sharkey et al., 2022](https://www.lesswrong.com/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition)) and GloVe word vectors ([Pennington et al., 2014](https://aclanthology.org/D14-1162/)).

## Installation

1. Create a Python virtual environment and install dependencies. I recommend [uv](https://github.com/astral-sh/uv).
2. Download `https://nlp.stanford.edu/data/glove.6B.zip` and extract it to the `data` subdirectory.

## Reproducing figures

To reproduce Figure 4, run `python experiments/2d.py`.
This script generates CSV data and PNG plots in the `results/2d` subdirectory.

To reproduce Figures 6 and 13–16, run `./l1_coef.sh`.
This script generates trained autoencoders in the `models` subdirectory and CSV data in the `results/l1_coef` subdirectory.
Each call to `python experiments/l1_coef.py` takes about 80 minutes on an RTX 3090 GPU and there are 16 calls.
For each number of latents and dataset, it trains autoencoders with 10 different random seeds and 21 different L1 coefficients.

To reproduce Figures 7 and 17–20, run `./glove.sh`.
This script generates trained autoencoders in the `models` subdirectory and CSV data in the `results/glove` subdirectory.
Each call to `python experiments/glove.py` takes about X minutes on an RTX 3090 GPU and there are 64 calls.
For each number of latents and dataset, it trains autoencoders with 21 different L1 coefficients.

To reproduce Figures X, run `./pythia.sh`.
This script generates trained autoencoders in the `models` subdirectory and CSV data in the `results/pythia` subdirectory.
Each call to `python experiments/pythia.py` takes about 20 minutes on an RTX 3090 GPU and there are 64 calls.
For each number of latents and dataset, it trains autoencoders with 21 different L1 coefficients.
