import os

import pandas as pd
import torch
from torch import nn
from transformers import GPTNeoXModel


@torch.no_grad()
def main() -> None:
    dir = "results/norm"
    os.makedirs(dir, exist_ok=True)
    rows: list[dict] = []

    for model_name in [
        "EleutherAI/pythia-14m",
        "EleutherAI/pythia-70m-deduped",
        "EleutherAI/pythia-160m-deduped",
        "EleutherAI/pythia-410m-deduped",
        "EleutherAI/pythia-1b-deduped",
        "EleutherAI/pythia-1.4b-deduped",
        "EleutherAI/pythia-2.8b-deduped",
        "EleutherAI/pythia-6.9b-deduped",
    ]:
        model_main = GPTNeoXModel.from_pretrained(model_name, revision="main")
        model_step0 = GPTNeoXModel.from_pretrained(model_name, revision="step0")

        model_rows: list[dict] = []
        for name, param_main in model_main.named_parameters():
            param_step0: nn.Parameter = model_step0.state_dict()[name]
            model_rows.append(
                {
                    "name": name,
                    "count": param_main.numel(),
                    "mean_main": param_main.mean().item(),
                    "std_main": param_main.std().item(),
                    "mean_step0": param_step0.mean().item(),
                    "std_step0": param_step0.std().item(),
                    "abs_mean_main": param_main.abs().mean().item(),
                    "abs_std_main": param_main.abs().std().item(),
                    "abs_mean_step0": param_step0.abs().mean().item(),
                    "abs_std_step0": param_step0.abs().std().item(),
                }
            )
        df = pd.DataFrame(model_rows)
        df.to_csv(os.path.join(dir, f"{model_name.replace('/', '_')}.csv"), index=False)

        count = sum(df["count"])
        rows.append(
            {
                "model_name": model_name,
                "mean_main": sum(df["mean_main"] * df["count"]) / count,
                "mean_step0": sum(df["mean_step0"] * df["count"]) / count,
                "abs_mean_main": sum(df["abs_mean_main"] * df["count"]) / count,
                "abs_mean_step0": sum(df["abs_mean_step0"] * df["count"]) / count,
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(dir, "summary.csv"), index=False)


if __name__ == "__main__":
    main()
