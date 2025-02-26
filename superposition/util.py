import hashlib
import json
import random
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel
from torch import Tensor


def get_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def seed_everything(seed: int) -> int:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def normalize(x: Tensor, dim: int = 0, eps: float = 1e-8) -> Tensor:
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
    return x / torch.max(norm, eps * torch.ones_like(norm))


def standardize(x: Tensor, eps: float = 1e-8) -> tuple[Tensor, Tensor, Tensor]:
    mean = x.mean(dim=-1, keepdim=True)
    x = x - mean
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, mean, std


def hash_obj(obj: BaseModel | Any) -> str:
    hash = hashlib.blake2b()
    json_str = obj.model_dump_json() if isinstance(obj, BaseModel) else json.dumps(obj)
    hash.update(json_str.encode())
    return hash.hexdigest()
