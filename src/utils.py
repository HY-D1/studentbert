"""Shared utilities used across the project."""
import random

import numpy as np
import torch


def get_device() -> str:
    """Return the best available device. MPS on M1, CUDA on cluster, else CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_seed(seed: int = 42) -> None:
    """Seed all RNGs for reproducibility. Call once at the start of every run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Note: full determinism on MPS/CUDA is not guaranteed; this covers the common cases.
