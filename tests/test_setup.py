"""Minimal tests to confirm the package imports and device helper works."""
import random

import numpy as np
import torch

from src.utils import get_device, set_seed


def test_get_device_returns_valid():
    assert get_device() in {"mps", "cuda", "cpu"}


def test_set_seed_runs():
    def draws():
        return random.random(), np.random.rand(3).tolist(), torch.rand(3).tolist()

    set_seed(123)
    first = draws()
    set_seed(123)
    assert draws() == first, "set_seed must make python, numpy and torch draws repeatable"
    set_seed(124)
    assert draws() != first
