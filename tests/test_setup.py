"""Minimal tests to confirm the package imports and device helper works."""
from src.utils import get_device, set_seed


def test_get_device_returns_valid():
    assert get_device() in {"mps", "cuda", "cpu"}


def test_set_seed_runs():
    set_seed(123)  # should not raise
