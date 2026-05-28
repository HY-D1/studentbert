"""Sanity-check the local environment before writing project code.

Run:  python scripts/check_setup.py
Expected: every line prints a version, MPS reports True on M1, tensor test passes.
"""
import sys


def main() -> None:
    print(f"Python      : {sys.version.split()[0]}")

    import numpy
    print(f"NumPy       : {numpy.__version__}")

    import pandas
    print(f"pandas      : {pandas.__version__}")

    import sklearn
    print(f"scikit-learn: {sklearn.__version__}")

    import torch
    print(f"PyTorch     : {torch.__version__}")
    print(f"MPS avail   : {torch.backends.mps.is_available()}")
    print(f"MPS built   : {torch.backends.mps.is_built()}")
    print(f"CUDA avail  : {torch.cuda.is_available()}  (False on Mac, True on cluster)")

    import transformers
    print(f"transformers: {transformers.__version__}")

    import wandb
    print(f"wandb       : {wandb.__version__}")

    # Pick best available device; this same helper is reused across the project.
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    x = torch.randn(3, 3, device=device)
    y = x @ x.T  # exercise the backend with a matmul
    assert y.shape == (3, 3)
    print(f"Tensor test : OK on '{device}'  (shape {tuple(y.shape)})")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
