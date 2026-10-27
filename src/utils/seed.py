"""
Random Seed Management Utility for Reproducibility in KT and MMKT Experiments.
Ensures deterministic behavior across Python, NumPy, PyTorch CPU, MPS, and CUDA.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds across all libraries to guarantee deterministic reproducibility.

    Args:
        seed: Integer seed value (default: 42).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS currently does not have separate manual_seed_all, covered by torch.manual_seed
        pass
