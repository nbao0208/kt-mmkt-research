"""
Factory and initialization module for pyKT baseline models (DKT, DKVMN, SAKT).

Strictly initializes models directly from the pyKT library (pykt.models) without
re-implementing the architectures from scratch, ensuring 100% adherence to pyKT benchmarks.
Synchronizes internal pyKT device variables across CUDA, MPS, and CPU.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn

# [H3 Fix] Unconditionally alias FloatTensor and LongTensor in torch.cuda to CPU types
# to prevent accidental CUDA allocation in DataLoader/pickle operations
torch.cuda.FloatTensor = torch.FloatTensor
torch.cuda.LongTensor = torch.LongTensor

import pykt.models.utils as p_utils
from pykt.models.dkt import DKT
from pykt.models.dkvmn import DKVMN
from pykt.models.sakt import SAKT


def sync_pykt_device(device: torch.device) -> None:
    """
    Synchronize global device setting inside pyKT modules.
    
    pyKT hardcodes `device = "cpu" if not torch.cuda.is_available() else "cuda"`,
    which causes device mismatch errors when running on Apple Silicon (MPS).
    This function updates pykt.models.utils.device to match the execution device.
    
    Args:
        device: Target PyTorch torch.device.
    """
    p_utils.device = device
    import pykt.models.init_model as p_init
    p_init.device = device
    import pykt.models.evaluate_model as p_eval
    p_eval.device = device


def init_pykt_baseline(
    model_name: str,
    model_config: Dict[str, Any],
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Initialize a Knowledge Tracing baseline model directly from pyKT.

    Supported models:
    - 'dkt': Deep Knowledge Tracing (LSTM on concept interactions).
    - 'dkvmn': Dynamic Key-Value Memory Networks.
    - 'sakt': Self-Attentive Knowledge Tracing.

    Args:
        model_name: Name of the baseline model ('dkt', 'dkvmn', 'sakt').
        model_config: Dictionary containing model architecture hyperparameters.
        device: Target device for execution (default: auto-detected).

    Returns:
        Instantiated nn.Module moved to the target device.
    """
    model_name = model_name.lower().strip()

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # Synchronize device within pyKT internals
    sync_pykt_device(device)

    num_c = model_config.get("num_c", 865)
    dropout = model_config.get("dropout", 0.20)
    emb_type = model_config.get("emb_type", "qid")

    if model_name == "dkt":
        emb_size = model_config.get("emb_size", 128)
        model = DKT(
            num_c=num_c,
            emb_size=emb_size,
            dropout=dropout,
            emb_type=emb_type,
        )
    elif model_name == "dkvmn":
        dim_s = model_config.get("dim_s", 128)
        size_m = model_config.get("size_m", 50)
        model = DKVMN(
            num_c=num_c,
            dim_s=dim_s,
            size_m=size_m,
            dropout=dropout,
            emb_type=emb_type,
        )
    elif model_name == "sakt":
        seq_len = model_config.get("seq_len", 200)
        emb_size = model_config.get("emb_size", 128)
        num_attn_heads = model_config.get("num_attn_heads", 4)
        num_en = model_config.get("num_en", 2)
        model = SAKT(
            num_c=num_c,
            seq_len=seq_len,
            emb_size=emb_size,
            num_attn_heads=num_attn_heads,
            dropout=dropout,
            num_en=num_en,
            emb_type=emb_type,
        )
    else:
        raise ValueError(
            f"Unsupported pyKT baseline model '{model_name}'. "
            f"Expected one of ['dkt', 'dkvmn', 'sakt']."
        )

    return model.to(device)

