"""
Dataset Routing and Configuration Resolution Utilities for XES3G5M and Eedi (Task 3 & 4).

Centralizes dataset detection, dataset-isolated checkpoint/report/log paths,
DataLoader construction, and fold-specific ACD-MMKTv3 auxiliary metadata loading
so all training, evaluation, and ablation scripts operate identically across datasets.
"""

import os
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from src.data.eedi_dataset import get_eedi_dataloaders, load_eedi_static_item_bank
from src.data.xes3g5m_dataset import get_xes3g5m_dataloaders, load_static_item_bank_optimized


def resolve_dataset_key(
    dataset_cfg: Optional[Dict[str, Any]] = None,
    dataset_arg: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> str:
    """
    Determine canonical dataset key ('eedi' or 'xes3g5m') from CLI flag, dataset config, or path.
    """
    if dataset_arg:
        arg_low = dataset_arg.strip().lower()
        if arg_low.startswith("eedi"):
            return "eedi"
        if arg_low.startswith("xes"):
            return "xes3g5m"
        raise ValueError(f"Unsupported dataset identifier: {dataset_arg}")

    if dataset_cfg:
        name = str(dataset_cfg.get("dataset_name", "")).strip().lower()
        if name.startswith("eedi"):
            return "eedi"
        cfg_dir = str(dataset_cfg.get("data_dir", "")).strip().lower()
        if "eedi" in cfg_dir:
            return "eedi"

    if data_dir and "eedi" in str(data_dir).strip().lower():
        return "eedi"

    return "xes3g5m"


def resolve_dataset_paths(
    dataset_key: str,
    dataset_cfg: Optional[Dict[str, Any]] = None,
    train_cfg: Optional[Dict[str, Any]] = None,
    checkpoint_dir: Optional[str] = None,
    output_report_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Resolve dataset-isolated data, checkpoint, report, and log directories.
    Preserves exact legacy XES3G5M paths while isolating Eedi outputs under */eedi/.
    """
    is_eedi = dataset_key == "eedi"
    default_data_dir = "data/processed/Eedi" if is_eedi else "data/processed/XES3G5M"
    data_dir = (
        (dataset_cfg or {}).get("data_dir")
        or (train_cfg or {}).get("data_dir")
        or default_data_dir
    )

    # Isolate Eedi checkpoints/reports/logs unless explicitly overridden by caller
    default_ckpt = "outputs/checkpoints/eedi" if is_eedi else "outputs/checkpoints"
    default_rep = "outputs/reports/eedi" if is_eedi else "outputs/reports"
    default_log = "outputs/logs/eedi" if is_eedi else "outputs/logs"

    if checkpoint_dir is not None and (not is_eedi or checkpoint_dir != "outputs/checkpoints"):
        resolved_ckpt = checkpoint_dir
    else:
        resolved_ckpt = (train_cfg or {}).get("checkpoint_dir") or default_ckpt

    if output_report_dir is not None and (not is_eedi or output_report_dir != "outputs/reports"):
        resolved_rep = output_report_dir
    else:
        resolved_rep = (train_cfg or {}).get("output_report_dir") or default_rep

    if log_dir is not None and (not is_eedi or log_dir != "outputs/logs"):
        resolved_log = log_dir
    else:
        resolved_log = (train_cfg or {}).get("log_dir") or default_log

    display_name = "Eedi NeurIPS 2020 (Task 3 & 4)" if is_eedi else "XES3G5M"
    return {
        "dataset_key": dataset_key,
        "display_name": display_name,
        "data_dir": data_dir,
        "checkpoint_dir": resolved_ckpt,
        "output_report_dir": resolved_rep,
        "log_dir": resolved_log,
    }


def load_static_item_bank_for_dataset(
    dataset_key: str,
    data_dir: str,
    model_type: str = "acd_mmkt_v3",
) -> Dict[str, Any]:
    """Load Tier 1 static item bank using the dataset-appropriate loader."""
    if dataset_key == "eedi":
        return load_eedi_static_item_bank(data_dir, model_type=model_type)
    return load_static_item_bank_optimized(data_dir, model_type=model_type)


def get_dataloaders_for_dataset(
    dataset_key: str,
    data_dir: str,
    val_fold: int = 0,
    model_type: str = "acd_mmkt_v3",
    batch_size: int = 64,
    test_batch_size: int = 128,
    p_drop: float = 0.15,
    test_stride: int = 100,
    num_workers: int = 0,
    test_split: str = "unified",
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """Construct Train, Validation, and Test DataLoaders for either XES3G5M or Eedi."""
    if dataset_key == "eedi":
        return get_eedi_dataloaders(
            data_dir=data_dir,
            val_fold=val_fold,
            model_type=model_type,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            p_drop=p_drop,
            test_stride=test_stride,
            num_workers=num_workers,
            test_split=test_split,
        )
    return get_xes3g5m_dataloaders(
        data_dir=data_dir,
        val_fold=val_fold,
        model_type=model_type,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        p_drop=p_drop,
        test_stride=test_stride,
        num_workers=num_workers,
    )


def resolve_split_tag(dataset_key: str, test_split: Optional[str] = "unified") -> str:
    """Return filename suffix to isolate non-unified Eedi splits without altering defaults."""
    split_val = (test_split or "unified").strip().lower()
    if dataset_key == "eedi" and split_val != "unified":
        return f"_{split_val}"
    return ""


def load_acd_v3_fold_metadata(
    data_dir: str,
    val_fold: int,
    num_questions: int,
    default_error_prob: float = 0.20,
    default_logit_prior: float = -1.37,
) -> Dict[str, Any]:
    """
    Load fold-specific empirical Bayes priors and auxiliary metadata from acd_v3_aux_metadata.pt.
    Falls back cleanly using dataset item count if metadata file is unavailable.
    """
    aux_path = os.path.join(data_dir, "acd_v3_aux_metadata.pt")
    if os.path.exists(aux_path):
        aux_data = torch.load(aux_path, map_location="cpu", weights_only=False)
        return {
            "aux_data": aux_data,
            "aux_path": aux_path,
            "mu_concept_prior": aux_data["mu_concept_folds"][val_fold],
            "mu_prior_prob": aux_data["mu_prior_folds"][val_fold],
            "n_q": aux_data["N_q_folds"][val_fold],
            "d_q": aux_data["D_q_folds"][val_fold],
            "omega_q": aux_data["omega_q_folds"][val_fold],
        }

    return {
        "aux_data": None,
        "aux_path": None,
        "mu_concept_prior": torch.full((num_questions,), default_logit_prior, dtype=torch.float32),
        "mu_prior_prob": torch.full((num_questions,), default_error_prob, dtype=torch.float32),
        "n_q": torch.zeros(num_questions, dtype=torch.int64),
        "d_q": torch.full((num_questions,), default_error_prob, dtype=torch.float32),
        "omega_q": torch.zeros(num_questions, dtype=torch.float32),
    }
