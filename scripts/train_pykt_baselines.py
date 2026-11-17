"""
Training script for pyKT baseline models (DKT, DKVMN, SAKT) on XES3G5M.

Executes 5-fold cross-validation or single fold training using pyKT's native architectures.

Usage:
    # Train Fold 0 for DKT
    conda run -n kt-research-env python scripts/train_pykt_baselines.py --model dkt --fold 0

    # Train all 5 folds for DKVMN
    conda run -n kt-research-env python scripts/train_pykt_baselines.py --model dkvmn --fold -1
"""

import os
import sys
import argparse
import json
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.utils.config import load_yaml_config
from src.data.dataset_router import resolve_dataset_key, resolve_dataset_paths
from src.data.pykt_dataset import get_pykt_dataloaders
from src.training.pykt_trainer import PyKTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train pyKT baseline models on XES3G5M or Eedi.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["xes3g5m", "eedi"],
        help="Target dataset shortcut ('xes3g5m' or 'eedi').",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["dkt", "dkvmn", "sakt"],
        help="Baseline model architecture identifier.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold index (0..4) or -1 to train all 5 folds sequentially.",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default=None,
        help="Path to model YAML configuration file.",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default=None,
        help="Path to training YAML configuration file.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Path to dataset YAML configuration file.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Optional override for checkpoint base directory.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override maximum number of epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override training batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Override early stopping patience.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution device ('auto', 'cuda', 'mps', 'cpu').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (defaults to train_config seed, 42).",
    )
    return parser.parse_args()


def train_single_fold(
    model_name: str,
    fold: int,
    model_cfg: dict,
    train_cfg: dict,
    dataset_cfg: dict,
    device: torch.device,
    logger,
) -> dict:
    """Train a single fold of the specified pyKT model."""
    logger.info(f"========== Initiating {model_name.upper()} Training for Fold {fold} ==========")
    raw_data_dir = dataset_cfg.get("raw_data_dir", "data/raw/XES3G5M/kc_level")
    train_valid_file = dataset_cfg.get("train_valid_file", "train_valid_sequences.csv")
    batch_size = train_cfg.get("batch_size", 256)
    num_workers = train_cfg.get("num_workers", 0)

    train_loader, val_loader = get_pykt_dataloaders(
        raw_data_dir=raw_data_dir,
        train_valid_file=train_valid_file,
        fold=fold,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    trainer = PyKTTrainer(
        model_name=model_name,
        model_config=model_cfg,
        train_config=train_cfg,
        fold=fold,
        device=device,
        logger=logger,
    )

    results = trainer.fit(train_loader, val_loader)

    # Save fold history JSON
    history_file = os.path.join(trainer.save_dir, "training_history.json")
    with open(history_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"========== Completed {model_name.upper()} Fold {fold} | Best Val AUC: {results['best_val_auc']:.4f} ==========")
    return results


def main():
    args = parse_args()

    # Load dataset config first to resolve dataset key and isolated defaults
    dataset_config_path = args.dataset_config or (
        "configs/dataset/eedi_pykt.yaml" if args.dataset == "eedi" else "configs/dataset/xes3g5m_pykt.yaml"
    )
    dataset_cfg = load_yaml_config(dataset_config_path)
    dataset_key = resolve_dataset_key(dataset_cfg=dataset_cfg, dataset_arg=args.dataset)

    train_config_path = args.train_config or (
        "configs/train_pykt_eedi.yaml"
        if (dataset_key == "eedi" and os.path.exists("configs/train_pykt_eedi.yaml"))
        else "configs/train_pykt.yaml"
    )
    model_config_path = args.model_config or f"configs/model/{args.model}.yaml"

    model_cfg = load_yaml_config(model_config_path)
    train_cfg = load_yaml_config(train_config_path)

    paths = resolve_dataset_paths(
        dataset_key=dataset_key,
        dataset_cfg=dataset_cfg,
        train_cfg=train_cfg,
        checkpoint_dir=args.save_dir,
    )
    train_cfg["save_dir"] = paths["checkpoint_dir"]

    # Dynamically override concept count and max sequence length from dataset config
    if "num_c" in dataset_cfg:
        model_cfg["num_c"] = int(dataset_cfg["num_c"])
    elif "num_leaf_kcs" in dataset_cfg:
        model_cfg["num_c"] = int(dataset_cfg["num_leaf_kcs"])
    if args.model == "sakt" and "max_seq_len" in dataset_cfg:
        model_cfg["seq_len"] = int(dataset_cfg["max_seq_len"])
    model_cfg["dataset_name"] = paths["display_name"]

    seed = int(args.seed) if args.seed is not None else int(train_cfg.get("seed", 42))
    set_seed(seed)

    # Configure log file in dataset-isolated logs directory
    log_dir = paths["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{args.model}.log")
    logger = get_logger(f"TrainPyKT_{args.model.upper()}", log_file=log_file)

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    logger.info(
        f"Dataset: {paths['display_name']} (key={dataset_key}) | num_c={model_cfg['num_c']} | "
        f"save_dir={train_cfg['save_dir']} | Operating device: {device}"
    )

    # Overrides
    if args.epochs is not None:
        train_cfg["num_epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    if args.patience is not None:
        train_cfg["patience"] = args.patience

    folds_to_train = [args.fold] if args.fold >= 0 else [0, 1, 2, 3, 4]
    all_results = []

    for f in folds_to_train:
        res = train_single_fold(
            model_name=args.model,
            fold=f,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            dataset_cfg=dataset_cfg,
            device=device,
            logger=logger,
        )
        all_results.append(res)

    if len(all_results) > 1:
        val_aucs = [r["best_val_auc"] for r in all_results]
        logger.info(f"5-Fold Training Complete for {args.model.upper()} on {paths['display_name']}!")
        logger.info(f"Val AUCs: {val_aucs}")
        logger.info(f"Mean Val AUC: {np.mean(val_aucs):.4f} +/- {np.std(val_aucs):.4f}")


if __name__ == "__main__":
    main()

