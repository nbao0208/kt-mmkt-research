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
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.utils.config import load_yaml_config
from src.data.pykt_dataset import get_pykt_dataloaders
from src.training.pykt_trainer import PyKTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train pyKT baseline models on XES3G5M.")
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
        default="configs/train_pykt.yaml",
        help="Path to training YAML configuration file.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="configs/dataset/xes3g5m_pykt.yaml",
        help="Path to dataset YAML configuration file.",
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
        default=42,
        help="Random seed for reproducibility.",
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
    set_seed(args.seed)

    # Load configurations
    model_config_path = args.model_config or f"configs/model/{args.model}.yaml"
    model_cfg = load_yaml_config(model_config_path)
    train_cfg = load_yaml_config(args.train_config)
    dataset_cfg = load_yaml_config(args.dataset_config)

    # [M3 Fix] Configure log file in outputs/logs/
    log_dir = train_cfg.get("log_dir", "outputs/logs")
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

    logger.info(f"Operating device: {device}")

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
        logger.info(f"5-Fold Training Complete for {args.model.upper()}!")
        logger.info(f"Val AUCs: {val_aucs}")
        logger.info(f"Mean Val AUC: {np.mean(val_aucs):.4f} +/- {np.std(val_aucs):.4f}")


if __name__ == "__main__":
    main()

