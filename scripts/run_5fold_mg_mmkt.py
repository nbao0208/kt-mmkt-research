"""
CLI Entry Point for Running and Synthesizing 5-Fold Cross-Validation on XES3G5M.

Implements Plan Step 6.1.1:
- Trains or evaluates across all 5 cross-validation folds (Folds 0..4).
- Gathers Validation and Test ROC-AUC and Accuracy per fold.
- Computes and prints Mean ± Std summary table.
- Saves structured reports in JSON and Markdown formats.

Usage:
    conda run -n kt-research-env python scripts/run_5fold_mg_mmkt.py --num-epochs 30
    conda run -n kt-research-env python scripts/run_5fold_mg_mmkt.py --eval-only --checkpoint-dir outputs/checkpoints
"""

import argparse
import os
import sys
import json
from typing import Dict, Any, List
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.utils.config import load_yaml_config
from src.data.xes3g5m_dataset import get_xes3g5m_dataloaders
from src.models.mmkt.mg_mmkt import MGMMKT
from src.models.baselines.sakt import IDOnlySAKT
from src.training.trainer import MGMMKTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="5-Fold Cross-Validation runner for KT models.")
    parser.add_argument("--config-model", type=str, default="configs/model/mg_mmkt.yaml", help="Path to model config")
    parser.add_argument("--config-train", type=str, default="configs/train.yaml", help="Path to training config")
    parser.add_argument("--config-dataset", type=str, default="configs/dataset/xes3g5m.yaml", help="Path to dataset config")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Folds to execute (default: 0 1 2 3 4)")
    parser.add_argument("--num-epochs", type=int, default=None, help="Number of epochs per fold")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'mps', 'cpu', or 'auto')")
    parser.add_argument("--save-base-dir", type=str, default="outputs/checkpoints", help="Base directory for fold checkpoints")
    parser.add_argument("--output-report-dir", type=str, default="outputs/reports", help="Directory for summary reports")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing fold checkpoints without re-training")

    # Ablation presets
    parser.add_argument(
        "--ablation",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Ablation preset: 1=Full, 2=w/o Solution, 3=w/o Visual, 4=Naive Concat",
    )
    parser.add_argument("--model-type", type=str, choices=["mg_mmkt", "sakt"], default="mg_mmkt", help="Model type to train")

    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger("run_5fold")
    logger.info("=== 5-Fold Cross-Validation Protocol (pyKT Compliant) ===")

    # 1. Device Selection
    device_str = args.device
    if device_str == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = torch.device(device_str)
    logger.info(f"Compute Device: {device}")

    # 2. Load Base Configurations
    model_cfg = load_yaml_config(args.config_model)
    train_cfg = load_yaml_config(args.config_train)
    dataset_cfg = load_yaml_config(args.config_dataset)

    num_epochs = args.num_epochs if args.num_epochs is not None else train_cfg.get("num_epochs", 30)
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 64)
    lr = args.lr if args.lr is not None else float(train_cfg.get("lr", 1e-3))
    base_seed = train_cfg.get("seed", 42)
    data_dir = dataset_cfg.get("data_dir", "data/processed/XES3G5M")

    # Apply ablation overrides
    if args.ablation == 2:
        model_cfg["use_solution"] = False
    elif args.ablation == 3:
        model_cfg["use_visual"] = False
    elif args.ablation == 4:
        model_cfg["use_learnable_imputation"] = False
        model_cfg["use_gated_fusion"] = False
        model_cfg["fusion_mode"] = "naive_concat"

    fold_results = []

    for fold in args.folds:
        set_seed(base_seed + fold)
        fold_save_dir = os.path.join(args.save_base_dir, f"mg_mmkt_fold{fold}")
        os.makedirs(fold_save_dir, exist_ok=True)
        fold_log_file = os.path.join(fold_save_dir, f"train_fold{fold}.log")
        fold_logger = get_logger(f"run_5fold_fold{fold}", log_file=fold_log_file)

        fold_logger.info(f"\n=======================================================")
        fold_logger.info(f"            EXECUTING FOLD {fold} / 5                 ")
        fold_logger.info(f"=======================================================")

        train_loader, val_loader, test_loader, item_bank = get_xes3g5m_dataloaders(
            data_dir=data_dir,
            val_fold=fold,
            batch_size=batch_size,
            test_batch_size=train_cfg.get("test_batch_size", 128),
            p_drop=dataset_cfg.get("p_drop", 0.20),
            test_stride=dataset_cfg.get("test_stride", 100),
            num_workers=train_cfg.get("num_workers", 0),
        )

        # Build Model
        if args.model_type == "sakt":
            model = IDOnlySAKT(
                num_questions=model_cfg.get("num_questions", 7652),
                num_kcs=model_cfg.get("num_kcs", 1175),
                d_model=model_cfg.get("d_model", 128),
                num_heads=model_cfg.get("num_heads", 4),
                max_seq_len=model_cfg.get("max_seq_len", 200),
                dropout=model_cfg.get("dropout", 0.20),
            ).to(device)
        else:
            model = MGMMKT(
                num_questions=model_cfg.get("num_questions", 7652),
                num_kcs=model_cfg.get("num_kcs", 1175),
                d_raw_txt=model_cfg.get("d_raw_txt", 768),
                d_raw_img=model_cfg.get("d_raw_img", 768),
                d_model=model_cfg.get("d_model", 128),
                num_heads=model_cfg.get("num_heads", 4),
                max_seq_len=model_cfg.get("max_seq_len", 200),
                dropout=model_cfg.get("dropout", 0.20),
                use_solution=model_cfg.get("use_solution", True),
                use_visual=model_cfg.get("use_visual", True),
                use_learnable_imputation=model_cfg.get("use_learnable_imputation", True),
                use_gated_fusion=model_cfg.get("use_gated_fusion", True),
                fusion_mode=model_cfg.get("fusion_mode", "gated"),
                static_item_bank=item_bank,
            ).to(device)

        trainer = MGMMKTTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            lr=lr,
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
            max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
            patience=int(train_cfg.get("patience", 10)),
            device=device,
            save_dir=fold_save_dir,
            model_cfg=model_cfg,
            warmup_epochs=train_cfg.get("warmup_epochs", 2),
            eta_min=float(train_cfg.get("eta_min", 1e-5)),
            logger=fold_logger,
        )

        best_ckpt_file = os.path.join(fold_save_dir, "best_model.pt")

        if args.eval_only:
            if not os.path.exists(best_ckpt_file):
                fold_logger.warning(f"Checkpoint {best_ckpt_file} not found. Skipping fold {fold}.")
                logger.warning(f"Checkpoint {best_ckpt_file} not found. Skipping fold {fold}.")
                continue
            fold_logger.info(f"Loading weights from {best_ckpt_file}...")
            ckpt = torch.load(best_ckpt_file, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            val_res = trainer.evaluate(val_loader, split_name=f"Fold-{fold} Val")
            test_res = trainer.evaluate(test_loader, split_name=f"Fold-{fold} Test")
            best_val_auc = val_res["auc"]
            best_val_acc = val_res["acc"]
            test_auc = test_res["auc"]
            test_acc = test_res["acc"]
        else:
            fit_summary = trainer.fit(num_epochs=num_epochs)
            best_val_auc = fit_summary["best_val_auc"]
            best_val_acc = fit_summary["history"][fit_summary["best_epoch"] - 1]["val_acc"]
            test_metrics = fit_summary["test_metrics"] or trainer.evaluate(test_loader, split_name=f"Fold-{fold} Test")
            test_auc = test_metrics["auc"]
            test_acc = test_metrics["acc"]

        fold_record = {
            "fold": fold,
            "val_auc": float(best_val_auc),
            "val_acc": float(best_val_acc),
            "test_auc": float(test_auc),
            "test_acc": float(test_acc),
        }
        fold_results.append(fold_record)
        fold_logger.info(f"Fold {fold} Result: Val AUC: {best_val_auc:.4f} | Test AUC: {test_auc:.4f} | Test ACC: {test_acc:.4f}")
        logger.info(f"Fold {fold} Result: Val AUC: {best_val_auc:.4f} | Test AUC: {test_auc:.4f} | Test ACC: {test_acc:.4f}")

    if not fold_results:
        logger.error("No fold results were collected.")
        return

    # 3. Compute Mean ± Std across executed folds
    val_aucs = [r["val_auc"] for r in fold_results]
    val_accs = [r["val_acc"] for r in fold_results]
    test_aucs = [r["test_auc"] for r in fold_results]
    test_accs = [r["test_acc"] for r in fold_results]

    summary = {
        "num_folds": len(fold_results),
        "folds": fold_results,
        "mean_val_auc": float(np.mean(val_aucs)),
        "std_val_auc": float(np.std(val_aucs)),
        "mean_val_acc": float(np.mean(val_accs)),
        "std_val_acc": float(np.std(val_accs)),
        "mean_test_auc": float(np.mean(test_aucs)),
        "std_test_auc": float(np.std(test_aucs)),
        "mean_test_acc": float(np.mean(test_accs)),
        "std_test_acc": float(np.std(test_accs)),
    }

    # 4. Print Summary Table
    logger.info("\n" + "=" * 65)
    logger.info("           5-FOLD CROSS-VALIDATION SYNTHESIS REPORT            ")
    logger.info("=" * 65)
    logger.info(f"{'Fold':<8} | {'Val AUC':<10} | {'Val ACC':<10} | {'Test AUC':<10} | {'Test ACC':<10}")
    logger.info("-" * 65)
    for r in fold_results:
        logger.info(f"Fold {r['fold']:<3} | {r['val_auc']:<10.4f} | {r['val_acc']:<10.4f} | {r['test_auc']:<10.4f} | {r['test_acc']:<10.4f}")
    logger.info("-" * 65)
    logger.info(f"{'Mean':<8} | {summary['mean_val_auc']:<10.4f} | {summary['mean_val_acc']:<10.4f} | {summary['mean_test_auc']:<10.4f} | {summary['mean_test_acc']:<10.4f}")
    logger.info(f"{'Std':<8} | ±{summary['std_val_auc']:<9.4f} | ±{summary['std_val_acc']:<9.4f} | ±{summary['std_test_auc']:<9.4f} | ±{summary['std_test_acc']:<9.4f}")
    logger.info("=" * 65)

    # 5. Save Report
    os.makedirs(args.output_report_dir, exist_ok=True)
    report_json_path = os.path.join(args.output_report_dir, "5fold_summary.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary report saved to {report_json_path}")


if __name__ == "__main__":
    main()
