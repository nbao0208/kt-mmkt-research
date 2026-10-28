"""
CLI Entry Point for Running and Synthesizing 5-Fold Cross-Validation on XES3G5M (pyKT Compliant).

Supports:
1. Single-Device Sequential Execution (Apple Silicon M1 Pro MPS / Single CUDA GPU / CPU).
2. Dual-GPU Asynchronous Parallel Scheduling (e.g. Kaggle 2x NVIDIA T4):
   - Worker 0 on cuda:0: Folds [0, 1, 2]
   - Worker 1 on cuda:1: Folds [3, 4]
   - Automatically balances load across dual GPUs with zero DDP communication overhead.
3. Eval-only mode to summarize existing pre-trained fold checkpoints.
4. Computes Mean ± Std across 5 folds and formats Markdown and JSON summary tables.

Usage:
    # Single-device sequential execution
    conda run -n kt-research-env python scripts/run_5fold_mg_mmkt.py --num-epochs 30

    # Dual-GPU parallel execution (Kaggle 2x T4)
    conda run -n kt-research-env python scripts/run_5fold_mg_mmkt.py --dual-gpu --num-epochs 30

    # Evaluate existing checkpoints
    conda run -n kt-research-env python scripts/run_5fold_mg_mmkt.py --eval-only --save-base-dir outputs/checkpoints
"""

import argparse
import os
import sys
import json
import time
import subprocess
from typing import Dict, Any, List
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.utils.config import load_yaml_config
from src.data.xes3g5m_dataset import get_xes3g5m_dataloaders, load_static_item_bank_optimized
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
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cuda:0', 'cuda:1', 'mps', 'cpu', or 'auto')")
    parser.add_argument("--dual-gpu", action="store_true", help="Enable Asynchronous Dual-GPU scheduling (cuda:0 folds 0,1,2 + cuda:1 folds 3,4)")
    parser.add_argument("--save-base-dir", type=str, default="outputs/checkpoints", help="Base directory for fold checkpoints")
    parser.add_argument("--output-report-dir", type=str, default="outputs/reports", help="Directory for summary reports")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing fold checkpoints without re-training")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of DataLoader workers (default: from config or 0)")
    parser.add_argument("--worker-only", action="store_true", help="Internal flag: worker only executes assigned folds without synthesizing summary reports")

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


def run_single_fold_training(
    fold: int,
    args: argparse.Namespace,
    model_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    data_dir: str,
    device: torch.device,
    fold_save_dir: str,
    batch_size: int,
    num_epochs: int,
    lr: float,
    num_workers: int,
    base_seed: int,
    logger: Any,
) -> Dict[str, Any]:
    """
    Train and evaluate a single cross-validation fold for MG-MMKT or SAKT.
    """
    set_seed(base_seed + fold)
    os.makedirs(fold_save_dir, exist_ok=True)
    fold_log_file = os.path.join(fold_save_dir, f"train_fold{fold}.log")
    fold_logger = get_logger(f"run_5fold_fold{fold}", log_file=fold_log_file)

    fold_logger.info("=" * 60)
    fold_logger.info(f"            EXECUTING FOLD {fold} / 5 on {device}          ")
    fold_logger.info("=" * 60)

    train_loader, val_loader, test_loader, item_bank = get_xes3g5m_dataloaders(
        data_dir=data_dir,
        val_fold=fold,
        batch_size=batch_size,
        test_batch_size=train_cfg.get("test_batch_size", 128),
        p_drop=dataset_cfg.get("p_drop", 0.20),
        test_stride=dataset_cfg.get("test_stride", 100),
        num_workers=num_workers,
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
            return None
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

        # Persist training summary JSON for downstream synthesis and plotting
        fit_summary["best_val_acc"] = float(best_val_acc)
        fit_summary["test_metrics"] = {"auc": float(test_auc), "acc": float(test_acc)}
        summary_path = os.path.join(fold_save_dir, "training_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f_out:
            json.dump(fit_summary, f_out, indent=2)

    fold_record = {
        "fold": fold,
        "val_auc": float(best_val_auc),
        "val_acc": float(best_val_acc),
        "test_auc": float(test_auc),
        "test_acc": float(test_acc),
    }
    fold_logger.info(f"Fold {fold} Result: Val AUC: {best_val_auc:.4f} | Test AUC: {test_auc:.4f} | Test ACC: {test_acc:.4f}")
    logger.info(f"Fold {fold} Result: Val AUC: {best_val_auc:.4f} | Test AUC: {test_auc:.4f} | Test ACC: {test_acc:.4f}")
    return fold_record


def evaluate_existing_fold(
    fold: int,
    ckpt_path: str,
    data_dir: str,
    model_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    device: torch.device,
    model_type: str,
    logger: Any,
) -> Dict[str, Any]:
    """
    Evaluate an existing checkpoint without retraining.
    """
    _, val_loader, test_loader, item_bank = get_xes3g5m_dataloaders(
        data_dir=data_dir,
        val_fold=fold,
        batch_size=128,
        test_batch_size=128,
        p_drop=0.0,
        test_stride=dataset_cfg.get("test_stride", 100),
        num_workers=0,
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    saved_cfg = ckpt.get("model_cfg") or {}
    effective_cfg = {**model_cfg, **saved_cfg}

    if model_type == "sakt":
        model = IDOnlySAKT(
            num_questions=effective_cfg.get("num_questions", 7652),
            num_kcs=effective_cfg.get("num_kcs", 1175),
            d_model=effective_cfg.get("d_model", 128),
            num_heads=effective_cfg.get("num_heads", 4),
            max_seq_len=effective_cfg.get("max_seq_len", 200),
            dropout=effective_cfg.get("dropout", 0.20),
        ).to(device)
    else:
        model = MGMMKT(
            num_questions=effective_cfg.get("num_questions", 7652),
            num_kcs=effective_cfg.get("num_kcs", 1175),
            d_raw_txt=effective_cfg.get("d_raw_txt", 768),
            d_raw_img=effective_cfg.get("d_raw_img", 768),
            d_model=effective_cfg.get("d_model", 128),
            num_heads=effective_cfg.get("num_heads", 4),
            max_seq_len=effective_cfg.get("max_seq_len", 200),
            dropout=effective_cfg.get("dropout", 0.20),
            use_solution=effective_cfg.get("use_solution", True),
            use_visual=effective_cfg.get("use_visual", True),
            use_learnable_imputation=effective_cfg.get("use_learnable_imputation", True),
            use_gated_fusion=effective_cfg.get("use_gated_fusion", True),
            fusion_mode=effective_cfg.get("fusion_mode", "gated"),
            static_item_bank=item_bank,
        ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    trainer = MGMMKTTrainer(
        model=model,
        train_loader=val_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        save_dir=os.path.dirname(ckpt_path),
        model_cfg=effective_cfg,
        logger=logger,
    )

    val_res = trainer.evaluate(val_loader, split_name=f"Fold-{fold} Val")
    test_res = trainer.evaluate(test_loader, split_name=f"Fold-{fold} Test")

    record = {
        "fold": fold,
        "val_auc": float(val_res["auc"]),
        "val_acc": float(val_res["acc"]),
        "test_auc": float(test_res["auc"]),
        "test_acc": float(test_res["acc"]),
    }
    logger.info(f"Fold {fold} Evaluated: Val AUC: {record['val_auc']:.4f} | Test AUC: {record['test_auc']:.4f}")
    return record


def main():
    args = parse_args()
    logger = get_logger("run_5fold")
    logger.info("=== 5-Fold Cross-Validation Protocol for MG-MMKT (pyKT Compliant) ===")

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
    num_workers = args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 0)

    # Apply ablation overrides
    if args.ablation == 1:
        model_cfg["use_solution"] = True
        model_cfg["use_visual"] = True
        model_cfg["use_learnable_imputation"] = True
        model_cfg["use_gated_fusion"] = True
        model_cfg["fusion_mode"] = "gated"
    elif args.ablation == 2:
        model_cfg["use_solution"] = False
    elif args.ablation == 3:
        model_cfg["use_visual"] = False
    elif args.ablation == 4:
        model_cfg["use_learnable_imputation"] = False
        model_cfg["use_gated_fusion"] = False
        model_cfg["fusion_mode"] = "naive_concat"

    fold_results: List[Dict[str, Any]] = []

    # 3. Branching: Dual-GPU parallel vs Sequential execution
    if args.dual_gpu and torch.cuda.device_count() >= 2 and len(args.folds) >= 2 and not args.eval_only and not args.worker_only:
        # Ensure mg_static_item_bank.pt exists on disk before spawning parallel workers to avoid race conditions
        load_static_item_bank_optimized(data_dir, model_type="mg_mmkt")

        mid = (len(args.folds) + 1) // 2
        folds_gpu0 = args.folds[:mid]
        folds_gpu1 = args.folds[mid:]

        logger.info("Initializing Asynchronous Dual-GPU 5-Fold Execution:")
        logger.info(f"- GPU 0 (cuda:0): Folds {folds_gpu0} | batch_size={batch_size}, epochs={num_epochs}")
        logger.info(f"- GPU 1 (cuda:1): Folds {folds_gpu1} | batch_size={batch_size}, epochs={num_epochs}")

        # Launch GPU 1 process in background
        cmd_gpu1 = [
            sys.executable,
            __file__,
            "--device", "cuda:1",
            "--folds", *[str(f) for f in folds_gpu1],
            "--batch-size", str(batch_size),
            "--num-epochs", str(num_epochs),
            "--save-base-dir", str(args.save_base_dir),
            "--output-report-dir", str(args.output_report_dir),
            "--config-model", str(args.config_model),
            "--config-train", str(args.config_train),
            "--config-dataset", str(args.config_dataset),
            "--worker-only",
        ]
        if args.lr is not None:
            cmd_gpu1.extend(["--lr", str(args.lr)])
        if args.num_workers is not None:
            cmd_gpu1.extend(["--num-workers", str(args.num_workers)])
        if args.ablation is not None:
            cmd_gpu1.extend(["--ablation", str(args.ablation)])
        if args.model_type != "mg_mmkt":
            cmd_gpu1.extend(["--model-type", str(args.model_type)])

        logger.info(f"Spawning background worker on cuda:1: {' '.join(cmd_gpu1)}")
        proc_gpu1 = subprocess.Popen(cmd_gpu1)

        # Run GPU 0 folds in current process
        for f in folds_gpu0:
            fold_save_dir = os.path.join(args.save_base_dir, f"mg_mmkt_fold{f}")
            res = run_single_fold_training(
                fold=f,
                args=args,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                dataset_cfg=dataset_cfg,
                data_dir=data_dir,
                device=torch.device("cuda:0"),
                fold_save_dir=fold_save_dir,
                batch_size=batch_size,
                num_epochs=num_epochs,
                lr=lr,
                num_workers=num_workers,
                base_seed=base_seed,
                logger=logger,
            )
            if res:
                fold_results.append(res)

        # Wait for GPU 1 to complete
        logger.info(f"Waiting for cuda:1 worker (Folds {folds_gpu1}) to finish...")
        proc_gpu1.wait()
        if proc_gpu1.returncode != 0:
            logger.error(f"Worker on cuda:1 exited with non-zero code {proc_gpu1.returncode}.")

        # Evaluate/retrieve results for GPU 1 folds
        for f in folds_gpu1:
            ckpt_file = os.path.join(args.save_base_dir, f"mg_mmkt_fold{f}", "best_model.pt")
            summary_file = os.path.join(args.save_base_dir, f"mg_mmkt_fold{f}", "training_summary.json")
            if os.path.exists(summary_file):
                try:
                    with open(summary_file, "r", encoding="utf-8") as f_sum:
                        s_data = json.load(f_sum)
                    best_ep_idx = max(0, int(s_data.get("best_epoch", 1)) - 1)
                    history_list = s_data.get("history", [])
                    best_val_acc = s_data.get(
                        "best_val_acc",
                        history_list[best_ep_idx].get("val_acc", 0.0) if best_ep_idx < len(history_list) else 0.0,
                    )
                    fold_results.append({
                        "fold": f,
                        "val_auc": float(s_data.get("best_val_auc", 0.0)),
                        "val_acc": float(best_val_acc),
                        "test_auc": float(s_data.get("test_metrics", {}).get("auc", 0.0)),
                        "test_acc": float(s_data.get("test_metrics", {}).get("acc", 0.0)),
                    })
                    continue
                except Exception as exc:
                    logger.warning(f"Failed to parse {summary_file}: {exc}. Falling back to checkpoint evaluation.")
            if os.path.exists(ckpt_file):
                res = evaluate_existing_fold(
                    fold=f,
                    ckpt_path=ckpt_file,
                    data_dir=data_dir,
                    model_cfg=model_cfg,
                    dataset_cfg=dataset_cfg,
                    device=torch.device("cuda:0"),
                    model_type=args.model_type,
                    logger=logger,
                )
                fold_results.append(res)

    else:
        # Standard sequential execution across specified folds
        for fold in args.folds:
            fold_save_dir = os.path.join(args.save_base_dir, f"mg_mmkt_fold{fold}")
            if args.eval_only:
                ckpt_path = os.path.join(fold_save_dir, "best_model.pt")
                if not os.path.exists(ckpt_path):
                    logger.warning(f"Checkpoint for fold {fold} not found at {ckpt_path}, skipping.")
                    continue
                res = evaluate_existing_fold(
                    fold=fold,
                    ckpt_path=ckpt_path,
                    data_dir=data_dir,
                    model_cfg=model_cfg,
                    dataset_cfg=dataset_cfg,
                    device=device,
                    model_type=args.model_type,
                    logger=logger,
                )
            else:
                res = run_single_fold_training(
                    fold=fold,
                    args=args,
                    model_cfg=model_cfg,
                    train_cfg=train_cfg,
                    dataset_cfg=dataset_cfg,
                    data_dir=data_dir,
                    device=device,
                    fold_save_dir=fold_save_dir,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    lr=lr,
                    num_workers=num_workers,
                    base_seed=base_seed,
                    logger=logger,
                )
            if res:
                fold_results.append(res)

    if args.worker_only:
        logger.info("Worker process completed assigned folds successfully.")
        return

    if not fold_results:
        logger.error("No fold results were collected.")
        return

    # 4. Compute Mean ± Std across executed folds
    fold_results = sorted(fold_results, key=lambda x: x["fold"])
    val_aucs = [r["val_auc"] for r in fold_results]
    val_accs = [r["val_acc"] for r in fold_results]
    test_aucs = [r["test_auc"] for r in fold_results]
    test_accs = [r["test_acc"] for r in fold_results]

    summary = {
        "num_folds": len(fold_results),
        "model_type": args.model_type,
        "ablation": args.ablation,
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

    # 5. Print Summary Table
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

    # 6. Save Report
    os.makedirs(args.output_report_dir, exist_ok=True)
    report_name = f"5fold_summary_abl{args.ablation}.json" if args.ablation else "5fold_summary.json"
    report_json_path = os.path.join(args.output_report_dir, report_name)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary report saved to {report_json_path}")


if __name__ == "__main__":
    main()
