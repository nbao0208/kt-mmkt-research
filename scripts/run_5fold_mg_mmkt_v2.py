"""
CLI Entry Point for Running and Synthesizing 5-Fold Cross-Validation for MG-MMKT v2 (pyKT Compliant).

Features:
1. Sequential Single-Device Execution (Apple Silicon MPS / Single CUDA / CPU).
2. Asynchronous Dual-GPU Scheduling for Multi-GPU setups:
   - Worker 0 on cuda:0: Folds [0, 1, 2]
   - Worker 1 on cuda:1: Folds [3, 4]
3. Eval-only mode to evaluate and synthesize existing fold checkpoints.
4. Computes 5-Fold Single-Model Mean +- Std and 5-Fold Soft Ensemble (pyKT standard).
5. Compares performance directly with MG-MMKT v1 benchmark.
6. Outputs comprehensive Markdown and JSON reports to outputs/reports/.

Usage:
    conda run -n kt-research-env python scripts/run_5fold_mg_mmkt_v2.py --num-epochs 30
    conda run -n kt-research-env python scripts/run_5fold_mg_mmkt_v2.py --eval-only
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
from src.data.xes3g5m_dataset import get_xes3g5m_dataloaders
from src.models.mmkt.mg_mmkt_v2 import MGMMKTv2
from src.training.mg_v2_trainer import MGMMKTv2Trainer
from src.evaluation.metrics import calculate_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="5-Fold Cross-Validation runner for MG-MMKT v2.")
    parser.add_argument("--config-model", type=str, default="configs/model/mg_mmkt_v2.yaml", help="Path to model config")
    parser.add_argument("--config-train", type=str, default="configs/train_mg_mmkt_v2.yaml", help="Path to training config")
    parser.add_argument("--config-dataset", type=str, default="configs/dataset/xes3g5m.yaml", help="Path to dataset config")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Folds to execute (default: 0 1 2 3 4)")
    parser.add_argument("--num-epochs", type=int, default=None, help="Number of epochs per fold")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'mps', 'cpu', or 'auto')")
    parser.add_argument("--dual-gpu", action="store_true", help="Enable Asynchronous Dual-GPU scheduling (cuda:0 folds 0,1,2 + cuda:1 folds 3,4)")
    parser.add_argument("--save-base-dir", type=str, default="outputs/checkpoints", help="Base directory for fold checkpoints")
    parser.add_argument("--output-report-dir", type=str, default="outputs/reports", help="Directory for summary reports")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing fold checkpoints without re-training")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of DataLoader workers (default: 0)")
    parser.add_argument("--worker-only", action="store_true", help="Internal flag: worker only executes assigned folds")

    # Ablation presets
    parser.add_argument("--ablation", type=int, choices=[1, 2, 3, 4, 5, 6], default=None, help="Ablation preset (1..6)")
    parser.add_argument("--wo-lupi", action="store_true", help="Ablation: Disable privileged cognitive distillation (Config 2)")
    parser.add_argument("--wo-dual-pathway", action="store_true", help="Ablation: Disable dual-pathway history (Config 3)")
    parser.add_argument("--wo-glu-ffn", action="store_true", help="Ablation: Disable GLU and post-fusion residual FFN (Config 4)")
    parser.add_argument("--wo-disentangled-qk", action="store_true", help="Ablation: Disable disentangled QK (Config 5a)")
    parser.add_argument("--wo-relative-forgetting", action="store_true", help="Ablation: Disable monotonic forgetting decay (Config 5b)")
    parser.add_argument("--wo-structural-decoupling", action="store_true", help="Ablation: Disable frequency gate and ID dropout (Config 6a)")
    parser.add_argument("--wo-kc-imputation", action="store_true", help="Ablation: Disable KC-conditioned missing vector imputation (Config 6b)")

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
    Train and evaluate a single cross-validation fold for MG-MMKT v2.
    """
    set_seed(base_seed + fold)
    os.makedirs(fold_save_dir, exist_ok=True)
    fold_log_file = os.path.join(fold_save_dir, f"train_fold{fold}.log")
    fold_logger = get_logger(f"run_5fold_v2_fold{fold}", log_file=fold_log_file)

    fold_logger.info("=" * 60)
    fold_logger.info(f"=== Running MG-MMKT v2 Fold {fold}/4 on {device} ===")

    p_drop = float(train_cfg.get("p_drop", 0.20))
    test_stride = int(train_cfg.get("test_stride", 100))
    test_batch_size = int(train_cfg.get("test_batch_size", 128))

    train_loader, val_loader, test_loader, item_bank = get_xes3g5m_dataloaders(
        data_dir=data_dir,
        val_fold=fold,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        p_drop=p_drop,
        test_stride=test_stride,
        num_workers=num_workers,
    )

    ckpt_path = os.path.join(fold_save_dir, "best_model.pt")

    # In eval-only mode, restore exact model configuration from checkpoint prior to model initialization
    if args.eval_only:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path} for eval-only mode.")
        fold_logger.info(f"Loading pre-trained checkpoint from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device)
        if "model_cfg" in ckpt and ckpt["model_cfg"] is not None:
            model_cfg = dict(model_cfg)
            model_cfg.update(ckpt["model_cfg"])
        best_val_auc = ckpt.get("val_auc", 0.0)
        best_epoch = ckpt.get("epoch", 0)

    model = MGMMKTv2(
        num_questions=model_cfg.get("num_questions", 7652),
        num_kcs=model_cfg.get("num_kcs", 1175),
        d_raw_txt=model_cfg.get("d_raw_txt", 768),
        d_raw_img=model_cfg.get("d_raw_img", 768),
        d_model=model_cfg.get("d_model", 128),
        d_ffn=model_cfg.get("d_ffn", 256),
        num_heads=model_cfg.get("num_heads", 4),
        max_seq_len=model_cfg.get("max_seq_len", 200),
        p_id_drop=float(model_cfg.get("p_id_drop", 0.20)),
        dropout=float(model_cfg.get("dropout", 0.20)),
        use_lupi=model_cfg.get("use_lupi", True),
        use_dual_pathway_history=model_cfg.get("use_dual_pathway_history", True),
        use_glu=model_cfg.get("use_glu", True),
        use_post_fusion_ffn=model_cfg.get("use_post_fusion_ffn", True),
        use_disentangled_qk=model_cfg.get("use_disentangled_qk", True),
        use_relative_forgetting=model_cfg.get("use_relative_forgetting", True),
        use_structural_decoupling=model_cfg.get("use_structural_decoupling", True),
        use_kc_imputation=model_cfg.get("use_kc_imputation", True),
        use_solution=model_cfg.get("use_solution", True),
        use_visual=model_cfg.get("use_visual", True),
        static_item_bank=item_bank,
    )

    if not args.eval_only:
        lambda_priv = float(model_cfg.get("lambda_priv", 0.10))
        lambda_gate = float(model_cfg.get("lambda_gate", 0.05))

        trainer = MGMMKTv2Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            lr=lr,
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
            max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
            patience=int(train_cfg.get("patience", 10)),
            lambda_priv=lambda_priv,
            lambda_gate=lambda_gate,
            device=device,
            save_dir=fold_save_dir,
            model_cfg=model.config,
            warmup_epochs=train_cfg.get("warmup_epochs", 2),
            eta_min=float(train_cfg.get("eta_min", 1e-5)),
            logger=fold_logger,
        )
        fit_results = trainer.fit(num_epochs=num_epochs)
        best_val_auc = fit_results["best_val_auc"]
        best_epoch = fit_results["best_epoch"]
    else:
        model.load_state_dict(ckpt["model_state_dict"])

    # Independent Test Set Evaluation
    fold_logger.info("Evaluating fold model on test set and caching probability predictions...")
    model.to(device)
    model.eval()

    test_preds = []
    test_targets = []

    with torch.no_grad():
        for batch in test_loader:
            q_seq = batch["q_seq"].to(device)
            r_seq = batch["r_seq"].to(device)
            seq_mask = batch["seq_mask"].to(device)
            eval_mask = batch["eval_mask"].to(device)
            raw_mod_masks = batch.get("raw_modality_masks", batch.get("modality_masks")).to(device)

            outputs = model(
                q_seq=q_seq,
                r_seq=r_seq,
                seq_mask=seq_mask,
                modality_masks=raw_mod_masks,
                raw_modality_masks=raw_mod_masks,
            )

            probs = outputs["probs"].cpu().numpy()
            targets = r_seq.cpu().numpy()
            eval_mask_np = eval_mask.cpu().numpy().astype(bool)

            valid_probs = probs[eval_mask_np]
            valid_targets = targets[eval_mask_np]

            test_preds.append(valid_probs)
            test_targets.append(valid_targets)

    flat_preds = np.concatenate(test_preds)
    flat_targets = np.concatenate(test_targets)
    test_metrics = calculate_metrics(flat_preds, flat_targets)

    # Cache predictions for ensemble calculation
    preds_cache_path = os.path.join(fold_save_dir, "test_preds.npy")
    targets_cache_path = os.path.join(fold_save_dir, "test_targets.npy")
    np.save(preds_cache_path, flat_preds.astype(np.float32))
    np.save(targets_cache_path, flat_targets.astype(np.int64))

    fold_logger.info(
        f"Fold {fold} Finished -> Best Val AUC: {best_val_auc:.4f} (Epoch {best_epoch}) | "
        f"Test AUC: {test_metrics['auc']:.4f} | Test ACC: {test_metrics['acc']:.4f}"
    )

    return {
        "fold": fold,
        "best_epoch": best_epoch,
        "val_auc": float(best_val_auc),
        "test_auc": float(test_metrics["auc"]),
        "test_acc": float(test_metrics["acc"]),
        "num_test_samples": int(test_metrics["num_samples"]),
        "preds_path": preds_cache_path,
        "targets_path": targets_cache_path,
    }


def synthesize_5fold_results(
    fold_results: List[Dict[str, Any]],
    output_dir: str,
    logger: Any,
) -> Dict[str, Any]:
    """
    Synthesize 5-Fold Cross-Validation metrics, compute Soft Ensemble,
    and compare against MG-MMKT v1 benchmark.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_folds = len(fold_results)

    val_aucs = [r["val_auc"] for r in fold_results]
    test_aucs = [r["test_auc"] for r in fold_results]
    test_accs = [r["test_acc"] for r in fold_results]

    mean_val_auc = float(np.mean(val_aucs))
    std_val_auc = float(np.std(val_aucs))
    mean_test_auc = float(np.mean(test_aucs))
    std_test_auc = float(np.std(test_aucs))
    mean_test_acc = float(np.mean(test_accs))
    std_test_acc = float(np.std(test_accs))

    # 5-Fold Soft Ensemble (pyKT standard: mean probability per interaction)
    all_fold_preds = [np.load(r["preds_path"]) for r in fold_results]
    targets = np.load(fold_results[0]["targets_path"])

    ensemble_probs = np.mean(all_fold_preds, axis=0)
    ensemble_metrics = calculate_metrics(ensemble_probs, targets)

    # Benchmark comparison values for MG-MMKT v1
    v1_benchmark = {
        "mean_test_auc": 0.8290,
        "std_test_auc": 0.0001,
        "ensemble_test_auc": 0.8321,
    }

    diff_mean_auc = mean_test_auc - v1_benchmark["mean_test_auc"]
    diff_ensemble_auc = ensemble_metrics["auc"] - v1_benchmark["ensemble_test_auc"]

    report_data = {
        "model_name": "mg_mmkt_v2",
        "num_folds": num_folds,
        "single_fold_results": fold_results,
        "summary": {
            "val_auc_mean": mean_val_auc,
            "val_auc_std": std_val_auc,
            "test_auc_mean": mean_test_auc,
            "test_auc_std": std_test_auc,
            "test_acc_mean": mean_test_acc,
            "test_acc_std": std_test_acc,
            "ensemble_test_auc": ensemble_metrics["auc"],
            "ensemble_test_acc": ensemble_metrics["acc"],
            "num_test_interactions": ensemble_metrics["num_samples"],
        },
        "comparison_with_v1": {
            "v1_mean_test_auc": v1_benchmark["mean_test_auc"],
            "v2_mean_test_auc": mean_test_auc,
            "diff_mean_test_auc": diff_mean_auc,
            "v1_ensemble_test_auc": v1_benchmark["ensemble_test_auc"],
            "v2_ensemble_test_auc": ensemble_metrics["auc"],
            "diff_ensemble_test_auc": diff_ensemble_auc,
        },
    }

    # Save JSON Report
    json_path = os.path.join(output_dir, "mg_mmkt_v2_5fold_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Save Markdown Report
    md_path = os.path.join(output_dir, "mg_mmkt_v2_5fold_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Báo Cáo Tổng Hợp Kết Quả Thực Nghiệm 5-Fold Cross-Validation: MG-MMKT v2\n\n")
        f.write("## 1. Kết Quả Chi Tiết Từng Fold Trên Tập Test Độc Lập\n\n")
        f.write("| Fold | Epoch Tốt Nhất | Validation AUC | Test ROC-AUC | Test Accuracy | Số Lượng Mẫu Đánh Giá |\n")
        f.write("| :--: | :------------: | :------------: | :----------: | :-----------: | :--------------------: |\n")
        for r in fold_results:
            f.write(f"| Fold {r['fold']} | {r['best_epoch']:02d} | {r['val_auc']:.4f} | {r['test_auc']:.4f} | {r['test_acc']:.4f} | {r['num_test_samples']:,} |\n")
        f.write(f"| **Trung Bình (Mean ± Std)** | - | **{mean_val_auc:.4f} ± {std_val_auc:.4f}** | **{mean_test_auc:.4f} ± {std_test_auc:.4f}** | **{mean_test_acc:.4f} ± {std_test_acc:.4f}** | **{ensemble_metrics['num_samples']:,}** |\n\n")

        f.write("## 2. Hiệu Năng Tổ Hợp Mềm 5-Fold Ensemble và Đối Sánh Trực Tiếp Với MG-MMKT v1\n\n")
        f.write("| Mô Hình | Giao Thức Đánh Giá | Test ROC-AUC | Test Accuracy | Mức Độ Tăng Trưởng (Delta vs v1) |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: |\n")
        f.write(f"| **MG-MMKT v1 (Đường cơ sở)** | 5-Fold Single Mean | {v1_benchmark['mean_test_auc']:.4f} ± {v1_benchmark['std_test_auc']:.4f} | - | Gốc |\n")
        f.write(f"| **MG-MMKT v2 (Đề xuất)** | 5-Fold Single Mean | **{mean_test_auc:.4f} ± {std_test_auc:.4f}** | **{mean_test_acc:.4f} ± {std_test_acc:.4f}** | **+{diff_mean_auc:+.4f}** |\n")
        f.write(f"| **MG-MMKT v1 (Đường cơ sở)** | 5-Fold Soft Ensemble | {v1_benchmark['ensemble_test_auc']:.4f} | - | Gốc |\n")
        f.write(f"| **MG-MMKT v2 (Đề xuất)** | 5-Fold Soft Ensemble | **{ensemble_metrics['auc']:.4f}** | **{ensemble_metrics['acc']:.4f}** | **+{diff_ensemble_auc:+.4f}** |\n\n")

    logger.info("=" * 60)
    logger.info("=== 5-Fold Cross-Validation Synthesis Complete ===")
    logger.info(f"5-Fold Single-Model Test AUC: {mean_test_auc:.4f} +- {std_test_auc:.4f}")
    logger.info(f"5-Fold Soft Ensemble Test AUC: {ensemble_metrics['auc']:.4f} (Delta vs v1: {diff_ensemble_auc:+.4f})")
    logger.info(f"Reports saved to {json_path} and {md_path}")
    logger.info("=" * 60)

    return report_data


def main():
    args = parse_args()
    logger = get_logger("run_5fold_mg_mmkt_v2")

    # Load Configurations
    model_cfg = load_yaml_config(args.config_model)
    train_cfg = load_yaml_config(args.config_train)
    dataset_cfg = load_yaml_config(args.config_dataset)

    # Apply ablation configurations
    if args.ablation == 1:
        # Full MG-MMKT v2
        pass
    elif args.ablation == 2 or args.wo_lupi:
        model_cfg["use_lupi"] = False
        model_cfg["lambda_priv"] = 0.0
    elif args.ablation == 3 or args.wo_dual_pathway:
        model_cfg["use_dual_pathway_history"] = False
    elif args.ablation == 4 or args.wo_glu_ffn:
        model_cfg["use_glu"] = False
        model_cfg["use_post_fusion_ffn"] = False
    elif args.ablation == 5:
        model_cfg["use_disentangled_qk"] = False
        model_cfg["use_relative_forgetting"] = False
    elif args.ablation == 6:
        model_cfg["use_structural_decoupling"] = False
        model_cfg["use_kc_imputation"] = False
        model_cfg["lambda_gate"] = 0.0

    if args.wo_disentangled_qk:
        model_cfg["use_disentangled_qk"] = False
    if args.wo_relative_forgetting:
        model_cfg["use_relative_forgetting"] = False
    if args.wo_structural_decoupling:
        model_cfg["use_structural_decoupling"] = False
    if args.wo_kc_imputation:
        model_cfg["use_kc_imputation"] = False

    # Overrides
    num_epochs = args.num_epochs if args.num_epochs is not None else train_cfg.get("num_epochs", 30)
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 64)
    lr = args.lr if args.lr is not None else float(train_cfg.get("lr", 1e-3))
    base_seed = int(train_cfg.get("seed", 42))
    num_workers = args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 0)
    data_dir = dataset_cfg.get("processed_data_dir", dataset_cfg.get("data_dir", train_cfg.get("data_dir", "data/processed/XES3G5M")))

    # Dual-GPU Execution Handling
    if args.dual_gpu and not args.worker_only:
        logger.info("Spawning Asynchronous Dual-GPU workers: GPU 0 (Folds 0,1,2) & GPU 1 (Folds 3,4)...")
        extra_args = [
            "--config-model", args.config_model,
            "--config-train", args.config_train,
            "--config-dataset", args.config_dataset,
            "--save-base-dir", args.save_base_dir,
            "--num-epochs", str(num_epochs),
            "--batch-size", str(batch_size),
            "--lr", str(lr),
            "--num-workers", str(num_workers),
        ]
        if args.ablation is not None:
            extra_args.extend(["--ablation", str(args.ablation)])
        if args.wo_lupi:
            extra_args.append("--wo-lupi")
        if args.wo_dual_pathway:
            extra_args.append("--wo-dual-pathway")
        if args.wo_glu_ffn:
            extra_args.append("--wo-glu-ffn")
        if args.wo_disentangled_qk:
            extra_args.append("--wo-disentangled-qk")
        if args.wo_relative_forgetting:
            extra_args.append("--wo-relative-forgetting")
        if args.wo_structural_decoupling:
            extra_args.append("--wo-structural-decoupling")
        if args.wo_kc_imputation:
            extra_args.append("--wo-kc-imputation")

        cmd_w0 = [
            sys.executable,
            __file__,
            "--device", "cuda:0",
            "--folds", "0", "1", "2",
            "--worker-only",
        ] + extra_args
        cmd_w1 = [
            sys.executable,
            __file__,
            "--device", "cuda:1",
            "--folds", "3", "4",
            "--worker-only",
        ] + extra_args

        p0 = subprocess.Popen(cmd_w0)
        p1 = subprocess.Popen(cmd_w1)
        p0.wait()
        p1.wait()
        if p0.returncode != 0 or p1.returncode != 0:
            logger.error("Dual-GPU training workers failed.")
            sys.exit(1)
        logger.info("Both Dual-GPU workers completed successfully.")
        # Switch to eval-only mode for main process aggregation so folds are not retrained
        args.eval_only = True

    # Device Selection for Single Worker
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    # Execute assigned folds
    fold_results = []
    for f in args.folds:
        fold_save_dir = os.path.join(args.save_base_dir, f"mg_mmkt_v2_fold{f}")
        res = run_single_fold_training(
            fold=f,
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
        fold_results.append(res)

    if not args.worker_only and len(args.folds) == 5:
        synthesize_5fold_results(
            fold_results=fold_results,
            output_dir=args.output_report_dir,
            logger=logger,
        )


if __name__ == "__main__":
    main()

