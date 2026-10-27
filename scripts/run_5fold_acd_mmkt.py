"""
CLI Entry Point for 5-Fold Cross-Validation Protocol on ACD-MMKT (pyKT Compliant).

Supports:
1. Single-Device Sequential Execution (Apple Silicon M1 Pro MPS / Single CUDA GPU).
2. Dual-GPU Asynchronous Parallel Scheduling:
   - Worker 0 on cuda:0 (RTX 5060 Ti): Folds [0, 1, 2] with batch_size=128, grad_accum=1.
   - Worker 1 on cuda:1 (RTX 3060): Folds [3, 4] with batch_size=64, grad_accum=2.
   - Automatically balances load across heterogeneous GPUs without DDP straggler wait.
3. Eval-only mode to summarize pre-trained fold checkpoints.
4. Computes Mean ± Std across 5 folds and formats Markdown and JSON summary tables.

Usage:
    # Run 5-fold CV sequentially on current device
    conda run -n kt-research-env python scripts/run_5fold_acd_mmkt.py --num-epochs 30

    # Run dual-GPU parallel 5-fold CV on workstation
    conda run -n kt-research-env python scripts/run_5fold_acd_mmkt.py --dual-gpu --num-epochs 30

    # Evaluate existing checkpoints
    conda run -n kt-research-env python scripts/run_5fold_acd_mmkt.py --eval-only --checkpoint-dir outputs/checkpoints
"""

import argparse
import os
import sys
import json
import subprocess
import time
from typing import Dict, Any, List
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.utils.config import load_yaml_config
from src.data.xes3g5m_dataset import get_xes3g5m_dataloaders
from src.models.mmkt.acd_mmkt import ACDMMKT
from src.training.acd_trainer import ACDMMKTTrainer
from src.evaluation.metrics import calculate_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="5-Fold Cross-Validation runner for ACD-MMKT.")
    parser.add_argument("--config-model", type=str, default="configs/model/acd_mmkt.yaml", help="Path to model config")
    parser.add_argument("--config-train", type=str, default="configs/train_acd_mmkt.yaml", help="Path to training config")
    parser.add_argument("--config-dataset", type=str, default="configs/dataset/xes3g5m.yaml", help="Path to dataset config")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Folds to execute (default: 0 1 2 3 4)")
    parser.add_argument("--num-epochs", type=int, default=None, help="Number of Stage 2 epochs per fold")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'mps', 'cpu', or 'auto')")
    parser.add_argument("--dual-gpu", action="store_true", help="Enable Asynchronous Dual-GPU scheduling (cuda:0 folds 0,1,2 + cuda:1 folds 3,4)")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints", help="Base directory for fold checkpoints")
    parser.add_argument("--output-report-dir", type=str, default="outputs/reports", help="Directory for summary reports")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing fold checkpoints without retraining")
    parser.add_argument("--skip-stage1", action="store_true", help="Skip Stage 1 pre-training")

    parser.add_argument("--grad-accum-steps", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--precision", type=str, choices=["fp32", "amp_fp16", "amp_bf16"], default=None, help="Computation precision")
    parser.add_argument("--profile", type=str, choices=["m1_pro", "dual_gpu_5060ti", "dual_gpu_3060", "kaggle_t4"], default=None, help="Hardware execution profile")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of DataLoader worker subprocesses")
    parser.add_argument("--worker-only", action="store_true", help="Internal flag: worker only executes assigned folds without synthesizing summary reports")

    # Ablation presets
    parser.add_argument(
        "--ablation",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="Ablation preset: 1=Full, 2=w/o Distillation, 3=w/o Fine-Grained, 4=w/o HGNN/Rasch, 5=w/o DeepIRT",
    )

    return parser.parse_args()


def run_single_fold_training(
    fold: int,
    args: argparse.Namespace,
    model_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    data_dir: str,
    device: torch.device,
    save_dir: str,
    batch_size: int,
    grad_accum_steps: int,
    precision: str,
    logger: Any,
) -> Dict[str, Any]:
    """
    Train and evaluate a single cross-validation fold.
    """
    logger.info(f"--- Launching Fold {fold} on {device} (BatchSize={batch_size}, GradAccum={grad_accum_steps}, Precision={precision}) ---")
    set_seed(train_cfg.get("seed", 42) + fold)

    logger.info(f"Step 1/4: Loading XES3G5M sequence datasets and item bank for Fold {fold}...")
    train_loader, val_loader, test_loader, item_bank = get_xes3g5m_dataloaders(
        data_dir=data_dir,
        val_fold=fold,
        model_type="acd_mmkt",
        batch_size=batch_size,
        test_batch_size=train_cfg.get("test_batch_size", 128),
        p_drop=float(train_cfg.get("p_drop", 0.20)),
        test_stride=int(train_cfg.get("test_stride", 100)),
        num_workers=int(train_cfg.get("num_workers", 0)),
    )

    logger.info(f"Step 2/4: Initializing ACDMMKT model and transferring static item bank to {device}...")
    model = ACDMMKT(
        num_questions=model_cfg.get("num_questions", 7652),
        num_kcs=model_cfg.get("num_kcs", 1175),
        d_raw_txt=model_cfg.get("d_raw_txt", 768),
        d_raw_img=model_cfg.get("d_raw_img", 768),
        d_ctx=model_cfg.get("d_ctx", 21),
        d_model=model_cfg.get("d_model", 128),
        num_heads=model_cfg.get("num_heads", 4),
        max_seq_len=model_cfg.get("max_seq_len", 200),
        dropout=model_cfg.get("dropout", 0.20),
        hgnn_layers=model_cfg.get("hgnn_layers", 2),
        tau_temp=model_cfg.get("tau_temp", 0.10),
        beta_distill=model_cfg.get("beta_distill", 0.10),
        beta_ctt=model_cfg.get("beta_ctt", 0.20),
        beta_disen=model_cfg.get("beta_disen", 0.05),
        lambda_res=model_cfg.get("lambda_res", 1e-3),
        irt_scale=model_cfg.get("irt_scale", 4.0),
        use_solution=model_cfg.get("use_solution", True),
        use_visual=model_cfg.get("use_visual", True),
        use_distillation=model_cfg.get("use_distillation", True),
        use_fine_grained=model_cfg.get("use_fine_grained", True),
        use_hypergraph_rasch=model_cfg.get("use_hypergraph_rasch", True),
        use_deep_irt=model_cfg.get("use_deep_irt", True),
        static_item_bank=item_bank,
    ).to(device)

    # Immediately release CPU item bank reference and run garbage collection
    del item_bank
    import gc
    gc.collect()

    logger.info(f"Step 3/4: Initializing ACDMMKTTrainer on {device}...")
    trainer = ACDMMKTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        d_ctt=train_loader.dataset.ctt_difficulty,
        ctt_valid_mask=train_loader.dataset.ctt_valid_mask,
        lr=float(args.lr if args.lr is not None else train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        patience=int(train_cfg.get("patience", 10)),
        device=device,
        precision=precision,
        grad_accum_steps=grad_accum_steps,
        save_dir=save_dir,
        model_cfg=model_cfg,
        warmup_epochs=int(train_cfg.get("warmup_epochs", 2)),
        eta_min=float(train_cfg.get("eta_min", 1e-5)),
        beta_distill=float(model_cfg.get("beta_distill", 0.10)),
        beta_ctt=float(model_cfg.get("beta_ctt", 0.20)),
        beta_disen=float(model_cfg.get("beta_disen", 0.05)),
        lambda_res=float(model_cfg.get("lambda_res", 1e-3)),
        tau_temp=float(model_cfg.get("tau_temp", 0.10)),
        logger=logger,
    )

    num_epochs = args.num_epochs if args.num_epochs is not None else int(train_cfg.get("num_epochs", 50))
    stage1_epochs = int(train_cfg.get("stage1_epochs", 15))
    stage1_batch_size = int(train_cfg.get("stage1_batch_size", 256))
    stage1_lr = float(train_cfg.get("stage1_lr", 1e-3))

    logger.info(f"Step 4/4: Launching Two-Stage Curriculum Training for Fold {fold} ({num_epochs} Stage-2 epochs)...")

    results = trainer.fit(
        num_epochs=num_epochs,
        run_stage1=(not args.skip_stage1),
        stage1_epochs=stage1_epochs,
        stage1_batch_size=stage1_batch_size,
        stage1_lr=stage1_lr,
    )

    out_record = {
        "fold": fold,
        "best_epoch": results["best_epoch"],
        "val_auc": results["best_val_auc"],
        "val_acc": results["history"][results["best_epoch"] - 1]["val_acc"] if results["best_epoch"] > 0 else 0.0,
        "test_auc": results["test_metrics"]["auc"] if results["test_metrics"] else 0.0,
        "test_acc": results["test_metrics"]["acc"] if results["test_metrics"] else 0.0,
        "checkpoint": results["best_checkpoint"],
    }

    # Clean up fold instances and release CUDA memory cache between folds
    del model, trainer, train_loader, val_loader, test_loader
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out_record


def evaluate_existing_fold(
    fold: int,
    ckpt_path: str,
    data_dir: str,
    model_cfg: Dict[str, Any],
    device: torch.device,
    logger: Any,
) -> Dict[str, Any]:
    """
    Evaluate an existing fold checkpoint.
    """
    logger.info(f"Evaluating existing checkpoint for Fold {fold}: {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_cfg" in ckpt and ckpt["model_cfg"] is not None:
        for k, v in ckpt["model_cfg"].items():
            model_cfg[k] = v

    _, val_loader, test_loader, item_bank = get_xes3g5m_dataloaders(
        data_dir=data_dir,
        val_fold=fold,
        model_type="acd_mmkt",
        batch_size=128,
        test_batch_size=128,
        p_drop=0.0,
        test_stride=100,
        num_workers=0,
    )

    model = ACDMMKT(
        num_questions=model_cfg.get("num_questions", 7652),
        num_kcs=model_cfg.get("num_kcs", 1175),
        d_raw_txt=model_cfg.get("d_raw_txt", 768),
        d_raw_img=model_cfg.get("d_raw_img", 768),
        d_ctx=model_cfg.get("d_ctx", 21),
        d_model=model_cfg.get("d_model", 128),
        num_heads=model_cfg.get("num_heads", 4),
        max_seq_len=model_cfg.get("max_seq_len", 200),
        dropout=model_cfg.get("dropout", 0.20),
        hgnn_layers=model_cfg.get("hgnn_layers", 2),
        tau_temp=model_cfg.get("tau_temp", 0.10),
        beta_distill=model_cfg.get("beta_distill", 0.10),
        beta_ctt=model_cfg.get("beta_ctt", 0.20),
        beta_disen=model_cfg.get("beta_disen", 0.05),
        lambda_res=model_cfg.get("lambda_res", 1e-3),
        irt_scale=model_cfg.get("irt_scale", 4.0),
        use_solution=model_cfg.get("use_solution", True),
        use_visual=model_cfg.get("use_visual", True),
        use_distillation=model_cfg.get("use_distillation", True),
        use_fine_grained=model_cfg.get("use_fine_grained", True),
        use_hypergraph_rasch=model_cfg.get("use_hypergraph_rasch", True),
        use_deep_irt=model_cfg.get("use_deep_irt", True),
        static_item_bank=item_bank,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    def _eval(loader):
        preds, targets = [], []
        with torch.no_grad():
            for batch in loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                out = model(batch)
                active_p = out["preds"][batch["eval_mask"]].cpu().numpy().tolist()
                active_y = batch["r_seq"][batch["eval_mask"]].cpu().numpy().tolist()
                preds.extend(active_p)
                targets.extend(active_y)
        return calculate_metrics(preds, targets)

    val_res = _eval(val_loader)
    test_res = _eval(test_loader)

    return {
        "fold": fold,
        "best_epoch": ckpt.get("epoch", -1),
        "val_auc": val_res["auc"],
        "val_acc": val_res["acc"],
        "test_auc": test_res["auc"],
        "test_acc": test_res["acc"],
        "checkpoint": ckpt_path,
    }


def main():
    args = parse_args()
    logger = get_logger("run_5fold_acd")
    logger.info("=== 5-Fold Cross-Validation Runner for ACD-MMKT ===")

    model_cfg = load_yaml_config(args.config_model)
    train_cfg = load_yaml_config(args.config_train)
    dataset_cfg = load_yaml_config(args.config_dataset)
    data_dir = dataset_cfg.get("data_dir", "data/processed/XES3G5M")

    # Apply hardware profile if selected
    if args.profile and "profiles" in train_cfg and args.profile in train_cfg["profiles"]:
        prof = train_cfg["profiles"][args.profile]
        logger.info(f"Applying hardware profile '{args.profile}': {prof}")
        for k, v in prof.items():
            train_cfg[k] = v

    if args.num_workers is not None:
        train_cfg["num_workers"] = args.num_workers

    # Apply ablation overrides
    if args.ablation == 2:
        model_cfg["use_distillation"] = False
    elif args.ablation == 3:
        model_cfg["use_fine_grained"] = False
    elif args.ablation == 4:
        model_cfg["use_hypergraph_rasch"] = False
    elif args.ablation == 5:
        model_cfg["use_deep_irt"] = False

    fold_results = []

    if args.eval_only:
        # Evaluate existing checkpoints
        for f in args.folds:
            ckpt_path = os.path.join(args.checkpoint_dir, f"acd_mmkt_fold{f}", "best_model.pt")
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(args.checkpoint_dir, f"fold_{f}", "best_model.pt")
            if not os.path.exists(ckpt_path):
                logger.warning(f"Checkpoint for fold {f} not found at {ckpt_path}, skipping.")
                continue

            device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
            res = evaluate_existing_fold(f, ckpt_path, data_dir, model_cfg, device, logger)
            fold_results.append(res)
    elif args.dual_gpu and torch.cuda.device_count() >= 2:
        # Asynchronous Dual-GPU Scheduling
        batch_size_gpu = args.batch_size if args.batch_size is not None else int(train_cfg.get("batch_size", 64))
        grad_accum_gpu = args.grad_accum_steps if args.grad_accum_steps is not None else int(train_cfg.get("grad_accum_steps", 2))
        precision_gpu = args.precision if args.precision is not None else str(train_cfg.get("precision", "amp_fp16"))

        logger.info("Initializing Asynchronous Dual-GPU 5-Fold Execution:")
        logger.info(f"- GPU 0 (cuda:0): Folds [0, 1, 2] | batch_size={batch_size_gpu}, grad_accum={grad_accum_gpu}, precision={precision_gpu}")
        logger.info(f"- GPU 1 (cuda:1): Folds [3, 4] | batch_size={batch_size_gpu}, grad_accum={grad_accum_gpu}, precision={precision_gpu}")

        # Launch GPU 1 process in background
        cmd_gpu1 = [
            sys.executable,
            __file__,
            "--device", "cuda:1",
            "--folds", "3", "4",
            "--batch-size", str(batch_size_gpu),
            "--grad-accum-steps", str(grad_accum_gpu),
            "--precision", precision_gpu,
            "--checkpoint-dir", str(args.checkpoint_dir),
            "--output-report-dir", str(args.output_report_dir),
            "--config-model", str(args.config_model),
            "--config-train", str(args.config_train),
            "--config-dataset", str(args.config_dataset),
            "--num-epochs", str(args.num_epochs or train_cfg.get("num_epochs", 50)),
            "--worker-only",
        ]
        if args.lr is not None:
            cmd_gpu1.extend(["--lr", str(args.lr)])
        if args.num_workers is not None:
            cmd_gpu1.extend(["--num-workers", str(args.num_workers)])
        if args.profile is not None:
            cmd_gpu1.extend(["--profile", str(args.profile)])
        if args.skip_stage1:
            cmd_gpu1.append("--skip-stage1")
        if args.ablation is not None:
            cmd_gpu1.extend(["--ablation", str(args.ablation)])

        logger.info(f"Spawning background worker on cuda:1: {' '.join(cmd_gpu1)}")
        proc_gpu1 = subprocess.Popen(cmd_gpu1)

        # Run GPU 0 folds in current process
        for f in [0, 1, 2]:
            save_dir = os.path.join(args.checkpoint_dir, f"acd_mmkt_fold{f}")
            res = run_single_fold_training(
                fold=f,
                args=args,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                data_dir=data_dir,
                device=torch.device("cuda:0"),
                save_dir=save_dir,
                batch_size=batch_size_gpu,
                grad_accum_steps=grad_accum_gpu,
                precision=precision_gpu,
                logger=logger,
            )
            fold_results.append(res)

        # Wait for GPU 1 to complete
        logger.info("Waiting for cuda:1 worker (Folds 3, 4) to finish...")
        proc_gpu1.wait()

        # Load results of Folds 3 and 4
        for f in [3, 4]:
            ckpt_path = os.path.join(args.checkpoint_dir, f"acd_mmkt_fold{f}", "best_model.pt")
            if os.path.exists(ckpt_path):
                res = evaluate_existing_fold(f, ckpt_path, data_dir, model_cfg, torch.device("cuda:0"), logger)
                fold_results.append(res)
    else:
        # Single-device sequential execution
        if args.device == "auto":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        else:
            device = torch.device(args.device)

        batch_size = args.batch_size if args.batch_size is not None else int(train_cfg.get("batch_size", 64))
        grad_accum = args.grad_accum_steps if args.grad_accum_steps is not None else int(train_cfg.get("grad_accum_steps", 1))
        precision = args.precision if args.precision is not None else train_cfg.get("precision", "amp_fp16" if "cuda" in str(device) else "fp32")

        for f in args.folds:
            save_dir = os.path.join(args.checkpoint_dir, f"acd_mmkt_fold{f}")
            res = run_single_fold_training(
                fold=f,
                args=args,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                data_dir=data_dir,
                device=device,
                save_dir=save_dir,
                batch_size=batch_size,
                grad_accum_steps=grad_accum,
                precision=precision,
                logger=logger,
            )
            fold_results.append(res)

    # If worker only, do not write summary report
    if args.worker_only:
        logger.info(f"Worker process completed assigned folds: {args.folds}. Exiting without generating global summary report.")
        return

    # 4. Synthesize Summary Report
    if len(fold_results) == 0:
        logger.error("No fold results available to summarize!")
        return

    val_aucs = [r["val_auc"] for r in fold_results]
    val_accs = [r["val_acc"] for r in fold_results]
    test_aucs = [r["test_auc"] for r in fold_results]
    test_accs = [r["test_acc"] for r in fold_results]

    summary_table = f"""# Báo Cáo Tổng Hợp Đánh Giá Chéo 5-Fold Cross-Validation (ACD-MMKT)

| Fold | Epoch Tốt Nhất | Validation AUC | Validation ACC | Test Micro AUC | Test ACC |
| :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in sorted(fold_results, key=lambda x: x["fold"]):
        summary_table += f"| Fold {r['fold']} | {r['best_epoch']} | {r['val_auc']:.4f} | {r['val_acc']:.4f} | {r['test_auc']:.4f} | {r['test_acc']:.4f} |\n"

    summary_table += f"""| **Mean ± Std** | - | **{np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}** | **{np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}** | **{np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}** | **{np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}** |
"""
    logger.info("\n" + summary_table)

    os.makedirs(args.output_report_dir, exist_ok=True)
    report_json_path = os.path.join(args.output_report_dir, "5fold_acd_mmkt_summary.json")
    report_md_path = os.path.join(args.output_report_dir, "5fold_acd_mmkt_summary.md")

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": "ACD-MMKT",
                "ablation": args.ablation,
                "fold_results": fold_results,
                "summary": {
                    "val_auc_mean": float(np.mean(val_aucs)),
                    "val_auc_std": float(np.std(val_aucs)),
                    "val_acc_mean": float(np.mean(val_accs)),
                    "val_acc_std": float(np.std(val_accs)),
                    "test_auc_mean": float(np.mean(test_aucs)),
                    "test_auc_std": float(np.std(test_aucs)),
                    "test_acc_mean": float(np.mean(test_accs)),
                    "test_acc_std": float(np.std(test_accs)),
                },
            },
            f,
            indent=2,
        )

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(summary_table)

    logger.info(f"Summary reports saved to {report_json_path} and {report_md_path}")


if __name__ == "__main__":
    main()
