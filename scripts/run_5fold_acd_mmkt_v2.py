"""
CLI Entry Point for 5-Fold Cross-Validation Protocol on ACD-MMKTv2 (pyKT Compliant).
Asymmetric Causal-Disentangled Multimodal Knowledge Tracing v2.

Supports:
1. Single-Device Sequential Execution (Apple Silicon MPS / Single CUDA GPU / CPU).
2. Heterogeneous Dual-GPU Asynchronous Parallel Scheduling:
   - Worker 0 on cuda:0 (e.g. RTX 5060 Ti): Folds [0, 1, 2] with batch_size=128, grad_accum=1.
   - Worker 1 on cuda:1 (e.g. RTX 3060): Folds [3, 4] with batch_size=64, grad_accum=2.
   - Automatically eliminates DDP straggler wait time.
3. 5-Fold Soft Probability Ensemble on independent test set (1,099,197 interactions).
4. Computes Mean +/- Std across 5 folds and formats comprehensive Markdown and JSON summary tables.
5. Eval-only mode to evaluate and summarize existing fold checkpoints.
6. Ablation study configurations (Ablation 1..6).

Usage:
    # Run full 5-fold CV sequentially
    conda run -n kt-research-env python scripts/run_5fold_acd_mmkt_v2.py --num-epochs 50

    # Run dual-GPU parallel execution
    conda run -n kt-research-env python scripts/run_5fold_acd_mmkt_v2.py --dual-gpu --num-epochs 50

    # Evaluate existing checkpoints and build ensemble
    conda run -n kt-research-env python scripts/run_5fold_acd_mmkt_v2.py --eval-only --checkpoint-dir outputs/checkpoints
"""

import argparse
import os
import sys
import json
import subprocess
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.utils.config import load_yaml_config
from src.data.xes3g5m_dataset import get_xes3g5m_dataloaders
from src.models.mmkt.acd_mmkt_v2 import ACDMMKTv2
from src.training.acd_v2_trainer import ACDMMKTv2Trainer
from src.evaluation.metrics import calculate_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="5-Fold Cross-Validation runner for ACD-MMKTv2.")
    parser.add_argument("--config-model", type=str, default="configs/model/acd_mmkt_v2.yaml", help="Path to model config")
    parser.add_argument("--config-train", type=str, default="configs/train_acd_mmkt_v2.yaml", help="Path to training config")
    parser.add_argument("--config-dataset", type=str, default="configs/dataset/xes3g5m.yaml", help="Path to dataset config")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Folds to execute (default: 0 1 2 3 4)")
    parser.add_argument("--num-epochs", type=int, default=None, help="Number of Stage 2 epochs per fold")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'xla', 'mps', 'cpu', or 'auto')")
    parser.add_argument("--dual-gpu", action="store_true", help="Enable Dual-GPU scheduling (cuda:0 folds 0,1,2 + cuda:1 folds 3,4)")
    parser.add_argument("--multi-tpu", action="store_true", help="Enable Multi-TPU core parallel execution across folds (TPU v5e-8)")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints", help="Base directory for checkpoints")
    parser.add_argument("--output-report-dir", type=str, default="outputs/reports", help="Directory for summary reports")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing fold checkpoints without retraining")
    parser.add_argument("--skip-stage1", action="store_true", help="Skip Stage 1 pre-training")
    parser.add_argument("--profile", type=str, choices=["m1_pro", "dual_gpu_5060ti", "dual_gpu_3060", "kaggle_t4", "kaggle_tpu_v5e", "colab_t4", "colab_l4", "colab_a100", "colab_tpu_v5e", "colab_tpu_v6e"], default=None, help="Hardware profile")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of DataLoader workers")
    parser.add_argument("--grad-accum-steps", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--precision", type=str, choices=["fp32", "amp_fp16", "amp_bf16"], default=None, help="Compute precision")
    parser.add_argument("--worker-only", action="store_true", help="Internal flag: worker only executes assigned folds")

    # Ablation presets
    parser.add_argument(
        "--ablation",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=None,
        help="Ablation preset: 1=Full, 2=w/o Solution, 3=w/o Fine-Grained, 4=w/o HGNN/Rasch, 5=w/o DeepIRT, 6=w/o Stage1",
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
    """Train and evaluate a single cross-validation fold."""
    logger.info(f"--- Launching Fold {fold} on {device} (BatchSize={batch_size}, GradAccum={grad_accum_steps}, Precision={precision}) ---")
    set_seed(train_cfg.get("seed", 42) + fold)

    train_loader, val_loader, test_loader, item_bank = get_xes3g5m_dataloaders(
        data_dir=data_dir,
        val_fold=fold,
        model_type="acd_mmkt_v2",
        batch_size=batch_size,
        test_batch_size=train_cfg.get("test_batch_size", 128),
        p_drop=float(train_cfg.get("p_drop", 0.15)),
        test_stride=int(train_cfg.get("test_stride", 100)),
        num_workers=args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 0),
    )

    model = ACDMMKTv2(
        static_item_bank=item_bank,
        d_raw_txt=int(model_cfg.get("d_raw_txt", 768)),
        d_raw_img=int(model_cfg.get("d_raw_img", 768)),
        d_model=int(model_cfg.get("d_model", 128)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        max_seq_len=int(model_cfg.get("max_seq_len", 200)),
        dropout=float(model_cfg.get("dropout", 0.15)),
        p_drop=float(model_cfg.get("p_visual_drop", train_cfg.get("p_drop", 0.15))),
        hgnn_layers=int(model_cfg.get("hgnn_layers", 2)),
        use_solution=bool(model_cfg.get("use_solution", True)),
        use_fine_grained=bool(model_cfg.get("use_fine_grained", True)),
        use_hypergraph=bool(model_cfg.get("use_hypergraph", True)),
        use_gram_schmidt=bool(model_cfg.get("use_gram_schmidt", True)),
        use_deep_irt=bool(model_cfg.get("use_deep_irt", True)),
    )

    trainer = ACDMMKTv2Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr=args.lr if args.lr is not None else float(train_cfg.get("lr", 1e-3)),
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
        stage1_epochs=int(train_cfg.get("stage1_epochs", 15)),
        stage1_batch_size=int(train_cfg.get("stage1_batch_size", 256)),
        stage1_lr=float(train_cfg.get("stage1_lr", 1e-3)),
        beta_d=float(model_cfg.get("beta_difficulty", 0.20)),
        kappa=float(model_cfg.get("kappa_smoothing", 10.0)),
        use_stage1_pretrain=bool(model_cfg.get("use_stage1_pretrain", True)) and not args.skip_stage1,
        logger=logger,
    )

    num_epochs = args.num_epochs if args.num_epochs is not None else train_cfg.get("num_epochs", 50)
    fit_res = trainer.fit(num_epochs=num_epochs)
    return fit_res


@torch.no_grad()
def evaluate_checkpoint_predictions(
    checkpoint_path: str,
    test_loader: Any,
    device: torch.device,
    item_bank: Dict[str, torch.Tensor],
    logger: Any,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Load a trained checkpoint and return predictions, targets, and metrics."""
    logger.info(f"Loading checkpoint {checkpoint_path} on {device}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_cfg = ckpt.get("model_cfg", {})

    model = ACDMMKTv2(
        static_item_bank=item_bank,
        d_raw_txt=int(model_cfg.get("d_raw_txt", 768)),
        d_raw_img=int(model_cfg.get("d_raw_img", 768)),
        d_model=int(model_cfg.get("d_model", 128)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        max_seq_len=int(model_cfg.get("max_seq_len", 200)),
        dropout=float(model_cfg.get("dropout", 0.15)),
        p_drop=float(model_cfg.get("p_visual_drop", 0.0)),
        hgnn_layers=int(model_cfg.get("hgnn_layers", 2)),
        use_solution=bool(model_cfg.get("use_solution", True)),
        use_fine_grained=bool(model_cfg.get("use_fine_grained", True)),
        use_hypergraph=bool(model_cfg.get("use_hypergraph", True)),
        use_gram_schmidt=bool(model_cfg.get("use_gram_schmidt", True)),
        use_deep_irt=bool(model_cfg.get("use_deep_irt", True)),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.precompute_item_cache()

    preds = []
    targets = []

    for batch in test_loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        eval_mask = batch["eval_mask"].bool()
        out = model(batch, use_cache=True)
        probs = out["probs"]

        if eval_mask.any():
            preds.extend(probs[eval_mask].cpu().numpy())
            targets.extend(batch["r_seq"][eval_mask].cpu().numpy())

    model.clear_item_cache()
    preds_arr = np.array(preds, dtype=np.float64)
    targets_arr = np.array(targets, dtype=np.int64)
    metrics = calculate_metrics(preds_arr, targets_arr)

    return preds_arr, targets_arr, metrics


def main():
    args = parse_args()
    logger = get_logger("run_5fold_acd_mmkt_v2")
    logger.info("=== pyKT 5-Fold Cross-Validation Protocol: ACD-MMKTv2 ===")

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

    # Apply ablations
    ablation_suffix = ""
    if args.ablation == 2:
        model_cfg["use_solution"] = False
        ablation_suffix = "_wo_solution"
    elif args.ablation == 3:
        model_cfg["use_fine_grained"] = False
        ablation_suffix = "_wo_fine_grained"
    elif args.ablation == 4:
        model_cfg["use_hypergraph"] = False
        model_cfg["use_gram_schmidt"] = False
        ablation_suffix = "_wo_hypergraph_rasch"
    elif args.ablation == 5:
        model_cfg["use_deep_irt"] = False
        ablation_suffix = "_wo_deep_irt"
    elif args.ablation == 6:
        model_cfg["use_stage1_pretrain"] = False
        ablation_suffix = "_wo_stage1"

    # Multi-TPU Core Dispatch (Kaggle TPU v5e-8 has 8 cores: run folds 0,1,2,3,4 in parallel)
    if args.multi_tpu and not args.worker_only:
        logger.info("Initializing Asynchronous Multi-TPU Core Execution across Folds 0..4...")
        cmd_base = [sys.executable, sys.argv[0], "--worker-only"]
        if args.config_model:
            cmd_base.extend(["--config-model", args.config_model])
        if args.config_train:
            cmd_base.extend(["--config-train", args.config_train])
        if args.config_dataset:
            cmd_base.extend(["--config-dataset", args.config_dataset])
        if args.checkpoint_dir:
            cmd_base.extend(["--checkpoint-dir", args.checkpoint_dir])
        if args.output_report_dir:
            cmd_base.extend(["--output-report-dir", args.output_report_dir])
        if args.num_epochs is not None:
            cmd_base.extend(["--num-epochs", str(args.num_epochs)])
        if args.batch_size is not None:
            cmd_base.extend(["--batch-size", str(args.batch_size)])
        if args.lr is not None:
            cmd_base.extend(["--lr", str(args.lr)])
        if args.grad_accum_steps is not None:
            cmd_base.extend(["--grad-accum-steps", str(args.grad_accum_steps)])
        if args.precision is not None:
            cmd_base.extend(["--precision", args.precision])
        if args.num_workers is not None:
            cmd_base.extend(["--num-workers", str(args.num_workers)])
        if args.skip_stage1:
            cmd_base.append("--skip-stage1")
        if args.ablation is not None:
            cmd_base.extend(["--ablation", str(args.ablation)])
        if args.eval_only:
            cmd_base.append("--eval-only")

        prof = args.profile if args.profile else "kaggle_tpu_v5e"
        workers = []
        for f in args.folds:
            env = os.environ.copy()
            env["PJRT_DEVICE"] = "TPU"
            env["TPU_VISIBLE_DEVICES"] = str(f)
            logger.info(f"Spawning TPU worker for Fold {f} on TPU Core {f} (TPU_VISIBLE_DEVICES={f})...")
            p = subprocess.Popen(cmd_base + [
                "--folds", str(f),
                "--profile", prof,
                "--device", "xla",
            ], env=env)
            workers.append(p)

        for p in workers:
            p.wait()
        logger.info("All Multi-TPU training workers finished.")

    # Dual-GPU Dispatch
    elif args.dual_gpu and not args.worker_only and torch.cuda.device_count() >= 2:
        logger.info("Initializing Asynchronous Dual-GPU Execution (cuda:0 -> Folds [0,1,2] | cuda:1 -> Folds [3,4])...")
        cmd_base = [sys.executable, sys.argv[0], "--worker-only"]
        if args.config_model:
            cmd_base.extend(["--config-model", args.config_model])
        if args.config_train:
            cmd_base.extend(["--config-train", args.config_train])
        if args.config_dataset:
            cmd_base.extend(["--config-dataset", args.config_dataset])
        if args.checkpoint_dir:
            cmd_base.extend(["--checkpoint-dir", args.checkpoint_dir])
        if args.output_report_dir:
            cmd_base.extend(["--output-report-dir", args.output_report_dir])
        if args.num_epochs is not None:
            cmd_base.extend(["--num-epochs", str(args.num_epochs)])
        if args.batch_size is not None:
            cmd_base.extend(["--batch-size", str(args.batch_size)])
        if args.lr is not None:
            cmd_base.extend(["--lr", str(args.lr)])
        if args.grad_accum_steps is not None:
            cmd_base.extend(["--grad-accum-steps", str(args.grad_accum_steps)])
        if args.precision is not None:
            cmd_base.extend(["--precision", args.precision])
        if args.num_workers is not None:
            cmd_base.extend(["--num-workers", str(args.num_workers)])
        if args.skip_stage1:
            cmd_base.append("--skip-stage1")
        if args.ablation is not None:
            cmd_base.extend(["--ablation", str(args.ablation)])
        if args.eval_only:
            cmd_base.append("--eval-only")

        # Worker 0 (cuda:0)
        prof_0 = args.profile if args.profile else "dual_gpu_5060ti"
        prof_1 = args.profile if args.profile else "dual_gpu_3060"
        p0 = subprocess.Popen(cmd_base + [
            "--folds", "0", "1", "2",
            "--profile", prof_0,
            "--device", "cuda:0",
        ])
        # Worker 1 (cuda:1)
        p1 = subprocess.Popen(cmd_base + [
            "--folds", "3", "4",
            "--profile", prof_1,
            "--device", "cuda:1",
        ])

        p0.wait()
        p1.wait()
        logger.info("Dual-GPU training workers finished.")

    elif not args.eval_only:
        # Sequential Single-Device Execution
        device_str = args.device if args.device != "auto" else (
            "xla" if "PJRT_DEVICE" in os.environ and os.environ.get("PJRT_DEVICE") == "TPU" or "TPU_NAME" in os.environ else
            "cuda:0" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        )
        if "xla" in device_str:
            try:
                import torch_xla.core.xla_model as xm
                device = xm.xla_device()
            except ImportError:
                device = torch.device(device_str)
        else:
            device = torch.device(device_str)
        batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 64)
        grad_accum = args.grad_accum_steps if args.grad_accum_steps is not None else train_cfg.get("grad_accum_steps", 1)
        precision = args.precision if args.precision is not None else train_cfg.get("precision", "fp32")

        for f in args.folds:
            save_dir = os.path.join(args.checkpoint_dir, f"acd_mmkt_v2{ablation_suffix}_fold{f}")
            run_single_fold_training(
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

    if args.worker_only:
        return

    # Synthesize 5-Fold Results & Soft Ensemble
    logger.info("=== Synthesizing 5-Fold Cross-Validation & Ensemble Results ===")
    eval_device_str = (
        "xla" if "PJRT_DEVICE" in os.environ and os.environ.get("PJRT_DEVICE") == "TPU" or "TPU_NAME" in os.environ else
        "cuda:0" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    )
    if "xla" in eval_device_str:
        try:
            import torch_xla.core.xla_model as xm
            eval_device = xm.xla_device()
        except ImportError:
            eval_device = torch.device(eval_device_str)
    else:
        eval_device = torch.device(eval_device_str)

    # Load test dataloader
    _, _, test_loader, item_bank = get_xes3g5m_dataloaders(
        data_dir=data_dir,
        val_fold=0,
        model_type="acd_mmkt_v2",
        batch_size=128,
        test_batch_size=128,
        p_drop=0.0,
        test_stride=100,
        num_workers=0,
    )

    fold_metrics = []
    all_fold_preds = []
    ground_truth_targets = None

    for f in range(5):
        ckpt_path = os.path.join(args.checkpoint_dir, f"acd_mmkt_v2{ablation_suffix}_fold{f}", "best_model.pt")
        if not os.path.exists(ckpt_path):
            logger.warning(f"Checkpoint for Fold {f} not found at {ckpt_path}. Skipping.")
            continue

        preds_f, targets_f, m_f = evaluate_checkpoint_predictions(
            checkpoint_path=ckpt_path,
            test_loader=test_loader,
            device=eval_device,
            item_bank=item_bank,
            logger=logger,
        )
        fold_metrics.append((f, m_f))
        all_fold_preds.append(preds_f)
        ground_truth_targets = targets_f
        logger.info(f"Fold {f} Test -> Micro ROC-AUC: {m_f['auc']:.5f} | Accuracy: {m_f['acc']:.5f}")

    if len(fold_metrics) == 0:
        logger.error("No valid checkpoints found for synthesis.")
        return

    # Compute Mean +/- Std across evaluated folds
    aucs = [m["auc"] for _, m in fold_metrics]
    accs = [m["acc"] for _, m in fold_metrics]
    mean_auc, std_auc = float(np.mean(aucs)), float(np.std(aucs))
    mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))

    # 5-Fold Soft Ensemble (if all 5 folds present)
    ensemble_metrics = {}
    if len(all_fold_preds) == 5 and ground_truth_targets is not None:
        ensemble_preds = np.mean(all_fold_preds, axis=0)
        ensemble_metrics = calculate_metrics(ensemble_preds, ground_truth_targets)
        logger.info(
            f"=== 5-Fold Soft Ensemble -> Micro ROC-AUC: {ensemble_metrics['auc']:.5f} | "
            f"Accuracy: {ensemble_metrics['acc']:.5f} ==="
        )

    logger.info(
        f"=== 5-Fold Individual Average -> "
        f"Micro ROC-AUC: {mean_auc:.5f} +/- {std_auc:.5f} | "
        f"Accuracy: {mean_acc:.5f} +/- {std_acc:.5f} ==="
    )

    # Save summary report
    summary = {
        "model": f"ACD-MMKTv2{ablation_suffix}",
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "individual_folds": [{"fold": f, "metrics": m} for f, m in fold_metrics],
        "ensemble": ensemble_metrics,
    }

    os.makedirs(args.output_report_dir, exist_ok=True)
    report_json_path = os.path.join(args.output_report_dir, f"acd_mmkt_v2{ablation_suffix}_5fold_summary.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved 5-fold cross-validation report to: {report_json_path}")


if __name__ == "__main__":
    main()

