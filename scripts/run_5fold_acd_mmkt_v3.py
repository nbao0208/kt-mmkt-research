"""
CLI Entry Point for 5-Fold Cross-Validation Protocol on ACD-MMKTv3 (pyKT Compliant).
Asymmetric Causal-Disentangled Multimodal Knowledge Tracing v3.

Supports:
1. Single-Device Sequential Execution (Apple Silicon MPS / Single CUDA GPU / CPU).
2. Heterogeneous Dual-GPU Asynchronous Scheduling:
   - Worker 0 on cuda:0: Folds [0, 1, 2]
   - Worker 1 on cuda:1: Folds [3, 4]
3. 5-Fold Soft Probability Ensemble on independent test set (1,099,197 interactions).
4. Statistical significance testing (Paired t-test and Wilcoxon signed-rank test) against baselines.
5. Computes Mean +/- Std across 5 folds and formats comprehensive Markdown and JSON summary tables.
6. Eval-only mode to evaluate and summarize existing fold checkpoints.
7. Ablation study configurations (Ablation 1..6).

Usage:
    # Run full 5-fold CV sequentially
    conda run -n kt-research-env python scripts/run_5fold_acd_mmkt_v3.py --num-epochs 50

    # Evaluate existing checkpoints and build ensemble
    conda run -n kt-research-env python scripts/run_5fold_acd_mmkt_v3.py --eval-only --checkpoint-dir outputs/checkpoints
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as stats
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.xes3g5m_dataset import get_xes3g5m_dataloaders
from src.evaluation.metrics import calculate_metrics
from src.models.mmkt.acd_mmkt_v3 import ACDMMKTv3
from src.training.acd_v3_trainer import ACDMMKTv3Trainer
from src.utils.config import load_yaml_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed


ABLATION_SUFFIX_MAP = {
    1: "",
    2: "_ablation2_wo_solution",
    3: "_ablation3_wo_visual",
    4: "_ablation4_unanchored",
    5: "_ablation5_polluted_exp",
    6: "_ablation6_wo_2pl_irt",
}


def parse_args():
    parser = argparse.ArgumentParser(description="5-Fold Cross-Validation runner for ACD-MMKTv3.")
    parser.add_argument("--config-model", type=str, default="configs/model/acd_mmkt_v3.yaml", help="Path to model config")
    parser.add_argument("--config-train", type=str, default="configs/train_acd_mmkt_v3.yaml", help="Path to training config")
    parser.add_argument("--config-dataset", type=str, default="configs/dataset/xes3g5m.yaml", help="Path to dataset config")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Folds to execute (default: 0 1 2 3 4)")
    parser.add_argument("--num-epochs", type=int, default=None, help="Number of epochs per fold")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'mps', 'cpu', or 'auto')")
    parser.add_argument("--dual-gpu", action="store_true", help="Enable Dual-GPU scheduling")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints", help="Base directory for checkpoints")
    parser.add_argument("--output-report-dir", type=str, default="outputs/reports", help="Directory for summary reports")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing fold checkpoints without retraining")
    parser.add_argument(
        "--profile",
        type=str,
        choices=[
            "m1_pro",
            "dual_gpu_5060ti",
            "dual_gpu_3060",
            "kaggle_t4",
            "kaggle_tpu_v5e",
            "colab_t4",
            "colab_l4",
            "colab_a100",
            "colab_tpu_v5e",
        ],
        default=None,
        help="Hardware profile",
    )
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
        help="Ablation preset: 1=Full, 2=w/o Solution, 3=w/o Visual, 4=Unanchored Concept, 5=Polluted Experience, 6=w/o 2PL-IRT",
    )
    return parser.parse_args()


def load_real_benchmarks_from_reports(base_reports_dir: str = "outputs/reports") -> Dict[str, Dict[str, Any]]:
    """
    Load empirical 5-fold evaluation AUCs from real benchmark reports on disk.
    Strictly avoids hardcoded mockup/dummy values. Any baseline without existing
    completed 5-fold evaluation is marked as 'pending'.

    Returns:
        Dictionary mapping model names to their benchmark metadata:
        {
            "MODEL_NAME": {
                "fold_aucs": List[float] or None,
                "status": "available" or "pending",
                "source": str or None
            }
        }
    """
    benchmarks: Dict[str, Dict[str, Any]] = {}

    # 1. pyKT baselines (DKT, DKVMN, SAKT)
    pykt_path = os.path.join(base_reports_dir, "pykt_baselines_comprehensive_evaluation.json")
    if os.path.exists(pykt_path):
        try:
            with open(pykt_path, "r", encoding="utf-8") as f:
                pykt_data = json.load(f)
            for m in ["dkt", "dkvmn", "sakt"]:
                if m in pykt_data and "fold_aucs" in pykt_data[m] and len(pykt_data[m]["fold_aucs"]) == 5:
                    benchmarks[m.upper()] = {
                        "fold_aucs": [float(x) for x in pykt_data[m]["fold_aucs"]],
                        "status": "available",
                        "source": pykt_path,
                    }
        except Exception as e:
            print(f"[Warning] Failed to parse pykt benchmark report {pykt_path}: {e}")

    # 2. MG-MMKT (v1)
    mg_v1_path = os.path.join(base_reports_dir, "full_model", "5fold_summary.json")
    if not os.path.exists(mg_v1_path):
        mg_v1_path = os.path.join(base_reports_dir, "comprehensive_test_evaluation.json")
    if os.path.exists(mg_v1_path):
        try:
            with open(mg_v1_path, "r", encoding="utf-8") as f:
                v1_data = json.load(f)
            if "folds" in v1_data and len(v1_data["folds"]) == 5:
                benchmarks["MG-MMKT"] = {
                    "fold_aucs": [float(fold["test_auc"]) for fold in v1_data["folds"]],
                    "status": "available",
                    "source": mg_v1_path,
                }
            elif "full_model_5fold_results" in v1_data and len(v1_data["full_model_5fold_results"]) == 5:
                aucs = []
                for fold in v1_data["full_model_5fold_results"]:
                    if "stratified_metrics" in fold and "overall" in fold["stratified_metrics"]:
                        aucs.append(float(fold["stratified_metrics"]["overall"]["auc"]))
                    elif "test_auc" in fold:
                        aucs.append(float(fold["test_auc"]))
                if len(aucs) == 5:
                    benchmarks["MG-MMKT"] = {
                        "fold_aucs": aucs,
                        "status": "available",
                        "source": mg_v1_path,
                    }
            elif "summaries_5fold" in v1_data and "full_model" in v1_data["summaries_5fold"]:
                folds = v1_data["summaries_5fold"]["full_model"].get("folds", [])
                if len(folds) == 5:
                    benchmarks["MG-MMKT"] = {
                        "fold_aucs": [float(fold["test_auc"]) for fold in folds],
                        "status": "available",
                        "source": mg_v1_path,
                    }
        except Exception as e:
            print(f"[Warning] Failed to parse MG-MMKT v1 report {mg_v1_path}: {e}")

    # 3. MG-MMKTv2
    mg_v2_path = os.path.join(base_reports_dir, "v2_full_5fold", "mg_mmkt_v2_5fold_summary.json")
    if os.path.exists(mg_v2_path):
        try:
            with open(mg_v2_path, "r", encoding="utf-8") as f:
                v2_data = json.load(f)
            if "single_fold_results" in v2_data and len(v2_data["single_fold_results"]) == 5:
                benchmarks["MG-MMKTv2"] = {
                    "fold_aucs": [float(fold["test_auc"]) for fold in v2_data["single_fold_results"]],
                    "status": "available",
                    "source": mg_v2_path,
                }
        except Exception as e:
            print(f"[Warning] Failed to parse MG-MMKT v2 report {mg_v2_path}: {e}")

    # 4. Other standard pyKT baselines: Mark as pending if no 5-fold evaluation is on disk
    expected_baselines = ["DKT", "DKVMN", "SAKT", "MG-MMKT", "MG-MMKTv2", "SAINT", "GKT", "SparseKT"]
    for expected in expected_baselines:
        if expected not in benchmarks:
            benchmarks[expected] = {
                "fold_aucs": None,
                "status": "pending",
                "source": None,
            }

    return benchmarks


def train_and_eval_single_fold(
    val_fold: int,
    args: argparse.Namespace,
    model_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    device_str: str,
    logger: Any,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Train (if needed) and evaluate a single fold on the test set.

    Returns:
        Tuple of (metrics_dict, test_predictions_array, test_targets_array).
    """
    logger.info("=" * 60)
    logger.info(f"=== PROCESSING FOLD {val_fold} ON DEVICE {device_str} ===")

    data_dir = dataset_cfg.get("data_dir", "data/processed/XES3G5M")
    suffix = ABLATION_SUFFIX_MAP.get(args.ablation, "")
    fold_save_dir = os.path.join(args.checkpoint_dir, f"acd_mmkt_v3_fold{val_fold}{suffix}")
    best_model_path = os.path.join(fold_save_dir, "best_model.pt")
    pred_save_path = os.path.join(fold_save_dir, "test_predictions.npy")
    target_save_path = os.path.join(fold_save_dir, "test_targets.npy")

    os.makedirs(fold_save_dir, exist_ok=True)

    # Fast cache return if in eval-only mode and predictions & targets already exist on disk
    if args.eval_only and os.path.exists(pred_save_path) and os.path.exists(target_save_path):
        logger.info(f"Loading existing fold {val_fold} cached test predictions from {pred_save_path}...")
        preds_arr = np.load(pred_save_path)
        targets_arr = np.load(target_save_path)
        metrics = calculate_metrics(torch.from_numpy(preds_arr), torch.from_numpy(targets_arr))
        logger.info(
            f"Fold {val_fold} Cached Test Results: AUC = {metrics['auc']:.4f}, ACC = {metrics['acc']:.4f} (N={len(preds_arr):,})"
        )
        return metrics, preds_arr, targets_arr

    batch_size = args.batch_size or train_cfg.get("batch_size", 64)
    test_batch_size = train_cfg.get("test_batch_size", 128)
    num_workers = args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 0)
    p_drop = train_cfg.get("p_drop", model_cfg.get("p_visual_drop", 0.15))

    # 1. Load Data
    train_loader, val_loader, test_loader, item_bank = get_xes3g5m_dataloaders(
        data_dir=data_dir,
        val_fold=val_fold,
        model_type="acd_mmkt_v3",
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        p_drop=p_drop,
        test_stride=train_cfg.get("test_stride", 100),
        num_workers=num_workers,
    )

    aux_path = os.path.join(data_dir, "acd_v3_aux_metadata.pt")
    if os.path.exists(aux_path):
        aux_data = torch.load(aux_path, map_location="cpu")
        mu_concept_prior = aux_data["mu_concept_folds"][val_fold]
        n_q = aux_data["N_q_folds"][val_fold]
        d_q = aux_data["D_q_folds"][val_fold]
        omega_q = aux_data["omega_q_folds"][val_fold]
    else:
        mu_concept_prior = torch.full((7652,), -1.37)
        n_q = torch.zeros(7652, dtype=torch.int64)
        d_q = torch.full((7652,), 0.20)
        omega_q = torch.zeros(7652)

    device = torch.device(device_str)

    # 2. Instantiate Model
    model = ACDMMKTv3(
        static_item_bank=item_bank,
        d_raw_txt=model_cfg.get("d_raw_txt", 768),
        d_raw_img=model_cfg.get("d_raw_img", 768),
        d_model=model_cfg.get("d_model", 128),
        num_heads=model_cfg.get("num_heads", 4),
        max_seq_len=model_cfg.get("max_seq_len", 200),
        dropout=model_cfg.get("dropout", 0.15),
        p_drop=p_drop,
        hgnn_layers=model_cfg.get("hgnn_layers", 2),
        gamma_vis=model_cfg.get("gamma_vis", 1.5),
        delta_a=model_cfg.get("delta_a", 0.1),
        use_solution=model_cfg.get("use_solution", True),
        use_visual=model_cfg.get("use_visual", True),
        unanchored_concept=model_cfg.get("unanchored_concept", False),
        polluted_experience_attn=model_cfg.get("polluted_experience_attn", False),
        use_deep_irt=model_cfg.get("use_deep_irt", True),
        use_hypergraph=model_cfg.get("use_hypergraph", True),
        use_gram_schmidt=model_cfg.get("use_gram_schmidt", True),
        post_diag_mode=model_cfg.get("post_diag_mode", "none"),
    ).to(device)

    # Free raw static item bank immediately after model weights initialization
    del item_bank
    gc.collect()

    # 3. Training Loop (if not in eval-only mode)
    if not args.eval_only:
        trainer = ACDMMKTv3Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=None,
            n_q=n_q,
            d_q=d_q,
            omega_q=omega_q,
            mu_concept_prior=mu_concept_prior,
            lr=args.lr or train_cfg.get("lr", 1e-3),
            weight_decay=train_cfg.get("weight_decay", 1e-4),
            max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
            patience=train_cfg.get("patience", 10),
            device=device,
            precision=args.precision or train_cfg.get("precision", "fp32"),
            grad_accum_steps=args.grad_accum_steps or train_cfg.get("grad_accum_steps", 1),
            save_dir=fold_save_dir,
            val_fold=val_fold,
            model_cfg=model_cfg,
            warmup_epochs=train_cfg.get("warmup_epochs", 2),
            eta_min=train_cfg.get("eta_min", 1e-5),
            beta_d=train_cfg.get("beta_d", 0.20),
            lambda_sparse=train_cfg.get("lambda_sparse", 0.01),
            kappa=train_cfg.get("kappa", 10.0),
            logger=logger,
        )
        trainer.train(num_epochs=args.num_epochs or train_cfg.get("num_epochs", 50))

    # 4. Load Best Checkpoint and Run Evaluation on Independent Test Set
    logger.info(f"Loading best checkpoint from {best_model_path} for Test evaluation...")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Checkpoint not found: {best_model_path}")

    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Precompute cache for fast evaluation
    model.precompute_item_bank_cache(mu_concept_prior.to(device))

    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    for batch in test_loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        eval_mask = batch["eval_mask"].bool()
        targets = batch["r_seq"]

        with torch.no_grad():
            outputs = model(batch, mu_concept_prior=mu_concept_prior.to(device), use_cache=True)

        probs = outputs["probs"]
        mask_np = eval_mask.cpu().numpy()
        probs_np = probs.cpu().numpy()
        targets_np = targets.cpu().numpy()

        all_preds.append(probs_np[mask_np])
        all_targets.append(targets_np[mask_np])

    model.clear_item_bank_cache()

    preds_arr = np.concatenate(all_preds, axis=0).astype(np.float32)
    targets_arr = np.concatenate(all_targets, axis=0).astype(np.int64)

    # 2 arguments passed into calculate_metrics
    metrics = calculate_metrics(
        torch.from_numpy(preds_arr),
        torch.from_numpy(targets_arr),
    )
    logger.info(f"Fold {val_fold} Test Results: AUC = {metrics['auc']:.4f}, ACC = {metrics['acc']:.4f} (N={len(preds_arr):,})")

    # Save fold predictions and targets arrays for reproducible soft ensembling
    np.save(pred_save_path, preds_arr)
    np.save(target_save_path, targets_arr)

    # Explicit memory cleanup
    del model, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics, preds_arr, targets_arr


def run_dual_gpu_parallel(args, model_cfg, train_cfg, dataset_cfg, logger):
    """
    Launch asynchronous dual-GPU execution:
    Worker 0 (cuda:0) handles Folds [0, 1, 2].
    Worker 1 (cuda:1) handles Folds [3, 4].
    Forwards hardware profiles, hyperparameters, and config files completely.
    """
    logger.info("=== LAUNCHING DUAL-GPU PARALLEL 5-FOLD CV ===")
    common_args = [
        sys.executable,
        __file__,
        "--worker-only",
        "--config-model",
        args.config_model,
        "--config-train",
        args.config_train,
        "--config-dataset",
        args.config_dataset,
        "--checkpoint-dir",
        args.checkpoint_dir,
        "--output-report-dir",
        args.output_report_dir,
    ]
    if args.ablation:
        common_args.extend(["--ablation", str(args.ablation)])
    if args.num_epochs:
        common_args.extend(["--num-epochs", str(args.num_epochs)])
    if args.batch_size:
        common_args.extend(["--batch-size", str(args.batch_size)])
    if args.lr:
        common_args.extend(["--lr", str(args.lr)])
    if args.num_workers is not None:
        common_args.extend(["--num-workers", str(args.num_workers)])
    if args.grad_accum_steps is not None:
        common_args.extend(["--grad-accum-steps", str(args.grad_accum_steps)])
    if args.precision:
        common_args.extend(["--precision", str(args.precision)])
    if args.eval_only:
        common_args.append("--eval-only")

    # Asymmetric hardware profile allocation for Dual-GPU scheduling
    worker0_profile = ["--profile", args.profile or "dual_gpu_5060ti"]
    worker1_profile = ["--profile", args.profile or "dual_gpu_3060"]

    worker0_cmd = common_args + worker0_profile + ["--device", "cuda:0", "--folds", "0", "1", "2"]
    worker1_cmd = common_args + worker1_profile + ["--device", "cuda:1", "--folds", "3", "4"]

    logger.info(f"Worker 0 Command: {' '.join(worker0_cmd)}")
    logger.info(f"Worker 1 Command: {' '.join(worker1_cmd)}")

    p0 = subprocess.Popen(worker0_cmd)
    p1 = subprocess.Popen(worker1_cmd)
    p0.wait()
    p1.wait()


def generate_markdown_report(
    report_data: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Format and write a clean, comprehensive GitHub-flavored Markdown report.
    """
    fold_metrics = report_data["single_fold_metrics"]
    mean_m = report_data["mean_metrics"]
    ens_m = report_data["ensemble_5fold_metrics"]
    stat_tests = report_data["statistical_tests"]
    ablation = report_data["ablation_preset"]

    lines = [
        f"# ACD-MMKTv3 5-Fold Cross-Validation Evaluation Report (pyKT Protocol)",
        f"",
        f"- **Model**: ACD-MMKTv3 (Asymmetric Causal-Disentangled Multimodal Knowledge Tracing v3)",
        f"- **Ablation Preset**: `{ablation}`",
        f"- **Generated At**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Total Runtime**: {report_data.get('total_time_mins', 0.0):.2f} minutes",
        f"",
        f"## 1. Single-Fold Test Results (XES3G5M Independent Test Set)",
        f"",
        f"| Fold | Test AUC | Test ACC |",
        f"|:---:|:---:|:---:|",
    ]

    for fold_k, m in sorted(fold_metrics.items()):
        lines.append(f"| {fold_k.upper()} | {m['auc']:.4f} | {m['acc']:.4f} |")

    lines.extend([
        f"| **Mean ± Std** | **{mean_m['auc']:.4f} ± {mean_m['auc_std']:.4f}** | **{mean_m['acc']:.4f} ± {mean_m['acc_std']:.4f}** |",
        f"",
        f"## 2. 5-Fold Soft Probability Ensemble",
        f"",
        f"- **Ensemble Test AUC**: `{ens_m['auc']:.4f}`",
        f"- **Ensemble Test ACC**: `{ens_m['acc']:.4f}`",
        f"",
        f"## 3. Empirical Benchmark Comparison & Statistical Significance",
        f"",
        f"| Baseline Model | Status | Mean AUC | Delta AUC | Paired t-test p-value | Wilcoxon p-value | Significance | Data Source |",
        f"|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|",
    ])

    for model_name, res in stat_tests.items():
        if res["status"] == "available":
            p_t = res["paired_t_pval"]
            p_w = res["wilcoxon_pval"]
            sig_str = "p < 0.01 ***" if p_t < 0.01 else "p < 0.05 *" if p_t < 0.05 else "n.s."
            src = os.path.basename(res.get("source", "disk"))
            lines.append(
                f"| {model_name} | {res['status']} | {res['base_mean_auc']:.4f} | {res['delta_auc']:+.4f} | "
                f"{p_t:.2e} | {p_w:.2e} | {sig_str} | `{src}` |"
            )
        else:
            lines.append(f"| {model_name} | pending | N/A | N/A | N/A | N/A | pending | N/A |")

    lines.extend([
        f"",
        f"> **Note**: All benchmark AUCs are dynamically loaded from verified cross-validation experiment reports on disk. "
        f"No mockup or placeholder data was used.",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()

    model_cfg = load_yaml_config(args.config_model)
    train_cfg = load_yaml_config(args.config_train)
    dataset_cfg = load_yaml_config(args.config_dataset)

    # Set seed for reproducible cross-validation
    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    # Configure file logging
    suffix = ABLATION_SUFFIX_MAP.get(args.ablation, "")
    log_file = os.path.join(
        "outputs",
        "logs",
        f"run_5fold_acd_mmkt_v3{suffix}.log",
    )
    logger = get_logger("run_5fold_acd_mmkt_v3", log_file=log_file)

    # Apply hardware profile if specified
    if args.profile and "profiles" in train_cfg and args.profile in train_cfg["profiles"]:
        prof = train_cfg["profiles"][args.profile]
        logger.info(f"Applying Hardware Profile '{args.profile}': {prof}")
        for k, v in prof.items():
            train_cfg[k] = v
        if args.device == "auto" and "device" in prof:
            args.device = prof["device"]

    # Command-line overrides take highest priority
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.precision is not None:
        train_cfg["precision"] = args.precision
    if args.grad_accum_steps is not None:
        train_cfg["grad_accum_steps"] = args.grad_accum_steps
    if args.num_workers is not None:
        train_cfg["num_workers"] = args.num_workers

    # Apply ablation overrides
    if args.ablation is not None:
        logger.info(f"Applying Ablation Preset {args.ablation}:")
        if args.ablation == 2:
            model_cfg["use_solution"] = False
        elif args.ablation == 3:
            model_cfg["use_visual"] = False
        elif args.ablation == 4:
            model_cfg["unanchored_concept"] = True
            train_cfg["lambda_sparse"] = 0.0
        elif args.ablation == 5:
            model_cfg["polluted_experience_attn"] = True
        elif args.ablation == 6:
            model_cfg["use_deep_irt"] = False

    # Check Dual-GPU request
    if args.dual_gpu and not args.worker_only:
        run_dual_gpu_parallel(args, model_cfg, train_cfg, dataset_cfg, logger)
        # Re-run in eval-only mode to assemble results from completed workers
        args.eval_only = True

    # Resolve compute device
    if args.device == "auto":
        device_str = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device_str = args.device

    fold_metrics = {}
    fold_preds = []
    targets_all = None

    t0 = time.time()

    for fold in args.folds:
        metrics, preds, targets = train_and_eval_single_fold(
            val_fold=fold,
            args=args,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            dataset_cfg=dataset_cfg,
            device_str=device_str,
            logger=logger,
        )
        fold_metrics[f"fold_{fold}"] = metrics
        fold_preds.append(preds)
        if targets_all is None:
            targets_all = targets

    # If this was a sub-worker in dual-GPU mode, terminate here
    if args.worker_only:
        logger.info(f"Worker on {device_str} completed assigned folds.")
        return

    # Compute 5-Fold Soft Probability Ensemble
    logger.info("=" * 60)
    logger.info("=== COMPUTING 5-FOLD SOFT PROBABILITY ENSEMBLE ===")
    fold_preds_mat = np.stack(fold_preds, axis=0)  # [5, N]
    ensemble_preds = np.mean(fold_preds_mat, axis=0)  # [N]

    ensemble_metrics = calculate_metrics(
        torch.from_numpy(ensemble_preds),
        torch.from_numpy(targets_all),
    )
    logger.info(f"5-Fold Soft Ensemble: AUC = {ensemble_metrics['auc']:.4f}, ACC = {ensemble_metrics['acc']:.4f}")

    # Compute Statistics across evaluated Folds
    aucs = [fold_metrics[f"fold_{f}"]["auc"] for f in args.folds]
    accs = [fold_metrics[f"fold_{f}"]["acc"] for f in args.folds]

    mean_auc, std_auc = float(np.mean(aucs)), float(np.std(aucs))
    mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))
    logger.info(f"5-Fold CV Summary: AUC = {mean_auc:.4f} +/- {std_auc:.4f}, ACC = {mean_acc:.4f} +/- {std_acc:.4f}")

    # Benchmark Comparison against Baselines loaded from reports
    logger.info("=" * 60)
    logger.info("=== BENCHMARK COMPARISON & PAIRED STATISTICAL TESTING ===")

    benchmarks = load_real_benchmarks_from_reports(base_reports_dir=args.output_report_dir)

    stat_tests = {}
    for base_name, base_info in benchmarks.items():
        base_aucs = base_info.get("fold_aucs")
        status = base_info.get("status", "pending")
        source = base_info.get("source")

        if status == "available" and base_aucs is not None and len(base_aucs) == len(aucs) and len(aucs) == 5:
            t_stat, t_pval = stats.ttest_rel(aucs, base_aucs)
            try:
                w_stat, w_pval = stats.wilcoxon(aucs, base_aucs)
            except Exception:
                w_stat, w_pval = 0.0, 1.0
            stat_tests[base_name] = {
                "status": "available",
                "source": source,
                "base_mean_auc": float(np.mean(base_aucs)),
                "delta_auc": float(mean_auc - np.mean(base_aucs)),
                "paired_t_stat": float(t_stat),
                "paired_t_pval": float(t_pval),
                "wilcoxon_stat": float(w_stat),
                "wilcoxon_pval": float(w_pval),
                "significant_p05": bool(t_pval < 0.05),
                "significant_p01": bool(t_pval < 0.01),
            }
            logger.info(
                f"  vs {base_name:12s}: Delta AUC = {mean_auc - np.mean(base_aucs):+.4f} | "
                f"t-test p = {t_pval:.2e} ({'p<0.01 ***' if t_pval < 0.01 else 'p<0.05 *' if t_pval < 0.05 else 'n.s.'}) | "
                f"Wilcoxon p = {w_pval:.2e}"
            )
        else:
            stat_tests[base_name] = {
                "status": "pending",
                "source": None,
                "base_mean_auc": None,
                "delta_auc": None,
                "paired_t_stat": None,
                "paired_t_pval": None,
                "wilcoxon_stat": None,
                "wilcoxon_pval": None,
                "significant_p05": None,
                "significant_p01": None,
            }
            logger.info(f"  vs {base_name:12s}: PENDING (no completed 5-fold evaluation report found)")

    # Save comprehensive report (JSON & Markdown)
    os.makedirs(args.output_report_dir, exist_ok=True)
    report_data = {
        "model_name": "ACD-MMKTv3",
        "ablation_preset": args.ablation or 1,
        "single_fold_metrics": fold_metrics,
        "mean_metrics": {
            "auc": mean_auc,
            "auc_std": std_auc,
            "acc": mean_acc,
            "acc_std": std_acc,
        },
        "ensemble_5fold_metrics": {
            "auc": float(ensemble_metrics["auc"]),
            "acc": float(ensemble_metrics["acc"]),
        },
        "statistical_tests": stat_tests,
        "total_time_mins": (time.time() - t0) / 60,
    }

    suffix = ABLATION_SUFFIX_MAP.get(args.ablation, "")
    report_path_json = os.path.join(args.output_report_dir, f"acd_mmkt_v3_5fold_report{suffix}.json")
    report_path_md = os.path.join(args.output_report_dir, f"acd_mmkt_v3_5fold_report{suffix}.md")

    with open(report_path_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"Saved comprehensive 5-fold evaluation JSON report to {report_path_json}")

    generate_markdown_report(report_data, report_path_md)
    logger.info(f"Saved comprehensive 5-fold evaluation Markdown report to {report_path_md}")


if __name__ == "__main__":
    main()
