"""
End-to-End Master Runner for pyKT Baselines (DKT, DKVMN, SAKT) on XES3G5M and Eedi.

Executes:
1. 5-Fold Cross-Validation training for the selected baseline models.
2. Full question-level unrolled anti-leakage evaluation on test interactions
   (1,099,197 on XES3G5M; 124,960 / 74,276 / 50,684 on Eedi unified/public/private).
3. 5-Fold soft probability ensemble.
4. Multimodal, cold-start, remedial, and Eedi pedagogical stratification.
5. Paired statistical significance tests against MMKT models.
6. Dataset-isolated JSON & Markdown report generation.

Usage:
    # XES3G5M (default)
    conda run -n kt-research-env python scripts/run_5fold_pykt_baselines.py --model all

    # Eedi
    conda run -n kt-research-env python scripts/run_5fold_pykt_baselines.py --dataset eedi --model all --all_eedi_splits
"""

import argparse
import os
import subprocess
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset_router import resolve_dataset_key, resolve_dataset_paths
from src.utils.config import load_yaml_config
from src.utils.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Master runner for pyKT baselines.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["xes3g5m", "eedi"],
        help="Target dataset shortcut ('xes3g5m' or 'eedi').",
    )
    parser.add_argument(
        "--test_split",
        "--test-split",
        dest="test_split",
        type=str,
        default="unified",
        choices=["unified", "public", "private"],
        help="Eedi test split ('unified', 'public', or 'private').",
    )
    parser.add_argument(
        "--all_eedi_splits",
        action="store_true",
        help="When running on Eedi, evaluate all 3 splits (unified, public, private).",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Path to pyKT dataset YAML config.",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default=None,
        help="Path to pyKT train YAML config.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["dkt", "dkvmn", "sakt", "all"],
        help="Model family to run ('dkt', 'dkvmn', 'sakt', or 'all').",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Maximum epochs per fold (default: None -> follows config YAML).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience (default: None -> follows config YAML).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Training batch size (default: None -> follows config YAML).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: None -> follows config YAML).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: None -> follows config YAML).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Specific cross-validation folds to execute (default: 0 1 2 3 4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution device ('auto', 'cuda', 'mps', 'cpu', or explicit 'cuda:0', 'cuda:1').",
    )
    parser.add_argument(
        "--dual-gpu",
        action="store_true",
        help="Enable Dual-GPU scheduling (Worker 0: cuda:0 on folds [0,1,2] + Worker 1: cuda:1 on folds [3,4]).",
    )
    parser.add_argument(
        "--worker-only",
        action="store_true",
        help="Internal flag: worker only executes assigned folds without running evaluation.",
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip training and directly run evaluation on existing checkpoints.",
    )
    parser.add_argument(
        "--reuse_preds",
        action="store_true",
        help="Reuse existing test_preds*.npy in checkpoint dirs if available during evaluation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_config_path = args.dataset_config or (
        "configs/dataset/eedi_pykt.yaml" if args.dataset == "eedi" else "configs/dataset/xes3g5m_pykt.yaml"
    )
    dataset_cfg = load_yaml_config(dataset_config_path) if os.path.exists(dataset_config_path) else {}
    dataset_key = resolve_dataset_key(dataset_cfg=dataset_cfg, dataset_arg=args.dataset)

    train_config_path = args.train_config or (
        "configs/train_pykt_eedi.yaml"
        if (dataset_key == "eedi" and os.path.exists("configs/train_pykt_eedi.yaml"))
        else "configs/train_pykt.yaml"
    )
    train_cfg = load_yaml_config(train_config_path) if os.path.exists(train_config_path) else {}
    paths = resolve_dataset_paths(dataset_key=dataset_key, dataset_cfg=dataset_cfg, train_cfg=train_cfg)

    log_dir = paths["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger("Run5FoldPyKT", log_file=os.path.join(log_dir, "run_5fold_pykt.log"))

    models_to_run = ["dkt", "dkvmn", "sakt"] if args.model == "all" else [args.model]
    logger.info(
        f"Dataset: {paths['display_name']} (key={dataset_key}) | "
        f"Target pyKT baseline models: {models_to_run} | Folds: {args.folds}"
    )

    python_bin = sys.executable

    shared_cfg_flags = [
        "--dataset", dataset_key,
        "--dataset_config", dataset_config_path,
        "--train_config", train_config_path,
    ]

    # Phase 1: Training
    if not args.skip_train:
        if args.dual_gpu and not args.worker_only:
            logger.info("⚡ [DUAL-GPU MODE] Dispatching cross-validation folds across cuda:0 and cuda:1")
            half = (len(args.folds) + 1) // 2
            folds_w0 = args.folds[:half]
            folds_w1 = args.folds[half:]

            for m in models_to_run:
                logger.info(f"################### DUAL-GPU TRAINING: {m.upper()} ###################")
                logger.info(f"   Worker 0 (cuda:0) assigned folds: {folds_w0}")
                logger.info(f"   Worker 1 (cuda:1) assigned folds: {folds_w1}")

                cmd0 = (
                    [
                        python_bin,
                        "scripts/run_5fold_pykt_baselines.py",
                        "--model", m,
                        "--device", "cuda:0",
                    ]
                    + shared_cfg_flags
                    + ["--folds"]
                    + [str(f) for f in folds_w0]
                    + ["--worker-only"]
                )

                cmd1 = (
                    [
                        python_bin,
                        "scripts/run_5fold_pykt_baselines.py",
                        "--model", m,
                        "--device", "cuda:1",
                    ]
                    + shared_cfg_flags
                    + ["--folds"]
                    + [str(f) for f in folds_w1]
                    + ["--worker-only"]
                )

                for c_arr in [cmd0, cmd1]:
                    if args.epochs is not None:
                        c_arr.extend(["--epochs", str(args.epochs)])
                    if args.patience is not None:
                        c_arr.extend(["--patience", str(args.patience)])
                    if args.batch_size is not None:
                        c_arr.extend(["--batch_size", str(args.batch_size)])
                    if args.lr is not None:
                        c_arr.extend(["--lr", str(args.lr)])
                    if args.seed is not None:
                        c_arr.extend(["--seed", str(args.seed)])

                p0 = subprocess.Popen(cmd0)
                p1 = subprocess.Popen(cmd1) if len(folds_w1) > 0 else None

                ret0 = p0.wait()
                ret1 = p1.wait() if p1 else 0

                if ret0 != 0 or ret1 != 0:
                    logger.error(f"Dual-GPU worker training failed for model {m} (Worker 0: {ret0}, Worker 1: {ret1})")
                    sys.exit(ret0 or ret1)
                logger.info(f"Dual-GPU training completed for {m.upper()} across all assigned folds.")
        else:
            for m in models_to_run:
                logger.info(f"################### STARTING TRAINING: {m.upper()} ###################")
                for f in args.folds:
                    cmd = (
                        [
                            python_bin,
                            "scripts/train_pykt_baselines.py",
                            "--model", m,
                            "--fold", str(f),
                            "--device", args.device,
                        ]
                        + shared_cfg_flags
                    )
                    if args.epochs is not None:
                        cmd.extend(["--epochs", str(args.epochs)])
                    if args.patience is not None:
                        cmd.extend(["--patience", str(args.patience)])
                    if args.batch_size is not None:
                        cmd.extend(["--batch_size", str(args.batch_size)])
                    if args.lr is not None:
                        cmd.extend(["--lr", str(args.lr)])
                    if args.seed is not None:
                        cmd.extend(["--seed", str(args.seed)])

                    logger.info(f"Running command: {' '.join(cmd)}")
                    res = subprocess.run(cmd)
                    if res.returncode != 0:
                        logger.error(f"Training failed for model {m} fold {f} with exit code {res.returncode}")
                        sys.exit(res.returncode)

    if args.worker_only:
        logger.info("Worker-only execution finished.")
        return

    # Phase 2: Evaluation
    logger.info("################### STARTING COMPREHENSIVE EVALUATION ###################")
    eval_device = "cuda:0" if args.dual_gpu else args.device
    splits_to_eval = (
        ["unified", "public", "private"]
        if (dataset_key == "eedi" and args.all_eedi_splits)
        else [args.test_split]
    )

    for sp in splits_to_eval:
        eval_cmd = (
            [
                python_bin,
                "scripts/evaluate_pykt_baselines.py",
                "--model", args.model,
                "--device", eval_device,
                "--test_split", sp,
            ]
            + shared_cfg_flags
        )
        if args.reuse_preds:
            eval_cmd.append("--reuse_preds")
        logger.info(f"Running evaluation command: {' '.join(eval_cmd)}")
        res = subprocess.run(eval_cmd)
        if res.returncode != 0:
            logger.error(f"Evaluation failed on split '{sp}' with exit code {res.returncode}")
            sys.exit(res.returncode)

    logger.info("################### 5-FOLD PYKT BASELINES PIPELINE COMPLETE ###################")


if __name__ == "__main__":
    main()
