"""
CLI Entry Point for Training MG-MMKT on XES3G5M with pyKT 5-Fold Cross-Validation.

Usage:
    conda run -n kt-research-env python scripts/train_mg_mmkt.py --val-fold 0 --num-epochs 30
    conda run -n kt-research-env python scripts/train_mg_mmkt.py --wo-solution  # Ablation 2
    conda run -n kt-research-env python scripts/train_mg_mmkt.py --wo-visual    # Ablation 3
"""

import argparse
import os
import sys
import json
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.utils.config import load_yaml_config, merge_configs
from src.data.xes3g5m_dataset import get_xes3g5m_dataloaders
from src.models.mmkt.mg_mmkt import MGMMKT
from src.training.trainer import MGMMKTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train MG-MMKT on XES3G5M dataset.")
    parser.add_argument("--config-model", type=str, default="configs/model/mg_mmkt.yaml", help="Path to model config")
    parser.add_argument("--config-train", type=str, default="configs/train.yaml", help="Path to training config")
    parser.add_argument("--config-dataset", type=str, default="configs/dataset/xes3g5m.yaml", help="Path to dataset config")
    parser.add_argument("--val-fold", type=int, default=None, help="Designated validation fold (0..4)")
    parser.add_argument("--num-epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'mps', 'cpu', or 'auto')")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of DataLoader worker subprocesses (default: from config or 0)")
    parser.add_argument("--save-dir", type=str, default=None, help="Checkpoint save directory")

    # Training hyperparameter overrides
    parser.add_argument("--warmup-epochs", type=int, default=None, help="Number of linear warmup epochs")
    parser.add_argument("--eta-min", type=float, default=None, help="Minimum learning rate for cosine annealing")

    # Ablation presets and flags
    parser.add_argument(
        "--ablation",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Ablation study configuration: 1=Full, 2=w/o Solution, 3=w/o Visual, 4=Naive Concat (Zero Imputation + Linear Concat)",
    )
    parser.add_argument("--wo-solution", action="store_true", help="Ablation: Disable solution representation in history (Config 2)")
    parser.add_argument("--wo-visual", action="store_true", help="Ablation: Disable question and solution visual modalities (Config 3)")
    parser.add_argument("--naive-concat", action="store_true", help="Ablation: Naive linear projection of concatenated modalities with zero-imputation (Config 4)")
    parser.add_argument("--zero-imputation", action="store_true", help="Ablation: Use zero-padding instead of learnable e_null tokens")
    parser.add_argument("--wo-gated-fusion", action="store_true", help="Ablation: Use equal weighting instead of masked softmax gate")

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load Configurations
    model_cfg = load_yaml_config(args.config_model)
    train_cfg = load_yaml_config(args.config_train)
    dataset_cfg = load_yaml_config(args.config_dataset)

    # Apply overrides
    val_fold = args.val_fold if args.val_fold is not None else train_cfg.get("val_fold", 0)
    num_epochs = args.num_epochs if args.num_epochs is not None else train_cfg.get("num_epochs", 30)
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 64)
    lr = args.lr if args.lr is not None else float(train_cfg.get("lr", 1e-3))
    seed = args.seed if args.seed is not None else train_cfg.get("seed", 42)
    save_dir = args.save_dir if args.save_dir is not None else train_cfg.get("save_dir", f"outputs/checkpoints/mg_mmkt_fold{val_fold}")
    num_workers = args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 0)
    warmup_epochs = args.warmup_epochs if args.warmup_epochs is not None else train_cfg.get("warmup_epochs", 2)
    eta_min = args.eta_min if args.eta_min is not None else float(train_cfg.get("eta_min", 1e-5))

    # Apply ablation presets or explicit flags
    if args.ablation == 1:
        # Full MG-MMKT
        model_cfg["use_solution"] = True
        model_cfg["use_visual"] = True
        model_cfg["use_learnable_imputation"] = True
        model_cfg["use_gated_fusion"] = True
        model_cfg["fusion_mode"] = "gated"
    elif args.ablation == 2 or args.wo_solution:
        # Configuration 2: w/o Solution Modalities
        model_cfg["use_solution"] = False
    elif args.ablation == 3 or args.wo_visual:
        # Configuration 3: w/o Visual Modalities
        model_cfg["use_visual"] = False
    elif args.ablation == 4 or args.naive_concat:
        # Configuration 4: Zero-imputation + Naive Concat (Issue H2)
        model_cfg["use_learnable_imputation"] = False
        model_cfg["use_gated_fusion"] = False
        model_cfg["fusion_mode"] = "naive_concat"

    # Additional individual flags
    if args.zero_imputation:
        model_cfg["use_learnable_imputation"] = False
    if args.wo_gated_fusion:
        model_cfg["use_gated_fusion"] = False
        if model_cfg.get("fusion_mode") != "naive_concat":
            model_cfg["fusion_mode"] = "equal_weight"

    # 2. Set Random Seed
    set_seed(seed)

    # 3. Setup Logger
    log_dir = train_cfg.get("log_dir", "outputs/logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_fold{val_fold}.log")
    logger = get_logger("train_mg_mmkt", log_file=log_file)
    logger.info("=== Initializing MG-MMKT Training Pipeline ===")
    logger.info(f"Model Config: {model_cfg}")
    logger.info(f"Val Fold: {val_fold} | Epochs: {num_epochs} | Batch Size: {batch_size} | LR: {lr:.2e} | Seed: {seed} | Workers: {num_workers}")

    # 4. Device Selection
    device_str = args.device if args.device != "auto" else train_cfg.get("device", "auto")
    if device_str == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = torch.device(device_str)
    logger.info(f"Compute Device: {device}")

    # 5. Build DataLoaders
    data_dir = dataset_cfg.get("data_dir", "data/processed/XES3G5M")
    p_drop = dataset_cfg.get("p_drop", 0.20)
    test_stride = dataset_cfg.get("test_stride", 100)

    logger.info(f"Loading data from {data_dir} (p_drop={p_drop}, test_stride={test_stride}, num_workers={num_workers})...")
    train_loader, val_loader, test_loader, item_bank = get_xes3g5m_dataloaders(
        data_dir=data_dir,
        val_fold=val_fold,
        batch_size=batch_size,
        test_batch_size=train_cfg.get("test_batch_size", 128),
        p_drop=p_drop,
        test_stride=test_stride,
        num_workers=num_workers,
    )
    logger.info(f"Loaded: Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    # 6. Instantiate Model
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
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"MGMMKT Model Instantiated. Trainable Parameters: {num_params:,}")

    # 7. Trainer Setup
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
        save_dir=save_dir,
        model_cfg=model_cfg,
        warmup_epochs=warmup_epochs,
        eta_min=eta_min,
        logger=logger,
    )

    # 8. Run Training
    summary = trainer.fit(num_epochs=num_epochs)

    # Save summary metrics to json
    report_path = os.path.join(save_dir, "training_summary.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=== MG-MMKT Training Finished Successfully ===")
    logger.info(f"Best Val AUC: {summary['best_val_auc']:.4f} at epoch {summary['best_epoch']}")
    if summary["test_metrics"] is not None:
        logger.info(f"Final Test AUC: {summary['test_metrics']['auc']:.4f} | ACC: {summary['test_metrics']['acc']:.4f}")
    logger.info(f"Full summary saved to {report_path}")


if __name__ == "__main__":
    main()
