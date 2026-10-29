"""
CLI Entry Point for Training MG-MMKT v2 on XES3G5M with pyKT 5-Fold Cross-Validation.

Usage:
    conda run -n kt-research-env python scripts/train_mg_mmkt_v2.py --val-fold 0 --num-epochs 30
    conda run -n kt-research-env python scripts/train_mg_mmkt_v2.py --ablation 2  # w/o LUPI
    conda run -n kt-research-env python scripts/train_mg_mmkt_v2.py --ablation 3  # w/o Dual-Pathway History
    conda run -n kt-research-env python scripts/train_mg_mmkt_v2.py --ablation 6  # w/o Structural Decoupling & KC-Imputation
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
from src.utils.config import load_yaml_config
from src.data.xes3g5m_dataset import get_xes3g5m_dataloaders
from src.models.mmkt.mg_mmkt_v2 import MGMMKTv2
from src.training.mg_v2_trainer import MGMMKTv2Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train MG-MMKT v2 on XES3G5M dataset.")
    parser.add_argument("--config-model", type=str, default="configs/model/mg_mmkt_v2.yaml", help="Path to model config")
    parser.add_argument("--config-train", type=str, default="configs/train_mg_mmkt_v2.yaml", help="Path to training config")
    parser.add_argument("--config-dataset", type=str, default="configs/dataset/xes3g5m.yaml", help="Path to dataset config")
    parser.add_argument("--val-fold", type=int, default=None, help="Designated validation fold (0..4)")
    parser.add_argument("--num-epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'mps', 'cpu', or 'auto')")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of DataLoader workers (default: 0)")
    parser.add_argument("--save-dir", type=str, default=None, help="Checkpoint save directory")

    # Multi-task loss weights
    parser.add_argument("--lambda-priv", type=float, default=None, help="Privileged distillation loss weight (default: 0.10)")
    parser.add_argument("--lambda-gate", type=float, default=None, help="Gate suppression regularization weight (default: 0.05)")

    # Ablation presets (Plan Section 8.1)
    parser.add_argument(
        "--ablation",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=None,
        help=(
            "Ablation presets: "
            "1=Full v2, 2=w/o LUPI, 3=w/o Dual-Pathway History, "
            "4=w/o GLU & Cross-Dim FFN, 5=w/o Disentangled QK & Relative Forgetting, "
            "6=w/o Structural Decoupling & KC Imputation"
        ),
    )
    parser.add_argument("--wo-lupi", action="store_true", help="Ablation: Disable privileged cognitive distillation (Config 2)")
    parser.add_argument("--wo-dual-pathway", action="store_true", help="Ablation: Disable dual-pathway history (Config 3)")
    parser.add_argument("--wo-glu-ffn", action="store_true", help="Ablation: Disable GLU and post-fusion residual FFN (Config 4)")
    parser.add_argument("--wo-disentangled-qk", action="store_true", help="Ablation: Disable disentangled QK (Config 5a)")
    parser.add_argument("--wo-relative-forgetting", action="store_true", help="Ablation: Disable monotonic forgetting decay (Config 5b)")
    parser.add_argument("--wo-structural-decoupling", action="store_true", help="Ablation: Disable frequency gate and ID dropout (Config 6a)")
    parser.add_argument("--wo-kc-imputation", action="store_true", help="Ablation: Disable KC-conditioned missing vector imputation (Config 6b)")

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load Configurations
    model_cfg = load_yaml_config(args.config_model)
    train_cfg = load_yaml_config(args.config_train)
    dataset_cfg = load_yaml_config(args.config_dataset)

    # Overrides
    val_fold = args.val_fold if args.val_fold is not None else train_cfg.get("val_fold", 0)
    num_epochs = args.num_epochs if args.num_epochs is not None else train_cfg.get("num_epochs", 30)
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 64)
    lr = args.lr if args.lr is not None else float(train_cfg.get("lr", 1e-3))
    seed = args.seed if args.seed is not None else train_cfg.get("seed", 42)
    save_dir = args.save_dir if args.save_dir is not None else train_cfg.get("save_dir", f"outputs/checkpoints/mg_mmkt_v2_fold{val_fold}")
    num_workers = args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 0)
    warmup_epochs = train_cfg.get("warmup_epochs", 2)
    eta_min = float(train_cfg.get("eta_min", 1e-5))
    lambda_priv = args.lambda_priv if args.lambda_priv is not None else float(model_cfg.get("lambda_priv", 0.10))
    lambda_gate = args.lambda_gate if args.lambda_gate is not None else float(model_cfg.get("lambda_gate", 0.05))

    # Apply ablation configurations
    if args.ablation == 1:
        # Full MG-MMKT v2
        pass
    elif args.ablation == 2 or args.wo_lupi:
        model_cfg["use_lupi"] = False
        lambda_priv = 0.0
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
        lambda_gate = 0.0

    if args.wo_disentangled_qk:
        model_cfg["use_disentangled_qk"] = False
    if args.wo_relative_forgetting:
        model_cfg["use_relative_forgetting"] = False
    if args.wo_structural_decoupling:
        model_cfg["use_structural_decoupling"] = False
    if args.wo_kc_imputation:
        model_cfg["use_kc_imputation"] = False

    # 2. Setup Logging and Reproducibility
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "train.log")
    logger = get_logger("train_mg_mmkt_v2", log_file=log_file)
    logger.info("=" * 60)
    logger.info(f"Starting MG-MMKT v2 Training (Fold {val_fold})")
    logger.info(f"Seed: {seed} | Epochs: {num_epochs} | Batch Size: {batch_size} | LR: {lr}")
    logger.info(f"Loss weights -> lambda_priv: {lambda_priv} | lambda_gate: {lambda_gate}")
    set_seed(seed)

    # 3. Device Selection
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)
    logger.info(f"Compute Device: {device}")

    # 4. DataLoaders
    data_dir = dataset_cfg.get("processed_data_dir", dataset_cfg.get("data_dir", train_cfg.get("data_dir", "data/processed/XES3G5M")))
    p_drop = float(train_cfg.get("p_drop", 0.20))
    test_stride = int(train_cfg.get("test_stride", 100))

    logger.info(f"Constructing pyKT-compliant 5-Fold DataLoaders from {data_dir}...")
    train_loader, val_loader, test_loader, item_bank = get_xes3g5m_dataloaders(
        data_dir=data_dir,
        val_fold=val_fold,
        batch_size=batch_size,
        test_batch_size=train_cfg.get("test_batch_size", 128),
        p_drop=p_drop,
        test_stride=test_stride,
        num_workers=num_workers,
    )
    logger.info(f"DataLoaders ready: Train ({len(train_loader.dataset):,} windows) | Val ({len(val_loader.dataset):,} windows) | Test ({len(test_loader.dataset):,} windows)")

    # 5. Model Initialization
    logger.info(f"Instantiating MG-MMKT v2 model (d={model_cfg.get('d_model', 128)}, heads={model_cfg.get('num_heads', 4)})...")
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

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total Trainable Parameters: {num_params:,}")

    # 6. Trainer Execution
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
        save_dir=save_dir,
        model_cfg=model.config,
        warmup_epochs=warmup_epochs,
        eta_min=eta_min,
        logger=logger,
    )

    results = trainer.fit(num_epochs=num_epochs)

    # 7. Save Final Report
    report_path = os.path.join(save_dir, "training_summary.json")
    summary = {
        "model_name": "mg_mmkt_v2",
        "val_fold": val_fold,
        "best_epoch": results["best_epoch"],
        "best_val_auc": results["best_val_auc"],
        "test_metrics": results["test_metrics"],
        "model_config": model.config,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Training successfully completed. Summary written to {report_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

