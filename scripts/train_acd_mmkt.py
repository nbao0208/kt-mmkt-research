"""
CLI Entry Point for Training ACD-MMKT on XES3G5M with pyKT 5-Fold Cross-Validation.

Usage:
    # Full Two-Stage Training on Fold 0
    conda run -n kt-research-env python scripts/train_acd_mmkt.py --val-fold 0 --num-epochs 30

    # With hardware profile for MacBook Pro M1 Pro
    conda run -n kt-research-env python scripts/train_acd_mmkt.py --val-fold 0 --profile m1_pro

    # Ablation Studies
    conda run -n kt-research-env python scripts/train_acd_mmkt.py --val-fold 0 --ablation 2  # w/o Distillation
    conda run -n kt-research-env python scripts/train_acd_mmkt.py --val-fold 0 --ablation 3  # w/o Fine-Grained
    conda run -n kt-research-env python scripts/train_acd_mmkt.py --val-fold 0 --ablation 4  # w/o HGNN/Rasch
    conda run -n kt-research-env python scripts/train_acd_mmkt.py --val-fold 0 --ablation 5  # w/o DeepIRT
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
from src.models.mmkt.acd_mmkt import ACDMMKT
from src.training.acd_trainer import ACDMMKTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train ACD-MMKT on XES3G5M dataset.")
    parser.add_argument("--config-model", type=str, default="configs/model/acd_mmkt.yaml", help="Path to model config")
    parser.add_argument("--config-train", type=str, default="configs/train_acd_mmkt.yaml", help="Path to training config")
    parser.add_argument("--config-dataset", type=str, default="configs/dataset/xes3g5m.yaml", help="Path to dataset config")
    parser.add_argument("--val-fold", type=int, default=None, help="Designated validation fold (0..4)")
    parser.add_argument("--num-epochs", type=int, default=None, help="Number of Stage 2 fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cuda', 'cuda:0', 'mps', 'cpu')")
    parser.add_argument("--profile", type=str, choices=["m1_pro", "dual_gpu_5060ti", "dual_gpu_3060"], default=None, help="Hardware execution profile")
    parser.add_argument("--precision", type=str, choices=["fp32", "amp_fp16", "amp_bf16"], default=None, help="Computation precision")
    parser.add_argument("--grad-accum-steps", type=int, default=None, help="Gradient accumulation steps K_acc")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of DataLoader worker subprocesses")
    parser.add_argument("--save-dir", type=str, default=None, help="Checkpoint save directory")

    # Stage 1 curriculum overrides
    parser.add_argument("--skip-stage1", action="store_true", help="Skip Stage 1 item-bank pre-training")
    parser.add_argument("--stage1-epochs", type=int, default=None, help="Number of Stage 1 pre-training epochs")
    parser.add_argument("--stage1-lr", type=float, default=None, help="Stage 1 learning rate")

    # Ablation presets and explicit flags
    parser.add_argument(
        "--ablation",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="Ablation preset: 1=Full, 2=w/o Distillation, 3=w/o Fine-Grained, 4=w/o HGNN/Rasch, 5=w/o DeepIRT",
    )
    parser.add_argument("--wo-distillation", action="store_true", help="Ablation 2: Disable student solution complexity distillation")
    parser.add_argument("--wo-fine-grained", action="store_true", help="Ablation 3: Disable token/patch cross-attention (use global pooling)")
    parser.add_argument("--wo-hypergraph-rasch", action="store_true", help="Ablation 4: Disable HGNN and Rasch decomposition")
    parser.add_argument("--wo-deep-irt", action="store_true", help="Ablation 5: Disable DeepIRT readout (use black-box MLP)")

    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger("train_acd_mmkt")

    # 1. Load Configurations
    model_cfg = load_yaml_config(args.config_model)
    train_cfg = load_yaml_config(args.config_train)
    dataset_cfg = load_yaml_config(args.config_dataset)

    # Apply hardware profile if selected
    if args.profile and "profiles" in train_cfg and args.profile in train_cfg["profiles"]:
        prof = train_cfg["profiles"][args.profile]
        logger.info(f"Applying hardware profile '{args.profile}': {prof}")
        for k, v in prof.items():
            train_cfg[k] = v

    # Resolve overrides
    val_fold = args.val_fold if args.val_fold is not None else train_cfg.get("val_fold", 0)
    num_epochs = args.num_epochs if args.num_epochs is not None else train_cfg.get("num_epochs", 50)
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 64)
    test_batch_size = train_cfg.get("test_batch_size", 128)
    lr = args.lr if args.lr is not None else float(train_cfg.get("lr", 1e-3))
    seed = args.seed if args.seed is not None else train_cfg.get("seed", 42)
    save_dir = args.save_dir if args.save_dir is not None else train_cfg.get("save_dir", f"outputs/checkpoints/acd_mmkt_fold{val_fold}")
    num_workers = args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 0)
    precision = args.precision if args.precision is not None else train_cfg.get("precision", "fp32")
    grad_accum_steps = args.grad_accum_steps if args.grad_accum_steps is not None else train_cfg.get("grad_accum_steps", 1)
    data_dir = dataset_cfg.get("data_dir", "data/processed/XES3G5M")

    # Device selection
    if args.device:
        device_str = args.device
    elif "device" in train_cfg and train_cfg["device"] != "auto":
        device_str = train_cfg["device"]
    else:
        device_str = "cuda:0" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_str)

    # Seed
    set_seed(seed)
    logger.info(f"Environment: Device={device} | Precision={precision} | Seed={seed} | Fold={val_fold}")

    # Apply ablation overrides
    if args.ablation == 1:
        model_cfg["use_distillation"] = True
        model_cfg["use_fine_grained"] = True
        model_cfg["use_hypergraph_rasch"] = True
        model_cfg["use_deep_irt"] = True
    elif args.ablation == 2 or args.wo_distillation:
        model_cfg["use_distillation"] = False
        logger.info("Ablation 2 active: Cross-modal solution distillation disabled.")
    elif args.ablation == 3 or args.wo_fine_grained:
        model_cfg["use_fine_grained"] = False
        logger.info("Ablation 3 active: Fine-grained token/patch cross-attention disabled (global pooling).")
    elif args.ablation == 4 or args.wo_hypergraph_rasch:
        model_cfg["use_hypergraph_rasch"] = False
        logger.info("Ablation 4 active: 2-layer HGNN and Rasch decomposition disabled.")
    elif args.ablation == 5 or args.wo_deep_irt:
        model_cfg["use_deep_irt"] = False
        logger.info("Ablation 5 active: DeepIRT readout disabled (black-box MLP readout).")

    # 2. DataLoaders (model_type='acd_mmkt' loads full static_item_bank.pt and fold-specific CTT labels)
    logger.info(f"Loading XES3G5M data for Fold {val_fold} (model_type='acd_mmkt')...")
    train_loader, val_loader, test_loader, static_item_bank = get_xes3g5m_dataloaders(
        data_dir=data_dir,
        val_fold=val_fold,
        model_type="acd_mmkt",
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        p_drop=float(train_cfg.get("p_drop", 0.20)),
        test_stride=int(train_cfg.get("test_stride", 100)),
        num_workers=num_workers,
    )

    # 3. Instantiate ACD-MMKT Model
    logger.info("Initializing ACD-MMKT model architecture...")
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
        static_item_bank=static_item_bank,
    )

    # 4. Instantiate Trainer
    trainer = ACDMMKTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        d_ctt=train_loader.dataset.ctt_difficulty,
        ctt_valid_mask=train_loader.dataset.ctt_valid_mask,
        lr=lr,
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

    # 5. Execute Two-Stage Curriculum Training
    stage1_epochs = args.stage1_epochs if args.stage1_epochs is not None else int(train_cfg.get("stage1_epochs", 15))
    stage1_lr = args.stage1_lr if args.stage1_lr is not None else float(train_cfg.get("stage1_lr", 1e-3))
    stage1_batch_size = int(train_cfg.get("stage1_batch_size", 256))

    results = trainer.fit(
        num_epochs=num_epochs,
        run_stage1=(not args.skip_stage1),
        stage1_epochs=stage1_epochs,
        stage1_batch_size=stage1_batch_size,
        stage1_lr=stage1_lr,
    )

    # Save summary report
    summary_path = os.path.join(save_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": results["best_epoch"],
                "best_val_auc": results["best_val_auc"],
                "test_metrics": results["test_metrics"],
                "model_cfg": model_cfg,
                "history": results["history"],
            },
            f,
            indent=2,
        )

    logger.info(f"Training completed successfully! Summary saved to {summary_path}")
    if results["test_metrics"]:
        logger.info(
            f"Final Test Evaluation: Global AUC: {results['test_metrics']['auc']:.4f} | ACC: {results['test_metrics']['acc']:.4f}"
        )


if __name__ == "__main__":
    main()
