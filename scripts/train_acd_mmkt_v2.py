"""
CLI Entry Point for Training ACD-MMKTv2 on XES3G5M with pyKT 5-Fold Cross-Validation.
Asymmetric Causal-Disentangled Multimodal Knowledge Tracing v2.

Usage:
    # Full Two-Stage Training on Fold 0
    conda run -n kt-research-env python scripts/train_acd_mmkt_v2.py --val-fold 0 --num-epochs 50

    # With hardware profile for Apple Silicon (M-series MPS)
    conda run -n kt-research-env python scripts/train_acd_mmkt_v2.py --val-fold 0 --profile m1_pro

    # Ablation Studies
    conda run -n kt-research-env python scripts/train_acd_mmkt_v2.py --val-fold 0 --ablation 2  # w/o Solution Blueprint
    conda run -n kt-research-env python scripts/train_acd_mmkt_v2.py --val-fold 0 --ablation 3  # w/o Fine-Grained Alignment
    conda run -n kt-research-env python scripts/train_acd_mmkt_v2.py --val-fold 0 --ablation 4  # w/o HGNN & Gram-Schmidt
    conda run -n kt-research-env python scripts/train_acd_mmkt_v2.py --val-fold 0 --ablation 5  # w/o Monotonic 2PL-IRT
    conda run -n kt-research-env python scripts/train_acd_mmkt_v2.py --val-fold 0 --ablation 6  # w/o Stage 1 Pre-training
"""

import argparse
import os
import sys
import json

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.utils.config import load_yaml_config
from src.data.xes3g5m_dataset import get_xes3g5m_dataloaders
from src.models.mmkt.acd_mmkt_v2 import ACDMMKTv2
from src.training.acd_v2_trainer import ACDMMKTv2Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train ACD-MMKTv2 on XES3G5M dataset.")
    parser.add_argument("--config-model", type=str, default="configs/model/acd_mmkt_v2.yaml", help="Path to model config")
    parser.add_argument("--config-train", type=str, default="configs/train_acd_mmkt_v2.yaml", help="Path to training config")
    parser.add_argument("--config-dataset", type=str, default="configs/dataset/xes3g5m.yaml", help="Path to dataset config")
    parser.add_argument("--val-fold", type=int, default=None, help="Designated validation fold (0..4)")
    parser.add_argument("--num-epochs", type=int, default=None, help="Number of Stage 2 fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cuda', 'cuda:0', 'xla', 'mps', 'cpu')")
    parser.add_argument("--profile", type=str, choices=["m1_pro", "dual_gpu_5060ti", "dual_gpu_3060", "kaggle_t4", "kaggle_tpu_v5e"], default=None, help="Hardware execution profile")
    parser.add_argument("--precision", type=str, choices=["fp32", "amp_fp16", "amp_bf16"], default=None, help="Computation precision")
    parser.add_argument("--grad-accum-steps", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of DataLoader workers")
    parser.add_argument("--save-dir", type=str, default=None, help="Checkpoint save directory")

    # Stage 1 overrides
    parser.add_argument("--skip-stage1", action="store_true", help="Skip Stage 1 item-bank pre-training")
    parser.add_argument("--stage1-epochs", type=int, default=None, help="Number of Stage 1 pre-training epochs")
    parser.add_argument("--stage1-lr", type=float, default=None, help="Stage 1 learning rate")

    # Ablation presets and explicit flags
    parser.add_argument(
        "--ablation",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=None,
        help="Ablation preset: 1=Full, 2=w/o Solution, 3=w/o Fine-Grained, 4=w/o HGNN/Rasch, 5=w/o DeepIRT, 6=w/o Stage1",
    )
    parser.add_argument("--wo-solution", action="store_true", help="Ablation 2: Disable solution cognitive blueprint")
    parser.add_argument("--wo-fine-grained", action="store_true", help="Ablation 3: Disable token/patch cross-attention")
    parser.add_argument("--wo-hypergraph-rasch", action="store_true", help="Ablation 4: Disable HGNN and Gram-Schmidt Rasch")
    parser.add_argument("--wo-deep-irt", action="store_true", help="Ablation 5: Disable monotonic 2PL-IRT readout")
    parser.add_argument("--wo-stage1", action="store_true", help="Ablation 6: Disable Stage 1 item-bank pre-training")

    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger("train_acd_mmkt_v2")

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
    save_dir = args.save_dir if args.save_dir is not None else train_cfg.get("save_dir", f"outputs/checkpoints/acd_mmkt_v2_fold{val_fold}")
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
        device_str = (
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

    set_seed(seed)
    logger.info(f"Environment: Device={device} | Precision={precision} | Seed={seed} | Fold={val_fold}")

    # Apply ablation overrides
    if args.ablation == 1:
        pass
    elif args.ablation == 2 or args.wo_solution:
        logger.info("Ablation 2: Disabling solution cognitive blueprint (use_solution=False)")
        model_cfg["use_solution"] = False
    elif args.ablation == 3 or args.wo_fine_grained:
        logger.info("Ablation 3: Disabling fine-grained alignment (use_fine_grained=False)")
        model_cfg["use_fine_grained"] = False
    elif args.ablation == 4 or args.wo_hypergraph_rasch:
        logger.info("Ablation 4: Disabling HGNN & Gram-Schmidt Rasch (use_hypergraph=False, use_gram_schmidt=False)")
        model_cfg["use_hypergraph"] = False
        model_cfg["use_gram_schmidt"] = False
    elif args.ablation == 5 or args.wo_deep_irt:
        logger.info("Ablation 5: Disabling Monotonic 2PL-IRT readout (use_deep_irt=False)")
        model_cfg["use_deep_irt"] = False
    elif args.ablation == 6 or args.wo_stage1 or args.skip_stage1:
        logger.info("Ablation 6: Disabling Stage 1 item-bank pre-training (use_stage1_pretrain=False)")
        model_cfg["use_stage1_pretrain"] = False

    use_stage1 = model_cfg.get("use_stage1_pretrain", True) and not args.skip_stage1
    stage1_epochs = args.stage1_epochs if args.stage1_epochs is not None else train_cfg.get("stage1_epochs", 15)
    stage1_lr = args.stage1_lr if args.stage1_lr is not None else float(train_cfg.get("stage1_lr", 1e-3))

    # 2. Data Loaders
    logger.info(f"Loading data from {data_dir} (val_fold={val_fold}, model_type='acd_mmkt_v2')...")
    train_loader, val_loader, test_loader, item_bank = get_xes3g5m_dataloaders(
        data_dir=data_dir,
        val_fold=val_fold,
        model_type="acd_mmkt_v2",
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        p_drop=float(train_cfg.get("p_drop", 0.15)),
        test_stride=int(train_cfg.get("test_stride", 100)),
        num_workers=num_workers,
    )

    # 3. Model Instantiation
    logger.info("Instantiating ACD-MMKTv2 model...")
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

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model instantiated with {total_params:,} trainable parameters.")

    # 4. Trainer Initialization & Execution
    trainer = ACDMMKTv2Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
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
        stage1_epochs=stage1_epochs,
        stage1_batch_size=int(train_cfg.get("stage1_batch_size", 256)),
        stage1_lr=stage1_lr,
        beta_d=float(model_cfg.get("beta_difficulty", 0.20)),
        kappa=float(model_cfg.get("kappa_smoothing", 10.0)),
        use_stage1_pretrain=use_stage1,
        logger=logger,
    )

    fit_results = trainer.fit(num_epochs=num_epochs)
    logger.info(f"Training completed successfully. Checkpoint saved at: {fit_results['checkpoint_path']}")


if __name__ == "__main__":
    main()
