"""
CLI Entry Point for Training ACD-MMKTv3 on XES3G5M or Eedi (Task 3 & 4) with pyKT 5-Fold CV.
Asymmetric Causal-Disentangled Multimodal Knowledge Tracing v3.

Usage:
    # Full End-to-End Training on XES3G5M Fold 0
    conda run -n kt-research-env python scripts/train_acd_mmkt_v3.py --val-fold 0 --num-epochs 50

    # Full End-to-End Training on Eedi Fold 0 (with Apple Silicon MPS profile)
    conda run -n kt-research-env python scripts/train_acd_mmkt_v3.py --dataset eedi --val-fold 0 --profile m1_pro

    # Ablation Studies (works identically on both XES3G5M and Eedi)
    conda run -n kt-research-env python scripts/train_acd_mmkt_v3.py --dataset eedi --val-fold 0 --ablation 2
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
from src.data.dataset_router import (
    resolve_dataset_key,
    resolve_dataset_paths,
    get_dataloaders_for_dataset,
)
from src.models.mmkt.acd_mmkt_v3 import ACDMMKTv3
from src.training.acd_v3_trainer import ACDMMKTv3Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train ACD-MMKTv3 on XES3G5M or Eedi dataset.")
    parser.add_argument("--dataset", type=str, choices=["xes3g5m", "eedi"], default=None, help="Target dataset shortcut ('xes3g5m' or 'eedi')")
    parser.add_argument("--config-model", type=str, default="configs/model/acd_mmkt_v3.yaml", help="Path to model config")
    parser.add_argument("--config-train", type=str, default=None, help="Path to training config")
    parser.add_argument("--config-dataset", type=str, default=None, help="Path to dataset config")
    parser.add_argument("--val-fold", type=int, default=None, help="Designated validation fold (0..4)")
    parser.add_argument("--test-split", type=str, choices=["unified", "public", "private"], default=None, help="Eedi test split ('unified', 'public', or 'private')")
    parser.add_argument("--post-diag-mode", type=str, choices=["none", "behavioral", "diagnostic"], default=None, help="Override post-response diagnostic vector mode")
    parser.add_argument("--num-epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cuda', 'cuda:0', 'mps', 'cpu')")
    parser.add_argument("--profile", type=str, choices=["m1_pro", "dual_gpu_5060ti", "dual_gpu_3060", "kaggle_t4", "kaggle_tpu_v5e", "colab_t4", "colab_l4", "colab_a100", "colab_tpu_v5e"], default=None, help="Hardware execution profile")
    parser.add_argument("--precision", type=str, choices=["fp32", "amp_fp16", "amp_bf16"], default=None, help="Computation precision")
    parser.add_argument("--grad-accum-steps", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of DataLoader workers")
    parser.add_argument("--save-dir", type=str, default=None, help="Checkpoint save directory")

    # Ablation presets and explicit flags
    parser.add_argument(
        "--ablation",
        type=int,
        choices=[0, 1, 2, 3, 4, 5, 6],
        default=None,
        help="Ablation preset: 0=Full, 1=w/o All Multimodal (Pure KC), 2=w/o Solution, 3=w/o Visual, 4=Unanchored Concept, 5=Polluted Experience, 6=w/o 2PL-IRT",
    )
    parser.add_argument("--wo-all-multimodal", action="store_true", help="Ablation 1: Disable all multimodal inputs (Pure KC)")
    parser.add_argument("--wo-solution", action="store_true", help="Ablation 2: Disable solution algorithmic stream")
    parser.add_argument("--wo-visual", action="store_true", help="Ablation 3: Disable text-guided visual stream")
    parser.add_argument("--unanchored-concept", action="store_true", help="Ablation 4: Learn unanchored concept difficulty and disable lambda_sparse")
    parser.add_argument("--polluted-experience", action="store_true", help="Ablation 5: Pass y_tau through causal self-attention")
    parser.add_argument("--wo-deep-irt", action="store_true", help="Ablation 6: Replace 2PL-IRT with black-box MLP")

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Resolve Dataset & Load Configurations
    config_dataset_path = args.config_dataset
    if config_dataset_path is None:
        config_dataset_path = "configs/dataset/eedi_task34.yaml" if args.dataset == "eedi" else "configs/dataset/xes3g5m.yaml"
    dataset_cfg = load_yaml_config(config_dataset_path)
    dataset_key = resolve_dataset_key(dataset_cfg=dataset_cfg, dataset_arg=args.dataset)

    config_train_path = args.config_train
    if config_train_path is None:
        eedi_train_cfg = "configs/train_acd_mmkt_v3_eedi.yaml"
        if dataset_key == "eedi" and os.path.exists(eedi_train_cfg):
            config_train_path = eedi_train_cfg
        else:
            config_train_path = "configs/train_acd_mmkt_v3.yaml"

    model_cfg = load_yaml_config(args.config_model)
    train_cfg = load_yaml_config(config_train_path)
    paths = resolve_dataset_paths(dataset_key=dataset_key, dataset_cfg=dataset_cfg, train_cfg=train_cfg)

    # Apply hardware profile if selected
    if args.profile and "profiles" in train_cfg and args.profile in train_cfg["profiles"]:
        prof = train_cfg["profiles"][args.profile]
        for k, v in prof.items():
            train_cfg[k] = v

    # Resolve overrides
    val_fold = args.val_fold if args.val_fold is not None else train_cfg.get("val_fold", 0)
    num_epochs = args.num_epochs if args.num_epochs is not None else train_cfg.get("num_epochs", 50)
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 64)
    test_batch_size = train_cfg.get("test_batch_size", 128)
    lr = args.lr if args.lr is not None else train_cfg.get("lr", 1e-3)
    seed = args.seed if args.seed is not None else train_cfg.get("seed", 42)
    device = args.device if args.device is not None else train_cfg.get("device", None)
    precision = args.precision if args.precision is not None else train_cfg.get("precision", "fp32")
    grad_accum_steps = args.grad_accum_steps if args.grad_accum_steps is not None else train_cfg.get("grad_accum_steps", 1)
    num_workers = args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 0)
    test_split = args.test_split or train_cfg.get("test_split", "unified")

    if args.post_diag_mode is not None:
        model_cfg["post_diag_mode"] = args.post_diag_mode
    elif "post_diag_mode" in train_cfg and train_cfg["post_diag_mode"] is not None:
        model_cfg["post_diag_mode"] = train_cfg["post_diag_mode"]

    # Record dataset identity in model_cfg for checkpoint restoration
    model_cfg["dataset_name"] = paths["display_name"]
    model_cfg["num_questions"] = dataset_cfg.get("num_questions", model_cfg.get("num_questions", 7652))
    model_cfg["num_kcs"] = dataset_cfg.get("num_kcs", model_cfg.get("num_kcs", 1175))

    # Resolve Ablation Preset and flags
    ablation_suffix = ""
    if args.ablation is not None:
        if args.ablation == 0:
            ablation_suffix = ""
        elif args.ablation == 1:
            model_cfg["use_multimodal"] = False
            train_cfg["beta_d"] = 0.0
            train_cfg["lambda_sparse"] = 0.0
            train_cfg["p_drop"] = 0.0
            model_cfg["p_visual_drop"] = 0.0
            ablation_suffix = "_ablation1_wo_all_multimodal"
        elif args.ablation == 2:
            model_cfg["use_solution"] = False
            train_cfg["beta_d"] = 0.20
            train_cfg["lambda_sparse"] = 0.01
            ablation_suffix = "_ablation2_wo_solution"
        elif args.ablation == 3:
            model_cfg["use_visual"] = False
            train_cfg["beta_d"] = 0.20
            train_cfg["lambda_sparse"] = 0.01
            train_cfg["p_drop"] = 0.0
            model_cfg["p_visual_drop"] = 0.0
            ablation_suffix = "_ablation3_wo_visual"
        elif args.ablation == 4:
            model_cfg["unanchored_concept"] = True
            train_cfg["lambda_sparse"] = 0.0
            ablation_suffix = "_ablation4_unanchored"
        elif args.ablation == 5:
            model_cfg["polluted_experience_attn"] = True
            ablation_suffix = "_ablation5_polluted_exp"
        elif args.ablation == 6:
            model_cfg["use_deep_irt"] = False
            ablation_suffix = "_ablation6_wo_2pl_irt"

    # Apply individual ablation flags if set
    if args.wo_all_multimodal:
        model_cfg["use_multimodal"] = False
        train_cfg["beta_d"] = 0.0
        train_cfg["lambda_sparse"] = 0.0
        train_cfg["p_drop"] = 0.0
        model_cfg["p_visual_drop"] = 0.0
        ablation_suffix = ablation_suffix or "_ablation1_wo_all_multimodal"
    if args.wo_solution:
        model_cfg["use_solution"] = False
        ablation_suffix = ablation_suffix or "_ablation2_wo_solution"
    if args.wo_visual:
        model_cfg["use_visual"] = False
        train_cfg["p_drop"] = 0.0
        model_cfg["p_visual_drop"] = 0.0
        ablation_suffix = ablation_suffix or "_ablation3_wo_visual"
    if args.unanchored_concept:
        model_cfg["unanchored_concept"] = True
        train_cfg["lambda_sparse"] = 0.0
        ablation_suffix = ablation_suffix or "_unanchored"
    if args.polluted_experience:
        model_cfg["polluted_experience_attn"] = True
        ablation_suffix = ablation_suffix or "_polluted_exp"
    if args.wo_deep_irt:
        model_cfg["use_deep_irt"] = False
        ablation_suffix = ablation_suffix or "_wo_2pl_irt"

    # Dynamic save_dir per dataset, fold, and ablation
    if args.save_dir is not None:
        save_dir = args.save_dir
    elif train_cfg.get("save_dir") is not None:
        save_dir = train_cfg["save_dir"]
        if ablation_suffix and not save_dir.endswith(ablation_suffix):
            save_dir = f"{save_dir}{ablation_suffix}"
    else:
        save_dir = os.path.join(paths["checkpoint_dir"], f"acd_mmkt_v3_fold{val_fold}{ablation_suffix}")

    log_dir = paths["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_acd_mmkt_v3_fold{val_fold}{ablation_suffix}.log")
    logger = get_logger(f"train_v3_{dataset_key}_f{val_fold}", log_file=log_file)

    if args.profile:
        logger.info(f"Applied hardware profile '{args.profile}'.")
    logger.info(f"Dataset: {paths['display_name']} (key={dataset_key}) | post_diag_mode={model_cfg.get('post_diag_mode', 'none')}")
    logger.info(f"Target Checkpoint Directory: {save_dir}")

    # Synchronize p_drop across configs
    p_drop = train_cfg.get("p_drop", model_cfg.get("p_visual_drop", 0.15))
    train_cfg["p_drop"] = p_drop
    model_cfg["p_visual_drop"] = p_drop
    model_cfg["p_drop"] = p_drop

    data_dir = paths["data_dir"]

    # Set Reproducibility Seed
    set_seed(seed)

    # 2. Construct DataLoaders
    logger.info(f"Loading {paths['display_name']} data from {data_dir} for Validation Fold {val_fold} (model_type='acd_mmkt_v3')...")
    train_loader, val_loader, test_loader, static_item_bank = get_dataloaders_for_dataset(
        dataset_key=dataset_key,
        data_dir=data_dir,
        val_fold=val_fold,
        model_type="acd_mmkt_v3",
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        p_drop=p_drop,
        test_stride=train_cfg.get("test_stride", dataset_cfg.get("test_stride", 100)),
        num_workers=num_workers,
        test_split=test_split,
    )

    # Retrieve fold statistics
    train_dataset = train_loader.dataset
    item_bank_meta = train_dataset.get_static_item_bank()
    n_q = item_bank_meta.get("N_q")
    d_q = item_bank_meta.get("D_q")
    omega_q = item_bank_meta.get("omega_q")
    mu_concept_prior = item_bank_meta.get("mu_concept")

    # 3. Instantiate ACD-MMKTv3 Model
    logger.info("Instantiating ACD-MMKTv3 model architecture...")
    model = ACDMMKTv3(
        static_item_bank=static_item_bank,
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
        use_multimodal=model_cfg.get("use_multimodal", True),
        use_solution=model_cfg.get("use_solution", True),
        use_visual=model_cfg.get("use_visual", True),
        unanchored_concept=model_cfg.get("unanchored_concept", False),
        polluted_experience_attn=model_cfg.get("polluted_experience_attn", False),
        use_deep_irt=model_cfg.get("use_deep_irt", True),
        use_hypergraph=model_cfg.get("use_hypergraph", True),
        use_gram_schmidt=model_cfg.get("use_gram_schmidt", True),
        post_diag_mode=model_cfg.get("post_diag_mode", "none"),
    )

    # Free memory from dictionary reference
    del static_item_bank
    import gc
    gc.collect()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model successfully built. Total trainable parameters: {total_params:,}")

    # 4. Initialize Trainer and Execute Training Loop
    trainer = ACDMMKTv3Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        n_q=n_q,
        d_q=d_q,
        omega_q=omega_q,
        mu_concept_prior=mu_concept_prior,
        lr=lr,
        weight_decay=train_cfg.get("weight_decay", 1e-4),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        patience=train_cfg.get("patience", 10),
        device=device,
        precision=precision,
        grad_accum_steps=grad_accum_steps,
        save_dir=save_dir,
        val_fold=val_fold,
        model_cfg=model_cfg,
        warmup_epochs=train_cfg.get("warmup_epochs", 2),
        eta_min=train_cfg.get("eta_min", 1e-5),
        beta_d=train_cfg.get("beta_d", 0.20),
        lambda_sparse=train_cfg.get("lambda_sparse", 0.01),
        kappa=train_cfg.get("kappa", 10.0),
        logger=logger,
    )

    train_results = trainer.train(num_epochs=num_epochs)

    # Save training summary JSON
    summary_path = os.path.join(save_dir, "train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(train_results, f, indent=2)
    logger.info(f"Saved training execution summary to {summary_path}")


if __name__ == "__main__":
    main()

