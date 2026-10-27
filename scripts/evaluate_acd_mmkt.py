"""
CLI Entry Point for Evaluating Trained ACD-MMKT on XES3G5M Test Set.

Performs:
1. pyKT-compliant Global Micro ROC-AUC and Accuracy evaluation (1,099,197 test interactions).
2. Multimodal Stratified Evaluation (with/without image, with/without solution image, cold-start items).
3. Psychometric difficulty correlation (Pearson r and Spearman rho between mu_q and empirical CTT D_q).
4. Three-tier Ante-hoc Explainability report extraction and Markdown serialization.

Usage:
    conda run -n kt-research-env python scripts/evaluate_acd_mmkt.py --checkpoint outputs/checkpoints/acd_mmkt_fold0/best_model.pt
"""

import argparse
import os
import sys
import json
from collections import Counter
import numpy as np
import scipy.stats as stats
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import get_logger
from src.utils.config import load_yaml_config
from src.data.xes3g5m_dataset import get_xes3g5m_dataloaders
from src.models.mmkt.acd_mmkt import ACDMMKT
from src.evaluation.metrics import calculate_metrics
from src.evaluation.acd_explainability import ACDExplainabilityEngine


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained ACD-MMKT model on XES3G5M.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--config-model", type=str, default="configs/model/acd_mmkt.yaml", help="Path to model config")
    parser.add_argument("--config-dataset", type=str, default="configs/dataset/xes3g5m.yaml", help="Path to dataset config")
    parser.add_argument("--val-fold", type=int, default=0, help="Fold index (default: 0)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--test-stride", type=int, default=100, help="Test strided context window step (default: 100)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'mps', 'cpu', or 'auto')")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader worker subprocesses (default: 0)")
    parser.add_argument("--output-report", type=str, default=None, help="Optional output JSON report path")
    parser.add_argument("--explain-output-md", type=str, default=None, help="Optional output Markdown path for explanation sample")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger("evaluate_acd_mmkt")
    logger.info("=== ACD-MMKT Multi-Tier Evaluation and Explainability Analysis ===")

    # 1. Device Selection
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)
    logger.info(f"Compute Device: {device}")

    # 2. Load Configurations & Data
    model_cfg = load_yaml_config(args.config_model)
    dataset_cfg = load_yaml_config(args.config_dataset)
    data_dir = dataset_cfg.get("data_dir", "data/processed/XES3G5M")

    # 3. Load Checkpoint & Restore Exact Model Configuration
    logger.info(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_cfg" in ckpt and ckpt["model_cfg"] is not None:
        saved_cfg = ckpt["model_cfg"]
        logger.info(f"Restoring exact architecture hyperparameters from checkpoint: {saved_cfg}")
        for k, v in saved_cfg.items():
            model_cfg[k] = v

    logger.info(f"Loading data from {data_dir} (val_fold={args.val_fold}, model_type='acd_mmkt')...")
    train_loader, val_loader, test_loader, item_bank = get_xes3g5m_dataloaders(
        data_dir=data_dir,
        val_fold=args.val_fold,
        model_type="acd_mmkt",
        batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        p_drop=0.0,
        test_stride=args.test_stride,
        num_workers=args.num_workers,
    )

    # Count training question frequencies
    train_folds = [f for f in range(5) if f != args.val_fold]
    q_counts = Counter()
    for f_idx in train_folds:
        seq_path = os.path.join(data_dir, f"sequences_fold_{f_idx}.pt")
        if os.path.exists(seq_path):
            f_data = torch.load(seq_path, map_location="cpu")
            valid_qs = f_data["q_seq"][f_data["seq_mask"]].numpy().tolist()
            q_counts.update(valid_qs)

    # 4. Instantiate Model & Load Weights
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
    logger.info("Model weights loaded successfully.")

    # 5. Global Micro Evaluation & Stratified Analysis
    all_preds = []
    all_targets = []

    # Stratified collection lists
    visual_q_preds, visual_q_targets = [], []
    text_only_q_preds, text_only_q_targets = [], []
    cold_start_preds, cold_start_targets = [], []
    frequent_preds, frequent_targets = [], []
    # Stratification by history solution image at preceding step t-1 (Issue M2, Plan Step 8.1.1)
    hist_sol_img_preds, hist_sol_img_targets = [], []
    hist_sol_txt_only_preds, hist_sol_txt_only_targets = [], []

    # Psychometric item difficulty tracking
    estimated_mu_dict = {}
    sample_batch_for_explain = None

    logger.info("Evaluating on independent test set...")
    with torch.no_grad():
        for batch in test_loader:
            if sample_batch_for_explain is None:
                sample_batch_for_explain = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            outputs = model(batch)

            preds = outputs["preds"].cpu().numpy()          # [B, L]
            targets = batch["r_seq"].cpu().numpy()          # [B, L]
            eval_mask = batch["eval_mask"].cpu().numpy()    # [B, L]
            q_seq = batch["q_seq"].cpu().numpy()            # [B, L]
            mod_masks = batch["modality_masks"].cpu().numpy()# [B, L, 3]
            mu_qs = outputs["mu_q"].cpu().numpy()           # [B, L]

            B, L = preds.shape
            for b in range(B):
                for t in range(L):
                    if not eval_mask[b, t]:
                        continue

                    p = float(preds[b, t])
                    y = int(targets[b, t])
                    q = int(q_seq[b, t])
                    has_img = bool(mod_masks[b, t, 0] > 0.5)
                    q_freq = q_counts.get(q, 0)
                    estimated_mu_dict[q] = float(mu_qs[b, t])

                    all_preds.append(p)
                    all_targets.append(y)

                    if has_img:
                        visual_q_preds.append(p)
                        visual_q_targets.append(y)
                    else:
                        text_only_q_preds.append(p)
                        text_only_q_targets.append(y)

                    if q_freq < 10:
                        cold_start_preds.append(p)
                        cold_start_targets.append(y)
                    else:
                        frequent_preds.append(p)
                        frequent_targets.append(y)

                    # Stratify by history solution image at preceding step t - 1
                    if t >= 1:
                        prev_has_sol_img = bool(mod_masks[b, t - 1, 2] > 0.5)
                        prev_has_sol_txt = bool(mod_masks[b, t - 1, 1] > 0.5)
                        if prev_has_sol_img:
                            hist_sol_img_preds.append(p)
                            hist_sol_img_targets.append(y)
                        elif prev_has_sol_txt:
                            hist_sol_txt_only_preds.append(p)
                            hist_sol_txt_only_targets.append(y)

    global_metrics = calculate_metrics(all_preds, all_targets)
    visual_metrics = calculate_metrics(visual_q_preds, visual_q_targets)
    text_metrics = calculate_metrics(text_only_q_preds, text_only_q_targets)
    cold_metrics = calculate_metrics(cold_start_preds, cold_start_targets)
    frequent_metrics = calculate_metrics(frequent_preds, frequent_targets)
    hist_sol_img_metrics = calculate_metrics(hist_sol_img_preds, hist_sol_img_targets)
    hist_sol_txt_metrics = calculate_metrics(hist_sol_txt_only_preds, hist_sol_txt_only_targets)

    # 6. Psychometric Correlation: Item-Bank Level (Issue M5) & Sequential Last-Seen
    ctt_diff_fold = item_bank["acd_ctt_difficulty_folds"][args.val_fold].numpy()
    ctt_valid_fold = item_bank["acd_ctt_valid_mask"][args.val_fold].numpy()

    # 6a. Intrinsic Item-Bank Difficulty Correlation (Forward Item Bank directly)
    with torch.no_grad():
        item_bank_out = model.forward_item_bank()
        bank_mu_q = item_bank_out["mu_q"].squeeze(-1).cpu().numpy()  # [7652]
    valid_mask_bank = ctt_valid_fold.astype(bool)
    if valid_mask_bank.sum() > 10:
        bank_mus = bank_mu_q[valid_mask_bank]
        ctt_mus_bank = ctt_diff_fold[valid_mask_bank]
        bank_pearson_r, _ = stats.pearsonr(bank_mus, ctt_mus_bank)
        bank_spearman_rho, _ = stats.spearmanr(bank_mus, ctt_mus_bank)
    else:
        bank_pearson_r, bank_spearman_rho = 0.0, 0.0

    # 6b. Sequential Last-Seen Interaction Difficulty Correlation
    eval_qs = [q for q in estimated_mu_dict if ctt_valid_fold[q]]
    if len(eval_qs) > 10:
        model_mus = np.array([estimated_mu_dict[q] for q in eval_qs])
        ctt_mus = np.array([ctt_diff_fold[q] for q in eval_qs])
        pearson_r, p_pearson = stats.pearsonr(model_mus, ctt_mus)
        spearman_rho, p_spearman = stats.spearmanr(model_mus, ctt_mus)
    else:
        pearson_r, spearman_rho = 0.0, 0.0

    logger.info("==========================================================================")
    logger.info(f"Global Test Micro AUC: {global_metrics['auc']:.4f} | ACC: {global_metrics['acc']:.4f} | Samples: {global_metrics['num_samples']:,}")
    logger.info(f"Visual Questions AUC:  {visual_metrics['auc']:.4f} | ACC: {visual_metrics['acc']:.4f} | Samples: {visual_metrics['num_samples']:,}")
    logger.info(f"Text-Only Questions:   {text_metrics['auc']:.4f} | ACC: {text_metrics['acc']:.4f} | Samples: {text_metrics['num_samples']:,}")
    logger.info(f"Cold-Start (<10) AUC:  {cold_metrics['auc']:.4f} | ACC: {cold_metrics['acc']:.4f} | Samples: {cold_metrics['num_samples']:,}")
    logger.info(f"Frequent Questions:    {frequent_metrics['auc']:.4f} | ACC: {frequent_metrics['acc']:.4f} | Samples: {frequent_metrics['num_samples']:,}")
    logger.info(f"Hist Sol-Image (t-1):  {hist_sol_img_metrics['auc']:.4f} | ACC: {hist_sol_img_metrics['acc']:.4f} | Samples: {hist_sol_img_metrics['num_samples']:,}")
    logger.info(f"Hist Sol-TextOnly(t-1):{hist_sol_txt_metrics['auc']:.4f} | ACC: {hist_sol_txt_metrics['acc']:.4f} | Samples: {hist_sol_txt_metrics['num_samples']:,}")
    logger.info(f"Item-Bank Intrinsic Difficulty Corr: Pearson r = {bank_pearson_r:.4f} | Spearman rho = {bank_spearman_rho:.4f}")
    logger.info(f"Sequential Last-Seen Difficulty Corr: Pearson r = {pearson_r:.4f} | Spearman rho = {spearman_rho:.4f}")
    logger.info("==========================================================================")

    # 7. Generate Explainability Sample
    explainer = ACDExplainabilityEngine(model, device=device)
    sample_explanation = explainer.explain_interaction(
        batch=sample_batch_for_explain,
        batch_idx=0,
        time_step=-1,
        top_k_history=5,
    )

    explain_md_path = args.explain_output_md or "outputs/reports/acd_mmkt_sample_explanation.md"
    explainer.export_report_markdown(sample_explanation, explain_md_path)
    logger.info(f"Exported sample Three-Tier explanation report to {explain_md_path}")

    # 8. Save Comprehensive Evaluation Report
    report_data = {
        "checkpoint": args.checkpoint,
        "val_fold": args.val_fold,
        "global_metrics": global_metrics,
        "stratified_metrics": {
            "visual_questions": visual_metrics,
            "text_only_questions": text_metrics,
            "cold_start_questions": cold_metrics,
            "frequent_questions": frequent_metrics,
            "hist_solution_image_preceding": hist_sol_img_metrics,
            "hist_solution_text_only_preceding": hist_sol_txt_metrics,
        },
        "psychometric_correlation": {
            "item_bank_intrinsic": {
                "num_valid_items": int(valid_mask_bank.sum()),
                "pearson_r": round(float(bank_pearson_r), 4),
                "spearman_rho": round(float(bank_spearman_rho), 4),
            },
            "sequential_last_seen": {
                "num_evaluated_questions": len(eval_qs),
                "pearson_r": round(float(pearson_r), 4),
                "spearman_rho": round(float(spearman_rho), 4),
            },
        },
        "model_cfg": model_cfg,
        "sample_explanation": sample_explanation,
    }

    out_json = args.output_report or f"outputs/reports/eval_acd_mmkt_fold{args.val_fold}.json"
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    logger.info(f"Saved evaluation report to {out_json}")


if __name__ == "__main__":
    main()
