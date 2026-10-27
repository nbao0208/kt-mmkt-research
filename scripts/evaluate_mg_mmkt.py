"""
CLI Entry Point for Evaluating Trained MG-MMKT on XES3G5M Test Set.

Performs:
1. Global micro ROC-AUC and Accuracy evaluation according to pyKT standards.
2. Multimodal Stratified Evaluation across question image presence and question frequency.
3. First-order Explainability extraction: Modality Attribution and Historical Evidence Top-K.

Usage:
    conda run -n kt-research-env python scripts/evaluate_mg_mmkt.py --checkpoint outputs/checkpoints/mg_mmkt_fold0/best_model.pt
"""

import argparse
import os
import sys
import json
from collections import Counter
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import get_logger
from src.utils.config import load_yaml_config
from src.data.xes3g5m_dataset import get_xes3g5m_dataloaders
from src.models.mmkt.mg_mmkt import MGMMKT
from src.evaluation.metrics import calculate_metrics, multimodal_stratified_evaluation
from src.evaluation.explainability import FirstOrderExplainer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained MG-MMKT model on XES3G5M.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--config-model", type=str, default="configs/model/mg_mmkt.yaml", help="Path to model config")
    parser.add_argument("--config-dataset", type=str, default="configs/dataset/xes3g5m.yaml", help="Path to dataset config")
    parser.add_argument("--val-fold", type=int, default=0, help="Fold index (default: 0)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--test-stride", type=int, default=100, help="Test strided context window step (default: 100)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'mps', 'cpu', or 'auto')")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader worker subprocesses (default: 0)")
    parser.add_argument("--output-report", type=str, default=None, help="Optional output JSON report path")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger("evaluate_mg_mmkt")
    logger.info("=== MG-MMKT Evaluation and Explainability Analysis ===")

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

    logger.info(f"Loading data from {data_dir} (val_fold={args.val_fold}, test_stride={args.test_stride})...")
    train_loader, val_loader, test_loader, item_bank = get_xes3g5m_dataloaders(
        data_dir=data_dir,
        val_fold=args.val_fold,
        batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        p_drop=0.0,
        test_stride=args.test_stride,
        num_workers=args.num_workers,
    )

    # 3. Compute Training Question Frequencies (Direct from training fold files - Issue L2)
    logger.info("Computing question frequencies directly from training fold sequence files...")
    train_folds = [f for f in range(5) if f != args.val_fold]
    q_counts = Counter()
    for f_idx in train_folds:
        seq_path = os.path.join(data_dir, f"sequences_fold_{f_idx}.pt")
        if os.path.exists(seq_path):
            f_data = torch.load(seq_path, map_location="cpu")
            q_seq_t = f_data["q_seq"]
            seq_mask_t = f_data["seq_mask"]
            valid_qs = q_seq_t[seq_mask_t].numpy().tolist()
            q_counts.update(valid_qs)
    logger.info(f"Counted {sum(q_counts.values()):,} interactions across {len(q_counts):,} unique training questions.")

    # 4. Load Checkpoint and Restore Model Configuration (Issue H3)
    logger.info(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_cfg" in ckpt and ckpt["model_cfg"] is not None:
        saved_cfg = ckpt["model_cfg"]
        logger.info(f"Restoring model architecture configuration from checkpoint: {saved_cfg}")
        for k, v in saved_cfg.items():
            model_cfg[k] = v

    # Instantiate Model & Load Weights
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
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 5. Full Inference & Tensor Collection on Test Set
    logger.info(f"Running inference across {len(test_loader)} test batches...")
    all_preds = []
    all_targets = []
    all_q_ids = []
    all_m_img_q = []
    all_prev_m_img_s = []
    all_alpha_q = []
    all_beta_s = []

    sample_evidence = []
    max_evidence_samples = 10
    explainer = FirstOrderExplainer()

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            outputs = model(batch)
            preds = outputs["preds"]  # [B, L]
            targets = batch["r_seq"]  # [B, L]
            eval_mask = batch["eval_mask"]  # [B, L]
            q_seq = batch["q_seq"]  # [B, L]
            mod_masks = batch["modality_masks"]  # [B, L, 3]
            alpha_q = outputs["alpha_q"]  # [B, L, num_branches]
            beta_s = outputs["beta_s"]  # [B, L, 2]
            attn_weights = outputs["attn_weights"]  # [B, L, L]
            leaf_kcs = batch.get("leaf_kcs")
            leaf_kc_mask = batch.get("leaf_kc_mask")
            context_start = batch.get("context_start", torch.zeros(preds.shape[0], dtype=torch.int64))

            # Shift solution image presence and attribution beta_s by 1 step to align with historical interaction (\tau = t - 1)
            prev_m_img_s = torch.zeros_like(mod_masks[:, :, 2])
            prev_m_img_s[:, 1:] = mod_masks[:, :-1, 2]

            prev_beta_s = torch.zeros_like(beta_s)
            prev_beta_s[:, 1:] = beta_s[:, :-1]

            # Collect active items
            active_p = preds[eval_mask].cpu().numpy()
            active_t = targets[eval_mask].cpu().numpy()
            active_q = q_seq[eval_mask].cpu().numpy()
            active_m_img = mod_masks[:, :, 0][eval_mask].cpu().numpy()
            active_prev_s_img = prev_m_img_s[eval_mask].cpu().numpy()
            active_alpha = alpha_q[eval_mask].cpu().numpy()
            active_prev_beta = prev_beta_s[eval_mask].cpu().numpy()

            all_preds.extend(active_p.tolist())
            all_targets.extend(active_t.tolist())
            all_q_ids.extend(active_q.tolist())
            all_m_img_q.extend(active_m_img.tolist())
            all_prev_m_img_s.extend(active_prev_s_img.tolist())
            all_alpha_q.extend(active_alpha.tolist())
            all_beta_s.extend(active_prev_beta.tolist())

            # Extract sample historical evidence across multiple students/batches (Issue L1)
            if len(sample_evidence) < max_evidence_samples:
                for b in range(preds.shape[0]):
                    if len(sample_evidence) >= max_evidence_samples:
                        break
                    valid_steps = torch.where(eval_mask[b])[0]
                    if len(valid_steps) > 0:
                        target_t = int(valid_steps[-1].item())
                        c_start = int(context_start[b].item()) if isinstance(context_start, torch.Tensor) else 0
                        ev = explainer.extract_top_k_historical_evidence(
                            attn_weights=attn_weights[b].cpu().numpy(),
                            step_t=target_t,
                            q_seq=q_seq[b].cpu().numpy(),
                            r_seq=targets[b].cpu().numpy(),
                            modality_masks=mod_masks[b].cpu().numpy(),
                            leaf_kcs=leaf_kcs[b].cpu().numpy() if leaf_kcs is not None else None,
                            leaf_kc_mask=leaf_kc_mask[b].cpu().numpy() if leaf_kc_mask is not None else None,
                            context_start=c_start,
                            k=3,
                        )
                        sample_evidence.append({
                            "batch_idx": batch_idx,
                            "student_batch_idx": b,
                            "target_step": target_t,
                            "actual_target_step": c_start + target_t,
                            "target_q": int(q_seq[b, target_t].item()),
                            "target_pred": float(preds[b, target_t].item()),
                            "target_label": int(targets[b, target_t].item()),
                            "top_k_evidence": ev,
                        })

    all_preds_np = np.array(all_preds)
    all_targets_np = np.array(all_targets)
    all_q_ids_np = np.array(all_q_ids)
    all_m_img_q_np = np.array(all_m_img_q)
    all_prev_m_img_s_np = np.array(all_prev_m_img_s)
    all_alpha_q_np = np.array(all_alpha_q)
    all_beta_s_np = np.array(all_beta_s)

    # 6. Compute Stratified Metrics
    stratified_metrics = multimodal_stratified_evaluation(
        preds=all_preds_np,
        targets=all_targets_np,
        q_ids=all_q_ids_np,
        m_img_q=all_m_img_q_np,
        prev_m_img_s=all_prev_m_img_s_np,
        train_question_counts=dict(q_counts),
        rare_threshold=10,
    )

    # 7. Compute Modality Attribution Statistics
    modality_report = explainer.analyze_modality_attribution(
        all_alpha_q=all_alpha_q_np,
        all_m_img_q=all_m_img_q_np,
    )
    solution_report = explainer.analyze_solution_attribution(
        all_beta_s=all_beta_s_np,
        all_m_img_s=all_prev_m_img_s_np,
    )

    # 8. Print Results
    logger.info("==========================================================================")
    logger.info("                     MG-MMKT EVALUATION REPORT                            ")
    logger.info("==========================================================================")
    logger.info(f"Overall Test AUC: {stratified_metrics['overall']['auc']:.4f} | ACC: {stratified_metrics['overall']['acc']:.4f} (N = {stratified_metrics['overall']['num_samples']:,})")
    logger.info("--------------------------------------------------------------------------")
    logger.info(f"Questions With Image: AUC: {stratified_metrics['questions_with_image']['auc']:.4f} | ACC: {stratified_metrics['questions_with_image']['acc']:.4f} (N = {stratified_metrics['questions_with_image']['num_samples']:,})")
    logger.info(f"Questions Text-Only:  AUC: {stratified_metrics['questions_text_only']['auc']:.4f} | ACC: {stratified_metrics['questions_text_only']['acc']:.4f} (N = {stratified_metrics['questions_text_only']['num_samples']:,})")
    logger.info("--------------------------------------------------------------------------")
    logger.info(f"Prior Sol Had Image:  AUC: {stratified_metrics['prior_step_had_solution_image']['auc']:.4f} | ACC: {stratified_metrics['prior_step_had_solution_image']['acc']:.4f} (N = {stratified_metrics['prior_step_had_solution_image']['num_samples']:,})")
    logger.info(f"Prior Sol Text-Only:  AUC: {stratified_metrics['prior_step_solution_text_only']['auc']:.4f} | ACC: {stratified_metrics['prior_step_solution_text_only']['acc']:.4f} (N = {stratified_metrics['prior_step_solution_text_only']['num_samples']:,})")
    logger.info("--------------------------------------------------------------------------")
    logger.info(f"Rare Questions:       AUC: {stratified_metrics['rare_questions']['auc']:.4f} | ACC: {stratified_metrics['rare_questions']['acc']:.4f} (N = {stratified_metrics['rare_questions']['num_samples']:,})")
    logger.info(f"Frequent Questions:   AUC: {stratified_metrics['frequent_questions']['auc']:.4f} | ACC: {stratified_metrics['frequent_questions']['acc']:.4f} (N = {stratified_metrics['frequent_questions']['num_samples']:,})")
    logger.info("==========================================================================")
    logger.info("                 FIRST-ORDER MODALITY ATTRIBUTION                         ")
    logger.info("==========================================================================")
    with_img_alpha = modality_report["with_image"]
    text_only_alpha = modality_report["text_only"]
    logger.info(f"Question With Image: Alpha_ID: {with_img_alpha['mean_alpha_id']:.3f} | Alpha_TXT: {with_img_alpha['mean_alpha_txt']:.3f} | Alpha_IMG: {with_img_alpha['mean_alpha_img']:.3f}")
    logger.info(f"Question Text-Only:  Alpha_ID: {text_only_alpha['mean_alpha_id']:.3f} | Alpha_TXT: {text_only_alpha['mean_alpha_txt']:.3f} | Alpha_IMG: {text_only_alpha['mean_alpha_img']:.3f}")
    if "statistical_test" in modality_report:
        st = modality_report["statistical_test"]
        logger.info(f"Statistical Test ({st.get('test_name')}): U={st.get('statistic', 0):.2f}, p-value={st.get('p_value', 1.0):.2e} (Significant: {st.get('significant_at_005')})")
    logger.info("--------------------------------------------------------------------------")
    logger.info(f"Prior Solution With Image (tau=t-1): Beta_TXT: {solution_report['with_image']['mean_beta_txt']:.3f} | Beta_IMG: {solution_report['with_image']['mean_beta_img']:.3f}")
    logger.info(f"Prior Solution Text-Only  (tau=t-1): Beta_TXT: {solution_report['text_only']['mean_beta_txt']:.3f} | Beta_IMG: {solution_report['text_only']['mean_beta_img']:.3f}")
    logger.info("==========================================================================")

    # 9. Save JSON Report
    output_path = args.output_report or os.path.join(
        os.path.dirname(args.checkpoint), "test_evaluation_report.json"
    )
    full_report = {
        "checkpoint": args.checkpoint,
        "model_config": model_cfg,
        "stratified_metrics": stratified_metrics,
        "question_modality_attribution": modality_report,
        "solution_modality_attribution": solution_report,
        "sample_historical_evidence": sample_evidence,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    logger.info(f"Full evaluation and explainability report saved to {output_path}")


if __name__ == "__main__":
    main()
