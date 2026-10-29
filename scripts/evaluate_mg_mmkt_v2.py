"""
CLI Entry Point for Evaluating Trained MG-MMKT v2 on XES3G5M Test Set.

Performs:
1. pyKT-compliant Global Micro ROC-AUC and Accuracy evaluation on test set (exactly 1,099,197 interactions).
2. Multimodal Stratified Evaluation:
   - Question image present vs text-only.
   - Post-response remedial learning gain (following r_{t-1} = 0) vs mastery (following r_{t-1} = 1).
   - Prior solution image present vs text-only.
   - Question frequency tiers (Rare <= 10, Medium 11-100, Frequent > 100).
3. 3-Level Explainability Extraction:
   - Level 1: Modality attribution distribution (alpha^q, beta^s) and Mann-Whitney U test.
   - Level 2: Top-K historical evidence with pathway labeling (Mastery vs Remedial) and head decay theta_h.
   - Level 3: Implicit cognitive demand attribution (concept cosine, algorithmic demand cosine, diagnosis).

Usage:
    conda run -n kt-research-env python scripts/evaluate_mg_mmkt_v2.py --checkpoint outputs/checkpoints/mg_mmkt_v2_fold0/best_model.pt
"""

import argparse
import os
import sys
import json
from collections import Counter
from typing import Dict, Any, List
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import get_logger
from src.utils.config import load_yaml_config
from src.data.xes3g5m_dataset import get_xes3g5m_dataloaders
from src.models.mmkt.mg_mmkt_v2 import MGMMKTv2
from src.evaluation.metrics import calculate_metrics
from src.evaluation.explainability_v2 import MultiLevelExplainerV2


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained MG-MMKT v2 model on XES3G5M.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--config-model", type=str, default="configs/model/mg_mmkt_v2.yaml", help="Path to model config")
    parser.add_argument("--config-dataset", type=str, default="configs/dataset/xes3g5m.yaml", help="Path to dataset config")
    parser.add_argument("--val-fold", type=int, default=0, help="Fold index used for train/val split (default: 0)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--test-stride", type=int, default=100, help="Test strided context window step (default: 100)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'mps', 'cpu', or 'auto')")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers (default: 0)")
    parser.add_argument("--output-report", type=str, default=None, help="Optional output JSON report path")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger("evaluate_mg_mmkt_v2")
    logger.info("=" * 60)
    logger.info("=== MG-MMKT v2 Evaluation and 3-Level Explainability Analysis ===")

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
    data_dir = dataset_cfg.get("processed_data_dir", dataset_cfg.get("data_dir", "data/processed/XES3G5M"))

    logger.info(f"Loading data from {data_dir} (val_fold={args.val_fold}, test_stride={args.test_stride})...")
    _, _, test_loader, item_bank = get_xes3g5m_dataloaders(
        data_dir=data_dir,
        val_fold=args.val_fold,
        batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        p_drop=0.0,
        test_stride=args.test_stride,
        num_workers=args.num_workers,
    )

    # 3. Compute Training Question Frequencies (Direct from training fold files)
    logger.info("Computing question frequencies directly from 4 training fold sequence files...")
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

    # 4. Load Checkpoint & Restore Model Configuration
    logger.info(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_cfg" in ckpt and ckpt["model_cfg"] is not None:
        saved_cfg = ckpt["model_cfg"]
        logger.info(f"Restoring model architecture configuration from checkpoint: {saved_cfg}")
        for k, v in saved_cfg.items():
            model_cfg[k] = v

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
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Extract head decay rates theta_h
    head_thetas = None
    if hasattr(model.causal_attention, "raw_theta"):
        head_thetas = torch.nn.functional.softplus(model.causal_attention.raw_theta).detach().cpu().numpy().tolist()
        logger.info(f"Learned Attention Head Forgetting Decay Rates (theta_1..theta_H): {[f'{th:.4f}' for th in head_thetas]}")

    # 5. Full Evaluation Loop over Test Set
    logger.info("Executing comprehensive test set evaluation...")
    all_preds = []
    all_targets = []
    all_q_ids = []
    all_m_img_q = []
    all_m_img_s = []
    all_prev_r = []
    all_prev_m_img_s = []
    all_shared_kc = []

    # Explainability buffers
    all_alpha_q = []
    all_beta_s = []

    # Global Level 3 Cognitive Demand buffers across all evaluated steps
    all_cos_hx = []
    all_norm_hx = []
    all_cos_hz = []
    all_norm_hz = []
    all_cos_distill = []

    # Representative cases for Level 2 and Level 3
    sample_cases = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            q_seq = batch["q_seq"].to(device)
            r_seq = batch["r_seq"].to(device)
            seq_mask = batch["seq_mask"].to(device)
            eval_mask = batch["eval_mask"].to(device)
            eval_mask_bool = eval_mask.bool()
            raw_mod_masks = batch.get("raw_modality_masks", batch.get("modality_masks")).to(device)
            context_start = batch.get("context_start", torch.zeros_like(q_seq[:, 0]))

            outputs = model(
                q_seq=q_seq,
                r_seq=r_seq,
                seq_mask=seq_mask,
                modality_masks=raw_mod_masks,
                raw_modality_masks=raw_mod_masks,
            )

            probs = outputs["probs"].cpu().numpy()
            alpha_q = outputs["alpha_q"].cpu().numpy()
            beta_s = outputs["beta_s"].cpu().numpy()
            gamma_bar = outputs["gamma_bar"].cpu().numpy()
            h = outputs["h"]
            x = outputs["x"]
            hat_z_s = outputs["hat_z_s"]
            z_s_clean = outputs["z_s_clean"]

            # Compute vectorized Level 3 alignment metrics for evaluated steps
            if hat_z_s is not None and z_s_clean is not None:
                cos_hx = F.cosine_similarity(h, x, dim=-1, eps=1e-8)
                norm_hx = torch.norm(h * x, p=2, dim=-1)
                cos_hz = F.cosine_similarity(h, hat_z_s, dim=-1, eps=1e-8)
                norm_hz = torch.norm(h * hat_z_s, p=2, dim=-1)
                cos_dist = F.cosine_similarity(hat_z_s, z_s_clean, dim=-1, eps=1e-8)

                all_cos_hx.append(cos_hx[eval_mask_bool].cpu().numpy())
                all_norm_hx.append(norm_hx[eval_mask_bool].cpu().numpy())
                all_cos_hz.append(cos_hz[eval_mask_bool].cpu().numpy())
                all_norm_hz.append(norm_hz[eval_mask_bool].cpu().numpy())
                all_cos_distill.append(cos_dist[eval_mask_bool].cpu().numpy())

            eval_mask_np = eval_mask_bool.cpu().numpy()
            q_seq_np = q_seq.cpu().numpy()
            r_seq_np = r_seq.cpu().numpy()
            mod_masks_np = raw_mod_masks.cpu().numpy()

            leaf_kcs_np = batch["leaf_kcs"].cpu().numpy() if "leaf_kcs" in batch else None
            leaf_kc_mask_np = batch["leaf_kc_mask"].cpu().numpy().astype(bool) if "leaf_kc_mask" in batch else None

            for b in range(q_seq.size(0)):
                for t in range(q_seq.size(1)):
                    if eval_mask_np[b, t]:
                        all_preds.append(float(probs[b, t]))
                        all_targets.append(int(r_seq_np[b, t]))
                        all_q_ids.append(int(q_seq_np[b, t]))
                        all_m_img_q.append(float(mod_masks_np[b, t, 0]))
                        all_m_img_s.append(float(mod_masks_np[b, t, 2]))
                        all_alpha_q.append(alpha_q[b, t])
                        all_beta_s.append(beta_s[b, t])

                        # Track previous step state (for remedial vs mastery gain & shared KC condition)
                        if t > 0:
                            all_prev_r.append(int(r_seq_np[b, t - 1]))
                            all_prev_m_img_s.append(float(mod_masks_np[b, t - 1, 2]))
                            if leaf_kcs_np is not None and leaf_kc_mask_np is not None:
                                kcs_curr = set(leaf_kcs_np[b, t][leaf_kc_mask_np[b, t]].tolist())
                                kcs_prev = set(leaf_kcs_np[b, t - 1][leaf_kc_mask_np[b, t - 1]].tolist())
                                all_shared_kc.append(bool(len(kcs_curr.intersection(kcs_prev)) > 0))
                            else:
                                all_shared_kc.append(False)
                        else:
                            all_prev_r.append(-1)
                            all_prev_m_img_s.append(-1.0)
                            all_shared_kc.append(False)

                        # Capture a few representative explainability case studies
                        if len(sample_cases) < 10 and (len(sample_cases) == 0 or np.random.rand() < 0.005):
                            kcs_history = None
                            if leaf_kcs_np is not None and leaf_kc_mask_np is not None:
                                kcs_history = [
                                    leaf_kcs_np[b, tau][leaf_kc_mask_np[b, tau]].tolist()
                                    for tau in range(t)
                                ]
                            sample_cases.append({
                                "b": b,
                                "t": t,
                                "uid": int(batch["uid"][b].item()),
                                "context_start": int(context_start[b].item()),
                                "q_history": q_seq_np[b, :t].tolist(),
                                "r_history": r_seq_np[b, :t].tolist(),
                                "kcs_history": kcs_history,
                                "gamma_bar_row": gamma_bar[b, t, :t],
                                "h_t": h[b, t].cpu(),
                                "x_t": x[b, t].cpu(),
                                "hat_z_t_s": hat_z_s[b, t].cpu() if hat_z_s is not None else None,
                                "z_t_s_clean": z_s_clean[b, t].cpu() if z_s_clean is not None else None,
                                "prob_t": float(probs[b, t]),
                                "target_t": int(r_seq_np[b, t]),
                            })

    flat_preds = np.array(all_preds, dtype=np.float64)
    flat_targets = np.array(all_targets, dtype=np.int64)
    flat_q_ids = np.array(all_q_ids, dtype=np.int64)
    flat_m_img_q = np.array(all_m_img_q, dtype=np.float32)
    flat_m_img_s = np.array(all_m_img_s, dtype=np.float32)
    flat_prev_r = np.array(all_prev_r, dtype=np.int64)
    flat_prev_m_img_s = np.array(all_prev_m_img_s, dtype=np.float32)
    flat_shared_kc = np.array(all_shared_kc, dtype=bool)
    mat_alpha_q = np.array(all_alpha_q, dtype=np.float32)
    mat_beta_s = np.array(all_beta_s, dtype=np.float32)

    logger.info(f"Evaluated Total Active Interactions: {len(flat_preds):,}")

    # 6. Overall Performance
    overall_metrics = calculate_metrics(flat_preds, flat_targets)
    logger.info(f"-> Global Test ROC-AUC: {overall_metrics['auc']:.4f} | Accuracy: {overall_metrics['acc']:.4f}")

    # 7. Stratified Evaluation
    stratified_report = {}

    # 7a. Question Image Modality Stratification
    q_img_mask = (flat_m_img_q == 1.0)
    q_txt_mask = (flat_m_img_q == 0.0)
    stratified_report["questions_with_image"] = calculate_metrics(flat_preds[q_img_mask], flat_targets[q_img_mask])
    stratified_report["questions_text_only"] = calculate_metrics(flat_preds[q_txt_mask], flat_targets[q_txt_mask])
    logger.info(
        f"Question Image: With Image AUC={stratified_report['questions_with_image']['auc']:.4f} ({q_img_mask.sum():,} steps) | "
        f"Text Only AUC={stratified_report['questions_text_only']['auc']:.4f} ({q_txt_mask.sum():,} steps)"
    )

    # 7b. Remedial vs Mastery Pedagogical Stratification (Status at t-1)
    prev_valid_mask = (flat_prev_r >= 0)
    after_wrong_mask = prev_valid_mask & (flat_prev_r == 0)
    after_wrong_same_kc_mask = after_wrong_mask & flat_shared_kc
    after_right_mask = prev_valid_mask & (flat_prev_r == 1)
    after_right_same_kc_mask = after_right_mask & flat_shared_kc

    stratified_report["after_wrong_response_remedial"] = calculate_metrics(flat_preds[after_wrong_mask], flat_targets[after_wrong_mask])
    stratified_report["after_wrong_same_kc_remedial"] = calculate_metrics(flat_preds[after_wrong_same_kc_mask], flat_targets[after_wrong_same_kc_mask])
    stratified_report["after_correct_response_mastery"] = calculate_metrics(flat_preds[after_right_mask], flat_targets[after_right_mask])
    stratified_report["after_correct_same_kc_mastery"] = calculate_metrics(flat_preds[after_right_same_kc_mask], flat_targets[after_right_same_kc_mask])
    logger.info(
        f"Pedagogical History: After Wrong (Remedial All) AUC={stratified_report['after_wrong_response_remedial']['auc']:.4f} ({after_wrong_mask.sum():,} steps) | "
        f"After Wrong Same KC (Remedial) AUC={stratified_report['after_wrong_same_kc_remedial']['auc']:.4f} ({after_wrong_same_kc_mask.sum():,} steps) | "
        f"After Right (Mastery All) AUC={stratified_report['after_correct_response_mastery']['auc']:.4f} ({after_right_mask.sum():,} steps) | "
        f"After Right Same KC (Mastery) AUC={stratified_report['after_correct_same_kc_mastery']['auc']:.4f} ({after_right_same_kc_mask.sum():,} steps)"
    )

    # 7c. Prior Step Solution Modality Stratification (Status at t-1)
    prev_sol_valid = (flat_prev_m_img_s >= 0.0)
    prev_sol_img_mask = prev_sol_valid & (flat_prev_m_img_s == 1.0)
    prev_sol_txt_only_mask = prev_sol_valid & (flat_prev_m_img_s == 0.0)

    stratified_report["prior_step_with_solution_image"] = calculate_metrics(flat_preds[prev_sol_img_mask], flat_targets[prev_sol_img_mask])
    stratified_report["prior_step_solution_text_only"] = calculate_metrics(flat_preds[prev_sol_txt_only_mask], flat_targets[prev_sol_txt_only_mask])
    logger.info(
        f"Prior Solution Modality: With Solution Image AUC={stratified_report['prior_step_with_solution_image']['auc']:.4f} ({prev_sol_img_mask.sum():,} steps) | "
        f"Solution Text Only AUC={stratified_report['prior_step_solution_text_only']['auc']:.4f} ({prev_sol_txt_only_mask.sum():,} steps)"
    )

    # 7d. Question Frequency Stratification (Rare <= 10, Medium 11-100, Frequent > 100)
    q_counts_arr = np.array([q_counts.get(int(q), 0) for q in flat_q_ids])
    rare_mask = (q_counts_arr <= 10)
    med_mask = (q_counts_arr > 10) & (q_counts_arr <= 100)
    freq_mask = (q_counts_arr > 100)

    stratified_report["rare_questions_cold_start"] = calculate_metrics(flat_preds[rare_mask], flat_targets[rare_mask])
    stratified_report["medium_frequency_questions"] = calculate_metrics(flat_preds[med_mask], flat_targets[med_mask])
    stratified_report["frequent_questions"] = calculate_metrics(flat_preds[freq_mask], flat_targets[freq_mask])

    logger.info(
        f"Frequency: Rare (<=10) AUC={stratified_report['rare_questions_cold_start']['auc']:.4f} ({rare_mask.sum():,} steps) | "
        f"Medium (11-100) AUC={stratified_report['medium_frequency_questions']['auc']:.4f} ({med_mask.sum():,} steps) | "
        f"Frequent (>100) AUC={stratified_report['frequent_questions']['auc']:.4f} ({freq_mask.sum():,} steps)"
    )

    # 8. Multi-Level Explainability Extraction
    explainer = MultiLevelExplainerV2()
    logger.info("Extracting 3-Level Explainability Report...")

    # Level 1: Modality Attribution
    l1_report = explainer.analyze_level1_modality_attribution(
        all_alpha_q=mat_alpha_q,
        all_beta_s=mat_beta_s,
        all_m_img_q=flat_m_img_q,
        all_m_img_s=flat_m_img_s,
    )
    logger.info(
        f"Level 1 Question Image Attribution: With Image mean={l1_report['question_level']['with_image']['mean_alpha_img']:.4f} vs "
        f"Text Only mean={l1_report['question_level']['text_only']['mean_alpha_img']:.4f}"
    )

    # Level 2 & Level 3 Case Studies
    detailed_cases = []
    for sc in sample_cases:
        top_k_evidence = explainer.extract_level2_historical_evidence(
            gamma_bar_row=sc["gamma_bar_row"],
            q_history=sc["q_history"],
            r_history=sc["r_history"],
            kcs_history=sc.get("kcs_history"),
            context_start=sc["context_start"],
            top_k=3,
            head_thetas=head_thetas,
        )

        l3_diag = {}
        if sc["hat_z_t_s"] is not None:
            l3_diag = explainer.analyze_level3_cognitive_demand(
                h_t=sc["h_t"],
                x_t=sc["x_t"],
                hat_z_t_s=sc["hat_z_t_s"],
                z_t_s_clean=sc["z_t_s_clean"],
                prob_t=sc["prob_t"],
            )

        detailed_cases.append({
            "uid": sc["uid"],
            "step": sc["context_start"] + sc["t"],
            "target_response": sc["target_t"],
            "predicted_prob": sc["prob_t"],
            "level2_top_evidence": top_k_evidence,
            "level3_diagnosis": l3_diag,
        })

    # Level 3: Global Cognitive Demand Statistics across the Test Set
    global_l3_stats = {}
    if len(all_cos_hx) > 0:
        global_cos_hx = np.concatenate(all_cos_hx)
        global_norm_hx = np.concatenate(all_norm_hx)
        global_cos_hz = np.concatenate(all_cos_hz)
        global_norm_hz = np.concatenate(all_norm_hz)
        global_cos_distill = np.concatenate(all_cos_distill) if len(all_cos_distill) > 0 else np.array([])

        pred_correct_mask = (flat_preds >= 0.5)
        pred_incorrect_mask = (flat_preds < 0.5)

        global_l3_stats["overall"] = {
            "mean_concept_cosine": float(np.mean(global_cos_hx)),
            "std_concept_cosine": float(np.std(global_cos_hx)),
            "mean_concept_hadamard_norm": float(np.mean(global_norm_hx)),
            "std_concept_hadamard_norm": float(np.std(global_norm_hx)),
            "mean_cognitive_demand_cosine": float(np.mean(global_cos_hz)),
            "std_cognitive_demand_cosine": float(np.std(global_cos_hz)),
            "mean_cognitive_demand_hadamard_norm": float(np.mean(global_norm_hz)),
            "std_cognitive_demand_hadamard_norm": float(np.std(global_norm_hz)),
            "mean_distillation_fidelity_cosine": float(np.mean(global_cos_distill)) if len(global_cos_distill) > 0 else None,
            "std_distillation_fidelity_cosine": float(np.std(global_cos_distill)) if len(global_cos_distill) > 0 else None,
        }
        global_l3_stats["predicted_proficient"] = {
            "mean_concept_cosine": float(np.mean(global_cos_hx[pred_correct_mask])) if np.any(pred_correct_mask) else 0.0,
            "mean_cognitive_demand_cosine": float(np.mean(global_cos_hz[pred_correct_mask])) if np.any(pred_correct_mask) else 0.0,
        }
        global_l3_stats["predicted_struggling"] = {
            "mean_concept_cosine": float(np.mean(global_cos_hx[pred_incorrect_mask])) if np.any(pred_incorrect_mask) else 0.0,
            "mean_cognitive_demand_cosine": float(np.mean(global_cos_hz[pred_incorrect_mask])) if np.any(pred_incorrect_mask) else 0.0,
        }
        logger.info(
            f"Global Level 3 Cognitive Alignment -> Concept Cosine: {global_l3_stats['overall']['mean_concept_cosine']:.4f} (std={global_l3_stats['overall']['std_concept_cosine']:.4f}) | "
            f"Cognitive Demand Cosine: {global_l3_stats['overall']['mean_cognitive_demand_cosine']:.4f} (std={global_l3_stats['overall']['std_cognitive_demand_cosine']:.4f})"
        )

    # 9. Synthesize and Export JSON Report
    final_report = {
        "model_name": "mg_mmkt_v2",
        "val_fold": args.val_fold,
        "total_test_interactions": len(flat_preds),
        "overall": overall_metrics,
        "stratified": stratified_report,
        "level1_modality_attribution": l1_report,
        "level2_head_decay_rates": head_thetas,
        "sample_explainability_cases": detailed_cases,
        "level3_global_statistics": global_l3_stats,
    }

    output_path = args.output_report
    if output_path is None:
        save_dir = os.path.dirname(args.checkpoint)
        output_path = os.path.join(save_dir, "test_evaluation_report.json")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    logger.info(f"Evaluation report successfully written to: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

