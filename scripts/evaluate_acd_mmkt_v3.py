"""
CLI Entry Point for Evaluating Trained ACD-MMKTv3 on XES3G5M Test Set.
Asymmetric Causal-Disentangled Multimodal Knowledge Tracing v3.

Performs:
1. pyKT-compliant Global Micro ROC-AUC and Accuracy evaluation (1,099,197 test interactions).
2. Multimodal Stratified Evaluation:
   - Stem image present vs text-only.
   - Solution image present vs text-only.
3. Signed Visual Cognitive Load Stratification:
   - Visual Scaffold / Aid (Delta_mu_vis < 0).
   - Visual Overload / Distractor (Delta_mu_vis > 0).
4. Item Frequency Stratification:
   - Rare / Cold-start (N_q <= 10) vs Medium (11 <= N_q <= 100) vs Frequent (N_q > 100).
   - Remedial learning response (r_{t-1} = 0 vs r_{t-1} = 1).
5. Psychometric Difficulty Correlation:
   - Pearson r and Spearman rho between model estimated difficulties and empirical error rates.
   - 100% Decision Faithfulness verification: (p_hat >= 0.5) <=> (theta >= tilde_mu).
6. 4-Tier Explainability Diagnostic report extraction and Markdown serialization.

Usage:
    conda run -n kt-research-env python scripts/evaluate_acd_mmkt_v3.py --checkpoint outputs/checkpoints/acd_mmkt_v3_fold0/best_model.pt
"""

import argparse
import os
import sys
import json
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import scipy.stats as stats
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import get_logger
from src.utils.config import load_yaml_config
from src.data.xes3g5m_dataset import get_xes3g5m_dataloaders
from src.models.mmkt.acd_mmkt_v3 import ACDMMKTv3
from src.evaluation.metrics import calculate_metrics, multimodal_stratified_evaluation
from src.evaluation.acd_v3_explainability import ACDv3ExplainabilityEngine


def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Safe Pearson correlation guarding against zero-variance constant arrays under Ablations."""
    if len(x) < 2 or np.std(x) <= 1e-8 or np.std(y) <= 1e-8:
        return 0.0, 1.0
    r, p = stats.pearsonr(x, y)
    return (0.0 if np.isnan(r) else float(r)), (1.0 if np.isnan(p) else float(p))


def safe_spearmanr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Safe Spearman correlation guarding against zero-variance constant arrays under Ablations."""
    if len(x) < 2 or np.std(x) <= 1e-8 or np.std(y) <= 1e-8:
        return 0.0, 1.0
    r, p = stats.spearmanr(x, y)
    return (0.0 if np.isnan(r) else float(r)), (1.0 if np.isnan(p) else float(p))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained ACD-MMKTv3 model on XES3G5M.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--config-model", type=str, default="configs/model/acd_mmkt_v3.yaml", help="Path to model config")
    parser.add_argument("--config-dataset", type=str, default="configs/dataset/xes3g5m.yaml", help="Path to dataset config")
    parser.add_argument("--val-fold", type=int, default=None, help="Fold index (default: auto-detected from checkpoint)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--test-stride", type=int, default=100, help="Test strided context window step (default: 100)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'mps', 'cpu', or 'auto')")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (default: 0)")
    parser.add_argument("--output-report", type=str, default=None, help="Optional output JSON report path")
    parser.add_argument("--explain-output-md", type=str, default=None, help="Optional output Markdown path for explanation sample")
    return parser.parse_args()


def main():
    args = parse_args()
    log_dir = "outputs/logs"
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger("evaluate_acd_mmkt_v3", log_file=os.path.join(log_dir, "evaluate_acd_mmkt_v3.log"))
    logger.info("=== ACD-MMKTv3 Comprehensive Evaluation and Psychometric Analysis ===")

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

    # Auto-resolve validation fold to prevent cross-fold anchor leakage
    val_fold = args.val_fold
    if val_fold is None:
        val_fold = int(ckpt.get("val_fold", 0))
        logger.info(f"Auto-resolved validation fold from checkpoint: fold {val_fold}")
    else:
        logger.info(f"Using explicitly specified validation fold: fold {val_fold}")

    logger.info(f"Loading data from {data_dir} (val_fold={val_fold}, model_type='acd_mmkt_v3')...")
    _, _, test_loader, item_bank = get_xes3g5m_dataloaders(
        data_dir=data_dir,
        val_fold=val_fold,
        model_type="acd_mmkt_v3",
        batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        p_drop=0.0,
        test_stride=args.test_stride,
        num_workers=args.num_workers,
    )

    # Load fold auxiliary metadata
    aux_path = os.path.join(data_dir, "acd_v3_aux_metadata.pt")
    if os.path.exists(aux_path):
        aux_data = torch.load(aux_path, map_location="cpu")
        mu_concept_prior = aux_data["mu_concept_folds"][val_fold].to(device)
        mu_prior_prob = aux_data["mu_prior_folds"][val_fold].numpy()
        n_q_train = aux_data["N_q_folds"][val_fold].numpy()
    else:
        mu_concept_prior = torch.full((7652,), -1.37, device=device)
        mu_prior_prob = np.full(7652, 0.20)
        n_q_train = np.zeros(7652)

    # 4. Instantiate Model and Load Weights
    logger.info("Instantiating ACDMMKTv3 architecture...")
    model = ACDMMKTv3(
        static_item_bank=item_bank,
        d_raw_txt=model_cfg.get("d_raw_txt", 768),
        d_raw_img=model_cfg.get("d_raw_img", 768),
        d_model=model_cfg.get("d_model", 128),
        num_heads=model_cfg.get("num_heads", 4),
        max_seq_len=model_cfg.get("max_seq_len", 200),
        dropout=model_cfg.get("dropout", 0.15),
        p_drop=0.0,
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

    # Cache raw modality masks and static context before releasing item_bank
    raw_masks_all = item_bank["modality_masks"].numpy()
    static_ctx_all = item_bank.get("question_static_ctx", torch.zeros((7652, 15), dtype=torch.float32)).numpy()
    del item_bank
    import gc
    gc.collect()

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    # Precompute all 7,652 static question representations once for inference
    logger.info("Precomputing offline item bank representations for fast evaluation...")
    model.precompute_item_bank_cache(mu_concept_prior)
    cached_props = model.cached_item_properties

    # 5. Global Micro pyKT Evaluation Loop
    logger.info("Executing evaluation on test interactions...")
    all_preds = []
    all_targets = []
    all_q_ids = []
    all_thetas = []
    all_tilde_mus = []
    all_prev_r = []
    all_prev_sol_img = []

    total_valid = 0
    sample_batch_for_explain = None

    for batch_idx, batch in enumerate(test_loader):
        if sample_batch_for_explain is None:
            sample_batch_for_explain = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        eval_mask = batch["eval_mask"].bool()
        targets = batch["r_seq"]
        q_seq = batch["q_seq"]

        with torch.no_grad():
            outputs = model(batch, mu_concept_prior=mu_concept_prior, use_cache=True)

        mask_np = eval_mask.cpu().numpy().astype(bool)
        probs_np = outputs["probs"].cpu().numpy()
        targets_np = targets.cpu().numpy()
        q_np = q_seq.cpu().numpy()
        theta_np = outputs["theta"].cpu().numpy()
        tilde_mu_np = outputs["tilde_mu"].cpu().numpy()

        # Vectorized slice collection
        all_preds.append(probs_np[mask_np])
        all_targets.append(targets_np[mask_np])
        all_q_ids.append(q_np[mask_np])
        all_thetas.append(theta_np[mask_np])
        all_tilde_mus.append(tilde_mu_np[mask_np])

        # Previous response (shifted right, step 0 is -1)
        prev_r_mat = np.pad(targets_np[:, :-1], ((0, 0), (1, 0)), constant_values=-1)
        all_prev_r.append(prev_r_mat[mask_np])

        # Prior step had solution image (shifted right, step 0 is 0)
        sol_img_mat = (raw_masks_all[q_np, 2] > 0.5).astype(np.int64)
        prev_sol_img_mat = np.pad(sol_img_mat[:, :-1], ((0, 0), (1, 0)), constant_values=0)
        all_prev_sol_img.append(prev_sol_img_mat[mask_np])

        total_valid += int(mask_np.sum())

    logger.info(f"Total evaluated test interactions: {total_valid:,} (Expected: 1,099,197)")

    preds_arr = np.concatenate(all_preds).astype(np.float32)
    targets_arr = np.concatenate(all_targets).astype(np.int64)
    q_arr = np.concatenate(all_q_ids).astype(np.int64)
    thetas_arr = np.concatenate(all_thetas).astype(np.float32)
    tilde_mus_arr = np.concatenate(all_tilde_mus).astype(np.float32)
    prev_r_arr = np.concatenate(all_prev_r).astype(np.int64)
    prev_sol_img_arr = np.concatenate(all_prev_sol_img).astype(np.int64)

    # 6. Global pyKT Metrics (called with exactly 2 arguments)
    global_metrics = calculate_metrics(preds_arr, targets_arr)
    auc_global = global_metrics["auc"]
    acc_global = global_metrics["acc"]
    logger.info(f"=== GLOBAL PYKT METRICS === AUC: {auc_global:.4f}, ACC: {acc_global:.4f}")

    # 7. Mathematical Faithfulness Verification: (p_hat >= 0.5) <=> (theta >= tilde_mu)
    faithfulness_mask = (preds_arr >= 0.5) == (thetas_arr >= tilde_mus_arr)
    faithfulness_pct = float(faithfulness_mask.mean()) * 100.0
    logger.info(f"2PL-IRT Decision Faithfulness: {faithfulness_pct:.2f}% (Matches: {int(faithfulness_mask.sum())} / {len(preds_arr)})")

    # 8. Multimodal Stratifications
    q_has_stem_img = raw_masks_all[q_arr, 0] > 0.5
    q_has_sol_img = raw_masks_all[q_arr, 2] > 0.5

    def get_subset_metrics(subset_mask, name):
        count = int(subset_mask.sum())
        if count == 0:
            return {"count": 0, "auc": 0.0, "acc": 0.0}
        sub_m = calculate_metrics(preds_arr[subset_mask], targets_arr[subset_mask])
        logger.info(f"  [{name}] Count: {count:,}, AUC: {sub_m['auc']:.4f}, ACC: {sub_m['acc']:.4f}")
        return {"count": count, "auc": float(sub_m["auc"]), "acc": float(sub_m["acc"])}

    logger.info("=== MULTIMODAL STRATIFICATION ===")
    strat_stem_img = get_subset_metrics(q_has_stem_img, "Stem Image Present (m_img^q = 1)")
    strat_stem_txt = get_subset_metrics(~q_has_stem_img, "Stem Text Only (m_img^q = 0)")
    strat_sol_img = get_subset_metrics(q_has_sol_img, "Solution Image Present (m_img^s = 1)")
    strat_sol_txt = get_subset_metrics(~q_has_sol_img, "Solution Text Only (m_img^s = 0)")

    # Prior step solution modality slice
    strat_prev_sol_img = get_subset_metrics(prev_sol_img_arr == 1, "Prior Step Had Solution Image (m_{img,t-1}^s = 1)")
    strat_prev_sol_txt = get_subset_metrics(prev_sol_img_arr == 0, "Prior Step Solution Text Only (m_{img,t-1}^s = 0)")

    # 9. Signed Visual Load Stratification (Delta_mu_vis < 0 vs Delta_mu_vis > 0)
    delta_vis_all = cached_props["delta_mu_vis"].cpu().numpy()
    q_delta_vis = delta_vis_all[q_arr]

    vis_scaffold_mask = q_has_stem_img & (q_delta_vis < 0.0)
    vis_overload_mask = q_has_stem_img & (q_delta_vis > 0.0)

    logger.info("=== SIGNED VISUAL LOAD STRATIFICATION ===")
    strat_vis_scaffold = get_subset_metrics(vis_scaffold_mask, "Visual Scaffold / Aid (Delta_mu_vis < 0)")
    strat_vis_overload = get_subset_metrics(vis_overload_mask, "Visual Overload / Distractor (Delta_mu_vis > 0)")

    # Signed visual load error rate residual analysis (D_q^test - mu_prior)
    test_q_counts = Counter(q_arr)
    test_q_errors = Counter()
    for q_id, target in zip(q_arr, targets_arr):
        if target == 0:
            test_q_errors[q_id] += 1

    stem_img_active_qs = [q for q, c in test_q_counts.items() if c >= 5 and raw_masks_all[q, 0] > 0.5]
    mean_res_scaffold = 0.0
    mean_res_overload = 0.0
    if len(stem_img_active_qs) > 0:
        stem_img_qs_arr = np.array(stem_img_active_qs)
        q_err_rates = np.array([test_q_errors[q] / test_q_counts[q] for q in stem_img_active_qs])
        q_priors = mu_prior_prob[stem_img_qs_arr]
        diff_residuals = q_err_rates - q_priors
        q_delta_vis_items = delta_vis_all[stem_img_qs_arr]

        scaffold_items = q_delta_vis_items < 0.0
        overload_items = q_delta_vis_items > 0.0

        mean_res_scaffold = float(diff_residuals[scaffold_items].mean()) if scaffold_items.any() else 0.0
        mean_res_overload = float(diff_residuals[overload_items].mean()) if overload_items.any() else 0.0

        logger.info(f"Signed Visual Load Pedagogical Validation (N={len(stem_img_active_qs)} items):")
        logger.info(f"  Visual Scaffold (Delta_mu_vis < 0, N={int(scaffold_items.sum())}): Mean (D_q^test - mu_prior) = {mean_res_scaffold:+.4f}")
        logger.info(f"  Visual Overload (Delta_mu_vis > 0, N={int(overload_items.sum())}): Mean (D_q^test - mu_prior) = {mean_res_overload:+.4f}")

    # 10. Frequency Stratification
    q_n_train = n_q_train[q_arr]
    mask_rare = q_n_train <= 10
    mask_medium = (q_n_train > 10) & (q_n_train <= 100)
    mask_frequent = q_n_train > 100

    logger.info("=== ITEM FREQUENCY & COLD-START STRATIFICATION ===")
    strat_rare = get_subset_metrics(mask_rare, "Rare / Cold-start (N_q <= 10)")
    strat_medium = get_subset_metrics(mask_medium, "Medium (11 <= N_q <= 100)")
    strat_frequent = get_subset_metrics(mask_frequent, "Frequent (N_q > 100)")

    # Remedial learning slice
    mask_remedial = prev_r_arr == 0
    mask_mastery = prev_r_arr == 1
    logger.info("=== REMEDIAL LEARNING SLICE ===")
    strat_remedial = get_subset_metrics(mask_remedial, "After Error (r_{t-1} = 0)")
    strat_mastery = get_subset_metrics(mask_mastery, "After Correct (r_{t-1} = 1)")

    # 11. Psychometric Difficulty Decomposition Correlations
    logger.info("=== PSYCHOMETRIC DIFFICULTY DECOMPOSITION CORRELATIONS ===")
    active_test_qs = [q for q, count in test_q_counts.items() if count >= 5]
    active_qs_arr = np.array(active_test_qs)
    emp_rates = np.array([test_q_errors[q] / test_q_counts[q] for q in active_test_qs])
    model_mus = cached_props["mu_q"].cpu().numpy()[active_qs_arr]
    model_tilde_mus = cached_props["tilde_mu"].cpu().numpy()[active_qs_arr]
    concept_mus = cached_props["mu_concept"].cpu().numpy()[active_qs_arr]
    delta_stem_mus = cached_props["delta_mu_stem"].cpu().numpy()[active_qs_arr]
    delta_txt_mus = cached_props["delta_mu_txt"].cpu().numpy()[active_qs_arr]
    delta_sol_mus = cached_props["delta_mu_sol"].cpu().numpy()[active_qs_arr]

    pearson_r, p_val = safe_pearsonr(model_mus, emp_rates)
    spearman_rho, rho_pval = safe_spearmanr(model_tilde_mus, emp_rates)
    concept_r, _ = safe_pearsonr(concept_mus, emp_rates)
    delta_stem_r, _ = safe_pearsonr(delta_stem_mus, emp_rates)
    delta_txt_r, _ = safe_pearsonr(delta_txt_mus, emp_rates)
    delta_sol_r, _ = safe_pearsonr(delta_sol_mus, emp_rates)

    stem_img_mask_in_active = raw_masks_all[active_qs_arr, 0] > 0.5
    if stem_img_mask_in_active.any():
        delta_vis_active = delta_vis_all[active_qs_arr[stem_img_mask_in_active]]
        emp_rates_vis = emp_rates[stem_img_mask_in_active]
        vis_r, _ = safe_pearsonr(delta_vis_active, emp_rates_vis)
    else:
        vis_r = 0.0

    logger.info(f"Correlation with Test Empirical Error Rate (N_items={len(active_test_qs)}):")
    logger.info(f"  Total mu_q: Pearson r = {pearson_r:.4f} (p={p_val:.2e}), Spearman rho = {spearman_rho:.4f} (p={rho_pval:.2e})")
    logger.info(f"  Concept Anchor mu_concept: Pearson r = {concept_r:.4f}")
    logger.info(f"  Stem Load Delta_mu_stem: Pearson r = {delta_stem_r:.4f}")
    logger.info(f"  Stem Text Load Delta_mu_txt: Pearson r = {delta_txt_r:.4f}")
    logger.info(f"  Stem Visual Load Delta_mu_vis (on image items): Pearson r = {vis_r:.4f}")
    logger.info(f"  Solution Load Delta_mu_sol: Pearson r = {delta_sol_r:.4f}")

    # Cognitive gate lambda_q statistics across math domains
    domain_names = [
        "Plane Geometry", "Solid Geometry", "Functions", "Algebra",
        "Arithmetic", "Probability & Combinatorics", "Number Theory",
        "Sequences", "Trigonometry", "Analytic Geometry",
    ]
    bar_lambda_all = cached_props["bar_lambda_q"].cpu().numpy().reshape(-1)
    domain_stats = {}
    logger.info("=== COGNITIVE GATE INTENSITY (bar_lambda_q) ACROSS MATH DOMAINS ===")
    for d_idx, d_name in enumerate(domain_names):
        d_mask = (static_ctx_all[:, 2 + d_idx] > 0.5) & (raw_masks_all[:, 0] > 0.5)
        if d_mask.any():
            mean_l = float(bar_lambda_all[d_mask].mean())
            std_l = float(bar_lambda_all[d_mask].std())
            domain_stats[d_name] = {"count": int(d_mask.sum()), "mean_lambda": mean_l, "std_lambda": std_l}
            logger.info(f"  [{d_name}] N={int(d_mask.sum())}, Mean bar_lambda_q: {mean_l:.4f} ± {std_l:.4f}")

    # 12. 4-Tier Explainability Diagnostic Sample
    explain_engine = ACDv3ExplainabilityEngine(model=model, aux_metadata_path=aux_path, device=device)
    explanation = explain_engine.explain_interaction(
        batch=sample_batch_for_explain,
        sample_idx=0,
        target_step=-1,
        mu_concept_prior=mu_concept_prior,
    )
    md_report = explain_engine.format_markdown_report(explanation)

    if args.explain_output_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.explain_output_md)), exist_ok=True)
        with open(args.explain_output_md, "w", encoding="utf-8") as f:
            f.write(md_report)
        logger.info(f"Saved 4-tier explainability report to {args.explain_output_md}")

    # Compile Evaluation Summary Dictionary
    eval_summary = {
        "model_name": "ACD-MMKTv3",
        "checkpoint": args.checkpoint,
        "val_fold": val_fold,
        "total_evaluated_interactions": total_valid,
        "global_metrics": {
            "auc": float(auc_global),
            "acc": float(acc_global),
            "decision_faithfulness_pct": float(faithfulness_pct),
        },
        "multimodal_stratification": {
            "stem_image": strat_stem_img,
            "stem_text_only": strat_stem_txt,
            "solution_image": strat_sol_img,
            "solution_text_only": strat_sol_txt,
            "prior_step_solution_image": strat_prev_sol_img,
            "prior_step_solution_text_only": strat_prev_sol_txt,
        },
        "signed_visual_load_stratification": {
            "visual_scaffold": strat_vis_scaffold,
            "visual_overload": strat_vis_overload,
            "mean_error_residual_scaffold": mean_res_scaffold,
            "mean_error_residual_overload": mean_res_overload,
        },
        "frequency_stratification": {
            "rare_cold_start": strat_rare,
            "medium": strat_medium,
            "frequent": strat_frequent,
        },
        "remedial_stratification": {
            "after_error": strat_remedial,
            "after_correct": strat_mastery,
        },
        "psychometric_correlation": {
            "pearson_r_mu_q": float(pearson_r),
            "spearman_rho_tilde_mu": float(spearman_rho),
            "pearson_r_mu_concept": float(concept_r),
            "pearson_r_delta_stem": float(delta_stem_r),
            "pearson_r_delta_txt": float(delta_txt_r),
            "pearson_r_delta_vis": float(vis_r),
            "pearson_r_delta_sol": float(delta_sol_r),
            "active_items_count": len(active_test_qs),
        },
        "domain_gate_intensity": domain_stats,
    }

    if args.output_report:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_report)), exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(eval_summary, f, indent=2)
        logger.info(f"Saved complete evaluation report to {args.output_report}")

    logger.info("=== Comprehensive Evaluation Completed Successfully ===")


if __name__ == "__main__":
    main()

