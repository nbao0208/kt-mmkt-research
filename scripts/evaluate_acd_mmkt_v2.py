"""
CLI Entry Point for Evaluating Trained ACD-MMKTv2 on XES3G5M Test Set.
Asymmetric Causal-Disentangled Multimodal Knowledge Tracing v2.

Performs:
1. pyKT-compliant Global Micro ROC-AUC and Accuracy evaluation (1,099,197 test interactions).
2. Multimodal Stratified Evaluation:
   - Question image present vs text-only.
   - Solution image present vs text-only.
   - Question frequency stratification (Rare / Cold-start <= 10 vs Medium vs Frequent).
3. Psychometric difficulty correlation (Pearson r and Spearman rho between mu_q and test empirical error rates).
4. 100% Decision Faithfulness Verification: (p_hat >= 0.5) <=> (theta >= tilde_mu).
5. 4-Tier Explainability Diagnostic report extraction and Markdown serialization.

Usage:
    conda run -n kt-research-env python scripts/evaluate_acd_mmkt_v2.py --checkpoint outputs/checkpoints/acd_mmkt_v2_fold0/best_model.pt
"""

import argparse
import os
import sys
import json
from collections import Counter
import numpy as np
import scipy.stats as stats
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import get_logger
from src.utils.config import load_yaml_config
from src.data.xes3g5m_dataset import get_xes3g5m_dataloaders
from src.models.mmkt.acd_mmkt_v2 import ACDMMKTv2
from src.evaluation.metrics import calculate_metrics, multimodal_stratified_evaluation
from src.evaluation.acd_v2_explainability import ACDv2ExplainabilityEngine


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained ACD-MMKTv2 model on XES3G5M.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--config-model", type=str, default="configs/model/acd_mmkt_v2.yaml", help="Path to model config")
    parser.add_argument("--config-dataset", type=str, default="configs/dataset/xes3g5m.yaml", help="Path to dataset config")
    parser.add_argument("--val-fold", type=int, default=0, help="Fold index (default: 0)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--test-stride", type=int, default=100, help="Test strided context window step (default: 100)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'mps', 'cpu', or 'auto')")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (default: 0)")
    parser.add_argument("--output-report", type=str, default=None, help="Optional output JSON report path")
    parser.add_argument("--explain-output-md", type=str, default=None, help="Optional output Markdown path for explanation sample")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger("evaluate_acd_mmkt_v2")
    logger.info("=== ACD-MMKTv2 Comprehensive Evaluation and Psychometric Analysis ===")

    # 1. Device Selection
    if args.device == "auto":
        if "PJRT_DEVICE" in os.environ and os.environ.get("PJRT_DEVICE") == "TPU" or "TPU_NAME" in os.environ:
            try:
                import torch_xla.core.xla_model as xm
                device = xm.xla_device()
            except ImportError:
                device = torch.device("xla")
        else:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
            )
    elif "xla" in args.device:
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
        except ImportError:
            device = torch.device(args.device)
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

    logger.info(f"Loading data from {data_dir} (val_fold={args.val_fold}, model_type='acd_mmkt_v2')...")
    train_loader, val_loader, test_loader, item_bank = get_xes3g5m_dataloaders(
        data_dir=data_dir,
        val_fold=args.val_fold,
        model_type="acd_mmkt_v2",
        batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        p_drop=0.0,
        test_stride=args.test_stride,
        num_workers=args.num_workers,
    )

    # Question occurrence counts across training folds
    train_folds = [f for f in range(5) if f != args.val_fold]
    q_train_counts = Counter()
    for f_idx in train_folds:
        seq_path = os.path.join(data_dir, f"sequences_fold_{f_idx}.pt")
        if os.path.exists(seq_path):
            f_data = torch.load(seq_path, map_location="cpu")
            valid_qs = f_data["q_seq"][f_data["seq_mask"]].numpy().tolist()
            q_train_counts.update(valid_qs)

    # 4. Instantiate Model & Load Weights
    model = ACDMMKTv2(
        static_item_bank=item_bank,
        d_raw_txt=int(model_cfg.get("d_raw_txt", 768)),
        d_raw_img=int(model_cfg.get("d_raw_img", 768)),
        d_model=int(model_cfg.get("d_model", 128)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        max_seq_len=int(model_cfg.get("max_seq_len", 200)),
        dropout=float(model_cfg.get("dropout", 0.15)),
        hgnn_layers=int(model_cfg.get("hgnn_layers", 2)),
        use_solution=bool(model_cfg.get("use_solution", True)),
        use_fine_grained=bool(model_cfg.get("use_fine_grained", True)),
        use_hypergraph=bool(model_cfg.get("use_hypergraph", True)),
        use_gram_schmidt=bool(model_cfg.get("use_gram_schmidt", True)),
        use_deep_irt=bool(model_cfg.get("use_deep_irt", True)),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Model weights loaded successfully. Precomputing item bank cache...")
    model.precompute_item_cache()

    # 5. Full Test Set Evaluation
    logger.info("Executing evaluation on test set (1,099,197 pyKT interactions)...")
    all_preds = []
    all_targets = []
    all_q_ids = []
    all_m_img_q = []
    all_m_img_s = []

    first_batch_sample = None

    with torch.no_grad():
        for b_idx, batch in enumerate(test_loader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            if first_batch_sample is None:
                first_batch_sample = {k: v.cpu() for k, v in batch.items() if isinstance(v, torch.Tensor)}

            eval_mask = batch["eval_mask"].bool()
            targets = batch["r_seq"]
            q_seq = batch["q_seq"]
            raw_mods = batch["raw_modality_masks"]

            out = model(batch, use_cache=True)
            probs = out["probs"]

            if eval_mask.any():
                v_preds = probs[eval_mask].cpu().numpy()
                v_targets = targets[eval_mask].cpu().numpy()
                v_qs = q_seq[eval_mask].cpu().numpy()
                v_img_q = raw_mods[..., 0][eval_mask].cpu().numpy()
                v_img_s = raw_mods[..., 2][eval_mask].cpu().numpy()

                all_preds.extend(v_preds)
                all_targets.extend(v_targets)
                all_q_ids.extend(v_qs)
                all_m_img_q.extend(v_img_q)
                all_m_img_s.extend(v_img_s)

    preds_arr = np.array(all_preds, dtype=np.float64)
    targets_arr = np.array(all_targets, dtype=np.int64)
    q_ids_arr = np.array(all_q_ids, dtype=np.int64)
    m_img_q_arr = np.array(all_m_img_q, dtype=np.float32)
    m_img_s_arr = np.array(all_m_img_s, dtype=np.float32)

    total_evaluated = len(preds_arr)
    logger.info(f"Total evaluated pyKT test interactions: {total_evaluated:,}")

    # 6. Global Micro Metrics & Stratification
    overall_metrics = calculate_metrics(preds_arr, targets_arr)

    # Question occurrence counts array
    q_counts_arr = np.array([q_train_counts.get(int(q), 0) for q in q_ids_arr])

    # 3-level frequency stratification
    rare_mask = q_counts_arr <= 10
    medium_mask = (q_counts_arr > 10) & (q_counts_arr <= 100)
    frequent_mask = q_counts_arr > 100

    rare_metrics = calculate_metrics(preds_arr[rare_mask], targets_arr[rare_mask]) if rare_mask.any() else {"auc": 0.0, "acc": 0.0, "num_samples": 0}
    medium_metrics = calculate_metrics(preds_arr[medium_mask], targets_arr[medium_mask]) if medium_mask.any() else {"auc": 0.0, "acc": 0.0, "num_samples": 0}
    frequent_metrics = calculate_metrics(preds_arr[frequent_mask], targets_arr[frequent_mask]) if frequent_mask.any() else {"auc": 0.0, "acc": 0.0, "num_samples": 0}

    # Modality stratification
    stem_img_mask = m_img_q_arr > 0.5
    stem_text_mask = m_img_q_arr <= 0.5
    sol_img_mask = m_img_s_arr > 0.5
    sol_text_mask = m_img_s_arr <= 0.5

    stem_img_metrics = calculate_metrics(preds_arr[stem_img_mask], targets_arr[stem_img_mask]) if stem_img_mask.any() else {"auc": 0.0, "acc": 0.0, "num_samples": 0}
    stem_text_metrics = calculate_metrics(preds_arr[stem_text_mask], targets_arr[stem_text_mask]) if stem_text_mask.any() else {"auc": 0.0, "acc": 0.0, "num_samples": 0}
    sol_img_metrics = calculate_metrics(preds_arr[sol_img_mask], targets_arr[sol_img_mask]) if sol_img_mask.any() else {"auc": 0.0, "acc": 0.0, "num_samples": 0}
    sol_text_metrics = calculate_metrics(preds_arr[sol_text_mask], targets_arr[sol_text_mask]) if sol_text_mask.any() else {"auc": 0.0, "acc": 0.0, "num_samples": 0}

    stratified = {
        "frequency": {
            "rare_le_10": rare_metrics,
            "medium_11_to_100": medium_metrics,
            "frequent_gt_100": frequent_metrics,
        },
        "stem_modality": {
            "with_image": stem_img_metrics,
            "text_only": stem_text_metrics,
        },
        "solution_modality": {
            "with_image": sol_img_metrics,
            "text_only": sol_text_metrics,
        },
    }

    logger.info(
        f"--- Overall pyKT Test Metrics ---\n"
        f"  Global Micro ROC-AUC: {overall_metrics['auc']:.5f}\n"
        f"  Global Micro Accuracy: {overall_metrics['acc']:.5f}\n"
        f"  Sample Count: {overall_metrics['num_samples']:,}"
    )

    logger.info(
        f"--- 3-Tier Question Frequency Stratification ---\n"
        f"  Rare (N_q <= 10):        AUC = {rare_metrics['auc']:.5f} (n={rare_metrics['num_samples']:,})\n"
        f"  Medium (11 <= N_q <= 100): AUC = {medium_metrics['auc']:.5f} (n={medium_metrics['num_samples']:,})\n"
        f"  Frequent (N_q > 100):    AUC = {frequent_metrics['auc']:.5f} (n={frequent_metrics['num_samples']:,})"
    )

    logger.info(
        f"--- Multimodal Modality Stratification ---\n"
        f"  Stem with Image:     AUC = {stem_img_metrics['auc']:.5f} (n={stem_img_metrics['num_samples']:,})\n"
        f"  Stem Text-Only:      AUC = {stem_text_metrics['auc']:.5f} (n={stem_text_metrics['num_samples']:,})\n"
        f"  Solution with Image: AUC = {sol_img_metrics['auc']:.5f} (n={sol_img_metrics['num_samples']:,})\n"
        f"  Solution Text-Only:  AUC = {sol_text_metrics['auc']:.5f} (n={sol_text_metrics['num_samples']:,})"
    )

    # 7. Psychometric Correlation Analysis & Gate Intensity Stats
    test_q_attempts = Counter(q_ids_arr.tolist())
    test_q_errors = Counter(q_ids_arr[targets_arr == 0].tolist())
    test_empirical_d = {}
    for q_idx, att in test_q_attempts.items():
        if att >= 5:
            test_empirical_d[q_idx] = test_q_errors[q_idx] / float(att)

    cache = model.cached_item_properties
    model_mu_prob = cache["mu_q"].squeeze(-1).cpu().numpy()
    model_tilde_mu = cache["tilde_mu"].squeeze(-1).cpu().numpy()
    model_mu_concept = cache["mu_concept"].squeeze(-1).cpu().numpy()
    model_delta_stem = cache["delta_mu_stem"].squeeze(-1).cpu().numpy()
    model_delta_sol = cache["delta_mu_sol"].squeeze(-1).cpu().numpy()

    common_qs = sorted(list(test_empirical_d.keys()))

    def compute_corr(model_vals, target_vals):
        if len(model_vals) > 10 and np.std(model_vals) > 1e-7 and np.std(target_vals) > 1e-7:
            r, _ = stats.pearsonr(model_vals, target_vals)
            rho, _ = stats.spearmanr(model_vals, target_vals)
            return float(r), float(rho)
        return 0.0, 0.0

    if len(common_qs) > 10:
        emp_d_vals = np.array([test_empirical_d[q] for q in common_qs])
        corr_prob_r, corr_prob_rho = compute_corr([model_mu_prob[q] for q in common_qs], emp_d_vals)
        corr_tilde_r, corr_tilde_rho = compute_corr([model_tilde_mu[q] for q in common_qs], emp_d_vals)
        corr_concept_r, corr_concept_rho = compute_corr([model_mu_concept[q] for q in common_qs], emp_d_vals)
        corr_stem_r, corr_stem_rho = compute_corr([model_delta_stem[q] for q in common_qs], emp_d_vals)
        corr_sol_r, corr_sol_rho = compute_corr([model_delta_sol[q] for q in common_qs], emp_d_vals)
    else:
        corr_prob_r, corr_prob_rho = 0.0, 0.0
        corr_tilde_r, corr_tilde_rho = 0.0, 0.0
        corr_concept_r, corr_concept_rho = 0.0, 0.0
        corr_stem_r, corr_stem_rho = 0.0, 0.0
        corr_sol_r, corr_sol_rho = 0.0, 0.0

    # Gate intensity statistics on items with images
    bank_masks = model.bank_modality_masks.cpu().numpy()
    has_img_q = bank_masks[:, 0] > 0.5
    has_img_s = bank_masks[:, 2] > 0.5
    lambda_q_all = cache["lambda_q"].cpu().numpy()
    lambda_s_all = cache["lambda_s"].cpu().numpy()

    mean_lambda_q = float(lambda_q_all[has_img_q].mean()) if has_img_q.any() else 0.0
    mean_lambda_s = float(lambda_s_all[has_img_s].mean()) if has_img_s.any() else 0.0

    logger.info(
        f"--- Psychometric Difficulty Calibration (Tested on {len(common_qs)} questions with >=5 test attempts) ---\n"
        f"  Total Prob Difficulty (mu_q):     Pearson r = {corr_prob_r:+.4f} | Spearman rho = {corr_prob_rho:+.4f}\n"
        f"  Total Logit Difficulty (tilde_mu): Pearson r = {corr_tilde_r:+.4f} | Spearman rho = {corr_tilde_rho:+.4f}\n"
        f"  Concept Difficulty (mu_concept):  Pearson r = {corr_concept_r:+.4f} | Spearman rho = {corr_concept_rho:+.4f}\n"
        f"  Stem Difficulty (delta_mu_stem):  Pearson r = {corr_stem_r:+.4f} | Spearman rho = {corr_stem_rho:+.4f}\n"
        f"  Solution Difficulty (delta_mu_sol): Pearson r = {corr_sol_r:+.4f} | Spearman rho = {corr_sol_rho:+.4f}"
    )

    logger.info(
        f"--- Intrinsic Visual Gate Intensity Statistics ---\n"
        f"  Stem Gate Mean (lambda_q, m_img^q=1):     {mean_lambda_q:.4f}\n"
        f"  Solution Gate Mean (lambda_s, m_img^s=1): {mean_lambda_s:.4f}"
    )

    # 8. Explainability Extraction & Markdown Report
    explainer = ACDv2ExplainabilityEngine(model=model, device=device)
    faithfulness_res = explainer.verify_mathematical_faithfulness(test_loader, max_samples=2000)
    logger.info(
        f"--- 100% Decision Faithfulness Verification ---\n"
        f"  Checked Samples: {faithfulness_res['total_checked']:,}\n"
        f"  Consistent Decisions: {faithfulness_res['consistent_count']:,} ({faithfulness_res['faithfulness_percentage']:.2f}%)\n"
        f"  Contradictions: {faithfulness_res['contradictions']}"
    )

    # Generate sample case study explanation
    case_study = explainer.explain_interaction(
        batch=first_batch_sample,
        sample_idx=0,
        target_step=-1,
    )
    if args.explain_output_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.explain_output_md)), exist_ok=True)
        with open(args.explain_output_md, "w", encoding="utf-8") as f:
            f.write(case_study["markdown_report"])
        logger.info(f"Saved 4-tier explainability report to: {args.explain_output_md}")

    # Output full report
    full_report = {
        "overall": overall_metrics,
        "stratified": stratified,
        "psychometrics": {
            "mu_prob": {"pearson_r": corr_prob_r, "spearman_rho": corr_prob_rho},
            "tilde_mu": {"pearson_r": corr_tilde_r, "spearman_rho": corr_tilde_rho},
            "mu_concept": {"pearson_r": corr_concept_r, "spearman_rho": corr_concept_rho},
            "delta_mu_stem": {"pearson_r": corr_stem_r, "spearman_rho": corr_stem_rho},
            "delta_mu_sol": {"pearson_r": corr_sol_r, "spearman_rho": corr_sol_rho},
            "evaluated_questions_count": len(common_qs),
        },
        "gate_statistics": {
            "mean_lambda_stem": mean_lambda_q,
            "mean_lambda_sol": mean_lambda_s,
        },
        "faithfulness": faithfulness_res,
        "checkpoint": args.checkpoint,
        "val_fold": args.val_fold,
    }

    if args.output_report:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_report)), exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2)
        logger.info(f"Saved full JSON evaluation report to: {args.output_report}")

    model.clear_item_cache()
    logger.info("Evaluation finished successfully.")


if __name__ == "__main__":
    main()

