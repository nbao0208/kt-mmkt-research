"""
Comprehensive Evaluation and Synthesis for MG-MMKT and Ablation Models on XES3G5M.

Executes:
1. 5-Fold Cross-Validation evaluation of Full MG-MMKT on the test set.
2. 5-Fold Ensemble inference (soft voting / probability averaging).
3. Stratified evaluation of Ablation models (Ablation 2: w/o Solution, Ablation 3: w/o Visual, Ablation 4: Naive Concat).
4. First-order Explainability extraction (Modality Attribution, Mann-Whitney U tests, Attention Sink, Historical Evidence).
5. Comprehensive JSON serialization for Markdown reporting.

Usage:
    conda run -n kt-research-env python scripts/run_comprehensive_evaluation.py
"""

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
from src.models.mmkt.mg_mmkt import MGMMKT
from src.evaluation.metrics import calculate_metrics, multimodal_stratified_evaluation
from src.evaluation.explainability import FirstOrderExplainer


def evaluate_single_checkpoint(
    ckpt_path: str,
    test_loader: Any,
    item_bank: Any,
    device: torch.device,
    q_counts: Dict[int, int],
    collect_evidence: bool = False,
    rare_threshold: int = 10,
    logger: Any = None,
) -> Dict[str, Any]:
    """
    Evaluate a single model checkpoint on the test set.
    """
    if logger:
        logger.info(f"Evaluating checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    saved_cfg = ckpt.get("model_cfg", {})

    model = MGMMKT(
        num_questions=saved_cfg.get("num_questions", 7652),
        num_kcs=saved_cfg.get("num_kcs", 1175),
        d_raw_txt=saved_cfg.get("d_raw_txt", 768),
        d_raw_img=saved_cfg.get("d_raw_img", 768),
        d_model=saved_cfg.get("d_model", 128),
        num_heads=saved_cfg.get("num_heads", 4),
        max_seq_len=saved_cfg.get("max_seq_len", 200),
        dropout=saved_cfg.get("dropout", 0.20),
        use_solution=saved_cfg.get("use_solution", True),
        use_visual=saved_cfg.get("use_visual", True),
        use_learnable_imputation=saved_cfg.get("use_learnable_imputation", True),
        use_gated_fusion=saved_cfg.get("use_gated_fusion", True),
        fusion_mode=saved_cfg.get("fusion_mode", "gated"),
        static_item_bank=item_bank,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds = []
    all_targets = []
    all_q_ids = []
    all_m_img_q = []
    all_prev_m_img_s = []
    all_alpha_q = []
    all_beta_s = []

    sample_evidence = []
    max_evidence_samples = 5
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

            # Shift solution image flags and beta_s to align with history tau = t - 1
            prev_m_img_s = torch.zeros_like(mod_masks[:, :, 2])
            prev_m_img_s[:, 1:] = mod_masks[:, :-1, 2]

            prev_beta_s = torch.zeros_like(beta_s)
            prev_beta_s[:, 1:] = beta_s[:, :-1]

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

            if collect_evidence and len(sample_evidence) < max_evidence_samples:
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

    all_preds_np = np.array(all_preds, dtype=np.float64)
    all_targets_np = np.array(all_targets, dtype=np.int64)
    all_q_ids_np = np.array(all_q_ids, dtype=np.int64)
    all_m_img_q_np = np.array(all_m_img_q, dtype=np.float32)
    all_prev_m_img_s_np = np.array(all_prev_m_img_s, dtype=np.float32)
    all_alpha_q_np = np.array(all_alpha_q, dtype=np.float32)
    all_beta_s_np = np.array(all_beta_s, dtype=np.float32)

    stratified = multimodal_stratified_evaluation(
        preds=all_preds_np,
        targets=all_targets_np,
        q_ids=all_q_ids_np,
        m_img_q=all_m_img_q_np,
        prev_m_img_s=all_prev_m_img_s_np,
        train_question_counts=q_counts,
        rare_threshold=rare_threshold,
    )

    modality_report = explainer.analyze_modality_attribution(
        all_alpha_q=all_alpha_q_np,
        all_m_img_q=all_m_img_q_np,
    )
    solution_report = explainer.analyze_solution_attribution(
        all_beta_s=all_beta_s_np,
        all_m_img_s=all_prev_m_img_s_np,
    )

    return {
        "preds": all_preds_np,
        "targets": all_targets_np,
        "q_ids": all_q_ids_np,
        "m_img_q": all_m_img_q_np,
        "prev_m_img_s": all_prev_m_img_s_np,
        "stratified_metrics": stratified,
        "modality_attribution": modality_report,
        "solution_attribution": solution_report,
        "sample_evidence": sample_evidence,
        "model_config": saved_cfg,
    }


def main():
    logger = get_logger("comprehensive_eval")
    logger.info("=== Starting Comprehensive Evaluation of MG-MMKT on XES3G5M ===")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    )
    logger.info(f"Compute Device: {device}")

    dataset_cfg = load_yaml_config("configs/dataset/xes3g5m.yaml")
    data_dir = dataset_cfg.get("data_dir", "data/processed/XES3G5M")

    # Load dataloaders
    _, _, test_loader, item_bank = get_xes3g5m_dataloaders(
        data_dir=data_dir,
        val_fold=0,
        batch_size=256,
        test_batch_size=256,
        p_drop=0.0,
        test_stride=100,
        num_workers=0,
    )

    # Compute Question Frequencies from Training Sequences (Folds 0-4)
    logger.info("Computing global training question frequencies...")
    q_counts = Counter()
    for f_idx in range(5):
        seq_path = os.path.join(data_dir, f"sequences_fold_{f_idx}.pt")
        if os.path.exists(seq_path):
            f_data = torch.load(seq_path, map_location="cpu")
            q_seq_t = f_data["q_seq"]
            seq_mask_t = f_data["seq_mask"]
            valid_qs = q_seq_t[seq_mask_t].numpy().tolist()
            q_counts.update(valid_qs)

    logger.info(f"Total training interactions: {sum(q_counts.values()):,}, unique questions: {len(q_counts):,}")

    # 1. Evaluate All 5 Folds of Full MG-MMKT
    full_fold_results = []
    full_fold_preds = []
    targets_ref = None
    q_ids_ref = None
    m_img_q_ref = None
    prev_m_img_s_ref = None

    for f in range(5):
        ckpt_path = f"outputs/checkpoints/mg_mmkt_fold{f}/best_model.pt"
        if not os.path.exists(ckpt_path):
            logger.error(f"Checkpoint not found: {ckpt_path}")
            continue

        res = evaluate_single_checkpoint(
            ckpt_path=ckpt_path,
            test_loader=test_loader,
            item_bank=item_bank,
            device=device,
            q_counts=dict(q_counts),
            collect_evidence=(f == 0),
            logger=logger,
        )
        full_fold_preds.append(res["preds"])
        if targets_ref is None:
            targets_ref = res["targets"]
            q_ids_ref = res["q_ids"]
            m_img_q_ref = res["m_img_q"]
            prev_m_img_s_ref = res["prev_m_img_s"]

        # Store serializable metrics
        full_fold_results.append({
            "fold": f,
            "stratified_metrics": res["stratified_metrics"],
            "modality_attribution": res["modality_attribution"],
            "solution_attribution": res["solution_attribution"],
            "sample_evidence": res["sample_evidence"],
        })
        logger.info(f"Fold {f}: Test AUC = {res['stratified_metrics']['overall']['auc']:.4f}, ACC = {res['stratified_metrics']['overall']['acc']:.4f}")

    # Compute 5-Fold Ensemble
    logger.info("Computing 5-Fold Ensemble Predictions...")
    ensemble_preds = np.mean(np.vstack(full_fold_preds), axis=0)
    ensemble_stratified = multimodal_stratified_evaluation(
        preds=ensemble_preds,
        targets=targets_ref,
        q_ids=q_ids_ref,
        m_img_q=m_img_q_ref,
        prev_m_img_s=prev_m_img_s_ref,
        train_question_counts=dict(q_counts),
        rare_threshold=10,
    )
    logger.info(f"==> 5-Fold Ensemble: Test AUC = {ensemble_stratified['overall']['auc']:.4f}, ACC = {ensemble_stratified['overall']['acc']:.4f}")

    # 2. Evaluate Ablation Models on Fold 0
    ablation_results = {}
    ablation_targets = {
        "ablation2_wo_solution": "outputs/checkpoints/mg_mmkt_ablation2/mg_mmkt_fold0/best_model.pt",
        "ablation3_wo_visual": "outputs/checkpoints/mg_mmkt_ablation3/mg_mmkt_fold0/best_model.pt",
        "ablation4_naive_concat": "outputs/checkpoints/mg_mmkt_ablation4/mg_mmkt_fold0/best_model.pt",
    }

    for abl_key, ckpt_path in ablation_targets.items():
        if os.path.exists(ckpt_path):
            logger.info(f"Evaluating ablation preset {abl_key}...")
            res = evaluate_single_checkpoint(
                ckpt_path=ckpt_path,
                test_loader=test_loader,
                item_bank=item_bank,
                device=device,
                q_counts=dict(q_counts),
                collect_evidence=False,
                logger=logger,
            )
            ablation_results[abl_key] = {
                "checkpoint": ckpt_path,
                "stratified_metrics": res["stratified_metrics"],
                "modality_attribution": res["modality_attribution"],
                "solution_attribution": res["solution_attribution"],
            }
            logger.info(f"{abl_key}: Test AUC = {res['stratified_metrics']['overall']['auc']:.4f}, ACC = {res['stratified_metrics']['overall']['acc']:.4f}")

    # 3. Aggregate 5-Fold Cross Validation Statistics
    def compute_stratum_stats(fold_list, stratum):
        aucs = [f["stratified_metrics"][stratum]["auc"] for f in fold_list]
        accs = [f["stratified_metrics"][stratum]["acc"] for f in fold_list]
        return {
            "mean_auc": float(np.mean(aucs)),
            "std_auc": float(np.std(aucs)),
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)),
            "num_samples": fold_list[0]["stratified_metrics"][stratum]["num_samples"],
        }

    strata_keys = [
        "overall",
        "questions_with_image",
        "questions_text_only",
        "prior_step_had_solution_image",
        "prior_step_solution_text_only",
        "rare_questions",
        "frequent_questions",
    ]

    full_model_aggregate_strata = {k: compute_stratum_stats(full_fold_results, k) for k in strata_keys}

    # Aggregate Modality Attribution stats across 5 folds
    def compute_attribution_stats(fold_list):
        alpha_id_with = [f["modality_attribution"]["with_image"]["mean_alpha_id"] for f in fold_list]
        alpha_txt_with = [f["modality_attribution"]["with_image"]["mean_alpha_txt"] for f in fold_list]
        alpha_img_with = [f["modality_attribution"]["with_image"]["mean_alpha_img"] for f in fold_list]

        alpha_id_without = [f["modality_attribution"]["text_only"]["mean_alpha_id"] for f in fold_list]
        alpha_txt_without = [f["modality_attribution"]["text_only"]["mean_alpha_txt"] for f in fold_list]
        alpha_img_without = [f["modality_attribution"]["text_only"]["mean_alpha_img"] for f in fold_list]

        beta_txt_with = [f["solution_attribution"]["with_image"]["mean_beta_txt"] for f in fold_list]
        beta_img_with = [f["solution_attribution"]["with_image"]["mean_beta_img"] for f in fold_list]

        beta_txt_without = [f["solution_attribution"]["text_only"]["mean_beta_txt"] for f in fold_list]
        beta_img_without = [f["solution_attribution"]["text_only"]["mean_beta_img"] for f in fold_list]

        return {
            "question_with_image": {
                "alpha_id": f"{np.mean(alpha_id_with):.3f} ± {np.std(alpha_id_with):.3f}",
                "alpha_txt": f"{np.mean(alpha_txt_with):.3f} ± {np.std(alpha_txt_with):.3f}",
                "alpha_img": f"{np.mean(alpha_img_with):.3f} ± {np.std(alpha_img_with):.3f}",
            },
            "question_text_only": {
                "alpha_id": f"{np.mean(alpha_id_without):.3f} ± {np.std(alpha_id_without):.3f}",
                "alpha_txt": f"{np.mean(alpha_txt_without):.3f} ± {np.std(alpha_txt_without):.3f}",
                "alpha_img": f"{np.mean(alpha_img_without):.3f} ± {np.std(alpha_img_without):.3f}",
            },
            "solution_with_image": {
                "beta_txt": f"{np.mean(beta_txt_with):.3f} ± {np.std(beta_txt_with):.3f}",
                "beta_img": f"{np.mean(beta_img_with):.3f} ± {np.std(beta_img_with):.3f}",
            },
            "solution_text_only": {
                "beta_txt": f"{np.mean(beta_txt_without):.3f} ± {np.std(beta_txt_without):.3f}",
                "beta_img": f"{np.mean(beta_img_without):.3f} ± {np.std(beta_img_without):.3f}",
            },
        }

    modality_stats = compute_attribution_stats(full_fold_results)

    # 4. Load Pre-computed 5-Fold Summaries for all Models
    def load_summary(path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    summaries_5fold = {
        "full_model": load_summary("outputs/reports/full_model/5fold_summary.json"),
        "ablation2_wo_solution": load_summary("outputs/reports/mg_mmkt_ablation2/5fold_summary.json"),
        "ablation3_wo_visual": load_summary("outputs/reports/mg_mmkt_ablation3/5fold_summary.json"),
        "ablation4_naive_concat": load_summary("outputs/reports/mg_mmkt_ablation4/5fold_summary.json"),
    }

    # 5. Compile Final Comprehensive Evaluation Report
    comprehensive_report = {
        "dataset": "XES3G5M",
        "total_test_interactions": len(targets_ref),
        "num_folds": 5,
        "full_model_5fold_results": full_fold_results,
        "full_model_aggregate_strata": full_model_aggregate_strata,
        "full_model_ensemble": ensemble_stratified,
        "full_model_modality_attribution_stats": modality_stats,
        "ablation_fold0_comparison": ablation_results,
        "summaries_5fold": summaries_5fold,
        "qualitative_evidence_samples": full_fold_results[0]["sample_evidence"],
    }

    output_file = "outputs/reports/comprehensive_test_evaluation.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(comprehensive_report, f, indent=2)

    logger.info(f"Successfully compiled comprehensive report to {output_file}")


if __name__ == "__main__":
    main()

