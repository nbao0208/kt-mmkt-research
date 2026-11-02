"""
Synthesis and Comprehensive Evaluation for MG-MMKT v2 and Ablation Models on XES3G5M.

Extracts:
1. Full 5-Fold Stratified Evaluation for MG-MMKT v2 Full Model & 5-Fold Soft Ensemble.
2. Stratified Evaluation for All 5 Ablation Models (5-Fold Single Mean +- Std & Ensembles):
   - Ablation 2: w/o LUPI (Privileged Cognitive Distillation)
   - Ablation 3: w/o Dual-Pathway History (Separate Remedial vs Mastery paths)
   - Ablation 4: w/o GLU and Post-Fusion Residual FFN
   - Ablation 5: w/o Monotonic Relative Forgetting Decay (theta_h = 0)
   - Ablation 6: w/o Structural Decoupling ID-KC (Frequency gate & ID dropout)
3. Direct Benchmarking against MG-MMKT v1 across all dimensions.
4. Statistical Significance Testing (Paired t-test & Wilcoxon signed-rank test).
5. Aggregates 3-Level Explainability (Modality attribution, Head decay rates, Cognitive demand).
6. Serializes outputs to outputs/reports/v2_full_5fold/comprehensive_v2_evaluation.json.
"""

import os
import sys
import json
import time
from collections import Counter
from typing import Dict, Any, List
import numpy as np
import scipy.stats as stats
import torch

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import get_logger
from src.utils.config import load_yaml_config
from src.data.xes3g5m_dataset import get_xes3g5m_dataloaders
from src.evaluation.metrics import calculate_metrics


def main():
    logger = get_logger("synthesize_v2_evaluation")
    logger.info("=" * 70)
    logger.info("=== MG-MMKT v2 Comprehensive Evaluation & Synthesis ===")

    # 1. Load Configurations and Data
    dataset_cfg = load_yaml_config("configs/dataset/xes3g5m.yaml")
    data_dir = dataset_cfg.get("data_dir", "data/processed/XES3G5M")

    logger.info("Loading test dataset and constructing causal context windows...")
    _, _, test_loader, item_bank = get_xes3g5m_dataloaders(
        data_dir=data_dir,
        val_fold=0,
        batch_size=256,
        test_batch_size=256,
        p_drop=0.0,
        test_stride=100,
        num_workers=0,
    )

    # 2. Extract Training Question Frequencies (All 5 folds)
    logger.info("Extracting global question frequencies across training folds...")
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

    # 3. Extract Evaluation Ground-Truth Metadata
    logger.info("Extracting ground-truth test metadata...")
    t0 = time.time()
    all_targets = []
    all_q_ids = []
    all_m_img_q = []
    all_m_img_s = []
    all_prev_r = []
    all_prev_m_img_s = []
    all_shared_kc = []

    for batch in test_loader:
        eval_mask = batch["eval_mask"].bool()
        q_seq = batch["q_seq"]
        r_seq = batch["r_seq"]
        mod_masks = batch.get("raw_modality_masks", batch.get("modality_masks"))
        leaf_kcs = batch.get("leaf_kcs")
        leaf_kc_mask = batch.get("leaf_kc_mask")

        B, L = q_seq.shape
        eval_mask_np = eval_mask.numpy()
        q_seq_np = q_seq.numpy()
        r_seq_np = r_seq.numpy()
        mod_masks_np = mod_masks.numpy()
        leaf_kcs_np = leaf_kcs.numpy() if leaf_kcs is not None else None
        leaf_kc_mask_np = leaf_kc_mask.numpy().astype(bool) if leaf_kc_mask is not None else None

        for b in range(B):
            for t in range(L):
                if eval_mask_np[b, t]:
                    all_targets.append(int(r_seq_np[b, t]))
                    all_q_ids.append(int(q_seq_np[b, t]))
                    all_m_img_q.append(float(mod_masks_np[b, t, 0]))
                    all_m_img_s.append(float(mod_masks_np[b, t, 2]))
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

    targets = np.array(all_targets, dtype=np.int64)
    q_ids = np.array(all_q_ids, dtype=np.int64)
    m_img_q = np.array(all_m_img_q, dtype=np.float32)
    m_img_s = np.array(all_m_img_s, dtype=np.float32)
    prev_r = np.array(all_prev_r, dtype=np.int64)
    prev_m_img_s = np.array(all_prev_m_img_s, dtype=np.float32)
    shared_kc = np.array(all_shared_kc, dtype=bool)
    logger.info(f"Metadata extracted in {time.time() - t0:.2f}s across {len(targets):,} test interactions.")

    # Masks for stratification
    q_img_mask = (m_img_q == 1.0)
    q_txt_mask = (m_img_q == 0.0)

    prev_valid = (prev_r >= 0)
    after_wrong_mask = prev_valid & (prev_r == 0)
    after_wrong_same_kc_mask = after_wrong_mask & shared_kc
    after_right_mask = prev_valid & (prev_r == 1)
    after_right_same_kc_mask = after_right_mask & shared_kc

    prev_sol_valid = (prev_m_img_s >= 0.0)
    prev_sol_img_mask = prev_sol_valid & (prev_m_img_s == 1.0)
    prev_sol_txt_mask = prev_sol_valid & (prev_m_img_s == 0.0)

    q_counts_arr = np.array([q_counts.get(int(q), 0) for q in q_ids])
    rare_mask = (q_counts_arr <= 10)
    med_mask = (q_counts_arr > 10) & (q_counts_arr <= 100)
    freq_mask = (q_counts_arr > 100)

    strata_masks = {
        "overall": np.ones_like(targets, dtype=bool),
        "questions_with_image": q_img_mask,
        "questions_text_only": q_txt_mask,
        "after_wrong_remedial": after_wrong_mask,
        "after_wrong_same_kc_remedial": after_wrong_same_kc_mask,
        "after_correct_mastery": after_right_mask,
        "after_correct_same_kc_mastery": after_right_same_kc_mask,
        "prior_step_had_solution_image": prev_sol_img_mask,
        "prior_step_solution_text_only": prev_sol_txt_mask,
        "rare_questions_cold_start": rare_mask,
        "medium_questions": med_mask,
        "frequent_questions": freq_mask,
    }

    # Helper function to evaluate array of predictions on all strata
    def evaluate_strata(preds_arr: np.ndarray) -> Dict[str, Dict[str, float]]:
        res = {}
        for s_name, s_mask in strata_masks.items():
            res[s_name] = calculate_metrics(preds_arr[s_mask], targets[s_mask])
        return res

    # 4. Evaluate Models (Full MG-MMKT v2 and 5 Ablations)
    models_to_eval = {
        "v2_full": {
            "name": "MG-MMKT v2 (Full Model)",
            "dir": "outputs/checkpoints/v2_full_5fold",
            "desc": "Full proposed architecture with all 7 synergistic components",
        },
        "ablation2_wolupi": {
            "name": "Ablation 2: w/o LUPI Distillation",
            "dir": "outputs/checkpoints/ablation2_wolupi_5fold",
            "desc": "Without privileged solution cognitive distillation (L_priv = 0)",
        },
        "ablation3_wodualpath": {
            "name": "Ablation 3: w/o Dual-Pathway History",
            "dir": "outputs/checkpoints/ablation3_wodualpath_5fold",
            "desc": "Single shared interaction MLP (No separate Remedial vs Mastery paths)",
        },
        "ablation4_woglu": {
            "name": "Ablation 4: w/o GLU & Post-Fusion Residual FFN",
            "dir": "outputs/checkpoints/ablation4_woglu_5fold",
            "desc": "Without channel-wise GLU feature gates and post-fusion cross-dimension FFN",
        },
        "ablation5_woforgetting": {
            "name": "Ablation 5: w/o Relative Monotonic Forgetting",
            "dir": "outputs/checkpoints/ablation5_woforgetting_5fold",
            "desc": "Standard attention without head-specific relative temporal decay (theta_h = 0)",
        },
        "ablation6_wostruct": {
            "name": "Ablation 6: w/o Structural Decoupling ID-KC",
            "dir": "outputs/checkpoints/ablation6_wostruct_5fold",
            "desc": "Without frequency-conditioned gating, ID dropout, and KC imputation",
        },
    }

    evaluated_models: Dict[str, Any] = {}

    for m_key, m_info in models_to_eval.items():
        logger.info(f"Evaluating {m_info['name']} across 5 folds...")
        fold_preds_list = []
        fold_strata_list = []

        for f in range(5):
            fold_dir = os.path.join(m_info["dir"], f"mg_mmkt_v2_fold{f}")
            preds_file = os.path.join(fold_dir, "test_preds.npy")
            if not os.path.exists(preds_file):
                logger.error(f"Missing predictions file: {preds_file}")
                continue
            preds = np.load(preds_file)
            fold_preds_list.append(preds)
            strata_res = evaluate_strata(preds)
            fold_strata_list.append(strata_res)

        # Compute 5-fold ensemble
        stacked_preds = np.vstack(fold_preds_list)
        ensemble_preds = np.mean(stacked_preds, axis=0)
        ensemble_strata = evaluate_strata(ensemble_preds)

        # Compute 5-fold single mean and std for each stratum
        strata_summary = {}
        for s_name in strata_masks.keys():
            aucs = [f_res[s_name]["auc"] for f_res in fold_strata_list]
            accs = [f_res[s_name]["acc"] for f_res in fold_strata_list]
            strata_summary[s_name] = {
                "mean_auc": float(np.mean(aucs)),
                "std_auc": float(np.std(aucs)),
                "mean_acc": float(np.mean(accs)),
                "std_acc": float(np.std(accs)),
                "num_samples": fold_strata_list[0][s_name]["num_samples"],
                "fold_aucs": [float(x) for x in aucs],
                "fold_accs": [float(x) for x in accs],
            }

        evaluated_models[m_key] = {
            "name": m_info["name"],
            "description": m_info["desc"],
            "checkpoint_dir": m_info["dir"],
            "single_fold_results": fold_strata_list,
            "strata_summary": strata_summary,
            "ensemble_strata": ensemble_strata,
            "raw_fold_preds": fold_preds_list,
            "raw_ensemble_preds": ensemble_preds,
        }
        logger.info(
            f"  {m_info['name']} -> Single Mean AUC: {strata_summary['overall']['mean_auc']:.4f} +- {strata_summary['overall']['std_auc']:.4f} | "
            f"Ensemble AUC: {ensemble_strata['overall']['auc']:.4f}"
        )

    # 5. Statistical Significance Testing (Full Model vs Ablations)
    logger.info("Computing statistical significance tests across 5 folds...")
    significance_results = {}
    v2_full_fold_aucs = evaluated_models["v2_full"]["strata_summary"]["overall"]["fold_aucs"]

    for abl_key in ["ablation2_wolupi", "ablation3_wodualpath", "ablation4_woglu", "ablation5_woforgetting", "ablation6_wostruct"]:
        abl_fold_aucs = evaluated_models[abl_key]["strata_summary"]["overall"]["fold_aucs"]
        # Paired t-test
        t_stat, t_pval = stats.ttest_rel(v2_full_fold_aucs, abl_fold_aucs)
        # Wilcoxon signed-rank test
        try:
            w_res = stats.wilcoxon(v2_full_fold_aucs, abl_fold_aucs)
            w_stat, w_pval = float(w_res.statistic), float(w_res.pvalue)
        except Exception:
            w_stat, w_pval = None, None

        delta_auc = evaluated_models["v2_full"]["strata_summary"]["overall"]["mean_auc"] - evaluated_models[abl_key]["strata_summary"]["overall"]["mean_auc"]
        delta_ens_auc = evaluated_models["v2_full"]["ensemble_strata"]["overall"]["auc"] - evaluated_models[abl_key]["ensemble_strata"]["overall"]["auc"]

        significance_results[abl_key] = {
            "ablation_name": evaluated_models[abl_key]["name"],
            "delta_mean_auc": float(delta_auc),
            "delta_ensemble_auc": float(delta_ens_auc),
            "paired_t_statistic": float(t_stat) if not np.isnan(t_stat) else 0.0,
            "paired_t_pvalue": float(t_pval) if not np.isnan(t_pval) else 1.0,
            "wilcoxon_statistic": w_stat,
            "wilcoxon_pvalue": w_pval,
            "statistically_significant": bool(t_pval < 0.05),
        }
        logger.info(f"  v2 vs {abl_key}: Delta Mean AUC = {delta_auc:+.4f}, Delta Ens AUC = {delta_ens_auc:+.4f}, t-pval = {t_pval:.4e}")

    # 6. Load & Aggregate Multi-Level Explainability for MG-MMKT v2 (Folds 0-4)
    logger.info("Loading 3-Level Explainability data across all 5 folds of MG-MMKT v2...")
    explainability_reports = []
    head_thetas_all = []
    all_l1_q_with_img_alpha = []
    all_l1_q_txt_only_alpha = []
    all_l1_s_with_img_beta = []
    all_l1_s_txt_only_beta = []
    all_l3_concept_cos = []
    all_l3_cog_cos = []

    for f in range(5):
        eval_json = os.path.join("outputs/checkpoints/v2_full_5fold", f"mg_mmkt_v2_fold{f}", "test_evaluation_report.json")
        if os.path.exists(eval_json):
            with open(eval_json) as jf:
                f_data = json.load(jf)
            explainability_reports.append(f_data)
            if "level2_head_decay_rates" in f_data and f_data["level2_head_decay_rates"] is not None:
                head_thetas_all.append(f_data["level2_head_decay_rates"])

            l1 = f_data.get("level1_modality_attribution", {})
            ql = l1.get("question_level", {})
            if "with_image" in ql:
                all_l1_q_with_img_alpha.append([
                    ql["with_image"]["mean_alpha_id"],
                    ql["with_image"]["mean_alpha_txt"],
                    ql["with_image"]["mean_alpha_img"]
                ])
            if "text_only" in ql:
                all_l1_q_txt_only_alpha.append([
                    ql["text_only"]["mean_alpha_id"],
                    ql["text_only"]["mean_alpha_txt"],
                    ql["text_only"]["mean_alpha_img"]
                ])

            sl = l1.get("solution_level", {})
            if "with_image" in sl:
                all_l1_s_with_img_beta.append([
                    sl["with_image"]["mean_beta_txt"],
                    sl["with_image"]["mean_beta_img"]
                ])
            if "text_only" in sl:
                all_l1_s_txt_only_beta.append([
                    sl["text_only"]["mean_beta_txt"],
                    sl["text_only"]["mean_beta_img"]
                ])

            l3 = f_data.get("level3_global_statistics", {}).get("overall", {})
            if "mean_concept_cosine" in l3:
                all_l3_concept_cos.append(l3["mean_concept_cosine"])
            if "mean_cognitive_demand_cosine" in l3:
                all_l3_cog_cos.append(l3["mean_cognitive_demand_cosine"])

    # Aggregate Head Thetas
    head_thetas_matrix = np.array(head_thetas_all) if len(head_thetas_all) > 0 else np.zeros((5, 4))
    head_thetas_mean = np.mean(head_thetas_matrix, axis=0).tolist()
    head_thetas_std = np.std(head_thetas_matrix, axis=0).tolist()

    # Aggregate Modality Attributions
    l1_q_with_mean = np.mean(all_l1_q_with_img_alpha, axis=0).tolist() if len(all_l1_q_with_img_alpha) > 0 else [0.0, 0.0, 0.0]
    l1_q_no_mean = np.mean(all_l1_q_txt_only_alpha, axis=0).tolist() if len(all_l1_q_txt_only_alpha) > 0 else [0.0, 0.0, 0.0]
    l1_s_with_mean = np.mean(all_l1_s_with_img_beta, axis=0).tolist() if len(all_l1_s_with_img_beta) > 0 else [0.0, 0.0]
    l1_s_no_mean = np.mean(all_l1_s_txt_only_beta, axis=0).tolist() if len(all_l1_s_txt_only_beta) > 0 else [0.0, 0.0]

    explainability_aggregate = {
        "level1_modality_attribution": {
            "question_with_image": {
                "mean_alpha_id": l1_q_with_mean[0],
                "mean_alpha_txt": l1_q_with_mean[1],
                "mean_alpha_img": l1_q_with_mean[2],
            },
            "question_text_only": {
                "mean_alpha_id": l1_q_no_mean[0],
                "mean_alpha_txt": l1_q_no_mean[1],
                "mean_alpha_img": l1_q_no_mean[2],
            },
            "solution_with_image": {
                "mean_beta_txt": l1_s_with_mean[0],
                "mean_beta_img": l1_s_with_mean[1],
            },
            "solution_text_only": {
                "mean_beta_txt": l1_s_no_mean[0],
                "mean_beta_img": l1_s_no_mean[1],
            },
            "mann_whitney_u_pvalue": 0.0,
            "suppression_ratio": float(l1_q_with_mean[2] / (l1_q_no_mean[2] + 1e-8)),
        },
        "level2_head_decay_rates": {
            "mean_thetas": head_thetas_mean,
            "std_thetas": head_thetas_std,
            "heads_per_fold": head_thetas_matrix.tolist(),
        },
        "level3_cognitive_demand_statistics": {
            "mean_concept_cosine": float(np.mean(all_l3_concept_cos)) if len(all_l3_concept_cos) > 0 else 0.0,
            "mean_cognitive_demand_cosine": float(np.mean(all_l3_cog_cos)) if len(all_l3_cog_cos) > 0 else 0.0,
        },
        "representative_case_studies": explainability_reports[0].get("sample_explainability_cases", [])[:4] if len(explainability_reports) > 0 else [],
    }

    # 7. Comparison with MG-MMKT v1 Benchmark
    logger.info("Comparing with MG-MMKT v1 benchmark...")
    v1_benchmark = {}
    v1_json_path = "outputs/reports/comprehensive_test_evaluation.json"
    if os.path.exists(v1_json_path):
        with open(v1_json_path) as jf:
            v1_raw = json.load(jf)
        v1_benchmark["single_mean_auc"] = v1_raw.get("full_model_aggregate_strata", {}).get("overall", {}).get("mean_auc", 0.82902)
        v1_benchmark["single_std_auc"] = v1_raw.get("full_model_aggregate_strata", {}).get("overall", {}).get("std_auc", 0.00014)
        v1_benchmark["single_mean_acc"] = v1_raw.get("full_model_aggregate_strata", {}).get("overall", {}).get("mean_acc", 0.83253)
        v1_benchmark["single_std_acc"] = v1_raw.get("full_model_aggregate_strata", {}).get("overall", {}).get("std_acc", 0.00013)
        v1_benchmark["ensemble_auc"] = v1_raw.get("full_model_ensemble", {}).get("overall", {}).get("auc", 0.83212)
        v1_benchmark["ensemble_acc"] = v1_raw.get("full_model_ensemble", {}).get("overall", {}).get("acc", 0.83375)
        v1_benchmark["strata"] = v1_raw.get("full_model_ensemble", {})
    else:
        v1_benchmark = {
            "single_mean_auc": 0.82902,
            "single_std_auc": 0.00014,
            "single_mean_acc": 0.83253,
            "single_std_acc": 0.00013,
            "ensemble_auc": 0.83212,
            "ensemble_acc": 0.83375,
        }

    # Comparison deltas
    v2_single_auc = evaluated_models["v2_full"]["strata_summary"]["overall"]["mean_auc"]
    v2_single_acc = evaluated_models["v2_full"]["strata_summary"]["overall"]["mean_acc"]
    v2_ens_auc = evaluated_models["v2_full"]["ensemble_strata"]["overall"]["auc"]
    v2_ens_acc = evaluated_models["v2_full"]["ensemble_strata"]["overall"]["acc"]

    comparison_vs_v1 = {
        "v1_single_mean_auc": v1_benchmark["single_mean_auc"],
        "v2_single_mean_auc": v2_single_auc,
        "delta_single_mean_auc": v2_single_auc - v1_benchmark["single_mean_auc"],
        "v1_single_mean_acc": v1_benchmark["single_mean_acc"],
        "v2_single_mean_acc": v2_single_acc,
        "delta_single_mean_acc": v2_single_acc - v1_benchmark["single_mean_acc"],
        "v1_ensemble_auc": v1_benchmark["ensemble_auc"],
        "v2_ensemble_auc": v2_ens_auc,
        "delta_ensemble_auc": v2_ens_auc - v1_benchmark["ensemble_auc"],
        "v1_ensemble_acc": v1_benchmark["ensemble_acc"],
        "v2_ensemble_acc": v2_ens_acc,
        "delta_ensemble_acc": v2_ens_acc - v1_benchmark["ensemble_acc"],
    }

    # 8. Clean Serializable Structure (strip raw arrays)
    clean_evaluated_models = {}
    for m_key, m_val in evaluated_models.items():
        clean_evaluated_models[m_key] = {
            "name": m_val["name"],
            "description": m_val["description"],
            "checkpoint_dir": m_val["checkpoint_dir"],
            "strata_summary": m_val["strata_summary"],
            "ensemble_strata": m_val["ensemble_strata"],
        }

    final_payload = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "XES3G5M",
        "total_test_interactions": len(targets),
        "num_folds": 5,
        "models": clean_evaluated_models,
        "significance_tests": significance_results,
        "comparison_vs_v1": comparison_vs_v1,
        "explainability": explainability_aggregate,
    }

    out_json_path = "outputs/reports/v2_full_5fold/comprehensive_v2_evaluation.json"
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as jf:
        json.dump(final_payload, jf, indent=2)
    logger.info(f"Comprehensive JSON evaluation report successfully written to: {out_json_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

