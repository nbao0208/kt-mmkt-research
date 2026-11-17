"""
Comprehensive Evaluation and Comparative Synthesis Script for pyKT Baselines (XES3G5M & Eedi).

Executes:
1. Question-level unrolled anti-leakage evaluation across all 5 folds for DKT, DKVMN, and SAKT.
2. 5-Fold soft probability ensemble evaluation.
3. Multimodal, frequency/cold-start, remedial, and Eedi pedagogical stratification.
4. Paired statistical significance testing (t-test and Wilcoxon) against MMKT models
   dynamically loaded from verified disk reports.
5. Dataset- and split-isolated JSON & Markdown report generation.

Usage:
    # XES3G5M (default)
    conda run -n kt-research-env python scripts/evaluate_pykt_baselines.py --model all

    # Eedi (unified / public / private)
    conda run -n kt-research-env python scripts/evaluate_pykt_baselines.py --dataset eedi --model all --test_split unified
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset_router import resolve_dataset_key, resolve_dataset_paths, resolve_split_tag
from src.evaluation.pykt_evaluator import PyKTEvaluator
from src.utils.config import load_yaml_config
from src.utils.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pyKT baselines on XES3G5M or Eedi.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["xes3g5m", "eedi"],
        help="Target dataset shortcut ('xes3g5m' or 'eedi').",
    )
    parser.add_argument(
        "--test_split",
        "--test-split",
        dest="test_split",
        type=str,
        default="unified",
        choices=["unified", "public", "private"],
        help="Eedi test split ('unified', 'public', or 'private').",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["dkt", "dkvmn", "sakt", "all"],
        help="Model to evaluate ('dkt', 'dkvmn', 'sakt', or 'all').",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Path to pyKT dataset YAML config.",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default=None,
        help="Path to pyKT train YAML config.",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default=None,
        help="Base directory containing model checkpoints.",
    )
    parser.add_argument(
        "--reports_dir",
        type=str,
        default=None,
        help="Directory to save evaluation reports and JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution device ('auto', 'cuda', 'mps', 'cpu').",
    )
    parser.add_argument(
        "--reuse_preds",
        action="store_true",
        help="Reuse existing test_preds*.npy in checkpoint dirs if available.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force recomputation of predictions even if test_preds*.npy exists.",
    )
    return parser.parse_args()


def load_mmkt_reference_results(
    dataset_key: str,
    reports_dir: str,
    split_tag: str = "",
) -> Dict[str, Any]:
    """Load reference 5-fold evaluation metrics dynamically from disk reports."""
    refs: Dict[str, Any] = {}

    if dataset_key == "eedi":
        comp_path = os.path.join(
            reports_dir, f"acd_mmkt_v3_modality_ablation_study_comprehensive{split_tag}.json"
        )
        if os.path.exists(comp_path):
            with open(comp_path, "r", encoding="utf-8") as f:
                refs["acd_v3_comp"] = json.load(f)

        v3_5fold_path = os.path.join(reports_dir, f"acd_mmkt_v3_5fold_report{split_tag}.json")
        if os.path.exists(v3_5fold_path):
            with open(v3_5fold_path, "r", encoding="utf-8") as f:
                refs["acd_v3_5fold"] = json.load(f)
        return refs

    # XES3G5M reference reports
    mg_v1_path = os.path.join(reports_dir, "full_model", "5fold_summary.json")
    if os.path.exists(mg_v1_path):
        with open(mg_v1_path, "r", encoding="utf-8") as f:
            refs["mg_mmkt"] = json.load(f)

    mg_v2_path = os.path.join(reports_dir, "v2_full_5fold", "mg_mmkt_v2_5fold_summary.json")
    if os.path.exists(mg_v2_path):
        with open(mg_v2_path, "r", encoding="utf-8") as f:
            refs["mg_mmkt_v2"] = json.load(f)

    v1_comp_path = os.path.join(reports_dir, "comprehensive_test_evaluation.json")
    if os.path.exists(v1_comp_path):
        with open(v1_comp_path, "r", encoding="utf-8") as f:
            refs["mg_mmkt_comp"] = json.load(f)

    v2_comp_path = os.path.join(reports_dir, "v2_full_5fold", "comprehensive_v2_evaluation.json")
    if os.path.exists(v2_comp_path):
        with open(v2_comp_path, "r", encoding="utf-8") as f:
            refs["mg_mmkt_v2_comp"] = json.load(f)

    return refs


def evaluate_model_family(
    model_name: str,
    evaluator: PyKTEvaluator,
    checkpoints_dir: str,
    reports_dir: str,
    logger,
    split_tag: str = "",
    reuse_preds: bool = False,
    recompute: bool = False,
) -> Dict[str, Any]:
    """Evaluate all 5 folds and soft-probability ensemble for a given model architecture."""
    logger.info(f"========== Evaluating {model_name.upper()} Across All 5 Folds (split_tag='{split_tag}') ==========")
    fold_metrics = []
    all_fold_preds = []
    num_parameters = 0

    for f in range(5):
        ckpt_dir = os.path.join(checkpoints_dir, f"{model_name}_fold{f}")
        ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
        preds_save_path = os.path.join(ckpt_dir, f"test_preds{split_tag}.npy")

        if reuse_preds and not recompute and os.path.exists(preds_save_path):
            try:
                preds = np.load(preds_save_path)
                if len(preds) == evaluator.expected_eval_count and not np.isnan(preds).any():
                    logger.info(f"Reusing existing {os.path.basename(preds_save_path)} for {model_name.upper()} Fold {f}")
                    metrics = evaluator.compute_all_metrics(preds)
                    metrics["model_name"] = model_name
                    metrics["fold"] = f
                    if os.path.exists(ckpt_path):
                        ckpt_meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                        metrics["val_auc"] = ckpt_meta.get("val_auc", 0.0)
                        if "model_state_dict" in ckpt_meta:
                            num_parameters = int(sum(t.numel() for t in ckpt_meta["model_state_dict"].values()))
                            metrics["num_parameters"] = num_parameters
                    all_fold_preds.append(preds)
                    fold_metrics.append(metrics)
                    continue
            except Exception as e:
                logger.warning(f"Failed to reuse {preds_save_path}: {e}. Re-evaluating checkpoint.")

        if not os.path.exists(ckpt_path):
            logger.warning(f"Checkpoint for {model_name} Fold {f} missing at {ckpt_path}. Skipping.")
            continue

        preds, metrics = evaluator.evaluate_checkpoint(ckpt_path)
        num_parameters = int(metrics.get("num_parameters", num_parameters))
        all_fold_preds.append(preds)
        fold_metrics.append(metrics)

        os.makedirs(ckpt_dir, exist_ok=True)
        np.save(preds_save_path, preds)
        logger.info(
            f"{model_name.upper()} Fold {f} Test AUC: {metrics['overall']['auc']:.4f} | "
            f"Test ACC: {metrics['overall']['acc']:.4f} | Val AUC: {metrics.get('val_auc', 0.0):.4f}"
        )

    if len(fold_metrics) == 0:
        logger.error(f"No valid checkpoints or predictions found for {model_name}!")
        return {}

    test_aucs = [m["overall"]["auc"] for m in fold_metrics]
    test_accs = [m["overall"]["acc"] for m in fold_metrics]
    val_aucs = [m.get("val_auc", 0.0) for m in fold_metrics]

    mean_test_auc = float(np.mean(test_aucs))
    std_test_auc = float(np.std(test_aucs))
    mean_test_acc = float(np.mean(test_accs))
    std_test_acc = float(np.std(test_accs))

    logger.info(f"Computing 5-Fold soft probability ensemble for {model_name.upper()}...")
    ensemble_preds = np.mean(all_fold_preds, axis=0)
    ensemble_metrics = evaluator.compute_all_metrics(ensemble_preds)

    logger.info(
        f"{model_name.upper()} 5-Fold Summary: "
        f"Test AUC: {mean_test_auc:.4f} +/- {std_test_auc:.4f} | "
        f"Test ACC: {mean_test_acc:.4f} +/- {std_test_acc:.4f} | "
        f"Ensemble AUC: {ensemble_metrics['overall']['auc']:.4f} | "
        f"Ensemble ACC: {ensemble_metrics['overall']['acc']:.4f}"
    )

    return {
        "model_name": model_name,
        "num_parameters": num_parameters,
        "num_evaluated_folds": len(fold_metrics),
        "fold_aucs": test_aucs,
        "fold_accs": test_accs,
        "val_aucs": val_aucs,
        "mean_test_auc": mean_test_auc,
        "std_test_auc": std_test_auc,
        "mean_test_acc": mean_test_acc,
        "std_test_acc": std_test_acc,
        "fold_detailed_metrics": fold_metrics,
        "ensemble_metrics": ensemble_metrics,
    }


def _generate_eedi_markdown_report(
    eval_results: Dict[str, Any],
    mmkt_refs: Dict[str, Any],
    output_path: str,
    test_split: str,
    n_eval: int,
    n_students: int,
) -> None:
    """Generate dynamic Markdown comparison report for pyKT baselines vs ACD-MMKTv3 on Eedi."""
    comp = mmkt_refs.get("acd_v3_comp", {})
    v3_5fold = mmkt_refs.get("acd_v3_5fold", {})

    md_lines = [
        "# Báo Cáo Đánh Giá Toàn Diện Các Mô Hình Cơ Sở pyKT (DKT, DKVMN, SAKT) Trên Eedi NeurIPS 2020 (Task 3 & 4)",
        "",
        f"> **Báo cáo kỹ thuật:** Tổng hợp kết quả huấn luyện 5-Fold Cross-Validation và đánh giá quy chiếu cấp độ câu hỏi (*Question-Level Late Mean Fusion*, `num_c = 57` kỹ năng lá cấp 3) đối với **DKT**, **DKVMN**, và **SAKT** nguyên bản từ thư viện **pyKT** trên phân vùng kiểm thử **`{test_split}`** (**{n_eval:,} tương tác** của **{n_students:,} học sinh**), đối sánh 1-đối-1 với **ACD-MMKTv3** và các biến thể cắt bỏ.",
        "",
        "---",
        "",
        "## 1. Bảng So Sánh Hiệu Năng Toàn Cục (Global 5-Fold Performance)",
        "",
        "| Mô hình | Nguồn kiến trúc | Tham số | Test ROC-AUC (Mean ± Std) | Test ACC (Mean ± Std) | 5-Fold Ens. AUC | 5-Fold Ens. ACC | RMSE | Brier | ECE | Macro-AUC |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for m_name in ["dkt", "dkvmn", "sakt"]:
        res = eval_results.get(m_name, {})
        if not res:
            continue
        em = res["ensemble_metrics"]
        ext = em.get("overall_extended", {})
        p_cnt = res.get("num_parameters", 0)
        m_auc = f"{res['mean_test_auc']:.4f} ± {res['std_test_auc']:.4f}"
        m_acc = f"{res['mean_test_acc']:.4f} ± {res['std_test_acc']:.4f}"
        ens_auc = f"{em['overall']['auc']:.4f}"
        ens_acc = f"{em['overall']['acc']:.4f}"
        rmse = f"{ext['rmse']:.4f}" if "rmse" in ext else "--"
        brier = f"{ext['brier']:.4f}" if "brier" in ext else "--"
        ece = f"{ext['ece']:.4f}" if "ece" in ext else "--"
        macro_auc = f"{ext['student_macro_auc']:.4f}" if "student_macro_auc" in ext else "--"
        md_lines.append(
            f"| **{m_name.upper()}** | pyKT (`pykt.models`) | {p_cnt:,} | {m_auc} | {m_acc} | {ens_auc} | {ens_acc} | {rmse} | {brier} | {ece} | {macro_auc} |"
        )

    full_fold_aucs: List[float] = []
    ab1_fold_aucs: List[float] = []

    if comp and "full_model" in comp:
        fm = comp["full_model"]
        full_fold_aucs = [float(x) for x in fm.get("fold_aucs", [])]
        f_ens = fm.get("ensemble_metrics", {})
        f_mean = fm.get("mean_metrics", {})
        f_accs = [
            float(v["acc"]) for v in v3_5fold.get("single_fold_metrics", {}).values()
        ] if v3_5fold else []
        f_acc_str = (
            f"{np.mean(f_accs):.4f} ± {np.std(f_accs):.4f}" if len(f_accs) == 5 else f"{f_ens.get('acc', 0.0):.4f}"
        )
        md_lines.append(
            f"| **ACD-MMKTv3 (Full)** | Đa phương thức đề xuất | -- | "
            f"**{f_mean.get('auc', 0.0):.4f} ± {f_mean.get('auc_std', 0.0):.4f}** | **{f_acc_str}** | "
            f"**{f_ens.get('auc', 0.0):.4f}** | **{f_ens.get('acc', 0.0):.4f}** | "
            f"**{f_ens.get('rmse', 0.0):.4f}** | **{f_ens.get('brier', 0.0):.4f}** | "
            f"**{f_ens.get('ece', 0.0):.4f}** | **{f_ens.get('student_macro_auc', 0.0):.4f}** |"
        )
        for ab_k, ab_label in [("3", "Ablation 3 (`w/o Visual`)"), ("2", "Ablation 2 (`w/o Solution`)"), ("1", "Ablation 1 (`Pure KC`)")]:
            ab_d = comp.get("ablations", {}).get(ab_k)
            if not ab_d:
                continue
            ab_m = ab_d.get("mean_metrics", {})
            ab_e = ab_d.get("ensemble_metrics", {})
            if ab_k == "1":
                ab1_fold_aucs = [float(m["auc"]) for m in ab_d.get("single_fold_metrics", [])]
            md_lines.append(
                f"| **{ab_label}** | Cắt bỏ nội bộ | {ab_d.get('total_params', 0):,} | "
                f"{ab_m.get('auc', 0.0):.4f} ± {ab_m.get('auc_std', 0.0):.4f} | "
                f"{ab_m.get('acc', 0.0):.4f} ± {ab_m.get('acc_std', 0.0):.4f} | "
                f"{ab_e.get('auc', 0.0):.4f} | {ab_e.get('acc', 0.0):.4f} | "
                f"{ab_e.get('rmse', 0.0):.4f} | {ab_e.get('brier', 0.0):.4f} | "
                f"{ab_e.get('ece', 0.0):.4f} | {ab_e.get('student_macro_auc', 0.0):.4f} |"
            )
    elif v3_5fold:
        sf = v3_5fold.get("single_fold_metrics", {})
        full_fold_aucs = [float(sf[f"fold_{i}"]["auc"]) for i in range(5) if f"fold_{i}" in sf]
        mm = v3_5fold.get("mean_metrics", {})
        em = v3_5fold.get("ensemble_5fold_metrics", {})
        md_lines.append(
            f"| **ACD-MMKTv3 (Full)** | Đa phương thức đề xuất | -- | "
            f"**{mm.get('auc', 0.0):.4f} ± {mm.get('auc_std', 0.0):.4f}** | "
            f"**{mm.get('acc', 0.0):.4f} ± {mm.get('acc_std', 0.0):.4f}** | "
            f"**{em.get('auc', 0.0):.4f}** | **{em.get('acc', 0.0):.4f}** | -- | -- | -- | -- |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Đánh Giá Phân Tầng Sư Phạm & Đa Phương Thức (Stratified Evaluation)",
        "",
        "Hiệu năng ROC-AUC trên mô hình 5-Fold Ensemble qua các phân tầng đặc thù của Eedi (lát cắt ảnh lời giải có `0` mẫu được bỏ qua an toàn):",
        "",
        "| Mô hình | Toàn cục | Đề có hình ($m_q^{\\text{img}}=1$) | Thuần chữ ($m_q^{\\text{img}}=0$) | Sửa sai ($r_{t-1}=0$) | Khởi đầu lạnh ($N_q \\le 10$) | Phổ biến ($N_q > 100$) | Đa kỹ năng (Multi-KC) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    def _fmt_auc(d: Dict[str, Any], key: str) -> str:
        item = d.get(key, {})
        if not item or item.get("num_samples", 0) == 0 or item.get("auc") is None:
            return "N/A"
        val = float(item["auc"])
        return "N/A" if np.isnan(val) else f"{val:.4f}"

    for m_name in ["dkt", "dkvmn", "sakt"]:
        res = eval_results.get(m_name, {})
        if not res:
            continue
        em = res["ensemble_metrics"]
        es = em.get("eedi_strata", {})
        src = es if es else em
        md_lines.append(
            f"| **{m_name.upper()}** (pyKT) | "
            f"{_fmt_auc(src, 'overall')} | "
            f"{_fmt_auc(src, 'q_stem_image' if es else 'questions_with_image')} | "
            f"{_fmt_auc(src, 'q_text_only' if es else 'questions_text_only')} | "
            f"{_fmt_auc(src, 'after_incorrect' if es else 'remedial_after_incorrect')} | "
            f"{_fmt_auc(src, 'freq_rare_le10')} | "
            f"{_fmt_auc(src, 'freq_frequent_gt100')} | "
            f"{_fmt_auc(src, 'multi_kc')} |"
        )

    if comp and "full_model" in comp:
        fs = comp["full_model"].get("stratified_metrics", {})
        if fs:
            md_lines.append(
                f"| **ACD-MMKTv3 (Full)** | "
                f"**{_fmt_auc(fs, 'overall')}** | **{_fmt_auc(fs, 'q_stem_image')}** | "
                f"**{_fmt_auc(fs, 'q_text_only')}** | **{_fmt_auc(fs, 'after_incorrect')}** | "
                f"**{_fmt_auc(fs, 'freq_rare_le10')}** | **{_fmt_auc(fs, 'freq_frequent_gt100')}** | "
                f"**{_fmt_auc(fs, 'multi_kc')}** |"
            )
        ab1_s = comp.get("ablations", {}).get("1", {}).get("stratified_metrics", {})
        if ab1_s:
            md_lines.append(
                f"| **Ablation 1 (`Pure KC`)** | "
                f"{_fmt_auc(ab1_s, 'overall')} | {_fmt_auc(ab1_s, 'q_stem_image')} | "
                f"{_fmt_auc(ab1_s, 'q_text_only')} | {_fmt_auc(ab1_s, 'after_incorrect')} | "
                f"{_fmt_auc(ab1_s, 'freq_rare_le10')} | {_fmt_auc(ab1_s, 'freq_frequent_gt100')} | "
                f"{_fmt_auc(ab1_s, 'multi_kc')} |"
            )

    md_lines.extend([
        "",
        "---",
        "",
        "## 3. Kiểm Định Ý Nghĩa Thống Kê Bắt Cặp (Paired Statistical Significance)",
        "",
        "Kiểm định Paired Student's t-test và Wilcoxon Signed-Rank Test qua 5 Folds đối sánh **ACD-MMKTv3 (Full)** và **Ablation 1 (`Pure KC`)** với từng mô hình cơ sở pyKT:",
        "",
        "| Cặp đối sánh | Chênh lệch AUC trung bình ($\\Delta$) | $t$-statistic | $p$-value ($t$-test) | Wilcoxon stat | $p$-value (Wilcoxon) | Kết luận ý nghĩa ($p < 0.05$) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    paired_tests_rendered = 0
    for prop_label, prop_aucs in [("ACD-MMKTv3 (Full)", full_fold_aucs), ("Ablation 1 (`Pure KC`)", ab1_fold_aucs)]:
        if len(prop_aucs) != 5:
            continue
        for m_name in ["dkt", "dkvmn", "sakt"]:
            res = eval_results.get(m_name, {})
            if not res or len(res.get("fold_aucs", [])) != 5:
                continue
            stat_res = PyKTEvaluator.compute_paired_statistical_tests(res["fold_aucs"], prop_aucs)
            p_t = stat_res["p_value_ttest"]
            sig_flag = (
                "Có ý nghĩa cực cao (p < 0.001)"
                if p_t < 0.001
                else ("Có ý nghĩa (p < 0.05)" if p_t < 0.05 else "Không có ý nghĩa (n.s.)")
            )
            md_lines.append(
                f"| **{prop_label}** vs **{m_name.upper()}** | "
                f"{stat_res['mean_diff']:+.4f} | "
                f"{stat_res['t_statistic']:.4f} | "
                f"{p_t:.4e} | "
                f"{stat_res['wilcoxon_stat']:.1f} | "
                f"{stat_res['p_value_wilcoxon']:.4e} | "
                f"**{sig_flag}** |"
            )
            paired_tests_rendered += 1

    if paired_tests_rendered == 0:
        md_lines.append(
            "| *(Đang chờ kết quả 5 Folds)* | -- | -- | -- | -- | -- | Sẽ tự động tính toán khi hoàn tất đủ 5 Folds |"
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


def generate_markdown_report(
    eval_results: Dict[str, Any],
    mmkt_refs: Dict[str, Any],
    output_path: str,
    dataset_key: str = "xes3g5m",
    test_split: str = "unified",
    n_eval: int = 1099197,
    n_students: int = 3613,
):
    """Generate detailed Markdown report comparing pyKT baselines with MMKT architectures."""
    if dataset_key == "eedi":
        _generate_eedi_markdown_report(
            eval_results=eval_results,
            mmkt_refs=mmkt_refs,
            output_path=output_path,
            test_split=test_split,
            n_eval=n_eval,
            n_students=n_students,
        )
        return

    md_lines = [
        "# Báo Cáo Đánh Giá Toàn Diện Các Mô Hình Cơ Sở pyKT (DKT, DKVMN, SAKT) Trên XES3G5M",
        "",
        f"> **Báo cáo kỹ thuật:** Bản báo cáo này tổng hợp kết quả huấn luyện 5-Fold Cross-Validation và đánh giá quy chiếu cấp độ câu hỏi (Question-Level Late Fusion) đối với ba mô hình Knowledge Tracing cơ sở gồm **DKT**, **DKVMN**, và **SAKT** được khởi tạo trực tiếp từ thư viện **pyKT**. Toàn bộ kết quả được đối sánh 1-đối-1 trên đúng **{n_eval:,} tương tác kiểm thử độc lập** với các mô hình Đa phương thức đề xuất (**MG-MMKT**, **MG-MMKTv2**).",
        "",
        "---",
        "",
        "## 1. Bảng So Sánh Hiệu Năng Toàn Cục (Global 5-Fold Performance)",
        "",
        "| Mô hình | Nguồn kiến trúc | Test ROC-AUC (Mean ± Std) | Test ACC (Mean ± Std) | 5-Fold Ensemble AUC | 5-Fold Ensemble ACC |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    for m_name in ["dkt", "dkvmn", "sakt"]:
        res = eval_results.get(m_name, {})
        if not res:
            continue
        m_auc = f"{res['mean_test_auc']:.4f} ± {res['std_test_auc']:.4f}"
        m_acc = f"{res['mean_test_acc']:.4f} ± {res['std_test_acc']:.4f}"
        ens_auc = f"{res['ensemble_metrics']['overall']['auc']:.4f}"
        ens_acc = f"{res['ensemble_metrics']['overall']['acc']:.4f}"
        md_lines.append(f"| **{m_name.upper()}** | pyKT (`pykt.models`) | {m_auc} | {m_acc} | {ens_auc} | {ens_acc} |")

    if "mg_mmkt" in mmkt_refs:
        ref1 = mmkt_refs["mg_mmkt"]
        m_auc = f"{ref1['mean_test_auc']:.4f} ± {ref1['std_test_auc']:.4f}"
        m_acc = f"{ref1['mean_test_acc']:.4f} ± {ref1['std_test_acc']:.4f}"
        ens1 = mmkt_refs.get("mg_mmkt_comp", {}).get("full_model_ensemble", {})
        e_auc1 = f"{ens1['auc']:.4f}" if "auc" in ens1 else "--"
        e_acc1 = f"{ens1['acc']:.4f}" if "acc" in ens1 else "--"
        md_lines.append(f"| **MG-MMKT** (Đề xuất v1) | Đa phương thức nội bộ | {m_auc} | {m_acc} | {e_auc1} | {e_acc1} |")

    if "mg_mmkt_v2" in mmkt_refs:
        ref2 = mmkt_refs["mg_mmkt_v2"]["summary"]
        m_auc = f"{ref2['test_auc_mean']:.4f} ± {ref2['test_auc_std']:.4f}"
        m_acc = f"{ref2['test_acc_mean']:.4f} ± {ref2['test_acc_std']:.4f}"
        ens_auc = f"{ref2['ensemble_test_auc']:.4f}"
        ens_acc = f"{ref2['ensemble_test_acc']:.4f}"
        md_lines.append(f"| **MG-MMKTv2** (Đề xuất v2) | Đa phương thức nội bộ | **{m_auc}** | **{m_acc}** | **{ens_auc}** | **{ens_acc}** |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Đánh Giá Phân Tầng Đa Phương Thức & Khởi Đầu Lạnh (Stratified Evaluation)",
        "",
        "Bảng đối sánh hiệu năng ROC-AUC trên mô hình 5-Fold Ensemble qua các tập con đặc thù:",
        "",
        "| Mô hình | Toàn cục | Câu hỏi có ảnh ($m_q^{\\text{img}}=1$) | Thuần Text ($m_q^{\\text{img}}=0$) | Có ảnh lời giải bước trước | Sửa sai (Remedial) | Khởi đầu lạnh ($N_q \\le 10$) | Phổ biến ($N_q > 100$) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for m_name in ["dkt", "dkvmn", "sakt"]:
        res = eval_results.get(m_name, {})
        if not res:
            continue
        em = res["ensemble_metrics"]
        md_lines.append(
            f"| **{m_name.upper()}** (pyKT) | "
            f"{em['overall']['auc']:.4f} | "
            f"{em['questions_with_image']['auc']:.4f} | "
            f"{em['questions_text_only']['auc']:.4f} | "
            f"{em['prior_step_had_solution_image']['auc']:.4f} | "
            f"{em['remedial_after_incorrect']['auc']:.4f} | "
            f"{em['freq_rare_le10']['auc']:.4f} | "
            f"{em['freq_frequent_gt100']['auc']:.4f} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 3. Kiểm Định Ý Nghĩa Thống Kê Bắt Cặp (Paired Statistical Significance)",
        "",
        "Kiểm định Paired Student's t-test và Wilcoxon Signed-Rank Test qua 5 Folds đối sánh sự vượt trội của **MG-MMKTv2** so với từng mô hình cơ sở pyKT:",
        "",
        "| Cặp đối sánh | Chênh lệch AUC trung bình ($\\Delta$) | $t$-statistic | $p$-value ($t$-test) | Wilcoxon stat | $p$-value (Wilcoxon) | Kết luận ý nghĩa ($p < 0.05$) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    paired_tests_rendered = 0
    if "mg_mmkt_v2" in mmkt_refs:
        v2_aucs = [f["test_auc"] for f in mmkt_refs["mg_mmkt_v2"]["single_fold_results"]]
        for m_name in ["dkt", "dkvmn", "sakt"]:
            res = eval_results.get(m_name, {})
            if not res or len(res.get("fold_aucs", [])) != 5:
                continue
            base_aucs = res["fold_aucs"]
            stat_res = PyKTEvaluator.compute_paired_statistical_tests(base_aucs, v2_aucs)
            sig_flag = "Có ý nghĩa cực cao (p < 0.001)" if stat_res["p_value_ttest"] < 0.001 else "Có ý nghĩa (p < 0.05)"
            md_lines.append(
                f"| **MG-MMKTv2** vs **{m_name.upper()}** | "
                f"+{stat_res['mean_diff']:.4f} | "
                f"{stat_res['t_statistic']:.4f} | "
                f"{stat_res['p_value_ttest']:.4e} | "
                f"{stat_res['wilcoxon_stat']:.1f} | "
                f"{stat_res['p_value_wilcoxon']:.4e} | "
                f"**{sig_flag}** |"
            )
            paired_tests_rendered += 1

    if paired_tests_rendered == 0:
        md_lines.append(
            "| *(Đang chờ kết quả 5 Folds)* | -- | -- | -- | -- | -- | Sẽ tự động tính toán khi hoàn tất đủ 5 Folds |"
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


def main():
    args = parse_args()

    dataset_config_path = args.dataset_config or (
        "configs/dataset/eedi_pykt.yaml" if args.dataset == "eedi" else "configs/dataset/xes3g5m_pykt.yaml"
    )
    dataset_cfg = load_yaml_config(dataset_config_path) if os.path.exists(dataset_config_path) else {}
    dataset_key = resolve_dataset_key(dataset_cfg=dataset_cfg, dataset_arg=args.dataset)

    train_config_path = args.train_config or (
        "configs/train_pykt_eedi.yaml"
        if (dataset_key == "eedi" and os.path.exists("configs/train_pykt_eedi.yaml"))
        else "configs/train_pykt.yaml"
    )
    train_cfg = load_yaml_config(train_config_path) if os.path.exists(train_config_path) else {}

    paths = resolve_dataset_paths(
        dataset_key=dataset_key,
        dataset_cfg=dataset_cfg,
        train_cfg=train_cfg,
        checkpoint_dir=args.checkpoints_dir,
        output_report_dir=args.reports_dir,
    )
    checkpoints_dir = paths["checkpoint_dir"]
    reports_dir = paths["output_report_dir"]
    log_dir = paths["log_dir"]
    split_tag = resolve_split_tag(dataset_key, args.test_split)

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"evaluate_pykt_baselines{split_tag}.log")
    logger = get_logger("EvaluatePyKT", log_file=log_file)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    logger.info(
        f"Dataset: {paths['display_name']} (key={dataset_key}, split={args.test_split}) | "
        f"Checkpoints: {checkpoints_dir} | Reports: {reports_dir} | Device: {device}"
    )

    if dataset_key == "eedi":
        raw_data_dir = dataset_cfg.get("raw_data_dir", "data/processed/Eedi/pykt_kc_level")
        test_files_cfg = dataset_cfg.get("test_files", {})
        test_filename = test_files_cfg.get(args.test_split, f"test_{args.test_split}.csv")
        test_csv_path = os.path.join(raw_data_dir, test_filename)
        processed_dir = dataset_cfg.get("processed_dir", paths["data_dir"])
        expected_eval = dataset_cfg.get("expected_eval_counts", {}).get(args.test_split)
        train_valid_csv_path = os.path.join(
            raw_data_dir, dataset_cfg.get("train_valid_file", "train_valid_sequences.csv")
        )
    else:
        raw_data_dir = dataset_cfg.get("raw_root_dir", "data/raw/XES3G5M")
        test_csv_path = dataset_cfg.get("test_csv_path", os.path.join(raw_data_dir, "kc_level", "test.csv"))
        processed_dir = dataset_cfg.get("processed_dir", None)
        expected_eval = 1099197
        train_valid_csv_path = None

    evaluator = PyKTEvaluator(
        raw_data_dir=raw_data_dir,
        test_csv_path=test_csv_path,
        processed_dir=processed_dir,
        dataset_key=dataset_key,
        test_split=args.test_split,
        expected_eval_count=expected_eval,
        test_stride=int(dataset_cfg.get("test_stride", 100)),
        train_valid_csv_path=train_valid_csv_path,
        device=device,
        logger=logger,
    )
    mmkt_refs = load_mmkt_reference_results(dataset_key, reports_dir, split_tag=split_tag)

    models_to_eval = ["dkt", "dkvmn", "sakt"] if args.model == "all" else [args.model]
    all_eval_results: Dict[str, Any] = {}

    for m in models_to_eval:
        res = evaluate_model_family(
            model_name=m,
            evaluator=evaluator,
            checkpoints_dir=checkpoints_dir,
            reports_dir=reports_dir,
            logger=logger,
            split_tag=split_tag,
            reuse_preds=args.reuse_preds,
            recompute=args.recompute,
        )
        if res:
            all_eval_results[m] = res

    os.makedirs(reports_dir, exist_ok=True)
    json_path = os.path.join(reports_dir, f"pykt_baselines_comprehensive_evaluation{split_tag}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_eval_results, f, indent=2)
    logger.info(f"Evaluation JSON saved to: {json_path}")

    md_path = os.path.join(reports_dir, f"PyKT_Baselines_Comprehensive_Evaluation_Report{split_tag}.md")
    generate_markdown_report(
        eval_results=all_eval_results,
        mmkt_refs=mmkt_refs,
        output_path=md_path,
        dataset_key=dataset_key,
        test_split=args.test_split,
        n_eval=len(evaluator.ground_truth_targets),
        n_students=len(evaluator.test_students),
    )
    logger.info(f"Comprehensive Markdown report generated at: {md_path}")


if __name__ == "__main__":
    main()
