"""
Evaluation Metrics for Multimodal Knowledge Tracing.

Implements standard pyKT metrics (Global Micro ROC-AUC and Accuracy)
along with Multimodal Stratified Evaluation across:
1. Question image presence (Geometry / Visual vs Text-only questions).
2. Prior solution image presence (Visual post-response feedback vs Text-only feedback).
3. Question frequency stratification (Rare / Cold-start vs Frequent questions).
"""

from typing import Dict, Any, Optional, Union, List
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


def calculate_metrics(
    preds: Union[np.ndarray, List[float]],
    targets: Union[np.ndarray, List[int]],
) -> Dict[str, float]:
    """
    Calculate global micro ROC-AUC and Accuracy at standard threshold 0.5.

    Args:
        preds: Array or list of predicted probabilities.
        targets: Array or list of binary target responses.

    Returns:
        Dictionary with 'auc', 'acc', and 'num_samples'.
    """
    preds_arr = np.asarray(preds, dtype=np.float64)
    targets_arr = np.asarray(targets, dtype=np.int64)

    if len(targets_arr) == 0:
        return {"auc": 0.0, "acc": 0.0, "num_samples": 0}

    # Verify both classes exist for AUC calculation
    unique_classes = np.unique(targets_arr)
    if len(unique_classes) < 2:
        auc = 0.5  # Degenerate case when all targets are identical
    else:
        auc = float(roc_auc_score(targets_arr, preds_arr))

    binary_preds = (preds_arr >= 0.5).astype(np.int64)
    acc = float(accuracy_score(targets_arr, binary_preds))

    return {
        "auc": auc,
        "acc": acc,
        "num_samples": int(len(targets_arr)),
    }


def multimodal_stratified_evaluation(
    preds: np.ndarray,
    targets: np.ndarray,
    q_ids: np.ndarray,
    m_img_q: np.ndarray,
    prev_m_img_s: Optional[np.ndarray] = None,
    train_question_counts: Optional[Dict[int, int]] = None,
    rare_threshold: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Perform fine-grained stratified evaluation across visual modality presence and question frequency.

    Args:
        preds: Flattened array of evaluated predictions.
        targets: Flattened array of evaluated target labels.
        q_ids: Question indices corresponding to evaluated predictions.
        m_img_q: Binary flags indicating if evaluated question has an image.
        prev_m_img_s: Optional binary flags indicating if previous step had a solution image.
        train_question_counts: Optional dictionary mapping q_id to its occurrence count in train set.
        rare_threshold: Frequency threshold for rare / cold-start classification (default: 10).

    Returns:
        Nested dictionary of stratified metrics.
    """
    results = {}

    # Overall Global Metrics
    results["overall"] = calculate_metrics(preds, targets)

    # 1. Stratification by Question Image Presence
    img_mask = (m_img_q == 1.0)
    txt_only_mask = (m_img_q == 0.0)

    results["questions_with_image"] = calculate_metrics(preds[img_mask], targets[img_mask])
    results["questions_text_only"] = calculate_metrics(preds[txt_only_mask], targets[txt_only_mask])

    # 2. Stratification by Prior Solution Image Presence (if available)
    if prev_m_img_s is not None:
        sol_img_mask = (prev_m_img_s == 1.0)
        sol_txt_mask = (prev_m_img_s == 0.0)
        results["prior_step_had_solution_image"] = calculate_metrics(
            preds[sol_img_mask], targets[sol_img_mask]
        )
        results["prior_step_solution_text_only"] = calculate_metrics(
            preds[sol_txt_mask], targets[sol_txt_mask]
        )

    # 3. Stratification by Question Frequency (Rare / Cold-Start vs Frequent)
    if train_question_counts is not None:
        is_rare = np.array([train_question_counts.get(int(q), 0) < rare_threshold for q in q_ids])
        is_frequent = ~is_rare

        results["rare_questions"] = calculate_metrics(preds[is_rare], targets[is_rare])
        results["frequent_questions"] = calculate_metrics(preds[is_frequent], targets[is_frequent])

    return results
