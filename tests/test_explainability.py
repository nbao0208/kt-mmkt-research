"""
Unit tests for First-Order Explainability Extractor and Statistical Attribution.
"""

import pytest
import numpy as np
from src.evaluation.explainability import FirstOrderExplainer


def test_first_order_explainer_3_branches():
    """Verify modality attribution analysis with visual modality (3 branches)."""
    explainer = FirstOrderExplainer()

    N = 100
    np.random.seed(42)
    # 40 with image, 60 text only
    m_img_q = np.zeros(N)
    m_img_q[:40] = 1.0

    # Alpha Q [N, 3]
    alpha_q = np.random.uniform(0.1, 1.0, (N, 3))
    # Give higher visual attention to questions with image
    alpha_q[:40, 2] += 2.0
    alpha_q = alpha_q / alpha_q.sum(axis=-1, keepdims=True)

    report = explainer.analyze_modality_attribution(alpha_q, m_img_q)

    assert report["num_questions_with_image"] == 40
    assert report["num_questions_text_only"] == 60
    assert report["has_visual_branch"] is True
    assert "statistical_test" in report
    assert report["with_image"]["mean_alpha_img"] > report["text_only"]["mean_alpha_img"]
    assert report["statistical_test"]["significant_at_005"] is True


def test_first_order_explainer_2_branches_no_visual():
    """Verify modality attribution analysis when visual modality is disabled (2 branches)."""
    explainer = FirstOrderExplainer()

    N = 50
    m_img_q = np.zeros(N)
    alpha_q = np.random.uniform(0.1, 1.0, (N, 2))
    alpha_q = alpha_q / alpha_q.sum(axis=-1, keepdims=True)

    # Must NOT raise IndexError!
    report = explainer.analyze_modality_attribution(alpha_q, m_img_q)

    assert report["has_visual_branch"] is False
    assert report["with_image"]["mean_alpha_img"] == 0.0
    assert report["text_only"]["mean_alpha_img"] == 0.0


def test_solution_attribution():
    """Verify solution modality attribution analysis (text vs image)."""
    explainer = FirstOrderExplainer()

    N = 60
    m_img_s = np.zeros(N)
    m_img_s[:20] = 1.0

    beta_s = np.random.uniform(0.2, 0.8, (N, 2))
    beta_s[:20, 1] += 1.5
    beta_s = beta_s / beta_s.sum(axis=-1, keepdims=True)

    report = explainer.analyze_solution_attribution(beta_s, m_img_s)

    assert report["num_solutions_with_image"] == 20
    assert report["num_solutions_text_only"] == 40
    assert report["with_image"]["mean_beta_img"] > report["text_only"]["mean_beta_img"]


def test_historical_evidence_with_skills_and_offsets():
    """Verify historical evidence retrieval contains skill IDs, prior sink weight, and correct offsets."""
    explainer = FirstOrderExplainer()

    L = 20
    target_t = 10
    attn = np.random.uniform(0.01, 0.5, (L, L))
    # Sink weight
    attn[target_t, 0] = 0.15
    # Historical steps
    attn[target_t, 5] = 0.85
    # Normalize row
    attn[target_t] = attn[target_t] / attn[target_t].sum()

    q_seq = np.arange(100, 100 + L)
    r_seq = np.ones(L, dtype=np.int64)
    mod_masks = np.ones((L, 3))
    leaf_kcs = np.full((L, 6), 55)
    leaf_kc_mask = np.zeros((L, 6), dtype=bool)
    leaf_kc_mask[:, :2] = True  # each question has 2 KCs

    evidence = explainer.extract_top_k_historical_evidence(
        attn_weights=attn,
        step_t=target_t,
        q_seq=q_seq,
        r_seq=r_seq,
        modality_masks=mod_masks,
        leaf_kcs=leaf_kcs,
        leaf_kc_mask=leaf_kc_mask,
        context_start=150,
        k=3,
    )

    assert len(evidence) == 3
    top1 = evidence[0]
    # Column 5 in attention matrix corresponds to tau = 4 in local window
    assert top1["history_step"] == 4
    assert top1["actual_history_step"] == 154
    assert top1["actual_target_step"] == 160
    assert top1["prior_sink_weight"] == pytest.approx(float(attn[target_t, 0]))
    assert top1["skill_ids"] == [55, 55]
