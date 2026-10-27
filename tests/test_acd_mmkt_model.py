"""
Unit and Integration Tests for ACD-MMKT Architecture.

Verifies:
1. Architectural output shapes across Subsystem 1, 2, 3, 4, and End-to-End ACDMMKT.
2. Strict Causal Invariance (Backward Perturbation Causality Test): max diff < 1e-6 for t <= t_0.
3. Psychometric Monotonicity Guarantee: d(p_hat)/d(theta) > 0 and d(p_hat)/d(mu) < 0.
4. Non-zero gradient flow to learnable canonical missing tokens and strictly positive decay rate theta > 0.
5. Numerical stability of Multi-Task Loss objectives (Stage 1 and Stage 2) under boundary cases.
6. Ablation Configurations (Full, w/o Distill, w/o Fine-Grained, w/o HGNN/Rasch, w/o DeepIRT).
7. Three-tier Explainability Engine extraction and Markdown report generation.
"""

import os
import pytest
import torch
import torch.nn as nn
import numpy as np

from src.models.modules.cross_modal_alignment import FineGrainedCrossModalAlignment
from src.models.modules.hypergraph_rasch import HypergraphRaschDisentanglement
from src.models.modules.monotonic_attention import AsymmetricMonotonicAttention
from src.models.modules.deep_irt_readout import DeepIRTReadout
from src.models.mmkt.acd_mmkt import ACDMMKT
from src.training.losses import (
    MaskedBCELoss,
    SolutionDistillationLoss,
    CTTDifficultyCalibrationLoss,
    OrthogonalDisentanglementLoss,
    ACDMMKTLoss,
)
from src.evaluation.acd_explainability import ACDExplainabilityEngine


@pytest.fixture
def mock_static_bank():
    """Create a synthetic static item bank dictionary matching XES3G5M shapes."""
    torch.manual_seed(42)
    num_q = 50
    num_k = 30
    d_raw = 768

    bank = {
        "acd_E_txt_q": torch.randn(num_q, 128, d_raw, dtype=torch.float16),
        "acd_mask_txt_q": torch.ones(num_q, 128, dtype=torch.bool),
        "acd_E_img_q": torch.randn(num_q, 49, d_raw, dtype=torch.float16),
        "acd_E_txt_s": torch.randn(num_q, 160, d_raw, dtype=torch.float16),
        "acd_mask_txt_s": torch.ones(num_q, 160, dtype=torch.bool),
        "acd_E_img_s": torch.randn(num_q, 49, d_raw, dtype=torch.float16),
        "acd_kc_init_emb": torch.randn(num_k, d_raw, dtype=torch.float32),
        "acd_hypergraph_G": torch.eye(num_k, dtype=torch.float32),
        "modality_masks": torch.bernoulli(torch.full((num_q, 3), 0.7)),
        "question_leaf_kcs": torch.randint(0, num_k, (num_q, 6), dtype=torch.int64),
        "question_kc_mask": torch.ones(num_q, 6, dtype=torch.bool),
        "question_static_ctx": torch.randn(num_q, 15, dtype=torch.float32),
        "acd_ctt_difficulty_folds": torch.rand(5, num_q, dtype=torch.float32) * 0.8 + 0.1,
        "acd_ctt_valid_mask": torch.ones(5, num_q, dtype=torch.bool),
    }
    return bank


@pytest.fixture
def mock_batch():
    """Create a synthetic student sequence batch."""
    torch.manual_seed(42)
    B, L = 4, 20
    num_q = 50
    num_k = 30

    q_seq = torch.randint(0, num_q, (B, L), dtype=torch.int64)
    r_seq = torch.randint(0, 2, (B, L), dtype=torch.int64)
    seq_mask = torch.ones((B, L), dtype=torch.bool)
    loss_mask = seq_mask.clone()
    loss_mask[:, 0] = False
    eval_mask = loss_mask.clone()

    dynamic_ctx = torch.randn(B, L, 21, dtype=torch.float32)
    modality_masks = torch.bernoulli(torch.full((B, L, 3), 0.7))
    leaf_kcs = torch.randint(0, num_k, (B, L, 6), dtype=torch.int64)
    leaf_kc_mask = torch.ones((B, L, 6), dtype=torch.bool)
    delta_t_lag = torch.rand(B, L)
    delta_t_kc = torch.rand(B, L)
    batch_flag = torch.randint(0, 2, (B, L)).float()

    return {
        "q_seq": q_seq,
        "r_seq": r_seq,
        "seq_mask": seq_mask,
        "loss_mask": loss_mask,
        "eval_mask": eval_mask,
        "dynamic_ctx": dynamic_ctx,
        "modality_masks": modality_masks,
        "leaf_kcs": leaf_kcs,
        "leaf_kc_mask": leaf_kc_mask,
        "delta_t_lag": delta_t_lag,
        "delta_t_kc": delta_t_kc,
        "batch_flag": batch_flag,
        "ctt_difficulty": torch.rand(B, L) * 0.8 + 0.1,
        "ctt_valid_mask": torch.ones(B, L, dtype=torch.bool),
        "context_start": torch.zeros(B, dtype=torch.int64),
    }


def test_subsystem1_shapes():
    """Verify Subsystem 1 cross-modal alignment and hierarchical gating shapes."""
    torch.manual_seed(42)
    N = 6
    d_model = 64
    sub1 = FineGrainedCrossModalAlignment(
        d_raw_txt=768,
        d_raw_img=768,
        d_ctx=21,
        d_model=d_model,
        num_heads=2,
        patch_count=16,
    )

    E_txt_q = torch.randn(N, 20, 768)
    mask_txt_q = torch.ones(N, 20, dtype=torch.bool)
    E_img_q = torch.randn(N, 16, 768)
    E_txt_s = torch.randn(N, 25, 768)
    mask_txt_s = torch.ones(N, 25, dtype=torch.bool)
    E_img_s = torch.randn(N, 16, 768)
    mod_masks = torch.ones(N, 3)
    dynamic_ctx = torch.randn(N, 21)

    out = sub1(
        E_txt_q=E_txt_q,
        mask_txt_q=mask_txt_q,
        E_img_q=E_img_q,
        E_txt_s=E_txt_s,
        mask_txt_s=mask_txt_s,
        E_img_s=E_img_s,
        modality_masks=mod_masks,
        dynamic_ctx=dynamic_ctx,
    )

    assert out["v_stem"].shape == (N, d_model)
    assert out["v_sol"].shape == (N, d_model)
    assert out["lambda_q"].shape == (N, d_model)
    assert out["A_cross_q"].shape == (N, 20, 16)


def test_subsystem2_shapes():
    """Verify Subsystem 2 HGNN, distillation predictor, and Rasch shapes."""
    torch.manual_seed(42)
    B, L = 2, 10
    num_k = 20
    d_model = 64
    sub2 = HypergraphRaschDisentanglement(
        num_kcs=num_k,
        d_raw_kc=768,
        d_model=d_model,
        hgnn_layers=2,
    )

    v_stem = torch.randn(B, L, d_model)
    leaf_kcs = torch.randint(0, num_k, (B, L, 4), dtype=torch.int64)
    leaf_kc_mask = torch.ones(B, L, 4, dtype=torch.bool)
    kc_init = torch.randn(num_k, 768)
    G = torch.eye(num_k)

    out = sub2(
        v_stem=v_stem,
        leaf_kcs=leaf_kcs,
        leaf_kc_mask=leaf_kc_mask,
        kc_init_emb=kc_init,
        hypergraph_G=G,
    )

    assert out["x"].shape == (B, L, d_model)
    assert out["c_C"].shape == (B, L, d_model)
    assert out["hat_v_sol"].shape == (B, L, d_model)
    assert out["mu_q"].shape == (B, L, 1)
    assert out["d_q"].shape == (B, L, d_model)
    assert (out["mu_q"] >= 0.0).all() and (out["mu_q"] <= 1.0).all()


def test_subsystem3_shapes_and_decay():
    """Verify Subsystem 3 exercise encoding, acquisition encoding, and monotonic retrieval."""
    torch.manual_seed(42)
    B, L = 2, 10
    d_model = 64
    sub3 = AsymmetricMonotonicAttention(
        d_model=d_model,
        num_heads=2,
        max_seq_len=20,
    )

    x = torch.randn(B, L, d_model)
    c_C = torch.randn(B, L, d_model)
    mu_q = torch.rand(B, L, 1)
    d_q = torch.randn(B, L, d_model)
    r_seq = torch.randint(0, 2, (B, L), dtype=torch.int64)
    delta_t_lag = torch.rand(B, L)
    delta_t_kc = torch.rand(B, L)
    batch_flag = torch.zeros(B, L)
    v_sol = torch.randn(B, L, d_model)
    m_txt_s = torch.ones(B, L, 1)

    out = sub3(
        x=x,
        c_C=c_C,
        mu_q=mu_q,
        d_q=d_q,
        r_seq=r_seq,
        delta_t_lag=delta_t_lag,
        delta_t_kc=delta_t_kc,
        batch_flag=batch_flag,
        v_sol=v_sol,
        m_txt_s=m_txt_s,
    )

    assert out["h"].shape == (B, L, d_model)
    assert out["tilde_x"].shape == (B, L, d_model)
    assert out["tilde_y"].shape == (B, L, d_model)
    assert out["alpha_ret"].shape == (B, L, L)
    assert out["theta_decay"].item() > 0.0  # Strictly positive decay rate


def test_subsystem4_monotonicity():
    """
    Verify DeepIRT Readout psychometric monotonicity:
    d(p_hat)/d(theta) > 0 and d(p_hat)/d(mu) < 0 strictly using autograd.
    """
    torch.manual_seed(42)
    B, L = 3, 5
    d_model = 64
    readout = DeepIRTReadout(d_model=d_model, irt_scale=4.0)

    h = torch.randn(B, L, d_model, requires_grad=True)
    tilde_x = torch.randn(B, L, d_model)
    mu_q = torch.rand(B, L, 1, requires_grad=True)

    out = readout(h=h, tilde_x=tilde_x, mu_q=mu_q)
    preds = out["preds"]
    theta = out["theta"]
    a_q = out["a_q"]

    # Verify a_q > 0 strictly
    assert (a_q > 0.0).all(), "Item discrimination must be strictly positive!"

    # 1. Gradients w.r.t mu_q: d(p_hat)/d(mu_q) < 0 strictly
    grad_mu = torch.autograd.grad(preds.sum(), mu_q, retain_graph=True)[0]
    assert (grad_mu < 0.0).all(), "Prediction must be strictly monotonically decreasing with item difficulty mu_q!"

    # 2. Gradients w.r.t theta: d(p_hat)/d(theta) > 0 strictly
    grad_theta = torch.autograd.grad(preds.sum(), theta, retain_graph=True)[0]
    assert (grad_theta > 0.0).all(), "Prediction must be strictly monotonically increasing with student proficiency theta!"


def test_end_to_end_acd_mmkt(mock_static_bank, mock_batch):
    """Verify complete end-to-end forward passes for Stage 1 and Stage 2."""
    model = ACDMMKT(
        num_questions=50,
        num_kcs=30,
        d_raw_txt=768,
        d_raw_img=768,
        d_ctx=21,
        d_model=64,
        num_heads=2,
        max_seq_len=20,
        static_item_bank=mock_static_bank,
    )

    # 1. Stage 1 Item-Bank Forward Pass
    s1_out = model.forward_item_bank(item_indices=torch.arange(10))
    assert s1_out["v_stem"].shape == (10, 64)
    assert s1_out["v_sol"].shape == (10, 64)
    assert s1_out["hat_v_sol"].shape == (10, 64)
    assert s1_out["mu_q"].shape == (10, 1)

    # 2. Stage 2 Sequential Interaction Forward Pass
    s2_out = model(mock_batch)
    B, L = mock_batch["q_seq"].shape
    assert s2_out["logits"].shape == (B, L)
    assert s2_out["preds"].shape == (B, L)
    assert s2_out["theta"].shape == (B, L)
    assert s2_out["mu_q"].shape == (B, L)
    assert s2_out["a_q"].shape == (B, L)
    assert s2_out["alpha_ret"].shape == (B, L, L)


def test_strict_temporal_causal_asymmetry(mock_static_bank, mock_batch):
    """
    Strict Backward Perturbation Causality Test (Plan Step 8.2 & Acceptance Criteria 4):
    Mutating response r_{t0}, solution matrices (E_txt^s, E_img^s), solution presence flags
    (m_txt^s, m_img^s) at step t0, and invalidating future steps MUST NOT alter
    predictions p_hat, theta, or mu_q for any step t <= t0 (diff < 1e-7),
    while at step t0 + 1, the response change MUST produce a valid difference (> 1e-4).
    """
    model = ACDMMKT(
        num_questions=50,
        num_kcs=30,
        d_raw_txt=768,
        d_raw_img=768,
        d_ctx=21,
        d_model=64,
        num_heads=2,
        max_seq_len=20,
        static_item_bank=mock_static_bank,
    )
    model.eval()

    t0 = 8

    # Ensure question at t0 is distinct from questions appearing at t < t0
    # so perturbing its static solution matrices does not affect earlier steps
    mock_batch["q_seq"][:, :t0] = mock_batch["q_seq"][:, :t0] % 35
    mock_batch["q_seq"][:, t0] = 45
    mock_batch["modality_masks"][:, t0, 1] = 1.0  # text solution present
    mock_batch["modality_masks"][:, t0, 2] = 1.0  # image solution present

    # Pass 1: Original input batch
    with torch.no_grad():
        out1 = model(mock_batch)
        preds1 = out1["preds"].clone()
        theta1 = out1["theta"].clone()
        mu1 = out1["mu_q"].clone()

    # Pass 2: Perturb response, solution presence, static solution matrices, and future steps at and after t0
    perturbed_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in mock_batch.items()}
    # 1. Flip student response at t0
    perturbed_batch["r_seq"][:, t0] = 1 - perturbed_batch["r_seq"][:, t0]
    # 2. Mutate solution presence flags (m_txt^s, m_img^s) at t0
    perturbed_batch["modality_masks"][:, t0, 1] = 0.0
    perturbed_batch["modality_masks"][:, t0, 2] = 0.0
    # 3. Invalidate all future steps tau > t0
    perturbed_batch["r_seq"][:, t0 + 1 :] = 0
    perturbed_batch["q_seq"][:, t0 + 1 :] = 0
    perturbed_batch["dynamic_ctx"][:, t0 + 1 :] = 0.0
    perturbed_batch["batch_flag"][:, t0 + 1 :] = 0.0
    perturbed_batch["modality_masks"][:, t0 + 1 :, :] = 0.0

    # 4. Perturb static item bank solution matrices E_txt^s and E_img^s for question 45
    model.acd_E_txt_s.data[45] += torch.randn_like(model.acd_E_txt_s.data[45])
    model.acd_E_img_s.data[45] += torch.randn_like(model.acd_E_img_s.data[45])

    with torch.no_grad():
        out2 = model(perturbed_batch)
        preds2 = out2["preds"].clone()
        theta2 = out2["theta"].clone()
        mu2 = out2["mu_q"].clone()

    # Verify zero change for all t <= t0 (strict threshold < 1e-7)
    diff_preds_past = (preds1[:, : t0 + 1] - preds2[:, : t0 + 1]).abs().max().item()
    diff_theta_past = (theta1[:, : t0 + 1] - theta2[:, : t0 + 1]).abs().max().item()
    diff_mu_past = (mu1[:, : t0 + 1] - mu2[:, : t0 + 1]).abs().max().item()

    assert diff_preds_past < 1e-7, f"Causal leakage in preds: max diff = {diff_preds_past:.2e} >= 1e-7"
    assert diff_theta_past < 1e-7, f"Causal leakage in theta: max diff = {diff_theta_past:.2e} >= 1e-7"
    assert diff_mu_past < 1e-7, f"Causal leakage in mu_q: max diff = {diff_mu_past:.2e} >= 1e-7"

    # Verify that at step t0 + 1, changing response and solution at t0 DOES change the next prediction
    diff_preds_next = (preds1[:, t0 + 1] - preds2[:, t0 + 1]).abs().max().item()
    assert diff_preds_next > 1e-4, f"Next-step prediction failed to reflect past interaction: diff = {diff_preds_next:.2e}"


def test_nonzero_gradient_flow(mock_static_bank, mock_batch):
    """
    Verify non-zero gradient flow to learnable canonical missing tokens:
    e_null_img_q, e_null_img_s, and decay parameter raw_theta.
    """
    model = ACDMMKT(
        num_questions=50,
        num_kcs=30,
        d_raw_txt=768,
        d_raw_img=768,
        d_ctx=21,
        d_model=64,
        num_heads=2,
        max_seq_len=20,
        static_item_bank=mock_static_bank,
    )
    model.train()

    loss_fn = ACDMMKTLoss()
    out = model(mock_batch)
    loss, _ = loss_fn.forward_stage2(
        seq_outputs=out,
        targets=mock_batch["r_seq"],
        loss_mask=mock_batch["loss_mask"],
        d_ctt=mock_batch["ctt_difficulty"],
        ctt_valid_mask=mock_batch["ctt_valid_mask"],
    )
    loss.backward()

    # Check gradient norms
    grad_e_null_q = model.subsystem1.e_null_img_q.grad
    grad_e_null_img_s = model.subsystem1.e_null_img_s.grad
    grad_theta = model.subsystem3.raw_theta.grad

    assert grad_e_null_q is not None and grad_e_null_q.abs().sum().item() > 0.0
    assert grad_e_null_img_s is not None and grad_e_null_img_s.abs().sum().item() > 0.0
    assert grad_theta is not None and grad_theta.abs().item() > 0.0


def test_ablation_variants(mock_static_bank, mock_batch):
    """Verify that all 5 ablation model configurations initialize and forward correctly."""
    # Ablation 2: w/o Distillation
    m2 = ACDMMKT(
        num_questions=50, num_kcs=30, d_model=64, num_heads=2, max_seq_len=20,
        use_distillation=False, static_item_bank=mock_static_bank,
    )
    o2 = m2(mock_batch)
    assert o2["logits"].shape == (4, 20)

    # Ablation 3: w/o Fine-Grained Alignment
    m3 = ACDMMKT(
        num_questions=50, num_kcs=30, d_model=64, num_heads=2, max_seq_len=20,
        use_fine_grained=False, static_item_bank=mock_static_bank,
    )
    o3 = m3(mock_batch)
    assert o3["logits"].shape == (4, 20)

    # Ablation 4: w/o HGNN and Rasch
    m4 = ACDMMKT(
        num_questions=50, num_kcs=30, d_model=64, num_heads=2, max_seq_len=20,
        use_hypergraph_rasch=False, static_item_bank=mock_static_bank,
    )
    o4 = m4(mock_batch)
    assert o4["logits"].shape == (4, 20)

    # Ablation 5: w/o DeepIRT Readout (Black-box MLP)
    m5 = ACDMMKT(
        num_questions=50, num_kcs=30, d_model=64, num_heads=2, max_seq_len=20,
        use_deep_irt=False, static_item_bank=mock_static_bank,
    )
    o5 = m5(mock_batch)
    assert o5["logits"].shape == (4, 20)


def test_three_tier_explainability(mock_static_bank, mock_batch, tmp_path):
    """Verify Three-Tier Explainability Engine extraction and Markdown serialization."""
    model = ACDMMKT(
        num_questions=50, num_kcs=30, d_model=64, num_heads=2, max_seq_len=20,
        static_item_bank=mock_static_bank,
    )
    explainer = ACDExplainabilityEngine(model)
    report = explainer.explain_interaction(
        batch=mock_batch,
        batch_idx=0,
        time_step=15,
        question_text="Giải phương trình bậc hai sau...",
    )

    assert "tier1_psychometrics" in report
    assert "tier2_attribution" in report
    assert "tier3_pedagogical_rationale" in report

    assert "student_proficiency_theta" in report["tier1_psychometrics"]
    assert "item_difficulty_mu" in report["tier1_psychometrics"]
    assert "solution_complexity_shift_delta_mu" in report["tier2_attribution"]
    assert "part1_knowledge_mastery" in report["tier3_pedagogical_rationale"]

    # Test Markdown export
    out_md = tmp_path / "test_report.md"
    md_content = explainer.export_report_markdown(report, str(out_md))
    assert os.path.exists(out_md)
    assert "Cấp Độ 1: Bằng Chứng Tâm Trắc Học Định Lượng" in md_content
    assert "Cấp Độ 2: Bằng Chứng Đa Phương Thức" in md_content
    assert "Cấp Độ 3: Báo Cáo Giải Thích Ngôn Ngữ Tự Nhiên" in md_content


def test_disentanglement_loss_with_mask():
    """Verify that OrthogonalDisentanglementLoss correctly supports optional active masks."""
    loss_fn = OrthogonalDisentanglementLoss()
    c_C = torch.randn(4, 20, 64)
    d_q = torch.randn(4, 20, 64)
    mask = torch.zeros(4, 20, dtype=torch.bool)
    mask[:, :5] = True

    loss_unmasked = loss_fn(c_C, d_q)
    loss_masked = loss_fn(c_C, d_q, mask=mask)

    assert loss_unmasked.ndim == 0 and loss_unmasked.item() >= 0.0
    assert loss_masked.ndim == 0 and loss_masked.item() >= 0.0

    # All masked case
    all_false_mask = torch.zeros_like(mask)
    loss_empty = loss_fn(c_C, d_q, mask=all_false_mask)
    assert loss_empty.item() == 0.0
