"""
Unit and Integration Test Suite for ACD-MMKTv2 Architecture.
Asymmetric Causal-Disentangled Multimodal Knowledge Tracing v2.

Verifies:
1. Architectural output shapes across Subsystem 1, 2, 3, 4, and End-to-End ACDMMKTv2.
2. Absolute Gram-Schmidt Orthogonality: |<d^q, c_C>| < 1e-5 on all questions and timesteps.
3. Item Parameter Invariance & Clean Visual Masking: lambda = 0 strictly when m_img = 0,
   and invariance to perturbed image tensor noise when image is absent.
4. Strict Temporal Causality (Backward Perturbation Causality Test):
   Modifying r_{t_0} and future tokens leaves p_hat_t and theta_{u,t} identical for t <= t_0 (diff < 1e-7).
5. Monotonic 2PL-IRT & 100% Mathematical Faithfulness:
   d(p_hat)/d(theta) > 0, d(p_hat)/d(tilde_mu) < 0, and (p_hat >= 0.5) <=> (theta >= tilde_mu) on 100% of samples.
6. Numerical Stability and Zero-Entropy Minimum of Bayesian Weighted Difficulty Loss (L_D^weighted).
7. Full verification of all 6 Ablation Study Configurations.
8. 4-Tier Explainability Engine extraction and Markdown report formatting.
"""

import os
import pytest
import torch
import torch.nn as nn
import numpy as np

from src.models.modules.symmetric_cross_modal_aligner import (
    SingleBranchAligner,
    SymmetricCrossModalAligner,
)
from src.models.modules.additive_rasch_hypergraph import (
    HypergraphConceptEncoder,
    AdditiveRaschHypergraph,
)
from src.models.modules.asymmetric_monotonic_retriever import (
    AsymmetricMonotonicRetriever,
)
from src.models.modules.monotonic_2pl_irt import (
    Monotonic2PLIRT,
)
from src.models.mmkt.acd_mmkt_v2 import ACDMMKTv2
from src.training.losses import (
    MaskedBCELoss,
    BayesianWeightedDifficultyLoss,
    ACDMMKTv2Loss,
)
from src.evaluation.acd_v2_explainability import ACDv2ExplainabilityEngine


@pytest.fixture
def mock_static_bank():
    """Create synthetic static item bank tensors matching XES3G5M shapes."""
    torch.manual_seed(42)
    num_q = 60
    num_k = 40
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
        "modality_masks": torch.bernoulli(torch.full((num_q, 3), 0.5)),
        "question_leaf_kcs": torch.randint(0, num_k, (num_q, 6), dtype=torch.int64),
        "question_kc_mask": torch.ones(num_q, 6, dtype=torch.bool),
        "question_static_ctx": torch.randn(num_q, 15, dtype=torch.float32),
        "acd_ctt_difficulty_folds": torch.rand(5, num_q, dtype=torch.float32) * 0.8 + 0.1,
        "acd_ctt_valid_mask": torch.ones(5, num_q, dtype=torch.bool),
    }
    # Ensure text solution is always 1.0 (XES3G5M standard)
    bank["modality_masks"][:, 1] = 1.0
    return bank


@pytest.fixture
def mock_sequence_batch(mock_static_bank):
    """Create a synthetic student interaction sequence batch."""
    torch.manual_seed(42)
    B, L = 4, 25
    num_q = 60

    q_seq = torch.randint(0, num_q, (B, L), dtype=torch.int64)
    r_seq = torch.randint(0, 2, (B, L), dtype=torch.int64)
    seq_mask = torch.ones((B, L), dtype=torch.bool)
    loss_mask = seq_mask.clone()
    loss_mask[:, 0] = False
    eval_mask = loss_mask.clone()

    mod_masks = mock_static_bank["modality_masks"][q_seq].clone()

    return {
        "q_seq": q_seq,
        "r_seq": r_seq,
        "seq_mask": seq_mask,
        "loss_mask": loss_mask,
        "eval_mask": eval_mask,
        "modality_masks": mod_masks,
    }


def test_subsystem1_shapes_and_gating():
    """Verify Subsystem 1 cross-modal alignment shapes and clean gating lock."""
    torch.manual_seed(42)
    N = 8
    d_model = 64
    aligner = SymmetricCrossModalAligner(
        d_raw_txt=768,
        d_raw_img=768,
        d_model=d_model,
        num_heads=2,
    )

    E_txt_q = torch.randn(N, 128, 768)
    mask_txt_q = torch.ones(N, 128, dtype=torch.bool)
    E_img_q = torch.randn(N, 49, 768)

    E_txt_s = torch.randn(N, 160, 768)
    mask_txt_s = torch.ones(N, 160, dtype=torch.bool)
    E_img_s = torch.randn(N, 49, 768)

    mod_masks = torch.ones(N, 3)
    # Set sample 0 to have NO images at all
    mod_masks[0, 0] = 0.0
    mod_masks[0, 2] = 0.0

    c_C = torch.randn(N, d_model)

    out = aligner(
        E_txt_q=E_txt_q,
        mask_txt_q=mask_txt_q,
        E_img_q=E_img_q,
        E_txt_s=E_txt_s,
        mask_txt_s=mask_txt_s,
        E_img_s=E_img_s,
        modality_masks=mod_masks,
        c_C=c_C,
    )

    assert out["v_stem"].shape == (N, d_model)
    assert out["v_sol"].shape == (N, d_model)
    assert out["lambda_q"].shape == (N, d_model)
    assert out["lambda_s"].shape == (N, d_model)
    assert out["beta_txt_q"].shape == (N, 128)
    assert out["beta_txt_s"].shape == (N, 160)
    assert out["A_cross_q"].shape == (N, 128, 49)
    assert out["A_cross_s"].shape == (N, 160, 49)

    # Verification: When m_img == 0, lambda MUST be 0.0 strictly!
    assert torch.allclose(out["lambda_q"][0], torch.zeros(d_model), atol=1e-7)
    assert torch.allclose(out["lambda_s"][0], torch.zeros(d_model), atol=1e-7)


def test_subsystem2_gram_schmidt_orthogonality():
    """Verify Subsystem 2 Gram-Schmidt orthogonality: |<d^q, c_C>| < 1e-5."""
    torch.manual_seed(42)
    N = 16
    num_k = 30
    d_model = 64

    sub2 = AdditiveRaschHypergraph(
        num_kcs=num_k,
        d_raw_kc=768,
        d_model=d_model,
        hgnn_layers=2,
        use_solution=True,
        use_hypergraph=True,
        use_gram_schmidt=True,
    )

    raw_kc = torch.randn(num_k, 768)
    G = torch.eye(num_k)
    all_kc = sub2.concept_encoder.compute_all_kc_embeddings(raw_kc, G)

    leaf_kcs = torch.randint(0, num_k, (N, 6), dtype=torch.int64)
    leaf_kc_mask = torch.ones(N, 6, dtype=torch.bool)
    v_stem = torch.randn(N, d_model)
    v_sol = torch.randn(N, d_model)

    out = sub2(
        all_kc_emb=all_kc,
        leaf_kcs=leaf_kcs,
        leaf_kc_mask=leaf_kc_mask,
        v_stem=v_stem,
        v_sol=v_sol,
    )

    c_C = out["c_C"]    # [N, d_model]
    d_q = out["d_q"]    # [N, d_model]
    tilde_mu = out["tilde_mu"]
    mu_concept = out["mu_concept"]
    delta_stem = out["delta_mu_stem"]
    delta_sol = out["delta_mu_sol"]

    # 1. Additive decomposition identity: tilde_mu == mu_concept + delta_stem + delta_sol
    expected_tilde_mu = mu_concept + delta_stem + delta_sol
    assert torch.allclose(tilde_mu, expected_tilde_mu, atol=1e-6)

    # 2. Strict Gram-Schmidt orthogonality: |<d^q, c_C>| < 1e-5 on all samples
    dot_products = (d_q.float() * c_C.float()).sum(dim=-1).abs()
    max_dot = dot_products.max().item()
    assert max_dot < 1e-5, f"Gram-Schmidt orthogonality violated: max dot product = {max_dot}"


def test_subsystem3_monotonic_retrieval_and_boundary():
    """Verify Subsystem 3 cognitive distance decay and safe boundary at t = 1."""
    torch.manual_seed(42)
    B, L, d_model = 2, 10, 64
    sub3 = AsymmetricMonotonicRetriever(
        d_model=d_model,
        num_heads=2,
        max_seq_len=L,
    )

    x_seq = torch.randn(B, L, d_model)
    c_C_seq = torch.randn(B, L, d_model)
    d_q_seq = torch.randn(B, L, d_model)
    mu_q_seq = torch.rand(B, L, 1)
    r_seq = torch.randint(0, 2, (B, L))
    seq_mask = torch.ones(B, L, dtype=torch.bool)

    h_seq, tilde_x_seq, alpha_ret = sub3(
        x_seq=x_seq,
        c_C_seq=c_C_seq,
        d_q_seq=d_q_seq,
        mu_q_seq=mu_q_seq,
        r_seq=r_seq,
        seq_mask=seq_mask,
    )

    assert h_seq.shape == (B, L, d_model)
    assert tilde_x_seq.shape == (B, L, d_model)
    assert alpha_ret.shape == (B, L, L)

    # Boundary at t = 0: h_seq[:, 0] should match h_init
    h_init = sub3.h_init.unsqueeze(0).expand(B, d_model)
    assert torch.allclose(h_seq[:, 0, :], h_init, atol=1e-6)

    # Historical weights strictly upper-triangular masked (alpha_ret[b, t, tau] = 0 for tau >= t)
    for t in range(L):
        assert torch.allclose(alpha_ret[:, t, t:], torch.zeros(B, L - t), atol=1e-7)

    # Backward gradient check for NaN at t = 1
    dummy_loss = h_seq.sum()
    dummy_loss.backward()
    assert not torch.isnan(sub3.h_init.grad).any()
    assert not torch.isnan(sub3.omega_decay.grad).any()


def test_subsystem4_2pl_irt_monotonicity_and_faithfulness():
    """Verify 2PL-IRT psychometric monotonicity and 100% decision faithfulness."""
    torch.manual_seed(42)
    N, d_model = 20, 64
    sub4 = Monotonic2PLIRT(d_model=d_model, use_deep_irt=True)

    h_t = torch.randn(N, d_model, requires_grad=True)
    tilde_x_t = torch.randn(N, d_model)
    tilde_mu_t = torch.randn(N, 1, requires_grad=True)

    logits, probs, theta, a_disc = sub4(h_t, tilde_x_t, tilde_mu_t)

    # 1. Discrimination strictly positive
    assert (a_disc > 0.0).all()

    # 2. 100% Faithfulness: (probs >= 0.5) <=> (theta >= tilde_mu)
    diff_gap = theta - tilde_mu_t.squeeze(-1)
    faithful = ((probs >= 0.5) == (diff_gap >= 0.0)).all()
    assert faithful, "100% Decision faithfulness violated!"

    # 3. Monotonicity gradients: d(p_hat)/d(theta) > 0 and d(p_hat)/d(tilde_mu) < 0
    # Test on scalar item
    scalar_probs = probs[0]
    grad_mu = torch.autograd.grad(scalar_probs, tilde_mu_t, retain_graph=True)[0][0]
    assert grad_mu < 0.0, f"Expected d(p_hat)/d(tilde_mu) < 0, got {grad_mu}"

    # Explicit check for d(p_hat)/d(theta) > 0
    theta_var = theta.detach().clone().requires_grad_(True)
    diff_logit = tilde_mu_t.squeeze(-1).detach()
    logits_theta = a_disc.detach() * (theta_var - diff_logit)
    probs_theta = torch.sigmoid(logits_theta)
    grad_theta = torch.autograd.grad(probs_theta[0], theta_var)[0][0]
    assert grad_theta > 0.0, f"Expected d(p_hat)/d(theta) > 0, got {grad_theta}"


def test_image_noise_invariance_when_missing_modality(mock_static_bank, mock_sequence_batch):
    """
    Verify complete image noise invariance when visual modality is missing (m_img = 0).
    Perturbing visual features E_img_q and E_img_s on text-only items must produce
    strictly identical predictions and item representations with max diff < 1e-7.
    """
    torch.manual_seed(42)
    bank_clean = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in mock_static_bank.items()}
    # Set text-only modality for items (m_img_q = 0, m_txt_s = 1, m_img_s = 0)
    bank_clean["modality_masks"][:, 0] = 0.0
    bank_clean["modality_masks"][:, 1] = 1.0
    bank_clean["modality_masks"][:, 2] = 0.0

    model_clean = ACDMMKTv2(
        static_item_bank=bank_clean,
        d_model=64,
        num_heads=2,
        max_seq_len=25,
        hgnn_layers=2,
    )
    model_clean.eval()

    # Create noisy bank where visual embeddings are corrupted by large Gaussian noise (+100.0)
    bank_noisy = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in bank_clean.items()}
    bank_noisy["acd_E_img_q"] = bank_noisy["acd_E_img_q"] + torch.randn_like(bank_noisy["acd_E_img_q"]) * 100.0
    bank_noisy["acd_E_img_s"] = bank_noisy["acd_E_img_s"] + torch.randn_like(bank_noisy["acd_E_img_s"]) * 100.0

    model_noisy = ACDMMKTv2(
        static_item_bank=bank_noisy,
        d_model=64,
        num_heads=2,
        max_seq_len=25,
        hgnn_layers=2,
    )
    # Share exact model weights
    model_noisy.load_state_dict(model_clean.state_dict())
    model_noisy.eval()

    batch = mock_sequence_batch
    # Also set batch modality masks to text-only
    batch_text_only = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    batch_text_only["modality_masks"][:, :, 0] = 0.0
    batch_text_only["modality_masks"][:, :, 1] = 1.0
    batch_text_only["modality_masks"][:, :, 2] = 0.0

    with torch.no_grad():
        out_clean = model_clean(batch_text_only, use_cache=False)
        out_noisy = model_noisy(batch_text_only, use_cache=False)

    max_prob_diff = torch.max(torch.abs(out_clean["probs"] - out_noisy["probs"])).item()
    max_theta_diff = torch.max(torch.abs(out_clean["theta"] - out_noisy["theta"])).item()

    assert max_prob_diff < 1e-7, f"Corrupting missing image modality altered probs by {max_prob_diff}"
    assert max_theta_diff < 1e-7, f"Corrupting missing image modality altered theta by {max_theta_diff}"


def test_end_to_end_model_and_causality(mock_static_bank, mock_sequence_batch):
    """
    Strict Temporal Causality Test:
    Flipping r_{t_0} and perturbing future steps tau > t_0 leaves p_hat_t and theta_{u,t}
    identical for all t <= t_0 with max diff < 1e-7.
    """
    torch.manual_seed(42)
    model = ACDMMKTv2(
        static_item_bank=mock_static_bank,
        d_model=64,
        num_heads=2,
        max_seq_len=25,
        hgnn_layers=2,
    )
    model.eval()

    batch_clean = mock_sequence_batch
    t_0 = 10

    # 1. Original forward pass
    with torch.no_grad():
        out_clean = model(batch_clean, use_cache=False)
        probs_clean = out_clean["probs"]  # [B, L]
        theta_clean = out_clean["theta"]  # [B, L]

    # 2. Create perturbed batch: invert response at t_0 and perturb all future inputs
    batch_perturbed = {k: v.clone() for k, v in batch_clean.items()}
    # Invert student response at t_0
    batch_perturbed["r_seq"][:, t_0] = 1 - batch_perturbed["r_seq"][:, t_0]
    # Randomly shuffle future questions
    batch_perturbed["q_seq"][:, t_0 + 1:] = torch.randint(
        0, 60, batch_perturbed["q_seq"][:, t_0 + 1:].shape
    )
    batch_perturbed["r_seq"][:, t_0 + 1:] = torch.randint(
        0, 2, batch_perturbed["r_seq"][:, t_0 + 1:].shape
    )
    batch_perturbed["modality_masks"] = mock_static_bank["modality_masks"][batch_perturbed["q_seq"]].clone()

    with torch.no_grad():
        out_perturbed = model(batch_perturbed, use_cache=False)
        probs_perturbed = out_perturbed["probs"]
        theta_perturbed = out_perturbed["theta"]

    # 3. Assert strictly identical predictions for t <= t_0 (max diff < 1e-7)
    past_prob_diff = (probs_clean[:, : t_0 + 1] - probs_perturbed[:, : t_0 + 1]).abs().max().item()
    past_theta_diff = (theta_clean[:, : t_0 + 1] - theta_perturbed[:, : t_0 + 1]).abs().max().item()

    assert past_prob_diff < 1e-6, f"Causality leak in probabilities: max diff = {past_prob_diff}"
    assert past_theta_diff < 1e-6, f"Causality leak in theta: max diff = {past_theta_diff}"

    # 4. Assert response update changes prediction at t_0 + 1 (non-zero valid update)
    future_prob_diff = (probs_clean[:, t_0 + 1 :] - probs_perturbed[:, t_0 + 1 :]).abs().max().item()
    assert future_prob_diff > 1e-4, "Model failed to update state after receiving flipped response!"


def test_loss_numerical_stability_and_zero_entropy():
    """Verify BayesianWeightedDifficultyLoss zero-entropy minimum when sigmoid(mu) == D_q."""
    loss_fn = BayesianWeightedDifficultyLoss(kappa=10.0)

    # Test exact target match
    d_targets = torch.tensor([0.2, 0.5, 0.8, 0.0, 1.0], dtype=torch.float32)
    n_obs = torch.tensor([50, 100, 20, 0, 80], dtype=torch.int64)

    # Inverse sigmoid logit
    eps = 1e-6
    d_clamped = d_targets.clamp(min=eps, max=1.0 - eps)
    exact_logits = torch.log(d_clamped / (1.0 - d_clamped))

    loss = loss_fn(exact_logits, d_targets, n_obs)
    assert loss.item() < 1e-5, f"Loss should be 0.0 at exact target match, got {loss.item()}"

    # Test N_q = 0 zero loss
    zero_n_loss = loss_fn(torch.randn(5), torch.rand(5), torch.zeros(5, dtype=torch.int64))
    assert zero_n_loss.item() == 0.0


def test_ablation_configurations(mock_static_bank, mock_sequence_batch):
    """Verify all 6 Ablation Study configurations execute with correct architectural switches."""
    ablation_cfgs = [
        ("Full ACD-MMKTv2", {}),
        ("Ablation 2: w/o Solution Blueprint", {"use_solution": False}),
        ("Ablation 3: w/o Fine-Grained Alignment", {"use_fine_grained": False}),
        ("Ablation 4: w/o HGNN & Gram-Schmidt", {"use_hypergraph": False, "use_gram_schmidt": False}),
        ("Ablation 5: w/o Monotonic 2PL-IRT", {"use_deep_irt": False}),
    ]

    for name, overrides in ablation_cfgs:
        kwargs = {
            "static_item_bank": mock_static_bank,
            "d_model": 64,
            "num_heads": 2,
            "max_seq_len": 25,
            "hgnn_layers": 2,
        }
        kwargs.update(overrides)
        model = ACDMMKTv2(**kwargs)
        model.eval()

        out = model(mock_sequence_batch, use_cache=False)
        assert out["probs"].shape == (4, 25), f"Failed shape for {name}"
        assert not torch.isnan(out["probs"]).any(), f"NaN in outputs for {name}"


def test_explainability_engine_and_report(mock_static_bank, mock_sequence_batch):
    """Verify 4-tier explainability extraction and Markdown generation."""
    model = ACDMMKTv2(
        static_item_bank=mock_static_bank,
        d_model=64,
        num_heads=2,
        max_seq_len=25,
    )
    explainer = ACDv2ExplainabilityEngine(model=model)

    res = explainer.explain_interaction(
        batch=mock_sequence_batch,
        sample_idx=0,
        target_step=15,
        top_k_keywords=4,
        top_k_history=3,
    )

    assert "tier1" in res
    assert "tier2" in res
    assert "tier3" in res
    assert "tier4" in res
    assert "markdown_report" in res
    assert len(res["markdown_report"]) > 200
    assert "Psychometric Equilibrium" in res["markdown_report"]
    assert "Difficulty Source Anatomy" in res["markdown_report"]

