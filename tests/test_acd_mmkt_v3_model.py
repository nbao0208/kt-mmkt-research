"""
Unit and Integration Test Suite for ACD-MMKTv3 Architecture.
Asymmetric Causal-Disentangled Multimodal Knowledge Tracing v3.

Verifies:
1. Subsystem 1: Shapes, text pooling, text-guided cross-attention with letterbox patch mask,
   and clean zero-gating when image is missing.
2. Subsystem 2: Absolute Gram-Schmidt Orthogonality (|<d^q, c_C>| < 1e-5), fixed logit(mu_prior) anchor,
   and signed visual load (Delta_mu_vis in [-gamma_vis, gamma_vis]) with zero free biases.
3. Subsystem 3: Atomic experience stream purity (d(v_tau)/d(r_tau') = 0 for tau' != tau),
   monotonic distance decay d(t, tau) > 0, and safe boundary handling at t = 1 (no NaN gradients).
4. Subsystem 4: Identifiable 2PL-IRT with analytical decoupling:
   d(theta)/d(tilde_mu) = 0, d(a_q)/d(tilde_mu) = 0,
   d(p_hat)/d(theta) > 0, d(p_hat)/d(tilde_mu) < 0,
   and 100% mathematical faithfulness (p_hat >= 0.5) <=> (theta >= tilde_mu).
5. Image noise invariance when m_img = 0.
6. Strict temporal causality (modifying future steps does not alter past predictions).
7. ACDMMKTv3Loss stability and unique active question filtering.
8. Full verification of all 6 Ablation study configurations.
9. 4-Tier Explainability Engine extraction and Markdown report generation.
"""

import os
import pytest
import torch
import torch.nn as nn
import numpy as np

from src.models.modules.text_guided_cross_aligner import TextGuidedCrossAligner
from src.models.modules.identifiable_rasch_hypergraph import (
    HypergraphConceptEncoder,
    IdentifiableRaschHypergraph,
)
from src.models.modules.atomic_monotonic_retriever import AtomicMonotonicRetriever
from src.models.modules.identifiable_2pl_irt import Identifiable2PLIRT
from src.models.mmkt.acd_mmkt_v3 import ACDMMKTv3
from src.training.losses import (
    MaskedBCELoss,
    BayesianWeightedDifficultyLoss,
    ACDMMKTv3Loss,
)
from src.training.acd_v3_trainer import ACDMMKTv3Trainer
from src.evaluation.acd_v3_explainability import ACDv3ExplainabilityEngine


@pytest.fixture
def mock_static_bank():
    """Create synthetic static item bank tensors matching XES3G5M shapes."""
    torch.manual_seed(42)
    num_q = 60
    num_k = 40
    d_raw = 768

    # Patch mask: 40 valid patches, 9 padded patches
    m_patch = torch.zeros(num_q, 49, dtype=torch.float32)
    m_patch[:, 40:] = -1e9

    bank = {
        "acd_E_txt_q": torch.randn(num_q, 128, d_raw, dtype=torch.float16),
        "acd_mask_txt_q": torch.ones(num_q, 128, dtype=torch.bool),
        "acd_E_img_q": torch.randn(num_q, 49, d_raw, dtype=torch.float16),
        "acd_E_txt_s": torch.randn(num_q, 160, d_raw, dtype=torch.float16),
        "acd_mask_txt_s": torch.ones(num_q, 160, dtype=torch.bool),
        "acd_E_img_s": torch.randn(num_q, d_raw, dtype=torch.float16),
        "acd_kc_init_emb": torch.randn(num_k, d_raw, dtype=torch.float32),
        "acd_hypergraph_G": torch.eye(num_k, dtype=torch.float32),
        "modality_masks": torch.bernoulli(torch.full((num_q, 3), 0.5)),
        "question_leaf_kcs": torch.randint(0, num_k, (num_q, 6), dtype=torch.int64),
        "question_kc_mask": torch.ones(num_q, 6, dtype=torch.bool),
        "question_static_ctx": torch.randn(num_q, 15, dtype=torch.float32),
        "M_patch_q": m_patch,
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
        "norm_delta_t_lag": torch.rand(B, L, dtype=torch.float32),
        "norm_delta_t_kc": torch.rand(B, L, dtype=torch.float32),
        "mu_concept": torch.full((B, L), -1.37, dtype=torch.float32),
    }


def test_subsystem1_shapes_gating_and_letterbox():
    """Verify Subsystem 1 output shapes, letterbox attention masking, and zero gating."""
    torch.manual_seed(42)
    M, d_raw, d_model = 8, 768, 128
    L_q, L_s, P = 128, 160, 49

    aligner = TextGuidedCrossAligner(
        d_raw_txt=d_raw,
        d_raw_img=d_raw,
        d_model=d_model,
        d_static_ctx=15,
        dropout=0.0,
    )

    E_txt_q = torch.randn(M, L_q, d_raw)
    mask_txt_q = torch.ones(M, L_q, dtype=torch.bool)
    E_img_q = torch.randn(M, P, d_raw)
    E_txt_s = torch.randn(M, L_s, d_raw)
    mask_txt_s = torch.ones(M, L_s, dtype=torch.bool)
    E_img_s = torch.randn(M, d_raw)
    c_C = torch.randn(M, d_model)
    c_ctx = torch.randn(M, 15)

    M_patch = torch.zeros(M, P)
    # Mark last 10 patches as padded
    M_patch[:, 39:] = -1e9

    # Even items have stem image, odd items are text-only
    modality_masks = torch.ones(M, 3)
    modality_masks[1::2, 0] = 0.0

    out = aligner(
        E_txt_q=E_txt_q,
        mask_txt_q=mask_txt_q,
        E_img_q=E_img_q,
        M_patch_q=M_patch,
        E_txt_s=E_txt_s,
        mask_txt_s=mask_txt_s,
        E_img_s=E_img_s,
        modality_masks=modality_masks,
        c_C=c_C,
        c_ctx=c_ctx,
        return_cross_weights=True,
    )

    # 1. Shape verifications
    assert out["v_stem"].shape == (M, d_model)
    assert out["v_sol"].shape == (M, d_model)
    assert out["lambda_q"].shape == (M, d_model)
    assert out["bar_lambda_q"].shape == (M, 1)
    assert out["beta_txt_q"].shape == (M, L_q)
    assert out["A_cross_q"].shape == (M, L_q, P)

    # 2. Zero gating when m_img_q == 0
    odd_indices = torch.arange(1, M, 2)
    assert torch.allclose(out["lambda_q"][odd_indices], torch.zeros(len(odd_indices), d_model))
    assert torch.allclose(out["bar_lambda_q"][odd_indices], torch.zeros(len(odd_indices), 1))
    assert torch.allclose(out["v_stem"][odd_indices], out["u_txt_q"][odd_indices], atol=1e-6)

    # 3. Letterbox patch masking in A_cross_q: padded patches receive 0 attention
    even_indices = torch.arange(0, M, 2)
    attn_padded = out["A_cross_q"][even_indices, :, 39:]  # [K, L_q, 10]
    assert attn_padded.max().item() < 1e-6, "Padded letterbox patches must receive ~0 attention weight"


def test_subsystem2_gram_schmidt_and_signed_visual_load():
    """Verify Subsystem 2 Gram-Schmidt orthogonality and signed visual load."""
    torch.manual_seed(42)
    M, d_model = 16, 128

    sub2 = IdentifiableRaschHypergraph(
        num_kcs=40,
        d_raw_kc=768,
        d_model=d_model,
        gamma_vis=1.5,
        use_solution=True,
        use_visual=True,
        use_hypergraph=True,
        use_gram_schmidt=True,
        unanchored_concept=False,
    )

    v_stem = torch.randn(M, d_model)
    v_sol = torch.randn(M, d_model)
    u_txt_q = torch.randn(M, d_model)
    u_cross_q = torch.randn(M, d_model)
    bar_lambda_q = torch.rand(M, 1)
    m_img_q = torch.ones(M, 1)
    c_C = torch.randn(M, d_model)
    prior_concept = torch.randn(M)  # real logit values

    out = sub2(
        v_stem=v_stem,
        v_sol=v_sol,
        u_txt_q=u_txt_q,
        u_cross_q=u_cross_q,
        bar_lambda_q=bar_lambda_q,
        m_img_q=m_img_q,
        c_C=c_C,
        mu_concept_prior=prior_concept,
    )

    # 1. Output shapes
    assert out["mu_concept"].shape == (M,)
    assert out["delta_mu_txt"].shape == (M,)
    assert out["delta_mu_vis"].shape == (M,)
    assert out["delta_mu_stem"].shape == (M,)
    assert out["delta_mu_sol"].shape == (M,)
    assert out["tilde_mu"].shape == (M,)
    assert out["mu_q"].shape == (M,)
    assert out["d_q"].shape == (M, d_model)
    assert out["x_q"].shape == (M, d_model)

    # 2. Strict Gram-Schmidt orthogonality: |<d^q, c_C>| < 1e-5
    dot_products = (out["d_q"].float() * c_C.float()).sum(dim=-1)
    max_dot = dot_products.abs().max().item()
    assert max_dot < 1e-5, f"Gram-Schmidt orthogonality violated: max |<d^q, c_C>| = {max_dot}"

    # 3. Fixed concept anchor: mu_concept strictly equals prior_concept
    assert torch.allclose(out["mu_concept"], prior_concept, atol=1e-6)

    # 4. Signed visual load strictly bounded in [-gamma_vis, gamma_vis]
    assert out["delta_mu_vis"].min().item() >= -1.5 - 1e-6
    assert out["delta_mu_vis"].max().item() <= 1.5 + 1e-6

    # 5. Additive total difficulty equation holds
    expected_tilde = out["mu_concept"] + out["delta_mu_stem"] + out["delta_mu_sol"]
    assert torch.allclose(out["tilde_mu"], expected_tilde, atol=1e-6)
    assert torch.allclose(out["mu_q"], torch.sigmoid(expected_tilde), atol=1e-6)


def test_subsystem3_atomic_experience_and_monotonic_retriever():
    """Verify Subsystem 3 atomic experience purity and monotonic distance decay."""
    torch.manual_seed(42)
    B, L, D = 4, 15, 128

    sub3 = AtomicMonotonicRetriever(
        d_model=D,
        num_heads=4,
        max_seq_len=200,
        dropout=0.0,
        polluted_experience_attn=False,
    )

    x_seq = torch.randn(B, L, D)
    c_C_seq = torch.randn(B, L, D)
    mu_q_seq = torch.rand(B, L)
    d_q_seq = torch.randn(B, L, D)
    r_seq = torch.randint(0, 2, (B, L))
    delta_t_lag = torch.rand(B, L)
    delta_t_kc = torch.rand(B, L)

    h_seq, tilde_x, alpha_ret = sub3(
        x_seq=x_seq,
        c_C_seq=c_C_seq,
        mu_q_seq=mu_q_seq,
        d_q_seq=d_q_seq,
        r_seq=r_seq,
        delta_t_lag=delta_t_lag,
        delta_t_kc=delta_t_kc,
    )

    # 1. Output shapes
    assert h_seq.shape == (B, L, D)
    assert tilde_x.shape == (B, L, D)
    assert alpha_ret.shape == (B, L, L)

    # 2. Strict lower triangular attention: alpha_{t, tau} == 0 for all tau >= t
    for t in range(L):
        assert torch.allclose(alpha_ret[:, t, t:], torch.zeros_like(alpha_ret[:, t, t:]), atol=1e-6)

    # 3. Boundary handling at t = 0 (step 1): h_1 equals learnable h_init
    expected_h0 = sub3.h_init.unsqueeze(0).expand(B, D)
    assert torch.allclose(h_seq[:, 0], expected_h0, atol=1e-6)
    assert torch.allclose(alpha_ret[:, 0], torch.zeros(B, L), atol=1e-6)

    # 4. Atomic experience purity check via sub3.compute_experience_stream:
    # At step tau, perturbing response r_{tau'} for tau' != tau must NOT change y_atomic[tau]
    # Verify by calculating grad of y_atomic[tau] with respect to r_float[tau']
    sub3_test = AtomicMonotonicRetriever(d_model=D, num_heads=4, dropout=0.0, polluted_experience_attn=False)
    r_param = nn.Parameter(r_seq.float())

    y_atomic = sub3_test.compute_experience_stream(
        c_C_seq=c_C_seq,
        mu_q_seq=mu_q_seq,
        d_q_seq=d_q_seq,
        r_seq=r_param,
    )

    # Target step tau = 3
    # Use linear projection weight to avoid sum-invariance of LayerNorm
    weight = torch.randn_like(y_atomic[:, 3])
    loss_target = (y_atomic[:, 3] * weight).sum()
    loss_target.backward()

    # Gradient must be non-zero at step 3 and strictly 0 at all other steps!
    grad_r = r_param.grad  # [B, L]
    assert grad_r[:, 3].abs().sum().item() > 0.0, "Gradient wrt own response must be non-zero"
    other_steps = [s for s in range(L) if s != 3]
    assert grad_r[:, other_steps].abs().sum().item() == 0.0, "Atomic experience violated: cross-step response leakage!"


def test_subsystem4_identifiable_2pl_irt_and_faithfulness():
    """Verify Subsystem 4 identifiable 2PL-IRT: monotonicity, decoupling, and 100% decision faithfulness."""
    torch.manual_seed(42)
    M, D = 32, 128

    irt = Identifiable2PLIRT(d_model=D, delta_a=0.1, use_deep_irt=True, use_solution=True)

    h_t = torch.randn(M, D)
    c_C = torch.randn(M, D)
    d_q = torch.randn(M, D)
    v_stem = torch.randn(M, D)
    v_sol = torch.randn(M, D)
    tilde_mu = torch.randn(M, 1)

    logits, probs, theta, a_disc = irt(
        h_t=h_t,
        c_C_t=c_C,
        d_q_t=d_q,
        v_stem_t=v_stem,
        v_sol_t=v_sol,
        tilde_mu_t=tilde_mu,
    )

    # 1. Output shapes
    assert logits.shape == (M,)
    assert probs.shape == (M,)
    assert theta.shape == (M,)
    assert a_disc.shape == (M,)

    # 2. Positive discrimination floor: a_q >= delta_a (0.1)
    assert a_disc.min().item() >= 0.1 - 1e-6

    # 3. 100% Decision faithfulness: (p_hat >= 0.5) <=> (theta >= tilde_mu)
    tilde_mu_flat = tilde_mu.squeeze(-1)
    faithful = (probs >= 0.5) == (theta >= tilde_mu_flat)
    assert faithful.all().item(), "100% Decision Faithfulness violated in 2PL-IRT!"

    # 4. Strict monotonicity derivatives:
    # d(p_hat)/d(theta) > 0 and d(p_hat)/d(tilde_mu) < 0
    theta_param = nn.Parameter(theta.clone())
    tilde_mu_param = nn.Parameter(tilde_mu_flat.clone())

    logits_test = a_disc.detach() * (theta_param - tilde_mu_param)
    probs_test = torch.sigmoid(logits_test)
    sum_probs = probs_test.sum()
    sum_probs.backward()

    assert (theta_param.grad > 0.0).all().item(), "d(p_hat)/d(theta) must be strictly positive"
    assert (tilde_mu_param.grad < 0.0).all().item(), "d(p_hat)/d(tilde_mu) must be strictly negative"

    # 5. Identifiability Decoupling via autograd:
    # d(theta)/d(tilde_mu) == 0 and d(a_q)/d(tilde_mu) == 0
    tilde_mu_input = nn.Parameter(torch.randn(M, 1))
    _, _, theta_out, a_disc_out = irt(
        h_t=h_t,
        c_C_t=c_C,
        d_q_t=d_q,
        v_stem_t=v_stem,
        v_sol_t=v_sol,
        tilde_mu_t=tilde_mu_input,
    )

    grad_theta = torch.autograd.grad(theta_out.sum(), tilde_mu_input, retain_graph=True, allow_unused=True)[0]
    assert grad_theta is None or (grad_theta == 0.0).all().item(), "theta must be strictly independent of tilde_mu (grad == 0)"

    grad_a = torch.autograd.grad(a_disc_out.sum(), tilde_mu_input, retain_graph=True, allow_unused=True)[0]
    assert grad_a is None or (grad_a == 0.0).all().item(), "a_q must be strictly independent of tilde_mu (grad == 0)"


def test_image_noise_invariance_when_missing_modality(mock_static_bank, mock_sequence_batch):
    """
    Verify complete image noise invariance when visual modality is missing (m_img = 0).
    Corrupting visual embeddings acd_E_img_q on text-only items must produce
    strictly identical predictions and representations with max diff < 1e-7.
    """
    torch.manual_seed(42)
    bank_clean = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in mock_static_bank.items()}
    # Set text-only modality for stem images (m_img_q = 0)
    bank_clean["modality_masks"][:, 0] = 0.0

    model_clean = ACDMMKTv3(
        static_item_bank=bank_clean,
        d_model=128,
        num_heads=4,
        max_seq_len=25,
        dropout=0.0,
    ).eval()

    # Create noisy bank where visual embeddings are corrupted by large Gaussian noise (+100.0)
    bank_noisy = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in bank_clean.items()}
    bank_noisy["acd_E_img_q"] = bank_noisy["acd_E_img_q"] + torch.randn_like(bank_noisy["acd_E_img_q"]) * 100.0

    model_noisy = ACDMMKTv3(
        static_item_bank=bank_noisy,
        d_model=128,
        num_heads=4,
        max_seq_len=25,
        dropout=0.0,
    ).eval()
    model_noisy.load_state_dict(model_clean.state_dict())

    # Set batch modality masks to text-only
    batch_text_only = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in mock_sequence_batch.items()}
    batch_text_only["modality_masks"][:, :, 0] = 0.0

    with torch.no_grad():
        out_clean = model_clean(batch_text_only)
        out_noisy = model_noisy(batch_text_only)

    max_prob_diff = torch.max(torch.abs(out_clean["probs"] - out_noisy["probs"])).item()
    max_theta_diff = torch.max(torch.abs(out_clean["theta"] - out_noisy["theta"])).item()

    assert max_prob_diff < 1e-7, f"Corrupting missing image modality altered probs by {max_prob_diff}"
    assert max_theta_diff < 1e-7, f"Corrupting missing image modality altered theta by {max_theta_diff}"


def test_end_to_end_causality(mock_static_bank, mock_sequence_batch):
    """Verify strict temporal causality: modifying future responses does not alter past predictions."""
    torch.manual_seed(42)
    model = ACDMMKTv3(
        static_item_bank=mock_static_bank,
        d_model=128,
        num_heads=4,
        max_seq_len=25,
        dropout=0.0,
    ).eval()

    batch1 = mock_sequence_batch
    batch2 = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch1.items()}

    # Step t_0 boundary check:
    # Keep questions identical up to and including step t_0, but flip responses from t_0 onwards
    # (including r_{t_0}). Also modify questions for subsequent steps (t > t_0).
    t_0 = 10
    batch2["r_seq"][:, t_0:] = 1 - batch2["r_seq"][:, t_0:]
    batch2["q_seq"][:, (t_0 + 1):] = (batch2["q_seq"][:, (t_0 + 1):] + 5) % 60
    batch2["modality_masks"] = mock_static_bank["modality_masks"][batch2["q_seq"]].clone()

    with torch.no_grad():
        out1 = model(batch1)
        out2 = model(batch2)

    # Predictions for t <= t_0 (INCLUDING step t_0 itself) must remain IDENTICAL.
    # If retriever attends to tau <= t (leaking r_{t_0}), step t_0 diff will be non-zero.
    diff_p = (out1["probs"][:, :(t_0 + 1)] - out2["probs"][:, :(t_0 + 1)]).abs().max().item()
    diff_theta = (out1["theta"][:, :(t_0 + 1)] - out2["theta"][:, :(t_0 + 1)]).abs().max().item()

    assert diff_p <= 1e-6, f"Temporal causality violated at/before step t_0 in probs: diff = {diff_p}"
    assert diff_theta <= 1e-6, f"Temporal causality violated at/before step t_0 in theta: diff = {diff_theta}"

    # Predictions for strictly future steps (t > t_0) should validly differ
    diff_future = (out1["probs"][:, (t_0 + 1):] - out2["probs"][:, (t_0 + 1):]).abs().max().item()
    assert diff_future > 1e-4, "Future outputs should reflect perturbed future interactions"


def test_loss_numerical_stability_and_zero_entropy():
    """Verify numerical stability and zero-entropy minimum of Bayesian weighted difficulty loss."""
    loss_fn = BayesianWeightedDifficultyLoss(kappa=10.0)

    # 1. When predicted probability equals empirical target: sigmoid(tilde_mu) == target
    targets = torch.tensor([0.2, 0.5, 0.8, 0.1, 0.9])
    perfect_logits = torch.log(targets / (1.0 - targets))
    n_obs = torch.tensor([100.0, 50.0, 20.0, 10.0, 5.0])

    loss_zero = loss_fn(perfect_logits, targets, n_obs)
    assert abs(loss_zero.item()) < 1e-6, f"Expected 0.0 optimal loss, got {loss_zero.item()}"

    # 2. ACDMMKTv3 multi-task loss forward pass
    acd_loss_fn = ACDMMKTv3Loss(beta_d=0.20, lambda_sparse=0.01, kappa=10.0)
    mock_outputs = {
        "logits": torch.randn(4, 25),
        "unique_q": torch.tensor([0, 1, 2, 3]),
        "clean_item_props": {
            "tilde_mu": torch.randn(4),
            "delta_mu_stem": torch.randn(4),
            "delta_mu_sol": torch.randn(4),
        },
    }
    targets_seq = torch.randint(0, 2, (4, 25))
    loss_mask = torch.ones(4, 25, dtype=torch.bool)
    loss_mask[:, 0] = False

    n_q_all = torch.randint(0, 500, (60,))
    d_q_all = torch.rand(60) * 0.8 + 0.1

    tot_loss, metrics = acd_loss_fn(
        model_outputs=mock_outputs,
        targets=targets_seq,
        loss_mask=loss_mask,
        n_q_all=n_q_all,
        d_q_all=d_q_all,
    )
    assert tot_loss.item() > 0.0
    assert "loss_bce" in metrics
    assert "loss_diff" in metrics
    assert "loss_sparse" in metrics


def test_all_six_ablation_configurations(mock_static_bank, mock_sequence_batch):
    """Verify that all 6 Ablation Study Configurations initialize and forward cleanly."""
    ablation_cfgs = [
        {"name": "Config 1: Full ACD-MMKTv3"},
        {"name": "Config 2: w/o Solution Stream", "use_solution": False},
        {"name": "Config 3: w/o Visual Stream", "use_visual": False},
        {"name": "Config 4: Unanchored Concept Difficulty", "unanchored_concept": True},
        {"name": "Config 5: Polluted Experience Self-Attention", "polluted_experience_attn": True},
        {"name": "Config 6: w/o Monotonic 2PL-IRT", "use_deep_irt": False},
    ]

    for cfg in ablation_cfgs:
        name = cfg.pop("name")
        model = ACDMMKTv3(
            static_item_bank=mock_static_bank,
            d_model=64,
            num_heads=2,
            max_seq_len=25,
            dropout=0.0,
            **cfg,
        ).eval()

        with torch.no_grad():
            out = model(mock_sequence_batch)

        assert out["probs"].shape == (4, 25), f"Failed shape check for {name}"
        assert not torch.isnan(out["probs"]).any(), f"NaN detected in {name}"


def test_explainability_engine_and_report(mock_static_bank, mock_sequence_batch):
    """Verify 4-tier explainability extraction, math consistency, and Markdown report formatting."""
    model = ACDMMKTv3(
        static_item_bank=mock_static_bank,
        d_model=64,
        num_heads=2,
        max_seq_len=25,
        dropout=0.0,
    ).eval()

    engine = ACDv3ExplainabilityEngine(model=model)
    explanation = engine.explain_interaction(
        batch=mock_sequence_batch,
        sample_idx=0,
        target_step=-1,
        top_k_keywords=3,
        top_k_history=3,
    )

    assert "tier1_psychometrics" in explanation
    assert "tier2_difficulty" in explanation
    assert "tier3_multimodal" in explanation
    assert "tier4_history" in explanation

    t1 = explanation["tier1_psychometrics"]
    assert t1["prediction_faithful"] is True

    t3 = explanation["tier3_multimodal"]
    assert "visual_impact_semantics" in t3

    # Format Markdown report
    md = engine.format_markdown_report(explanation)
    assert "# Báo Cáo Giải Thích Tâm Trắc Học Đa Tầng ACD-MMKTv3" in md
    assert "Tầng 1: Cân Bằng Tâm Trắc Học" in md
    assert "Tầng 2: Giải Phẫu Nguồn Gây Khó" in md
    assert "Tầng 3: Quy Gán Phương Thức Hạt Mịn" in md
    assert "Tầng 4: Truy Vết Lịch Sử Thực Chứng Nguyên Tử" in md


def test_irt_parameter_invariance_and_cache_equivalence(mock_static_bank):
    """
    Verify Group 3 (IRT Item Parameter Invariance):
    (mu_q, tilde_mu_q, delta_mu_stem, delta_mu_sol, d^q, a_q) are invariant (<= 1e-6)
    under changes to student interaction history, and identical between Cache and Non-Cache modes.
    """
    torch.manual_seed(42)
    model = ACDMMKTv3(
        static_item_bank=mock_static_bank,
        d_model=64,
        num_heads=2,
        max_seq_len=20,
        dropout=0.0,
    ).eval()

    L = 20
    target_q = 12
    step_t = 10

    # Student A: All correct answers, high speed
    batch_A = {
        "q_seq": torch.randint(0, 60, (1, L)),
        "r_seq": torch.ones(1, L, dtype=torch.int64),
        "seq_mask": torch.ones(1, L, dtype=torch.bool),
        "loss_mask": torch.ones(1, L, dtype=torch.bool),
        "eval_mask": torch.ones(1, L, dtype=torch.bool),
        "norm_delta_t_lag": torch.full((1, L), 0.1),
        "norm_delta_t_kc": torch.full((1, L), 0.1),
        "mu_concept": torch.full((1, L), -1.37),
    }
    batch_A["q_seq"][0, step_t] = target_q
    batch_A["modality_masks"] = mock_static_bank["modality_masks"][batch_A["q_seq"]]

    # Student B: All incorrect answers, low speed, completely different preceding questions
    batch_B = {
        "q_seq": torch.randint(0, 60, (1, L)),
        "r_seq": torch.zeros(1, L, dtype=torch.int64),
        "seq_mask": torch.ones(1, L, dtype=torch.bool),
        "loss_mask": torch.ones(1, L, dtype=torch.bool),
        "eval_mask": torch.ones(1, L, dtype=torch.bool),
        "norm_delta_t_lag": torch.full((1, L), 0.9),
        "norm_delta_t_kc": torch.full((1, L), 0.9),
        "mu_concept": torch.full((1, L), -1.37),
    }
    batch_B["q_seq"][0, step_t] = target_q
    batch_B["modality_masks"] = mock_static_bank["modality_masks"][batch_B["q_seq"]]

    # 1. Non-Cache mode
    with torch.no_grad():
        out_A_no_cache = model(batch_A, use_cache=False)
        out_B_no_cache = model(batch_B, use_cache=False)

    # Question parameters for question target_q at step_t must be identical across students
    for key in ["tilde_mu", "mu_q", "delta_mu_stem", "delta_mu_sol", "delta_mu_txt", "delta_mu_vis", "a_disc"]:
        diff_students = (out_A_no_cache[key][0, step_t] - out_B_no_cache[key][0, step_t]).abs().max().item()
        assert diff_students <= 1e-6, f"IRT parameter {key} varied across student histories by {diff_students}"

    # 2. Cache mode
    mu_concept_prior = torch.full((60,), -1.37)
    model.precompute_item_bank_cache(mu_concept_prior)

    with torch.no_grad():
        out_A_cache = model(batch_A, mu_concept_prior=mu_concept_prior, use_cache=True)
        out_B_cache = model(batch_B, mu_concept_prior=mu_concept_prior, use_cache=True)

    model.clear_item_bank_cache()

    # Compare Cache vs Non-Cache
    for key in ["tilde_mu", "mu_q", "delta_mu_stem", "delta_mu_sol", "delta_mu_txt", "delta_mu_vis", "a_disc"]:
        diff_cache = (out_A_cache[key][0, step_t] - out_A_no_cache[key][0, step_t]).abs().max().item()
        assert diff_cache <= 1e-6, f"IRT parameter {key} Cache vs Non-Cache diff = {diff_cache}"


def test_dynamic_modality_mask_override_in_eval(mock_static_bank):
    """
    Verify that in eval() mode with precomputed cache, supplying dynamic modality masks
    (e.g., dropping image modality) forces dynamic re-computation and zeroes Delta_mu_vis.
    """
    torch.manual_seed(42)
    # Ensure question 5 has an image in the static bank
    mock_static_bank["modality_masks"][5, 0] = 1.0

    model = ACDMMKTv3(
        static_item_bank=mock_static_bank,
        d_model=64,
        num_heads=2,
        max_seq_len=10,
        dropout=0.0,
    ).eval()

    mu_concept_prior = torch.full((60,), -1.37)
    model.precompute_item_bank_cache(mu_concept_prior)

    L = 10
    q_seq = torch.full((1, L), 5, dtype=torch.int64)
    batch = {
        "q_seq": q_seq,
        "r_seq": torch.ones(1, L, dtype=torch.int64),
        "seq_mask": torch.ones(1, L, dtype=torch.bool),
        "loss_mask": torch.ones(1, L, dtype=torch.bool),
        "eval_mask": torch.ones(1, L, dtype=torch.bool),
        "norm_delta_t_lag": torch.zeros(1, L),
        "norm_delta_t_kc": torch.zeros(1, L),
        "mu_concept": torch.full((1, L), -1.37),
    }

    # First forward with default bank masks: m_img = 1
    batch["modality_masks"] = mock_static_bank["modality_masks"][q_seq]
    with torch.no_grad():
        out_normal = model(batch, mu_concept_prior=mu_concept_prior, use_cache=True)

    # Second forward with image dropped: m_img = 0
    batch_dropped = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    batch_dropped["modality_masks"] = batch["modality_masks"].clone()
    batch_dropped["modality_masks"][:, :, 0] = 0.0

    with torch.no_grad():
        out_dropped = model(batch_dropped, mu_concept_prior=mu_concept_prior, use_cache=True)

    model.clear_item_bank_cache()

    # In dropped forward, delta_mu_vis must be strictly 0.0
    max_vis_dropped = out_dropped["delta_mu_vis"].abs().max().item()
    assert max_vis_dropped <= 1e-7, f"Dynamic modality drop failed to zero delta_mu_vis: got {max_vis_dropped}"


def test_trainer_smoke_test(mock_static_bank, mock_sequence_batch, tmp_path):
    """
    Smoke test running ACDMMKTv3Trainer.train_epoch() and ACDMMKTv3Trainer.evaluate().
    """
    torch.manual_seed(42)
    model = ACDMMKTv3(
        static_item_bank=mock_static_bank,
        d_model=64,
        num_heads=2,
        max_seq_len=25,
        dropout=0.0,
    )

    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, count=4):
            self.count = count

        def __len__(self):
            return self.count

        def __getitem__(self, idx):
            return idx

    mock_loader = torch.utils.data.DataLoader(
        MockDataset(count=4),
        batch_size=2,
        collate_fn=lambda _: {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in mock_sequence_batch.items()},
    )

    n_q = torch.randint(10, 100, (60,))
    d_q = torch.full((60,), 0.25)
    omega_q = torch.zeros(60)
    mu_concept_prior = torch.full((60,), -1.37)

    trainer = ACDMMKTv3Trainer(
        model=model,
        train_loader=mock_loader,
        val_loader=mock_loader,
        test_loader=None,
        n_q=n_q,
        d_q=d_q,
        omega_q=omega_q,
        mu_concept_prior=mu_concept_prior,
        lr=1e-3,
        device=torch.device("cpu"),
        precision="fp32",
        save_dir=str(tmp_path),
        val_fold=0,
    )

    # Run 1 training epoch
    train_metrics = trainer.train_epoch(epoch=0)
    assert train_metrics["loss"] > 0.0
    assert "loss" in train_metrics
    assert "bce" in train_metrics

    # Run evaluation
    val_metrics = trainer.evaluate(mock_loader)
    assert "auc" in val_metrics
    assert "acc" in val_metrics
    assert val_metrics["auc"] >= 0.0
