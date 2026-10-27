"""
Unit and Integration Tests for MG-MMKT Architecture.

Verifies:
1. Architectural output shapes across Layer 1, Layer 2, Layer 3, and End-to-End MGMMKT.
2. Strict Causal Invariance (Backward Perturbation Causality Test): max diff < 1e-6.
3. Non-zero gradient flow to learnable null tokens (e_null_img_q, e_null_txt_s, e_null_img_s).
4. Extreme missingness stability (zero-flags for all modalities without NaN/Inf).
5. Single-batch optimization sanity check (rapid loss reduction).
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from src.models.modules.modality_imputation import ModalityImputationLayer
from src.models.modules.gated_fusion import (
    QuestionMaskedGatedFusion,
    SolutionMaskedGatedFusion,
)
from src.models.modules.causal_attention import CausalSAKTLayer
from src.models.mmkt.mg_mmkt import MGMMKT
from src.models.baselines.sakt import IDOnlySAKT
from src.training.losses import MaskedBCELoss


@pytest.fixture
def mock_batch():
    """Create a synthetic mock batch with realistic dimensions."""
    torch.manual_seed(42)
    B, L = 4, 30
    num_q = 7652
    num_k = 1175
    d_raw = 768

    q_seq = torch.randint(0, num_q, (B, L), dtype=torch.int64)
    r_seq = torch.randint(0, 2, (B, L), dtype=torch.int64)
    seq_mask = torch.ones((B, L), dtype=torch.bool)
    loss_mask = seq_mask.clone()
    loss_mask[:, 0] = False
    eval_mask = loss_mask.clone()

    # Modality masks: [B, L, 3] with mixed presence
    modality_masks = torch.bernoulli(torch.full((B, L, 3), 0.5))

    # Leaf KCs
    leaf_kcs = torch.randint(0, num_k, (B, L, 6), dtype=torch.int64)
    leaf_kc_mask = torch.ones((B, L, 6), dtype=torch.bool)

    # Static raw features
    v_txt_q = torch.randn(B, L, d_raw)
    v_img_q = torch.randn(B, L, d_raw)
    v_txt_s = torch.randn(B, L, d_raw)
    v_img_s = torch.randn(B, L, d_raw)

    return {
        "q_seq": q_seq,
        "r_seq": r_seq,
        "seq_mask": seq_mask,
        "loss_mask": loss_mask,
        "eval_mask": eval_mask,
        "modality_masks": modality_masks,
        "leaf_kcs": leaf_kcs,
        "leaf_kc_mask": leaf_kc_mask,
        "v_txt_q": v_txt_q,
        "v_img_q": v_img_q,
        "v_txt_s": v_txt_s,
        "v_img_s": v_img_s,
    }


def test_layer1_modality_imputation(mock_batch):
    """Verify Layer 1 projections, LayerNorm, and learnable token imputation shapes."""
    d_model = 64
    layer1 = ModalityImputationLayer(
        d_raw_txt=768,
        d_raw_img=768,
        d_model=d_model,
        use_learnable_imputation=True,
    )

    tilde_txt_q, tilde_img_q, tilde_txt_s, tilde_img_s = layer1(
        v_txt_q=mock_batch["v_txt_q"],
        v_img_q=mock_batch["v_img_q"],
        v_txt_s=mock_batch["v_txt_s"],
        v_img_s=mock_batch["v_img_s"],
        modality_masks=mock_batch["modality_masks"],
    )

    B, L = mock_batch["q_seq"].shape
    assert tilde_txt_q.shape == (B, L, d_model)
    assert tilde_img_q.shape == (B, L, d_model)
    assert tilde_txt_s.shape == (B, L, d_model)
    assert tilde_img_s.shape == (B, L, d_model)


def test_layer2_gated_fusion(mock_batch):
    """Verify Layer 2 gating weights sum to 1.0 and output fused representations."""
    B, L = mock_batch["q_seq"].shape
    d_model = 64

    q_fusion = QuestionMaskedGatedFusion(d_model=d_model, use_visual=True, use_gated_fusion=True)
    s_fusion = SolutionMaskedGatedFusion(d_model=d_model, use_visual=True, use_gated_fusion=True)

    e_q = torch.randn(B, L, d_model)
    e_c = torch.randn(B, L, d_model)
    tilde_txt_q = torch.randn(B, L, d_model)
    tilde_img_q = torch.randn(B, L, d_model)
    tilde_txt_s = torch.randn(B, L, d_model)
    tilde_img_s = torch.randn(B, L, d_model)

    m_img_q = mock_batch["modality_masks"][:, :, 0:1]
    m_txt_s = mock_batch["modality_masks"][:, :, 1:2]
    m_img_s = mock_batch["modality_masks"][:, :, 2:3]

    x_q, alpha_q = q_fusion(e_q, e_c, tilde_txt_q, tilde_img_q, m_img_q)
    z_s, beta_s = s_fusion(tilde_txt_s, tilde_img_s, m_txt_s, m_img_s)

    assert x_q.shape == (B, L, d_model)
    assert z_s.shape == (B, L, d_model)
    assert alpha_q.shape == (B, L, 3)
    assert beta_s.shape == (B, L, 2)

    # Weights must sum to 1.0 along the modality dimension
    torch.testing.assert_close(alpha_q.sum(dim=-1), torch.ones(B, L), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(beta_s.sum(dim=-1), torch.ones(B, L), atol=1e-5, rtol=1e-5)


def test_layer3_causal_sakt(mock_batch):
    """Verify Layer 3 SAKT prediction outputs and causal mask bounds."""
    B, L = mock_batch["q_seq"].shape
    d_model = 64

    sakt = CausalSAKTLayer(
        d_model=d_model,
        num_heads=4,
        max_seq_len=50,
        dropout=0.1,
        use_solution=True,
    )

    x_q = torch.randn(B, L, d_model)
    e_r = torch.randn(B, L, d_model)
    z_s = torch.randn(B, L, d_model)
    m_txt_s = mock_batch["modality_masks"][:, :, 1:2]

    logits, preds, attn_weights = sakt(x_q, e_r, z_s, m_txt_s)

    assert logits.shape == (B, L)
    assert preds.shape == (B, L)
    assert attn_weights.shape == (B, L, L)
    assert (preds >= 0.0).all() and (preds <= 1.0).all()


def test_strict_causal_invariance(mock_batch):
    """
    CRITICAL TEST: Backward Perturbation Causality Test.
    Perturbing response r_{t0}, solution s_{t0}, or future data tau > t0 MUST NOT change predictions at t <= t0.
    """
    model = MGMMKT(
        num_questions=7652,
        num_kcs=1175,
        d_model=64,
        num_heads=4,
        max_seq_len=50,
        dropout=0.0,
    )
    model.eval()

    # Pass 1: Original batch
    with torch.no_grad():
        out1 = model(
            batch_or_q_seq=mock_batch["q_seq"],
            r_seq=mock_batch["r_seq"],
            modality_masks=mock_batch["modality_masks"],
            leaf_kcs=mock_batch["leaf_kcs"],
            leaf_kc_mask=mock_batch["leaf_kc_mask"],
            v_txt_q=mock_batch["v_txt_q"],
            v_img_q=mock_batch["v_img_q"],
            v_txt_s=mock_batch["v_txt_s"],
            v_img_s=mock_batch["v_img_s"],
        )
        preds1 = out1["preds"]

    # Pick test pivot step t0
    t0 = 15
    B, L = mock_batch["q_seq"].shape

    # Create fully perturbed copy of inputs
    pert_r = mock_batch["r_seq"].clone()
    pert_q = mock_batch["q_seq"].clone()
    pert_txt_q = mock_batch["v_txt_q"].clone()
    pert_img_q = mock_batch["v_img_q"].clone()
    pert_txt_s = mock_batch["v_txt_s"].clone()
    pert_img_s = mock_batch["v_img_s"].clone()
    pert_mod = mock_batch["modality_masks"].clone()
    pert_leaf_kcs = mock_batch["leaf_kcs"].clone()
    pert_leaf_kc_mask = mock_batch["leaf_kc_mask"].clone()

    # 1. Perturb response and solution at pivot step t0 (enters y_{t0}, only visible at t > t0)
    pert_r[:, t0] = 1 - pert_r[:, t0]
    pert_txt_s[:, t0] = torch.randn_like(pert_txt_s[:, t0])
    pert_img_s[:, t0] = torch.randn_like(pert_img_s[:, t0])
    pert_mod[:, t0, 1] = 1.0 - pert_mod[:, t0, 1]  # Flip m_txt_s at t0
    pert_mod[:, t0, 2] = 1.0 - pert_mod[:, t0, 2]  # Flip m_img_s at t0

    # 2. Completely randomize all future interactions tau > t0 (Issue L3 / Plan Step 7.2)
    pert_r[:, t0 + 1:] = torch.randint(0, 2, (B, L - t0 - 1))
    pert_q[:, t0 + 1:] = torch.randint(0, 7652, (B, L - t0 - 1))
    pert_txt_q[:, t0 + 1:] = torch.randn_like(pert_txt_q[:, t0 + 1:])
    pert_img_q[:, t0 + 1:] = torch.randn_like(pert_img_q[:, t0 + 1:])
    pert_txt_s[:, t0 + 1:] = torch.randn_like(pert_txt_s[:, t0 + 1:])
    pert_img_s[:, t0 + 1:] = torch.randn_like(pert_img_s[:, t0 + 1:])
    pert_mod[:, t0 + 1:] = torch.randint(0, 2, (B, L - t0 - 1, 3)).float()
    pert_leaf_kcs[:, t0 + 1:] = torch.randint(0, 1175, (B, L - t0 - 1, 6))
    pert_leaf_kc_mask[:, t0 + 1:] = torch.randint(0, 2, (B, L - t0 - 1, 6)).bool()

    # Pass 2: Forward on perturbed input
    with torch.no_grad():
        out2 = model(
            batch_or_q_seq=pert_q,
            r_seq=pert_r,
            modality_masks=pert_mod,
            leaf_kcs=pert_leaf_kcs,
            leaf_kc_mask=pert_leaf_kc_mask,
            v_txt_q=pert_txt_q,
            v_img_q=pert_img_q,
            v_txt_s=pert_txt_s,
            v_img_s=pert_img_s,
        )
        preds2 = out2["preds"]

    # Invariance check: max absolute difference up to t0 must satisfy strict Checklist Criterion 3 (< 1e-7)
    diff = torch.abs(preds1[:, :t0 + 1] - preds2[:, :t0 + 1])
    max_diff = diff.max().item()
    assert max_diff < 1e-7, f"Causal leakage detected! Max diff at t <= t0: {max_diff:.2e} (Threshold: 1e-7)"


def test_null_modality_gradient_flow_realistic(mock_batch):
    """
    CRITICAL TEST: Gradient Flow to Learnable e_null Tokens Under Realistic Conditions (Issue L4 / Plan 7.2).
    Verifies that ||grad(e_null)||_2 > 0 under natural modality dropout and mixed image presence.
    """
    model = MGMMKT(
        num_questions=7652,
        num_kcs=1175,
        d_model=64,
        num_heads=4,
        max_seq_len=50,
        dropout=0.0,
    )
    model.train()

    # Simulate realistic batch:
    # On real data, m_txt_s is 1.0 for all items, but training-time modality dropout
    # randomly zeroes out ~20% of solution masks and question image masks
    B, L = mock_batch["q_seq"].shape
    torch.manual_seed(123)
    mod_masks = torch.ones((B, L, 3))
    # ~70% missing question image
    mod_masks[:, :, 0] = torch.bernoulli(torch.full((B, L), 0.30))
    # Modality dropout on solution text (~20% dropped to 0)
    mod_masks[:, :, 1] = torch.bernoulli(torch.full((B, L), 0.80))
    # ~85% missing solution image
    mod_masks[:, :, 2] = torch.bernoulli(torch.full((B, L), 0.15))

    outputs = model(
        batch_or_q_seq=mock_batch["q_seq"],
        r_seq=mock_batch["r_seq"],
        modality_masks=mod_masks,
        leaf_kcs=mock_batch["leaf_kcs"],
        leaf_kc_mask=mock_batch["leaf_kc_mask"],
        v_txt_q=mock_batch["v_txt_q"],
        v_img_q=mock_batch["v_img_q"],
        v_txt_s=mock_batch["v_txt_s"],
        v_img_s=mock_batch["v_img_s"],
    )

    loss_fn = MaskedBCELoss()
    loss = loss_fn(outputs["logits"], mock_batch["r_seq"], mock_batch["loss_mask"])
    loss.backward()

    # Check gradients of all three learnable null tokens
    grad_img_q = model.imputation.e_null_img_q.grad
    grad_txt_s = model.imputation.e_null_txt_s.grad
    grad_img_s = model.imputation.e_null_img_s.grad

    assert grad_img_q is not None and torch.norm(grad_img_q).item() > 0.0, "Zero gradient on e_null_img_q!"
    assert grad_txt_s is not None and torch.norm(grad_txt_s).item() > 0.0, "Zero gradient on e_null_txt_s under realistic dropout!"
    assert grad_img_s is not None and torch.norm(grad_img_s).item() > 0.0, "Zero gradient on e_null_img_s!"


def test_null_modality_gradient_flow_with_dataset():
    """
    CRITICAL INTEGRATION TEST: Gradient Flow to Learnable e_null Tokens Using XES3G5MDataset (Issue L4).
    Validates end-to-end that XES3G5MDataset(is_train=True, p_drop=0.20) dynamically generates
    modality dropout masks for solution text (m_txt_s = 0 on ~20% of steps) and natural missingness
    for images, allowing loss.backward() to compute non-zero gradients on all e_null parameters.
    """
    import os
    from torch.utils.data import DataLoader
    from src.data.xes3g5m_dataset import (
        XES3G5MDataset,
        collate_xes3g5m,
        load_static_item_bank_optimized,
    )

    data_dir = "data/processed/XES3G5M"
    seq_file = os.path.join(data_dir, "sequences_fold_0.pt")
    if not (os.path.exists(data_dir) and os.path.exists(seq_file)):
        pytest.skip(f"Processed dataset not found in {data_dir}")

    static_item_bank = load_static_item_bank_optimized(data_dir)
    train_dataset = XES3G5MDataset(
        static_item_bank=static_item_bank,
        sequences=seq_file,
        is_train=True,
        p_drop=0.20,
    )

    torch.manual_seed(42)
    loader = DataLoader(train_dataset, batch_size=8, shuffle=False, collate_fn=collate_xes3g5m)
    batch = next(iter(loader))

    # Verify that dynamic dropout dropped at least one solution text token in active steps
    loss_mask = batch["loss_mask"]
    m_txt_s = batch["modality_masks"][:, :, 1]
    dropped_txt_s_active = ((m_txt_s == 0.0) & loss_mask).sum().item()
    assert dropped_txt_s_active > 0, (
        f"Dynamic modality dropout failed to drop solution text on active steps! Count: {dropped_txt_s_active}"
    )

    # Initialize model with item bank
    model = MGMMKT(
        num_questions=7652,
        num_kcs=1175,
        d_model=64,
        num_heads=4,
        max_seq_len=200,
        dropout=0.0,
        static_item_bank=static_item_bank,
    )
    model.train()

    outputs = model(batch)
    loss_fn = MaskedBCELoss()
    loss = loss_fn(outputs["logits"], batch["r_seq"], batch["loss_mask"])
    loss.backward()

    # Verify gradients of all learnable null tokens
    grad_img_q = model.imputation.e_null_img_q.grad
    grad_txt_s = model.imputation.e_null_txt_s.grad
    grad_img_s = model.imputation.e_null_img_s.grad

    assert grad_img_q is not None and torch.norm(grad_img_q).item() > 0.0, (
        "Zero gradient on e_null_img_q with real XES3G5MDataset!"
    )
    assert grad_txt_s is not None and torch.norm(grad_txt_s).item() > 0.0, (
        "Zero gradient on e_null_txt_s with real XES3G5MDataset under dynamic p_drop=0.20!"
    )
    assert grad_img_s is not None and torch.norm(grad_img_s).item() > 0.0, (
        "Zero gradient on e_null_img_s with real XES3G5MDataset!"
    )


def test_extreme_missingness_stress(mock_batch):
    """
    CRITICAL TEST: Extreme Missingness Stability.
    When all modalities are absent (m = 0), forward and backward pass must NOT produce NaN or Inf.
    """
    model = MGMMKT(
        num_questions=7652,
        num_kcs=1175,
        d_model=64,
        num_heads=4,
        max_seq_len=50,
        dropout=0.0,
    )
    model.train()

    # Zero all modality presence flags
    zero_masks = torch.zeros_like(mock_batch["modality_masks"])

    outputs = model(
        batch_or_q_seq=mock_batch["q_seq"],
        r_seq=mock_batch["r_seq"],
        modality_masks=zero_masks,
        leaf_kcs=mock_batch["leaf_kcs"],
        leaf_kc_mask=mock_batch["leaf_kc_mask"],
        v_txt_q=mock_batch["v_txt_q"],
        v_img_q=mock_batch["v_img_q"],
        v_txt_s=mock_batch["v_txt_s"],
        v_img_s=mock_batch["v_img_s"],
    )

    logits = outputs["logits"]
    assert not torch.isnan(logits).any(), "NaN found in logits under extreme missingness!"
    assert not torch.isinf(logits).any(), "Inf found in logits under extreme missingness!"

    loss_fn = MaskedBCELoss()
    loss = loss_fn(logits, mock_batch["r_seq"], mock_batch["loss_mask"])
    assert not torch.isnan(loss).any(), "NaN found in loss under extreme missingness!"

    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN in gradient of {name} under extreme missingness!"


def test_single_batch_overfit_sanity(mock_batch):
    """
    Sanity Check: Model must rapidly reduce BCE loss on a single mini-batch.
    """
    model = MGMMKT(
        num_questions=7652,
        num_kcs=1175,
        d_model=64,
        num_heads=4,
        max_seq_len=50,
        dropout=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss_fn = MaskedBCELoss()

    initial_loss = None
    final_loss = None

    for step in range(25):
        optimizer.zero_grad()
        outputs = model(
            batch_or_q_seq=mock_batch["q_seq"],
            r_seq=mock_batch["r_seq"],
            modality_masks=mock_batch["modality_masks"],
            leaf_kcs=mock_batch["leaf_kcs"],
            leaf_kc_mask=mock_batch["leaf_kc_mask"],
            v_txt_q=mock_batch["v_txt_q"],
            v_img_q=mock_batch["v_img_q"],
            v_txt_s=mock_batch["v_txt_s"],
            v_img_s=mock_batch["v_img_s"],
        )
        loss = loss_fn(outputs["logits"], mock_batch["r_seq"], mock_batch["loss_mask"])
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert final_loss < initial_loss * 0.5, f"Model failed to overfit single batch! Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"


def test_ablation_configurations(mock_batch):
    """
    CRITICAL TEST: Forward and Backward pass across all Ablation Configurations (Issue L5 / Plan Step 7.1).
    - Config 1: Full MG-MMKT
    - Config 2: w/o Solution Modalities
    - Config 3: w/o Visual Modalities
    - Config 4: Naive Concat with Zero-Imputation
    """
    loss_fn = MaskedBCELoss()
    B, L = mock_batch["q_seq"].shape

    configs = [
        # Config 1: Full MG-MMKT
        {"use_solution": True, "use_visual": True, "use_learnable_imputation": True, "use_gated_fusion": True, "fusion_mode": "gated"},
        # Config 2: w/o Solution Modalities
        {"use_solution": False, "use_visual": True, "use_learnable_imputation": True, "use_gated_fusion": True, "fusion_mode": "gated"},
        # Config 3: w/o Visual Modalities
        {"use_solution": True, "use_visual": False, "use_learnable_imputation": True, "use_gated_fusion": True, "fusion_mode": "gated"},
        # Config 4: Zero Imputation + Naive Concat
        {"use_solution": True, "use_visual": True, "use_learnable_imputation": False, "use_gated_fusion": False, "fusion_mode": "naive_concat"},
    ]

    for cfg_idx, cfg in enumerate(configs, start=1):
        model = MGMMKT(
            num_questions=7652,
            num_kcs=1175,
            d_model=64,
            num_heads=4,
            max_seq_len=50,
            dropout=0.1,
            **cfg,
        )
        model.train()

        outputs = model(
            batch_or_q_seq=mock_batch["q_seq"],
            r_seq=mock_batch["r_seq"],
            modality_masks=mock_batch["modality_masks"],
            leaf_kcs=mock_batch["leaf_kcs"],
            leaf_kc_mask=mock_batch["leaf_kc_mask"],
            v_txt_q=mock_batch["v_txt_q"],
            v_img_q=mock_batch["v_img_q"],
            v_txt_s=mock_batch["v_txt_s"],
            v_img_s=mock_batch["v_img_s"],
        )

        logits = outputs["logits"]
        preds = outputs["preds"]
        assert logits.shape == (B, L), f"Config {cfg_idx} logits shape mismatch!"
        assert preds.shape == (B, L), f"Config {cfg_idx} preds shape mismatch!"
        assert not torch.isnan(logits).any(), f"Config {cfg_idx} produced NaNs in logits!"
        assert not torch.isinf(logits).any(), f"Config {cfg_idx} produced Infs in logits!"

        # Backward pass check
        loss = loss_fn(logits, mock_batch["r_seq"], mock_batch["loss_mask"])
        assert not torch.isnan(loss).any(), f"Config {cfg_idx} produced NaN loss!"
        loss.backward()

        # Check gradients exist on active parameters
        trainable_grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert len(trainable_grads) > 0, f"Config {cfg_idx} failed backward gradient accumulation!"


def test_id_only_sakt_baseline(mock_batch):
    """
    CRITICAL TEST: ID-Only SAKT Baseline Model (Issue M5 / Checklist Item 4).
    Verifies that the non-multimodal baseline runs forward and backward cleanly.
    """
    B, L = mock_batch["q_seq"].shape
    model = IDOnlySAKT(
        num_questions=7652,
        num_kcs=1175,
        d_model=64,
        num_heads=4,
        max_seq_len=50,
        dropout=0.1,
    )
    model.train()

    outputs = model(
        batch_or_q_seq=mock_batch["q_seq"],
        r_seq=mock_batch["r_seq"],
        leaf_kcs=mock_batch["leaf_kcs"],
        leaf_kc_mask=mock_batch["leaf_kc_mask"],
    )

    logits = outputs["logits"]
    preds = outputs["preds"]
    attn_weights = outputs["attn_weights"]

    assert logits.shape == (B, L)
    assert preds.shape == (B, L)
    assert attn_weights.shape == (B, L, L)
    assert not torch.isnan(logits).any()
    assert (preds >= 0.0).all() and (preds <= 1.0).all()

    loss_fn = MaskedBCELoss()
    loss = loss_fn(logits, mock_batch["r_seq"], mock_batch["loss_mask"])
    loss.backward()

    assert model.q_emb.weight.grad is not None and torch.norm(model.q_emb.weight.grad).item() > 0.0
    assert model.y_0.grad is not None and torch.norm(model.y_0.grad).item() > 0.0

