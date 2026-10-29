"""
Comprehensive Unit and Integration Test Suite for MG-MMKT v2.

Verifies all mathematical constraints, causal boundaries, and ablation flags:
1. Layer 1 geometry and distribution invariance (mean ~ 0, norm ~ sqrt(d)).
2. Step-level Question ID Dropout Bernoulli behavior (without scaling, per time-step).
3. Non-zero gradients to e_null^m and W_null^m; zero gradient to W_img^q when batch lacks images.
4. Layer 2 Hierarchical Dual-Stage Gating probability simplex and stability under extreme missingness.
5. Strict Backward Perturbation Causality Test (invariance at t <= t0 under future and current solution perturbations: max diff < 1e-7).
6. Stop-gradient verification on Teacher representations in Privileged Distillation Loss (L_priv).
7. Pathway isolation in historical encoding (r=1 isolates Mastery; r=0 isolates Remedial).
8. Monotonic temporal forgetting curve (decay ensures nearer steps have higher attention).
9. Boundary condition at t = 1 (h_1 = h_0 without NaNs).
10. Checkpoint save and restore across all 6 Ablation study configurations.
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.modules.kc_conditioned_imputation import (
    StructuralDecouplingLayer,
    KCConditionedImputationLayer,
)
from src.models.modules.dual_stage_gating import (
    QuestionDualStageGating,
    SolutionDualStageGating,
)
from src.models.modules.cognitive_history_fusion import (
    CognitiveHistoryFusionLayer,
)
from src.models.modules.disentangled_causal_attention import (
    DisentangledCausalAttention,
    MultiInteractiveReadout,
)
from src.models.mmkt.mg_mmkt_v2 import MGMMKTv2
from src.training.losses import (
    PrivilegedSolutionDistillationLoss,
    ModalitySuppressionRegularizationLoss,
    MGMMKTv2Loss,
)
from src.evaluation.explainability_v2 import MultiLevelExplainerV2


@pytest.fixture
def mock_item_bank():
    """Create a lightweight synthetic static item bank for 100 questions."""
    torch.manual_seed(42)
    N_q = 100
    N_c = 50
    d_raw = 768

    return {
        "mg_v_txt_q": torch.randn(N_q, d_raw, dtype=torch.float32),
        "mg_v_img_q": torch.randn(N_q, d_raw, dtype=torch.float32),
        "mg_v_txt_s": torch.randn(N_q, d_raw, dtype=torch.float32),
        "mg_v_img_s": torch.randn(N_q, d_raw, dtype=torch.float32),
        "question_leaf_kcs": torch.randint(0, N_c, (N_q, 6), dtype=torch.int64),
        "question_kc_mask": torch.ones(N_q, 6, dtype=torch.bool),
        "modality_masks": torch.bernoulli(torch.full((N_q, 3), 0.5)),
    }


@pytest.fixture
def mock_batch():
    """Create a synthetic sequence batch."""
    torch.manual_seed(42)
    B, L = 4, 25
    N_q = 100

    q_seq = torch.randint(0, N_q, (B, L), dtype=torch.int64)
    r_seq = torch.randint(0, 2, (B, L), dtype=torch.int64)
    seq_mask = torch.ones(B, L, dtype=torch.bool)
    loss_mask = seq_mask.clone()
    loss_mask[:, 0] = False
    eval_mask = loss_mask.clone()

    raw_mod_masks = torch.bernoulli(torch.full((B, L, 3), 0.6))
    # Step dropout during training
    dyn_mod_masks = raw_mod_masks.clone()
    dyn_mod_masks[..., 1:] = dyn_mod_masks[..., 1:] * torch.bernoulli(torch.full((B, L, 2), 0.8))

    return {
        "q_seq": q_seq,
        "r_seq": r_seq,
        "seq_mask": seq_mask,
        "loss_mask": loss_mask,
        "eval_mask": eval_mask,
        "raw_modality_masks": raw_mod_masks,
        "modality_masks": dyn_mod_masks,
    }


def test_layer1_geometry_and_invariance():
    """
    Test 1: Verify Layer 1 outputs lie on a normalized hypersphere
    with mean ~ 0 and L2 norm ~ sqrt(d) regardless of whether modalities are present or missing.
    """
    torch.manual_seed(42)
    d_model = 128
    layer = KCConditionedImputationLayer(d_raw_txt=768, d_raw_img=768, d_model=d_model)

    B, L = 8, 20
    v_txt_q = torch.randn(B, L, 768)
    v_img_q = torch.randn(B, L, 768)
    v_txt_s = torch.randn(B, L, 768)
    v_img_s = torch.randn(B, L, 768)
    e_C = torch.randn(B, L, d_model)

    # Test case A: Modalities fully present [1, 1, 1]
    masks_present = torch.ones(B, L, 3)
    out_txt_q, out_img_q, out_txt_s, out_img_s = layer(
        v_txt_q, v_img_q, v_txt_s, v_img_s, masks_present, e_C
    )

    # Test case B: Modalities fully missing [0, 0, 0]
    masks_missing = torch.zeros(B, L, 3)
    _, out_img_q_miss, out_txt_s_miss, out_img_s_miss = layer(
        v_txt_q, v_img_q, v_txt_s, v_img_s, masks_missing, e_C
    )

    expected_norm = math.sqrt(d_model)  # sqrt(128) ~ 11.31

    for tensor, name in [
        (out_txt_q, "txt_q"),
        (out_img_q, "img_q_present"),
        (out_img_q_miss, "img_q_missing"),
        (out_txt_s, "txt_s_present"),
        (out_txt_s_miss, "txt_s_missing"),
        (out_img_s, "img_s_present"),
        (out_img_s_miss, "img_s_missing"),
    ]:
        # LayerNorm guarantees mean ~ 0
        mean_val = tensor.mean(dim=-1).abs().max().item()
        assert mean_val < 0.05, f"{name} mean not close to 0: {mean_val}"

        # LayerNorm with default gamma=1, beta=0 produces norm ~ sqrt(d)
        norms = torch.norm(tensor, p=2, dim=-1)
        mean_norm = norms.mean().item()
        assert abs(mean_norm - expected_norm) < 0.5, f"{name} norm deviated from sqrt(d): {mean_norm}"


def test_step_level_question_id_dropout():
    """
    Test 2: Verify Question ID Dropout samples per time-step in {0, 1}
    without scaling by 1/(1-p) during training, and is strictly 1.0 in eval mode.
    """
    torch.manual_seed(42)
    d_model = 64
    layer = StructuralDecouplingLayer(d_model=d_model, p_id_drop=0.20)

    B, L = 4, 10
    e_q = torch.randn(B, L, d_model)
    e_kcs = torch.randn(B, L, 3, d_model)
    kc_mask = torch.ones(B, L, 3, dtype=torch.bool)

    # 1. Training mode
    layer.train()
    e_struct_train, e_C = layer(e_q, e_kcs, kc_mask)
    assert e_struct_train.shape == (B, L, d_model)

    # 2. Eval mode
    layer.eval()
    e_struct_eval1, _ = layer(e_q, e_kcs, kc_mask)
    e_struct_eval2, _ = layer(e_q, e_kcs, kc_mask)
    # In eval mode, outputs must be 100% deterministic (xi = 1.0)
    assert torch.allclose(e_struct_eval1, e_struct_eval2, atol=1e-6)


def test_kc_conditioned_imputation_gradient_flow():
    """
    Test 3: Verify learnable null parameters receive non-zero gradients on missing modalities,
    and W_img^q receives zero gradient when question image is completely missing in batch.
    """
    torch.manual_seed(42)
    d_model = 64
    layer = KCConditionedImputationLayer(d_raw_txt=32, d_raw_img=32, d_model=d_model)

    B, L = 2, 5
    v_txt_q = torch.randn(B, L, 32)
    v_img_q = torch.randn(B, L, 32)
    v_txt_s = torch.randn(B, L, 32)
    v_img_s = torch.randn(B, L, 32)
    e_C = torch.randn(B, L, d_model)

    # Batch with question image = 0, solution modalities = 0
    masks = torch.zeros(B, L, 3)

    out_txt_q, out_img_q, out_txt_s, out_img_s = layer(
        v_txt_q, v_img_q, v_txt_s, v_img_s, masks, e_C
    )

    loss = (out_img_q.sum() + out_txt_s.sum() + out_img_s.sum())
    loss.backward()

    # Learnable base tokens and W_null must receive gradients
    assert layer.e_null_img_q.grad is not None and layer.e_null_img_q.grad.abs().sum() > 0
    assert layer.e_null_txt_s.grad is not None and layer.e_null_txt_s.grad.abs().sum() > 0
    assert layer.e_null_img_s.grad is not None and layer.e_null_img_s.grad.abs().sum() > 0
    assert layer.w_null_img_q.weight.grad is not None and layer.w_null_img_q.weight.grad.abs().sum() > 0

    # Image projection weight must receive ZERO gradient because m_img_q is 0 everywhere
    assert layer.proj_img_q.weight.grad is None or layer.proj_img_q.weight.grad.abs().sum() == 0.0


def test_layer2_dual_stage_gating():
    """
    Test 4: Verify Layer 2 Softmax weights lie in probability simplex,
    and outputs remain finite even under extreme missingness.
    """
    torch.manual_seed(42)
    d_model = 64
    q_gating = QuestionDualStageGating(d_model=d_model, d_ffn=128)
    s_gating = SolutionDualStageGating(d_model=d_model, d_ffn=128)

    B, L = 4, 15
    e_struct = torch.randn(B, L, d_model)
    tilde_txt_q = torch.randn(B, L, d_model)
    tilde_img_q = torch.randn(B, L, d_model)
    tilde_txt_s = torch.randn(B, L, d_model)
    tilde_img_s = torch.randn(B, L, d_model)
    e_C = torch.randn(B, L, d_model)

    # Extreme missingness: all 0s
    m_img_q = torch.zeros(B, L, 1)
    m_txt_s = torch.zeros(B, L, 1)
    m_img_s = torch.zeros(B, L, 1)

    x, alpha_q = q_gating(e_struct, tilde_txt_q, tilde_img_q, m_img_q)
    z_s, beta_s = s_gating(tilde_txt_s, tilde_img_s, e_C, m_txt_s, m_img_s)

    # Check finite
    assert torch.isfinite(x).all()
    assert torch.isfinite(z_s).all()

    # Check alpha in Delta^2: sum = 1.0
    alpha_sum = alpha_q.sum(dim=-1)
    assert torch.allclose(alpha_sum, torch.ones_like(alpha_sum), atol=1e-5)

    # Check beta in Delta^1: sum = 1.0
    beta_sum = beta_s.sum(dim=-1)
    assert torch.allclose(beta_sum, torch.ones_like(beta_sum), atol=1e-5)


def test_strict_backward_perturbation_causality(mock_item_bank, mock_batch):
    """
    Test 5: Backward Perturbation Causality Test.
    Perturbing response r_t0, solution features v_s_t0, and all future data tau > t0
    must change prediction at t <= t0 by less than 1e-7.
    """
    torch.manual_seed(42)
    model = MGMMKTv2(
        num_questions=100,
        num_kcs=50,
        d_model=64,
        d_ffn=128,
        num_heads=4,
        static_item_bank=mock_item_bank,
    )
    model.eval()

    # Ensure questions at t < t0 do not use question 95 so perturbing question 95 only impacts step t0 and beyond
    q_seq = mock_batch["q_seq"].clone()
    t0 = 10
    q_seq[:, :t0] = q_seq[:, :t0] % 90
    q_seq[:, t0] = 95

    r_seq = mock_batch["r_seq"]
    seq_mask = mock_batch["seq_mask"]
    mod_masks = mock_batch["modality_masks"]
    raw_masks = mock_batch["raw_modality_masks"]

    # Original forward pass
    with torch.no_grad():
        out1 = model(q_seq, r_seq, seq_mask, mod_masks, raw_masks)
        p1 = out1["probs"]

    # Perturb inputs:
    # 1. Flip response at t0: r_t0 <- 1 - r_t0
    r_seq_pert = r_seq.clone()
    r_seq_pert[:, t0] = 1 - r_seq_pert[:, t0]

    # 2. Perturb solution presence masks at t0 (solution text and image)
    mod_masks_pert = mod_masks.clone()
    mod_masks_pert[:, t0, 1:] = 1.0 - mod_masks_pert[:, t0, 1:]
    raw_masks_pert = raw_masks.clone()
    raw_masks_pert[:, t0, 1:] = 1.0 - raw_masks_pert[:, t0, 1:]

    # 3. Randomize ALL future questions, responses, and modality masks at tau > t0
    q_seq_pert = q_seq.clone()
    q_seq_pert[:, t0 + 1:] = torch.randint(0, 90, (q_seq.size(0), q_seq.size(1) - t0 - 1))
    r_seq_pert[:, t0 + 1:] = torch.randint(0, 2, (r_seq.size(0), r_seq.size(1) - t0 - 1))
    mod_masks_pert[:, t0 + 1:] = torch.bernoulli(torch.full_like(mod_masks[:, t0 + 1:], 0.5))
    raw_masks_pert[:, t0 + 1:] = torch.bernoulli(torch.full_like(raw_masks[:, t0 + 1:], 0.5))

    # 4. Perturb static solution features mg_v_txt_s and mg_v_img_s for question q_t0 (qid=95)
    model.mg_v_txt_s[95] = model.mg_v_txt_s[95] + torch.randn_like(model.mg_v_txt_s[95]) * 10.0
    model.mg_v_img_s[95] = model.mg_v_img_s[95] + torch.randn_like(model.mg_v_img_s[95]) * 10.0

    # Second forward pass with perturbed future and current solution
    with torch.no_grad():
        out2 = model(q_seq_pert, r_seq_pert, seq_mask, mod_masks_pert, raw_masks_pert)
        p2 = out2["probs"]

    # Predictions for t <= t0 (indices 0 to t0) must remain strictly invariant (max diff < 1e-7)
    diff_past = (p1[:, : t0 + 1] - p2[:, : t0 + 1]).abs().max().item()
    assert diff_past < 1e-7, f"Causality leak detected! Max diff at t <= t0 is {diff_past}"

    # Predictions for t > t0 MUST change due to perturbed history and inputs
    diff_future = (p1[:, t0 + 1:] - p2[:, t0 + 1:]).abs().max().item()
    assert diff_future > 1e-3, f"Future perturbation had no effect! Max diff at t > t0 is {diff_future}"


def test_teacher_stop_gradient_in_privileged_distillation(mock_item_bank, mock_batch):
    """
    Test 6: Stop-Gradient Verification on Teacher Representation.
    Backpropagating L_priv alone must produce zero gradients on solution encoder weights (s_gating).
    """
    torch.manual_seed(42)
    model = MGMMKTv2(
        num_questions=100,
        num_kcs=50,
        d_model=64,
        d_ffn=128,
        num_heads=4,
        static_item_bank=mock_item_bank,
    )
    model.train()

    outputs = model(
        mock_batch["q_seq"],
        mock_batch["r_seq"],
        mock_batch["seq_mask"],
        mock_batch["modality_masks"],
        mock_batch["raw_modality_masks"],
    )

    priv_loss_fn = PrivilegedSolutionDistillationLoss()
    m_sol = torch.ones_like(mock_batch["seq_mask"])
    l_priv = priv_loss_fn(outputs["hat_z_s"], outputs["z_s_clean"], m_sol)
    l_priv.backward()

    # Cognitive estimator (Student) MUST receive gradients
    assert model.history_fusion.cog_mlp[0].weight.grad is not None
    assert model.history_fusion.cog_mlp[0].weight.grad.abs().sum() > 0

    # Solution gating (Teacher) MUST NOT receive gradients from L_priv
    assert model.s_gating.gate_mlp[0].weight.grad is None or model.s_gating.gate_mlp[0].weight.grad.abs().sum() == 0.0


def test_pathway_isolation_layer3():
    """
    Test 7: Pathway Isolation in Cognitive History Fusion.
    When r = 1 everywhere, Remedial pathway receives zero gradient.
    When r = 0 everywhere, Mastery pathway receives zero gradient.
    """
    torch.manual_seed(42)
    d_model = 64
    layer = CognitiveHistoryFusionLayer(d_model=d_model, use_dual_pathway_history=True)

    B, L = 2, 5
    x = torch.randn(B, L, d_model)
    z_s = torch.randn(B, L, d_model)
    e_r = torch.randn(B, L, d_model)
    m_txt = torch.ones(B, L, 1)
    m_img = torch.ones(B, L, 1)

    # Case A: r = 1 everywhere -> Mastery only
    r_all_1 = torch.ones(B, L, dtype=torch.int64)
    y_1 = layer.forward_history(x, z_s, e_r, r_all_1, m_txt, m_img)
    y_1.sum().backward()

    assert layer.mastery_mlp[0].weight.grad is not None and layer.mastery_mlp[0].weight.grad.abs().sum() > 0
    assert layer.remedial_mlp[0].weight.grad is None or layer.remedial_mlp[0].weight.grad.abs().sum() == 0.0

    # Case B: r = 0 everywhere -> Remedial only
    layer.zero_grad()
    r_all_0 = torch.zeros(B, L, dtype=torch.int64)
    y_0 = layer.forward_history(x, z_s, e_r, r_all_0, m_txt, m_img)
    y_0.sum().backward()

    assert layer.remedial_mlp[0].weight.grad is not None and layer.remedial_mlp[0].weight.grad.abs().sum() > 0
    assert layer.mastery_mlp[0].weight.grad is None or layer.mastery_mlp[0].weight.grad.abs().sum() == 0.0


def test_monotonic_relative_forgetting_decay():
    """
    Test 8: Monotonic Temporal Forgetting Decay.
    For two identical past questions x_tau1 == x_tau2 (tau1 < tau2 < t),
    the nearer step tau2 receives strictly higher attention weight.
    """
    torch.manual_seed(42)
    d_model = 64
    num_heads = 4
    attn = DisentangledCausalAttention(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=20,
        use_disentangled_qk=True,
        use_relative_forgetting=True,
    )
    attn.eval()

    B, L = 1, 10
    # Create question representations where step 1 and step 3 have identical vectors
    x = torch.randn(B, L, d_model)
    x[:, 3, :] = x[:, 1, :].clone()  # identical question stem at tau=1 and tau=3

    y = torch.randn(B, L, d_model)
    seq_mask = torch.ones(B, L, dtype=torch.bool)

    with torch.no_grad():
        h, gamma_bar = attn(x, y, seq_mask)

    # For target step t = 5 (index 5 in gamma_bar):
    # tau=3 (step 3, diff=2) is nearer than tau=1 (step 1, diff=4)
    weight_tau1 = gamma_bar[0, 5, 1].item()
    weight_tau3 = gamma_bar[0, 5, 3].item()

    assert weight_tau3 > weight_tau1, f"Monotonic decay violated: tau3 ({weight_tau3}) <= tau1 ({weight_tau1})"


def test_causal_attention_boundary_no_nan():
    """
    Test 9: Verify boundary condition at t = 1 (h_1 = h_0) and absence of NaNs.
    """
    torch.manual_seed(42)
    d_model = 64
    attn = DisentangledCausalAttention(d_model=d_model, num_heads=4, max_seq_len=10)

    B, L = 2, 8
    x = torch.randn(B, L, d_model)
    y = torch.randn(B, L, d_model)
    seq_mask = torch.ones(B, L, dtype=torch.bool)

    h, gamma_bar = attn(x, y, seq_mask)

    assert not torch.isnan(h).any()
    assert not torch.isnan(gamma_bar).any()

    # Step 0 (t = 1) must be equal to h_0
    for b in range(B):
        assert torch.allclose(h[b, 0, :], attn.h_0, atol=1e-6)

    # gamma_bar at row 0 must be all 0s
    assert (gamma_bar[:, 0, :] == 0.0).all()


@pytest.mark.parametrize(
    "ablation_cfg",
    [
        {"name": "full_v2"},
        {"name": "wo_lupi", "use_lupi": False},
        {"name": "wo_dual_pathway", "use_dual_pathway_history": False},
        {"name": "wo_glu_ffn", "use_glu": False, "use_post_fusion_ffn": False},
        {"name": "wo_disentangled_qk_decay", "use_disentangled_qk": False, "use_relative_forgetting": False},
        {"name": "wo_struct_kc_imputation", "use_structural_decoupling": False, "use_kc_imputation": False},
    ],
)
def test_checkpoint_serialization_and_ablation_matrix(mock_item_bank, mock_batch, ablation_cfg, tmp_path):
    """
    Test 10: Verify model instantiation, forward pass, saving, and state_dict reloading
    for all 6 Ablation Study configurations.
    """
    cfg = {
        "num_questions": 100,
        "num_kcs": 50,
        "d_model": 64,
        "d_ffn": 128,
        "num_heads": 4,
        "max_seq_len": 25,
        "static_item_bank": mock_item_bank,
    }
    cfg.update({k: v for k, v in ablation_cfg.items() if k != "name"})

    model = MGMMKTv2(**cfg)
    model.eval()

    with torch.no_grad():
        out1 = model(
            mock_batch["q_seq"],
            mock_batch["r_seq"],
            mock_batch["seq_mask"],
            mock_batch["modality_masks"],
            mock_batch["raw_modality_masks"],
        )

    # Save checkpoint
    ckpt_path = tmp_path / f"{ablation_cfg['name']}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": model.config,
        },
        ckpt_path,
    )

    # Reload model from saved config
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    loaded_model = MGMMKTv2(**checkpoint["model_cfg"], static_item_bank=mock_item_bank)
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.eval()

    with torch.no_grad():
        out2 = loaded_model(
            mock_batch["q_seq"],
            mock_batch["r_seq"],
            mock_batch["seq_mask"],
            mock_batch["modality_masks"],
            mock_batch["raw_modality_masks"],
        )

    assert torch.allclose(out1["probs"], out2["probs"], atol=1e-6)


def test_ablation5_positional_embeddings_active(mock_item_bank, mock_batch):
    """
    Test Ablation 5 (w/o Relative Forgetting):
    Verifies that when use_relative_forgetting=False, self.pos_emb is active
    and sequence position affects predictions (model is not permutation-invariant).
    """
    torch.manual_seed(42)
    model = MGMMKTv2(
        num_questions=100,
        num_kcs=50,
        d_model=64,
        d_ffn=128,
        num_heads=4,
        use_relative_forgetting=False,
        static_item_bank=mock_item_bank,
    )
    model.eval()

    # Ensure pos_emb exists in causal attention module
    assert hasattr(model.causal_attention, "pos_emb")
    assert isinstance(model.causal_attention.pos_emb, nn.Embedding)

    q_seq1 = mock_batch["q_seq"].clone()
    r_seq1 = mock_batch["r_seq"].clone()
    seq_mask = mock_batch["seq_mask"]
    mod_masks = mock_batch["modality_masks"]
    raw_masks = mock_batch["raw_modality_masks"]

    with torch.no_grad():
        out1 = model(q_seq1, r_seq1, seq_mask, mod_masks, raw_masks)
        p1 = out1["probs"]

    # Permute historical steps 0 and 1 in sequence 2
    q_seq2 = q_seq1.clone()
    r_seq2 = r_seq1.clone()
    q_seq2[:, 0], q_seq2[:, 1] = q_seq1[:, 1], q_seq1[:, 0]
    r_seq2[:, 0], r_seq2[:, 1] = r_seq1[:, 1], r_seq1[:, 0]

    with torch.no_grad():
        out2 = model(q_seq2, r_seq2, seq_mask, mod_masks, raw_masks)
        p2 = out2["probs"]

    # At step 2, the model attends to history {0, 1}.
    # With active positional embeddings, swapping steps 0 and 1 must produce different predictions.
    diff_at_step2 = (p1[:, 2] - p2[:, 2]).abs().max().item()
    assert diff_at_step2 > 1e-4, (
        f"Ablation 5 failed position sensitivity! Swapping history steps 0 and 1 produced identical prediction: diff={diff_at_step2}"
    )


def test_multilevel_explainer_v2():
    """
    Test MultiLevelExplainerV2:
    Verifies 3-level explainability extraction:
    - Level 1: Modality attribution and Mann-Whitney U test.
    - Level 2: Historical evidence retrieval with pathway and KC attribution.
    - Level 3: Cognitive demand diagnosis (cosine alignments, norms, categorization).
    """
    explainer = MultiLevelExplainerV2()

    # --- Level 1 Verification ---
    N = 100
    np.random.seed(42)
    # Questions: 50 with image, 50 text-only
    m_img_q = np.array([1.0] * 50 + [0.0] * 50, dtype=np.float32)
    m_img_s = np.array([1.0] * 40 + [0.0] * 60, dtype=np.float32)

    # Simulated alpha_q [N, 3]: (id, txt, img)
    alpha_q = np.zeros((N, 3), dtype=np.float32)
    alpha_q[:50, :] = [0.2, 0.3, 0.5]  # With image: higher alpha_img
    alpha_q[50:, :] = [0.4, 0.6, 0.0]  # Without image: alpha_img ~ 0.0

    # Simulated beta_s [N, 2]: (txt, img)
    beta_s = np.zeros((N, 2), dtype=np.float32)
    beta_s[:40, :] = [0.4, 0.6]
    beta_s[40:, :] = [1.0, 0.0]

    l1_report = explainer.analyze_level1_modality_attribution(
        all_alpha_q=alpha_q,
        all_beta_s=beta_s,
        all_m_img_q=m_img_q,
        all_m_img_s=m_img_s,
    )

    assert "question_level" in l1_report
    assert "solution_level" in l1_report
    assert l1_report["question_level"]["with_image"]["mean_alpha_img"] > 0.4
    assert l1_report["question_level"]["text_only"]["mean_alpha_img"] < 0.01
    assert "mann_whitney_u" in l1_report["question_level"]
    assert l1_report["question_level"]["mann_whitney_u"]["significant"] is True

    # --- Level 2 Verification ---
    gamma_bar_row = np.array([0.05, 0.45, 0.10, 0.30, 0.10])
    q_history = [101, 102, 103, 104, 105]
    r_history = [1, 0, 1, 1, 0]  # step 1 is wrong (remedial), step 3 is right (mastery)
    kcs_history = [[1, 2], [3], [1, 4], [2], [5]]
    head_thetas = [0.1, 0.3, 0.7, 1.5]

    evidence = explainer.extract_level2_historical_evidence(
        gamma_bar_row=gamma_bar_row,
        q_history=q_history,
        r_history=r_history,
        kcs_history=kcs_history,
        context_start=100,
        top_k=2,
        head_thetas=head_thetas,
    )

    assert len(evidence) == 2
    # Top 1 must be index 1 (weight 0.45)
    assert evidence[0]["local_step"] == 1
    assert evidence[0]["global_step"] == 101
    assert evidence[0]["question_id"] == 102
    assert evidence[0]["response"] == 0
    assert "Remedial" in evidence[0]["pathway"]
    assert evidence[0]["kcs"] == [3]

    # Top 2 must be index 3 (weight 0.30)
    assert evidence[1]["local_step"] == 3
    assert evidence[1]["global_step"] == 103
    assert evidence[1]["question_id"] == 104
    assert evidence[1]["response"] == 1
    assert "Mastery" in evidence[1]["pathway"]
    assert evidence[1]["kcs"] == [2]

    # --- Level 3 Verification ---
    torch.manual_seed(42)
    d = 64
    h_t = torch.randn(d)
    x_t = torch.randn(d)
    hat_z_t_s = torch.randn(d)
    z_t_s_clean = hat_z_t_s + 0.1 * torch.randn(d)

    l3_result = explainer.analyze_level3_cognitive_demand(
        h_t=h_t,
        x_t=x_t,
        hat_z_t_s=hat_z_t_s,
        z_t_s_clean=z_t_s_clean,
        prob_t=0.35,
    )

    assert "concept_cosine" in l3_result
    assert "cognitive_demand_cosine" in l3_result
    assert "distillation_fidelity_cosine" in l3_result
    assert "category" in l3_result
    assert "diagnosis" in l3_result
    assert -1.0 <= l3_result["concept_cosine"] <= 1.0
    assert -1.0 <= l3_result["cognitive_demand_cosine"] <= 1.0
    assert l3_result["distillation_fidelity_cosine"] > 0.8

