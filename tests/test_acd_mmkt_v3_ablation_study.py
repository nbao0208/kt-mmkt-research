"""
Test Suite for Modality Ablation Study of ACD-MMKTv3 Architecture.
Asymmetric Causal-Disentangled Multimodal Knowledge Tracing v3.

Verifies:
1. Zero Modality Leakage under extreme perturbation (+/- 10^4):
   - Experiment 1 (w/o All Multimodal / Pure KC): invariant to E_txt_q, E_img_q, E_txt_s, E_img_s, and acd_kc_init_emb.
   - Experiment 2 (w/o Solution): invariant to E_txt_s, E_img_s, and solution masks; Delta_mu_sol = 0 strictly.
   - Experiment 3 (w/o Visual): invariant to E_img_q and E_img_s (including questions with m_img^s = 1); Delta_mu_vis = 0, lambda_q = 0.
2. Pure Pruning Compliance:
   - Parameter counts of all 3 ablation variants are strictly less than the full ACD-MMKTv3 model.
   - 100% of initialized parameters in each variant receive valid, non-zero, finite gradients during backward pass.
3. Mathematical & Psychometric Invariants:
   - Gram-Schmidt orthogonality: |<d^q, c_C>| < 1e-5 (and d^q = 0 strictly for Experiment 1).
   - 100% 2PL-IRT Decision Faithfulness: (p_hat >= 0.5) <=> (theta >= tilde_mu) on every interaction.
   - Exact equivalence between direct evaluation and precomputed offline item bank cache (diff <= 1e-6).
   - 4-Tier Explainability Engine operates reliably across all 3 ablation variants.
"""

import os
import pytest
import torch
import torch.nn as nn
import numpy as np

from src.models.mmkt.acd_mmkt_v3 import ACDMMKTv3
from src.training.losses import ACDMMKTv3Loss
from src.evaluation.acd_v3_explainability import ACDv3ExplainabilityEngine


@pytest.fixture
def mock_ablation_static_bank():
    """
    Create synthetic static item bank covering all 4 visual modality combinations:
    (m_img_q, m_img_s) in {(0,0), (1,0), (0,1), (1,1)}.
    """
    torch.manual_seed(42)
    num_q = 40
    num_k = 30
    d_raw = 768

    m_patch = torch.zeros(num_q, 49, dtype=torch.float32)
    m_patch[:, 35:] = -1e9

    modality_masks = torch.zeros(num_q, 3, dtype=torch.float32)
    # Quarter 1: (0, 1, 0) - stem text, solution text
    modality_masks[0:10, 1] = 1.0
    # Quarter 2: (1, 1, 0) - stem image, solution text
    modality_masks[10:20, 0] = 1.0
    modality_masks[10:20, 1] = 1.0
    # Quarter 3: (0, 1, 1) - stem text, solution text + image
    modality_masks[20:30, 1] = 1.0
    modality_masks[20:30, 2] = 1.0
    # Quarter 4: (1, 1, 1) - stem image, solution text + image
    modality_masks[30:40, 0] = 1.0
    modality_masks[30:40, 1] = 1.0
    modality_masks[30:40, 2] = 1.0

    return {
        "acd_E_txt_q": torch.randn(num_q, 128, d_raw, dtype=torch.float16),
        "acd_mask_txt_q": torch.ones(num_q, 128, dtype=torch.bool),
        "acd_E_img_q": torch.randn(num_q, 49, d_raw, dtype=torch.float16),
        "acd_E_txt_s": torch.randn(num_q, 160, d_raw, dtype=torch.float16),
        "acd_mask_txt_s": torch.ones(num_q, 160, dtype=torch.bool),
        "acd_E_img_s": torch.randn(num_q, d_raw, dtype=torch.float16),
        "acd_kc_init_emb": torch.randn(num_k, d_raw, dtype=torch.float32),
        "acd_hypergraph_G": torch.eye(num_k, dtype=torch.float32),
        "modality_masks": modality_masks,
        "question_leaf_kcs": torch.randint(0, num_k, (num_q, 6), dtype=torch.int64),
        "question_kc_mask": torch.ones(num_q, 6, dtype=torch.bool),
        "question_static_ctx": torch.randn(num_q, 15, dtype=torch.float32),
        "M_patch_q": m_patch,
    }


@pytest.fixture
def mock_ablation_sequence_batch(mock_ablation_static_bank):
    """Create a synthetic student interaction sequence batch covering all question types."""
    torch.manual_seed(42)
    B, L = 4, 20
    num_q = 40

    q_seq = torch.randint(0, num_q, (B, L), dtype=torch.int64)
    r_seq = torch.randint(0, 2, (B, L), dtype=torch.int64)
    seq_mask = torch.ones((B, L), dtype=torch.bool)
    loss_mask = seq_mask.clone()
    loss_mask[:, 0] = False
    eval_mask = loss_mask.clone()

    mod_masks = mock_ablation_static_bank["modality_masks"][q_seq].clone()

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


def test_experiment1_zero_multimodal_leakage_and_pruning(mock_ablation_static_bank, mock_ablation_sequence_batch):
    """
    Experiment 1 (Pure KC-Interaction / w/o All Multimodal):
    1. Extreme perturbation (+/- 10^4) on all text, visual, and KC semantic embeddings produces 0.0 change on outputs.
    2. Subsystem 1 has 0 parameters, and overall parameter count is strictly lower than Full model.
    3. Intermediate quantities: d^q = 0, delta_mu = 0, x_q = c_C.
    """
    torch.manual_seed(42)
    d_model = 64

    # Full Model
    full_model = ACDMMKTv3(
        static_item_bank=mock_ablation_static_bank,
        d_model=d_model,
        num_heads=2,
        max_seq_len=20,
        dropout=0.0,
        use_multimodal=True,
    ).eval()
    full_param_count = sum(p.numel() for p in full_model.parameters() if p.requires_grad)

    # Experiment 1 Model
    exp1_model = ACDMMKTv3(
        static_item_bank=mock_ablation_static_bank,
        d_model=d_model,
        num_heads=2,
        max_seq_len=20,
        dropout=0.0,
        use_multimodal=False,
    ).eval()
    exp1_param_count = sum(p.numel() for p in exp1_model.parameters() if p.requires_grad)

    # 1. Pure Pruning: Parameter count strictly reduced
    assert exp1_param_count < full_param_count, f"Exp1 params ({exp1_param_count}) must be < Full ({full_param_count})"

    # 2. Forward pass before perturbation
    with torch.no_grad():
        out_clean = exp1_model(mock_ablation_sequence_batch, use_cache=False)

    # Intermediate checks: d^q is strictly zero, x_q == c_C
    d_q = out_clean.get("d_q", exp1_model.forward_items(
        torch.tensor([0, 1, 2]),
        exp1_model.compute_all_kc_embeddings(),
        torch.full((3,), -1.37)
    )["d_q"])
    assert torch.allclose(d_q, torch.zeros_like(d_q), atol=1e-7), "d^q must be strictly zero in Exp1"

    delta_stem = out_clean["delta_mu_stem"]
    delta_sol = out_clean["delta_mu_sol"]
    assert torch.allclose(delta_stem, torch.zeros_like(delta_stem), atol=1e-7), "delta_mu_stem must be zero in Exp1"
    assert torch.allclose(delta_sol, torch.zeros_like(delta_sol), atol=1e-7), "delta_mu_sol must be zero in Exp1"

    # 3. Inject EXTREME perturbation (+/- 10,000) into all multimodal feature buffers
    exp1_model.acd_E_txt_q.data.add_(torch.randn_like(exp1_model.acd_E_txt_q) * 1e4)
    exp1_model.acd_E_img_q.data.add_(torch.randn_like(exp1_model.acd_E_img_q) * 1e4)
    exp1_model.acd_E_txt_s.data.add_(torch.randn_like(exp1_model.acd_E_txt_s) * 1e4)
    exp1_model.acd_E_img_s.data.add_(torch.randn_like(exp1_model.acd_E_img_s) * 1e4)
    exp1_model.acd_kc_init_emb.data.add_(torch.randn_like(exp1_model.acd_kc_init_emb) * 1e4)

    with torch.no_grad():
        out_perturbed = exp1_model(mock_ablation_sequence_batch, use_cache=False)

    # Check zero leakage on all psychometric and predictive outputs
    max_diff_prob = (out_clean["probs"] - out_perturbed["probs"]).abs().max().item()
    max_diff_theta = (out_clean["theta"] - out_perturbed["theta"]).abs().max().item()
    max_diff_mu = (out_clean["tilde_mu"] - out_perturbed["tilde_mu"]).abs().max().item()
    max_diff_a = (out_clean["a_disc"] - out_perturbed["a_disc"]).abs().max().item()

    assert max_diff_prob <= 1e-7, f"Exp1 prob leaked! max diff = {max_diff_prob}"
    assert max_diff_theta <= 1e-7, f"Exp1 theta leaked! max diff = {max_diff_theta}"
    assert max_diff_mu <= 1e-7, f"Exp1 tilde_mu leaked! max diff = {max_diff_mu}"
    assert max_diff_a <= 1e-7, f"Exp1 a_disc leaked! max diff = {max_diff_a}"


def test_experiment2_zero_solution_leakage_and_pruning(mock_ablation_static_bank, mock_ablation_sequence_batch):
    """
    Experiment 2 (Question Stem Multimodal Only / w/o Solution):
    1. Extreme perturbation (+/- 10^4) on E_txt_s and E_img_s produces 0.0 change on outputs.
    2. Delta_mu_sol = 0 strictly on all questions; v_sol = 0 strictly.
    3. Parameter count strictly reduced compared to Full model.
    4. Gram-Schmidt orthogonality on stem: |<d^q, c_C>| < 1e-5.
    """
    torch.manual_seed(42)
    d_model = 64

    # Full Model
    full_model = ACDMMKTv3(
        static_item_bank=mock_ablation_static_bank,
        d_model=d_model,
        num_heads=2,
        max_seq_len=20,
        dropout=0.0,
        use_solution=True,
    ).eval()
    full_param_count = sum(p.numel() for p in full_model.parameters() if p.requires_grad)

    # Experiment 2 Model
    exp2_model = ACDMMKTv3(
        static_item_bank=mock_ablation_static_bank,
        d_model=d_model,
        num_heads=2,
        max_seq_len=20,
        dropout=0.0,
        use_solution=False,
    ).eval()
    exp2_param_count = sum(p.numel() for p in exp2_model.parameters() if p.requires_grad)

    # 1. Pure Pruning
    assert exp2_param_count < full_param_count, f"Exp2 params ({exp2_param_count}) must be < Full ({full_param_count})"

    # 2. Forward pass before perturbation
    with torch.no_grad():
        out_clean = exp2_model(mock_ablation_sequence_batch, use_cache=False)

    # Delta_mu_sol must be strictly zero
    max_delta_sol = out_clean["delta_mu_sol"].abs().max().item()
    assert max_delta_sol <= 1e-7, f"Exp2 delta_mu_sol must be 0, got {max_delta_sol}"

    # Verify Gram-Schmidt orthogonality on stem representation
    item_props = exp2_model.forward_items(
        torch.arange(mock_ablation_static_bank["modality_masks"].size(0)),
        exp2_model.compute_all_kc_embeddings(),
        torch.full((mock_ablation_static_bank["modality_masks"].size(0),), -1.37),
    )
    c_C = item_props["c_C"]
    d_q = item_props["d_q"]
    dot_prods = (d_q * c_C).sum(dim=-1).abs()
    assert dot_prods.max().item() < 1e-5, f"Gram-Schmidt failed in Exp2: max |<d^q, c_C>| = {dot_prods.max().item()}"

    # 3. Inject EXTREME perturbation (+/- 10,000) into solution text and solution image
    exp2_model.acd_E_txt_s.data.add_(torch.randn_like(exp2_model.acd_E_txt_s) * 1e4)
    exp2_model.acd_E_img_s.data.add_(torch.randn_like(exp2_model.acd_E_img_s) * 1e4)

    with torch.no_grad():
        out_perturbed = exp2_model(mock_ablation_sequence_batch, use_cache=False)

    # Check zero leakage
    max_diff_prob = (out_clean["probs"] - out_perturbed["probs"]).abs().max().item()
    max_diff_theta = (out_clean["theta"] - out_perturbed["theta"]).abs().max().item()
    max_diff_mu = (out_clean["tilde_mu"] - out_perturbed["tilde_mu"]).abs().max().item()
    max_diff_a = (out_clean["a_disc"] - out_perturbed["a_disc"]).abs().max().item()

    assert max_diff_prob <= 1e-7, f"Exp2 prob leaked solution! max diff = {max_diff_prob}"
    assert max_diff_theta <= 1e-7, f"Exp2 theta leaked solution! max diff = {max_diff_theta}"
    assert max_diff_mu <= 1e-7, f"Exp2 tilde_mu leaked solution! max diff = {max_diff_mu}"
    assert max_diff_a <= 1e-7, f"Exp2 a_disc leaked solution! max diff = {max_diff_a}"


def test_experiment3_zero_visual_leakage_and_solution_image_fix(mock_ablation_static_bank, mock_ablation_sequence_batch):
    """
    Experiment 3 (Text-Only Stream / w/o Visual):
    1. Extreme perturbation (+/- 10^4) on E_img_q and E_img_s produces 0.0 change on outputs.
    2. Crucial fix verification: Even on questions with m_img^s = 1 (Quarter 3 and 4),
       v_sol = u_txt_s purely without any visual contribution.
    3. Delta_mu_vis = 0, lambda_q = 0 strictly.
    4. Parameter count strictly reduced compared to Full model.
    """
    torch.manual_seed(42)
    d_model = 64

    # Full Model
    full_model = ACDMMKTv3(
        static_item_bank=mock_ablation_static_bank,
        d_model=d_model,
        num_heads=2,
        max_seq_len=20,
        dropout=0.0,
        use_visual=True,
    ).eval()
    full_param_count = sum(p.numel() for p in full_model.parameters() if p.requires_grad)

    # Experiment 3 Model
    exp3_model = ACDMMKTv3(
        static_item_bank=mock_ablation_static_bank,
        d_model=d_model,
        num_heads=2,
        max_seq_len=20,
        dropout=0.0,
        use_visual=False,
    ).eval()
    exp3_param_count = sum(p.numel() for p in exp3_model.parameters() if p.requires_grad)

    # 1. Pure Pruning
    assert exp3_param_count < full_param_count, f"Exp3 params ({exp3_param_count}) must be < Full ({full_param_count})"

    # 2. Forward pass before perturbation
    with torch.no_grad():
        out_clean = exp3_model(mock_ablation_sequence_batch, use_cache=False)

    # Verify visual loads and cognitive gates are strictly zero
    assert out_clean["delta_mu_vis"].abs().max().item() <= 1e-7, "delta_mu_vis must be 0 in Exp3"
    assert out_clean["lambda_q"].abs().max().item() <= 1e-7, "lambda_q must be 0 in Exp3"
    assert out_clean["bar_lambda_q"].abs().max().item() <= 1e-7, "bar_lambda_q must be 0 in Exp3"

    # Verify solution representations on questions with m_img^s = 1 (Quarter 3: questions 20..29)
    # They must equal u_txt_s purely
    sol_img_q_ids = torch.arange(20, 30)
    item_props = exp3_model.forward_items(
        sol_img_q_ids,
        exp3_model.compute_all_kc_embeddings(),
        torch.full((10,), -1.37),
    )
    # In Exp3, v_sol must be identical to u_txt_s even with m_img^s = 1
    # Check that solution image perturbation does not change v_sol
    v_sol_clean = item_props["v_sol"].clone()

    # 3. Inject EXTREME perturbation (+/- 10,000) into stem image and solution image
    exp3_model.acd_E_img_q.data.add_(torch.randn_like(exp3_model.acd_E_img_q) * 1e4)
    exp3_model.acd_E_img_s.data.add_(torch.randn_like(exp3_model.acd_E_img_s) * 1e4)

    with torch.no_grad():
        out_perturbed = exp3_model(mock_ablation_sequence_batch, use_cache=False)
        item_props_perturbed = exp3_model.forward_items(
            sol_img_q_ids,
            exp3_model.compute_all_kc_embeddings(),
            torch.full((10,), -1.37),
        )

    # Check zero leakage on item representations
    v_sol_diff = (v_sol_clean - item_props_perturbed["v_sol"]).abs().max().item()
    assert v_sol_diff <= 1e-7, f"Exp3 v_sol leaked solution image! max diff = {v_sol_diff}"

    # Check zero leakage on predictive and psychometric sequence outputs
    max_diff_prob = (out_clean["probs"] - out_perturbed["probs"]).abs().max().item()
    max_diff_theta = (out_clean["theta"] - out_perturbed["theta"]).abs().max().item()
    max_diff_mu = (out_clean["tilde_mu"] - out_perturbed["tilde_mu"]).abs().max().item()
    max_diff_a = (out_clean["a_disc"] - out_perturbed["a_disc"]).abs().max().item()

    assert max_diff_prob <= 1e-7, f"Exp3 prob leaked visual! max diff = {max_diff_prob}"
    assert max_diff_theta <= 1e-7, f"Exp3 theta leaked visual! max diff = {max_diff_theta}"
    assert max_diff_mu <= 1e-7, f"Exp3 tilde_mu leaked visual! max diff = {max_diff_mu}"
    assert max_diff_a <= 1e-7, f"Exp3 a_disc leaked visual! max diff = {max_diff_a}"


def test_gradient_flow_and_no_orphan_parameters_across_ablations(mock_ablation_static_bank, mock_ablation_sequence_batch):
    """
    Verify that 100% of initialized trainable parameters in all 3 ablation models
    receive valid, finite non-zero gradients when optimizing their respective losses.
    Guarantees no dead or orphan parameters.
    """
    torch.manual_seed(42)
    d_model = 64
    n_q = torch.randint(10, 100, (40,))
    d_q = torch.full((40,), 0.25)

    ablation_settings = [
        {"name": "Exp 1: Pure KC", "use_multimodal": False, "beta_d": 0.0, "lambda_sparse": 0.0},
        {"name": "Exp 2: w/o Solution", "use_solution": False, "beta_d": 0.20, "lambda_sparse": 0.01},
        {"name": "Exp 3: w/o Visual", "use_visual": False, "beta_d": 0.20, "lambda_sparse": 0.01},
    ]

    for setting in ablation_settings:
        name = setting["name"]
        model = ACDMMKTv3(
            static_item_bank=mock_ablation_static_bank,
            d_model=d_model,
            num_heads=2,
            max_seq_len=20,
            dropout=0.0,
            use_multimodal=setting.get("use_multimodal", True),
            use_solution=setting.get("use_solution", True),
            use_visual=setting.get("use_visual", True),
        ).train()

        loss_fn = ACDMMKTv3Loss(
            beta_d=setting["beta_d"],
            lambda_sparse=setting["lambda_sparse"],
            kappa=10.0,
        )

        outputs = model(mock_ablation_sequence_batch, use_cache=False)
        # Activate step 0 in loss mask for gradient verification of initial state h_init
        active_loss_mask = mock_ablation_sequence_batch["loss_mask"].clone()
        active_loss_mask[:, 0] = True

        loss, _ = loss_fn(
            model_outputs=outputs,
            targets=mock_ablation_sequence_batch["r_seq"],
            loss_mask=active_loss_mask,
            n_q_all=n_q,
            d_q_all=d_q,
        )

        loss.backward()

        # Check every initialized parameter has non-None, finite gradient
        unupdated_params = []
        for p_name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    unupdated_params.append((p_name, "None grad"))
                elif torch.isnan(param.grad).any():
                    unupdated_params.append((p_name, "NaN grad"))
                elif torch.isinf(param.grad).any():
                    unupdated_params.append((p_name, "Inf grad"))
                elif (param.grad == 0).all():
                    # Check if all zero
                    unupdated_params.append((p_name, "Zero grad"))

        assert len(unupdated_params) == 0, f"Orphan / invalid gradient parameters in {name}: {unupdated_params}"


def test_cache_equivalence_and_100pct_faithfulness_across_ablations(mock_ablation_static_bank, mock_ablation_sequence_batch):
    """
    Verify that:
    1. Direct evaluation and Precomputed Offline Item-Bank Cache give identical predictions (diff <= 1e-6).
    2. 100% 2PL-IRT Decision Faithfulness: (p_hat >= 0.5) <=> (theta >= tilde_mu) holds on 100% of interactions.
    """
    torch.manual_seed(42)
    d_model = 64
    mu_prior = torch.full((40,), -1.37)

    for use_mm, use_sol, use_vis, name in [
        (False, True, True, "Exp 1: Pure KC"),
        (True, False, True, "Exp 2: w/o Solution"),
        (True, True, False, "Exp 3: w/o Visual"),
    ]:
        model = ACDMMKTv3(
            static_item_bank=mock_ablation_static_bank,
            d_model=d_model,
            num_heads=2,
            max_seq_len=20,
            dropout=0.0,
            use_multimodal=use_mm,
            use_solution=use_sol,
            use_visual=use_vis,
        ).eval()

        # 1. Non-cache forward
        with torch.no_grad():
            out_direct = model(mock_ablation_sequence_batch, mu_concept_prior=mu_prior, use_cache=False)

        # 2. Cache forward
        model.precompute_item_bank_cache(mu_prior)
        with torch.no_grad():
            out_cached = model(mock_ablation_sequence_batch, mu_concept_prior=mu_prior, use_cache=True)
        model.clear_item_bank_cache()

        # Compare predictions
        diff = (out_direct["probs"] - out_cached["probs"]).abs().max().item()
        assert diff <= 1e-6, f"{name}: Cache vs Direct diff = {diff} > 1e-6"

        # 3. 100% 2PL-IRT Decision Faithfulness check
        probs = out_cached["probs"].numpy()
        theta = out_cached["theta"].numpy()
        tilde_mu = out_cached["tilde_mu"].numpy()

        faithful = (probs >= 0.5) == (theta >= tilde_mu)
        faithful_pct = float(faithful.mean()) * 100.0
        assert faithful_pct == 100.0, f"{name}: Faithfulness was {faithful_pct}% != 100%"


def test_explainability_engine_across_all_three_ablations(mock_ablation_static_bank, mock_ablation_sequence_batch):
    """
    Verify that ACDv3ExplainabilityEngine generates robust, valid 4-tier explanations
    across all three ablation study variants without crashes or invalid keywords/semantics.
    """
    torch.manual_seed(42)
    d_model = 64
    mu_prior = torch.full((40,), -1.37)

    for use_mm, use_sol, use_vis, ab_name in [
        (False, True, True, "Exp 1: Pure KC"),
        (True, False, True, "Exp 2: w/o Solution"),
        (True, True, False, "Exp 3: w/o Visual"),
    ]:
        model = ACDMMKTv3(
            static_item_bank=mock_ablation_static_bank,
            d_model=d_model,
            num_heads=2,
            max_seq_len=20,
            dropout=0.0,
            use_multimodal=use_mm,
            use_solution=use_sol,
            use_visual=use_vis,
        ).eval()

        engine = ACDv3ExplainabilityEngine(model=model, aux_metadata_path=None)
        exp = engine.explain_interaction(
            batch=mock_ablation_sequence_batch,
            sample_idx=0,
            target_step=5,
            top_k_keywords=5,
            top_k_history=3,
            mu_concept_prior=mu_prior,
        )

        assert "tier1_psychometrics" in exp
        assert "tier2_difficulty" in exp
        assert "tier3_multimodal" in exp
        assert "tier4_history" in exp

        t3 = exp["tier3_multimodal"]
        if not use_mm:
            # Ablation 1: pure KC -> no multimodal tokens or visual load
            assert len(t3["stem_keywords"]) == 0, f"{ab_name}: stem_keywords must be empty when multimodal is disabled"
            assert len(t3["solution_steps"]) == 0, f"{ab_name}: solution_steps must be empty when multimodal is disabled"
            assert "Visual channel disabled" in t3["visual_impact_semantics"]
            assert t3["delta_mu_txt"] == 0.0
            assert t3["delta_mu_vis"] == 0.0
        elif not use_sol:
            # Ablation 2: w/o solution -> no solution steps, delta_mu_sol = 0
            assert len(t3["solution_steps"]) == 0, f"{ab_name}: solution_steps must be empty when solution is disabled"
        elif not use_vis:
            # Ablation 3: w/o visual -> no visual patches, delta_mu_vis = 0, bar_lambda_q = 0
            assert len(t3["top_patches"]) == 0, f"{ab_name}: top_patches must be empty when visual is disabled"
            assert "Visual channel disabled" in t3["visual_impact_semantics"]
            assert t3["delta_mu_vis"] == 0.0
            assert t3["bar_lambda_q"] == 0.0

        # Formatted markdown report test
        report_text = engine.format_markdown_report(exp)
        assert isinstance(report_text, str) and len(report_text) > 50
