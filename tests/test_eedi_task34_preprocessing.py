"""
Comprehensive Verification Suite for Eedi NeurIPS 2020 (Task 3 & 4) Preprocessing.

Verifies all 10 Acceptance Gates specified in
plans/eedi_task34_mmkt_preprocessing_plan.md directly against the processed
Two-Tier artifacts under data/processed/Eedi/ and splits under data/splits/Eedi/,
as well as drop-in compatibility with ACDMMKTv3 (Full & Ablations 1-3),
MGMMKT, MGMMKTv2, and ACDv3ExplainabilityEngine without modifying model code.
"""

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.eedi_dataset import (
    EediDataset,
    get_eedi_dataloaders,
    load_eedi_static_item_bank,
)
from src.evaluation.acd_v3_explainability import ACDv3ExplainabilityEngine
from src.models.mmkt.acd_mmkt_v3 import ACDMMKTv3
from src.models.mmkt.mg_mmkt import MGMMKT
from src.models.mmkt.mg_mmkt_v2 import MGMMKTv2
from src.training.losses import ACDMMKTv3Loss


PROCESSED_DIR = Path("data/processed/Eedi")
FEATURES_DIR = Path("data/features/Eedi")
SPLITS_DIR = Path("data/splits/Eedi")
RAW_DIR = Path("data/raw/Eedi NeurIPS 2020/data_extracted/data")


def _artifacts_ready() -> bool:
    required = [
        PROCESSED_DIR / "static_item_bank.pt",
        PROCESSED_DIR / "mg_static_item_bank.pt",
        PROCESSED_DIR / "acd_v3_aux_metadata.pt",
        PROCESSED_DIR / "test_strata_masks.pt",
        PROCESSED_DIR / "sequences_test.pt",
        PROCESSED_DIR / "sequences_test_public.pt",
        PROCESSED_DIR / "sequences_test_private.pt",
    ] + [PROCESSED_DIR / f"sequences_fold_{f}.pt" for f in range(5)]
    return all(p.exists() for p in required)


requires_processed = pytest.mark.skipif(
    not _artifacts_ready(),
    reason="Processed Eedi artifacts not yet built in data/processed/Eedi/",
)


@pytest.fixture(scope="module")
def static_bank():
    return torch.load(PROCESSED_DIR / "static_item_bank.pt", map_location="cpu")


@pytest.fixture(scope="module")
def mg_bank():
    return torch.load(PROCESSED_DIR / "mg_static_item_bank.pt", map_location="cpu")


@pytest.fixture(scope="module")
def aux_meta():
    return torch.load(PROCESSED_DIR / "acd_v3_aux_metadata.pt", map_location="cpu", weights_only=False)


# ----------------------------------------------------------------------
# Gate 1: Two-Tier Artifact Completeness & Raw User Disjointness
# ----------------------------------------------------------------------
@requires_processed
def test_gate1_artifacts_and_user_disjointness(static_bank, mg_bank, aux_meta):
    assert (SPLITS_DIR / "train_val_folds.json").exists()
    assert (SPLITS_DIR / "test_public_uids.json").exists()
    assert (SPLITS_DIR / "test_private_uids.json").exists()

    with open(SPLITS_DIR / "train_val_folds.json") as f:
        folds = json.load(f)
    with open(SPLITS_DIR / "test_public_uids.json") as f:
        pub_uids = set(json.load(f))
    with open(SPLITS_DIR / "test_private_uids.json") as f:
        priv_uids = set(json.load(f))

    fold_sets = [set(folds[f"fold_{k}"]) for k in range(5)]
    all_train_uids = set().union(*fold_sets)
    assert sum(len(s) for s in fold_sets) == len(all_train_uids) == 4918
    assert len(pub_uids) == 615
    assert len(priv_uids) == 615
    assert not (all_train_uids & pub_uids)
    assert not (all_train_uids & priv_uids)
    assert not (pub_uids & priv_uids)


# ----------------------------------------------------------------------
# Gate 2: Official Answer Keys & Non-Leaking Distractors
# ----------------------------------------------------------------------
@requires_processed
def test_gate2_official_keys_and_distractors(static_bank, aux_meta):
    c_idx = static_bank["correct_answer_idx"]
    assert c_idx.shape == (948,) and c_idx.dtype == torch.int64
    assert int(c_idx.min()) == 0 and int(c_idx.max()) == 3

    # Verify the 4 questions with historical Eedi key revisions match latest timestamp (0-based A=0..D=3)
    assert int(c_idx[43]) == 1   # Q43: B (8)
    assert int(c_idx[84]) == 0   # Q84: A
    assert int(c_idx[206]) == 2  # Q206: C
    assert int(c_idx[860]) == 1  # Q860: B

    dom_dist = aux_meta["fold_dominant_distractor"]
    dist_ent = aux_meta["fold_distractor_entropy"]
    assert dom_dist.shape == (5, 948) and dom_dist.dtype == torch.int64
    assert dist_ent.shape == (5, 948) and dist_ent.dtype == torch.float32
    assert bool(torch.isfinite(dist_ent).all())
    assert float(dist_ent.min()) >= 0.0 and float(dist_ent.max()) <= 1.0 + 1e-6
    # Dominant distractor must never collide with official answer key
    assert not bool((dom_dist == c_idx.unsqueeze(0)).any())


# ----------------------------------------------------------------------
# Gate 3: CV Slide Anatomy, Modality Masks & Duplicate Consistency
# ----------------------------------------------------------------------
@requires_processed
def test_gate3_slide_anatomy_modality_masks_and_dedup(static_bank, aux_meta):
    mod = static_bank["modality_masks"]
    assert mod.shape == (948, 3) and mod.dtype == torch.float32
    assert bool((mod[:, 1] == 1.0).all())  # 100% text-anchored solution
    assert bool((mod[:, 2] == 0.0).all())  # No authentic solution images on Eedi
    assert set(mod[:, 0].unique().tolist()).issubset({0.0, 1.0})
    n_diag = int(mod[:, 0].sum().item())
    assert 300 <= n_diag <= 650

    # Text-only items must hold exact zero visual embeddings
    text_only_mask = mod[:, 0] == 0.0
    assert float(static_bank["acd_E_img_q"][text_only_mask].abs().max()) == 0.0
    assert float(static_bank["mg_v_img_q"][text_only_mask].abs().max()) == 0.0
    assert float(static_bank["acd_E_img_s"].abs().max()) == 0.0
    assert float(static_bank["mg_v_img_s"].abs().max()) == 0.0

    # Calculator icon meta sanity (zero unslashed calculator-allowed slides in Eedi Task 3&4)
    img_meta_q = aux_meta["image_meta_q"]
    assert len(img_meta_q) == 948
    assert all(float(m["calculator_allowed_flag"]) == 0.0 for m in img_meta_q)
    assert all(m["layout_type"] in {
        "horiz_below", "horiz_above_options", "horiz_ref",
        "vert_right", "vert_left", "grid_std", "grid_embedded",
    } for m in img_meta_q)

    # Exact MD5 image duplicate pair (220, 812) must share identical extracted representations
    assert torch.allclose(static_bank["acd_E_txt_q"][220], static_bank["acd_E_txt_q"][812])
    assert torch.allclose(static_bank["acd_E_txt_s"][220], static_bank["acd_E_txt_s"][812])
    assert torch.allclose(static_bank["acd_E_img_q"][220], static_bank["acd_E_img_q"][812])


# ----------------------------------------------------------------------
# Gate 4: Zero-Leakage OCR T_q, Solution T_s^q & 1-to-1 Token Alignment
# ----------------------------------------------------------------------
@requires_processed
def test_gate4_ocr_solutions_and_token_alignment(static_bank, aux_meta):
    ocr_path = FEATURES_DIR / "ocr_texts_q.jsonl"
    sol_path = FEATURES_DIR / "solutions_s.jsonl"
    assert ocr_path.exists() and sol_path.exists()

    leak_pat = re.compile(r"\b(correct\s+answer\s*(is|:)|official\s+key)\b", re.IGNORECASE)
    with open(ocr_path, encoding="utf-8") as f:
        ocr_recs = [json.loads(line) for line in f if line.strip()]
    with open(sol_path, encoding="utf-8") as f:
        sol_recs = [json.loads(line) for line in f if line.strip()]

    assert len(ocr_recs) == 947
    assert len(sol_recs) == 947
    for r in ocr_recs:
        assert r["text_q"].strip()
        assert "[DOMAIN:" in r["text_q"]
        assert not leak_pat.search(r["text_q"])
    for r in sol_recs:
        assert len(r["text_s"].split()) >= 12
        assert f"Standard Answer: ({r['correct_letter']})" in r["text_s"]

    # Strict 1-to-1 token_to_word alignment with mask_txt_q / mask_txt_s
    tok_q = aux_meta["token_to_word_q"]
    tok_s = aux_meta["token_to_word_s"]
    mask_q = static_bank["acd_mask_txt_q"]
    mask_s = static_bank["acd_mask_txt_s"]
    assert static_bank["acd_E_txt_q"].shape == (948, 128, 768)
    assert static_bank["acd_E_txt_s"].shape == (948, 160, 768)
    for q in range(948):
        assert len(tok_q[q]) == 128
        assert sum(1 for tk in tok_q[q] if tk != "<pad>") == int(mask_q[q].sum().item())
        assert len(tok_s[q]) == 160
        assert sum(1 for tk in tok_s[q] if tk != "<pad>") == int(mask_s[q].sum().item())

    # Spatial patch mask M_patch_q validity check (including text-only default mask = 0.0 / True)
    m_patch = aux_meta["M_patch_q"]
    p_valid = aux_meta["patch_valid_mask"]
    text_only_mask = static_bank["modality_masks"][:, 0] == 0.0
    assert m_patch.shape == (948, 49) and p_valid.shape == (948, 49)
    assert bool((p_valid.any(dim=1)).all())  # No all-masked rows (Softmax NaN guard)
    assert bool((m_patch[p_valid] == 0.0).all())
    assert bool((m_patch[~p_valid] == -1e9).all())
    assert bool(p_valid[text_only_mask].all())
    assert bool((m_patch[text_only_mask] == 0.0).all())


# ----------------------------------------------------------------------
# Gate 5: 85-KC Hierarchy, Hypergraph H/G & Spectral Bounds
# ----------------------------------------------------------------------
@requires_processed
def test_gate5_hypergraph_and_kc_hierarchy(static_bank):
    H = static_bank["acd_hypergraph_H"]
    G = static_bank["acd_hypergraph_G"]
    kc_init = static_bank["acd_kc_init_emb"]
    leaf_kcs = static_bank["question_leaf_kcs"]
    leaf_mask = static_bank["question_kc_mask"]
    all_kcs = static_bank["question_all_kcs"]
    all_mask = static_bank["question_all_kc_mask"]

    assert H.shape == (85, 948) and H.dtype == torch.float32
    assert G.shape == (85, 85) and G.dtype == torch.float32
    assert kc_init.shape == (85, 768) and kc_init.dtype == torch.float32
    assert bool(torch.isfinite(H).all() and torch.isfinite(G).all() and torch.isfinite(kc_init).all())

    assert leaf_kcs.shape == (948, 6) and leaf_mask.shape == (948, 6)
    assert all_kcs.shape == (948, 5) and all_mask.shape == (948, 5)
    assert int(leaf_kcs[leaf_mask].min()) >= 0 and int(leaf_kcs[leaf_mask].max()) <= 56
    assert int(all_kcs[all_mask].min()) >= 0 and int(all_kcs[all_mask].max()) <= 84

    # All 85 active KCs must have strictly positive vertex degree; weights strictly in {0, 0.15, 0.5, 1.0}
    assert bool((H.sum(dim=1) > 0).all())
    uniq_w = sorted(round(float(x), 4) for x in H.unique().tolist())
    assert uniq_w == [0.0, 0.15, 0.5, 1.0]
    assert torch.allclose(G, G.t(), atol=1e-6)
    # Eigenvalues of symmetric normalized hypergraph Laplacian kernel lie in [0, 1 + eps]
    evals = torch.linalg.eigvalsh(G)
    assert float(evals.min()) >= -1e-5 and float(evals.max()) <= 1.0 + 1e-4


# ----------------------------------------------------------------------
# Gate 6: Strided Context Windowing & Zero Lost / Zero Duplicated Eval Steps
# ----------------------------------------------------------------------
@requires_processed
def test_gate6_strided_windows_and_eval_counts():
    train_loss_total = 0
    for f in range(5):
        fd = torch.load(PROCESSED_DIR / f"sequences_fold_{f}.pt", map_location="cpu")
        assert fd["q_seq"].ndim == 2 and fd["q_seq"].size(1) == 200
        assert fd["r_seq"].dtype == torch.int64
        train_loss_total += int(fd["loss_mask"].sum().item())
        # Verify post_diag_vec shape [5, N_seq, 200, 8]
        assert fd["post_diag_vec"].shape == (5, fd["q_seq"].size(0), 200, 8)
    assert train_loss_total == 1_382_727 - 4_918

    pub = torch.load(PROCESSED_DIR / "sequences_test_public.pt", map_location="cpu")
    priv = torch.load(PROCESSED_DIR / "sequences_test_private.pt", map_location="cpu")
    uni = torch.load(PROCESSED_DIR / "sequences_test.pt", map_location="cpu")

    assert int(pub["eval_mask"].sum().item()) == 74_276
    assert int(priv["eval_mask"].sum().item()) == 50_684
    assert int(uni["eval_mask"].sum().item()) == 124_960
    assert pub["is_target_mask"].shape == (pub["q_seq"].size(0), 200, 10)
    assert priv["is_target_mask"].shape == (priv["q_seq"].size(0), 200, 10)
    assert uni["is_target_mask"].shape == (uni["q_seq"].size(0), 200, 10)


# ----------------------------------------------------------------------
# Gate 7: Empirical Bayes Shrunk CTT Priors & Task 3 Expert Pairs
# ----------------------------------------------------------------------
@requires_processed
def test_gate7_empirical_bayes_ctt_and_expert_pairs(aux_meta):
    nq = aux_meta["N_q_folds"]
    dq = aux_meta["D_q_folds"]
    omega = aux_meta["omega_q_folds"]
    mu_p = aux_meta["mu_prior_folds"]
    mu_c = aux_meta["mu_concept_folds"]

    assert nq.shape == (5, 948) and dq.shape == (5, 948)
    assert int(nq.min().item()) >= 3  # Rare question balancing ensured train presence
    expected_omega = nq.float() / (nq.float() + 10.0)
    assert torch.allclose(omega, expected_omega, atol=1e-6)
    # Failure probability bounds and exact logit relationship mu_concept = log(mu_prior / (1 - mu_prior))
    assert float(mu_p.min()) >= 1e-4 - 1e-7 and float(mu_p.max()) <= 1.0 - 1e-4 + 1e-7
    assert float(dq.min()) >= 1e-4 - 1e-7 and float(dq.max()) <= 1.0 - 1e-4 + 1e-7
    assert torch.allclose(mu_c, torch.log(mu_p / (1.0 - mu_p)), atol=1e-5)

    # Rare questions (N_q <= 10) have omega <= 0.5
    rare_mask = nq <= 10
    assert bool((omega[rare_mask] <= 0.5 + 1e-6).all())

    pairs = aux_meta["task3_expert_pairs"]
    assert len(pairs) == 50
    for p in pairs:
        assert 0 <= p["left"] < 948 and 0 <= p["right"] < 948
        assert p["consensus_pref"] in (1, 2)


# ----------------------------------------------------------------------
# Gate 8: Temporal Non-Negativity & Fold-Isolated Diagnostic Vectors
# ----------------------------------------------------------------------
@requires_processed
def test_gate8_temporal_dynamics_and_post_diag():
    fd0 = torch.load(PROCESSED_DIR / "sequences_fold_0.pt", map_location="cpu")
    for key in ("norm_delta_t_lag", "norm_delta_t_kc", "kc_first_flag", "batch_flag", "relative_quiz_pos"):
        t = fd0[key]
        assert bool(torch.isfinite(t).all())
        assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0 + 1e-6

    # Negative sentinel in raw delta_t_kc must cleanly produce norm_delta_t_kc==0.0 & kc_first_flag==1.0
    neg_mask = fd0["delta_t_kc"] < 0.0
    assert bool((fd0["norm_delta_t_kc"][neg_mask] == 0.0).all())
    assert bool((fd0["kc_first_flag"][neg_mask & fd0["seq_mask"]] == 1.0).all())

    # Step 0 causal Laplace-smoothed running accuracy equals global train prior
    first_win = fd0["context_start"] == 0
    assert torch.allclose(
        fd0["running_acc_prev"][first_win, 0],
        torch.full((int(first_win.sum()),), 0.537332),
        atol=1e-5,
    )
    # Step 1 obeys causal Laplace formula (r_0 + 2 * 0.537332) / 3.0
    valid_step1 = first_win & fd0["seq_mask"][:, 1]
    expected_step1 = (fd0["r_seq"][valid_step1, 0].float() + 2.0 * 0.537332) / 3.0
    assert torch.allclose(fd0["running_acc_prev"][valid_step1, 1], expected_step1, atol=1e-5)

    pdv = fd0["post_diag_vec"]
    assert bool(torch.isfinite(pdv).all())
    sm = fd0["seq_mask"].bool()
    # [0..3] one-hot AnswerValue sum == 1.0 on valid steps and 0.0 on padded steps
    assert torch.allclose(pdv[0][sm][..., :4].sum(dim=-1), torch.ones(int(sm.sum())))
    assert float(pdv[0][~sm].abs().max()) == 0.0
    # Confidence normalization at index 6 in [0, 1]
    assert float(pdv[:, :, :, 6].min()) >= 0.0 and float(pdv[:, :, :, 6].max()) <= 1.0


# ----------------------------------------------------------------------
# Gate 9: Drop-In Compatibility with ACDMMKTv3 (Full + Ablations 1-3) & MG-MMKT/v2
# ----------------------------------------------------------------------
@requires_processed
def test_gate9_model_and_ablation_compatibility(static_bank, mg_bank, aux_meta):
    sctx = static_bank["question_static_ctx"]
    assert sctx.shape == (948, 15)
    assert bool((sctx[:, 0] == 1.0).all())
    assert float(sctx.min()) >= 0.0 and float(sctx.max()) <= 1.0 + 1e-6

    train_loader, val_loader, test_loader, loaded_bank = get_eedi_dataloaders(
        data_dir=str(PROCESSED_DIR),
        val_fold=0,
        model_type="acd_mmkt_v3",
        batch_size=4,
        test_batch_size=4,
        num_workers=0,
        test_split="unified",
    )
    batch = next(iter(train_loader))
    assert batch["q_seq"].shape == (4, 200)
    assert batch["post_diag_vec"].shape == (4, 200, 8)
    mu_prior = loaded_bank["mu_concept_folds"][0]
    n_q_all = loaded_bank["N_q_folds"][0]
    d_q_all = loaded_bank["D_q_folds"][0]

    criterion = ACDMMKTv3Loss(beta_d=0.20, lambda_sparse=0.01, kappa=10.0)

    # Test Full ACD-MMKTv3 (diagnostic post-diag mode) + zero orphan gradients
    full_model = ACDMMKTv3(
        static_item_bank=loaded_bank,
        d_model=64,
        num_heads=4,
        max_seq_len=200,
        post_diag_mode="diagnostic",
    )
    assert full_model.sub1.proj_img_sol is None
    full_model.train()
    out = full_model(batch, mu_concept_prior=mu_prior, use_cache=True)
    loss, _ = criterion(out, batch["r_seq"], batch["loss_mask"], n_q_all, d_q_all)
    assert torch.isfinite(loss)
    loss.backward()
    for pname, param in full_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Orphan parameter without gradient on Eedi: {pname}"

    # Verify Ablation invariance under extreme 1e4 noise injection
    bank_noisy = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in loaded_bank.items()}
    bank_noisy["acd_E_txt_q"] = bank_noisy["acd_E_txt_q"] + 1e4
    bank_noisy["acd_E_txt_s"] = bank_noisy["acd_E_txt_s"] + 1e4
    bank_noisy["acd_E_img_q"] = bank_noisy["acd_E_img_q"] + 1e4

    # Ablation 1: Pure KC-Interaction (immune to all text + visual noise)
    m_pure_a = ACDMMKTv3(static_item_bank=loaded_bank, d_model=64, num_heads=4, max_seq_len=200, use_multimodal=False).eval()
    m_pure_b = ACDMMKTv3(static_item_bank=bank_noisy, d_model=64, num_heads=4, max_seq_len=200, use_multimodal=False).eval()
    m_pure_b.load_state_dict(m_pure_a.state_dict())
    with torch.no_grad():
        y1 = m_pure_a(batch, mu_concept_prior=mu_prior)["probs"]
        y2 = m_pure_b(batch, mu_concept_prior=mu_prior)["probs"]
    assert torch.allclose(y1, y2, atol=1e-5)

    # Ablation 2: w/o Solution (immune to solution noise)
    bank_sol_noisy = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in loaded_bank.items()}
    bank_sol_noisy["acd_E_txt_s"] = bank_sol_noisy["acd_E_txt_s"] + 1e4
    m_nosol_a = ACDMMKTv3(static_item_bank=loaded_bank, d_model=64, num_heads=4, max_seq_len=200, use_solution=False).eval()
    m_nosol_b = ACDMMKTv3(static_item_bank=bank_sol_noisy, d_model=64, num_heads=4, max_seq_len=200, use_solution=False).eval()
    m_nosol_b.load_state_dict(m_nosol_a.state_dict())
    with torch.no_grad():
        assert torch.allclose(
            m_nosol_a(batch, mu_concept_prior=mu_prior)["probs"],
            m_nosol_b(batch, mu_concept_prior=mu_prior)["probs"],
            atol=1e-5,
        )

    # Ablation 3: w/o Visual (immune to visual noise)
    bank_vis_noisy = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in loaded_bank.items()}
    bank_vis_noisy["acd_E_img_q"] = bank_vis_noisy["acd_E_img_q"] + 1e4
    m_novis_a = ACDMMKTv3(static_item_bank=loaded_bank, d_model=64, num_heads=4, max_seq_len=200, use_visual=False).eval()
    m_novis_b = ACDMMKTv3(static_item_bank=bank_vis_noisy, d_model=64, num_heads=4, max_seq_len=200, use_visual=False).eval()
    m_novis_b.load_state_dict(m_novis_a.state_dict())
    with torch.no_grad():
        assert torch.allclose(
            m_novis_a(batch, mu_concept_prior=mu_prior)["probs"],
            m_novis_b(batch, mu_concept_prior=mu_prior)["probs"],
            atol=1e-5,
        )

    # MG-MMKT & MG-MMKTv2 forward compatibility
    _, _, _, mg_loaded = get_eedi_dataloaders(
        data_dir=str(PROCESSED_DIR), val_fold=0, model_type="mg_mmkt_v2", batch_size=4, num_workers=0
    )
    mg1 = MGMMKT(num_questions=948, num_kcs=85, d_model=64, static_item_bank=mg_loaded).eval()
    mg2 = MGMMKTv2(num_questions=948, num_kcs=85, d_model=64, d_ffn=128, static_item_bank=mg_loaded).eval()
    with torch.no_grad():
        out_mg1 = mg1(batch)
        out_mg2 = mg2(
            batch["q_seq"],
            batch["r_seq"],
            batch["seq_mask"],
            modality_masks=batch["modality_masks"],
            raw_modality_masks=batch["raw_modality_masks"],
            leaf_kcs=batch["leaf_kcs"],
            leaf_kc_mask=batch["leaf_kc_mask"],
            delta_t_lag=batch["delta_t_lag"],
            delta_t_kc=batch["delta_t_kc"],
            batch_flag=batch["batch_flag"],
        )
    assert out_mg1["preds"].shape == (4, 200) and torch.isfinite(out_mg1["preds"]).all()
    assert out_mg2["probs"].shape == (4, 200) and torch.isfinite(out_mg2["probs"]).all()


# ----------------------------------------------------------------------
# Gate 10: Explainability Engine & Stratified Evaluation Masks
# ----------------------------------------------------------------------
@requires_processed
def test_gate10_explainability_and_stratified_masks(aux_meta):
    _, _, test_loader, loaded_bank = get_eedi_dataloaders(
        data_dir=str(PROCESSED_DIR),
        val_fold=0,
        model_type="acd_mmkt_v3",
        batch_size=4,
        test_batch_size=4,
        num_workers=0,
        test_split="unified",
    )
    batch = next(iter(test_loader))
    model = ACDMMKTv3(static_item_bank=loaded_bank, d_model=64, num_heads=4, max_seq_len=200).eval()
    engine = ACDv3ExplainabilityEngine(
        model=model,
        aux_metadata_path=str(PROCESSED_DIR / "acd_v3_aux_metadata.pt"),
        device=torch.device("cpu"),
    )
    valid_len = int(batch["seq_mask"][0].sum().item())
    report = engine.explain_interaction(
        batch=batch,
        sample_idx=0,
        target_step=min(10, valid_len - 1),
        mu_concept_prior=loaded_bank["mu_concept_folds"][0],
    )
    assert isinstance(report, dict) and report
    md = engine.format_markdown_report(report)
    assert isinstance(md, str) and len(md) > 20

    # Verify stratified test slice masks remain strict subsets of eval_mask and clean partitions
    strata_all = torch.load(PROCESSED_DIR / "test_strata_masks.pt", map_location="cpu")
    for split_name, fname in [
        ("public", "sequences_test_public.pt"),
        ("private", "sequences_test_private.pt"),
        ("unified", "sequences_test.pt"),
    ]:
        seq = torch.load(PROCESSED_DIR / fname, map_location="cpu")
        em = seq["eval_mask"].bool()
        sp_strata = strata_all[split_name]
        assert int((sp_strata["visual_diagram"] | sp_strata["text_only"]).sum()) == int(em.sum())
        assert int((sp_strata["cold_start_q"] | sp_strata["medium_freq_q"] | sp_strata["frequent_q"]).sum()) == int(em.sum())
        assert int((sp_strata["layout_horizontal"] | sp_strata["layout_grid"] | sp_strata["layout_vertical"]).sum()) == int(em.sum())
        assert int((sp_strata["single_kc"] | sp_strata["multi_kc"]).sum()) == int(em.sum())
        assert int((sp_strata["after_error"] | sp_strata["after_correct"]).sum()) == int(em.sum())
        assert not bool((sp_strata["after_error"] & sp_strata["after_correct"]).any())
        assert "after_dom_misconception" in sp_strata
        assert "after_overconfident_error" in sp_strata
        assert "domain_number" in sp_strata and "domain_algebra" in sp_strata and "domain_geometry" in sp_strata
        for s_name, mask in sp_strata.items():
            assert mask.shape == em.shape and mask.dtype == torch.bool
            assert not bool((mask & (~em)).any()), f"Stratum {split_name}.{s_name} leaked outside eval_mask!"


# ======================================================================
# Non-Skipped Unit & Integration Tests (Run Always Without Heavy Preproc)
# ======================================================================
def test_unit_slide_cv_saturation_and_hybrid_ocr_routing(tmp_path: Path, monkeypatch):
    """Verify region-aware OCR routing, saturation-based diagram detection, Step 3 cache sync, and Step 4 VLM->LLM order."""
    from unittest.mock import patch
    from PIL import Image
    import src.data.eedi_vlm_ocr as vlm_mod
    from src.data.eedi_slide_cv import extract_no_calc_template, extract_slide_rois_and_metadata
    from src.data.eedi_preprocessor import EediTask34Preprocessor
    from src.encoders import letterbox_image_and_mask

    # 1. Test real Eedi slides (4.jpg with visual options, 77.jpg with blue dot grid) if available
    img_dir = RAW_DIR / "images"
    av_path = FEATURES_DIR / "apple_vision_ocr.jsonl"
    if (img_dir / "0.jpg").exists() and (img_dir / "4.jpg").exists() and (img_dir / "77.jpg").exists() and av_path.exists():
        av_map = {}
        with open(av_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["qid"] in (4, 77):
                    av_map[int(rec["qid"])] = rec["items"]
        no_calc_tmpl = extract_no_calc_template(img_dir / "0.jpg")
        roi_dir = tmp_path / "rois"

        meta4 = extract_slide_rois_and_metadata(4, img_dir / "4.jpg", av_map.get(4, []), roi_dir, no_calc_tmpl)
        meta77 = extract_slide_rois_and_metadata(77, img_dir / "77.jpg", av_map.get(77, []), roi_dir, no_calc_tmpl)
        assert meta77["stem_sat_density"] >= 0.0008
        assert vlm_mod.compute_hybrid_has_stem_diagram(True, False, meta77) is True
        assert vlm_mod.compute_hybrid_has_stem_diagram(
            False,
            True,
            meta4,
            opts_norm={k: f"[VISUAL_DIAGRAM_OPTION_{k}] bar" for k in "ABCD"},
        ) is True

        # Verify composite image helper when options contain diagrams (4.jpg)
        comp = EediTask34Preprocessor._compose_stem_with_visual_options(
            meta4["roi_stem_path"], meta4["roi_opt_paths"]
        )
        lb_img, m_patch, p_valid, _ = letterbox_image_and_mask(comp, target_size=224, grid_size=7)
        assert lb_img.size == (224, 224)
        assert m_patch.shape == (49,) and p_valid.shape == (49,) and bool(p_valid.any())

    # 2. Negative control: pure text item must remain m_img^q == 0 even if VLM hallucinates diagram=True
    pure_text_meta = {
        "layout_type": "horiz_below",
        "stem_struct_density": 0.0002,
        "stem_sat_density": 0.0,
        "stem_has_long_stroke": False,
        "stem_only_small_box": False,
        "opt_cv_visual": {"A": False, "B": False, "C": False, "D": False},
        "badges": {"A": {"x0": 40, "y0": 260, "x1": 65, "y1": 285}},
        "opt_bboxes": {"A": [70, 255, 160, 295], "B": [170, 255, 260, 295], "C": [270, 255, 360, 295], "D": [370, 255, 460, 295]},
    }
    assert vlm_mod.compute_hybrid_has_stem_diagram(True, False, pure_text_meta) is False

    # 3. Region-aware OCR hints separate stem words from badge letters and option boxes
    ocr_items = [
        {"bbox": [0.10, 0.25, 0.40, 0.06], "text": "Solve 2x = 8"},
        {"bbox": [0.065, 0.55, 0.03, 0.04], "text": "A"},  # badge letter inside badges["A"] -> skipped
        {"bbox": [0.14, 0.55, 0.08, 0.05], "text": "4"},    # option A value inside opt_bboxes["A"]
    ]
    routed = vlm_mod._extract_region_aware_ocr_hints(ocr_items, pure_text_meta)
    assert routed["stem_hints"] == ["Solve 2x = 8"]
    assert routed["opt_hints"]["A"] == ["4"]

    # 4. Curriculum prefix T_q and Step 4.2 T_s^q formatting + Step 3 cache sync + Step 4 VLM->LLM order
    tq = vlm_mod.format_question_text_T_q(
        "Solve 2x = 8",
        {"A": "A. 2", "B": "4", "C": "6", "D": "8"},
        kc_prefix="[DOMAIN: Algebra | TOPIC: Equations | SKILL: Linear Equations]",
    )
    assert tq.startswith("[DOMAIN: Algebra | TOPIC: Equations | SKILL: Linear Equations] Question: Solve 2x = 8 [SEP]")
    assert "Option A: 2" in tq

    ts = vlm_mod._format_solution_text("B", "4", "Divide both sides by 2 to get x = 4.", "Option A subtracts 2 instead of dividing.")
    assert ts.startswith("Standard Answer: (B) 4 [SEP] Solution Analysis:")
    assert "| Misconception Note:" in ts

    # Step 3 cache refresh atomically rewrites outdated cached entry with curriculum prefix + fixed diagram flag
    dummy_slide = tmp_path / "imgs" / "0.jpg"
    dummy_slide.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 480), (255, 255, 255)).save(dummy_slide)
    ocr_jsonl = tmp_path / "ocr.jsonl"
    stale_rec = {
        "qid": 0,
        "stem_text": "Which bar chart shows 5?",
        "options": {"A": "A", "B": "B", "C": "C", "D": "D"},
        "text_q": "Question: Which bar chart shows 5?",
        "vlm_stem_has_diagram": False,
        "options_have_diagrams": True,
        "has_stem_diagram": False,
        "stem_visual_summary": "",
        "fallback_used": False,
    }
    ocr_jsonl.write_text(json.dumps(stale_rec) + "\n", encoding="utf-8")
    l_inv = {
        0: {**pure_text_meta, "opt_cv_visual": {"A": True, "B": True, "C": True, "D": True}},
        1: {**pure_text_meta},
    }
    synced_ocr = vlm_mod.run_step3_ocr_extraction(
        unique_rep_qids=[0],
        dup_to_rep={0: 0},
        images_dir=dummy_slide.parent,
        ocr_items_by_qid={0: []},
        layout_inventory=l_inv,
        out_jsonl_path=ocr_jsonl,
        kc_prefix_by_qid={0: "[DOMAIN: Number | TOPIC: Charts | SKILL: Reading Charts]"},
    )
    assert synced_ocr[0]["has_stem_diagram"] is True
    assert "[DOMAIN: Number" in synced_ocr[0]["text_q"]
    reloaded = json.loads(ocr_jsonl.read_text(encoding="utf-8").strip())
    assert reloaded["has_stem_diagram"] is True and "[DOMAIN: Number" in reloaded["text_q"]

    # Step 4 ordering: visual question calls VLM with image first, text question calls LLM without image
    (dummy_slide.parent / "1.jpg").write_bytes(dummy_slide.read_bytes())
    call_log = []

    def fake_call_ollama_json(model, prompt, img_b64=None, ollama_url="", num_predict=240, **kwargs):
        call_log.append((model, img_b64 is not None))
        return {
            "steps": "Evaluate the expression step by step so the exact result matches option B (4) cleanly.",
            "misconceptions": "Choosing A, C, or D results from sign error or adding rather than dividing.",
        }

    with patch.object(vlm_mod, "call_ollama_json", side_effect=fake_call_ollama_json):
        sols = vlm_mod.run_step4_solution_generation(
            unique_rep_qids=[1, 0],  # Q1 is text-only, Q0 is visual -> Q0 must run first with VLM!
            dup_to_rep={0: 0, 1: 1},
            images_dir=dummy_slide.parent,
            ocr_by_qid={
                0: {**synced_ocr[0], "has_stem_diagram": True, "options": {"A": "1", "B": "4", "C": "6", "D": "8"}},
                1: {"stem_text": "Compute 2+2", "options": {"A": "1", "B": "4", "C": "6", "D": "8"}, "has_stem_diagram": False, "options_have_diagrams": False, "stem_visual_summary": ""},
            },
            correct_idx_by_qid={0: 1, 1: 1},
            kc_names_by_qid={0: "Reading Charts", 1: "Addition"},
            out_jsonl_path=tmp_path / "sols.jsonl",
            vlm_model="vlm-test",
            llm_model="llm-test",
        )
    assert call_log == [("vlm-test", True), ("llm-test", False)]
    assert sols[0]["model_used"] == "vlm-test"
    assert sols[1]["model_used"] == "llm-test"
    assert "Standard Answer: (B) 4 [SEP] Solution Analysis:" in sols[0]["text_s"]


def test_unit_step1_step9_step10_step11_real_raw_slice():
    """Run Step 1 audit, Step 9 EB logit failure priors, Step 10 sequences, and Step 11 strata on a real Eedi slice."""
    if not RAW_DIR.exists():
        pytest.skip("Raw Eedi CSV directory missing")

    from src.data.eedi_preprocessor import (
        EediTask34Preprocessor,
        GLOBAL_ACC_PRIOR,
        KAPPA,
        M_ITEM,
        NUM_QUESTIONS,
    )

    pre = EediTask34Preprocessor()
    step1 = pre.run_step1_audit_and_mappings()
    assert len(step1["unique_rep_qids"]) == 947
    assert len(step1["l1_indices"]) == 3
    assert len(step1["leaf_to_l1"]) == 57
    assert step1["question_leaf_kcs"].shape == (948, 6)
    assert step1["question_all_kcs"].shape == (948, 5)
    assert len(step1["expert_pairs"]) == 50

    # Slice 200 real training users (40 per fold) & 30 public test users
    train_df = step1["train_df"]
    sampled_uids = sorted(int(u) for u in train_df["UserId"].unique()[:200])
    train_sub = train_df[train_df["UserId"].isin(sampled_uids)].copy()
    split_sub = {f"fold_{f}": sampled_uids[f::5] for f in range(5)}

    step9 = pre.run_step9_fold_priors(train_sub, split_sub, step1)
    nq = step9["N_q_folds"]
    dq = step9["D_q_folds"]
    mu_p = step9["mu_prior_folds"]
    mu_c = step9["mu_concept_folds"]
    omega = step9["omega_q_folds"]

    assert torch.allclose(mu_c, torch.log(mu_p / (1.0 - mu_p)), atol=1e-5)
    assert torch.allclose(omega, nq.float() / (nq.float() + KAPPA), atol=1e-6)
    # Check exact item error-rate direction (high accuracy in D_train^(0) => low failure D_q)
    tr0 = train_sub[train_sub["UserId"].isin(sampled_uids) & (~train_sub["UserId"].isin(split_sub["fold_0"]))]
    q_stats = tr0.groupby("QuestionId")["IsCorrect"].agg(["count", "mean"])
    q_test = int(q_stats.index[0])
    n0 = float(q_stats.loc[q_test, "count"])
    err0 = (1.0 - float(q_stats.loc[q_test, "mean"])) * n0
    expected_d = (err0 + M_ITEM * float(mu_p[0, q_test])) / (n0 + M_ITEM)
    assert abs(float(dq[0, q_test]) - max(1e-4, min(1.0 - 1e-4, expected_d))) < 1e-5

    # Build Step 10 sequences on a user with >= 3 steps and verify Laplace running_acc & 8-dim post_diag_vec
    seq_res = pre._build_split_sequences(train_sub[train_sub["UserId"].isin(sampled_uids[:10])], step1, step9["fold_dominant_distractor"], has_targets=False)
    sm0 = seq_res["seq_mask"][0]
    assert abs(float(seq_res["running_acc_prev"][0, 0]) - GLOBAL_ACC_PRIOR) < 1e-6
    r0 = float(seq_res["r_seq"][0, 0])
    assert abs(float(seq_res["running_acc_prev"][0, 1]) - (r0 + 2.0 * GLOBAL_ACC_PRIOR) / 3.0) < 1e-5
    pdv = seq_res["post_diag_vec"][0]
    assert pdv.shape[-1] == 8
    assert torch.allclose(pdv[0][sm0][..., :4].sum(dim=-1), torch.ones(int(sm0.sum())))

    # Build Step 10 static_ctx (normalized in [0,1]) and Step 11 strata masks
    pub_sub = step1["pub_df"][step1["pub_df"]["UserId"].isin(step1["pub_df"]["UserId"].unique()[:20])]
    pub_seq = pre._build_split_sequences(pub_sub, step1, step9["fold_dominant_distractor"], has_targets=True)
    dummy_layout = {
        q: {
            "layout_type": ["horiz_below", "grid_std", "vert_right"][q % 3],
            "stem_struct_density": 0.01,
            "calculator_allowed_flag": 0.0,
        }
        for q in range(NUM_QUESTIONS)
    }
    dummy_mod = torch.zeros((NUM_QUESTIONS, 3), dtype=torch.float32)
    dummy_mod[::2, 0] = 1.0
    dummy_mod[:, 1] = 1.0
    static_ctx = pre._build_question_static_ctx(
        step1,
        dummy_layout,
        {},
        {},
        {"modality_masks": dummy_mod, "patch_valid_mask": torch.ones((NUM_QUESTIONS, 49), dtype=torch.bool)},
        {"acd_mask_txt_q": torch.ones((NUM_QUESTIONS, 32), dtype=torch.bool), "acd_mask_txt_s": torch.ones((NUM_QUESTIONS, 40), dtype=torch.bool)},
    )
    assert bool((static_ctx[:, 0] == 1.0).all())
    assert float(static_ctx.min()) >= 0.0 and float(static_ctx.max()) <= 1.0

    q_counts = torch.from_numpy(np.bincount(train_sub["QuestionId"].to_numpy(dtype=np.int64), minlength=NUM_QUESTIONS)).long()
    strata = pre._compute_test_strata_masks(pub_seq, dummy_mod, q_counts, step1, dummy_layout)
    em = pub_seq["eval_mask"].bool()
    assert int((strata["layout_horizontal"] | strata["layout_grid"] | strata["layout_vertical"]).sum()) == int(em.sum())
    assert int((strata["after_error"] | strata["after_correct"]).sum()) == int(em.sum())
    assert int((strata["single_kc"] | strata["multi_kc"]).sum()) == int(em.sum())


def test_unit_acd_mmkt_v3_eedi_compatibility_and_explainability():
    """Verify ACDMMKTv3 Eedi compatibility: m_img^s==0 zero orphan grads, diagnostic post_diag, Ablations 1-3, RoBERTa filtering."""
    torch.manual_seed(0)
    num_q, num_kc, l_q, l_s, seq_len = 32, 85, 64, 96, 12

    mod_masks = torch.zeros((num_q, 3), dtype=torch.float32)
    mod_masks[:16, 0] = 1.0  # Half diagram stems
    mod_masks[:, 1] = 1.0     # 100% text solutions
    mod_masks[:, 2] = 0.0     # Zero solution images (Eedi signature!)

    H = torch.zeros((num_kc, num_q), dtype=torch.float32)
    leaf_kcs = torch.zeros((num_q, 6), dtype=torch.int64)
    kc_mask = torch.zeros((num_q, 6), dtype=torch.bool)
    for q in range(num_q):
        c1, c2 = q % 57, (q + 7) % 57
        leaf_kcs[q, 0] = c1
        leaf_kcs[q, 1] = c2
        kc_mask[q, :2] = True
        H[c1, q] = 1.0
        H[c2, q] = 1.0
        H[57 + (q % 28), q] = 0.5
    G = torch.eye(num_kc, dtype=torch.float32)

    mu_p = torch.full((5, num_q), 0.4627, dtype=torch.float32)
    mu_c = torch.log(mu_p / (1.0 - mu_p))  # ~ -0.1495 logit
    s_ctx = torch.rand((num_q, 15), dtype=torch.float32)
    s_ctx[:, 0] = 1.0

    bank = {
        "modality_masks": mod_masks,
        "question_leaf_kcs": leaf_kcs,
        "question_kc_mask": kc_mask,
        "question_static_ctx": s_ctx,
        "acd_E_txt_q": torch.randn((num_q, l_q, 768)).half(),
        "acd_mask_txt_q": torch.ones((num_q, l_q), dtype=torch.bool),
        "acd_E_txt_s": torch.randn((num_q, l_s, 768)).half(),
        "acd_mask_txt_s": torch.ones((num_q, l_s), dtype=torch.bool),
        "acd_E_img_q": torch.randn((num_q, 49, 768)).half(),
        "acd_E_img_s": torch.zeros((num_q, 49, 768)).half(),
        "acd_kc_init_emb": torch.randn((num_kc, 768), dtype=torch.float32),
        "acd_hypergraph_G": G,
        "M_patch_q": torch.zeros((num_q, 49), dtype=torch.float32),
        "N_q_folds": torch.full((5, num_q), 50, dtype=torch.int64),
        "D_q_folds": mu_p.clone(),
        "mu_concept_folds": mu_c,
        "d_post_diag": 8,
    }

    q_seq = torch.randint(0, num_q, (3, seq_len))
    r_seq = torch.randint(0, 2, (3, seq_len))
    seq_mask = torch.ones((3, seq_len), dtype=torch.bool)
    loss_mask = seq_mask.clone()
    loss_mask[:, 0] = False
    post_diag_vec = torch.zeros((3, seq_len, 8), dtype=torch.float32)
    post_diag_vec[:, :, 1] = 1.0
    post_diag_vec[:, :, 6] = 0.65

    batch = {
        "q_seq": q_seq,
        "r_seq": r_seq,
        "seq_mask": seq_mask,
        "loss_mask": loss_mask,
        "eval_mask": loss_mask,
        "modality_masks": mod_masks[q_seq],
        "raw_modality_masks": mod_masks[q_seq],
        "norm_delta_t_lag": torch.rand((3, seq_len)),
        "norm_delta_t_kc": torch.rand((3, seq_len)),
        "post_diag_vec": post_diag_vec,
    }

    # 1. No solution image projection parameters & dataset-mean default mu_concept
    model = ACDMMKTv3(
        static_item_bank=bank,
        d_model=32,
        num_heads=4,
        max_seq_len=seq_len,
        post_diag_mode="diagnostic",
    )
    assert model.sub1.proj_img_sol is None
    assert model.sub1.norm_img_sol is None
    assert model.sub1.gate_sol_img is None
    assert abs(model.default_mu_concept - float(mu_c.mean().item())) < 1e-5
    assert model.d_post_diag == 8

    out = model(batch, mu_concept_prior=mu_c[0])
    crit = ACDMMKTv3Loss(beta_d=0.20, lambda_sparse=0.01, kappa=10.0)
    loss, _ = crit(out, r_seq, loss_mask, bank["N_q_folds"][0], bank["D_q_folds"][0])
    assert torch.isfinite(loss)
    loss.backward()
    for pname, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"Invalid/orphan grad for {pname}"

    # 2. Ablations 1-3 dynamic L_q / L_s checks + strict zero-leakage noise invariance on Eedi bank
    bank_noisy = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in bank.items()}
    bank_noisy["acd_E_txt_q"] = bank_noisy["acd_E_txt_q"] + 1e3
    bank_noisy["acd_E_txt_s"] = bank_noisy["acd_E_txt_s"] + 1e3
    bank_noisy["acd_E_img_q"] = bank_noisy["acd_E_img_q"] + 1e3
    m1a = ACDMMKTv3(static_item_bank=bank, d_model=32, num_heads=4, max_seq_len=seq_len, use_multimodal=False).eval()
    m1b = ACDMMKTv3(static_item_bank=bank_noisy, d_model=32, num_heads=4, max_seq_len=seq_len, use_multimodal=False).eval()
    m1b.load_state_dict(m1a.state_dict())
    with torch.no_grad():
        assert torch.allclose(m1a(batch, mu_concept_prior=mu_c[0])["probs"], m1b(batch, mu_concept_prior=mu_c[0])["probs"], atol=1e-5)

    bank_sol_noisy = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in bank.items()}
    bank_sol_noisy["acd_E_txt_s"] = bank_sol_noisy["acd_E_txt_s"] + 1e3
    m2a = ACDMMKTv3(static_item_bank=bank, d_model=32, num_heads=4, max_seq_len=seq_len, use_solution=False).eval()
    m2b = ACDMMKTv3(static_item_bank=bank_sol_noisy, d_model=32, num_heads=4, max_seq_len=seq_len, use_solution=False).eval()
    m2b.load_state_dict(m2a.state_dict())
    with torch.no_grad():
        assert torch.allclose(m2a(batch, mu_concept_prior=mu_c[0])["probs"], m2b(batch, mu_concept_prior=mu_c[0])["probs"], atol=1e-5)

    bank_vis_noisy = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in bank.items()}
    bank_vis_noisy["acd_E_img_q"] = bank_vis_noisy["acd_E_img_q"] + 1e3
    m3a = ACDMMKTv3(static_item_bank=bank, d_model=32, num_heads=4, max_seq_len=seq_len, use_visual=False).eval()
    m3b = ACDMMKTv3(static_item_bank=bank_vis_noisy, d_model=32, num_heads=4, max_seq_len=seq_len, use_visual=False).eval()
    m3b.load_state_dict(m3a.state_dict())
    with torch.no_grad():
        assert torch.allclose(m3a(batch, mu_concept_prior=mu_c[0])["probs"], m3b(batch, mu_concept_prior=mu_c[0])["probs"], atol=1e-5)

    # 3. RoBERTa special tokens (<pad>, <s>, </s>, <unk>) filtered cleanly by ExplainabilityEngine
    model.eval()
    engine = ACDv3ExplainabilityEngine(model=model, device=torch.device("cpu"))
    toks_q = [["<s>", "<pad>", "</s>", "<unk>", "Ġtriangle", "Ġangle"] + ["<pad>"] * (l_q - 6) for _ in range(num_q)]
    toks_s = [["<s>", "</s>", "Ġtheorem", "<unk>", "Ġsum"] + ["<pad>"] * (l_s - 5) for _ in range(num_q)]
    engine.aux_metadata = {"token_to_word_q": toks_q, "token_to_word_s": toks_s}
    expl = engine.explain_interaction(batch, sample_idx=0, target_step=seq_len - 1, top_k_keywords=5, mu_concept_prior=mu_c[0])
    for kw in expl["tier3_multimodal"]["stem_keywords"]:
        assert kw["token"] not in ("<s>", "</s>", "<pad>", "<unk>")
    for st in expl["tier3_multimodal"]["solution_steps"]:
        assert st["token"] not in ("<s>", "</s>", "<pad>", "<unk>")
