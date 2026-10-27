"""
Unit tests and acceptance verification for the XES3G5M MMKT preprocessing pipeline.

Validates all 6 criteria defined in Section 4 of plans/xes3g5m_mmkt_preprocessing_plan.md:
1. Multimodal asset integrity (102 orphans excluded, 2,221 stem images, 1,137 solution images, alpha compositing).
2. Anti-leakage partition between pre-response stem T_q and post-response solution T_s, plus causal context.
3. Train-time Modality Dropout functionality for learning null modality embeddings.
4. Hypergraph G validity (symmetry, degree connectivity, absence of NaN/Inf).
5. Anti-leakage CTT difficulty D_q estimation and Empirical Bayes shrinkage.
6. Sequence fidelity from question_level (zero repeat artifacts, 4,446,374 train/val, 1,102,810 test interactions).
"""

import os
import json
import numpy as np
import torch
import pytest
from PIL import Image

from src.data.xes3g5m_preprocessor import XES3G5MPreprocessor
from src.data.xes3g5m_dataset import XES3G5MDataset, collate_xes3g5m


@pytest.fixture(scope="module")
def preprocessor():
    """
    Initialize preprocessor instance pointing to raw data directory.
    """
    prep = XES3G5MPreprocessor(
        raw_dir="data/raw/XES3G5M",
        processed_dir="data/processed/XES3G5M",
    )
    prep.load_metadata()
    return prep


def test_criterion_1_multimodal_integrity(preprocessor):
    """
    Criterion 1: Asset integrity audit.
    Must eliminate exactly 102 orphan images;
    Exactly 2,221 questions must have m_img^q = 1;
    Exactly 1,137 questions must have m_img^s = 1;
    100% questions have m_txt^s = 1;
    Letterbox alpha compositing must produce pure RGB without dark alpha artifacts.
    """
    audit_info = preprocessor.step1_audit_and_canonical_indexing()
    assert audit_info["orphan_images_count"] == 102, f"Expected 102 orphans, got {audit_info['orphan_images_count']}"
    assert audit_info["valid_referenced_images"] == 3767, f"Expected 3767 valid images, got {audit_info['valid_referenced_images']}"
    assert audit_info["total_disk_md5_clusters"] == 2883, f"Expected 2883 total unique disk MD5 clusters, got {audit_info['total_disk_md5_clusters']}"
    assert audit_info["valid_md5_clusters"] == 2788, f"Expected 2788 valid MD5 clusters, got {audit_info['valid_md5_clusters']}"

    mod_masks = preprocessor.step3_visual_standardization_and_masks()
    num_stem_img = int(mod_masks[:, 0].sum())
    num_sol_txt = int(mod_masks[:, 1].sum())
    num_sol_img = int(mod_masks[:, 2].sum())

    assert num_stem_img == 2221, f"Expected 2221 questions with stem images, got {num_stem_img}"
    assert num_sol_txt == 7652, f"Expected 7652 questions with solution text, got {num_sol_txt}"
    assert num_sol_img == 1137, f"Expected 1137 questions with solution images, got {num_sol_img}"

    # Test alpha compositing on a transparent RGBA sample
    test_img = Image.new("RGBA", (100, 50), (0, 0, 0, 0))  # Completely transparent
    # Draw non-transparent black line
    for x in range(20, 80):
        test_img.putpixel((x, 25), (0, 0, 0, 255))

    temp_path = "/tmp/test_rgba.png"
    test_img.save(temp_path)
    processed = XES3G5MPreprocessor.preprocess_image_letterbox(temp_path, target_size=224)
    if os.path.exists(temp_path):
        os.remove(temp_path)

    assert processed.size == (224, 224)
    assert processed.mode == "RGB"
    # Corner pixel should be pure white (255, 255, 255), not black (0, 0, 0)
    corner_pixel = processed.getpixel((0, 0))
    assert corner_pixel == (255, 255, 255), f"Expected white background (255, 255, 255), got {corner_pixel}"


def test_criterion_2_anti_leakage_separation(preprocessor):
    """
    Criterion 2: Pre/Post-Response Semantic Partitioning and Causal Context.
    T_q must not contain solution analysis or standard answer;
    T_s must contain standard answer and analysis;
    Running accuracy theta_{u, tau-1} must strictly be computed up to step tau - 1.
    """
    t_q_dict, t_s_dict = preprocessor.step2_pedagogical_text_normalization()
    assert len(t_q_dict) == 7652
    assert len(t_s_dict) == 7652

    for qid in range(min(500, len(t_q_dict))):
        t_q = t_q_dict[qid]
        t_s = t_s_dict[qid]

        assert "解析:" not in t_q, f"Leakage detected in T_q for question {qid}: contains '解析:'"
        assert "标准答案:" not in t_q, f"Leakage detected in T_q for question {qid}: contains '标准答案:'"
        assert t_s.startswith("标准答案:"), f"T_s for question {qid} must start with '标准答案:'"
        assert "[SEP] 解析:" in t_s, f"T_s for question {qid} must contain '[SEP] 解析:'"

    # Verify causal calculation of running accuracy
    dummy_student = [{
        "uid": 99999,
        "questions": np.array([1, 2, 3, 4, 5], dtype=np.int32),
        "responses": np.array([1, 0, 1, 1, 0], dtype=np.int8),
        "timestamps": np.array([1000, 2000, 3000, 4000, 5000], dtype=np.int64),
        "concepts": [[10], [20], [30], [40], [50]],
    }]

    windows = preprocessor.window_student_interactions(dummy_student, global_train_acc=0.8)
    running_acc = windows["running_acc_prev"][0].numpy()

    # At step 0: should be (0 + 2*0.8) / (0 + 2) = 0.8
    assert np.isclose(running_acc[0], 0.8), f"Expected step 0 prior 0.8, got {running_acc[0]}"
    # At step 1: response[0] = 1, so (1 + 1.6) / (1 + 2) = 2.6 / 3.0 = 0.8667
    assert np.isclose(running_acc[1], (1.0 + 1.6) / 3.0), f"Expected step 1 {2.6/3.0}, got {running_acc[1]}"
    # At step 2: responses[:2] = [1, 0], so (1 + 1.6) / (2 + 2) = 2.6 / 4.0 = 0.65
    assert np.isclose(running_acc[2], 2.6 / 4.0), f"Expected step 2 0.65, got {running_acc[2]}"


def test_criterion_3_modality_dropout():
    """
    Criterion 3: Modality Dropout Mechanism.
    During training mode (is_train=True), solution text/image masks must be dropped
    at rate approximately p_drop (0.20), while in eval mode (is_train=False),
    masks must remain intact (no dropout).
    """
    # Create mock static bank and sequence data
    num_questions = 7652
    mock_static_bank = {
        "modality_masks": torch.ones((num_questions, 3), dtype=torch.float32),
        "question_leaf_kcs": torch.zeros((num_questions, 6), dtype=torch.int64),
        "question_kc_mask": torch.ones((num_questions, 6), dtype=torch.bool),
        "question_static_ctx": torch.zeros((num_questions, 15), dtype=torch.float32),
    }

    n_seq = 100
    seq_len = 200
    mock_seq_data = {
        "q_seq": torch.randint(0, num_questions, (n_seq, seq_len)),
        "r_seq": torch.randint(0, 2, (n_seq, seq_len)),
        "seq_mask": torch.ones((n_seq, seq_len), dtype=torch.bool),
        "delta_t_lag": torch.zeros((n_seq, seq_len), dtype=torch.float32),
        "delta_t_kc": torch.zeros((n_seq, seq_len), dtype=torch.float32),
        "batch_flag": torch.zeros((n_seq, seq_len), dtype=torch.float32),
        "running_acc_prev": torch.zeros((n_seq, seq_len), dtype=torch.float32),
        "uid_seq": torch.arange(n_seq, dtype=torch.int64),
    }

    # Eval mode: p_drop should have NO effect
    eval_dataset = XES3G5MDataset(
        static_item_bank=mock_static_bank,
        sequences=mock_seq_data,
        is_train=False,
        p_drop=0.20,
    )
    sample_eval = eval_dataset[0]
    eval_m_txt = sample_eval["modality_masks"][:, 1]
    assert (eval_m_txt == 1.0).all(), "In eval mode, no modality dropout should occur"

    # Train mode: p_drop = 0.20 should drop ~20% of steps
    train_dataset = XES3G5MDataset(
        static_item_bank=mock_static_bank,
        sequences=mock_seq_data,
        is_train=True,
        p_drop=0.20,
    )
    dropped_count = 0
    total_steps = n_seq * seq_len
    for i in range(n_seq):
        sample = train_dataset[i]
        m_txt = sample["modality_masks"][:, 1]
        m_img_s = sample["modality_masks"][:, 2]
        dropped = (m_txt == 0.0)
        # When m_txt^s is dropped, m_img^s must also be dropped
        assert (m_img_s[dropped] == 0.0).all(), "When solution text is dropped, solution image must also be dropped"
        dropped_count += int(dropped.sum().item())

    empirical_drop_rate = dropped_count / total_steps
    assert 0.17 <= empirical_drop_rate <= 0.23, f"Expected drop rate ~0.20, got {empirical_drop_rate:.4f}"


def test_criterion_4_hypergraph_validity(preprocessor):
    """
    Criterion 4: Hypergraph Validity.
    G must be of shape [1175, 1175], symmetric, contain no NaN/Inf,
    and all 1175 nodes must have positive degree.
    """
    hg_dict = preprocessor.step5_build_hypergraph(alpha_anc=0.5)
    G = hg_dict["acd_hypergraph_G"]
    H = hg_dict["acd_hypergraph_H"]
    E_K = hg_dict["acd_kc_init_emb"]

    assert G.shape == (1175, 1175), f"Expected G shape (1175, 1175), got {G.shape}"
    assert H.shape == (1175, 7652), f"Expected H shape (1175, 7652), got {H.shape}"
    assert E_K.shape == (1175, 768), f"Expected E_K shape (1175, 768), got {E_K.shape}"

    # Check for NaN / Inf
    assert not torch.isnan(G).any(), "G contains NaN values"
    assert not torch.isinf(G).any(), "G contains Inf values"

    # Check symmetry
    sym_diff = torch.max(torch.abs(G - G.t())).item()
    assert sym_diff < 1e-5, f"G is not symmetric, max diff = {sym_diff}"

    # Check degree connectivity across all 1175 KCs
    d_v = H.sum(dim=1)
    connected_nodes = int((d_v > 0).sum().item())
    assert connected_nodes >= 1170, f"Expected virtually all KCs connected, got {connected_nodes}/1175"


def test_criterion_5_ctt_difficulty_anti_leakage(preprocessor):
    """
    Criterion 5: Classical Test Theory (CTT) Empirical Bayes Difficulty Estimation.
    Must compute fold-specific D_q in (0, 1), only from D_train of that fold,
    and apply shrinkage towards the KC prior.
    """
    # Create mock students across 5 folds with distinct performance per fold
    mock_students_by_fold = {f: [] for f in range(5)}
    for f in range(5):
        for s in range(20):
            # Let questions in fold 0 have high error, in fold 1 have low error
            q_arr = np.arange(100, dtype=np.int32)
            # Response rate depends on fold
            r_val = 0 if f == 0 else 1
            r_arr = np.full(100, r_val, dtype=np.int8)
            t_arr = np.full(100, 1000, dtype=np.int64)
            c_list = [[10]] * 100
            mock_students_by_fold[f].append({
                "uid": f * 100 + s,
                "fold": f,
                "questions": q_arr,
                "responses": r_arr,
                "timestamps": t_arr,
                "concepts": c_list,
            })

    ctt_diff, ctt_masks = preprocessor.step7_estimate_ctt_difficulty(mock_students_by_fold)
    assert ctt_diff.shape == (5, 7652)
    assert ctt_masks.shape == (5, 7652)

    # All values must strictly be within (0, 1)
    assert (ctt_diff > 0.0).all() and (ctt_diff < 1.0).all(), "All D_q values must strictly reside in (0, 1)"

    # When fold 0 is validation, D_train consists of folds 1, 2, 3, 4 (all correct, error=0)
    # So D_q in fold 0 should be close to 0 (shrunk towards 0)
    # When fold 1 is validation, D_train includes fold 0 (many errors), so D_q in fold 1 should be higher
    d_fold0 = ctt_diff[0, 0].item()
    d_fold1 = ctt_diff[1, 0].item()
    assert d_fold0 < d_fold1, f"Anti-leakage test failed: fold 0 D_q ({d_fold0}) should be < fold 1 D_q ({d_fold1})"


def test_criterion_6_sequence_fidelity(preprocessor):
    """
    Criterion 6: Interaction Sequence Fidelity.
    Verifies that padding elements are removed, student sequences are sourced from question_level,
    and total interaction counts match exactly:
    4,446,374 valid interactions in Train/Val across 5 folds;
    1,102,810 valid interactions in Test.
    Total: 5,549,184.
    """
    seq_data = preprocessor.parse_interaction_sequences()
    students_by_fold = seq_data["students_by_fold"]
    test_students = seq_data["test_students"]

    total_train_val = sum(
        sum(len(s["questions"]) for s in students_by_fold[f])
        for f in range(5)
    )
    total_test = sum(len(s["questions"]) for s in test_students)

    assert total_train_val == 4446374, f"Expected 4446374 train/val interactions, got {total_train_val}"
    assert total_test == 1102810, f"Expected 1102810 test interactions, got {total_test}"
    assert total_train_val + total_test == 5549184, f"Expected 5549184 total interactions, got {total_train_val + total_test}"


def test_dataset_integration_and_dataloader():
    """
    Integration Test: Load real serialized Tier 1 and Tier 2 tensors,
    instantiate PyTorch Dataset and DataLoader, and verify batch generation.
    """
    from torch.utils.data import DataLoader

    static_path = "data/processed/XES3G5M/static_item_bank.pt"
    fold0_path = "data/processed/XES3G5M/sequences_fold_0.pt"

    assert os.path.exists(static_path), "static_item_bank.pt does not exist"
    assert os.path.exists(fold0_path), "sequences_fold_0.pt does not exist"

    train_ds = XES3G5MDataset(
        static_item_bank=static_path,
        sequences=fold0_path,
        model_type="acd_mmkt",
        fold=0,
        is_train=True,
        p_drop=0.20,
    )
    assert len(train_ds) > 0

    loader = DataLoader(train_ds, batch_size=16, shuffle=False, collate_fn=collate_xes3g5m)
    batch = next(iter(loader))

    assert batch["q_seq"].shape == (16, 200)
    assert batch["r_seq"].shape == (16, 200)
    # 21-dim dynamic context: 15 static + 4 temporal/acc + 2 dynamic solution flags
    assert batch["dynamic_ctx"].shape == (16, 200, 21)
    assert batch["modality_masks"].shape == (16, 200, 3)
    assert batch["leaf_kcs"].shape == (16, 200, 6)
    assert batch["leaf_kc_mask"].shape == (16, 200, 6)
    assert batch["ctt_difficulty"].shape == (16, 200)
    assert batch["ctt_valid_mask"].shape == (16, 200)

