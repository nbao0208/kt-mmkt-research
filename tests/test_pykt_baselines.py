"""
Unit and integration tests for pyKT baseline models (DKT, DKVMN, SAKT).

Verifies:
1. Native pyKT architecture preservation (models instantiated directly from pykt.models).
2. Hardware device synchronization and monkey-patch safety on macOS MPS/CPU.
3. Forward propagation and loss computation across all 3 baseline models.
4. Anti-leakage unrolled question-level inference on multi-concept questions.
5. 1-to-1 alignment of test predictions with exactly 1,099,197 target interactions.
6. Paired statistical test computations.
"""

import os
import pytest
import numpy as np
import torch
import torch.nn as nn
from pykt.models.dkt import DKT
from pykt.models.dkvmn import DKVMN
from pykt.models.sakt import SAKT

from src.models.baselines.pykt_models import init_pykt_baseline, sync_pykt_device
from src.data.pykt_dataset import load_test_grouped_data, load_stratification_metadata
from src.evaluation.pykt_evaluator import PyKTEvaluator


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def test_native_pykt_model_instantiation(device):
    """
    Criterion 1: Ensure all three models are instantiated directly from pykt.models,
    without using hand-crafted reimplementations.
    """
    dkt = init_pykt_baseline("dkt", {"num_c": 865, "emb_size": 128, "dropout": 0.2}, device=device)
    assert isinstance(dkt, DKT), "DKT must be an instance of pykt.models.dkt.DKT"

    dkvmn = init_pykt_baseline("dkvmn", {"num_c": 865, "dim_s": 128, "size_m": 50, "dropout": 0.2}, device=device)
    assert isinstance(dkvmn, DKVMN), "DKVMN must be an instance of pykt.models.dkvmn.DKVMN"

    sakt = init_pykt_baseline("sakt", {"num_c": 865, "seq_len": 200, "emb_size": 128, "num_attn_heads": 4, "dropout": 0.2}, device=device)
    assert isinstance(sakt, SAKT), "SAKT must be an instance of pykt.models.sakt.SAKT"


def test_device_synchronization(device):
    """
    Ensure device synchronization updates pyKT's internal device variable.
    """
    import pykt.models.utils as p_utils
    sync_pykt_device(device)
    assert p_utils.device == device


def test_model_forward_and_loss(device):
    """
    Verify forward propagation and float32 masked BCE loss on the active device.
    """
    batch_size = 4
    seq_len = 20
    num_c = 865

    c = torch.randint(0, num_c, (batch_size, seq_len), device=device)
    r = torch.randint(0, 2, (batch_size, seq_len), device=device)
    cshft = torch.randint(0, num_c, (batch_size, seq_len), device=device)
    rshft = torch.randint(0, 2, (batch_size, seq_len), device=device)
    sm = (torch.rand(batch_size, seq_len, device=device) > 0.3)

    # 1. DKT
    dkt = init_pykt_baseline("dkt", {"num_c": num_c, "emb_size": 64}, device=device)
    dkt.train()
    y_dkt = dkt(c.long(), r.long())
    y_dkt = (y_dkt * torch.nn.functional.one_hot(cshft.long(), num_c)).sum(-1)
    loss_dkt = torch.nn.functional.binary_cross_entropy(
        torch.masked_select(y_dkt, sm).float(),
        torch.masked_select(rshft, sm).float(),
    )
    assert not torch.isnan(loss_dkt)
    loss_dkt.backward()

    # 2. DKVMN
    dkvmn = init_pykt_baseline("dkvmn", {"num_c": num_c, "dim_s": 64, "size_m": 20}, device=device)
    dkvmn.train()
    cc = torch.cat((c[:, 0:1], cshft), dim=1)
    cr = torch.cat((r[:, 0:1], rshft), dim=1)
    y_dkvmn = dkvmn(cc.long(), cr.long())[:, 1:]
    loss_dkvmn = torch.nn.functional.binary_cross_entropy(
        torch.masked_select(y_dkvmn, sm).float(),
        torch.masked_select(rshft, sm).float(),
    )
    assert not torch.isnan(loss_dkvmn)
    loss_dkvmn.backward()

    # 3. SAKT
    sakt = init_pykt_baseline("sakt", {"num_c": num_c, "seq_len": seq_len, "emb_size": 64, "num_attn_heads": 4}, device=device)
    sakt.train()
    y_sakt = sakt(c.long(), r.long(), cshft.long())
    loss_sakt = torch.nn.functional.binary_cross_entropy(
        torch.masked_select(y_sakt, sm).float(),
        torch.masked_select(rshft, sm).float(),
    )
    assert not torch.isnan(loss_sakt)
    loss_sakt.backward()


def test_test_question_grouping_count():
    """
    Criterion 4: Verify test question grouping matches exactly 1,099,197 evaluated interactions
    across all 3,613 students.
    """
    students = load_test_grouped_data("data/raw/XES3G5M/kc_level/test.csv")
    assert len(students) == 3613, f"Expected 3,613 students, got {len(students)}"

    total_eval = sum(len(s["eval_questions"]) for s in students)
    assert total_eval == 1099197, f"Expected exactly 1,099,197 evaluated questions, got {total_eval}"


def test_pykt_5fold_raw_student_isolation():
    """
    Verify strict 5-Fold student partition isolation directly on raw XES3G5M CSV files:
    - 0% student overlap between any pair of cross-validation folds.
    - 0% student overlap between any fold and the independent test set.
    """
    import pandas as pd

    train_csv = "data/raw/XES3G5M/kc_level/train_valid_sequences.csv"
    test_csv = "data/raw/XES3G5M/kc_level/test.csv"
    assert os.path.exists(train_csv), f"Missing raw file: {train_csv}"
    assert os.path.exists(test_csv), f"Missing raw file: {test_csv}"

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    fold_uids = [set(df_train[df_train["fold"] == f]["uid"].unique()) for f in range(5)]
    test_uids = set(df_test["uid"].unique())

    assert len(test_uids) == 3613, f"Expected 3,613 test students, got {len(test_uids)}"
    total_train_students = sum(len(u) for u in fold_uids)
    assert total_train_students == 14453, f"Expected 14,453 train students, got {total_train_students}"

    # Pairwise disjointness between folds
    for i in range(5):
        for j in range(i + 1, 5):
            overlap = fold_uids[i].intersection(fold_uids[j])
            assert len(overlap) == 0, f"Student leakage between Fold {i} and Fold {j}: {len(overlap)} shared students!"

    # Disjointness with test set
    for f in range(5):
        overlap = fold_uids[f].intersection(test_uids)
        assert len(overlap) == 0, f"Student leakage between Fold {f} and Test set: {len(overlap)} shared students!"


def test_paired_statistical_testing():
    """
    Verify paired t-test and Wilcoxon test output structure.
    """
    base_aucs = [0.7510, 0.7525, 0.7505, 0.7518, 0.7530]
    prop_aucs = [0.7710, 0.7730, 0.7700, 0.7725, 0.7735]

    res = PyKTEvaluator.compute_paired_statistical_tests(base_aucs, prop_aucs)
    assert res["mean_diff"] > 0
    assert res["p_value_ttest"] < 0.05
    assert res["p_value_wilcoxon"] <= 0.07


def test_anti_leakage_multi_concept_question_inference(device):
    """
    Criterion 4 [H5 Fix]: Verify strict anti-leakage question-level inference on multi-concept questions.

    Construct synthetic student sequences with multi-concept questions.
    Invert response r_{t_0} -> 1 - r_{t_0} at multi-concept question t_0 = 1.
    Verify that across all 3 models (DKT, DKVMN, SAKT), question predictions for t <= t_0
    remain strictly invariant (diff < 1e-6), proving zero leakage of r_{t_0}.
    """
    student_a = {
        "uid": 1,
        "all_questions": [
            {"q_id": 100, "concepts": [10], "response": 1},
            {"q_id": 101, "concepts": [20, 30, 40], "response": 0},  # t_0: multi-concept, response 0
            {"q_id": 102, "concepts": [50, 60], "response": 1},
            {"q_id": 103, "concepts": [70], "response": 0},
        ],
        "eval_questions": [
            {"q_id": 101, "concepts": [20, 30, 40], "response": 0},
            {"q_id": 102, "concepts": [50, 60], "response": 1},
            {"q_id": 103, "concepts": [70], "response": 0},
        ],
    }
    student_b = {
        "uid": 1,
        "all_questions": [
            {"q_id": 100, "concepts": [10], "response": 1},
            {"q_id": 101, "concepts": [20, 30, 40], "response": 1},  # t_0: inverted to response 1
            {"q_id": 102, "concepts": [50, 60], "response": 1},
            {"q_id": 103, "concepts": [70], "response": 0},
        ],
        "eval_questions": [
            {"q_id": 101, "concepts": [20, 30, 40], "response": 1},
            {"q_id": 102, "concepts": [50, 60], "response": 1},
            {"q_id": 103, "concepts": [70], "response": 0},
        ],
    }

    # Initialize evaluator
    evaluator = PyKTEvaluator.__new__(PyKTEvaluator)
    evaluator.device = device
    sync_pykt_device(device)

    # 1. Test DKT
    dkt = init_pykt_baseline("dkt", {"num_c": 865, "emb_size": 64, "dropout": 0.0}, device=device)
    dkt.eval()
    evaluator.test_students = [student_a]
    preds_dkt_a = evaluator.predict_dkt_all(dkt)
    evaluator.test_students = [student_b]
    preds_dkt_b = evaluator.predict_dkt_all(dkt)

    assert abs(preds_dkt_a[0] - preds_dkt_b[0]) < 1e-6, "DKT leaked r_{t_0} into prediction at t_0!"
    assert abs(preds_dkt_a[1] - preds_dkt_b[1]) > 1e-6, "DKT state did not update at t > t_0!"

    # 2. Test DKVMN
    dkvmn = init_pykt_baseline("dkvmn", {"num_c": 865, "dim_s": 64, "size_m": 20, "dropout": 0.0}, device=device)
    dkvmn.eval()
    evaluator.test_students = [student_a]
    preds_dkvmn_a = evaluator.predict_dkvmn_all(dkvmn)
    evaluator.test_students = [student_b]
    preds_dkvmn_b = evaluator.predict_dkvmn_all(dkvmn)

    assert abs(preds_dkvmn_a[0] - preds_dkvmn_b[0]) < 1e-6, "DKVMN leaked r_{t_0} into prediction at t_0!"
    assert abs(preds_dkvmn_a[1] - preds_dkvmn_b[1]) > 1e-6, "DKVMN memory did not update at t > t_0!"

    # 3. Test SAKT
    sakt = init_pykt_baseline("sakt", {"num_c": 865, "seq_len": 200, "emb_size": 64, "num_attn_heads": 4, "dropout": 0.0}, device=device)
    sakt.eval()
    evaluator.test_students = [student_a]
    preds_sakt_a = evaluator.predict_sakt_all(sakt)
    evaluator.test_students = [student_b]
    preds_sakt_b = evaluator.predict_sakt_all(sakt)

    assert abs(preds_sakt_a[0] - preds_sakt_b[0]) < 1e-6, "SAKT leaked r_{t_0} into prediction at t_0!"
    assert abs(preds_sakt_a[1] - preds_sakt_b[1]) > 1e-6, "SAKT attention did not update at t > t_0!"


def test_one_to_one_target_and_qid_alignment_with_raw_and_mmkt_test_set():
    """
    Criterion 5 [H5 Fix]: Verify exact 1-to-1 alignment between pyKT question-level test set
    and raw test_quelevel.csv (and optional sequences_test.pt) across all 1,099,197 samples.
    """
    import pandas as pd

    students = load_test_grouped_data("data/raw/XES3G5M/kc_level/test.csv")
    strat_meta = load_stratification_metadata("data/raw/XES3G5M", students)

    assert len(strat_meta["targets"]) == 1099197
    assert len(strat_meta["q_ids"]) == 1099197

    # 1. Verify against raw test_quelevel.csv
    que_csv_path = "data/raw/XES3G5M/question_level/test_quelevel.csv"
    if os.path.exists(que_csv_path):
        df_que = pd.read_csv(que_csv_path)
        que_targets, que_qids = [], []
        for _, row in df_que.iterrows():
            qs = [int(x) for x in row["questions"].split(",")]
            rs = [int(x) for x in row["responses"].split(",")]
            for q, r in zip(qs[1:], rs[1:]):
                que_targets.append(r)
                que_qids.append(q)
        assert np.array_equal(strat_meta["targets"], np.array(que_targets, dtype=np.int64)), "Targets do not match raw test_quelevel.csv 1-to-1!"
        assert np.array_equal(strat_meta["q_ids"], np.array(que_qids, dtype=np.int64)), "Question IDs do not match raw test_quelevel.csv 1-to-1!"

    # 2. Optional alignment check with MMKT sequences_test.pt if present
    seq_pt_path = "data/processed/XES3G5M/sequences_test.pt"
    if os.path.exists(seq_pt_path):
        from src.data.xes3g5m_dataset import build_test_strided_windows
        test_seq_data = torch.load(seq_pt_path, map_location="cpu")
        w_dict, eval_mask = build_test_strided_windows(test_seq_data, max_seq_len=200, stride=100)
        mmkt_targets = w_dict["r_seq"][eval_mask].numpy()
        mmkt_qids = w_dict["q_seq"][eval_mask].numpy()
        assert np.array_equal(strat_meta["targets"], mmkt_targets), "Targets do not match MMKT test set 1-to-1!"
        assert np.array_equal(strat_meta["q_ids"], mmkt_qids), "Question IDs do not match MMKT test set 1-to-1!"


def test_sakt_right_padding_and_shifted_query_equivalence(device):
    """
    Criterion 3 [H1 & H5 Fix]: Verify that batched SAKT right-padding inference matches
    exact unpadded sequence evaluation across history lengths {1, 5, 50, 199} (diff < 1e-6).
    """
    sync_pykt_device(device)
    model = init_pykt_baseline("sakt", {"num_c": 865, "seq_len": 200, "emb_size": 64, "num_attn_heads": 4, "dropout": 0.0, "num_en": 2}, device=device)
    model.eval()

    test_lengths = [1, 5, 50, 199]
    for h_len in test_lengths:
        torch.manual_seed(42 + h_len)
        c_hist = torch.randint(0, 865, (h_len,), device=device)
        r_hist = torch.randint(0, 2, (h_len,), device=device)
        target_c = torch.tensor(100, device=device)

        # Unpadded single forward pass
        c_unpad = c_hist.unsqueeze(0)
        r_unpad = r_hist.unsqueeze(0)
        qry_unpad = torch.zeros((1, h_len), dtype=torch.long, device=device)
        if h_len > 1:
            qry_unpad[0, :h_len - 1] = c_hist[1:]
        qry_unpad[0, h_len - 1] = target_c

        with torch.no_grad():
            out_unpad = model(c_unpad, r_unpad, qry_unpad)
            pred_unpad = out_unpad[0, h_len - 1].item()

        # Padded forward pass (length 199)
        c_pad = torch.zeros((1, 199), dtype=torch.long, device=device)
        r_pad = torch.zeros((1, 199), dtype=torch.long, device=device)
        qry_pad = torch.zeros((1, 199), dtype=torch.long, device=device)

        c_pad[0, :h_len] = c_hist
        r_pad[0, :h_len] = r_hist
        if h_len > 1:
            qry_pad[0, :h_len - 1] = c_hist[1:]
        qry_pad[0, h_len - 1] = target_c

        with torch.no_grad():
            out_pad = model(c_pad, r_pad, qry_pad)
            pred_pad = out_pad[0, h_len - 1].item()

        diff = abs(pred_unpad - pred_pad)
        assert diff < 1e-6, f"SAKT right-padded output mismatch for h_len={h_len}: unpadded={pred_unpad}, padded={pred_pad}, diff={diff}"

