"""
Integration Tests for pyKT 5-Fold Cross-Validation and Context Windowing Protocol.

Verifies:
1. Strict 5-Fold student partition isolation (0% student UID leakage between folds and test).
2. Test set context-preserving strided windowing (100% coverage, 0 duplicate evaluations).
3. Masked BCE Loss numerical stability under boundary cases.
"""

import os
import pytest
import torch
import numpy as np

from src.data.xes3g5m_dataset import (
    XES3G5MDataset,
    build_test_strided_windows,
    load_static_item_bank_optimized,
)
from src.training.losses import MaskedBCELoss


@pytest.fixture
def processed_dir():
    return "data/processed/XES3G5M"


def test_pykt_5fold_student_isolation(processed_dir):
    """
    Verify strict isolation across the 5 cross-validation folds and the test set.
    There must be 0% overlap of student UIDs between any two folds or between fold and test.
    """
    fold_uids = []
    for f in range(5):
        fold_file = os.path.join(processed_dir, f"sequences_fold_{f}.pt")
        assert os.path.exists(fold_file), f"Fold file missing: {fold_file}"
        data = torch.load(fold_file, map_location="cpu")
        uids = set(data["uid_seq"].tolist())
        fold_uids.append(uids)

    # Check pairwise disjointness of all 5 folds
    for i in range(5):
        for j in range(i + 1, 5):
            overlap = fold_uids[i].intersection(fold_uids[j])
            assert len(overlap) == 0, f"Student leakage between Fold {i} and Fold {j}: {len(overlap)} shared students!"

    # Check isolation with Test set
    test_file = os.path.join(processed_dir, "sequences_test.pt")
    assert os.path.exists(test_file), f"Test file missing: {test_file}"
    test_data = torch.load(test_file, map_location="cpu")
    test_uids = set(test_data["uid_seq"].tolist())

    for f in range(5):
        overlap = fold_uids[f].intersection(test_uids)
        assert len(overlap) == 0, f"Student leakage between Fold {f} and Test set: {len(overlap)} shared students!"


def test_pykt_interaction_counts(processed_dir):
    """
    Verify exact match with pyKT interaction totals:
    Train/Val: exactly 4,446,374 valid interactions across 14,453 students.
    Test: exactly 1,102,810 valid interactions across 3,613 students.
    """
    total_train_val = 0
    total_train_students = set()
    for f in range(5):
        data = torch.load(os.path.join(processed_dir, f"sequences_fold_{f}.pt"), map_location="cpu")
        total_train_val += int(data["seq_mask"].sum().item())
        total_train_students.update(data["uid_seq"].tolist())

    assert total_train_val == 4446374, f"Mismatch in Train/Val interactions: expected 4446374, got {total_train_val}"
    assert len(total_train_students) == 14453, f"Mismatch in Train/Val students: expected 14453, got {len(total_train_students)}"

    test_data = torch.load(os.path.join(processed_dir, "sequences_test.pt"), map_location="cpu")
    total_test = int(test_data["seq_mask"].sum().item())
    total_test_students = set(test_data["uid_seq"].tolist())

    assert total_test == 1102810, f"Mismatch in Test interactions: expected 1102810, got {total_test}"
    assert len(total_test_students) == 3613, f"Mismatch in Test students: expected 3613, got {len(total_test_students)}"


def test_strided_test_window_eval_count(processed_dir):
    """
    Verify that context-preserving strided windowing evaluates exactly (total_test_interactions - num_students).
    For XES3G5M: 1,102,810 - 3,613 = 1,099,197 evaluated interactions.
    """
    test_data = torch.load(os.path.join(processed_dir, "sequences_test.pt"), map_location="cpu")
    strided_dict, eval_mask = build_test_strided_windows(test_data, max_seq_len=200, stride=100)

    total_evaluated = int(eval_mask.sum().item())
    expected_evaluated = 1102810 - 3613
    assert total_evaluated == expected_evaluated, (
        f"Mismatch in test evaluated interactions: expected {expected_evaluated}, got {total_evaluated}"
    )

    # Verify context_start generation and dataset integration (Issue L1)
    assert "context_start" in strided_dict, "context_start missing from reconstructed test strided windows!"
    context_starts = strided_dict["context_start"]
    assert context_starts.shape[0] == strided_dict["q_seq"].shape[0]
    assert (context_starts >= 0).all()
    assert (context_starts > 0).any(), "No windows have context_start > 0 in multi-window test set!"

    # Verify context_start in DataLoader batch
    from torch.utils.data import DataLoader
    from src.data.xes3g5m_dataset import collate_xes3g5m
    test_ds = XES3G5MDataset(
        static_item_bank=load_static_item_bank_optimized(processed_dir),
        sequences=strided_dict,
        is_train=False,
        eval_mask=eval_mask,
    )
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_xes3g5m)
    batch = next(iter(test_loader))
    assert "context_start" in batch, "context_start missing from collated test batch!"
    assert batch["context_start"].shape == (16,)
    assert batch["context_start"].dtype == torch.int64


def test_masked_bce_loss_numerical_stability():
    """
    Verify MaskedBCELoss numerical stability under extreme logits and empty masks.
    """
    loss_fn = MaskedBCELoss()

    # Extreme logits (large positive and large negative)
    logits = torch.tensor([[-100.0, 100.0], [50.0, -50.0]])
    targets = torch.tensor([[0, 1], [1, 0]])
    mask = torch.tensor([[True, True], [True, True]])

    loss = loss_fn(logits, targets, mask)
    assert not torch.isnan(loss) and not torch.isinf(loss)
    assert loss.item() < 1e-4, f"Loss on confident correct predictions should be near zero: {loss.item()}"

    # Empty mask (no valid steps)
    empty_mask = torch.zeros((2, 2), dtype=torch.bool)
    loss_empty = loss_fn(logits, targets, empty_mask)
    assert not torch.isnan(loss_empty) and not torch.isinf(loss_empty)
    assert loss_empty.item() == 0.0
