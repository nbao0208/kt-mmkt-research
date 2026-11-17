"""
Automated Verification Suite for pyKT Baselines (DKT, DKVMN, SAKT) on Eedi Task 3 & 4.

Verifies all Pre-Training Verification Gates and Acceptance Criteria from
plans/eedi_task34_pykt_baselines_implementation_and_training_plan.md:
1. Native pyKT instantiation with dynamic num_c=57 (zero dead parameters from num_c=865).
2. Dataset/train configuration & output directory isolation under outputs/*/eedi/.
3. Strict 5-fold and test student isolation on generated pykt_kc_level CSVs matching data/splits/Eedi/.
4. Non-overlapping 200-step chunking, selectmasks {-1, 1} compatibility, and intact 2-KC boundaries.
5. Anti-leakage invariance on 2-KC questions across DKT, DKVMN, SAKT (diff < 1e-6).
6. 1-to-1 element-wise bijection with EediTestContext on unified (124,960), public (74,276), private (50,684).
7. Non-overlapping training question frequency counts and safe handling of empty solution-image strata.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from pykt.models.dkt import DKT
from pykt.models.dkvmn import DKVMN
from pykt.models.sakt import SAKT

from src.data.dataset_router import resolve_dataset_key, resolve_dataset_paths, resolve_split_tag
from src.data.pykt_dataset import (
    get_pykt_dataloaders,
    load_stratification_metadata,
    load_test_grouped_data,
)
from src.evaluation.eedi_test_context import EediTestContext
from src.evaluation.pykt_evaluator import PyKTEvaluator
from src.models.baselines.pykt_models import init_pykt_baseline, sync_pykt_device
from src.utils.config import load_yaml_config


EEDI_PYKT_DIR = Path("data/processed/Eedi/pykt_kc_level")
EEDI_PROC_DIR = Path("data/processed/Eedi")
EEDI_SPLITS_DIR = Path("data/splits/Eedi")

requires_eedi_pykt = pytest.mark.skipif(
    not (EEDI_PYKT_DIR / "train_valid_sequences.csv").exists(),
    reason="Eedi pyKT kc_level CSVs not yet built in data/processed/Eedi/pykt_kc_level/",
)


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def test_eedi_pykt_configs_and_isolation():
    """Verify Eedi pyKT dataset/train YAML configs and outputs/*/eedi/ isolation."""
    ds_cfg = load_yaml_config("configs/dataset/eedi_pykt.yaml")
    tr_cfg = load_yaml_config("configs/train_pykt_eedi.yaml")

    assert resolve_dataset_key(dataset_cfg=ds_cfg) == "eedi"
    assert int(ds_cfg["num_c"]) == 57
    assert int(ds_cfg["num_questions"]) == 948
    assert int(ds_cfg["max_seq_len"]) == 200
    assert ds_cfg["expected_eval_counts"] == {"unified": 124960, "public": 74276, "private": 50684}

    paths = resolve_dataset_paths("eedi", dataset_cfg=ds_cfg, train_cfg=tr_cfg)
    assert paths["checkpoint_dir"] == "outputs/checkpoints/eedi"
    assert paths["output_report_dir"] == "outputs/reports/eedi"
    assert paths["log_dir"] == "outputs/logs/eedi"
    assert resolve_split_tag("eedi", "unified") == ""
    assert resolve_split_tag("eedi", "public") == "_public"
    assert resolve_split_tag("eedi", "private") == "_private"


def test_eedi_pykt_models_num_c_57(device):
    """Verify native pyKT models instantiate with num_c=57 and no leftover 865-dim weights."""
    dkt = init_pykt_baseline("dkt", {"num_c": 57, "emb_size": 128, "dropout": 0.2, "emb_type": "qid"}, device=device)
    dkvmn = init_pykt_baseline("dkvmn", {"num_c": 57, "dim_s": 128, "size_m": 50, "dropout": 0.2, "emb_type": "qid"}, device=device)
    sakt = init_pykt_baseline("sakt", {"num_c": 57, "seq_len": 200, "emb_size": 128, "num_attn_heads": 4, "dropout": 0.2, "num_en": 2, "emb_type": "qid"}, device=device)

    assert isinstance(dkt, DKT) and dkt.num_c == 57
    assert isinstance(dkvmn, DKVMN) and dkvmn.num_c == 57
    assert isinstance(sakt, SAKT) and sakt.num_c == 57

    assert sum(p.numel() for p in dkt.parameters()) == 154041
    assert sum(p.numel() for p in dkvmn.parameters()) == 100737
    assert sum(p.numel() for p in sakt.parameters()) == 246785


@requires_eedi_pykt
def test_eedi_pykt_5fold_isolation_and_chunk_integrity():
    """
    Verify 0% UID leakage across 5 folds and 3 test splits, exact match with data/splits/Eedi/,
    strict 200-step chunk length with {-1, 1} selectmasks, and intact 2-KC question boundaries.
    """
    df_train = pd.read_csv(EEDI_PYKT_DIR / "train_valid_sequences.csv")
    df_uni = pd.read_csv(EEDI_PYKT_DIR / "test_unified.csv")
    df_pub = pd.read_csv(EEDI_PYKT_DIR / "test_public.csv")
    df_priv = pd.read_csv(EEDI_PYKT_DIR / "test_private.csv")

    with open(EEDI_SPLITS_DIR / "train_val_folds.json", "r", encoding="utf-8") as f:
        ref_folds = json.load(f)
    with open(EEDI_SPLITS_DIR / "test_public_uids.json", "r", encoding="utf-8") as f:
        ref_pub = set(json.load(f))
    with open(EEDI_SPLITS_DIR / "test_private_uids.json", "r", encoding="utf-8") as f:
        ref_priv = set(json.load(f))

    fold_uids = [set(df_train[df_train["fold"] == k]["uid"].astype(int).unique()) for k in range(5)]
    for k in range(5):
        assert fold_uids[k] == set(ref_folds[f"fold_{k}"])
    all_train = set().union(*fold_uids)
    assert len(all_train) == 4918

    pub_uids = set(df_pub["uid"].astype(int).unique())
    priv_uids = set(df_priv["uid"].astype(int).unique())
    uni_uids = set(df_uni["uid"].astype(int).unique())
    assert pub_uids == ref_pub and len(pub_uids) == 615
    assert priv_uids == ref_priv and len(priv_uids) == 615
    assert uni_uids == (pub_uids | priv_uids) and len(uni_uids) == 1230
    assert not (all_train & uni_uids)
    assert not (pub_uids & priv_uids)

    # Check chunk length == 200, selectmasks in {1, -1}, concepts in [0, 56], and no split 2-KC starts
    for _, row in df_train.iloc[:500].iterrows():
        qs = [int(x) for x in row["questions"].split(",")]
        cs = [int(x) for x in row["concepts"].split(",")]
        sm = [int(x) for x in row["selectmasks"].split(",")]
        rep = [int(x) for x in row["is_repeat"].split(",")]
        assert len(qs) == len(cs) == len(sm) == len(rep) == 200
        assert rep[0] == 0, "Chunk starts mid-question (is_repeat != 0)!"
        assert set(sm).issubset({1, -1})
        valid_c = [c for c in cs if c != -1]
        assert all(0 <= c < 57 for c in valid_c)


@requires_eedi_pykt
def test_eedi_pykt_ktdataset_dataloader_smoke():
    """Verify pyKT KTDataset loads Eedi 5-fold CSVs cleanly and yields concepts in [0, 56]."""
    train_loader, val_loader = get_pykt_dataloaders(
        raw_data_dir=str(EEDI_PYKT_DIR),
        train_valid_file="train_valid_sequences.csv",
        fold=0,
        batch_size=32,
        num_workers=0,
    )
    batch = next(iter(val_loader))
    assert batch["cseqs"].shape[1] == 199
    sm = batch["smasks"]
    assert sm.dtype == torch.bool
    valid_c = batch["shft_cseqs"][sm]
    assert int(valid_c.min()) >= 0 and int(valid_c.max()) < 57


@requires_eedi_pykt
@pytest.mark.parametrize(
    "split,expected_students,expected_eval",
    [("unified", 1230, 124960), ("public", 615, 74276), ("private", 615, 50684)],
)
def test_eedi_pykt_one_to_one_alignment_with_test_context(split, expected_students, expected_eval):
    """
    Verify 1-to-1 element-wise bijection between PyKT grouped test data and EediTestContext
    across unified, public, and private splits, plus non-overlapping train question counts.
    """
    test_csv = str(EEDI_PYKT_DIR / f"test_{split}.csv")
    students = load_test_grouped_data(test_csv, expected_eval_count=expected_eval)
    assert len(students) == expected_students

    meta = load_stratification_metadata(
        raw_data_dir=str(EEDI_PYKT_DIR),
        test_grouped_students=students,
        processed_dir=str(EEDI_PROC_DIR),
        dataset_key="eedi",
        test_split=split,
        test_csv_path=test_csv,
        expected_eval_count=expected_eval,
        train_valid_csv_path=str(EEDI_PYKT_DIR / "train_valid_sequences.csv"),
    )
    ctx = EediTestContext(processed_dir=str(EEDI_PROC_DIR), test_split=split)
    assert len(meta["targets"]) == expected_eval
    assert np.array_equal(meta["targets"], ctx.targets)
    assert np.array_equal(meta["q_ids"], ctx.q_ids)

    # Verify non-overlapping train question counts match acd_v3_aux_metadata.pt identically
    aux = torch.load(EEDI_PROC_DIR / "acd_v3_aux_metadata.pt", map_location="cpu", weights_only=False)
    ref_counts = aux["question_train_counts"].tolist()
    for qid in range(948):
        assert meta["train_question_counts"].get(qid, 0) == int(ref_counts[qid])


@requires_eedi_pykt
def test_eedi_pykt_anti_leakage_2kc_and_empty_sol_slice(device):
    """
    Verify 2-KC anti-leakage invariance with num_c=57 across DKT, DKVMN, SAKT
    and verify compute_all_metrics handles Eedi's 0-sample solution image slice safely.
    """
    evaluator = PyKTEvaluator(
        raw_data_dir=str(EEDI_PYKT_DIR),
        test_csv_path=str(EEDI_PYKT_DIR / "test_private.csv"),
        processed_dir=str(EEDI_PROC_DIR),
        dataset_key="eedi",
        test_split="private",
        expected_eval_count=50684,
        device=device,
    )

    # Test empty solution-image slice safety in compute_all_metrics
    rng = np.random.default_rng(42)
    fake_preds = rng.uniform(0.1, 0.9, size=50684).astype(np.float32)
    metrics = evaluator.compute_all_metrics(fake_preds)
    assert metrics["prior_step_had_solution_image"]["num_samples"] == 0
    assert "eedi_strata" in metrics
    assert "overall_extended" in metrics
    assert not np.isnan(metrics["overall_extended"]["ece"])

    # Anti-leakage test on 2-KC question with num_c=57
    student_a = {
        "uid": 1,
        "all_questions": [
            {"q_id": 0, "concepts": [3], "response": 1},
            {"q_id": 1, "concepts": [12, 41], "response": 0},  # 2-KC question
            {"q_id": 2, "concepts": [7], "response": 1},
        ],
        "eval_questions": [
            {"q_id": 1, "concepts": [12, 41], "response": 0},
            {"q_id": 2, "concepts": [7], "response": 1},
        ],
    }
    student_b = {
        "uid": 1,
        "all_questions": [
            {"q_id": 0, "concepts": [3], "response": 1},
            {"q_id": 1, "concepts": [12, 41], "response": 1},  # Inverted response
            {"q_id": 2, "concepts": [7], "response": 1},
        ],
        "eval_questions": [
            {"q_id": 1, "concepts": [12, 41], "response": 1},
            {"q_id": 2, "concepts": [7], "response": 1},
        ],
    }

    sync_pykt_device(device)
    for m_name, m_cfg, fn_name in [
        ("dkt", {"num_c": 57, "emb_size": 64, "dropout": 0.0}, "predict_dkt_all"),
        ("dkvmn", {"num_c": 57, "dim_s": 64, "size_m": 20, "dropout": 0.0}, "predict_dkvmn_all"),
        ("sakt", {"num_c": 57, "seq_len": 200, "emb_size": 64, "num_attn_heads": 4, "dropout": 0.0, "num_en": 2}, "predict_sakt_all"),
    ]:
        model = init_pykt_baseline(m_name, m_cfg, device=device)
        model.eval()
        evaluator.test_students = [student_a]
        p_a = getattr(evaluator, fn_name)(model)
        evaluator.test_students = [student_b]
        p_b = getattr(evaluator, fn_name)(model)

        assert abs(p_a[0] - p_b[0]) < 1e-6, f"{m_name.upper()} leaked 2-KC label at t_0!"
        assert abs(p_a[1] - p_b[1]) > 1e-6, f"{m_name.upper()} state failed to update at t_0 + 1!"


def test_eedi_pykt_cli_help():
    """Verify CLI entry points parse Eedi options cleanly."""
    for script in [
        "scripts/prepare_eedi_pykt_data.py",
        "scripts/train_pykt_baselines.py",
        "scripts/evaluate_pykt_baselines.py",
        "scripts/run_5fold_pykt_baselines.py",
    ]:
        res = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True)
        assert res.returncode == 0, f"{script} --help failed: {res.stderr}"
