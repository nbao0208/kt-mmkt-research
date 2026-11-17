"""
Eedi NeurIPS 2020 (Task 3 & 4) to pyKT Concept-Level (`kc_level`) Data Converter.

Generates deterministic, non-overlapping pyKT-compatible sequence CSVs under
`data/processed/Eedi/pykt_kc_level/` while maintaining 100% alignment with
the ACD-MMKTv3 Eedi preprocessing pipeline:
  1. 1-to-1 AnswerId metadata join and 4-level chronological sort
     (UserId, DateAnswered_sec, QuizId, AnswerId).
  2. Leaf-KC space mapping restricted strictly to the 57 Level-3 skills (0..56).
  3. Multi-concept question unrolling in ascending KC order with `is_repeat` flags.
  4. Non-overlapping 200-step chunking with `-1` right-padding and question-boundary
     preservation for 5-fold train/validation (`train_valid_sequences.csv`).
  5. Full unchunked student trajectories sorted by ascending UserId for unified,
     public, and private test splits (`test_unified.csv`, `test_public.csv`, `test_private.csv`).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml


EXPECTED_TRAIN_STUDENTS = 4918
EXPECTED_PUBLIC_STUDENTS = 615
EXPECTED_PRIVATE_STUDENTS = 615
EXPECTED_UNIFIED_EVAL = 124960
EXPECTED_PUBLIC_EVAL = 74276
EXPECTED_PRIVATE_EVAL = 50684
EXPECTED_NUM_QUESTIONS = 948
EXPECTED_NUM_LEAF_KCS = 57


def _load_leaf_kc_mapping(processed_dir: Path) -> Tuple[List[List[int]], Dict[str, int]]:
    """
    Load the canonical question-to-leaf-KC mapping (57 Level-3 KCs indexed 0..56)
    from the preprocessed Eedi static item bank.
    """
    mg_path = processed_dir / "mg_static_item_bank.pt"
    full_path = processed_dir / "static_item_bank.pt"
    bank_path = mg_path if mg_path.exists() else full_path
    if not bank_path.exists():
        raise FileNotFoundError(
            f"Neither {mg_path} nor {full_path} exists. Run Eedi MMKT preprocessing first."
        )

    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    q_leaf_kcs = bank["question_leaf_kcs"]
    q_kc_mask = bank["question_kc_mask"]

    num_q = int(q_leaf_kcs.size(0))
    if num_q != EXPECTED_NUM_QUESTIONS:
        raise AssertionError(f"Expected {EXPECTED_NUM_QUESTIONS} questions, got {num_q}")

    q_to_leaves: List[List[int]] = []
    unique_kcs = set()
    single_kc_q = 0
    multi_kc_q = 0

    for qid in range(num_q):
        leaves = sorted(int(c) for c in q_leaf_kcs[qid][q_kc_mask[qid]].tolist())
        if not leaves:
            raise AssertionError(f"Question {qid} has zero mapped leaf KCs.")
        if any(c < 0 or c >= EXPECTED_NUM_LEAF_KCS for c in leaves):
            raise AssertionError(f"Question {qid} has out-of-range leaf KC indices: {leaves}")
        unique_kcs.update(leaves)
        if len(leaves) == 1:
            single_kc_q += 1
        else:
            multi_kc_q += 1
        q_to_leaves.append(leaves)

    if len(unique_kcs) != EXPECTED_NUM_LEAF_KCS:
        raise AssertionError(
            f"Expected {EXPECTED_NUM_LEAF_KCS} active leaf KCs, found {len(unique_kcs)}"
        )

    stats = {
        "num_questions": num_q,
        "num_leaf_kcs": len(unique_kcs),
        "single_kc_questions": single_kc_q,
        "multi_kc_questions": multi_kc_q,
    }
    return q_to_leaves, stats


def _load_splits(splits_dir: Path) -> Tuple[Dict[int, int], List[int], List[int]]:
    """
    Load the verified 5-fold training split and public/private test UID lists
    produced by the Eedi MMKT preprocessor, asserting strict disjointness.
    """
    folds_path = splits_dir / "train_val_folds.json"
    pub_path = splits_dir / "test_public_uids.json"
    priv_path = splits_dir / "test_private_uids.json"

    for p in (folds_path, pub_path, priv_path):
        if not p.exists():
            raise FileNotFoundError(f"Required split file missing: {p}")

    with open(folds_path, "r", encoding="utf-8") as f:
        folds_dict = json.load(f)
    with open(pub_path, "r", encoding="utf-8") as f:
        pub_uids = [int(u) for u in json.load(f)]
    with open(priv_path, "r", encoding="utf-8") as f:
        priv_uids = [int(u) for u in json.load(f)]

    uid_to_fold: Dict[int, int] = {}
    for f_idx in range(5):
        f_key = f"fold_{f_idx}"
        if f_key not in folds_dict:
            raise KeyError(f"Missing '{f_key}' in {folds_path}")
        for u in folds_dict[f_key]:
            uid = int(u)
            if uid in uid_to_fold:
                raise AssertionError(f"Duplicate UID {uid} across folds in {folds_path}")
            uid_to_fold[uid] = f_idx

    train_set = set(uid_to_fold.keys())
    pub_set = set(pub_uids)
    priv_set = set(priv_uids)

    if len(train_set) != EXPECTED_TRAIN_STUDENTS:
        raise AssertionError(f"Expected {EXPECTED_TRAIN_STUDENTS} train students, got {len(train_set)}")
    if len(pub_set) != EXPECTED_PUBLIC_STUDENTS:
        raise AssertionError(f"Expected {EXPECTED_PUBLIC_STUDENTS} public students, got {len(pub_set)}")
    if len(priv_set) != EXPECTED_PRIVATE_STUDENTS:
        raise AssertionError(f"Expected {EXPECTED_PRIVATE_STUDENTS} private students, got {len(priv_set)}")
    if (train_set & pub_set) or (train_set & priv_set) or (pub_set & priv_set):
        raise AssertionError("Student UID overlap detected across train/public/private splits!")

    return uid_to_fold, pub_uids, priv_uids


def _unroll_user_dataframe(
    u_df: pd.DataFrame,
    q_to_leaves: List[List[int]],
) -> Tuple[List[int], List[int], List[int], List[int], List[int]]:
    """
    Unroll a chronologically sorted student DataFrame into leaf-KC-level sequences.
    Multi-KC questions expand into consecutive steps with ascending KC indices
    and `is_repeat` = 0 on the first KC, 1 on subsequent KCs.
    """
    q_seq: List[int] = []
    c_seq: List[int] = []
    r_seq: List[int] = []
    ts_seq: List[int] = []
    rep_seq: List[int] = []

    q_arr = u_df["QuestionId"].to_numpy(dtype=np.int64)
    r_arr = u_df["IsCorrect"].to_numpy(dtype=np.int64)
    ts_arr = u_df["DateAnswered_sec"].to_numpy(dtype=np.int64)

    for q_val, r_val, ts_val in zip(q_arr, r_arr, ts_arr):
        q_int = int(q_val)
        r_int = int(r_val)
        ts_int = int(ts_val)
        leaves = q_to_leaves[q_int]
        for idx, c_int in enumerate(leaves):
            q_seq.append(q_int)
            c_seq.append(c_int)
            r_seq.append(r_int)
            ts_seq.append(ts_int)
            rep_seq.append(0 if idx == 0 else 1)

    return q_seq, c_seq, r_seq, ts_seq, rep_seq


def _build_train_valid_csv(
    train_df: pd.DataFrame,
    uid_to_fold: Dict[int, int],
    q_to_leaves: List[List[int]],
    out_csv_path: Path,
    max_seq_len: int = 200,
) -> Dict[str, Any]:
    """
    Convert the 4,918 training student trajectories into non-overlapping `max_seq_len`-step
    chunks right-padded with `-1` for pyKT's `KTDataset` (selectmasks != -1 loss rule).

    Question-boundary protection ensures a 2-KC question is never split across a chunk
    boundary when slicing at `max_seq_len`.
    """
    sorted_df = train_df.sort_values(
        ["UserId", "DateAnswered_sec", "QuizId", "AnswerId"]
    ).reset_index(drop=True)

    rows: List[Dict[str, Any]] = []
    total_unrolled_steps = 0
    total_questions = 0
    total_repeat_steps = 0
    fold_student_counts = {f: 0 for f in range(5)}
    fold_chunk_counts = {f: 0 for f in range(5)}

    for uid_val, u_df in sorted_df.groupby("UserId", sort=True):
        uid = int(uid_val)
        fold = uid_to_fold[uid]
        fold_student_counts[fold] += 1

        q_seq, c_seq, r_seq, ts_seq, rep_seq = _unroll_user_dataframe(u_df, q_to_leaves)
        n_steps = len(q_seq)
        total_unrolled_steps += n_steps
        total_questions += len(u_df)
        total_repeat_steps += sum(rep_seq)

        start = 0
        chunk_idx = 0
        while start < n_steps:
            end = min(start + max_seq_len, n_steps)
            # Avoid cutting inside a multi-KC question across chunk boundaries
            if end < n_steps and rep_seq[end] == 1:
                while end > start and rep_seq[end] == 1:
                    end -= 1
            if end == start:
                end = min(start + max_seq_len, n_steps)

            chunk_len = end - start
            pad_len = max_seq_len - chunk_len

            q_chunk = q_seq[start:end] + [-1] * pad_len
            c_chunk = c_seq[start:end] + [-1] * pad_len
            r_chunk = r_seq[start:end] + [-1] * pad_len
            t_chunk = ts_seq[start:end] + [-1] * pad_len
            s_chunk = [1] * chunk_len + [-1] * pad_len
            rep_chunk = rep_seq[start:end] + [-1] * pad_len

            rows.append(
                {
                    "fold": fold,
                    "uid": uid,
                    "_chunk_idx": chunk_idx,
                    "questions": ",".join(map(str, q_chunk)),
                    "concepts": ",".join(map(str, c_chunk)),
                    "responses": ",".join(map(str, r_chunk)),
                    "timestamps": ",".join(map(str, t_chunk)),
                    "selectmasks": ",".join(map(str, s_chunk)),
                    "is_repeat": ",".join(map(str, rep_chunk)),
                }
            )
            fold_chunk_counts[fold] += 1
            chunk_idx += 1
            start = end

    out_df = pd.DataFrame(rows)
    out_df.sort_values(["fold", "uid", "_chunk_idx"], inplace=True)
    out_df.drop(columns=["_chunk_idx"], inplace=True)
    out_df.to_csv(out_csv_path, index=False)

    return {
        "file": str(out_csv_path),
        "num_students": sum(fold_student_counts.values()),
        "num_chunks": len(out_df),
        "total_questions": total_questions,
        "total_unrolled_steps": total_unrolled_steps,
        "total_repeat_steps": total_repeat_steps,
        "fold_student_counts": fold_student_counts,
        "fold_chunk_counts": fold_chunk_counts,
    }


def _build_test_csv(
    test_df: pd.DataFrame,
    q_to_leaves: List[List[int]],
    out_csv_path: Path,
    expected_students: int,
    expected_eval_interactions: int,
) -> Dict[str, Any]:
    """
    Convert a test split DataFrame into an unchunked pyKT `kc_level` test CSV sorted
    by ascending `UserId` with 4-level chronological ordering within each student.
    """
    sorted_df = test_df.sort_values(
        ["UserId", "DateAnswered_sec", "QuizId", "AnswerId"]
    ).reset_index(drop=True)

    rows: List[Dict[str, Any]] = []
    running_cidx = 0
    total_questions = 0
    total_eval_questions = 0
    total_repeat_steps = 0
    student_count = 0

    for uid_val, u_df in sorted_df.groupby("UserId", sort=True):
        uid = int(uid_val)
        student_count += 1
        n_q = len(u_df)
        total_questions += n_q
        if n_q > 1:
            total_eval_questions += n_q - 1

        q_seq, c_seq, r_seq, ts_seq, rep_seq = _unroll_user_dataframe(u_df, q_to_leaves)
        n_steps = len(q_seq)
        cidxs = list(range(running_cidx, running_cidx + n_steps))
        running_cidx += n_steps
        total_repeat_steps += sum(rep_seq)

        rows.append(
            {
                "fold": -1,
                "uid": uid,
                "questions": ",".join(map(str, q_seq)),
                "concepts": ",".join(map(str, c_seq)),
                "responses": ",".join(map(str, r_seq)),
                "timestamps": ",".join(map(str, ts_seq)),
                "is_repeat": ",".join(map(str, rep_seq)),
                "cidxs": ",".join(map(str, cidxs)),
            }
        )

    if student_count != expected_students:
        raise AssertionError(
            f"Expected {expected_students} students in {out_csv_path.name}, got {student_count}"
        )
    if total_eval_questions != expected_eval_interactions:
        raise AssertionError(
            f"Expected {expected_eval_interactions} evaluated interactions in {out_csv_path.name}, "
            f"got {total_eval_questions}"
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv_path, index=False)

    return {
        "file": str(out_csv_path),
        "num_students": student_count,
        "total_questions": total_questions,
        "evaluated_interactions": total_eval_questions,
        "total_unrolled_steps": running_cidx,
        "total_repeat_steps": total_repeat_steps,
    }


def prepare_eedi_pykt_kc_level(
    dataset_config_path: str = "configs/dataset/eedi_pykt.yaml",
    raw_root_dir: Optional[str] = None,
    processed_dir: Optional[str] = None,
    splits_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_seq_len: int = 200,
) -> Dict[str, Any]:
    """
    Build all pyKT concept-level (`kc_level`) sequence CSVs for Eedi Task 3 & 4
    inside `data/processed/Eedi/pykt_kc_level/` without writing to `data/raw/`.
    """
    cfg: Dict[str, Any] = {}
    if dataset_config_path and os.path.exists(dataset_config_path):
        with open(dataset_config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    raw_dir = Path(raw_root_dir or cfg.get("raw_root_dir", "data/raw/Eedi NeurIPS 2020/data_extracted/data"))
    proc_dir = Path(processed_dir or cfg.get("processed_dir", "data/processed/Eedi"))
    spl_dir = Path(splits_dir or cfg.get("splits_dir", "data/splits/Eedi"))
    out_dir = Path(output_dir or cfg.get("raw_data_dir", str(proc_dir / "pykt_kc_level")))
    seq_len = int(cfg.get("max_seq_len", max_seq_len))

    # Safety guard: never write generated artifacts into data/raw
    if "data/raw" in str(out_dir).replace("\\", "/"):
        raise ValueError(f"Refusing to write generated pyKT data into raw directory: {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print("[Eedi pyKT Prep] Loading canonical 57-leaf-KC mapping and verified splits...", flush=True)
    q_to_leaves, kc_stats = _load_leaf_kc_mapping(proc_dir)
    uid_to_fold, pub_uids, priv_uids = _load_splits(spl_dir)

    print(f"[Eedi pyKT Prep] Reading raw Eedi CSVs from {raw_dir}...", flush=True)
    train_df = pd.read_csv(raw_dir / "train_data" / "train_task_3_4.csv")
    pub_df = pd.read_csv(raw_dir / "test_data" / "test_public_task_4_more_splits.csv")
    priv_df = pd.read_csv(raw_dir / "test_data" / "test_private_task_4_more_splits.csv")
    ans_meta = pd.read_csv(raw_dir / "metadata" / "answer_metadata_task_3_4.csv")

    total_rows = len(train_df) + len(pub_df) + len(priv_df)
    if len(ans_meta) != total_rows:
        raise AssertionError(f"Answer metadata rows ({len(ans_meta)}) != total interactions ({total_rows})")

    ans_meta["DateAnswered_sec"] = (
        pd.to_datetime(ans_meta["DateAnswered"]).astype("datetime64[s]").astype("int64")
    )
    ans_indexed = ans_meta[["AnswerId", "DateAnswered_sec", "QuizId"]]

    train_merged = train_df.merge(ans_indexed, on="AnswerId", how="left", validate="one_to_one")
    pub_merged = pub_df.merge(ans_indexed, on="AnswerId", how="left", validate="one_to_one")
    priv_merged = priv_df.merge(ans_indexed, on="AnswerId", how="left", validate="one_to_one")

    # Verify exact UserId match with stored splits
    if set(train_merged["UserId"].astype(int).unique()) != set(uid_to_fold.keys()):
        raise AssertionError("Train UserIds do not match data/splits/Eedi/train_val_folds.json!")
    if set(pub_merged["UserId"].astype(int).unique()) != set(pub_uids):
        raise AssertionError("Public test UserIds do not match data/splits/Eedi/test_public_uids.json!")
    if set(priv_merged["UserId"].astype(int).unique()) != set(priv_uids):
        raise AssertionError("Private test UserIds do not match data/splits/Eedi/test_private_uids.json!")

    unified_merged = pd.concat([pub_merged, priv_merged], ignore_index=True)

    print("[Eedi pyKT Prep] Writing 5-fold non-overlapping train_valid_sequences.csv...", flush=True)
    train_stats = _build_train_valid_csv(
        train_df=train_merged,
        uid_to_fold=uid_to_fold,
        q_to_leaves=q_to_leaves,
        out_csv_path=out_dir / cfg.get("train_valid_file", "train_valid_sequences.csv"),
        max_seq_len=seq_len,
    )

    test_files_cfg = cfg.get("test_files", {})
    print("[Eedi pyKT Prep] Writing unchunked test CSVs (unified, public, private)...", flush=True)
    unified_stats = _build_test_csv(
        test_df=unified_merged,
        q_to_leaves=q_to_leaves,
        out_csv_path=out_dir / test_files_cfg.get("unified", "test_unified.csv"),
        expected_students=EXPECTED_PUBLIC_STUDENTS + EXPECTED_PRIVATE_STUDENTS,
        expected_eval_interactions=EXPECTED_UNIFIED_EVAL,
    )
    public_stats = _build_test_csv(
        test_df=pub_merged,
        q_to_leaves=q_to_leaves,
        out_csv_path=out_dir / test_files_cfg.get("public", "test_public.csv"),
        expected_students=EXPECTED_PUBLIC_STUDENTS,
        expected_eval_interactions=EXPECTED_PUBLIC_EVAL,
    )
    private_stats = _build_test_csv(
        test_df=priv_merged,
        q_to_leaves=q_to_leaves,
        out_csv_path=out_dir / test_files_cfg.get("private", "test_private.csv"),
        expected_students=EXPECTED_PRIVATE_STUDENTS,
        expected_eval_interactions=EXPECTED_PRIVATE_EVAL,
    )

    summary = {
        "dataset": "Eedi NeurIPS 2020 (Task 3 & 4)",
        "output_dir": str(out_dir),
        "max_seq_len": seq_len,
        "kc_stats": kc_stats,
        "train_valid": train_stats,
        "test_unified": unified_stats,
        "test_public": public_stats,
        "test_private": private_stats,
    }
    summary_path = out_dir / "pykt_kc_level_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Eedi pyKT Prep] Completed! Summary saved to {summary_path}", flush=True)
    return summary
