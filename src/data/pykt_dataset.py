"""
Data loading, preprocessing, and test set alignment module for pyKT baselines on XES3G5M.

Provides:
1. 5-Fold Cross-Validation data loaders using pyKT's native KTDataset.
2. Exact sequential question grouping for kc_level/test.csv to match 1,099,197 test interactions.
3. Multimodal stratification metadata extraction from processed XES3G5M directory.
"""

import os
import re
import json
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# [H3 Fix] Unconditionally alias FloatTensor and LongTensor in torch.cuda and pykt.datasets.data_loader
# to standard CPU torch.FloatTensor / torch.LongTensor.
# This ensures KTDataset always stores tensors on CPU RAM regardless of whether CUDA is available or not,
# preventing CUDA memory leaks, pickle errors, and DataLoader worker multiprocessing crashes.
torch.cuda.FloatTensor = torch.FloatTensor
torch.cuda.LongTensor = torch.LongTensor

import pykt.datasets.data_loader as p_dl
p_dl.FloatTensor = torch.FloatTensor
p_dl.LongTensor = torch.LongTensor

from pykt.datasets.data_loader import KTDataset
from src.utils.seed import seed_worker


def get_pykt_dataloaders(
    raw_data_dir: str = "data/raw/XES3G5M/kc_level",
    train_valid_file: str = "train_valid_sequences.csv",
    fold: int = 0,
    batch_size: int = 256,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Initialize training and validation DataLoaders for a specific cross-validation fold.
    
    Adheres strictly to the 5-fold cross-validation protocol:
    - Validation set: Sequences with fold == `fold`.
    - Training set: Sequences with fold in {0, 1, 2, 3, 4} \\ {fold}.

    Args:
        raw_data_dir: Path to directory containing raw kc_level CSV files.
        train_valid_file: Name of the 5-fold training/validation sequence file.
        fold: Validation fold index in {0, 1, 2, 3, 4}.
        batch_size: Batch size for training and validation.
        num_workers: Number of DataLoader worker subprocesses.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    seq_path = os.path.join(raw_data_dir, train_valid_file)
    if not os.path.exists(seq_path):
        raise FileNotFoundError(f"Training sequence file not found at: {seq_path}")

    all_folds = {0, 1, 2, 3, 4}
    train_folds = all_folds - {fold}
    val_folds = {fold}

    train_ds = KTDataset(
        file_path=seq_path,
        input_type=["questions", "concepts"],
        folds=train_folds,
        qtest=False,
    )
    val_ds = KTDataset(
        file_path=seq_path,
        input_type=["questions", "concepts"],
        folds=val_folds,
        qtest=False,
    )

    worker_fn = seed_worker if num_workers > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_fn,
    )

    return train_loader, val_loader


def load_test_grouped_data(
    test_csv_path: str = "data/raw/XES3G5M/kc_level/test.csv",
    expected_eval_count: Optional[int] = 1099197,
) -> List[Dict[str, Any]]:
    """
    Load a pyKT kc_level test CSV and group interactions by question for each student.
    
    In kc_level test CSVs, questions requiring multiple skills are unrolled into
    consecutive steps with `is_repeat == 0` for the first skill and `is_repeat == 1`
    for subsequent skills belonging to the same question.
    
    This function parses each student trajectory and groups skills by question.
    The first question (t = 0) has no prior history and is skipped from evaluation,
    yielding e.g. 1,099,197 interactions on XES3G5M or 124,960 / 74,276 / 50,684 on Eedi.

    Args:
        test_csv_path: Path to kc_level test CSV.
        expected_eval_count: Expected number of evaluated interactions (None to skip assertion).

    Returns:
        List of student dictionaries, each containing:
            - 'uid': Student ID.
            - 'all_questions': List of question interaction tuples.
            - 'eval_questions': Evaluated question tuples starting from index 1 (t > 0).
    """
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test CSV not found at: {test_csv_path}")

    df = pd.read_csv(test_csv_path)
    students = []
    total_eval_interactions = 0

    for idx, row in df.iterrows():
        uid = int(row["uid"])
        questions = [int(x) for x in row["questions"].split(",")]
        concepts = [int(x) for x in row["concepts"].split(",")]
        responses = [int(x) for x in row["responses"].split(",")]
        repeats = [int(x) for x in row["is_repeat"].split(",")]

        # Group concepts by question
        grouped_q = []
        cur_q_id = None
        cur_concepts = []
        cur_resp = None

        for q, c, r, rep in zip(questions, concepts, responses, repeats):
            if rep == 0:
                if cur_concepts:
                    grouped_q.append({
                        "q_id": cur_q_id,
                        "concepts": cur_concepts,
                        "response": cur_resp,
                    })
                cur_q_id = q
                cur_concepts = [c]
                cur_resp = r
            else:
                cur_concepts.append(c)

        if cur_concepts:
            grouped_q.append({
                "q_id": cur_q_id,
                "concepts": cur_concepts,
                "response": cur_resp,
            })

        # Evaluated interactions exclude the first question (t = 0)
        eval_q = grouped_q[1:] if len(grouped_q) > 1 else []
        total_eval_interactions += len(eval_q)

        students.append({
            "uid": uid,
            "all_questions": grouped_q,
            "eval_questions": eval_q,
        })

    if expected_eval_count is not None:
        assert total_eval_interactions == expected_eval_count, (
            f"Expected exactly {expected_eval_count:,} evaluated interactions, got {total_eval_interactions:,}!"
        )
    return students


def load_stratification_metadata(
    raw_data_dir: str = "data/raw/XES3G5M",
    test_grouped_students: Optional[List[Dict[str, Any]]] = None,
    processed_dir: Optional[str] = None,
    dataset_key: str = "xes3g5m",
    test_split: str = "unified",
    test_csv_path: Optional[str] = None,
    expected_eval_count: Optional[int] = None,
    test_stride: int = 100,
    train_valid_csv_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract stratification metadata aligned with evaluated test interactions.

    Supports both:
    - 'xes3g5m': sourced from raw XES3G5M kc_level CSVs and metadata/questions.json (1,099,197 interactions).
    - 'eedi': sourced from EediTestContext + non-overlapping Eedi pykt_kc_level/train_valid_sequences.csv
      with strict 1-to-1 element-wise verification against EediTestContext (124,960 / 74,276 / 50,684).

    Provides:
    - 'targets': np.ndarray of ground truth binary labels.
    - 'q_ids': np.ndarray of evaluated question IDs.
    - 'm_img_q': np.ndarray of question image indicators.
    - 'prev_m_img_s': np.ndarray of prior step solution image indicators.
    - 'remedial_mask': np.ndarray of bool indicating response at tau-1 == 0.
    - 'train_question_counts': Dict mapping question ID to occurrence count in training set.
    """
    ds_key = dataset_key.strip().lower()
    if ds_key == "eedi":
        from src.evaluation.eedi_test_context import EediTestContext, EXPECTED_EEDI_EVAL_COUNTS

        split_key = test_split.strip().lower()
        exp_count = expected_eval_count if expected_eval_count is not None else EXPECTED_EEDI_EVAL_COUNTS.get(split_key)
        proc_dir = processed_dir or "data/processed/Eedi"
        if test_grouped_students is None:
            resolved_test_csv = test_csv_path or os.path.join(
                raw_data_dir, f"test_{split_key}.csv"
            )
            if not os.path.exists(resolved_test_csv):
                resolved_test_csv = os.path.join(proc_dir, "pykt_kc_level", f"test_{split_key}.csv")
            test_grouped_students = load_test_grouped_data(
                resolved_test_csv, expected_eval_count=exp_count
            )

        targets_list: List[int] = []
        q_ids_list: List[int] = []
        remedial_mask_list: List[bool] = []
        for s in test_grouped_students:
            all_q = s["all_questions"]
            for t in range(1, len(all_q)):
                targets_list.append(int(all_q[t]["response"]))
                q_ids_list.append(int(all_q[t]["q_id"]))
                remedial_mask_list.append(int(all_q[t - 1]["response"]) == 0)

        targets_arr = np.asarray(targets_list, dtype=np.int64)
        q_ids_arr = np.asarray(q_ids_list, dtype=np.int64)
        remedial_arr = np.asarray(remedial_mask_list, dtype=bool)

        ctx = EediTestContext(processed_dir=proc_dir, test_split=split_key, test_stride=test_stride)
        assert np.array_equal(targets_arr, ctx.targets), (
            f"Eedi pyKT targets mismatch against EediTestContext on split '{split_key}'!"
        )
        assert np.array_equal(q_ids_arr, ctx.q_ids), (
            f"Eedi pyKT q_ids mismatch against EediTestContext on split '{split_key}'!"
        )

        strata_masks = ctx.strata_masks()
        m_img_q_arr = strata_masks["q_stem_image"].astype(np.float32)
        # Eedi Task 3 & 4 has zero authentic solution images across all 948 items
        prev_m_img_s_arr = np.zeros(len(targets_arr), dtype=np.float32)

        # Compute non-overlapping training question counts (never count overlapping strided windows)
        train_counts: Counter = Counter()
        tv_csv = train_valid_csv_path or os.path.join(raw_data_dir, "train_valid_sequences.csv")
        if not os.path.exists(tv_csv):
            tv_csv = os.path.join(proc_dir, "pykt_kc_level", "train_valid_sequences.csv")
        if os.path.exists(tv_csv):
            df_train = pd.read_csv(tv_csv)
            for _, row in df_train.iterrows():
                qs = [int(x) for x in row["questions"].split(",")]
                repeats = [int(x) for x in row["is_repeat"].split(",")]
                for q, rep in zip(qs, repeats):
                    if q != -1 and rep == 0:
                        train_counts[q] += 1
        else:
            aux_path = os.path.join(proc_dir, "acd_v3_aux_metadata.pt")
            if os.path.exists(aux_path):
                aux = torch.load(aux_path, map_location="cpu", weights_only=False)
                q_cnt_tensor = aux["question_train_counts"]
                for qid, cnt in enumerate(q_cnt_tensor.tolist()):
                    if cnt > 0:
                        train_counts[qid] = int(cnt)

        return {
            "targets": targets_arr,
            "q_ids": q_ids_arr,
            "m_img_q": m_img_q_arr,
            "prev_m_img_s": prev_m_img_s_arr,
            "remedial_mask": remedial_arr,
            "train_question_counts": dict(train_counts),
            "eedi_context": ctx,
            "strata_masks": strata_masks,
        }

    if test_grouped_students is None:
        resolved_test_csv = test_csv_path or os.path.join(raw_data_dir, "kc_level", "test.csv")
        exp_count = expected_eval_count if expected_eval_count is not None else 1099197
        test_grouped_students = load_test_grouped_data(
            resolved_test_csv, expected_eval_count=exp_count
        )

    # 1. Extract question and solution image indicators directly from raw metadata/questions.json
    q_meta_path = os.path.join(raw_data_dir, "metadata", "questions.json")
    has_stem_img_map: Dict[int, float] = {}
    has_sol_img_map: Dict[int, float] = {}

    if os.path.exists(q_meta_path):
        img_ref_regex = re.compile(r"((?:question|analysis)_\d+-image_\d+)(?:\.png)?")
        with open(q_meta_path, "r", encoding="utf-8") as f:
            questions_raw = json.load(f)

        for qid_str, q_info in questions_raw.items():
            qid = int(qid_str)
            content = q_info.get("content", "")
            opts = q_info.get("options", {})
            opt_str = " ".join(str(v) for v in opts.values()) if isinstance(opts, dict) else ""
            analysis = q_info.get("analysis", "")

            has_stem = 1.0 if len(img_ref_regex.findall(content + " " + opt_str)) > 0 else 0.0
            has_sol = 1.0 if len(img_ref_regex.findall(analysis)) > 0 else 0.0
            has_stem_img_map[qid] = has_stem
            has_sol_img_map[qid] = has_sol
    elif processed_dir and os.path.exists(os.path.join(processed_dir, "mg_static_item_bank.pt")):
        bank = torch.load(os.path.join(processed_dir, "mg_static_item_bank.pt"), map_location="cpu")
        m_masks = bank.get("modality_masks", bank.get("mod_mask"))
        for qid in range(len(m_masks)):
            has_stem_img_map[qid] = float(m_masks[qid, 0].item())
            has_sol_img_map[qid] = float(m_masks[qid, 2].item())

    # 2. Extract training question frequencies directly from raw kc_level/train_valid_sequences.csv
    train_counts: Counter = Counter()
    raw_seq_csv = os.path.join(raw_data_dir, "kc_level", "train_valid_sequences.csv")

    if os.path.exists(raw_seq_csv):
        df_train = pd.read_csv(raw_seq_csv)
        for _, row in df_train.iterrows():
            qs = [int(x) for x in row["questions"].split(",")]
            repeats = [int(x) for x in row["is_repeat"].split(",")]
            for q, rep in zip(qs, repeats):
                if rep == 0:
                    train_counts[q] += 1
    elif processed_dir:
        for f in range(5):
            fold_file = os.path.join(processed_dir, f"sequences_fold_{f}.pt")
            if os.path.exists(fold_file):
                f_data = torch.load(fold_file, map_location="cpu")
                valid_mask = f_data["seq_mask"]
                valid_qs = f_data["q_seq"][valid_mask].tolist()
                train_counts.update(valid_qs)

    targets_list = []
    q_ids_list = []
    m_img_q_list = []
    prev_m_img_s_list = []
    remedial_mask_list = []

    for s in test_grouped_students:
        all_q = s["all_questions"]
        # Step through evaluated questions t = 1 .. T-1
        for t in range(1, len(all_q)):
            cur_q = all_q[t]
            prev_q = all_q[t - 1]

            q_id = cur_q["q_id"]
            resp = cur_q["response"]
            prev_resp = prev_q["response"]
            prev_q_id = prev_q["q_id"]

            has_q_img = has_stem_img_map.get(q_id, 0.0)
            has_prev_s_img = has_sol_img_map.get(prev_q_id, 0.0)

            targets_list.append(resp)
            q_ids_list.append(q_id)
            m_img_q_list.append(has_q_img)
            prev_m_img_s_list.append(has_prev_s_img)
            remedial_mask_list.append(prev_resp == 0)

    return {
        "targets": np.asarray(targets_list, dtype=np.int64),
        "q_ids": np.asarray(q_ids_list, dtype=np.int64),
        "m_img_q": np.asarray(m_img_q_list, dtype=np.float32),
        "prev_m_img_s": np.asarray(prev_m_img_s_list, dtype=np.float32),
        "remedial_mask": np.asarray(remedial_mask_list, dtype=bool),
        "train_question_counts": dict(train_counts),
    }
