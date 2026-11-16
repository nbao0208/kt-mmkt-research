"""
PyTorch Dataset and DataLoader Utilities for Eedi NeurIPS 2020 (Task 3 & 4)
Multimodal Knowledge Tracing.

Supports:
- Two-tier storage architecture matching ACDMMKTv3, MGMMKT, MGMMKTv2, and pyKT.
- 5-Fold Cross-Validation merging (4 folds train, 1 fold val) partitioned by student UID.
- Context-preserving strided windowing (L=200, S=100) for unified, public, and private test splits.
- Asymmetric dynamic Modality Dropout for ACD-MMKTv3 (drops only stem visual channel
  with p_drop in [0.15, 0.20] while preserving 100% of stem and solution text)
  and standard solution dropout for MG-MMKT/v2.
- Clean support for post-response diagnostic vectors v_post_tau^diag ([B, L, 8]) strictly
  for causal history steps tau < t, plus Task 4 target masks ([B, L, 10]).
"""

from collections import defaultdict
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def seed_worker(worker_id: int) -> None:
    """Ensure DataLoader worker subprocesses receive distinct deterministic seeds."""
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_eedi_mg_static_item_bank(
    source_path: str,
    target_path: str,
) -> Dict[str, torch.Tensor]:
    """
    Extract lightweight MG-MMKT static item bank from the full Eedi static_item_bank.pt.

    Args:
        source_path: Path to static_item_bank.pt.
        target_path: Destination path for mg_static_item_bank.pt.

    Returns:
        Lightweight static item bank dictionary for MG-MMKT / MG-MMKTv2.
    """
    if os.path.exists(target_path):
        return torch.load(target_path, map_location="cpu")

    full_bank = torch.load(source_path, map_location="cpu")
    mg_bank = {
        "modality_masks": full_bank["modality_masks"],
        "question_leaf_kcs": full_bank["question_leaf_kcs"],
        "question_kc_mask": full_bank["question_kc_mask"],
        "question_static_ctx": full_bank["question_static_ctx"],
        "correct_answer_idx": full_bank["correct_answer_idx"],
        "mg_v_txt_q": full_bank["mg_v_txt_q"],
        "mg_v_txt_s": full_bank["mg_v_txt_s"],
        "mg_v_img_q": full_bank["mg_v_img_q"],
        "mg_v_img_s": full_bank["mg_v_img_s"],
        "option_multimodal_emb": full_bank["option_multimodal_emb"],
    }
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    torch.save(mg_bank, target_path)
    return mg_bank


def load_eedi_static_item_bank(
    processed_dir: str,
    model_type: str = "acd_mmkt_v3",
) -> Dict[str, Any]:
    """
    Load Tier 1 Static Item Bank for Eedi Task 3 & 4 and attach ACD-MMKTv3 auxiliary metadata.

    Args:
        processed_dir: Directory containing processed Eedi tensors (e.g. 'data/processed/Eedi').
        model_type: Model identifier ('acd_mmkt_v3', 'acd_mmkt_v2', 'acd_mmkt', 'mg_mmkt', 'mg_mmkt_v2').

    Returns:
        Dictionary of static item bank tensors and fold priors.
    """
    full_path = os.path.join(processed_dir, "static_item_bank.pt")
    mg_path = os.path.join(processed_dir, "mg_static_item_bank.pt")

    if model_type.lower() in ("acd_mmkt", "acd_mmkt_v2", "acd_mmkt_v3"):
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Full static item bank not found at {full_path}")
        try:
            bank = torch.load(full_path, mmap=True, map_location="cpu")
        except Exception:
            bank = torch.load(full_path, map_location="cpu")

        aux_path = os.path.join(processed_dir, "acd_v3_aux_metadata.pt")
        if os.path.exists(aux_path):
            aux_data = torch.load(aux_path, map_location="cpu", weights_only=False)
            if isinstance(bank, dict):
                for key in (
                    "M_patch_q",
                    "patch_valid_mask",
                    "N_q_folds",
                    "D_q_folds",
                    "omega_q_folds",
                    "mu_prior_folds",
                    "mu_concept_folds",
                    "fold_distractor_entropy",
                    "fold_dominant_distractor",
                    "task3_expert_pairs",
                    "d_post_diag",
                    "kc_id_to_raw",
                    "raw_to_kc_id",
                    "kc_descriptions",
                    "question_train_counts",
                ):
                    if key in aux_data:
                        bank[key] = aux_data[key]
        return bank

    if os.path.exists(mg_path):
        return torch.load(mg_path, map_location="cpu")
    if os.path.exists(full_path):
        return create_eedi_mg_static_item_bank(full_path, mg_path)
    raise FileNotFoundError(f"Neither {mg_path} nor {full_path} found in {processed_dir}")


def build_eedi_test_strided_windows(
    test_seq_data: Dict[str, torch.Tensor],
    max_seq_len: int = 200,
    stride: int = 100,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Ensure Eedi test sequences are formatted as context-preserving strided windows.
    If the serialized dictionary already contains pre-built strided windows
    (indicated by 'context_start' and 'eval_mask'), returns them directly without
    duplicating overlapping causal context.

    Args:
        test_seq_data: Sequence dictionary loaded from sequences_test*.pt.
        max_seq_len: Maximum window length (default: 200).
        stride: Evaluation stride for subsequent windows (default: 100).

    Returns:
        Tuple of (strided_seq_dict, eval_mask [N_seq, max_seq_len]).
    """
    if "context_start" in test_seq_data and "eval_mask" in test_seq_data:
        return test_seq_data, test_seq_data["eval_mask"].bool()

    uids = test_seq_data["uid_seq"].tolist()
    uid_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, u in enumerate(uids):
        uid_to_indices[int(u)].append(idx)

    tensor_keys = [
        k for k, v in test_seq_data.items()
        if isinstance(v, torch.Tensor) and v.ndim >= 2 and k != "uid_seq"
    ]
    out_lists: Dict[str, List[torch.Tensor]] = {k: [] for k in tensor_keys}
    out_uids: List[int] = []
    out_ctx_start: List[int] = []
    out_eval_mask: List[torch.Tensor] = []

    for u, indices in uid_to_indices.items():
        full_seq: Dict[str, torch.Tensor] = {}
        for k in tensor_keys:
            parts = []
            for idx in indices:
                valid_len = int(test_seq_data["seq_mask"][idx].sum().item())
                parts.append(test_seq_data[k][idx, :valid_len])
            full_seq[k] = torch.cat(parts, dim=0)

        total_steps = full_seq["q_seq"].size(0)
        w_len = min(max_seq_len, total_steps)

        for k in tensor_keys:
            shape = (max_seq_len,) + tuple(full_seq[k].shape[1:])
            chunk = torch.zeros(shape, dtype=full_seq[k].dtype)
            chunk[:w_len] = full_seq[k][:w_len]
            out_lists[k].append(chunk)

        eval_chunk = torch.zeros(max_seq_len, dtype=torch.bool)
        if w_len > 1:
            eval_chunk[1:w_len] = True
        out_eval_mask.append(eval_chunk)
        out_uids.append(u)
        out_ctx_start.append(0)

        cur_eval_start = max_seq_len
        while cur_eval_start < total_steps:
            cur_eval_end = min(cur_eval_start + stride, total_steps)
            context_start = max(0, cur_eval_end - max_seq_len)
            slice_len = cur_eval_end - context_start

            for k in tensor_keys:
                shape = (max_seq_len,) + tuple(full_seq[k].shape[1:])
                chunk = torch.zeros(shape, dtype=full_seq[k].dtype)
                chunk[:slice_len] = full_seq[k][context_start:cur_eval_end]
                out_lists[k].append(chunk)

            eval_chunk = torch.zeros(max_seq_len, dtype=torch.bool)
            eval_local_start = cur_eval_start - context_start
            eval_chunk[eval_local_start:slice_len] = True
            out_eval_mask.append(eval_chunk)
            out_uids.append(u)
            out_ctx_start.append(context_start)

            cur_eval_start = cur_eval_end

    reconstructed: Dict[str, torch.Tensor] = {
        k: torch.stack(v, dim=0) for k, v in out_lists.items()
    }
    reconstructed["uid_seq"] = torch.tensor(out_uids, dtype=torch.int64)
    reconstructed["context_start"] = torch.tensor(out_ctx_start, dtype=torch.int64)
    eval_mask_tensor = torch.stack(out_eval_mask, dim=0)
    reconstructed["eval_mask"] = eval_mask_tensor
    return reconstructed, eval_mask_tensor


class EediDataset(Dataset):
    """
    Two-Tier PyTorch Dataset for Eedi NeurIPS 2020 (Task 3 & 4) Multimodal Knowledge Tracing.
    """

    def __init__(
        self,
        static_item_bank: Union[str, Dict[str, Any]],
        sequences: Union[str, Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
        model_type: str = "acd_mmkt_v3",
        fold: int = 0,
        is_train: bool = True,
        p_drop: float = 0.15,
        eval_mask: Optional[torch.Tensor] = None,
        data_dir: Optional[str] = None,
    ):
        """
        Initialize EediDataset.

        Args:
            static_item_bank: Path to processed directory / .pt file or loaded item bank dict.
            sequences: Path to sequence .pt file, loaded dict, or list of fold dicts to merge.
            model_type: Model identifier ('acd_mmkt_v3', 'mg_mmkt', 'mg_mmkt_v2', etc.).
            fold: Active validation fold index (0..4).
            is_train: Training mode enables dynamic Modality Dropout.
            p_drop: Dropout rate (for ACD-MMKTv3 drops only stem image; for MG-MMKT drops solution text).
            eval_mask: Optional custom evaluation mask overriding stored eval_mask.
            data_dir: Optional directory containing acd_v3_aux_metadata.pt.
        """
        super().__init__()
        self.model_type = model_type.lower()
        self.fold = fold
        self.is_train = is_train
        self.p_drop = p_drop if is_train else 0.0

        if isinstance(static_item_bank, str):
            if os.path.isdir(static_item_bank):
                bank_dict = load_eedi_static_item_bank(static_item_bank, model_type=self.model_type)
            else:
                bank_dict = torch.load(static_item_bank, map_location="cpu")
        else:
            bank_dict = static_item_bank

        if isinstance(sequences, str):
            self.seq_data = torch.load(sequences, map_location="cpu")
        elif isinstance(sequences, list):
            self.seq_data = self._merge_sequence_dicts(sequences)
        else:
            self.seq_data = sequences

        self.num_sequences = self.seq_data["q_seq"].shape[0]
        self.max_seq_len = self.seq_data["q_seq"].shape[1]

        # Lightweight item-level references (< 1 MB)
        self.modality_masks = bank_dict["modality_masks"]
        num_q = self.modality_masks.size(0)
        self.num_questions = num_q

        self.question_leaf_kcs = bank_dict["question_leaf_kcs"]
        self.question_kc_mask = bank_dict["question_kc_mask"]
        self.question_static_ctx = bank_dict.get(
            "question_static_ctx", torch.zeros((num_q, 15), dtype=torch.float32)
        )

        # Load auxiliary metadata for ACD-MMKTv3 / v2
        self.N_q = torch.zeros(num_q, dtype=torch.int64)
        self.D_q = torch.full((num_q,), 0.4627, dtype=torch.float32)
        self.omega_q = torch.zeros(num_q, dtype=torch.float32)
        self.mu_concept = torch.full((num_q,), -0.1495, dtype=torch.float32)
        self.mu_prior = torch.full((num_q,), 0.4627, dtype=torch.float32)
        self.M_patch_q = torch.zeros((num_q, 49), dtype=torch.float32)
        self.patch_valid_mask = torch.ones((num_q, 49), dtype=torch.bool)

        aux_source = None
        if "N_q_folds" in bank_dict and bank_dict["N_q_folds"] is not None:
            aux_source = bank_dict
        else:
            candidate_dir = data_dir or (
                static_item_bank if isinstance(static_item_bank, str) and os.path.isdir(static_item_bank) else None
            )
            if candidate_dir:
                aux_path = os.path.join(candidate_dir, "acd_v3_aux_metadata.pt")
                if os.path.exists(aux_path):
                    aux_source = torch.load(aux_path, map_location="cpu", weights_only=False)

        if aux_source is not None:
            if "N_q_folds" in aux_source and aux_source["N_q_folds"] is not None:
                self.N_q = aux_source["N_q_folds"][self.fold]
            if "D_q_folds" in aux_source and aux_source["D_q_folds"] is not None:
                self.D_q = aux_source["D_q_folds"][self.fold]
            if "omega_q_folds" in aux_source and aux_source["omega_q_folds"] is not None:
                self.omega_q = aux_source["omega_q_folds"][self.fold]
            if "mu_concept_folds" in aux_source and aux_source["mu_concept_folds"] is not None:
                self.mu_concept = aux_source["mu_concept_folds"][self.fold]
            if "mu_prior_folds" in aux_source and aux_source["mu_prior_folds"] is not None:
                self.mu_prior = aux_source["mu_prior_folds"][self.fold]
            if "M_patch_q" in aux_source and aux_source["M_patch_q"] is not None:
                self.M_patch_q = aux_source["M_patch_q"]
            if "patch_valid_mask" in aux_source and aux_source["patch_valid_mask"] is not None:
                self.patch_valid_mask = aux_source["patch_valid_mask"]

        self.ctt_difficulty = self.D_q
        self.ctt_valid_mask = self.N_q >= 5

        self.static_item_bank = {
            "modality_masks": self.modality_masks,
            "question_leaf_kcs": self.question_leaf_kcs,
            "question_kc_mask": self.question_kc_mask,
            "question_static_ctx": self.question_static_ctx,
            "ctt_difficulty": self.ctt_difficulty,
            "ctt_valid_mask": self.ctt_valid_mask,
            "N_q": self.N_q,
            "D_q": self.D_q,
            "omega_q": self.omega_q,
            "mu_concept": self.mu_concept,
            "mu_prior": self.mu_prior,
            "M_patch_q": self.M_patch_q,
            "patch_valid_mask": self.patch_valid_mask,
        }

        seq_mask = self.seq_data["seq_mask"].bool()
        if "loss_mask" in self.seq_data:
            self.loss_mask = self.seq_data["loss_mask"].bool()
        else:
            self.loss_mask = seq_mask.clone()
            self.loss_mask[:, 0] = False

        if eval_mask is not None:
            self.custom_eval_mask = eval_mask.bool()
        elif "eval_mask" in self.seq_data:
            self.custom_eval_mask = self.seq_data["eval_mask"].bool()
        else:
            self.custom_eval_mask = self.loss_mask.clone()

        # Select fold-specific post-diagnostic vector if stored as [5, N_seq, L, 8] or [N_seq, L, 8]
        self.has_post_diag = "post_diag_vec" in self.seq_data
        self.has_target_mask = "is_target_mask" in self.seq_data

    def _merge_sequence_dicts(self, dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Merge sequence dictionaries from multiple training folds along sequence axis."""
        keys = dicts[0].keys()
        merged: Dict[str, torch.Tensor] = {}
        for k in keys:
            # post_diag_vec is stored fold-indexed [5, N_seq, L, 8] to avoid distractor leakage across CV folds
            cat_dim = 1 if (k == "post_diag_vec" and dicts[0][k].ndim == 4 and dicts[0][k].size(0) == 5) else 0
            merged[k] = torch.cat([d[k] for d in dicts], dim=cat_dim)
        return merged

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        q_seq = self.seq_data["q_seq"][idx]
        r_seq = self.seq_data["r_seq"][idx]
        seq_mask = self.seq_data["seq_mask"][idx]
        loss_mask = self.loss_mask[idx]
        eval_mask = self.custom_eval_mask[idx]
        uid = self.seq_data["uid_seq"][idx]

        delta_t_lag = self.seq_data["delta_t_lag"][idx]
        delta_t_kc = self.seq_data["delta_t_kc"][idx]
        norm_delta_t_lag = self.seq_data.get(
            "norm_delta_t_lag", torch.clamp(delta_t_lag, min=0.0, max=1.0)
        )[idx]
        if "norm_delta_t_kc" in self.seq_data:
            norm_delta_t_kc = self.seq_data["norm_delta_t_kc"][idx]
        else:
            norm_delta_t_kc = torch.where(
                delta_t_kc >= 0,
                torch.clamp(delta_t_kc / 16.11809565, min=0.0, max=1.0),
                torch.zeros_like(delta_t_kc),
            )
        kc_first_flag = self.seq_data.get(
            "kc_first_flag", (delta_t_kc < 0).float()
        )[idx]
        batch_flag = self.seq_data["batch_flag"][idx]
        running_acc = self.seq_data["running_acc_prev"][idx]

        raw_mod_mask_seq = self.modality_masks[q_seq].clone()
        mod_mask_seq = raw_mod_mask_seq.clone()
        leaf_kcs_seq = self.question_leaf_kcs[q_seq]
        leaf_mask_seq = self.question_kc_mask[q_seq]
        static_ctx_seq = self.question_static_ctx[q_seq]

        # Dynamic Modality Dropout strictly matching ACD-MMKTv3 vs MG-MMKT semantics
        if self.is_train and self.p_drop > 0.0:
            if self.model_type in ("acd_mmkt_v2", "acd_mmkt_v3"):
                # Text is 100% preserved; drop only stem visual channel (m_img^s is identically 0)
                drop_q = (torch.rand(self.max_seq_len) >= self.p_drop).float()
                mod_mask_seq[:, 0] = mod_mask_seq[:, 0] * drop_q
            else:
                # MG-MMKT / MG-MMKTv2: sample solution text dropout for e_null^{txt,s}
                drop_sol = (torch.rand(self.max_seq_len) >= self.p_drop).float()
                mod_mask_seq[:, 1] = mod_mask_seq[:, 1] * drop_sol
                mod_mask_seq[:, 2] = mod_mask_seq[:, 2] * drop_sol

        # Legacy 21-dim dynamic context for backward compatibility with MG-MMKT/ACD-MMKTv1
        dynamic_ctx_seq = torch.cat(
            [
                static_ctx_seq,
                running_acc.unsqueeze(-1),
                norm_delta_t_lag.unsqueeze(-1),
                batch_flag.unsqueeze(-1),
                norm_delta_t_kc.unsqueeze(-1),
                mod_mask_seq[:, 1:3],
            ],
            dim=-1,
        )

        item: Dict[str, torch.Tensor] = {
            "uid": torch.as_tensor(uid, dtype=torch.int64),
            "context_start": torch.as_tensor(
                self.seq_data.get("context_start", torch.zeros(self.num_sequences, dtype=torch.int64))[idx],
                dtype=torch.int64,
            ),
            "q_seq": q_seq,
            "r_seq": r_seq,
            "seq_mask": seq_mask,
            "loss_mask": loss_mask,
            "eval_mask": eval_mask,
            "delta_t_lag": delta_t_lag,
            "delta_t_kc": delta_t_kc,
            "norm_delta_t_lag": norm_delta_t_lag,
            "norm_delta_t_kc": norm_delta_t_kc,
            "kc_first_flag": kc_first_flag,
            "batch_flag": batch_flag,
            "running_acc_prev": running_acc,
            "static_ctx": static_ctx_seq,
            "dynamic_ctx": dynamic_ctx_seq,
            "modality_masks": mod_mask_seq,
            "raw_modality_masks": raw_mod_mask_seq,
            "leaf_kcs": leaf_kcs_seq,
            "leaf_kc_mask": leaf_mask_seq,
            "ctt_difficulty": self.ctt_difficulty[q_seq],
            "ctt_valid_mask": self.ctt_valid_mask[q_seq],
            "N_q": self.N_q[q_seq],
            "D_q": self.D_q[q_seq],
            "omega_q": self.omega_q[q_seq],
            "mu_concept": self.mu_concept[q_seq],
            "mu_prior": self.mu_prior[q_seq],
            "M_patch_q": self.M_patch_q[q_seq],
            "patch_valid_mask": self.patch_valid_mask[q_seq],
        }

        if "ans_val_seq" in self.seq_data:
            item["ans_val_seq"] = self.seq_data["ans_val_seq"][idx]
        if "relative_quiz_pos" in self.seq_data:
            item["relative_quiz_pos"] = self.seq_data["relative_quiz_pos"][idx]
        if self.has_post_diag:
            # Slice fold-aligned post-response diagnostic vector [L, 8] if per-fold tensor [5, N, L, 8]
            pv = self.seq_data["post_diag_vec"]
            if pv.ndim == 4 and pv.size(0) == 5:
                item["post_diag_vec"] = pv[self.fold, idx]
            else:
                item["post_diag_vec"] = pv[idx]
        if self.has_target_mask:
            item["is_target_mask"] = self.seq_data["is_target_mask"][idx]

        return item

    def get_static_item_bank(self) -> Dict[str, torch.Tensor]:
        """Return lightweight cached static item metadata."""
        return self.static_item_bank


def collate_eedi(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate sequence samples into mini-batch tensors."""
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


def get_eedi_dataloaders(
    data_dir: str = "data/processed/Eedi",
    val_fold: int = 0,
    model_type: str = "acd_mmkt_v3",
    batch_size: int = 64,
    test_batch_size: int = 128,
    p_drop: float = 0.15,
    test_stride: int = 100,
    num_workers: int = 0,
    test_split: str = "unified",
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Construct Train, Validation, and Test DataLoaders for Eedi Task 3 & 4 (5-fold CV).

    Args:
        data_dir: Processed Eedi directory.
        val_fold: Fold index (0..4) designated for validation.
        model_type: Model identifier ('acd_mmkt_v3', 'mg_mmkt', 'mg_mmkt_v2', etc.).
        batch_size: Training batch size.
        test_batch_size: Validation/test batch size.
        p_drop: Dynamic modality dropout rate during training.
        test_stride: Context-preserving test window stride (default: 100).
        num_workers: DataLoader workers.
        test_split: Which test split to load ('unified', 'public', or 'private').

    Returns:
        Tuple of (train_loader, val_loader, test_loader, static_item_bank).
    """
    static_item_bank = load_eedi_static_item_bank(data_dir, model_type=model_type)

    train_fold_dicts = []
    for f in range(5):
        if f != val_fold:
            fold_file = os.path.join(data_dir, f"sequences_fold_{f}.pt")
            train_fold_dicts.append(torch.load(fold_file, map_location="cpu"))

    train_dataset = EediDataset(
        static_item_bank=static_item_bank,
        sequences=train_fold_dicts,
        model_type=model_type,
        fold=val_fold,
        is_train=True,
        p_drop=p_drop,
        data_dir=data_dir,
    )

    val_file = os.path.join(data_dir, f"sequences_fold_{val_fold}.pt")
    val_dataset = EediDataset(
        static_item_bank=static_item_bank,
        sequences=val_file,
        model_type=model_type,
        fold=val_fold,
        is_train=False,
        p_drop=0.0,
        data_dir=data_dir,
    )

    test_filename = {
        "unified": "sequences_test.pt",
        "public": "sequences_test_public.pt",
        "private": "sequences_test_private.pt",
    }[test_split.lower()]
    test_file = os.path.join(data_dir, test_filename)
    raw_test_data = torch.load(test_file, map_location="cpu")
    strided_test_data, test_eval_mask = build_eedi_test_strided_windows(
        raw_test_data,
        max_seq_len=200,
        stride=test_stride,
    )
    test_dataset = EediDataset(
        static_item_bank=static_item_bank,
        sequences=strided_test_data,
        model_type=model_type,
        fold=val_fold,
        is_train=False,
        p_drop=0.0,
        eval_mask=test_eval_mask,
        data_dir=data_dir,
    )

    worker_fn = seed_worker if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_eedi,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_eedi,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_eedi,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_fn,
    )

    return train_loader, val_loader, test_loader, static_item_bank
