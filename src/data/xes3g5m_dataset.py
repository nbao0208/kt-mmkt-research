"""
PyTorch Dataset and DataLoader Utilities for XES3G5M Multimodal Knowledge Tracing.

Supports:
- Two-tier storage architecture with fast item bank indexing.
- 5-Fold Cross-Validation data merging (4 folds train, 1 fold val) strictly partitioned by student UID.
- Strided Context Windowing for test sequence evaluation preserving continuous causal student history.
- Training-time dynamic Modality Dropout (p_drop in [0.15, 0.20]) to ensure non-zero gradient flow to e_null.
- pyKT-compliant loss mask and evaluation mask generation (first interaction of student excluded from evaluation).
"""

from typing import Dict, Any, Optional, Union, Tuple, List
import os
from collections import defaultdict
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def seed_worker(worker_id: int) -> None:
    """Ensure DataLoader worker subprocesses receive distinct and epoch-varying seeds."""
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def create_mg_static_item_bank(
    source_path: str,
    target_path: str,
) -> Dict[str, torch.Tensor]:
    """
    Extract lightweight MG-MMKT static item bank (~48 MB) from the full 4.6 GB item bank.
    Reduces memory footprint and speeds up DataLoader initialization by ~100x.

    Args:
        source_path: Path to the full static_item_bank.pt.
        target_path: Destination path for mg_static_item_bank.pt.

    Returns:
        Extracted MG item bank dictionary.
    """
    if os.path.exists(target_path):
        return torch.load(target_path, map_location="cpu")

    print(f"Extracting MG-MMKT static item bank from {source_path} to {target_path}...")
    full_bank = torch.load(source_path, map_location="cpu")
    mg_bank = {
        "modality_masks": full_bank["modality_masks"],
        "question_leaf_kcs": full_bank["question_leaf_kcs"],
        "question_kc_mask": full_bank["question_kc_mask"],
        "question_static_ctx": full_bank["question_static_ctx"],
        "mg_v_txt_q": full_bank["mg_v_txt_q"],
        "mg_v_txt_s": full_bank["mg_v_txt_s"],
        "mg_v_img_q": full_bank["mg_v_img_q"],
        "mg_v_img_s": full_bank["mg_v_img_s"],
    }
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    torch.save(mg_bank, target_path)
    print(f"Successfully saved mg_static_item_bank.pt ({os.path.getsize(target_path) / 1e6:.2f} MB).")
    return mg_bank


def load_static_item_bank_optimized(
    processed_dir: str,
    model_type: str = "mg_mmkt",
) -> Dict[str, torch.Tensor]:
    """
    Load the static item bank. For acd_mmkt, loads the complete static_item_bank.pt
    containing fine-grained token/patch tensors and hypergraphs. For mg_mmkt,
    prioritizes the lightweight mg_static_item_bank.pt.

    Args:
        processed_dir: Path to directory containing processed XES3G5M files.
        model_type: Model identifier ('mg_mmkt' or 'acd_mmkt').

    Returns:
        Dictionary of static item properties.
    """
    full_path = os.path.join(processed_dir, "static_item_bank.pt")
    mg_path = os.path.join(processed_dir, "mg_static_item_bank.pt")

    if model_type.lower() == "acd_mmkt":
        if os.path.exists(full_path):
            try:
                # Use mmap=True to avoid loading 4.5 GB into physical CPU RAM
                return torch.load(full_path, mmap=True, map_location="cpu")
            except Exception:
                return torch.load(full_path, map_location="cpu")
        raise FileNotFoundError(f"Full static item bank not found at {full_path} for acd_mmkt")

    if os.path.exists(mg_path):
        return torch.load(mg_path, map_location="cpu")
    elif os.path.exists(full_path):
        return create_mg_static_item_bank(full_path, mg_path)
    else:
        raise FileNotFoundError(f"Neither {mg_path} nor {full_path} found in {processed_dir}")


class XES3G5MDataset(Dataset):
    """
    Two-Tier PyTorch Dataset for XES3G5M Multimodal Knowledge Tracing.
    Provides sequence batches with dynamic modality masking, loss masks, and evaluation masks.
    """

    def __init__(
        self,
        static_item_bank: Union[str, Dict[str, torch.Tensor]],
        sequences: Union[str, Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
        model_type: str = "mg_mmkt",
        fold: int = 0,
        is_train: bool = True,
        p_drop: float = 0.20,
        eval_mask: Optional[torch.Tensor] = None,
    ):
        """
        Initialize XES3G5MDataset.

        Args:
            static_item_bank: Path to static item bank or pre-loaded dictionary.
            sequences: Path to sequence file, sequence dict, or list of sequence dicts to concatenate.
            model_type: Model identifier ('mg_mmkt' or 'acd_mmkt').
            fold: Active cross-validation fold (0..4).
            is_train: Whether dataset is in training mode (enables Modality Dropout).
            p_drop: Probability of dropping solution modalities during training (default: 0.20).
            eval_mask: Optional custom evaluation mask tensor [N_seq, L] (e.g. for strided test evaluation).
        """
        super().__init__()
        self.model_type = model_type.lower()
        self.fold = fold
        self.is_train = is_train
        self.p_drop = p_drop if is_train else 0.0

        # Load Static Item Bank metadata (avoiding retaining 4.3 GB acd_E_* token/patch tensors in dataset memory)
        if isinstance(static_item_bank, str):
            if os.path.isdir(static_item_bank):
                bank_dict = load_static_item_bank_optimized(static_item_bank)
            else:
                bank_dict = torch.load(static_item_bank, map_location="cpu")
        else:
            bank_dict = static_item_bank

        # Load Sequence Store (merge multiple folds if list provided)
        if isinstance(sequences, str):
            self.seq_data = torch.load(sequences, map_location="cpu")
        elif isinstance(sequences, list):
            self.seq_data = self._merge_sequence_dicts(sequences)
        else:
            self.seq_data = sequences

        self.num_sequences = self.seq_data["q_seq"].shape[0]
        self.max_seq_len = self.seq_data["q_seq"].shape[1]

        # Cache reference tensors for quick lookup
        self.modality_masks = bank_dict["modality_masks"]  # [7652, 3]
        self.question_leaf_kcs = bank_dict["question_leaf_kcs"]  # [7652, 6]
        self.question_kc_mask = bank_dict["question_kc_mask"]  # [7652, 6]
        self.question_static_ctx = bank_dict.get(
            "question_static_ctx", torch.zeros((7652, 15), dtype=torch.float32)
        )

        # Fold-specific CTT difficulty labels if available
        if "acd_ctt_difficulty_folds" in bank_dict:
            self.ctt_difficulty = bank_dict["acd_ctt_difficulty_folds"][self.fold]  # [7652]
            self.ctt_valid_mask = bank_dict["acd_ctt_valid_mask"][self.fold]  # [7652]
        else:
            self.ctt_difficulty = torch.zeros(7652, dtype=torch.float32)
            self.ctt_valid_mask = torch.zeros(7652, dtype=torch.bool)

        # Retain only lightweight item metadata (<1 MB) so del static_item_bank can truly free the 4.3 GB dictionary
        self.static_item_bank = {
            "modality_masks": self.modality_masks,
            "question_leaf_kcs": self.question_leaf_kcs,
            "question_kc_mask": self.question_kc_mask,
            "question_static_ctx": self.question_static_ctx,
            "ctt_difficulty": self.ctt_difficulty,
            "ctt_valid_mask": self.ctt_valid_mask,
        }

        # Precompute or assign loss masks and evaluation masks
        # Standard pyKT KT loss mask: step 0 of each student sequence has loss_mask = 0, steps 1..L follow seq_mask
        seq_mask = self.seq_data["seq_mask"]  # [N_seq, L]
        self.loss_mask = seq_mask.clone()
        self.loss_mask[:, 0] = False  # Student has no prior interaction history at step 0

        if eval_mask is not None:
            self.custom_eval_mask = eval_mask
        else:
            self.custom_eval_mask = self.loss_mask.clone()

    def _merge_sequence_dicts(self, dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Merge multiple sequence dictionaries along dimension 0.
        """
        keys = dicts[0].keys()
        merged = {}
        for k in keys:
            tensors = [d[k] for d in dicts]
            merged[k] = torch.cat(tensors, dim=0)
        return merged

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve a single student interaction sequence window.

        Returns:
            Dictionary containing sequence features, modality masks, loss masks, and eval masks.
        """
        q_seq = self.seq_data["q_seq"][idx]  # [L]
        r_seq = self.seq_data["r_seq"][idx]  # [L]
        seq_mask = self.seq_data["seq_mask"][idx]  # [L]
        loss_mask = self.loss_mask[idx]  # [L]
        eval_mask = self.custom_eval_mask[idx]  # [L]
        uid = self.seq_data["uid_seq"][idx]

        delta_t_lag = self.seq_data.get(
            "delta_t_lag", torch.zeros_like(q_seq, dtype=torch.float32)
        )[idx]
        delta_t_kc = self.seq_data.get(
            "delta_t_kc", torch.zeros_like(q_seq, dtype=torch.float32)
        )[idx]
        batch_flag = self.seq_data.get(
            "batch_flag", torch.zeros_like(q_seq, dtype=torch.float32)
        )[idx]
        running_acc = self.seq_data.get(
            "running_acc_prev", torch.zeros_like(q_seq, dtype=torch.float32)
        )[idx]

        # Retrieve static item properties for this sequence
        raw_mod_mask_seq = self.modality_masks[q_seq].clone()  # [L, 3] clean ground-truth mask
        mod_mask_seq = raw_mod_mask_seq.clone()  # [L, 3] dynamic dropped mask
        leaf_kcs_seq = self.question_leaf_kcs[q_seq]  # [L, 6]
        leaf_mask_seq = self.question_kc_mask[q_seq]  # [L, 6]
        static_ctx_seq = self.question_static_ctx[q_seq]  # [L, 15]

        # Apply Dynamic Modality Dropout during training (p_drop on solution modalities)
        if self.is_train and self.p_drop > 0.0:
            # Independent Bernoulli mask per step
            drop_sample = torch.rand(self.max_seq_len) >= self.p_drop  # True with prob 1 - p_drop
            # If dropped, solution text and solution image are masked out
            mod_mask_seq[:, 1] = mod_mask_seq[:, 1] * drop_sample.float()
            mod_mask_seq[:, 2] = mod_mask_seq[:, 2] * drop_sample.float()

        # Build dynamic context vector (21 dimensions: 15 static + 4 dynamic temporal/proficiency + 2 solution presence flags)
        dynamic_ctx_seq = torch.cat(
            [
                static_ctx_seq,
                running_acc.unsqueeze(-1),
                delta_t_lag.unsqueeze(-1),
                batch_flag.unsqueeze(-1),
                delta_t_kc.unsqueeze(-1),
                mod_mask_seq[:, 1:3],
            ],
            dim=-1,
        )  # [L, 21]

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
            "batch_flag": batch_flag,
            "running_acc_prev": running_acc,
            "dynamic_ctx": dynamic_ctx_seq,
            "modality_masks": mod_mask_seq,
            "raw_modality_masks": raw_mod_mask_seq,
            "leaf_kcs": leaf_kcs_seq,
            "leaf_kc_mask": leaf_mask_seq,
            "ctt_difficulty": self.ctt_difficulty[q_seq],
            "ctt_valid_mask": self.ctt_valid_mask[q_seq],
        }

        return item

    def get_static_item_bank(self) -> Dict[str, torch.Tensor]:
        """
        Access the underlying Tier 1 static item bank.
        """
        return self.static_item_bank


def build_test_strided_windows(
    test_seq_data: Dict[str, torch.Tensor],
    max_seq_len: int = 200,
    stride: int = 100,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Construct context-preserving strided evaluation windows from continuous student test sequences.
    
    Guarantees:
    1. Every interaction at t > 0 is evaluated exactly once (total = total_valid_interactions - num_students).
    2. Interactions at t >= max_seq_len maintain at least (max_seq_len - stride) steps of causal history context.
    3. Reconstructs unbroken student sequences from raw sequence windows grouped by student UID.

    Args:
        test_seq_data: Dictionary loaded from sequences_test.pt.
        max_seq_len: Maximum window sequence length (default: 200).
        stride: Evaluation step stride for subsequent windows (default: 100).

    Returns:
        Tuple of (re-windowed sequence dictionary, corresponding eval_mask tensor).
    """
    uids = test_seq_data["uid_seq"].tolist()
    uid_to_indices = defaultdict(list)
    for idx, u in enumerate(uids):
        uid_to_indices[u].append(idx)

    new_q, new_r, new_mask, new_lag, new_kc, new_batch, new_acc, new_uids, new_eval_mask, new_context_start = (
        [], [], [], [], [], [], [], [], [], []
    )

    for u, indices in uid_to_indices.items():
        # Reconstruct full continuous student trajectory
        q_list, r_list, lag_list, kc_list, batch_list, acc_list = [], [], [], [], [], []
        for idx in indices:
            valid_len = int(test_seq_data["seq_mask"][idx].sum().item())
            q_list.append(test_seq_data["q_seq"][idx, :valid_len])
            r_list.append(test_seq_data["r_seq"][idx, :valid_len])
            lag_list.append(test_seq_data["delta_t_lag"][idx, :valid_len])
            kc_list.append(test_seq_data["delta_t_kc"][idx, :valid_len])
            batch_list.append(test_seq_data["batch_flag"][idx, :valid_len])
            acc_list.append(test_seq_data["running_acc_prev"][idx, :valid_len])

        full_q = torch.cat(q_list)
        full_r = torch.cat(r_list)
        full_lag = torch.cat(lag_list)
        full_kc = torch.cat(kc_list)
        full_batch = torch.cat(batch_list)
        full_acc = torch.cat(acc_list)
        total_steps = len(full_q)

        # Window 1: steps [0, min(max_seq_len, total_steps)]
        # Evaluated steps: [1, min(max_seq_len, total_steps) - 1] (step 0 has no history)
        w_len = min(max_seq_len, total_steps)
        chunk_q = torch.zeros(max_seq_len, dtype=torch.int64)
        chunk_r = torch.zeros(max_seq_len, dtype=torch.int64)
        chunk_mask = torch.zeros(max_seq_len, dtype=torch.bool)
        chunk_lag = torch.zeros(max_seq_len, dtype=torch.float32)
        chunk_kc = torch.zeros(max_seq_len, dtype=torch.float32)
        chunk_batch = torch.zeros(max_seq_len, dtype=torch.float32)
        chunk_acc = torch.zeros(max_seq_len, dtype=torch.float32)
        chunk_eval = torch.zeros(max_seq_len, dtype=torch.bool)

        chunk_q[:w_len] = full_q[:w_len]
        chunk_r[:w_len] = full_r[:w_len]
        chunk_mask[:w_len] = True
        chunk_lag[:w_len] = full_lag[:w_len]
        chunk_kc[:w_len] = full_kc[:w_len]
        chunk_batch[:w_len] = full_batch[:w_len]
        chunk_acc[:w_len] = full_acc[:w_len]
        if w_len > 1:
            chunk_eval[1:w_len] = True

        new_q.append(chunk_q)
        new_r.append(chunk_r)
        new_mask.append(chunk_mask)
        new_lag.append(chunk_lag)
        new_kc.append(chunk_kc)
        new_batch.append(chunk_batch)
        new_acc.append(chunk_acc)
        new_uids.append(u)
        new_eval_mask.append(chunk_eval)
        new_context_start.append(0)

        # Subsequent windows: start sliding from max_seq_len with step stride
        cur_eval_start = max_seq_len
        while cur_eval_start < total_steps:
            cur_eval_end = min(cur_eval_start + stride, total_steps)
            context_start = max(0, cur_eval_end - max_seq_len)
            slice_len = cur_eval_end - context_start

            chunk_q = torch.zeros(max_seq_len, dtype=torch.int64)
            chunk_r = torch.zeros(max_seq_len, dtype=torch.int64)
            chunk_mask = torch.zeros(max_seq_len, dtype=torch.bool)
            chunk_lag = torch.zeros(max_seq_len, dtype=torch.float32)
            chunk_kc = torch.zeros(max_seq_len, dtype=torch.float32)
            chunk_batch = torch.zeros(max_seq_len, dtype=torch.float32)
            chunk_acc = torch.zeros(max_seq_len, dtype=torch.float32)
            chunk_eval = torch.zeros(max_seq_len, dtype=torch.bool)

            chunk_q[:slice_len] = full_q[context_start:cur_eval_end]
            chunk_r[:slice_len] = full_r[context_start:cur_eval_end]
            chunk_mask[:slice_len] = True
            chunk_lag[:slice_len] = full_lag[context_start:cur_eval_end]
            chunk_kc[:slice_len] = full_kc[context_start:cur_eval_end]
            chunk_batch[:slice_len] = full_batch[context_start:cur_eval_end]
            chunk_acc[:slice_len] = full_acc[context_start:cur_eval_end]

            # In this window, only the newly introduced steps [cur_eval_start - context_start : slice_len] are evaluated
            eval_local_start = cur_eval_start - context_start
            chunk_eval[eval_local_start:slice_len] = True

            new_q.append(chunk_q)
            new_r.append(chunk_r)
            new_mask.append(chunk_mask)
            new_lag.append(chunk_lag)
            new_kc.append(chunk_kc)
            new_batch.append(chunk_batch)
            new_acc.append(chunk_acc)
            new_uids.append(u)
            new_eval_mask.append(chunk_eval)
            new_context_start.append(context_start)

            cur_eval_start = cur_eval_end

    reconstructed_dict = {
        "q_seq": torch.stack(new_q, dim=0),
        "r_seq": torch.stack(new_r, dim=0),
        "seq_mask": torch.stack(new_mask, dim=0),
        "delta_t_lag": torch.stack(new_lag, dim=0),
        "delta_t_kc": torch.stack(new_kc, dim=0),
        "batch_flag": torch.stack(new_batch, dim=0),
        "running_acc_prev": torch.stack(new_acc, dim=0),
        "uid_seq": torch.tensor(new_uids, dtype=torch.int64),
        "context_start": torch.tensor(new_context_start, dtype=torch.int64),
    }
    reconstructed_eval_mask = torch.stack(new_eval_mask, dim=0)

    return reconstructed_dict, reconstructed_eval_mask


def collate_xes3g5m(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Standard PyTorch collate function for batching XES3G5M sequence records.
    """
    collated = {}
    keys = batch[0].keys()
    for k in keys:
        collated[k] = torch.stack([b[k] for b in batch], dim=0)
    return collated


def get_xes3g5m_dataloaders(
    data_dir: str,
    val_fold: int = 0,
    model_type: str = "mg_mmkt",
    batch_size: int = 64,
    test_batch_size: int = 128,
    p_drop: float = 0.20,
    test_stride: int = 100,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, torch.Tensor]]:
    """
    Construct PyTorch DataLoaders for Train, Validation, and Test splits according to pyKT 5-fold CV protocol.

    Args:
        data_dir: Path to processed XES3G5M directory (e.g. 'data/processed/XES3G5M').
        val_fold: Fold index (0..4) designated as validation set. The other 4 folds form the train set.
        model_type: Model identifier ('mg_mmkt' or 'acd_mmkt').
        batch_size: Batch size for training.
        test_batch_size: Batch size for validation and testing.
        p_drop: Modality dropout rate during training.
        test_stride: Stride for context-preserving test windowing (default: 100).
        num_workers: Number of DataLoader worker subprocesses (default: 0).

    Returns:
        Tuple of (train_loader, val_loader, test_loader, static_item_bank).
    """
    static_item_bank = load_static_item_bank_optimized(data_dir, model_type=model_type)

    # 1. Train set: Merge 4 non-validation folds
    train_fold_dicts = []
    for f in range(5):
        if f != val_fold:
            fold_file = os.path.join(data_dir, f"sequences_fold_{f}.pt")
            train_fold_dicts.append(torch.load(fold_file, map_location="cpu"))

    train_dataset = XES3G5MDataset(
        static_item_bank=static_item_bank,
        sequences=train_fold_dicts,
        model_type=model_type,
        fold=val_fold,
        is_train=True,
        p_drop=p_drop,
    )

    # 2. Validation set: designated validation fold with is_train=False
    val_file = os.path.join(data_dir, f"sequences_fold_{val_fold}.pt")
    val_dataset = XES3G5MDataset(
        static_item_bank=static_item_bank,
        sequences=val_file,
        model_type=model_type,
        fold=val_fold,
        is_train=False,
        p_drop=0.0,
    )

    # 3. Test set: Strided context windows on sequences_test.pt
    test_file = os.path.join(data_dir, "sequences_test.pt")
    raw_test_data = torch.load(test_file, map_location="cpu")
    strided_test_data, test_eval_mask = build_test_strided_windows(
        raw_test_data,
        max_seq_len=200,
        stride=test_stride,
    )
    test_dataset = XES3G5MDataset(
        static_item_bank=static_item_bank,
        sequences=strided_test_data,
        model_type=model_type,
        fold=val_fold,
        is_train=False,
        p_drop=0.0,
        eval_mask=test_eval_mask,
    )

    # Instantiate DataLoaders
    worker_fn = seed_worker if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_xes3g5m,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_xes3g5m,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_xes3g5m,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_fn,
    )


    return train_loader, val_loader, test_loader, static_item_bank
