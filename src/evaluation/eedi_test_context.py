"""
Aligned Test Context and Task 3 Expert Pairwise Validation for Eedi NeurIPS 2020 (Task 3 & 4).

Mirrors the TestContext interface used by the unified XES3G5M benchmark and modality
ablation pipeline while reading directly from Eedi's pre-built context-preserving strided
test windows (`sequences_test*.pt`) and stratified evaluation masks (`test_strata_masks.pt`).
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.data.eedi_dataset import build_eedi_test_strided_windows


EXPECTED_EEDI_EVAL_COUNTS = {
    "unified": 124_960,
    "public": 74_276,
    "private": 50_684,
}


class EediTestContext:
    """
    Aligned per-interaction metadata for Eedi Task 4 evaluated test interactions.

    Attributes:
        targets       : Ground-truth binary responses r_t [N_eval].
        q_ids         : Question IDs in [0, 947] [N_eval].
        uids          : Original student IDs [N_eval].
        student_index : Dense student index in [0, S) [N_eval].
    """

    def __init__(
        self,
        processed_dir: str = "data/processed/Eedi",
        test_split: str = "unified",
        test_stride: int = 100,
    ):
        self.processed_dir = processed_dir
        self.test_split = test_split.lower()
        self.test_stride = test_stride
        self._strata_cache: Optional[Dict[str, np.ndarray]] = None
        self._build()

    def _build(self) -> None:
        filename = {
            "unified": "sequences_test.pt",
            "public": "sequences_test_public.pt",
            "private": "sequences_test_private.pt",
        }[self.test_split]

        seq_path = os.path.join(self.processed_dir, filename)
        if not os.path.exists(seq_path):
            raise FileNotFoundError(f"Eedi test sequence file not found: {seq_path}")

        raw_seq = torch.load(seq_path, map_location="cpu")
        strided_seq, eval_mask = build_eedi_test_strided_windows(
            raw_seq, max_seq_len=200, stride=self.test_stride
        )
        eval_mask_b = eval_mask.bool()

        expected_n = EXPECTED_EEDI_EVAL_COUNTS.get(self.test_split)
        n_eval = int(eval_mask_b.sum().item())
        if expected_n is not None:
            assert n_eval == expected_n, (
                f"Eedi '{self.test_split}' split has {n_eval} eval steps, expected {expected_n}"
            )

        self.targets = strided_seq["r_seq"][eval_mask_b].numpy().astype(np.int64)
        self.q_ids = strided_seq["q_seq"][eval_mask_b].numpy().astype(np.int64)
        uid_grid = strided_seq["uid_seq"].unsqueeze(1).expand_as(eval_mask_b)
        self.uids = uid_grid[eval_mask_b].numpy().astype(np.int64)
        _, self.student_index = np.unique(self.uids, return_inverse=True)
        self.student_index = self.student_index.astype(np.int64)

        strata_path = os.path.join(self.processed_dir, "test_strata_masks.pt")
        strata_map: Dict[str, np.ndarray] = {
            "overall": np.ones(n_eval, dtype=bool),
        }
        if os.path.exists(strata_path):
            raw_strata = torch.load(strata_path, map_location="cpu")
            split_strata = raw_strata.get(self.test_split, {})
            for key, mask_2d in split_strata.items():
                if isinstance(mask_2d, torch.Tensor) and mask_2d.shape == eval_mask_b.shape:
                    strata_map[key] = mask_2d.bool()[eval_mask_b].numpy().astype(bool)

            # Map canonical keys expected by the shared modality ablation pipeline
            alias_pairs = [
                ("q_stem_image", "visual_diagram"),
                ("q_text_only", "text_only"),
                ("after_incorrect", "after_error"),
                ("after_correct", "after_correct"),
                ("freq_rare_le10", "cold_start_q"),
                ("freq_medium_11_100", "medium_freq_q"),
                ("freq_frequent_gt100", "frequent_q"),
                ("single_kc", "single_kc"),
                ("multi_kc", "multi_kc"),
            ]
            for canonical_key, eedi_key in alias_pairs:
                if eedi_key in strata_map:
                    strata_map[canonical_key] = strata_map[eedi_key]

        self._strata_cache = strata_map

    def strata_masks(self) -> Dict[str, np.ndarray]:
        """Return aligned 1D boolean masks for all Eedi evaluation strata."""
        assert self._strata_cache is not None
        return self._strata_cache


def evaluate_task3_expert_pairs(
    a_q: np.ndarray,
    tilde_mu: np.ndarray,
    task3_expert_pairs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluate alignment between ACD-MMKTv3 item discrimination a_q and Eedi Task 3
    human-expert pairwise question quality judgments (50 pairs across public & private).

    Args:
        a_q: Item discrimination parameters [948].
        tilde_mu: Item difficulty logits [948].
        task3_expert_pairs: List of 50 expert consensus dictionaries.

    Returns:
        Dictionary of pairwise agreement rates and breakdowns.
    """
    if not task3_expert_pairs:
        return {"num_pairs": 0}

    a_arr = np.asarray(a_q, dtype=np.float64).reshape(-1)
    mu_arr = np.asarray(tilde_mu, dtype=np.float64).reshape(-1)

    disc_hits: List[int] = []
    pub_hits: List[int] = []
    priv_hits: List[int] = []
    strong_hits: List[int] = []
    balanced_hits: List[int] = []

    for pair in task3_expert_pairs:
        q_left = int(pair["left"])
        q_right = int(pair["right"])
        pref = int(pair["consensus_pref"])  # 1 = left preferred, 2 = right preferred

        # Higher discrimination a_q predicts stronger diagnostic differentiation
        pred_disc = 1 if a_arr[q_left] >= a_arr[q_right] else 2
        hit_disc = int(pred_disc == pref)
        disc_hits.append(hit_disc)

        # Balanced difficulty near 0 logit (-|tilde_mu|) as auxiliary diagnostic check
        pred_bal = 1 if abs(mu_arr[q_left]) <= abs(mu_arr[q_right]) else 2
        balanced_hits.append(int(pred_bal == pref))

        split_name = str(pair.get("split", "")).lower()
        if split_name == "public":
            pub_hits.append(hit_disc)
        elif split_name == "private":
            priv_hits.append(hit_disc)

        max_votes = max(int(pair.get("left_votes", 0)), int(pair.get("right_votes", 0)))
        if max_votes >= 4:
            strong_hits.append(hit_disc)

    return {
        "num_pairs": len(task3_expert_pairs),
        "a_q_agreement": float(np.mean(disc_hits)),
        "a_q_correct": int(np.sum(disc_hits)),
        "a_q_public_agreement": float(np.mean(pub_hits)) if pub_hits else 0.0,
        "a_q_private_agreement": float(np.mean(priv_hits)) if priv_hits else 0.0,
        "a_q_strong_consensus_agreement": float(np.mean(strong_hits)) if strong_hits else 0.0,
        "num_strong_consensus_pairs": len(strong_hits),
        "balanced_difficulty_agreement": float(np.mean(balanced_hits)),
    }
