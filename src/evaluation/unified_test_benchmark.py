"""
Unified Final Test Benchmark: inference engines and aligned test context for KT / MMKT models on XES3G5M.

Data protocol (strictly separated per model family):
- pyKT baselines (DKT, DKVMN, SAKT) were trained on the ORIGINAL XES3G5M release
  (data/raw/XES3G5M/kc_level). They are evaluated on the original `kc_level/test.csv` with the
  question-level, anti-leakage late-fusion protocol implemented in `PyKTEvaluator`.
- MMKT models (MG-MMKT, MG-MMKTv2, ACD-MMKTv3) were trained on the MMKT-PREPROCESSED XES3G5M
  (data/processed/XES3G5M). They are evaluated on `sequences_test.pt` rebuilt into
  context-preserving strided windows (max_len = 200, stride = 100), exactly as during training.

Both protocols score the same 1,099,197 question-level test interactions (every question except
the first one of each of the 3,613 test students) in the same order. `TestContext` exposes the
shared ground truth and stratification metadata derived from the original release, and every
MMKT prediction run is verified element-wise against it (targets and question IDs).
"""

import os
import re
import json
import time
from collections import Counter
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.data.xes3g5m_dataset import (
    XES3G5MDataset,
    build_test_strided_windows,
    collate_xes3g5m,
    load_static_item_bank_optimized,
)
from src.utils.logger import get_logger


NUM_TEST_INTERACTIONS = 1_099_197

# Registry of the six fully-trained (non-ablation) models evaluated in the benchmark.
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "dkt": {
        "display": "DKT",
        "family": "pykt",
        "data_source": "original XES3G5M (kc_level/test.csv)",
        "ckpt_pattern": "outputs/checkpoints/dkt_fold{fold}/best_model.pt",
        "legacy_preds_pattern": "outputs/checkpoints/dkt_fold{fold}/test_preds.npy",
    },
    "dkvmn": {
        "display": "DKVMN",
        "family": "pykt",
        "data_source": "original XES3G5M (kc_level/test.csv)",
        "ckpt_pattern": "outputs/checkpoints/dkvmn_fold{fold}/best_model.pt",
        "legacy_preds_pattern": "outputs/checkpoints/dkvmn_fold{fold}/test_preds.npy",
    },
    "sakt": {
        "display": "SAKT",
        "family": "pykt",
        "data_source": "original XES3G5M (kc_level/test.csv)",
        "ckpt_pattern": "outputs/checkpoints/sakt_fold{fold}/best_model.pt",
        "legacy_preds_pattern": "outputs/checkpoints/sakt_fold{fold}/test_preds.npy",
    },
    "mg_mmkt": {
        "display": "MG-MMKT",
        "family": "mg_mmkt",
        "data_source": "MMKT-preprocessed XES3G5M (sequences_test.pt)",
        "ckpt_pattern": "outputs/checkpoints/mg_mmkt_fold{fold}/best_model.pt",
        "legacy_preds_pattern": None,
    },
    "mg_mmkt_v2": {
        "display": "MG-MMKTv2",
        "family": "mg_mmkt_v2",
        "data_source": "MMKT-preprocessed XES3G5M (sequences_test.pt)",
        "ckpt_pattern": "outputs/checkpoints/v2_full_5fold/mg_mmkt_v2_fold{fold}/best_model.pt",
        "legacy_preds_pattern": "outputs/checkpoints/v2_full_5fold/mg_mmkt_v2_fold{fold}/test_preds.npy",
    },
    "acd_mmkt_v3": {
        "display": "ACD-MMKTv3",
        "family": "acd_mmkt_v3",
        "data_source": "MMKT-preprocessed XES3G5M (sequences_test.pt)",
        "ckpt_pattern": "outputs/checkpoints/acd_mmkt_v3_full_5fold/acd_mmkt_v3_fold{fold}/best_model.pt",
        "legacy_preds_pattern": "outputs/checkpoints/acd_mmkt_v3_full_5fold/acd_mmkt_v3_fold{fold}/test_predictions.npy",
    },
}

BASELINE_MODELS = ["dkt", "dkvmn", "sakt"]
MMKT_MODELS = ["mg_mmkt", "mg_mmkt_v2", "acd_mmkt_v3"]


def resolve_device(device_str: str) -> torch.device:
    """Resolve 'auto' to the best available accelerator (cuda > mps > cpu)."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def synchronize_device(device: torch.device) -> None:
    """Block until all queued kernels on the device are finished (accurate wall-clock timing)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


# ---------------------------------------------------------------------------------------------
# Shared test context (ground truth + stratification metadata from the ORIGINAL release)
# ---------------------------------------------------------------------------------------------
class TestContext:
    """
    Aligned per-interaction metadata for the 1,099,197 question-level test interactions.

    Built exclusively from the original XES3G5M release so that all strata are defined identically
    for every model regardless of its training data pipeline:
        targets          : ground-truth response r_t.
        q_ids            : question ID of the evaluated interaction.
        uids / student_index : student ID and its dense index.
        position         : question index t within the student trajectory (t >= 1).
        kc_history_len   : number of KC-level steps preceding question t (pyKT unrolled length).
        num_kcs          : number of KCs tagged on question t.
        m_img_q          : 1 if the question stem/options contain an image.
        m_img_s_q        : 1 if the question's own solution (analysis) contains an image.
        prev_m_img_s     : 1 if the solution of question t-1 contains an image.
        prev_response    : response r_{t-1}.
        train_q_count    : occurrences of question q in the full train+valid split.
    """

    def __init__(self, raw_root_dir: str = "data/raw/XES3G5M", processed_dir: str = "data/processed/XES3G5M"):
        self.raw_root_dir = raw_root_dir
        self.processed_dir = processed_dir
        self._build()

    def _build(self) -> None:
        test_csv = os.path.join(self.raw_root_dir, "kc_level", "test.csv")
        df = pd.read_csv(test_csv)

        targets, q_ids, uids, positions, kc_hist, num_kcs, prev_resp, prev_q = ([] for _ in range(8))
        for _, row in df.iterrows():
            uid = int(row["uid"])
            qs = [int(x) for x in row["questions"].split(",")]
            cs = [int(x) for x in row["concepts"].split(",")]
            rs = [int(x) for x in row["responses"].split(",")]
            reps = [int(x) for x in row["is_repeat"].split(",")]

            # Group unrolled KC-level steps back into questions (is_repeat == 1 continues a question)
            grouped = []
            for q, c, r, rep in zip(qs, cs, rs, reps):
                if q == -1:
                    continue
                if rep == 0:
                    grouped.append([q, [c], r])
                else:
                    grouped[-1][1].append(c)

            kc_steps_before = 0
            for t, (q, c_list, r) in enumerate(grouped):
                if t >= 1:
                    targets.append(r)
                    q_ids.append(q)
                    uids.append(uid)
                    positions.append(t)
                    kc_hist.append(kc_steps_before)
                    num_kcs.append(len(c_list))
                    prev_resp.append(grouped[t - 1][2])
                    prev_q.append(grouped[t - 1][0])
                kc_steps_before += len(c_list)

        self.targets = np.asarray(targets, dtype=np.int64)
        assert len(self.targets) == NUM_TEST_INTERACTIONS, f"Expected {NUM_TEST_INTERACTIONS} test interactions, got {len(self.targets)}"
        self.q_ids = np.asarray(q_ids, dtype=np.int64)
        self.uids = np.asarray(uids, dtype=np.int64)
        _, self.student_index = np.unique(self.uids, return_inverse=True)
        self.student_index = self.student_index.astype(np.int64)
        self.position = np.asarray(positions, dtype=np.int64)
        self.kc_history_len = np.asarray(kc_hist, dtype=np.int64)
        self.num_kcs = np.asarray(num_kcs, dtype=np.int64)
        self.prev_response = np.asarray(prev_resp, dtype=np.int64)
        prev_q_arr = np.asarray(prev_q, dtype=np.int64)

        # Image indicators parsed from raw metadata/questions.json (same regex as PyKT stratification)
        stem_img, sol_img = self._parse_image_flags()
        self.m_img_q = np.array([stem_img.get(int(q), 0.0) for q in self.q_ids], dtype=np.float32)
        self.m_img_s_q = np.array([sol_img.get(int(q), 0.0) for q in self.q_ids], dtype=np.float32)
        self.prev_m_img_s = np.array([sol_img.get(int(q), 0.0) for q in prev_q_arr], dtype=np.float32)

        # Question frequencies in the full train+valid split of the original release
        self.train_q_counts = self._count_train_questions()
        self.train_q_count = np.array([self.train_q_counts.get(int(q), 0) for q in self.q_ids], dtype=np.int64)

        # Cross-check image flags against the MMKT processed item bank modality masks
        self.modality_agreement = self._compare_with_processed_masks()

    def _parse_image_flags(self) -> Tuple[Dict[int, float], Dict[int, float]]:
        q_meta_path = os.path.join(self.raw_root_dir, "metadata", "questions.json")
        img_ref_regex = re.compile(r"((?:question|analysis)_\d+-image_\d+)(?:\.png)?")
        with open(q_meta_path, "r", encoding="utf-8") as f:
            questions_raw = json.load(f)
        stem, sol = {}, {}
        for qid_str, info in questions_raw.items():
            content = info.get("content", "")
            opts = info.get("options", {})
            opt_str = " ".join(str(v) for v in opts.values()) if isinstance(opts, dict) else ""
            analysis = info.get("analysis", "")
            stem[int(qid_str)] = 1.0 if img_ref_regex.findall(content + " " + opt_str) else 0.0
            sol[int(qid_str)] = 1.0 if img_ref_regex.findall(analysis) else 0.0
        return stem, sol

    def _count_train_questions(self) -> Dict[int, int]:
        path = os.path.join(self.raw_root_dir, "kc_level", "train_valid_sequences.csv")
        df = pd.read_csv(path)
        counts: Counter = Counter()
        for _, row in df.iterrows():
            qs = row["questions"].split(",")
            reps = row["is_repeat"].split(",")
            for q, rep in zip(qs, reps):
                if rep == "0" and q != "-1":
                    counts[int(q)] += 1
        return dict(counts)

    def _compare_with_processed_masks(self) -> Dict[str, float]:
        bank_path = os.path.join(self.processed_dir, "mg_static_item_bank.pt")
        if not os.path.exists(bank_path):
            return {}
        bank = torch.load(bank_path, map_location="cpu")
        masks = bank["modality_masks"].numpy()
        proc_stem = (masks[self.q_ids, 0] > 0.5).astype(np.float32)
        return {
            "stem_image_agreement_rate": float(np.mean(proc_stem == self.m_img_q)),
            "raw_stem_image_rate": float(self.m_img_q.mean()),
            "processed_stem_image_rate": float(proc_stem.mean()),
        }

    def strata_masks(self) -> Dict[str, np.ndarray]:
        """
        Boolean masks of every evaluation stratum (identical for all models).

        Returns:
            Ordered mapping stratum_name -> bool mask [N].
        """
        f = self.train_q_count
        pos = self.position
        return {
            "overall": np.ones_like(self.targets, dtype=bool),
            "q_stem_image": self.m_img_q == 1.0,
            "q_text_only": self.m_img_q == 0.0,
            "prev_solution_image": self.prev_m_img_s == 1.0,
            "prev_solution_text_only": self.prev_m_img_s == 0.0,
            "after_incorrect": self.prev_response == 0,
            "after_correct": self.prev_response == 1,
            "freq_unseen_0": f == 0,
            "freq_rare_le10": f <= 10,
            "freq_medium_11_100": (f > 10) & (f <= 100),
            "freq_frequent_gt100": f > 100,
            "single_kc": self.num_kcs == 1,
            "multi_kc": self.num_kcs >= 2,
            "pos_1_49": (pos >= 1) & (pos < 50),
            "pos_50_199": (pos >= 50) & (pos < 200),
            "pos_200_499": (pos >= 200) & (pos < 500),
            "pos_500_plus": pos >= 500,
            "kc_hist_le199": self.kc_history_len <= 199,
            "kc_hist_gt199": self.kc_history_len > 199,
        }

    def save(self, path: str) -> None:
        """Persist the aligned context arrays (npz) for the analysis stage."""
        np.savez_compressed(
            path,
            targets=self.targets, q_ids=self.q_ids, uids=self.uids, student_index=self.student_index,
            position=self.position, kc_history_len=self.kc_history_len, num_kcs=self.num_kcs,
            prev_response=self.prev_response, m_img_q=self.m_img_q, m_img_s_q=self.m_img_s_q,
            prev_m_img_s=self.prev_m_img_s, train_q_count=self.train_q_count,
        )


# ---------------------------------------------------------------------------------------------
# pyKT baselines (original release)
# ---------------------------------------------------------------------------------------------
class PyKTBaselineRunner:
    """
    Runs the question-level anti-leakage inference of `PyKTEvaluator` on kc_level test CSVs
    across XES3G5M or Eedi (unified, public, private).
    """

    def __init__(
        self,
        raw_root_dir: str,
        device: torch.device,
        logger=None,
        dataset_key: str = "xes3g5m",
        test_split: str = "unified",
        test_csv_path: Optional[str] = None,
        processed_dir: Optional[str] = None,
        expected_eval_count: Optional[int] = None,
    ):
        # Imported lazily: pykt_dataset patches torch.cuda tensor aliases at import time
        from src.evaluation.pykt_evaluator import PyKTEvaluator

        self.device = device
        self.logger = logger or get_logger("PyKTBaselineRunner")
        resolved_test_csv = test_csv_path
        if resolved_test_csv is None and dataset_key == "xes3g5m":
            resolved_test_csv = os.path.join(raw_root_dir, "kc_level", "test.csv")
        self.evaluator = PyKTEvaluator(
            raw_data_dir=raw_root_dir,
            test_csv_path=resolved_test_csv,
            processed_dir=processed_dir,
            dataset_key=dataset_key,
            test_split=test_split,
            expected_eval_count=expected_eval_count,
            device=device,
            logger=self.logger,
        )

    def predict(self, ckpt_path: str) -> Dict[str, Any]:
        """
        Load one fold checkpoint and predict all 1,099,197 question-level probabilities.

        Returns:
            Dict with 'preds', 'targets', 'q_ids' (None: implicit raw order), checkpoint meta, timing.
        """
        from src.models.baselines.pykt_models import init_pykt_baseline

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model = init_pykt_baseline(
            model_name=ckpt["model_name"],
            model_config=ckpt["model_config"],
            device=self.device,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        num_params = int(sum(p.numel() for p in model.parameters()))

        synchronize_device(self.device)
        t0 = time.perf_counter()
        preds = self.evaluator.predict_model(model, ckpt["model_name"])
        synchronize_device(self.device)
        elapsed = time.perf_counter() - t0

        return {
            "preds": preds.astype(np.float32),
            "targets": self.evaluator.ground_truth_targets.astype(np.int64),
            "q_ids": self.evaluator.strat_meta["q_ids"].astype(np.int64),
            "inference_seconds": elapsed,
            "num_parameters": num_params,
            "checkpoint_meta": {
                "best_epoch": int(ckpt.get("epoch", -1)),
                "val_auc": float(ckpt.get("val_auc", float("nan"))),
                "val_acc": float(ckpt.get("val_acc", float("nan"))),
                "model_config": ckpt["model_config"],
            },
            "diagnostics": {},
        }


# ---------------------------------------------------------------------------------------------
# MMKT models (MMKT-preprocessed release)
# ---------------------------------------------------------------------------------------------
class MMKTRunner:
    """
    Inference engine for MG-MMKT, MG-MMKTv2 and ACD-MMKTv3 on the MMKT-preprocessed test split.

    The strided test windows are built once (identical to `get_xes3g5m_dataloaders`); for each
    fold, a non-training `XES3G5MDataset` is instantiated with that fold index so that
    fold-specific auxiliary statistics (N_q, mu_concept, ...) match the ones used in training.
    """

    def __init__(
        self,
        family: str,
        data_dir: str,
        device: torch.device,
        batch_size: int = 64,
        test_stride: int = 100,
        logger=None,
    ):
        self.family = family
        self.data_dir = data_dir
        self.device = device
        self.batch_size = batch_size
        self.logger = logger or get_logger(f"MMKTRunner[{family}]")

        self.item_bank = load_static_item_bank_optimized(data_dir, model_type=family)
        raw_test = torch.load(os.path.join(data_dir, "sequences_test.pt"), map_location="cpu")
        self.strided_test, self.test_eval_mask = build_test_strided_windows(raw_test, max_seq_len=200, stride=test_stride)
        n_eval = int(self.test_eval_mask.sum().item())
        assert n_eval == NUM_TEST_INTERACTIONS, f"Strided windows evaluate {n_eval} interactions, expected {NUM_TEST_INTERACTIONS}"
        self.raw_modality_masks = self.item_bank["modality_masks"].numpy()
        self.model: Optional[torch.nn.Module] = None
        self.aux_v3 = None
        if family == "acd_mmkt_v3":
            aux_path = os.path.join(data_dir, "acd_v3_aux_metadata.pt")
            self.aux_v3 = torch.load(aux_path, map_location="cpu")

    def _build_loader(self, fold: int) -> torch.utils.data.DataLoader:
        dataset = XES3G5MDataset(
            static_item_bank=self.item_bank,
            sequences=self.strided_test,
            model_type=self.family,
            fold=fold,
            is_train=False,
            p_drop=0.0,
            eval_mask=self.test_eval_mask,
            data_dir=self.data_dir,
        )
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, collate_fn=collate_xes3g5m,
        )

    def _instantiate(self, model_cfg: Dict[str, Any]) -> torch.nn.Module:
        """Instantiate the architecture with exactly the hyper-parameters stored in the checkpoint."""
        if self.family == "mg_mmkt":
            from src.models.mmkt.mg_mmkt import MGMMKT

            return MGMMKT(
                num_questions=model_cfg.get("num_questions", 7652),
                num_kcs=model_cfg.get("num_kcs", 1175),
                d_raw_txt=model_cfg.get("d_raw_txt", 768),
                d_raw_img=model_cfg.get("d_raw_img", 768),
                d_model=model_cfg.get("d_model", 128),
                num_heads=model_cfg.get("num_heads", 4),
                max_seq_len=model_cfg.get("max_seq_len", 200),
                dropout=model_cfg.get("dropout", 0.20),
                use_solution=model_cfg.get("use_solution", True),
                use_visual=model_cfg.get("use_visual", True),
                use_learnable_imputation=model_cfg.get("use_learnable_imputation", True),
                use_gated_fusion=model_cfg.get("use_gated_fusion", True),
                fusion_mode=model_cfg.get("fusion_mode", "gated"),
                static_item_bank=self.item_bank,
            )
        if self.family == "mg_mmkt_v2":
            from src.models.mmkt.mg_mmkt_v2 import MGMMKTv2

            return MGMMKTv2(
                num_questions=model_cfg.get("num_questions", 7652),
                num_kcs=model_cfg.get("num_kcs", 1175),
                d_raw_txt=model_cfg.get("d_raw_txt", 768),
                d_raw_img=model_cfg.get("d_raw_img", 768),
                d_model=model_cfg.get("d_model", 128),
                d_ffn=model_cfg.get("d_ffn", 256),
                num_heads=model_cfg.get("num_heads", 4),
                max_seq_len=model_cfg.get("max_seq_len", 200),
                p_id_drop=float(model_cfg.get("p_id_drop", 0.20)),
                dropout=float(model_cfg.get("dropout", 0.20)),
                use_lupi=model_cfg.get("use_lupi", True),
                use_dual_pathway_history=model_cfg.get("use_dual_pathway_history", True),
                use_glu=model_cfg.get("use_glu", True),
                use_post_fusion_ffn=model_cfg.get("use_post_fusion_ffn", True),
                use_disentangled_qk=model_cfg.get("use_disentangled_qk", True),
                use_relative_forgetting=model_cfg.get("use_relative_forgetting", True),
                use_structural_decoupling=model_cfg.get("use_structural_decoupling", True),
                use_kc_imputation=model_cfg.get("use_kc_imputation", True),
                use_solution=model_cfg.get("use_solution", True),
                use_visual=model_cfg.get("use_visual", True),
                static_item_bank=self.item_bank,
            )
        if self.family == "acd_mmkt_v3":
            from src.models.mmkt.acd_mmkt_v3 import ACDMMKTv3

            return ACDMMKTv3(
                static_item_bank=self.item_bank,
                d_raw_txt=model_cfg.get("d_raw_txt", 768),
                d_raw_img=model_cfg.get("d_raw_img", 768),
                d_model=model_cfg.get("d_model", 128),
                num_heads=model_cfg.get("num_heads", 4),
                max_seq_len=model_cfg.get("max_seq_len", 200),
                dropout=model_cfg.get("dropout", 0.15),
                p_drop=0.0,
                hgnn_layers=model_cfg.get("hgnn_layers", 2),
                gamma_vis=model_cfg.get("gamma_vis", 1.5),
                delta_a=model_cfg.get("delta_a", 0.1),
                use_multimodal=model_cfg.get("use_multimodal", True),
                use_solution=model_cfg.get("use_solution", True),
                use_visual=model_cfg.get("use_visual", True),
                unanchored_concept=model_cfg.get("unanchored_concept", False),
                polluted_experience_attn=model_cfg.get("polluted_experience_attn", False),
                use_deep_irt=model_cfg.get("use_deep_irt", True),
                use_hypergraph=model_cfg.get("use_hypergraph", True),
                use_gram_schmidt=model_cfg.get("use_gram_schmidt", True),
                post_diag_mode=model_cfg.get("post_diag_mode", "none"),
            )
        raise ValueError(f"Unknown MMKT family: {self.family}")

    def predict(self, ckpt_path: str, fold: int) -> Dict[str, Any]:
        """
        Load one fold checkpoint and score all 1,099,197 evaluated test interactions.

        Returns:
            Dict with 'preds', 'targets', 'q_ids', checkpoint meta, timing and family diagnostics.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_cfg = dict(ckpt.get("model_cfg") or {})
        ckpt_fold = ckpt.get("val_fold", fold)
        assert int(ckpt_fold) == int(fold), f"Checkpoint fold {ckpt_fold} != requested fold {fold}"

        # The architecture (and its large non-persistent item-bank buffers) is built once per family
        if self.model is None:
            self.model = self._instantiate(model_cfg).to(self.device)
        missing_unexpected = self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.eval()
        num_params = int(sum(p.numel() for p in self.model.parameters()))

        loader = self._build_loader(fold)
        mu_concept_prior = None
        if self.family == "acd_mmkt_v3":
            mu_concept_prior = self.aux_v3["mu_concept_folds"][fold].to(self.device)

        diag = _DiagnosticsAccumulator(self.family)
        preds_chunks, target_chunks, q_chunks = [], [], []

        synchronize_device(self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            if self.family == "acd_mmkt_v3":
                self.model.precompute_item_bank_cache(mu_concept_prior)
            for batch in loader:
                batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                eval_mask = batch["eval_mask"].bool()

                if self.family == "mg_mmkt":
                    out = self.model(batch)
                    probs = out["preds"]
                elif self.family == "mg_mmkt_v2":
                    out = self.model(
                        q_seq=batch["q_seq"],
                        r_seq=batch["r_seq"],
                        seq_mask=batch["seq_mask"],
                        modality_masks=batch["modality_masks"],
                        raw_modality_masks=batch["raw_modality_masks"],
                    )
                    probs = out["probs"]
                else:
                    out = self.model(batch, mu_concept_prior=mu_concept_prior, use_cache=True)
                    probs = out["probs"]

                preds_chunks.append(probs[eval_mask].float().cpu().numpy())
                target_chunks.append(batch["r_seq"][eval_mask].cpu().numpy())
                q_chunks.append(batch["q_seq"][eval_mask].cpu().numpy())
                diag.update(out, batch, eval_mask)
        synchronize_device(self.device)
        elapsed = time.perf_counter() - t0

        diagnostics = diag.finalize()
        if self.family == "acd_mmkt_v3":
            props = self.model.cached_item_properties
            item_arrays = {}
            for key in ("mu_q", "tilde_mu", "mu_concept", "delta_mu_txt", "delta_mu_vis",
                        "delta_mu_stem", "delta_mu_sol", "a_q", "bar_lambda_q"):
                if key in props and isinstance(props[key], torch.Tensor):
                    arr = props[key].detach().float().cpu().numpy().reshape(-1)
                    if arr.shape[0] == 7652:
                        item_arrays[key] = arr
            diagnostics["item_parameters"] = item_arrays
            self.model.clear_item_bank_cache()

        return {
            "preds": np.concatenate(preds_chunks).astype(np.float32),
            "targets": np.concatenate(target_chunks).astype(np.int64),
            "q_ids": np.concatenate(q_chunks).astype(np.int64),
            "inference_seconds": elapsed,
            "num_parameters": num_params,
            "checkpoint_meta": {
                "best_epoch": int(ckpt.get("epoch", -1)),
                "val_auc": float(ckpt.get("val_auc", ckpt.get("best_val_auc", float("nan")))),
                "val_acc": float(ckpt.get("val_acc", (ckpt.get("val_metrics") or {}).get("acc", float("nan")))),
                "model_config": model_cfg,
                "load_state_dict": str(missing_unexpected),
            },
            "diagnostics": diagnostics,
        }


class _DiagnosticsAccumulator:
    """
    Streaming accumulator of family-specific interpretability statistics over evaluated steps.

    - MG-MMKT / MG-MMKTv2: mean question-gate weights alpha_q = (ID, TXT, IMG) split by stem-image
      presence, and mean solution-gate weights beta_s = (TXT, IMG) of step t-1 split by whether the
      previous solution has an image.
    - ACD-MMKTv3: 2PL-IRT decision faithfulness (p >= 0.5  <=>  theta >= tilde_mu) and theta stats.
    """

    def __init__(self, family: str):
        self.family = family
        self.sums: Dict[str, np.ndarray] = {}
        self.counts: Dict[str, float] = {}
        self.faithful = 0
        self.total = 0
        self.theta_sum = 0.0
        self.theta_sq_sum = 0.0

    def _add(self, key: str, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        s = values.double().sum(dim=0).cpu().numpy()
        self.sums[key] = self.sums.get(key, 0) + s
        self.counts[key] = self.counts.get(key, 0) + values.shape[0]

    def update(self, out: Dict[str, Any], batch: Dict[str, torch.Tensor], eval_mask: torch.Tensor) -> None:
        if self.family in ("mg_mmkt", "mg_mmkt_v2"):
            raw = batch["raw_modality_masks"]
            alpha = out["alpha_q"]
            beta = out["beta_s"]
            stem_img = raw[:, :, 0] > 0.5
            self._add("alpha_q|stem_image", alpha[eval_mask & stem_img])
            self._add("alpha_q|text_only", alpha[eval_mask & ~stem_img])
            # Solution gate of step t-1 is what conditions the prediction at step t
            prev_beta = torch.zeros_like(beta)
            prev_beta[:, 1:] = beta[:, :-1]
            prev_sol_img = torch.zeros_like(stem_img)
            prev_sol_img[:, 1:] = raw[:, :-1, 2] > 0.5
            self._add("beta_s_prev|prev_solution_image", prev_beta[eval_mask & prev_sol_img])
            self._add("beta_s_prev|prev_solution_text_only", prev_beta[eval_mask & ~prev_sol_img])
        elif self.family == "acd_mmkt_v3":
            p = out["probs"][eval_mask].float()
            th = out["theta"][eval_mask].float()
            mu = out["tilde_mu"][eval_mask].float()
            self.faithful += int(((p >= 0.5) == (th >= mu)).sum().item())
            self.total += int(p.numel())
            self.theta_sum += float(th.double().sum().item())
            self.theta_sq_sum += float((th.double() ** 2).sum().item())

    def finalize(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for k, s in self.sums.items():
            result[k] = {"mean": (s / max(self.counts[k], 1)).tolist(), "count": int(self.counts[k])}
        if self.family == "acd_mmkt_v3" and self.total > 0:
            mean_t = self.theta_sum / self.total
            result["decision_faithfulness_pct"] = 100.0 * self.faithful / self.total
            result["theta_mean"] = mean_t
            result["theta_std"] = float(np.sqrt(max(self.theta_sq_sum / self.total - mean_t ** 2, 0.0)))
        return result
