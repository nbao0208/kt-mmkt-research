"""
Question-Level Late Fusion Evaluator and Multimodal Benchmark Suite for pyKT Baselines.

Adheres strictly to the pyKT question-level evaluation protocol on XES3G5M:
1. Anti-leakage inference on multi-concept questions (is_repeat == 1).
2. Late Mean Fusion: p_hat(u, t) = mean_{c in Q(u, t)} p_hat(u, t, c).
3. 1-to-1 alignment with the exactly 1,099,197 evaluated test interactions of MMKT models.
4. Comprehensive global, 5-fold ensemble, multimodal stratified, and paired statistical testing.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score

from src.models.baselines.pykt_models import init_pykt_baseline, sync_pykt_device
from src.data.pykt_dataset import load_test_grouped_data, load_stratification_metadata
from src.evaluation.metrics import calculate_metrics, multimodal_stratified_evaluation
from src.evaluation.benchmark_statistics import compute_binary_metrics, student_macro_auc
from src.utils.logger import get_logger


class PyKTEvaluator:
    """
    Evaluator performing unrolled, anti-leakage question-level inference
    and multimodal comparative analysis for pyKT models across XES3G5M and Eedi.
    """

    def __init__(
        self,
        raw_data_dir: str = "data/raw/XES3G5M",
        test_csv_path: Optional[str] = None,
        processed_dir: Optional[str] = None,
        dataset_key: str = "xes3g5m",
        test_split: str = "unified",
        expected_eval_count: Optional[int] = None,
        test_stride: int = 100,
        train_valid_csv_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        logger: Optional[Any] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            raw_data_dir: Directory containing raw kc_level/metadata (XES3G5M) or pykt_kc_level (Eedi).
            test_csv_path: Optional explicit path to kc_level test CSV.
            processed_dir: Directory containing processed MMKT tensors (required for EediTestContext).
            dataset_key: Canonical dataset identifier ('xes3g5m' or 'eedi').
            test_split: Eedi evaluation split ('unified', 'public', or 'private').
            expected_eval_count: Expected number of evaluated question-level interactions.
            test_stride: Context-preserving stride used by EediTestContext (default: 100).
            train_valid_csv_path: Optional explicit path to train_valid_sequences.csv.
            device: Target evaluation device.
            logger: Custom logger instance.
        """
        self.raw_data_dir = raw_data_dir
        self.dataset_key = dataset_key.strip().lower()
        self.test_split = test_split.strip().lower()
        self.processed_dir = processed_dir or ("data/processed/Eedi" if self.dataset_key == "eedi" else None)
        self.test_stride = test_stride
        self.logger = logger or get_logger("PyKTEvaluator")

        if self.dataset_key == "eedi":
            from src.evaluation.eedi_test_context import EXPECTED_EEDI_EVAL_COUNTS

            self.expected_eval_count = (
                int(expected_eval_count)
                if expected_eval_count is not None
                else int(EXPECTED_EEDI_EVAL_COUNTS[self.test_split])
            )
            default_test_csv = os.path.join(raw_data_dir, f"test_{self.test_split}.csv")
            if not os.path.exists(default_test_csv):
                default_test_csv = os.path.join(
                    self.processed_dir, "pykt_kc_level", f"test_{self.test_split}.csv"
                )
            self.test_csv_path = test_csv_path or default_test_csv
        else:
            self.expected_eval_count = int(expected_eval_count) if expected_eval_count is not None else 1099197
            self.test_csv_path = test_csv_path or os.path.join(raw_data_dir, "kc_level", "test.csv")

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        sync_pykt_device(self.device)

        # Preload student grouped test sequences and stratification metadata
        self.logger.info(
            f"Loading test student sequences and stratification metadata for {self.dataset_key.upper()} "
            f"(split={self.test_split}, path={self.test_csv_path})..."
        )
        self.test_students = load_test_grouped_data(
            self.test_csv_path,
            expected_eval_count=self.expected_eval_count,
        )
        self.strat_meta = load_stratification_metadata(
            raw_data_dir=self.raw_data_dir,
            test_grouped_students=self.test_students,
            processed_dir=self.processed_dir,
            dataset_key=self.dataset_key,
            test_split=self.test_split,
            test_csv_path=self.test_csv_path,
            expected_eval_count=self.expected_eval_count,
            test_stride=self.test_stride,
            train_valid_csv_path=train_valid_csv_path,
        )
        self.eedi_ctx = self.strat_meta.get("eedi_context")
        self.ground_truth_targets = self.strat_meta["targets"]
        assert len(self.ground_truth_targets) == self.expected_eval_count, (
            f"Expected exactly {self.expected_eval_count:,} targets, got {len(self.ground_truth_targets):,}"
        )
        self.logger.info(
            f"Loaded {len(self.test_students)} students ({len(self.ground_truth_targets):,} interactions)."
        )

    def predict_dkt_all(self, model: nn.Module, batch_size: int = 1024) -> np.ndarray:
        """
        Predict question probabilities for all 3,613 test students using DKT.
        
        Adheres to pyKT's maxlen=200 context window specification (max_hist_len = 199):
        1. For questions where historical interaction steps <= 199:
           The hidden state trajectory is extracted from the causal LSTM unroll of the prefix.
        2. For questions where historical interaction steps > 199:
           To prevent out-of-distribution recurrent hidden state drift, the context window
           is strictly capped to the most recent 199 historical steps (c[h-199:h], r[h-199:h])
           and evaluated via batched inference.
        3. Anti-leakage Question-Level Late Mean Fusion:
           p_hat(u, t) = mean_{c in Q(u, t)} p_hat(u, t, c).
        """
        all_preds: List[float] = []

        long_q_sub_c: List[List[int]] = []
        long_q_sub_r: List[List[int]] = []
        long_q_targets: List[List[int]] = []
        long_q_positions: List[int] = []

        pred_index = 0

        for student in self.test_students:
            all_q = student["all_questions"]
            if len(all_q) <= 1:
                continue

            c_list, r_list, q_end_indices = [], [], []
            for q in all_q:
                for c in q["concepts"]:
                    c_list.append(c)
                    r_list.append(q["response"])
                q_end_indices.append(len(c_list) - 1)

            prefix_len = min(len(c_list), 199)
            c_prefix = torch.tensor([c_list[:prefix_len]], dtype=torch.long, device=self.device)
            r_prefix = torch.tensor([r_list[:prefix_len]], dtype=torch.long, device=self.device)

            with torch.no_grad():
                y_prefix = model(c_prefix, r_prefix)[0].cpu().numpy()  # [prefix_len, num_c]

            for t in range(1, len(all_q)):
                h_step = q_end_indices[t - 1]
                cur_concepts = all_q[t]["concepts"]

                if h_step < 199:
                    probs_at_step = y_prefix[h_step]
                    all_preds.append(float(np.mean(probs_at_step[cur_concepts])))
                else:
                    all_preds.append(0.0)
                    h_end = h_step + 1
                    long_q_sub_c.append(c_list[h_end - 199:h_end])
                    long_q_sub_r.append(r_list[h_end - 199:h_end])
                    long_q_targets.append(cur_concepts)
                    long_q_positions.append(pred_index)

                pred_index += 1

        num_long = len(long_q_positions)
        if num_long > 0:
            c_batch = torch.zeros((batch_size, 199), dtype=torch.long, device=self.device)
            r_batch = torch.zeros((batch_size, 199), dtype=torch.long, device=self.device)

            for b_start in range(0, num_long, batch_size):
                b_end = min(b_start + batch_size, num_long)
                cur_b = b_end - b_start

                for i in range(cur_b):
                    idx = b_start + i
                    c_batch[i] = torch.tensor(long_q_sub_c[idx], device=self.device)
                    r_batch[i] = torch.tensor(long_q_sub_r[idx], device=self.device)

                with torch.no_grad():
                    y_last = model(c_batch[:cur_b], r_batch[:cur_b])[:, -1, :].cpu().numpy()

                for i in range(cur_b):
                    idx = b_start + i
                    target_cs = long_q_targets[idx]
                    pos = long_q_positions[idx]
                    all_preds[pos] = float(np.mean(y_last[i, target_cs]))

        return np.asarray(all_preds, dtype=np.float32)

    def predict_dkvmn_all(self, model: nn.Module, batch_size: int = 512) -> np.ndarray:
        """
        Predict question probabilities for all 3,613 test students using DKVMN.
        
        Adheres to pyKT's maxlen=200 context window specification (max_hist_len = 199):
        1. For questions where historical interaction steps <= 199:
           Value memory updates sequentially from Mv0 over the prefix, freezing Mv at
           the end of question t-1 for evaluating question t concepts.
        2. For questions where historical interaction steps > 199:
           Context is strictly capped to the most recent 199 historical steps (c[h-199:h], r[h-199:h])
           to prevent memory state drift beyond the training sequence length.
        3. Anti-leakage Question-Level Late Mean Fusion:
           p_hat(u, t) = mean_{c in Q(u, t)} p_hat(u, t, c).
        """
        all_preds: List[float] = []

        long_q_sub_c: List[List[int]] = []
        long_q_sub_r: List[List[int]] = []
        long_q_targets: List[List[int]] = []
        long_q_positions: List[int] = []

        pred_index = 0

        for student in self.test_students:
            all_q = student["all_questions"]
            if len(all_q) <= 1:
                continue

            c_list, r_list, q_end_indices = [], [], []
            for q in all_q:
                for c in q["concepts"]:
                    c_list.append(c)
                    r_list.append(q["response"])
                q_end_indices.append(len(c_list) - 1)

            prefix_len = min(len(c_list), 199)
            c_prefix = torch.tensor([c_list[:prefix_len]], dtype=torch.long, device=self.device)
            r_prefix = torch.tensor([r_list[:prefix_len]], dtype=torch.long, device=self.device)

            with torch.no_grad():
                k = model.k_emb_layer(c_prefix)
                x = c_prefix + model.num_c * r_prefix
                v = model.v_emb_layer(x)
                w = torch.softmax(torch.matmul(k, model.Mk.T), dim=-1)
                e = torch.sigmoid(model.e_layer(v))
                a = torch.tanh(model.a_layer(v))

                Mvt = model.Mv0.unsqueeze(0)
                Mv = [Mvt]
                for et, at, wt in zip(e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)):
                    Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + (wt.unsqueeze(-1) * at.unsqueeze(1))
                    Mv.append(Mvt)
                Mv_seq = torch.stack(Mv, dim=1)[0]  # [prefix_len + 1, size_m, dim_s]

            # Collect queries for short-history questions
            q_lengths = []
            mv_indices = []
            flat_concepts = []
            short_q_slots = []

            for t in range(1, len(all_q)):
                h_step = q_end_indices[t - 1]
                cur_cs = all_q[t]["concepts"]

                if h_step < 199:
                    all_preds.append(0.0)  # placeholder for short
                    short_q_slots.append(pred_index)
                    mv_idx = h_step + 1
                    q_lengths.append(len(cur_cs))
                    for c in cur_cs:
                        mv_indices.append(mv_idx)
                        flat_concepts.append(c)
                else:
                    all_preds.append(0.0)  # placeholder for long
                    h_end = h_step + 1
                    long_q_sub_c.append(c_list[h_end - 199:h_end])
                    long_q_sub_r.append(r_list[h_end - 199:h_end])
                    long_q_targets.append(cur_cs)
                    long_q_positions.append(pred_index)

                pred_index += 1

            if len(flat_concepts) > 0:
                with torch.no_grad():
                    mv_sub = Mv_seq[mv_indices]
                    c_sub = torch.tensor(flat_concepts, dtype=torch.long, device=self.device).unsqueeze(1)
                    k_sub = model.k_emb_layer(c_sub)
                    w_sub = torch.softmax(torch.matmul(k_sub, model.Mk.T), dim=-1)
                    read_vec = (w_sub.unsqueeze(-1) * mv_sub.unsqueeze(1)).sum(-2)
                    f_sub = torch.tanh(model.f_layer(torch.cat([read_vec, k_sub], dim=-1)))
                    p_sub = torch.sigmoid(model.p_layer(f_sub)).squeeze(-1).cpu().numpy()

                if p_sub.ndim == 0:
                    p_sub = np.array([p_sub.item()])

                cur_idx = 0
                for s_i, m_len in enumerate(q_lengths):
                    pos = short_q_slots[s_i]
                    all_preds[pos] = float(np.mean(p_sub[cur_idx:cur_idx + m_len]))
                    cur_idx += m_len

        # Batched window inference for questions with history > 199
        num_long = len(long_q_positions)
        if num_long > 0:
            c_batch = torch.zeros((batch_size, 199), dtype=torch.long, device=self.device)
            r_batch = torch.zeros((batch_size, 199), dtype=torch.long, device=self.device)

            for b_start in range(0, num_long, batch_size):
                b_end = min(b_start + batch_size, num_long)
                cur_b = b_end - b_start

                for i in range(cur_b):
                    idx = b_start + i
                    c_batch[i] = torch.tensor(long_q_sub_c[idx], device=self.device)
                    r_batch[i] = torch.tensor(long_q_sub_r[idx], device=self.device)

                with torch.no_grad():
                    c_sub_b = c_batch[:cur_b]
                    r_sub_b = r_batch[:cur_b]
                    k = model.k_emb_layer(c_sub_b)
                    x = c_sub_b + model.num_c * r_sub_b
                    v = model.v_emb_layer(x)
                    w = torch.softmax(torch.matmul(k, model.Mk.T), dim=-1)
                    e = torch.sigmoid(model.e_layer(v))
                    a = torch.tanh(model.a_layer(v))

                    Mvt = model.Mv0.unsqueeze(0).repeat(cur_b, 1, 1)
                    for et, at, wt in zip(e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)):
                        Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + (wt.unsqueeze(-1) * at.unsqueeze(1))

                    # Read target concepts for each sample in batch
                    for i in range(cur_b):
                        idx = b_start + i
                        cur_cs = long_q_targets[idx]
                        pos = long_q_positions[idx]
                        c_q = torch.tensor(cur_cs, dtype=torch.long, device=self.device).unsqueeze(1)
                        k_q = model.k_emb_layer(c_q)
                        w_q = torch.softmax(torch.matmul(k_q, model.Mk.T), dim=-1)
                        mv_i = Mvt[i].unsqueeze(0).repeat(len(cur_cs), 1, 1)
                        read_vec = (w_q.unsqueeze(-1) * mv_i.unsqueeze(1)).sum(-2)
                        f_q = torch.tanh(model.f_layer(torch.cat([read_vec, k_q], dim=-1)))
                        p_q = torch.sigmoid(model.p_layer(f_q)).squeeze(-1).cpu().numpy()
                        all_preds[pos] = float(np.mean(p_q))

        return np.asarray(all_preds, dtype=np.float32)

    def predict_sakt_all(self, model: nn.Module, batch_size: int = 512) -> np.ndarray:
        """
        Predict question probabilities for all 3,613 test students using SAKT in batches.
        
        Strict mathematical alignment with pyKT's SAKT:
        1. Right-Padding: sub_hist_c and sub_hist_r (length h_len in [1, 199]) occupy positions 0 .. h_len - 1.
           Because of ut_mask (upper-triangular causal attention mask), positions 0 .. h_len - 1 NEVER attend
           to positions >= h_len, completely eliminating left-padded false interaction (c=0, r=0) leakage.
        2. Shifted Query Alignment:
           At history positions 0 .. h_len - 2: qry_mat[i, :h_len - 1] = c_mat[i, 1:h_len] (shifted concept query cshft).
           At prediction position h_len - 1: qry_mat[i, h_len - 1] = target_concept.
           This correctly aligns with multi-layer SAKT (num_en = 2) attention blocks.
        3. Extraction at h_len - 1:
           Prediction is gathered strictly at index h_len - 1: preds[i, h_len - 1].
        4. Late Mean Fusion:
           p_hat(u, t) = mean_{c in Q(u, t)} p_hat(u, t, c).
        """
        all_preds: List[float] = []

        q_kc_counts: List[int] = []
        kc_preds_list: List[float] = []

        c_mat = torch.zeros((batch_size, 199), dtype=torch.long, device=self.device)
        r_mat = torch.zeros((batch_size, 199), dtype=torch.long, device=self.device)
        qry_mat = torch.zeros((batch_size, 199), dtype=torch.long, device=self.device)
        h_lens_tensor = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        buf_hist_c: List[List[int]] = []
        buf_hist_r: List[List[int]] = []
        buf_qry: List[int] = []

        def flush_batch():
            cur_b = len(buf_qry)
            if cur_b == 0:
                return

            c_mat[:cur_b].zero_()
            r_mat[:cur_b].zero_()
            qry_mat[:cur_b].zero_()

            for i in range(cur_b):
                h_c = buf_hist_c[i]
                h_r = buf_hist_r[i]
                h_l = len(h_c)
                h_lens_tensor[i] = h_l
                t_c = torch.tensor(h_c, dtype=torch.long, device=self.device)
                t_r = torch.tensor(h_r, dtype=torch.long, device=self.device)
                c_mat[i, :h_l] = t_c
                r_mat[i, :h_l] = t_r
                if h_l > 1:
                    qry_mat[i, :h_l - 1] = t_c[1:]
                qry_mat[i, h_l - 1] = buf_qry[i]

            with torch.no_grad():
                out = model(c_mat[:cur_b], r_mat[:cur_b], qry_mat[:cur_b])
                row_indices = torch.arange(cur_b, device=self.device)
                col_indices = h_lens_tensor[:cur_b] - 1
                batch_preds = out[row_indices, col_indices].cpu().numpy()
                kc_preds_list.extend(batch_preds.tolist())

            buf_hist_c.clear()
            buf_hist_r.clear()
            buf_qry.clear()

        for student in self.test_students:
            all_q = student["all_questions"]
            if len(all_q) <= 1:
                continue

            c_list, r_list, q_end_indices = [], [], []
            for q in all_q:
                for c in q["concepts"]:
                    c_list.append(c)
                    r_list.append(q["response"])
                q_end_indices.append(len(c_list) - 1)

            for t in range(1, len(all_q)):
                h_end = q_end_indices[t - 1] + 1
                h_start = max(0, h_end - 199)
                sub_c = c_list[h_start:h_end]
                sub_r = r_list[h_start:h_end]
                cur_cs = all_q[t]["concepts"]
                q_kc_counts.append(len(cur_cs))

                for c in cur_cs:
                    buf_hist_c.append(sub_c)
                    buf_hist_r.append(sub_r)
                    buf_qry.append(c)
                    if len(buf_qry) >= batch_size:
                        flush_batch()

        flush_batch()

        cur_idx = 0
        all_kc_preds = np.asarray(kc_preds_list, dtype=np.float32)
        for m_len in q_kc_counts:
            all_preds.append(float(np.mean(all_kc_preds[cur_idx:cur_idx + m_len])))
            cur_idx += m_len

        return np.asarray(all_preds, dtype=np.float32)

    def predict_model(
        self,
        model: nn.Module,
        model_name: str,
    ) -> np.ndarray:
        """
        Generate aligned question-level predictions across all 3,613 test students.

        Args:
            model: PyTorch model loaded with trained weights.
            model_name: Identifier ('dkt', 'dkvmn', 'sakt').

        Returns:
            np.ndarray of shape [1,099,197] containing predicted probabilities.
        """
        model.eval()
        model_name = model_name.lower().strip()
        self.logger.info(f"Running anti-leakage question-level inference for {model_name.upper()}...")

        if model_name in ["dkt", "dkt+"]:
            preds_arr = self.predict_dkt_all(model)
        elif model_name in ["dkvmn", "deep_irt"]:
            preds_arr = self.predict_dkvmn_all(model)
        elif model_name in ["sakt"]:
            preds_arr = self.predict_sakt_all(model)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        assert len(preds_arr) == self.expected_eval_count, (
            f"Prediction length mismatch: expected {self.expected_eval_count:,}, got {len(preds_arr):,}"
        )
        return preds_arr

    def evaluate_checkpoint(
        self,
        checkpoint_path: str,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load checkpoint, run question-level inference, and compute stratified metrics.

        Args:
            checkpoint_path: Path to best_model.pt checkpoint file.

        Returns:
            Tuple of (predictions array [N_eval], evaluation results dict).
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model_name = ckpt["model_name"]
        model_config = ckpt["model_config"]
        fold = ckpt.get("fold", 0)

        model = init_pykt_baseline(
            model_name=model_name,
            model_config=model_config,
            device=self.device,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        num_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

        preds = self.predict_model(model, model_name)
        metrics = self.compute_all_metrics(preds)
        metrics["model_name"] = model_name
        metrics["fold"] = fold
        metrics["val_auc"] = ckpt.get("val_auc", 0.0)
        metrics["num_parameters"] = num_params

        return preds, metrics

    def compute_all_metrics(
        self,
        preds: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compute global, multimodal, remedial, cold-start, and (on Eedi) comprehensive
        calibration + pedagogical stratified metrics.

        Args:
            preds: Array of predicted probabilities aligned with ground_truth_targets.

        Returns:
            Nested dictionary of all evaluation metrics.
        """
        targets = self.strat_meta["targets"]
        q_ids = self.strat_meta["q_ids"]
        m_img_q = self.strat_meta["m_img_q"]
        prev_m_img_s = self.strat_meta["prev_m_img_s"]
        train_counts = self.strat_meta["train_question_counts"]

        # 1. Base multimodal stratification
        strat_results = multimodal_stratified_evaluation(
            preds=preds,
            targets=targets,
            q_ids=q_ids,
            m_img_q=m_img_q,
            prev_m_img_s=prev_m_img_s,
            train_question_counts=train_counts,
            rare_threshold=10,
        )

        # 2. Remedial learning stratification (response at tau-1 == 0 vs == 1)
        remedial_mask = self.strat_meta["remedial_mask"]

        strat_results["remedial_after_incorrect"] = calculate_metrics(
            preds[remedial_mask], targets[remedial_mask]
        )
        strat_results["normal_after_correct"] = calculate_metrics(
            preds[~remedial_mask], targets[~remedial_mask]
        )

        # 3. Frequency tiers: Rare (<=10), Medium (11..100), Frequent (>100)
        q_freqs = np.array([train_counts.get(int(q), 0) for q in q_ids])
        mask_rare = (q_freqs <= 10)
        mask_med = (q_freqs > 10) & (q_freqs <= 100)
        mask_freq = (q_freqs > 100)

        strat_results["freq_rare_le10"] = calculate_metrics(preds[mask_rare], targets[mask_rare])
        strat_results["freq_medium_11to100"] = calculate_metrics(preds[mask_med], targets[mask_med])
        strat_results["freq_frequent_gt100"] = calculate_metrics(preds[mask_freq], targets[mask_freq])

        # 4. Extended binary calibration + Eedi pedagogical strata (when EediTestContext is attached)
        if getattr(self, "eedi_ctx", None) is not None:
            ext_global = compute_binary_metrics(preds, targets)
            macro = student_macro_auc(preds, targets, self.eedi_ctx.student_index)
            ext_global["student_macro_auc"] = macro["macro_auc"]
            ext_global["student_macro_auc_std"] = macro["std_auc"]
            strat_results["overall_extended"] = ext_global

            eedi_strata: Dict[str, Dict[str, Any]] = {}
            for s_name, s_mask in self.eedi_ctx.strata_masks().items():
                if int(np.sum(s_mask)) == 0:
                    eedi_strata[s_name] = {
                        "auc": None, "acc": None, "rmse": None, "log_loss": None,
                        "brier": None, "ece": None, "pos_rate": None,
                        "mean_pred": None, "num_samples": 0,
                    }
                else:
                    eedi_strata[s_name] = compute_binary_metrics(preds[s_mask], targets[s_mask])
            strat_results["eedi_strata"] = eedi_strata

        return strat_results

    @staticmethod
    def compute_paired_statistical_tests(
        baseline_fold_aucs: List[float],
        proposed_fold_aucs: List[float],
    ) -> Dict[str, float]:
        """
        Execute paired Student's t-test and Wilcoxon signed-rank test across 5 folds.

        Args:
            baseline_fold_aucs: List of 5 fold AUC values for baseline model.
            proposed_fold_aucs: List of 5 fold AUC values for proposed model.

        Returns:
            Dictionary with t_statistic, p_value_ttest, wilcoxon_stat, and p_value_wilcoxon.
        """
        assert len(baseline_fold_aucs) == len(proposed_fold_aucs) == 5, (
            "Both models must have exactly 5 fold AUC values."
        )
        diff = np.array(proposed_fold_aucs) - np.array(baseline_fold_aucs)

        t_res = stats.ttest_rel(proposed_fold_aucs, baseline_fold_aucs)

        # Wilcoxon signed-rank test
        try:
            w_res = stats.wilcoxon(proposed_fold_aucs, baseline_fold_aucs)
            w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
        except Exception:
            w_stat, w_p = 0.0, 1.0

        return {
            "mean_diff": float(np.mean(diff)),
            "std_diff": float(np.std(diff)),
            "t_statistic": float(t_res.statistic),
            "p_value_ttest": float(t_res.pvalue),
            "wilcoxon_stat": w_stat,
            "p_value_wilcoxon": w_p,
        }
