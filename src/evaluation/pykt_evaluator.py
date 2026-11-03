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
from src.utils.logger import get_logger


class PyKTEvaluator:
    """
    Evaluator performing unrolled, anti-leakage question-level inference
    and multimodal comparative analysis for pyKT models.
    """

    def __init__(
        self,
        raw_data_dir: str = "data/raw/XES3G5M",
        test_csv_path: Optional[str] = None,
        processed_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
        logger: Optional[Any] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            raw_data_dir: Directory containing raw XES3G5M files (kc_level/, metadata/).
            test_csv_path: Optional path to kc_level/test.csv.
            processed_dir: Optional legacy fallback directory for processed MMKT tensors.
            device: Target evaluation device.
            logger: Custom logger instance.
        """
        self.raw_data_dir = raw_data_dir
        self.test_csv_path = test_csv_path or os.path.join(raw_data_dir, "kc_level", "test.csv")
        self.processed_dir = processed_dir
        self.logger = logger or get_logger("PyKTEvaluator")

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
        self.logger.info("Loading test student sequences and stratification metadata from raw XES3G5M...")
        self.test_students = load_test_grouped_data(self.test_csv_path)
        self.strat_meta = load_stratification_metadata(
            raw_data_dir=self.raw_data_dir,
            test_grouped_students=self.test_students,
            processed_dir=self.processed_dir,
        )
        self.ground_truth_targets = self.strat_meta["targets"]
        assert len(self.ground_truth_targets) == 1099197, (
            f"Expected exactly 1,099,197 targets, got {len(self.ground_truth_targets)}"
        )
        self.logger.info(f"Loaded {len(self.test_students)} students ({len(self.ground_truth_targets)} interactions).")

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

        assert len(preds_arr) == 1099197, (
            f"Prediction length mismatch: expected 1,099,197, got {len(preds_arr)}"
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
            Tuple of (predictions array [1099197], evaluation results dict).
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        model_name = ckpt["model_name"]
        model_config = ckpt["model_config"]
        fold = ckpt.get("fold", 0)

        model = init_pykt_baseline(
            model_name=model_name,
            model_config=model_config,
            device=self.device,
        )
        model.load_state_dict(ckpt["model_state_dict"])

        preds = self.predict_model(model, model_name)
        metrics = self.compute_all_metrics(preds)
        metrics["model_name"] = model_name
        metrics["fold"] = fold
        metrics["val_auc"] = ckpt.get("val_auc", 0.0)

        return preds, metrics

    def compute_all_metrics(
        self,
        preds: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compute global, multimodal, remedial, and cold-start stratified metrics.

        Args:
            preds: Array of predicted probabilities of length 1,099,197.

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
