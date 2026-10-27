"""
First-Order Explainability Extractor for MG-MMKT.

Implements Section 3 (First-Order Explainability) of docs/models/MG-MMKT.md:
1. Modality Attribution:
   - Question modality contribution weights alpha_t^q = (alpha_id, alpha_txt, alpha_img).
   - Solution modality contribution weights beta_tau^s = (beta_txt, beta_img).
   - Statistical verification of modality attribution on visual vs text-only questions.
2. Historical Interaction Evidence:
   - Top-K historical interaction retrieval based on SAKT causal attention weights gamma_{t, tau}.
   - Precise chronological index alignment accounting for right-shifted y_0 BOS token.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch


class FirstOrderExplainer:
    """
    Extracts and formats modality attribution and historical interaction evidence from MG-MMKT.
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize FirstOrderExplainer.

        Args:
            metadata: Optional dictionary with question texts, images, and KC names for rich reporting.
        """
        self.metadata = metadata or {}

    def analyze_modality_attribution(
        self,
        all_alpha_q: np.ndarray,
        all_m_img_q: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compute aggregate statistics for question modality attribution weights,
        comparing questions with image vs text-only questions, and perform statistical testing.

        Args:
            all_alpha_q: Modality weights array [N, num_branches] (either 3: id, txt, img or 2: id, txt).
            all_m_img_q: Binary flags indicating image presence [N].

        Returns:
            Dictionary containing mean, std, and statistical significance tests across question groups.
        """
        num_branches = all_alpha_q.shape[-1]
        has_visual = (num_branches >= 3)

        img_mask = (all_m_img_q == 1.0)
        txt_mask = (all_m_img_q == 0.0)

        alpha_with_img = all_alpha_q[img_mask] if np.any(img_mask) else np.zeros((0, num_branches))
        alpha_txt_only = all_alpha_q[txt_mask] if np.any(txt_mask) else np.zeros((0, num_branches))

        report: Dict[str, Any] = {
            "num_questions_with_image": int(img_mask.sum()),
            "num_questions_text_only": int(txt_mask.sum()),
            "has_visual_branch": has_visual,
        }

        if has_visual:
            mean_img_with = float(np.mean(alpha_with_img[:, 2])) if len(alpha_with_img) > 0 else 0.0
            std_img_with = float(np.std(alpha_with_img[:, 2])) if len(alpha_with_img) > 0 else 0.0
            mean_img_without = float(np.mean(alpha_txt_only[:, 2])) if len(alpha_txt_only) > 0 else 0.0
            std_img_without = float(np.std(alpha_txt_only[:, 2])) if len(alpha_txt_only) > 0 else 0.0

            report["with_image"] = {
                "mean_alpha_id": float(np.mean(alpha_with_img[:, 0])) if len(alpha_with_img) > 0 else 0.0,
                "mean_alpha_txt": float(np.mean(alpha_with_img[:, 1])) if len(alpha_with_img) > 0 else 0.0,
                "mean_alpha_img": mean_img_with,
                "std_alpha_img": std_img_with,
            }
            report["text_only"] = {
                "mean_alpha_id": float(np.mean(alpha_txt_only[:, 0])) if len(alpha_txt_only) > 0 else 0.0,
                "mean_alpha_txt": float(np.mean(alpha_txt_only[:, 1])) if len(alpha_txt_only) > 0 else 0.0,
                "mean_alpha_img": mean_img_without,
                "std_alpha_img": std_img_without,
            }

            # Statistical significance verification (Plan Step 6.2): Mann-Whitney U test
            if len(alpha_with_img) > 0 and len(alpha_txt_only) > 0:
                try:
                    from scipy import stats
                    u_stat, p_val = stats.mannwhitneyu(
                        alpha_with_img[:, 2], alpha_txt_only[:, 2], alternative="two-sided"
                    )
                    report["statistical_test"] = {
                        "test_name": "Mann-Whitney U Test (Image Attribution vs Missing)",
                        "statistic": float(u_stat),
                        "p_value": float(p_val),
                        "significant_at_005": bool(p_val < 0.05),
                    }
                except Exception as e:
                    report["statistical_test"] = {"error": str(e)}
        else:
            # Model trained without visual modality (2 columns: id, txt)
            report["with_image"] = {
                "mean_alpha_id": float(np.mean(alpha_with_img[:, 0])) if len(alpha_with_img) > 0 else 0.0,
                "mean_alpha_txt": float(np.mean(alpha_with_img[:, 1])) if len(alpha_with_img) > 0 else 0.0,
                "mean_alpha_img": 0.0,
                "std_alpha_img": 0.0,
            }
            report["text_only"] = {
                "mean_alpha_id": float(np.mean(alpha_txt_only[:, 0])) if len(alpha_txt_only) > 0 else 0.0,
                "mean_alpha_txt": float(np.mean(alpha_txt_only[:, 1])) if len(alpha_txt_only) > 0 else 0.0,
                "mean_alpha_img": 0.0,
                "std_alpha_img": 0.0,
            }

        return report

    def analyze_solution_attribution(
        self,
        all_beta_s: np.ndarray,
        all_m_img_s: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compute aggregate statistics for solution modality attribution weights beta_tau^s = (beta_txt, beta_img).

        Args:
            all_beta_s: Solution modality weights array [N, 2] (txt, img).
            all_m_img_s: Binary flags indicating solution image presence [N].

        Returns:
            Dictionary containing mean and std for solution branches.
        """
        sol_img_mask = (all_m_img_s == 1.0)
        sol_txt_mask = (all_m_img_s == 0.0)

        beta_with_img = all_beta_s[sol_img_mask] if np.any(sol_img_mask) else np.zeros((0, 2))
        beta_txt_only = all_beta_s[sol_txt_mask] if np.any(sol_txt_mask) else np.zeros((0, 2))

        return {
            "num_solutions_with_image": int(sol_img_mask.sum()),
            "num_solutions_text_only": int(sol_txt_mask.sum()),
            "with_image": {
                "mean_beta_txt": float(np.mean(beta_with_img[:, 0])) if len(beta_with_img) > 0 else 0.0,
                "mean_beta_img": float(np.mean(beta_with_img[:, 1])) if len(beta_with_img) > 0 else 0.0,
                "std_beta_img": float(np.std(beta_with_img[:, 1])) if len(beta_with_img) > 0 else 0.0,
            },
            "text_only": {
                "mean_beta_txt": float(np.mean(beta_txt_only[:, 0])) if len(beta_txt_only) > 0 else 0.0,
                "mean_beta_img": float(np.mean(beta_txt_only[:, 1])) if len(beta_txt_only) > 0 else 0.0,
                "std_beta_img": float(np.std(beta_txt_only[:, 1])) if len(beta_txt_only) > 0 else 0.0,
            },
        }

    def extract_top_k_historical_evidence(
        self,
        attn_weights: np.ndarray,
        step_t: int,
        q_seq: np.ndarray,
        r_seq: np.ndarray,
        modality_masks: np.ndarray,
        leaf_kcs: Optional[np.ndarray] = None,
        leaf_kc_mask: Optional[np.ndarray] = None,
        context_start: int = 0,
        k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Extract Top-K historical interactions with highest attention weights for prediction at step t.
        
        Note on Index Alignment:
        The sequence is right-shifted by 1 position using y_0 BOS token:
        Column 0 in attention matrix corresponds to y_0 (prior/sink).
        Column tau' in [1 .. t] corresponds to student historical step tau = tau' - 1.

        Args:
            attn_weights: SAKT attention weights matrix [L, L].
            step_t: Target prediction step index (0-indexed, where t >= 1).
            q_seq: Question sequence array [L].
            r_seq: Student response sequence array [L].
            modality_masks: Modality presence array [L, 3].
            leaf_kcs: Optional leaf KC indices array [L, num_leaves].
            leaf_kc_mask: Optional leaf KC mask array [L, num_leaves].
            context_start: Offset step in student's continuous trajectory for multi-window alignment.
            k: Number of top evidence items to retrieve (default: 3).

        Returns:
            List of dictionaries containing evidence for the top-k attended historical steps.
        """
        if step_t == 0:
            # Step 0 has no prior history (attention is entirely on BOS token y_0)
            return []

        # Attention row at target step t
        row_weights = attn_weights[step_t]  # [L]

        # Historical positions tau' in [1 .. step_t] correspond to student steps tau = tau' - 1 in [0 .. step_t - 1]
        hist_col_indices = np.arange(1, step_t + 1)
        hist_weights = row_weights[hist_col_indices]

        # Prior / Sink weight on y_0 (column 0) (Plan Step 4.2 / Issue L1)
        y0_weight = float(row_weights[0])

        # Sort historical interactions descending by attention weight
        num_candidates = len(hist_weights)
        top_k_count = min(k, num_candidates)
        top_sorted_indices = np.argsort(-hist_weights)[:top_k_count]

        evidence_list = []
        for rank, s_idx in enumerate(top_sorted_indices, start=1):
            tau = int(hist_col_indices[s_idx] - 1)  # Actual historical step index in local window
            weight = float(hist_weights[s_idx])

            # Extract associated skills C_tau (Plan Step 6.2 requirement)
            skill_ids: List[int] = []
            if leaf_kcs is not None and leaf_kc_mask is not None:
                valid_kc_indices = leaf_kc_mask[tau].astype(bool)
                skill_ids = [int(kc) for kc in leaf_kcs[tau][valid_kc_indices].tolist()]

            evidence_item = {
                "rank": rank,
                "history_step": tau,
                "target_step": step_t,
                "actual_history_step": context_start + tau,
                "actual_target_step": context_start + step_t,
                "attention_weight": weight,
                "prior_sink_weight": y0_weight,
                "question_id": int(q_seq[tau]),
                "skill_ids": skill_ids,
                "response": int(r_seq[tau]),
                "has_stem_image": bool(modality_masks[tau, 0] == 1.0),
                "has_sol_text": bool(modality_masks[tau, 1] == 1.0),
                "has_sol_image": bool(modality_masks[tau, 2] == 1.0),
            }
            evidence_list.append(evidence_item)

        return evidence_list
