"""
Multi-Level Explainability Extractor for MG-MMKT v2.

Implements Section 6 of docs/models/MG-MMKTv2.md and Plan Step 7:
1. Level 1 - Modality Attribution:
   - Question weights alpha_t^q = (alpha_id, alpha_txt, alpha_img) in Delta^2.
   - Solution weights beta_tau^s = (beta_txt, beta_img) in Delta^1.
   - Statistical testing (Mann-Whitney U) and missing suppression audit.
2. Level 2 - Disentangled Historical Evidence:
   - Top-K past interaction retrieval based on causal attention gamma_bar.
   - Chronological step alignment incorporating context_start offset.
   - Pedagogical pathway activation labeling:
     * "Mastery Demonstration Pathway" (Delta y^(1)) when r_tau = 1.
     * "Remedial Intervention Pathway" (Delta y^(0)) when r_tau = 0.
   - Attention head forgetting decay rates theta_1..theta_H.
3. Level 3 - Implicit Cognitive Demand Attribution:
   - Concept alignment: cos(h_t, x_t) and ||h_t * x_t||_2.
   - Algorithmic demand alignment: cos(h_t, hat_z_t^s) and ||h_t * hat_z_t^s||_2.
   - Distillation fidelity: cos(hat_z_t^s, z_t^(s, clean)).
   - Failure diagnostic categorization (Concept gap vs Algorithmic overload).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats


class MultiLevelExplainerV2:
    """
    3-Level Explainability Extractor for MG-MMKT v2.
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """
        Args:
            metadata: Optional dictionary containing question texts, images, and KC names.
        """
        self.metadata = metadata or {}

    def analyze_level1_modality_attribution(
        self,
        all_alpha_q: np.ndarray,
        all_beta_s: np.ndarray,
        all_m_img_q: np.ndarray,
        all_m_img_s: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Level 1: Aggregate statistics and hypothesis testing for question and solution modality gates.

        Args:
            all_alpha_q: Question gate weights [N, 3] for (id, txt, img).
            all_beta_s: Solution gate weights [N, 2] for (txt, img).
            all_m_img_q: Question image presence flags [N].
            all_m_img_s: Solution image presence flags [N].

        Returns:
            Dictionary with statistical breakdown and Mann-Whitney U test results.
        """
        q_img_mask = (all_m_img_q == 1.0)
        q_txt_mask = (all_m_img_q == 0.0)

        s_img_mask = (all_m_img_s == 1.0)
        s_txt_mask = (all_m_img_s == 0.0)

        # Question Modality Attribution
        alpha_with_img = all_alpha_q[q_img_mask] if np.any(q_img_mask) else np.zeros((0, 3))
        alpha_no_img = all_alpha_q[q_txt_mask] if np.any(q_txt_mask) else np.zeros((0, 3))

        report: Dict[str, Any] = {
            "question_level": {
                "num_with_image": int(q_img_mask.sum()),
                "num_text_only": int(q_txt_mask.sum()),
                "with_image": {
                    "mean_alpha_id": float(np.mean(alpha_with_img[:, 0])) if len(alpha_with_img) > 0 else 0.0,
                    "mean_alpha_txt": float(np.mean(alpha_with_img[:, 1])) if len(alpha_with_img) > 0 else 0.0,
                    "mean_alpha_img": float(np.mean(alpha_with_img[:, 2])) if len(alpha_with_img) > 0 else 0.0,
                    "std_alpha_img": float(np.std(alpha_with_img[:, 2])) if len(alpha_with_img) > 0 else 0.0,
                },
                "text_only": {
                    "mean_alpha_id": float(np.mean(alpha_no_img[:, 0])) if len(alpha_no_img) > 0 else 0.0,
                    "mean_alpha_txt": float(np.mean(alpha_no_img[:, 1])) if len(alpha_no_img) > 0 else 0.0,
                    "mean_alpha_img": float(np.mean(alpha_no_img[:, 2])) if len(alpha_no_img) > 0 else 0.0,
                    "std_alpha_img": float(np.std(alpha_no_img[:, 2])) if len(alpha_no_img) > 0 else 0.0,
                },
            },
            "solution_level": {
                "num_with_image": int(s_img_mask.sum()),
                "num_text_only": int(s_txt_mask.sum()),
                "with_image": {
                    "mean_beta_txt": float(np.mean(all_beta_s[s_img_mask, 0])) if np.any(s_img_mask) else 0.0,
                    "mean_beta_img": float(np.mean(all_beta_s[s_img_mask, 1])) if np.any(s_img_mask) else 0.0,
                },
                "text_only": {
                    "mean_beta_txt": float(np.mean(all_beta_s[s_txt_mask, 0])) if np.any(s_txt_mask) else 0.0,
                    "mean_beta_img": float(np.mean(all_beta_s[s_txt_mask, 1])) if np.any(s_txt_mask) else 0.0,
                },
            },
        }

        # Statistical significance test (Mann-Whitney U) on Question Image Weight
        if len(alpha_with_img) > 0 and len(alpha_no_img) > 0:
            try:
                u_stat, p_val = stats.mannwhitneyu(
                    alpha_with_img[:, 2], alpha_no_img[:, 2], alternative="greater"
                )
                report["question_level"]["mann_whitney_u"] = {
                    "statistic": float(u_stat),
                    "p_value": float(p_val),
                    "significant": bool(p_val < 0.05),
                }
            except Exception as e:
                report["question_level"]["mann_whitney_u"] = {"error": str(e)}

        return report

    def extract_level2_historical_evidence(
        self,
        gamma_bar_row: np.ndarray,
        q_history: List[int],
        r_history: List[int],
        kcs_history: Optional[List[List[int]]] = None,
        context_start: int = 0,
        top_k: int = 3,
        head_thetas: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Level 2: Extract Top-K historical evidence interactions for target step t.

        Args:
            gamma_bar_row: Attention weights row gamma_bar[t, :t] of length t.
            q_history: Question IDs for historical steps 0..t-1.
            r_history: Student correctness responses (0 or 1) for historical steps 0..t-1.
            kcs_history: Optional list of KC ID lists for historical steps.
            context_start: Window context start offset.
            top_k: Number of historical interactions to retrieve.
            head_thetas: Optional list of forgetting decay values for attention heads.

        Returns:
            List of dictionaries for each Top-K evidence interaction.
        """
        num_hist = len(q_history)
        if num_hist == 0:
            return []

        weights = np.array(gamma_bar_row[:num_hist])
        top_indices = np.argsort(weights)[::-1][: min(top_k, num_hist)]

        evidence_list = []
        for idx in top_indices:
            resp = int(r_history[idx])
            pathway = "Mastery Demonstration Pathway" if resp == 1 else "Remedial Intervention Pathway"
            global_step = context_start + int(idx)

            kcs = kcs_history[idx] if (kcs_history is not None and idx < len(kcs_history)) else []

            evidence = {
                "local_step": int(idx),
                "global_step": global_step,
                "attention_weight": float(weights[idx]),
                "question_id": int(q_history[idx]),
                "response": resp,
                "pathway": pathway,
                "activated_pathway": pathway,
                "kcs": kcs,
            }
            if head_thetas is not None:
                evidence["head_forgetting_rates"] = [float(th) for th in head_thetas]

            evidence_list.append(evidence)

        return evidence_list

    def analyze_level3_cognitive_demand(
        self,
        h_t: torch.Tensor,
        x_t: torch.Tensor,
        hat_z_t_s: torch.Tensor,
        z_t_s_clean: Optional[torch.Tensor] = None,
        prob_t: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Level 3: Diagnose student cognitive state and failure cause (Concept gap vs Algorithmic overload).

        Args:
            h_t: Retrieved knowledge state [d].
            x_t: Question representation [d].
            hat_z_t_s: Estimated cognitive demand representation [d].
            z_t_s_clean: Clean ground-truth solution representation [d] (if available).
            prob_t: Model predicted probability of success p_hat_t.

        Returns:
            Dictionary with cosine alignment metrics, norms, and diagnosis explanation.
        """
        h_t = h_t.float().view(-1)
        x_t = x_t.float().view(-1)
        hat_z = hat_z_t_s.float().view(-1)

        # 1. Concept Alignment
        cos_hx = float(F.cosine_similarity(h_t.unsqueeze(0), x_t.unsqueeze(0), eps=1e-8).item())
        norm_hx = float(torch.norm(h_t * x_t, p=2).item())

        # 2. Algorithmic / Cognitive Demand Alignment
        cos_hz = float(F.cosine_similarity(h_t.unsqueeze(0), hat_z.unsqueeze(0), eps=1e-8).item())
        norm_hz = float(torch.norm(h_t * hat_z, p=2).item())

        # 3. Distillation Fidelity (if ground truth solution provided)
        cos_distill = None
        if z_t_s_clean is not None:
            z_clean = z_t_s_clean.float().view(-1)
            cos_distill = float(F.cosine_similarity(hat_z.unsqueeze(0), z_clean.unsqueeze(0), eps=1e-8).item())

        # 4. Diagnostic Reasoning
        if prob_t >= 0.5:
            diagnosis = "Mastery: Student demonstrates solid concept mastery aligned with algorithmic demand."
            category = "Proficient"
        else:
            if cos_hx < 0.2 and cos_hz < 0.2:
                diagnosis = "Deficiency in both domain concept and algorithmic solution techniques."
                category = "General Knowledge Deficiency"
            elif cos_hx >= 0.3 and cos_hz < 0.15:
                diagnosis = "Understands question concept, but experiences cognitive overload on complex multi-step solution method."
                category = "Algorithmic Cognitive Overload"
            elif cos_hx < 0.15 and cos_hz >= 0.3:
                diagnosis = "Familiar with problem-solving steps, but lacks core conceptual foundation of question stem."
                category = "Conceptual Gap"
            else:
                diagnosis = "Marginal knowledge alignment below proficiency threshold."
                category = "Borderline Struggle"

        return {
            "predicted_prob": float(prob_t),
            "concept_cosine": cos_hx,
            "concept_hadamard_norm": norm_hx,
            "cognitive_demand_cosine": cos_hz,
            "cognitive_demand_hadamard_norm": norm_hz,
            "distillation_fidelity_cosine": cos_distill,
            "category": category,
            "diagnosis": diagnosis,
        }

