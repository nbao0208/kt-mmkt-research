"""
Four-Tier Explainability Framework for ACD-MMKTv2.
Asymmetric Causal-Disentangled Multimodal Knowledge Tracing v2.

Implements the exact 4-tier explainability specification:
- Tier 1: Psychometric Equilibrium Evidence (p_hat_t, theta_{u,t}, tilde_mu_{q,t}, mu_q, a_q, Delta_gap).
  100% mathematically faithful: p_hat_t >= 0.5 <=> theta_{u,t} >= tilde_mu_{q,t}.
- Tier 2: Difficulty Source Anatomy (mu_concept, Delta_mu_stem, Delta_mu_sol) on logit scale.
- Tier 3: Symmetric Fine-Grained Multimodal Attribution:
  - Question Stem: Top text keywords (beta_txt^q), visual gate (lambda_q), and attended visual patch (A_cross^q).
  - Solution: Top algorithmic steps (beta_txt^s), visual gate (lambda_s), and attended visual patch (A_cross^s).
- Tier 4: Empirical History Tracing:
  Top-K influential past interactions from monotonic attention (alpha_ret) with student responses (r_tau).
- Structured Markdown Report and pedagogical diagnosis synthesis.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import os
import torch
from torch.utils.data import DataLoader
import numpy as np


class ACDv2ExplainabilityEngine:
    """
    Explainability Engine for ACD-MMKTv2.
    Extracts and formats 4-tier multi-granular pedagogical explanations.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        aux_metadata_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize ACDv2ExplainabilityEngine.

        Args:
            model: Trained ACDMMKTv2 model.
            aux_metadata_path: Path to acd_v2_aux_metadata.pt.
            device: Compute device.
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.aux_metadata = None

        if aux_metadata_path and os.path.exists(aux_metadata_path):
            self.aux_metadata = torch.load(aux_metadata_path, map_location="cpu")
        else:
            default_aux = os.path.join("data", "processed", "XES3G5M", "acd_v2_aux_metadata.pt")
            if os.path.exists(default_aux):
                self.aux_metadata = torch.load(default_aux, map_location="cpu")

    def get_token_strings(self, q_id: int, branch: str = "q") -> List[str]:
        """Retrieve token strings list for a given question and branch ('q' or 's')."""
        if self.aux_metadata:
            key = "token_to_word_q" if branch == "q" else "token_to_word_s"
            if key in self.aux_metadata and q_id < len(self.aux_metadata[key]):
                return self.aux_metadata[key][q_id]
        return [f"token_{i}" for i in range(128 if branch == "q" else 160)]

    @torch.no_grad()
    def explain_interaction(
        self,
        batch: Dict[str, torch.Tensor],
        sample_idx: int = 0,
        target_step: int = -1,
        top_k_keywords: int = 5,
        top_k_history: int = 5,
    ) -> Dict[str, Any]:
        """
        Extract the full 4-tier evidence package for a given student interaction.

        Args:
            batch: Model sequence batch dictionary.
            sample_idx: Index of sequence in batch (default: 0).
            target_step: Step in sequence to explain (-1 = last valid step).
            top_k_keywords: Number of top text tokens to extract (default: 5).
            top_k_history: Number of historical interactions to trace (default: 5).

        Returns:
            Dictionary containing 4-tier evidence and diagnostic text.
        """
        self.model.eval()
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)

        raw_evidence = self.model.extract_evidence(
            batch=batch,
            sample_idx=sample_idx,
            target_step=target_step,
            top_k_history=top_k_history,
        )

        target_q_id = raw_evidence["target_q_id"]
        tier1 = raw_evidence["tier1_psychometrics"]
        tier2 = raw_evidence["tier2_difficulty"]
        tier3 = raw_evidence["tier3_multimodal"]
        tier4 = raw_evidence["tier4_history"]

        # Map Tier 3 attention distributions to token strings and image patch coordinates
        tokens_q = self.get_token_strings(target_q_id, branch="q")
        tokens_s = self.get_token_strings(target_q_id, branch="s")
        mask_txt_q = (
            self.model.acd_mask_txt_q[target_q_id].cpu().numpy()
            if hasattr(self.model, "acd_mask_txt_q")
            else None
        )
        mask_txt_s = (
            self.model.acd_mask_txt_s[target_q_id].cpu().numpy()
            if hasattr(self.model, "acd_mask_txt_s")
            else None
        )

        # Top stem keywords
        beta_q = tier3["beta_txt_q"]
        top_q_indices = np.argsort(beta_q)[::-1]
        stem_keywords = []
        special_tokens = {"[CLS]", "[SEP]", "[PAD]", "[UNK]"}
        for idx in top_q_indices:
            # Mask out padding tokens
            if mask_txt_q is not None and idx < len(mask_txt_q) and not bool(mask_txt_q[idx]):
                continue
            tok = tokens_q[idx] if idx < len(tokens_q) else f"token_{idx}"
            if tok in special_tokens:
                continue
            # Corresponding visual patch if image is present
            patch_coord = None
            if tier3["has_img_stem"]:
                patch_idx = int(np.argmax(tier3["A_cross_q"][idx]))
                patch_row = patch_idx // 7
                patch_col = patch_idx % 7
                patch_coord = {"patch_idx": patch_idx, "row": patch_row, "col": patch_col}

            stem_keywords.append({
                "token": tok,
                "token_idx": int(idx),
                "attention_weight": float(beta_q[idx]),
                "linked_patch": patch_coord,
            })
            if len(stem_keywords) >= top_k_keywords:
                break

        # Top solution algorithmic steps
        beta_s = tier3["beta_txt_s"]
        top_s_indices = np.argsort(beta_s)[::-1]
        sol_keywords = []
        for idx in top_s_indices:
            # Mask out padding tokens
            if mask_txt_s is not None and idx < len(mask_txt_s) and not bool(mask_txt_s[idx]):
                continue
            tok = tokens_s[idx] if idx < len(tokens_s) else f"token_{idx}"
            if tok in special_tokens:
                continue
            patch_coord = None
            if tier3["has_img_sol"]:
                patch_idx = int(np.argmax(tier3["A_cross_s"][idx]))
                patch_row = patch_idx // 7
                patch_col = patch_idx % 7
                patch_coord = {"patch_idx": patch_idx, "row": patch_row, "col": patch_col}

            sol_keywords.append({
                "token": tok,
                "token_idx": int(idx),
                "attention_weight": float(beta_s[idx]),
                "linked_patch": patch_coord,
            })
            if len(sol_keywords) >= top_k_keywords:
                break

        enriched_tier3 = {
            "lambda_q_mean": tier3["lambda_q_mean"],
            "lambda_s_mean": tier3["lambda_s_mean"],
            "has_img_stem": tier3["has_img_stem"],
            "has_img_sol": tier3["has_img_sol"],
            "stem_keywords": stem_keywords,
            "solution_keywords": sol_keywords,
        }

        # Synthesize pedagogical diagnosis
        report_md = self.generate_markdown_report(
            target_q_id=target_q_id,
            target_step=raw_evidence["target_step"],
            tier1=tier1,
            tier2=tier2,
            tier3=enriched_tier3,
            tier4=tier4,
        )

        return {
            "target_q_id": target_q_id,
            "target_step": raw_evidence["target_step"],
            "tier1": tier1,
            "tier2": tier2,
            "tier3": enriched_tier3,
            "tier4": tier4,
            "markdown_report": report_md,
        }

    def generate_markdown_report(
        self,
        target_q_id: int,
        target_step: int,
        tier1: Dict[str, Any],
        tier2: Dict[str, Any],
        tier3: Dict[str, Any],
        tier4: Dict[str, Any],
    ) -> str:
        """
        Format a complete 4-tier pedagogical diagnosis report in GitHub Markdown.
        """
        pred_label = "Correct (1)" if tier1["p_hat"] >= 0.5 else "Incorrect (0)"
        status_emoji = "✅" if tier1["p_hat"] >= 0.5 else "❌"

        lines = [
            f"# 📊 ACD-MMKTv2 Pedagogical Diagnostic Report",
            f"**Question ID:** `Q_{target_q_id}` | **Step:** `t = {target_step}` | "
            f"**Predicted Outcome:** {status_emoji} **{pred_label}** (Probability: `{tier1['p_hat']:.4f}`)",
            f"",
            f"---",
            f"## Tier 1 — Psychometric Equilibrium Evidence (2PL-IRT)",
            f"- **Student Proficiency (\\(\\theta_{{u,t}}\\)):** `{tier1['theta']:+.3f}` (Real logit scale)",
            f"- **Total Item Difficulty (\\(\\tilde{{\\mu}}_{{q,t}}\\)):** `{tier1['tilde_mu']:+.3f}` (Probability scale \\(\\mu_{{q,t}} = {tier1['mu_prob']:.3f}\\))",
            f"- **Item Discrimination (\\(a_{{q,t}}\\)):** `{tier1['a_disc']:.3f}` (Sensitivity scaling factor)",
            f"- **Competency Gap (\\(\\Delta_{{\\text{{gap}},t}} = \\theta_{{u,t}} - \\tilde{{\\mu}}_{{q,t}}\\)):** `{tier1['delta_gap']:+.3f}`",
            f"- **Faithfulness (100% Guaranteed):** `{'PASSED (Consistent)' if tier1['is_faithful_100'] else 'FAILED'}`",
            f"",
            f"> **Diagnostic Summary:** The student's current proficiency "
            f"{'exceeds' if tier1['delta_gap'] >= 0 else 'falls below'} the required difficulty by "
            f"`{abs(tier1['delta_gap']):.3f}` logits. "
            f"Since \\(a_{{q,t}} = {tier1['a_disc']:.2f} > 0\\), the predicted success probability strictly aligns with the proficiency-difficulty balance.",
            f"",
            f"---",
            f"## Tier 2 — Multi-Source Difficulty Source Anatomy",
            f"The intrinsic cognitive demand of `Q_{target_q_id}` decomposes additively into three pedagogical sources:",
            f"",
            f"| Difficulty Component | Logit Value | Relative Contribution | Pedagogical Role |",
            f"| :--- | :---: | :---: | :--- |",
            f"| **Baseline Concept (\\(\\mu_{{\\text{{concept}}}}\\))** | `{tier2['mu_concept']:+.3f}` | `{tier2['pct_concept']:.1f}%` | Core domain abstractness of the question's KC prerequisites. |",
            f"| **Stem Perception (\\(\\Delta \\mu_{{\\text{{stem}}}}\\))** | `{tier2['delta_mu_stem']:+.3f}` | `{tier2['pct_stem']:.1f}%` | Cognitive load from problem framing and diagram reading. |",
            f"| **Solution Algorithm (\\(\\Delta \\mu_{{\\text{{sol}}}}\\))** | `{tier2['delta_mu_sol']:+.3f}` | `{tier2['pct_sol']:.1f}%` | Inherent complexity of formula derivations and reasoning steps. |",
            f"",
            f"> **Difficulty Attribution:** The primary cognitive bottleneck originates from "
            f"**{'Concept Prerequisites' if tier2['pct_concept'] >= max(tier2['pct_stem'], tier2['pct_sol']) else 'Stem Presentation' if tier2['pct_stem'] >= tier2['pct_sol'] else 'Algorithmic Complexity'}**.",
            f"",
            f"---",
            f"## Tier 3 — Symmetric Fine-Grained Multimodal Attribution",
            f"### Branch 1: Problem Stem (Question Input)",
            f"- **Visual Reliance Gate (\\(\\bar{{\\lambda}}_t^q\\)):** `{tier3['lambda_q_mean']:.3f}` "
            f"({'Active image processing' if tier3['has_img_stem'] else 'Text-only item, visual gate locked to 0.000'})",
            f"- **Salient Text Keywords & Linked Visual Coordinates (7x7 Grid):**",
        ]

        for item in tier3["stem_keywords"]:
            tok = item["token"]
            w = item["attention_weight"]
            patch_info = f" -> Visual Grid Patch `(r={item['linked_patch']['row']}, c={item['linked_patch']['col']})`" if item["linked_patch"] else ""
            lines.append(f"  - Term `\"{tok}\"` (Attention: `{w:.4f}`){patch_info}")

        lines.extend([
            f"",
            f"### Branch 2: Solution Blueprint (Algorithmic Cognitive Demand)",
            f"- **Visual Geometry Gate (\\(\\bar{{\\lambda}}_t^s\\)):** `{tier3['lambda_s_mean']:.3f}` "
            f"({'Active auxiliary geometry processing' if tier3['has_img_sol'] else 'Text-only derivation, visual gate locked to 0.000'})",
            f"- **Salient Derivation Steps & Auxiliary Visual Structures:**",
        ])

        for item in tier3["solution_keywords"]:
            tok = item["token"]
            w = item["attention_weight"]
            patch_info = f" -> Diagram Element `(r={item['linked_patch']['row']}, c={item['linked_patch']['col']})`" if item["linked_patch"] else ""
            lines.append(f"  - Step `\"{tok}\"` (Attention: `{w:.4f}`){patch_info}")

        lines.extend([
            f"",
            f"---",
            f"## Tier 4 — Empirical Historical Knowledge Tracing",
            f"Top historical interactions retrieved by monotonic context-aware attention (\\(\\alpha_{{t,\\tau}}^{{\\text{{ret}}}}\\)):",
            f"",
            f"| Past Step (Global) | Historical Item | Historical Difficulty (\\(\\mu_q\\)) | Student Response | Retrieval Weight (\\(\\alpha\\)) | Diagnostic Impact |",
            f"| :---: | :---: | :---: | :---: | :---: | :--- |",
        ])

        for hist in tier4["top_history_interactions"]:
            step = hist["step_tau"]
            g_step = hist.get("global_step", step)
            qid = hist["q_id"]
            resp = "Correct (1) ✅" if hist["response_r"] == 1 else "Incorrect (0) ❌"
            w = hist["retrieval_weight"]
            diff_str = f"`{hist['mu_q']:.3f}`" if "mu_q" in hist else "N/A"
            impact = "Reinforced concept proficiency" if hist["response_r"] == 1 else "Exposed cognitive gap"
            lines.append(f"| `\\tau = {step}` (`g = {g_step}`) | `Q_{qid}` | {diff_str} | {resp} | `{w:.4f}` | {impact} |")

        lines.extend([
            f"",
            f"---",
            f"## Actionable Pedagogical Recommendations",
            f"1. **Remediation Target:** {'Proceed to higher difficulty extension problems.' if tier1['delta_gap'] >= 0 else 'Review the highlighted keywords in Branch 1 and review previous similar failed problems.'}",
            f"2. **Cognitive Load Adjustment:** {'Focus on streamlining algebraic derivation steps.' if tier2['pct_sol'] > 40 else 'Strengthen visual diagram decomposition skills.' if tier2['pct_stem'] > 40 else 'Review fundamental KC definitions and theorems.'}",
        ])

        return "\n".join(lines)

    @torch.no_grad()
    def verify_mathematical_faithfulness(self, dataloader: DataLoader, max_samples: int = 1000) -> Dict[str, Any]:
        """
        Verify that 100% of model predictions satisfy:
        (p_hat >= 0.5) <=> (theta >= tilde_mu).

        Returns:
            Dictionary with total checked, total consistent, and faithfulness percentage.
        """
        self.model.eval()
        self.model.precompute_item_cache()

        total_checked = 0
        consistent_count = 0

        for batch in dataloader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)

            eval_mask = batch["eval_mask"].bool()
            outputs = self.model(batch, use_cache=True)

            probs = outputs["probs"][eval_mask]
            thetas = outputs["theta"][eval_mask]
            tilde_mus = outputs["tilde_mu_seq"].squeeze(-1)[eval_mask]

            p_binary = (probs >= 0.5)
            gap_binary = (thetas >= tilde_mus)

            is_consistent = (p_binary == gap_binary)
            consistent_count += int(is_consistent.sum().item())
            total_checked += len(is_consistent)

            if total_checked >= max_samples:
                break

        self.model.clear_item_cache()
        pct = (consistent_count / max(1, total_checked)) * 100.0

        return {
            "total_checked": total_checked,
            "consistent_count": consistent_count,
            "faithfulness_percentage": pct,
            "contradictions": total_checked - consistent_count,
        }
