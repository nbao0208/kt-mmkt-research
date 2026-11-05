"""
Four-Tier Explainability Framework for ACD-MMKTv3.
Asymmetric Causal-Disentangled Multimodal Knowledge Tracing v3.

Implements the exact 4-tier explainability specification:
- Tier 1: Psychometric Equilibrium Evidence (p_hat_t, theta_{u,t}, tilde_mu_{q,t}, mu_q, a_q, Delta_gap).
  100% mathematically faithful: p_hat_t >= 0.5 <=> theta_{u,t} >= tilde_mu_{q,t}.
- Tier 2: Difficulty Source Anatomy (mu_concept, Delta_mu_stem, Delta_mu_sol) on logit scale.
- Tier 3: Fine-Grained Multimodal Attribution:
  - Text Stem: Top text keywords (beta_txt^q) and text difficulty load (Delta_mu_txt^q).
  - Visual Stem: Signed visual load (Delta_mu_vis^q in [-gamma_vis, gamma_vis]), visual gate (bar_lambda_q),
    and attended visual patch map (A_cross^q with letterbox masking).
    Signed interpretation: Delta_mu_vis < 0 is a visual scaffold/aid; Delta_mu_vis > 0 is cognitive visual load.
  - Solution: Top algorithmic steps (beta_txt^s) and algorithmic load (Delta_mu_sol).
- Tier 4: Atomic Empirical History Tracing:
  Top-K influential past interactions from monotonic attention (alpha_ret) with student responses (r_tau).
- Structured Markdown Report and pedagogical diagnosis synthesis.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import os
import torch
import numpy as np


class ACDv3ExplainabilityEngine:
    """
    Explainability Engine for ACD-MMKTv3.
    Extracts and formats 4-tier multi-granular pedagogical explanations.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        aux_metadata_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize ACDv3ExplainabilityEngine.

        Args:
            model: Trained ACDMMKTv3 model.
            aux_metadata_path: Path to acd_v3_aux_metadata.pt.
            device: Compute device.
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.aux_metadata = None

        if aux_metadata_path and os.path.exists(aux_metadata_path):
            self.aux_metadata = torch.load(aux_metadata_path, map_location="cpu")
        else:
            default_aux = os.path.join("data", "processed", "XES3G5M", "acd_v3_aux_metadata.pt")
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
        mu_concept_prior: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Extract the full 4-tier evidence package for a given student interaction.
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
            mu_concept_prior=mu_concept_prior,
        )

        target_q_id = raw_evidence["target_q_id"]
        tier1 = raw_evidence["tier1_psychometrics"]
        tier2 = raw_evidence["tier2_difficulty"]
        tier3 = raw_evidence["tier3_multimodal"]
        tier4 = raw_evidence["tier4_history"]

        # Map Tier 3 attention distributions to token strings
        tokens_q = self.get_token_strings(target_q_id, branch="q")
        tokens_s = self.get_token_strings(target_q_id, branch="s")

        # Top stem keywords
        beta_q = tier3["beta_txt_q"]
        top_q_indices = np.argsort(beta_q)[::-1]
        stem_keywords = []
        for idx in top_q_indices:
            tok = tokens_q[idx] if idx < len(tokens_q) else f"tok_{idx}"
            if tok in ("[PAD]", "[CLS]", "[SEP]"):
                continue
            stem_keywords.append({
                "token": tok,
                "token_idx": int(idx),
                "weight": float(beta_q[idx]),
            })
            if len(stem_keywords) >= top_k_keywords:
                break

        # Top solution steps
        beta_s = tier3["beta_txt_s"]
        top_s_indices = np.argsort(beta_s)[::-1]
        solution_steps = []
        if getattr(self.model, "use_solution", True):
            for idx in top_s_indices:
                if beta_s[idx] <= 1e-8:
                    continue
                tok = tokens_s[idx] if idx < len(tokens_s) else f"tok_{idx}"
                if tok in ("[PAD]", "[CLS]", "[SEP]"):
                    continue
                solution_steps.append({
                    "token": tok,
                    "token_idx": int(idx),
                    "weight": float(beta_s[idx]),
                })
                if len(solution_steps) >= top_k_keywords:
                    break

        # Top attended visual patches
        A_cross = tier3["A_cross_q"]
        top_patches = []
        if A_cross is not None:
            # Mathematical attribution: patch_weights = beta_q @ A_cross (weighted by token importance)
            # A_cross is [L_q, P], beta_q is [L_q]
            patch_weights = np.matmul(beta_q, A_cross)  # [P]
            top_p_indices = np.argsort(patch_weights)[::-1]
            for p_idx in top_p_indices:
                p_w = float(patch_weights[p_idx])
                if p_w <= 1e-8:
                    continue
                row = int(p_idx // 7)
                col = int(p_idx % 7)
                top_patches.append({
                    "patch_idx": int(p_idx),
                    "grid_pos": f"({row}, {col})",
                    "weight": p_w,
                })
                if len(top_patches) >= 5:
                    break

        # Image metadata
        img_info_q = {}
        if self.aux_metadata and "image_meta_q" in self.aux_metadata and target_q_id < len(self.aux_metadata["image_meta_q"]):
            img_info_q = self.aux_metadata["image_meta_q"][target_q_id]

        img_info_s = {}
        if self.aux_metadata and "image_meta_s" in self.aux_metadata and target_q_id < len(self.aux_metadata["image_meta_s"]):
            img_info_s = self.aux_metadata["image_meta_s"][target_q_id]

        # Signed visual load semantic description
        t_resolved = raw_evidence["target_step"]
        if "modality_masks" in batch and batch["modality_masks"] is not None:
            has_stem_img = bool(batch["modality_masks"][sample_idx, t_resolved, 0].item() > 0.5)
        elif img_info_q and "has_image" in img_info_q:
            has_stem_img = bool(img_info_q.get("has_image", False))
        elif hasattr(self.model, "bank_modality_masks") and self.model.bank_modality_masks is not None:
            has_stem_img = bool(self.model.bank_modality_masks[target_q_id, 0].item() > 0.5)
        else:
            has_stem_img = False

        delta_vis = float(tier3["delta_mu_vis"])
        if not has_stem_img:
            vis_semantics = "Text-Only Item (Câu hỏi thuần văn bản — m_img^q = 0, Delta mu_vis = 0.0)"
            delta_vis = 0.0
        elif delta_vis < -0.05:
            vis_semantics = "Visual Scaffold / Aid (Hỗ trợ trực quan hóa, làm giảm độ khó đề bài)"
        elif delta_vis > 0.05:
            vis_semantics = "Visual Cognitive Overload (Hình vẽ nhiều chi tiết, gây tăng tải nhận thức thị giác)"
        else:
            vis_semantics = "Neutral Visual Impact (Tác động thị giác trung tính)"

        explanation = {
            "target_q_id": target_q_id,
            "target_step": raw_evidence["target_step"],
            "tier1_psychometrics": tier1,
            "tier2_difficulty": tier2,
            "tier3_multimodal": {
                "delta_mu_txt": tier3["delta_mu_txt"],
                "delta_mu_vis": tier3["delta_mu_vis"],
                "visual_impact_semantics": vis_semantics,
                "bar_lambda_q": tier3["bar_lambda_q"],
                "stem_keywords": stem_keywords,
                "solution_steps": solution_steps,
                "top_patches": top_patches,
                "image_meta_q": img_info_q,
                "image_meta_s": img_info_s,
            },
            "tier4_history": tier4,
        }
        return explanation

    def format_markdown_report(self, explanation: Dict[str, Any]) -> str:
        """
        Format the extracted 4-tier evidence package into a publication-grade Markdown report.
        """
        t1 = explanation["tier1_psychometrics"]
        t2 = explanation["tier2_difficulty"]
        t3 = explanation["tier3_multimodal"]
        t4 = explanation["tier4_history"]
        qid = explanation["target_q_id"]
        step = explanation["target_step"]

        md = []
        md.append(f"# Báo Cáo Giải Thích Tâm Trắc Học Đa Tầng ACD-MMKTv3 (Câu hỏi Q_{qid}, Bước {step})\n")

        # Tier 1
        md.append("## Tầng 1: Cân Bằng Tâm Trắc Học (Psychometric Equilibrium)")
        md.append("| Đại lượng | Ký hiệu | Giá trị | Ý nghĩa Sư phạm |")
        md.append("| :--- | :---: | :---: | :--- |")
        md.append(f"| **Xác suất Trả lời Đúng** | $\\hat{{p}}_t$ | **{t1['p_hat']:.4f}** | {'Dự đoán ĐÚNG (>= 0.5)' if t1['p_hat'] >= 0.5 else 'Dự đoán SAI (< 0.5)'} |")
        md.append(f"| **Năng lực Người học** | $\\theta_{{u,t}}$ | **{t1['theta']:.4f}** | Trạng thái tri thức tích lũy trên thang Logit |")
        md.append(f"| **Độ khó Tổng thể** | $\\tilde{{\\mu}}_{{q,t}}$ | **{t1['tilde_mu']:.4f}** | Tổng tải nhận thức câu hỏi trên thang Logit (Xác suất: {t1['mu_q']:.4f}) |")
        md.append(f"| **Khoảng cách Năng lực** | $\\Delta_{{\\text{{gap}}}}$ | **{t1['delta_gap']:+.4f}** | $\\theta_{{u,t}} - \\tilde{{\\mu}}_{{q,t}}$ ({'Đạt chuẩn' if t1['delta_gap'] >= 0 else 'Chưa đạt chuẩn'}) |")
        md.append(f"| **Hệ số Phân biệt** | $a_{{q_t}}$ | **{t1['a_disc']:.4f}** | Độ dốc phân biệt nội tại ($a_{{q}} \\ge 0.1$) |")
        md.append(f"| **Tính Nhất quán Quyết định** | IRT Faithfulness | **{'100% NHẤT QUÁN' if t1['prediction_faithful'] else 'KHÔNG NHẤT QUÁN'}** | $\\hat{{p}}_t \\ge 0.5 \\iff \\theta_{{u,t}} \\ge \\tilde{{\\mu}}_{{q,t}}$ |\n")

        # Tier 2
        md.append("## Tầng 2: Giải Phẫu Nguồn Gây Khó (Difficulty Source Anatomy)")
        md.append("Phương trình phân rã cộng tính trên thang Logit: $\\tilde{\\mu}_{q} = \\mu_{\\text{concept}} + \\Delta \\mu_{\\text{stem}} + \\Delta \\mu_{\\text{sol}}$")
        md.append("| Thành phần Độ khó | Ký hiệu | Giá trị Logit | Tỷ trọng Đóng góp |")
        md.append("| :--- | :---: | :---: | :---: |")
        abs_tot = abs(t2["mu_concept"]) + abs(t2["delta_mu_stem"]) + abs(t2["delta_mu_sol"]) + 1e-8
        md.append(f"| **Độ khó Khái niệm Nền tảng** | $\\mu_{{\\text{{concept}}}}$ | {t2['mu_concept']:+.4f} | {abs(t2['mu_concept'])/abs_tot*100:.1f}% |")
        md.append(f"| **Tải Nhận thức Đề bài** | $\\Delta \\mu_{{\\text{{stem}}}}$ | {t2['delta_mu_stem']:+.4f} | {abs(t2['delta_mu_stem'])/abs_tot*100:.1f}% |")
        md.append(f"| **Tải Giải thuật Lời giải** | $\\Delta \\mu_{{\\text{{sol}}}}$ | {t2['delta_mu_sol']:+.4f} | {abs(t2['delta_mu_sol'])/abs_tot*100:.1f}% |")
        md.append(f"| **Tổng Logit Tái lập** | $\\tilde{{\\mu}}_{{q}}^{{\\text{{rec}}}}$ | {t2['tilde_mu_reconstructed']:+.4f} | Sai số tái lập: {abs(t2['tilde_mu_reconstructed'] - t1['tilde_mu']):.2e} |\n")

        # Tier 3
        md.append("## Tầng 3: Quy Gán Phương Thức Hạt Mịn (Fine-Grained Modality Attribution)")
        md.append(f"- **Tải văn bản đề bài ($\\Delta \\mu_{{\\text{{txt}}}}^q$):** {t3['delta_mu_txt']:+.4f}")
        md.append(f"- **Tải thị giác có dấu ($\\Delta \\mu_{{\\text{{vis}}}}^q$):** {t3['delta_mu_vis']:+.4f} -> **{t3['visual_impact_semantics']}**")
        md.append(f"- **Cường độ Cổng Thị giác ($\\bar{{\\lambda}}^q$):** {t3['bar_lambda_q']:.4f}")

        md.append("\n**Top Từ khóa Đề bài:**")
        for kw in t3["stem_keywords"]:
            md.append(f"- Token `{kw['token']}` (Vị trí {kw['token_idx']}): Trọng số chú ý $\\beta_{{q}} = {kw['weight']:.4f}$")

        if t3["solution_steps"]:
            md.append("\n**Top Bước giải Then chốt:**")
            for st in t3["solution_steps"]:
                md.append(f"- Token `{st['token']}` (Vị trí {st['token_idx']}): Trọng số chú ý $\\beta_{{s}} = {st['weight']:.4f}$")

        if t3["top_patches"]:
            md.append("\n**Top Vùng ảnh Đề bài được Chú ý Chéo (Loại bỏ Viền Letterbox):**")
            for p in t3["top_patches"]:
                md.append(f"- Patch `{p['patch_idx']}` tại tọa độ lưới `{p['grid_pos']}`: Trọng số $A_{{cross}} = {p['weight']:.4f}$")
        md.append("")

        # Tier 4
        md.append("## Tầng 4: Truy Vết Lịch Sử Thực Chứng Nguyên Tử (Atomic Empirical History Trace)")
        md.append(f"Top {len(t4['top_interactions'])} tương tác quá khứ có ảnh hưởng nhân quả lớn nhất tới trạng thái tri thức tại bước {step}:")
        md.append("| Bước Cục bộ | Bước Toàn cục | Mã Câu hỏi | Phản hồi Thực tế | Độ khó Câu quá khứ | Trọng số Chú ý $\\alpha_{{t,\\tau}}^{{\\text{{ret}}}}$ |")
        md.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
        for hi in t4["top_interactions"]:
            resp_str = "ĐÚNG (1)" if hi["r"] == 1 else "SAI (0)"
            md.append(f"| {hi['local_step']} | {hi['global_step']} | Q_{hi['q_id']} | {resp_str} | {hi['mu_q']:.4f} | **{hi['alpha_weight']:.4f}** |")

        return "\n".join(md)

