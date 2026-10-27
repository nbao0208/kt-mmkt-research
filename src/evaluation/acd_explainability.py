"""
Three-Tier Ante-hoc Explainability Framework for ACD-MMKT.

Implements Section 3.4 of docs/models/ACD-MMKT.md and Step 5 & 8 of
plans/acd_mmkt_implementation_and_training_plan.md:
- Tier 1: Quantitative Psychometric Gap Evidence (theta_{u, t}, mu_{q, t}, a_{q, t}, Delta_psy).
- Tier 2: Multimodal & Temporal Attribution Evidence:
  - Cross-modal keyword-to-patch attention maps A_cross.
  - Contextual gate intensity lambda_q.
  - Algorithmic solution complexity shift Delta_mu_sol = mu_q - mu_stem_only.
  - Monotonic retrieval trace of top-influential historical steps with solution exposure status.
- Tier 3: Closed-Loop Evidence-Grounded Pedagogical Rationale Synthesis:
  - (1) Knowledge mastery diagnostic.
  - (2) Content & cognitive error analysis grounded in problem text, visual cues, and solution history.
  - (3) Actionable next-step personalized learning recommendations.
"""

from typing import Dict, Any, Optional, List, Tuple
import os
import json
import torch
import numpy as np


class ACDExplainabilityEngine:
    """
    Explainability Engine for ACD-MMKT.
    Extracts, summarizes, and generates three-tier pedagogical explanation reports.
    """

    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None):
        """
        Initialize ACDExplainabilityEngine.

        Args:
            model: Trained ACDMMKT model instance.
            device: Compute device.
        """
        self.model = model
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def explain_interaction(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int = 0,
        time_step: int = -1,
        question_text: str = "",
        top_k_history: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate complete three-tier explanation report for a single student interaction step.

        Args:
            batch: Model input batch dictionary.
            batch_idx: Sequence index within the batch (default: 0).
            time_step: Target sequence step to explain (default: -1 for last valid step).
            question_text: Optional raw text of the question.
            top_k_history: Number of top historical steps to retrieve.

        Returns:
            Dictionary containing Tier 1, Tier 2, and Tier 3 evidence reports.
        """
        self.model.eval()
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)

        outputs = self.model(batch, return_cross_weights=True)

        B, L = batch["q_seq"].shape
        b = batch_idx

        # If time_step < 0, pick the last active step in the sequence
        if time_step < 0:
            seq_mask = batch.get("seq_mask", torch.ones_like(batch["q_seq"], dtype=torch.bool))
            valid_steps = torch.where(seq_mask[b])[0]
            t = int(valid_steps[-1].item()) if len(valid_steps) > 0 else (L - 1)
        else:
            t = time_step

        q_id = int(batch["q_seq"][b, t].item())
        r_gt = int(batch["r_seq"][b, t].item()) if "r_seq" in batch else None
        p_hat = float(outputs["preds"][b, t].item())

        # --- Tier 1: Psychometrics ---
        theta = float(outputs["theta"][b, t].item())
        mu_q = float(outputs["mu_q"][b, t].item())
        a_q = float(outputs["a_q"][b, t].item())
        psy_gap = float(outputs["psy_gap"][b, t].item())

        if psy_gap > 0.15:
            gap_summary = (
                f"Student proficiency (theta={theta:.2f}) comfortably exceeds item difficulty "
                f"(mu={mu_q:.2f}) by +{psy_gap:.2f}. Expected outcome: High probability of correct answer."
            )
        elif psy_gap < -0.15:
            gap_summary = (
                f"Student proficiency (theta={theta:.2f}) falls significantly below item difficulty "
                f"(mu={mu_q:.2f}) by {psy_gap:.2f}. Expected outcome: Vulnerable to mistakes / high error risk."
            )
        else:
            gap_summary = (
                f"Student proficiency (theta={theta:.2f}) is closely matched with item difficulty "
                f"(mu={mu_q:.2f}) (gap={psy_gap:+.2f}). Item is within Zone of Proximal Development (ZPD)."
            )

        tier1 = {
            "prediction_prob": round(p_hat, 4),
            "ground_truth": r_gt,
            "student_proficiency_theta": round(theta, 4),
            "item_difficulty_mu": round(mu_q, 4),
            "item_discrimination_a": round(a_q, 4),
            "psychometric_gap": round(psy_gap, 4),
            "pedagogical_summary": gap_summary,
        }

        # --- Tier 2: Attribution ---
        lambda_val = float(outputs["lambda_q"][b, t].mean().item())
        has_image = bool(batch["modality_masks"][b, t, 0].item() > 0.5)

        # Calculate solution complexity shift Delta_mu_sol = mu_q - mu_stem_only
        v_stem = outputs["v_stem"][b:b+1, t:t+1]
        c_C = outputs["c_C"][b:b+1, t:t+1]
        mu_stem_only = float(
            self.model.subsystem2.estimate_difficulty_stem_only(v_stem, c_C).item()
        )
        delta_mu_sol = mu_q - mu_stem_only

        if delta_mu_sol > 0.08:
            sol_summary = (
                f"Item difficulty is substantially amplified (+{delta_mu_sol:.2f}) by its latent solution "
                f"complexity (multi-step deductive reasoning / auxiliary steps required)."
            )
        elif delta_mu_sol < -0.05:
            sol_summary = (
                f"Problem text/visual presentation appears challenging initially, but underlying "
                f"solution procedure is concise and direct ({delta_mu_sol:.2f})."
            )
        else:
            sol_summary = "Problem presentation difficulty is congruent with solution procedure demands."

        # History retrieval trace
        # alpha_ret[b, t] has length L: index 0 is BOS, index 1..t maps to past steps 0..t-1
        alpha_row = outputs["alpha_ret"][b, t, 1 : t + 1].cpu()
        top_k = min(top_k_history, len(alpha_row))
        history_trace = []

        if top_k > 0:
            top_vals, top_idx = torch.topk(alpha_row, k=top_k)
            c_start = int(batch.get("context_start", torch.zeros(B, dtype=torch.int64))[b].item())

            for score, past_step in zip(top_vals.tolist(), top_idx.tolist()):
                past_q = int(batch["q_seq"][b, past_step].item())
                past_r = int(batch["r_seq"][b, past_step].item())
                had_sol = bool(batch["modality_masks"][b, past_step, 1].item() > 0.5)

                history_trace.append({
                    "step": past_step,
                    "absolute_step": c_start + past_step,
                    "question_id": past_q,
                    "student_response": past_r,
                    "attention_weight": round(score, 4),
                    "exposed_to_solution": had_sol,
                })

        # Cross-modal keyword-to-patch visual alignments (Tier 2, Plan Step 5.1)
        top_alignments = []
        if has_image and outputs.get("A_cross_q") is not None and "uniq_q" in outputs:
            uniq_q_ref = outputs["uniq_q"].cpu()
            A_cross_ref = outputs["A_cross_q"].cpu()
            matches = (uniq_q_ref == q_id).nonzero(as_tuple=True)[0]
            if len(matches) > 0:
                a_mat = A_cross_ref[matches[0]].float()  # [L_q, P]
                # Filter out [PAD] tokens using question text mask
                # ensuring Top-K cross-modal alignments map strictly to valid words
                if hasattr(self.model, "acd_mask_txt_q") and q_id < len(self.model.acd_mask_txt_q):
                    txt_mask = self.model.acd_mask_txt_q[q_id].float().unsqueeze(-1).cpu()  # [L_q, 1]
                    a_mat = a_mat * txt_mask

                flat_a = a_mat.view(-1)
                valid_pairs = int((flat_a > 0.0).sum().item())
                k_pairs = min(5, valid_pairs) if valid_pairs > 0 else min(5, flat_a.size(0))
                if k_pairs > 0:
                    words = question_text.strip().split() if question_text else []
                    top_weights, top_flat_indices = torch.topk(flat_a, k=k_pairs)
                    for w, flat_idx in zip(top_weights.tolist(), top_flat_indices.tolist()):
                        t_idx = flat_idx // a_mat.size(1)
                        p_idx = flat_idx % a_mat.size(1)
                        entry = {
                            "token_index": int(t_idx),
                            "patch_index": int(p_idx),
                            "patch_coord": [int(p_idx // 7), int(p_idx % 7)],
                            "attention_weight": round(float(w), 4),
                        }
                        if 0 <= t_idx < len(words):
                            entry["token_word"] = words[t_idx]
                        top_alignments.append(entry)

        tier2 = {
            "has_visual_modality": has_image,
            "visual_gate_intensity_lambda": round(lambda_val, 4),
            "solution_complexity_shift_delta_mu": round(delta_mu_sol, 4),
            "solution_complexity_interpretation": sol_summary,
            "top_influential_history": history_trace,
            "top_token_patch_alignments": top_alignments,
        }


        # --- Tier 3: Closed-Loop Pedagogical Rationale ---
        rationale = self._synthesize_pedagogical_report(
            q_id=q_id,
            tier1=tier1,
            tier2=tier2,
            question_text=question_text,
        )

        return {
            "question_id": q_id,
            "step_index": t,
            "tier1_psychometrics": tier1,
            "tier2_attribution": tier2,
            "tier3_pedagogical_rationale": rationale,
        }

    def _synthesize_pedagogical_report(
        self,
        q_id: int,
        tier1: Dict[str, Any],
        tier2: Dict[str, Any],
        question_text: str = "",
    ) -> Dict[str, str]:
        """
        Synthesize three-part closed-loop pedagogical rationale.
        """
        p_hat = tier1["prediction_prob"]
        theta = tier1["student_proficiency_theta"]
        mu_q = tier1["item_difficulty_mu"]
        gap = tier1["psychometric_gap"]
        delta_sol = tier2["solution_complexity_shift_delta_mu"]
        history = tier2["top_influential_history"]

        # Part 1: Mastery Assessment
        if theta >= 0.70:
            part1 = f"Học sinh thể hiện năng lực tổng thể vững vàng trên nhóm kỹ năng liên quan (theta = {theta:.2f})."
        elif theta >= 0.45:
            part1 = f"Học sinh có sự hiểu biết cơ bản nhưng chưa hoàn toàn ổn định trên nhóm kỹ năng này (theta = {theta:.2f})."
        else:
            part1 = f"Học sinh có lỗ hổng kiến thức nền tảng đáng kể trên nhóm kỹ năng này (theta = {theta:.2f})."

        # Part 2: Cognitive Reasoning
        predicted_correct = (p_hat >= 0.5)
        outcome_str = "trả lời ĐÚNG" if predicted_correct else "trả lời SAI / gặp khó khăn"

        hist_evidence_str = ""
        if len(history) > 0:
            top_step = history[0]
            status_sol = "đã được đọc lời giải củng cố" if top_step["exposed_to_solution"] else "chưa có lời giải chi tiết"
            hist_evidence_str = (
                f" Lịch sử làm bài tại tương tác gần nhất liên quan (Câu hỏi #{top_step['question_id']}, "
                f"kết quả: {'Đúng' if top_step['student_response'] == 1 else 'Sai'}, {status_sol}) "
                f"có ảnh hưởng nhận thức lớn nhất (trọng số chú ý: {top_step['attention_weight']:.2f})."
            )

        img_str = ""
        if tier2["has_visual_modality"]:
            img_str = f" Câu hỏi có hình minh họa phụ trợ (độ dung hợp cổng thị giác lambda = {tier2['visual_gate_intensity_lambda']:.2f})."

        part2 = (
            f"Dự đoán học sinh sẽ {outcome_str} (xác suất làm đúng: {p_hat:.1%}). "
            f"Nguyên nhân: Độ chênh lệch giữa năng lực học sinh và độ khó bài toán là {gap:+.2f} "
            f"(độ khó bài toán mu = {mu_q:.2f}). {tier2['solution_complexity_interpretation']}{img_str}{hist_evidence_str}"
        )

        # Part 3: Actionable Pedagogical Recommendation
        if not predicted_correct:
            if delta_sol > 0.05:
                part3 = (
                    "Khuyến nghị: Cung cấp bài tập phân rã các bước giải phụ trợ ngầm định; hướng dẫn học sinh "
                    "cách lập dàn ý biến đổi công thức từng bước trước khi điền kết quả cuối cùng."
                )
            elif tier2["has_visual_modality"]:
                part3 = (
                    "Khuyến nghị: Rèn luyện kỹ năng đọc hiểu hình học và liên hệ ký hiệu văn bản với các vùng hình vẽ "
                    "thông qua bài tập có chú giải thị giác tương tác."
                )
            else:
                part3 = (
                    "Khuyến nghị: Củng cố lại định lý và khái niệm nền tảng với các ví dụ mức độ dễ (mu < 0.3) "
                    "trước khi thử sức lại với dạng câu hỏi tương tự."
                )
        else:
            part3 = (
                "Khuyến nghị: Học sinh đã làm chủ tốt dạng bài này. Có thể nâng cao độ thử thách bằng cách "
                "giao các câu hỏi có độ khó cao hơn (mu > 0.7) hoặc các bài toán đòi hỏi tích hợp nhiều kỹ năng."
            )

        return {
            "part1_knowledge_mastery": part1,
            "part2_cognitive_reasoning": part2,
            "part3_learning_recommendation": part3,
        }

    def export_report_markdown(self, explanation: Dict[str, Any], filepath: str) -> str:
        """
        Format explanation report as standard GitHub markdown and save to disk.
        """
        t1 = explanation["tier1_psychometrics"]
        t2 = explanation["tier2_attribution"]
        t3 = explanation["tier3_pedagogical_rationale"]

        md_content = f"""# Báo Cáo Chẩn Đoán Sư Phạm & Giải Thích 3 Cấp Độ (ACD-MMKT)

**Câu hỏi ID:** `{explanation['question_id']}` | **Bước chuỗi:** `{explanation['step_index']}`  
**Dự đoán xác suất đúng:** `{t1['prediction_prob']:.2%}` | **Nhãn thực tế:** `{t1['ground_truth']}`

---

## Cấp Độ 1: Bằng Chứng Tâm Trắc Học Định Lượng (Psychometric Gap Evidence)

| Chỉ số Tâm trắc học | Ký hiệu Toán | Giá trị Ước lượng | Thang đo |
| :--- | :---: | :---: | :---: |
| **Mức độ Năng lực Học sinh** | $\\theta_{{u, t}}$ | **{t1['student_proficiency_theta']:.4f}** | $(0, 1)$ |
| **Độ khó Đa phương thức Bài tập** | $\\mu_{{q, t}}$ | **{t1['item_difficulty_mu']:.4f}** | $(0, 1)$ |
| **Độ phân biệt của Câu hỏi** | $a_{{q, t}}$ | **{t1['item_discrimination_a']:.4f}** | $> 0$ |
| **Khoảng cách Năng lực - Độ khó** | $\\Delta_{{\\text{{psy}}}} = \\theta - \\mu$ | **{t1['psychometric_gap']:+.4f}** | $(-1, +1)$ |

> **Nhận định Sư phạm:** {t1['pedagogical_summary']}

---

## Cấp Độ 2: Bằng Chứng Đa Phương Thức & Truy Vết Lịch Sử (Multimodal & Temporal Attribution)

- **Hiện diện Hình ảnh Đề bài:** `{'Có' if t2['has_visual_modality'] else 'Không'}` (Cường độ cổng ngữ cảnh $\\lambda_t^q = {t2['visual_gate_intensity_lambda']:.4f}$)
- **Độ lệch Độ khó do Tải Giải thuật:** $\\Delta \\mu_{{\\text{{sol}}}} = {t2['solution_complexity_shift_delta_mu']:+.4f}$  
  *{t2['solution_complexity_interpretation']}*

### Các Tương Tác Lịch Sử Có Ảnh Hưởng Lớn Nhất (Top Attended History):

| Thứ tự | Bước Thời gian | Câu hỏi ID | Kết quả Quá khứ | Đã Học Lời Giải? | Trọng số Chú ý $\\alpha_{{t, \\tau}}^{{\\text{{ret}}}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
"""
        for i, h in enumerate(t2["top_influential_history"], 1):
            res_str = "Đúng" if h["student_response"] == 1 else "Sai"
            sol_str = "Có" if h["exposed_to_solution"] else "Không"
            md_content += f"| {i} | Bước {h['absolute_step']} | #{h['question_id']} | {res_str} | {sol_str} | **{h['attention_weight']:.4f}** |\n"

        alignments = t2.get("top_token_patch_alignments", [])
        if len(alignments) > 0:
            has_words = any("token_word" in a for a in alignments)
            if has_words:
                md_content += """
### Căn Chỉnh Chéo Từ Khóa — Vùng Ảnh ($A_{\\text{cross}, t}^q$):

| Thứ tự | Vị trí Token Đề ($L_q$) | Từ Khóa | Chỉ số Vùng Ảnh ($P$) | Tọa độ Lưới ($7 \\times 7$) | Trọng số Chú ý Chéo $A_{\\text{cross}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
"""
                for i, a in enumerate(alignments, 1):
                    row, col = a["patch_coord"]
                    word_str = f"`{a['token_word']}`" if "token_word" in a else "-"
                    md_content += f"| {i} | Token #{a['token_index']} | {word_str} | Patch #{a['patch_index']} | Hàng {row}, Cột {col} | **{a['attention_weight']:.4f}** |\n"
            else:
                md_content += """
### Căn Chỉnh Chéo Từ Khóa — Vùng Ảnh ($A_{\\text{cross}, t}^q$):

| Thứ tự | Vị trí Token Đề ($L_q$) | Chỉ số Vùng Ảnh ($P$) | Tọa độ Lưới ($7 \\times 7$) | Trọng số Chú ý Chéo $A_{\\text{cross}}$ |
| :---: | :---: | :---: | :---: | :---: |
"""
                for i, a in enumerate(alignments, 1):
                    row, col = a["patch_coord"]
                    md_content += f"| {i} | Token #{a['token_index']} | Patch #{a['patch_index']} | Hàng {row}, Cột {col} | **{a['attention_weight']:.4f}** |\n"
        elif t2["has_visual_modality"]:
            md_content += "\n*Câu hỏi có ảnh nhưng không có trọng số chú ý chéo nổi bật.*\n"

        md_content += f"""
---


## Cấp Độ 3: Báo Cáo Giải Thích Ngôn Ngữ Tự Nhiên Đóng Vòng Lặp (Evidence-Grounded Rationale)

### 1. Đánh Giá Mức Độ Nắm Vững Kiến Thức:
{t3['part1_knowledge_mastery']}

### 2. Lý Giải Nguyên Nhân Dự Đoán Nhận Thức:
{t3['part2_cognitive_reasoning']}

### 3. Khuyến Nghị Lộ Trình Sư Phạm Tiếp Theo:
{t3['part3_learning_recommendation']}
"""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md_content)

        return md_content
