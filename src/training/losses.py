"""
Loss Functions for Knowledge Tracing and Multimodal Knowledge Tracing.

Implements:
1. MaskedBCELoss: Numerically stable masked Binary Cross-Entropy with Logits (pyKT standard).
2. SolutionDistillationLoss (L_distill): MSE + InfoNCE contrastive distillation from Teacher sg(v_sol)
   to Student hat_v_sol on items with solutions (m_txt_s == 1).
3. CTTDifficultyCalibrationLoss (L_D): BCE loss calibrating estimated difficulty logit mu_q to
   empirical CTT difficulty labels D_q on reliable questions (ctt_valid_mask == True).
4. OrthogonalDisentanglementLoss (L_disen): Penalizes cosine similarity squared between concept
   representation c_C and item-specific variation d_q.
5. DeepIRTResidualRegLoss (L_res_reg): L2 regularization on DeepIRT residual logit ell_res.
6. ACDMMKTLoss: Multi-task objective orchestrating Stage 1 and Stage 2 curriculum optimization.
"""

from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedBCELoss(nn.Module):
    """
    Masked Binary Cross-Entropy Loss with Logits.
    Ensures numerical stability via log-sum-exp formulation and normalizes
    strictly by the number of valid active interactions in the batch.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute masked BCE loss.

        Args:
            logits: Unnormalized model prediction logits [B, L].
            targets: Binary ground-truth student correctness labels [B, L] in {0, 1}.
            mask: Binary validity mask [B, L] (e.g. loss_mask or eval_mask).

        Returns:
            Scalar loss tensor normalized by valid interaction count.
        """
        loss_unreduced = F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            reduction="none",
        )
        masked_loss = loss_unreduced * mask.float()
        total_valid = torch.clamp(mask.float().sum(), min=1.0)
        return masked_loss.sum() / total_valid


class SolutionDistillationLoss(nn.Module):
    """
    Multimodal Solution Distillation Loss (Formula 71 in docs/models/ACD-MMKT.md).
    Combines MSE feature reconstruction and InfoNCE contrastive distillation
    from Teacher sg(v_sol) to Student hat_v_sol.
    """

    def __init__(self, tau_temp: float = 0.10):
        super().__init__()
        self.tau_temp = tau_temp

    def forward(
        self,
        hat_v_sol: torch.Tensor,
        v_sol: torch.Tensor,
        m_txt_s: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hat_v_sol: Student predicted solution representations [N, d].
            v_sol: Teacher ground-truth solution representations [N, d].
            m_txt_s: Solution text presence flag [N, 1] or [N].

        Returns:
            Scalar distillation loss.
        """
        # Ensure float32 for numerical stability in cosine similarity and InfoNCE
        hat_v = hat_v_sol.float()
        # CRITICAL: Stop gradient on Teacher representation to prevent representation collapse
        teacher_v = v_sol.detach().float()

        mask = (m_txt_s.view(-1) > 0.5)
        num_valid = int(mask.sum().item())
        if num_valid == 0:
            return torch.tensor(0.0, device=hat_v.device, requires_grad=True)

        valid_hat = hat_v[mask]          # [M, d]
        valid_teacher = teacher_v[mask]  # [M, d]
        d = valid_hat.size(-1)

        # 1. Normalized Mean Squared Error (divided by dimension d to match BCE scale)
        mse_loss = F.mse_loss(valid_hat, valid_teacher, reduction="mean")

        # 2. InfoNCE Contrastive Distillation Loss
        if num_valid > 1:
            u_norm = F.normalize(valid_hat, p=2, dim=-1, eps=1e-8)       # [M, d]
            v_norm = F.normalize(valid_teacher, p=2, dim=-1, eps=1e-8)   # [M, d]

            # Similarity matrix [M, M]
            sim_matrix = torch.matmul(u_norm, v_norm.t()) / self.tau_temp
            # Targets: diagonal pairs (i, i) are positive pairs
            labels = torch.arange(num_valid, device=hat_v.device)
            infonce_loss = F.cross_entropy(sim_matrix, labels)
        else:
            infonce_loss = torch.tensor(0.0, device=hat_v.device)

        return mse_loss + infonce_loss


class CTTDifficultyCalibrationLoss(nn.Module):
    """
    CTT Difficulty Calibration Loss (Section 4 in docs/models/ACD-MMKT.md).
    Calibrates estimated difficulty logit ell_mu to empirical CTT error rate D_q.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logit_mu: torch.Tensor,
        d_ctt: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logit_mu: Predicted difficulty logits [N] or [B, L].
            d_ctt: Empirical CTT error rates in (0, 1) [N] or [B, L].
            valid_mask: Reliability mask (True if N_train >= 5) [N] or [B, L].

        Returns:
            Scalar calibration loss.
        """
        flat_logits = logit_mu.reshape(-1).float()
        flat_targets = d_ctt.reshape(-1).float()
        flat_mask = valid_mask.reshape(-1).bool()

        if not flat_mask.any():
            return torch.tensor(0.0, device=logit_mu.device, requires_grad=True)

        active_logits = flat_logits[flat_mask]
        active_targets = flat_targets[flat_mask]

        # BCE with logits against soft targets minus empirical entropy H(D_q) (Plan Step 6.2)
        bce = F.binary_cross_entropy_with_logits(
            active_logits, active_targets, reduction="none"
        )
        eps = 1e-7
        target_entropy = -(
            active_targets * torch.log(active_targets.clamp(min=eps))
            + (1.0 - active_targets) * torch.log((1.0 - active_targets).clamp(min=eps))
        )
        # Shifted so that optimal calibration loss is 0.0 when sigmoid(logit_mu) == d_ctt
        return torch.clamp(bce - target_entropy, min=0.0).mean()



class OrthogonalDisentanglementLoss(nn.Module):
    """
    Orthogonal Disentanglement Loss (Section 4 in docs/models/ACD-MMKT.md).
    Regularizes cosine similarity squared between core concept c_C and variation d_q.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        c_C: torch.Tensor,
        d_q: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            c_C: Concept representations [..., d].
            d_q: Item-specific variation representations [..., d].
            mask: Optional boolean or float validity mask [...] indicating active interactions.

        Returns:
            Scalar orthogonality loss.
        """
        if mask is not None:
            flat_mask = mask.reshape(-1).bool()
            if not flat_mask.any():
                return torch.tensor(0.0, device=c_C.device, requires_grad=True)
            c_flat = c_C.reshape(-1, c_C.size(-1))[flat_mask].float()
            d_flat = d_q.reshape(-1, d_q.size(-1))[flat_mask].float()
        else:
            c_flat = c_C.reshape(-1, c_C.size(-1)).float()
            d_flat = d_q.reshape(-1, d_q.size(-1)).float()

        cos_sim = F.cosine_similarity(c_flat, d_flat, dim=-1, eps=1e-8)
        return torch.mean(cos_sim ** 2)


class DeepIRTResidualRegLoss(nn.Module):
    """
    L2 Regularization on DeepIRT Residual Logit (Section 3.4 in docs/models/ACD-MMKT.md).
    Prevents deep residual branch from overpowering psychometric IRT formulation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, ell_res: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ell_res: Residual logits [B, L].
            mask: Validity mask [B, L].

        Returns:
            Scalar L2 regularization loss.
        """
        res_sq = (ell_res.float() ** 2) * mask.float()
        total_valid = torch.clamp(mask.float().sum(), min=1.0)
        return res_sq.sum() / total_valid


class ACDMMKTLoss(nn.Module):
    """
    Multi-Task Objective for ACD-MMKT Curriculum Training.
    Orchestrates Stage 1 (Item-Bank Pre-training) and Stage 2 (Sequential KT Fine-tuning).
    """

    def __init__(
        self,
        beta_distill: float = 0.10,
        beta_ctt: float = 0.20,
        beta_disen: float = 0.05,
        lambda_res: float = 1e-3,
        tau_temp: float = 0.10,
    ):
        super().__init__()
        self.beta_distill = beta_distill
        self.beta_ctt = beta_ctt
        self.beta_disen = beta_disen
        self.lambda_res = lambda_res

        self.bce_loss_fn = MaskedBCELoss()
        self.distill_loss_fn = SolutionDistillationLoss(tau_temp=tau_temp)
        self.ctt_loss_fn = CTTDifficultyCalibrationLoss()
        self.disen_loss_fn = OrthogonalDisentanglementLoss()
        self.res_reg_loss_fn = DeepIRTResidualRegLoss()

    def forward_stage1(
        self,
        item_outputs: Dict[str, torch.Tensor],
        d_ctt: torch.Tensor,
        ctt_valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Stage 1 Multi-Task Loss: Pre-train item representations on question bank.
        L_Stage1 = beta_distill * L_distill + beta_ctt * L_D + beta_disen * L_disen.
        """
        hat_v_sol = item_outputs["hat_v_sol"]
        v_sol = item_outputs["v_sol"]
        m_txt_s = item_outputs["m_txt_s"]
        logit_mu = item_outputs["logit_mu"]
        c_C = item_outputs["c_C"]
        d_q = item_outputs["d_q"]
        q_idx = item_outputs.get("item_indices")

        # Distillation loss
        if self.beta_distill > 0.0:
            l_distill = self.distill_loss_fn(hat_v_sol, v_sol, m_txt_s)
        else:
            l_distill = torch.tensor(0.0, device=hat_v_sol.device)

        # CTT difficulty calibration loss
        if self.beta_ctt > 0.0:
            if q_idx is not None:
                cur_d_ctt = d_ctt[q_idx]
                cur_valid = ctt_valid_mask[q_idx]
            else:
                cur_d_ctt = d_ctt
                cur_valid = ctt_valid_mask
            l_ctt = self.ctt_loss_fn(logit_mu, cur_d_ctt, cur_valid)
        else:
            l_ctt = torch.tensor(0.0, device=logit_mu.device)

        # Orthogonal disentanglement loss
        if self.beta_disen > 0.0:
            l_disen = self.disen_loss_fn(c_C, d_q)
        else:
            l_disen = torch.tensor(0.0, device=c_C.device)

        total_loss = (
            self.beta_distill * l_distill
            + self.beta_ctt * l_ctt
            + self.beta_disen * l_disen
        )

        metrics = {
            "loss_stage1_total": total_loss.item(),
            "loss_distill": l_distill.item(),
            "loss_ctt": l_ctt.item(),
            "loss_disen": l_disen.item(),
        }
        return total_loss, metrics

    def forward_stage2(
        self,
        seq_outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        loss_mask: torch.Tensor,
        d_ctt: torch.Tensor,
        ctt_valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Stage 2 Multi-Task Loss: End-to-end sequential Knowledge Tracing fine-tuning.
        L_Stage2 = L_BCE + beta_distill * L_distill + beta_ctt * L_D + beta_disen * L_disen + lambda_res * L_res.
        """
        logits = seq_outputs["logits"]
        ell_res = seq_outputs["ell_res"]
        logit_mu = seq_outputs["logit_mu"]
        c_C = seq_outputs["c_C"]
        d_q = seq_outputs["d_q"]

        # Primary Knowledge Tracing BCE loss
        l_bce = self.bce_loss_fn(logits, targets, loss_mask)

        # Distillation loss computed on unique items in batch
        if self.beta_distill > 0.0:
            uniq_hat = seq_outputs["uniq_hat_v_sol"]
            uniq_v = seq_outputs["uniq_v_sol"]
            uniq_m_txt = seq_outputs["uniq_m_txt_s"]
            l_distill = self.distill_loss_fn(uniq_hat, uniq_v, uniq_m_txt)
        else:
            l_distill = torch.tensor(0.0, device=logits.device)

        # CTT difficulty calibration loss
        if self.beta_ctt > 0.0:
            l_ctt = self.ctt_loss_fn(logit_mu, d_ctt, ctt_valid_mask & loss_mask)
        else:
            l_ctt = torch.tensor(0.0, device=logits.device)

        # Orthogonal disentanglement loss (masked by loss_mask to exclude padding tokens)
        if self.beta_disen > 0.0:
            l_disen = self.disen_loss_fn(c_C, d_q, mask=loss_mask)
        else:
            l_disen = torch.tensor(0.0, device=logits.device)

        # Residual branch L2 regularization
        if self.lambda_res > 0.0:
            l_res = self.res_reg_loss_fn(ell_res, loss_mask)
        else:
            l_res = torch.tensor(0.0, device=logits.device)

        total_loss = (
            l_bce
            + self.beta_distill * l_distill
            + self.beta_ctt * l_ctt
            + self.beta_disen * l_disen
            + self.lambda_res * l_res
        )


        metrics = {
            "loss_total": total_loss.item(),
            "loss_bce": l_bce.item(),
            "loss_distill": l_distill.item(),
            "loss_ctt": l_ctt.item(),
            "loss_disen": l_disen.item(),
            "loss_res_reg": l_res.item(),
        }
        return total_loss, metrics
