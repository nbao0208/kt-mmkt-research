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
            logits.float(),
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


class PrivilegedSolutionDistillationLoss(nn.Module):
    """
    Privileged Solution Distillation Loss for MG-MMKT v2 (Formula in docs/models/MG-MMKTv2.md Section 5.2).
    Combines MSE feature reconstruction and Cosine distance between Student hat_z_s and Teacher sg(z_s_clean).
    Active on valid interactions where a solution exists (m_sol_t == 1).
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        hat_z_s: torch.Tensor,
        z_s_clean: torch.Tensor,
        m_sol: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hat_z_s: Student estimated cognitive demand [B, L, d].
            z_s_clean: Teacher clean solution representations [B, L, d].
            m_sol: Binary indicator of solution existence and sequence validity [B, L].

        Returns:
            Scalar distillation loss.
        """
        # Ensure float32 for stable distance computation
        hat_z = hat_z_s.float()
        # CRITICAL: Stop gradient on Teacher representation to isolate autograd flow
        teacher_z = z_s_clean.detach().float()

        mask = m_sol.bool()
        num_valid = mask.sum().float()

        if num_valid.item() == 0:
            return torch.tensor(0.0, device=hat_z.device, requires_grad=True)

        valid_hat = hat_z[mask]          # [M, d]
        valid_teacher = teacher_z[mask]  # [M, d]
        d = valid_hat.size(-1)

        # 1. Normalized Mean Squared Error (divided by dimension d)
        mse = torch.mean((valid_hat - valid_teacher) ** 2) / d

        # 2. Cosine Distance: 1 - cos(hat_z, teacher_z)
        hat_norm = F.normalize(valid_hat, p=2, dim=-1, eps=self.eps)
        teacher_norm = F.normalize(valid_teacher, p=2, dim=-1, eps=self.eps)
        cos_sim = (hat_norm * teacher_norm).sum(dim=-1)
        cos_dist = torch.mean(1.0 - cos_sim)

        return mse + cos_dist


class ModalitySuppressionRegularizationLoss(nn.Module):
    """
    Modality Suppression Regularization Loss for MG-MMKT v2 (Section 5.3).
    Penalizes non-zero gate weights on missing visual modalities (alpha_img_q and beta_img_s).
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        alpha_q: torch.Tensor,
        beta_s: torch.Tensor,
        m_img_q: torch.Tensor,
        m_img_s: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            alpha_q: Question gate weights [B, L, 3] (id, txt, img).
            beta_s: Solution gate weights [B, L, 2] (txt, img).
            m_img_q: Question image presence flags [B, L, 1] or [B, L].
            m_img_s: Solution image presence flags [B, L, 1] or [B, L].
            seq_mask: Interaction sequence validity mask [B, L].

        Returns:
            Scalar gate regularization loss.
        """
        alpha_img = alpha_q[..., 2].float()  # [B, L]
        beta_img = beta_s[..., 1].float()    # [B, L]
        m_q = m_img_q.view_as(alpha_img).float()
        m_s = m_img_s.view_as(beta_img).float()
        valid = seq_mask.float()

        # Penalize squared gate weight when modality is missing (1 - m)
        q_penalty = (1.0 - m_q) * (alpha_img ** 2)
        s_penalty = (1.0 - m_s) * (beta_img ** 2)

        total_penalty = (q_penalty + s_penalty) * valid
        total_valid = torch.clamp(valid.sum(), min=1.0)
        return total_penalty.sum() / total_valid


class MGMMKTv2Loss(nn.Module):
    """
    End-to-End Multi-Task Loss for MG-MMKT v2:
    L_total = L_BCE + lambda_priv * L_priv + lambda_gate * L_gate
    """

    def __init__(
        self,
        lambda_priv: float = 0.10,
        lambda_gate: float = 0.05,
    ):
        super().__init__()
        self.lambda_priv = lambda_priv
        self.lambda_gate = lambda_gate

        self.bce_loss_fn = MaskedBCELoss()
        self.priv_loss_fn = PrivilegedSolutionDistillationLoss()
        self.gate_reg_fn = ModalitySuppressionRegularizationLoss()

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        loss_mask: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total multi-task loss and component metrics.

        Args:
            model_outputs: Dictionary returned by MGMMKTv2.forward().
            targets: Binary ground-truth student correctness labels [B, L].
            loss_mask: pyKT loss mask (step 0 excluded) [B, L].
            seq_mask: Sequence validity mask (all active steps) [B, L].

        Returns:
            Tuple of:
                - total_loss: Scalar multi-task loss.
                - metrics: Dictionary of individual loss components.
        """
        logits = model_outputs["logits"]
        hat_z_s = model_outputs.get("hat_z_s")
        z_s_clean = model_outputs.get("z_s_clean")
        alpha_q = model_outputs["alpha_q"]
        beta_s = model_outputs["beta_s"]
        raw_mod_masks = model_outputs.get("raw_modality_masks")
        dyn_mod_masks = model_outputs.get("modality_masks")

        # 1. Primary Knowledge Tracing BCE loss (evaluated on loss_mask)
        l_bce = self.bce_loss_fn(logits, targets, loss_mask)

        # 2. Privileged Solution Distillation loss (LUPI)
        if self.lambda_priv > 0.0 and hat_z_s is not None and z_s_clean is not None and raw_mod_masks is not None:
            # Solution existence indicator: max(m_txt_s^raw, m_img_s^raw) * seq_mask
            m_sol_raw = torch.max(raw_mod_masks[..., 1], raw_mod_masks[..., 2])
            m_sol_valid = (m_sol_raw > 0.5) & seq_mask.bool()
            l_priv = self.priv_loss_fn(hat_z_s, z_s_clean, m_sol_valid)
        else:
            l_priv = torch.tensor(0.0, device=logits.device)

        # 3. Modality Suppression Regularization loss
        if self.lambda_gate > 0.0 and dyn_mod_masks is not None:
            m_img_q = dyn_mod_masks[..., 0]
            m_img_s = dyn_mod_masks[..., 2]
            l_gate = self.gate_reg_fn(alpha_q, beta_s, m_img_q, m_img_s, seq_mask)
        else:
            l_gate = torch.tensor(0.0, device=logits.device)

        total_loss = l_bce + self.lambda_priv * l_priv + self.lambda_gate * l_gate

        metrics = {
            "loss_total": total_loss.item(),
            "loss_bce": l_bce.item(),
            "loss_priv": l_priv.item(),
            "loss_gate": l_gate.item(),
        }
        return total_loss, metrics


class BayesianWeightedDifficultyLoss(nn.Module):
    """
    Bayesian Sample-Size Weighted Difficulty Calibration Loss (L_D^weighted).
    Implements Section 4.1 in docs/models/ACD-MMKTv2.md:
        omega_q = N_q / (N_q + kappa)
        L_D^weighted = (1 / |Q|) sum_{q in Q} omega_q * (BCE_logits(tilde_mu_q, D_q) - H(D_q))
    Normalized so that the optimal loss is 0.0 when sigmoid(tilde_mu_q) == D_q.
    """

    def __init__(self, kappa: float = 10.0):
        super().__init__()
        self.kappa = kappa

    def forward(
        self,
        logit_mu: torch.Tensor,
        d_err: torch.Tensor,
        n_obs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted difficulty calibration loss.

        Args:
            logit_mu: Model predicted difficulty logits [M].
            d_err: Empirical CTT error rates in [0, 1] [M].
            n_obs: Interaction attempt sample sizes N_q >= 0 [M].

        Returns:
            Scalar weighted calibration loss.
        """
        flat_logits = logit_mu.reshape(-1).float()
        flat_targets = d_err.reshape(-1).float()
        flat_n = n_obs.reshape(-1).float()

        if len(flat_logits) == 0:
            return torch.tensor(0.0, device=logit_mu.device, requires_grad=True)

        # Bayesian confidence weight omega_q = N_q / (N_q + kappa)
        omega = flat_n / (flat_n + self.kappa)  # [M] in [0, 1)

        # Numerically stable BCE with logits against soft empirical targets
        bce = F.binary_cross_entropy_with_logits(
            flat_logits, flat_targets, reduction="none"
        )

        # Target entropy H(D_q) with safe clamping
        eps = 1e-7
        p = flat_targets.clamp(min=eps, max=1.0 - eps)
        target_entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))

        # Shifted so that optimal loss is 0.0 and clamp to non-negative
        shifted_loss = torch.clamp(bce - target_entropy, min=0.0)
        weighted_loss = omega * shifted_loss

        return weighted_loss.mean()


class ACDMMKTv2Loss(nn.Module):
    """
    Multi-Task Loss Objective for ACD-MMKTv2.
    Combines:
    1. Primary pyKT sequential prediction BCE loss (L_BCE) on active interactions (loss_mask).
    2. Bayesian sample-size weighted difficulty calibration loss (L_D^weighted) on unique items.
    L_ACD = L_BCE + beta_D * L_D^weighted.
    """

    def __init__(self, beta_d: float = 0.20, kappa: float = 10.0):
        """
        Initialize ACDMMKTv2Loss.

        Args:
            beta_d: Weight for difficulty calibration loss (default: 0.20).
            kappa: Bayesian sample size smoothing constant (default: 10.0).
        """
        super().__init__()
        self.beta_d = beta_d
        self.bce_loss_fn = MaskedBCELoss()
        self.diff_loss_fn = BayesianWeightedDifficultyLoss(kappa=kappa)

    def forward_stage1(
        self,
        tilde_mu: torch.Tensor,
        d_err: torch.Tensor,
        n_obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Stage 1 loss: Pure item-bank pre-training on L_D^weighted.
        """
        diff_loss = self.diff_loss_fn(tilde_mu, d_err, n_obs)
        total_loss = self.beta_d * diff_loss

        metrics = {
            "loss_total": total_loss.item(),
            "loss_diff": diff_loss.item(),
        }
        return total_loss, metrics

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        loss_mask: torch.Tensor,
        n_q_all: torch.Tensor,
        d_q_all: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Stage 2 multi-task loss.

        Args:
            model_outputs: Output dictionary from ACDMMKTv2.forward().
            targets: Binary student response targets [B, L].
            loss_mask: Evaluation / loss mask (step 0 excluded) [B, L].
            n_q_all: Full fold sample size array [7652].
            d_q_all: Full fold empirical error rate array [7652].

        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        logits = model_outputs["logits"]

        # 1. Primary Knowledge Tracing BCE loss
        l_bce = self.bce_loss_fn(logits, targets, loss_mask)

        # 2. Bayesian weighted difficulty calibration loss on unique active questions
        active_q = model_outputs.get("active_unique_q")
        active_tilde_mu = model_outputs.get("active_unique_tilde_mu")

        if active_q is not None and len(active_q) > 0 and active_tilde_mu is not None and self.beta_d > 0.0:
            target_n = n_q_all[active_q]
            target_d = d_q_all[active_q]
            l_diff = self.diff_loss_fn(active_tilde_mu, target_d, target_n)
        else:
            l_diff = torch.tensor(0.0, device=logits.device)

        total_loss = l_bce + self.beta_d * l_diff

        metrics = {
            "loss_total": total_loss.item(),
            "loss_bce": l_bce.item(),
            "loss_diff": l_diff.item(),
        }
        return total_loss, metrics


class ACDMMKTv3Loss(nn.Module):
    """
    Multi-Task Loss Objective for ACD-MMKTv3.
    Combines:
    1. Primary pyKT sequential prediction BCE loss (L_BCE) on active interactions (loss_mask).
    2. Bayesian sample-size weighted difficulty calibration loss (L_D^weighted) on unique active questions (using clean modalities).
    3. Sparsity shrinkage penalty lambda_sparse * ((Delta_mu_stem)^2 + (Delta_mu_sol)^2) on unique active questions.
    L_ACD = L_BCE + beta_D * L_D^weighted + lambda_sparse * L_sparse.
    """

    def __init__(
        self,
        beta_d: float = 0.20,
        lambda_sparse: float = 0.01,
        kappa: float = 10.0,
    ):
        """
        Initialize ACDMMKTv3Loss.

        Args:
            beta_d: Weight for difficulty calibration loss (default: 0.20).
            lambda_sparse: Sparsity shrinkage penalty coefficient (default: 0.01).
            kappa: Bayesian sample size smoothing constant (default: 10.0).
        """
        super().__init__()
        self.beta_d = beta_d
        self.lambda_sparse = lambda_sparse
        self.bce_loss_fn = MaskedBCELoss()
        self.diff_loss_fn = BayesianWeightedDifficultyLoss(kappa=kappa)

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        loss_mask: torch.Tensor,
        n_q_all: torch.Tensor,
        d_q_all: torch.Tensor,
        unanchored_concept: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total ACD-MMKTv3 loss and metrics.

        Args:
            model_outputs: Output dictionary from ACDMMKTv3.forward().
            targets: Ground-truth student correctness labels [B, L].
            loss_mask: Valid interaction mask excluding t = 0 [B, L].
            n_q_all: Fold sample sizes array [7652].
            d_q_all: Fold empirical CTT error rates [7652].
            unanchored_concept: If True (Ablation 4), disables sparsity regularization.

        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        logits = model_outputs["logits"]

        # 1. Primary Knowledge Tracing BCE loss
        l_bce = self.bce_loss_fn(logits, targets, loss_mask)

        # 2. Extract unique active questions and clean item properties
        clean_props = model_outputs.get("clean_item_props")
        unique_q = model_outputs.get("unique_q")

        if clean_props is not None and unique_q is not None and len(unique_q) > 0:
            clean_tilde_mu = clean_props["tilde_mu"]
            clean_delta_stem = clean_props["delta_mu_stem"]
            clean_delta_sol = clean_props["delta_mu_sol"]

            target_n = n_q_all[unique_q].to(logits.device)
            target_d = d_q_all[unique_q].to(logits.device)

            if self.beta_d > 0.0:
                l_diff = self.diff_loss_fn(clean_tilde_mu, target_d, target_n)
            else:
                l_diff = torch.tensor(0.0, device=logits.device)

            if self.lambda_sparse > 0.0 and not unanchored_concept:
                l_sparse = (clean_delta_stem.pow(2) + clean_delta_sol.pow(2)).mean()
            else:
                l_sparse = torch.tensor(0.0, device=logits.device)
        else:
            l_diff = torch.tensor(0.0, device=logits.device)
            l_sparse = torch.tensor(0.0, device=logits.device)

        total_loss = l_bce + self.beta_d * l_diff + self.lambda_sparse * l_sparse

        metrics = {
            "loss_total": total_loss.item(),
            "loss_bce": l_bce.item(),
            "loss_diff": l_diff.item(),
            "loss_sparse": l_sparse.item(),
        }
        return total_loss, metrics


