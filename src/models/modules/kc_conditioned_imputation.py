"""
KC-Conditioned Normalized Imputation and Structural Decoupling Module (Layer 1 of MG-MMKT v2).

Implements:
1. Structural Decoupling with Frequency-Gating and time-step Question ID Dropout:
   - Pools associated Knowledge Components (KCs) via masked average.
   - Computes confidence frequency gate between KC concept and question ID.
   - Applies Bernoulli Question ID Dropout per step (without scaling) during training.
   - Normalizes combined structural representation via LN_struct.
2. KC-Conditioned Normalized Imputation inside LayerNorm:
   - Linear projections from raw frozen feature spaces (text: 768, vision: 768) to d_model (128).
   - Learnable canonical missing tokens e_null initialized via N(0, I).
   - KC-conditioning transformation matrices W_null.
   - Strict inside-LayerNorm switching ensuring mean ~ 0 and L2 norm ~ sqrt(d).
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuralDecouplingLayer(nn.Module):
    """
    Decouples Question ID from general Knowledge Component concepts with frequency-gating
    and time-step Question ID Dropout.
    """

    def __init__(
        self,
        d_model: int = 128,
        p_id_drop: float = 0.20,
        use_structural_decoupling: bool = True,
    ):
        """
        Args:
            d_model: Latent dimension d (default: 128).
            p_id_drop: Probability of dropping question ID during training (default: 0.20).
            use_structural_decoupling: Whether to use frequency gate and ID dropout.
        """
        super().__init__()
        self.d_model = d_model
        self.p_id_drop = p_id_drop
        self.use_structural_decoupling = use_structural_decoupling

        # Frequency confidence gate: W_freq [e_C ; e_q] + b_freq -> (0, 1)^d
        self.w_freq = nn.Linear(2 * d_model, d_model)
        self.ln_struct = nn.LayerNorm(d_model)

    def forward(
        self,
        e_q: torch.Tensor,
        e_kcs: torch.Tensor,
        kc_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for structural decoupling.

        Args:
            e_q: Question embeddings [B, L, d].
            e_kcs: Knowledge Component embeddings [B, L, num_kcs_per_q, d].
            kc_mask: Validity mask for leaf KCs [B, L, num_kcs_per_q] (bool or float).

        Returns:
            Tuple of:
                - e_struct: Normalized structural representations [B, L, d].
                - e_C: General concept KC average embeddings [B, L, d].
        """
        # 1. Masked average pooling over associated knowledge components C_tau
        mask_expanded = kc_mask.unsqueeze(-1).float()  # [B, L, K, 1]
        sum_kcs = (e_kcs * mask_expanded).sum(dim=2)   # [B, L, d]
        count_kcs = torch.clamp(mask_expanded.sum(dim=2), min=1.0)  # [B, L, 1]
        e_C = sum_kcs / count_kcs  # [B, L, d]

        if not self.use_structural_decoupling:
            # Ablation mode: simple additive combination without frequency gate or ID dropout
            e_struct = self.ln_struct(e_C + e_q)
            return e_struct, e_C

        # 2. Frequency gating: g_freq = sigmoid(W_freq [e_C ; e_q] + b_freq)
        concat_cq = torch.cat([e_C, e_q], dim=-1)  # [B, L, 2d]
        g_freq = torch.sigmoid(self.w_freq(concat_cq))  # [B, L, d]

        # 3. Question ID Dropout: Bernoulli mask per time-step [B, L, 1] without 1/(1-p) scaling
        if self.training and self.p_id_drop > 0.0:
            bernoulli_prob = 1.0 - self.p_id_drop
            xi = torch.bernoulli(
                torch.full((e_q.size(0), e_q.size(1), 1), bernoulli_prob, device=e_q.device)
            )  # [B, L, 1] in {0.0, 1.0}
        else:
            xi = 1.0

        # 4. Synthesize structural representation and normalize
        # e_struct = LN_struct(e_C + xi * (g_freq * e_q))
        gated_id = g_freq * e_q
        e_struct_raw = e_C + xi * gated_id
        e_struct = self.ln_struct(e_struct_raw)

        return e_struct, e_C


class KCConditionedImputationLayer(nn.Module):
    """
    KC-Conditioned Normalized Imputation Module (Layer 1 of MG-MMKT v2).
    Linear projections from raw feature space, learnable base missing vectors,
    KC-conditioning matrices, and switching inside LayerNorm.
    """

    def __init__(
        self,
        d_raw_txt: int = 768,
        d_raw_img: int = 768,
        d_model: int = 128,
        use_kc_imputation: bool = True,
        use_visual: bool = True,
        use_solution: bool = True,
    ):
        """
        Args:
            d_raw_txt: Raw dimension of frozen text embeddings (default: 768).
            d_raw_img: Raw dimension of frozen vision embeddings (default: 768).
            d_model: Hidden dimension d (default: 128).
            use_kc_imputation: Whether to condition missing vectors on e_C.
            use_visual: Whether visual modalities are active.
            use_solution: Whether solution modalities are active.
        """
        super().__init__()
        self.d_raw_txt = d_raw_txt
        self.d_raw_img = d_raw_img
        self.d_model = d_model
        self.use_kc_imputation = use_kc_imputation
        self.use_visual = use_visual
        self.use_solution = use_solution

        # Linear projections for 4 branches
        self.proj_txt_q = nn.Linear(d_raw_txt, d_model)
        self.proj_img_q = nn.Linear(d_raw_img, d_model)
        self.proj_txt_s = nn.Linear(d_raw_txt, d_model)
        self.proj_img_s = nn.Linear(d_raw_img, d_model)

        # Learnable canonical base missing tokens initialized with N(0, 1)
        self.e_null_img_q = nn.Parameter(torch.randn(d_model))
        self.e_null_txt_s = nn.Parameter(torch.randn(d_model))
        self.e_null_img_s = nn.Parameter(torch.randn(d_model))

        # KC-conditioning transformation matrices (Xavier uniform)
        self.w_null_img_q = nn.Linear(d_model, d_model, bias=False)
        self.w_null_txt_s = nn.Linear(d_model, d_model, bias=False)
        self.w_null_img_s = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.w_null_img_q.weight)
        nn.init.xavier_uniform_(self.w_null_txt_s.weight)
        nn.init.xavier_uniform_(self.w_null_img_s.weight)

        # Branch-specific Layer Normalizations
        self.ln_txt_q = nn.LayerNorm(d_model)
        self.ln_img_q = nn.LayerNorm(d_model)
        self.ln_txt_s = nn.LayerNorm(d_model)
        self.ln_img_s = nn.LayerNorm(d_model)

    def forward(
        self,
        v_txt_q: torch.Tensor,
        v_img_q: torch.Tensor,
        v_txt_s: torch.Tensor,
        v_img_s: torch.Tensor,
        modality_masks: torch.Tensor,
        e_C: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for KC-Conditioned Normalized Imputation.

        Args:
            v_txt_q: Raw text embeddings of question [B, L, d_raw_txt].
            v_img_q: Raw image embeddings of question [B, L, d_raw_img].
            v_txt_s: Raw text embeddings of solution [B, L, d_raw_txt].
            v_img_s: Raw image embeddings of solution [B, L, d_raw_img].
            modality_masks: Flags [B, L, 3] for (m_img_q, m_txt_s, m_img_s).
            e_C: Mean KC embeddings [B, L, d_model].

        Returns:
            Tuple of normalized branch representations in R^d:
                (tilde_v_txt_q, tilde_v_img_q, tilde_v_txt_s, tilde_v_img_s)
        """
        # Extract presence flags
        m_img_q = modality_masks[..., 0:1].float()  # [B, L, 1]
        m_txt_s = modality_masks[..., 1:2].float()  # [B, L, 1]
        m_img_s = modality_masks[..., 2:3].float()  # [B, L, 1]

        # 1. Question text branch (always present 100%)
        proj_txt_q = self.proj_txt_q(v_txt_q)
        tilde_v_txt_q = self.ln_txt_q(proj_txt_q)

        # Compute missing conditional replacements
        if self.use_kc_imputation:
            cond_img_q = self.e_null_img_q + self.w_null_img_q(e_C)
            cond_txt_s = self.e_null_txt_s + self.w_null_txt_s(e_C)
            cond_img_s = self.e_null_img_s + self.w_null_img_s(e_C)
        else:
            cond_img_q = self.e_null_img_q.expand_as(proj_txt_q)
            cond_txt_s = self.e_null_txt_s.expand_as(proj_txt_q)
            cond_img_s = self.e_null_img_s.expand_as(proj_txt_q)

        # 2. Question image branch (with inside-LayerNorm switching)
        proj_img_q = self.proj_img_q(v_img_q)
        # If visual branch is disabled globally by ablation, force mask to 0
        if not self.use_visual:
            m_img_q = torch.zeros_like(m_img_q)
        comb_img_q = m_img_q * proj_img_q + (1.0 - m_img_q) * cond_img_q
        tilde_v_img_q = self.ln_img_q(comb_img_q)

        # 3. Solution text branch (with inside-LayerNorm switching)
        proj_txt_s = self.proj_txt_s(v_txt_s)
        if not self.use_solution:
            m_txt_s = torch.zeros_like(m_txt_s)
        comb_txt_s = m_txt_s * proj_txt_s + (1.0 - m_txt_s) * cond_txt_s
        tilde_v_txt_s = self.ln_txt_s(comb_txt_s)

        # 4. Solution image branch (with inside-LayerNorm switching)
        proj_img_s = self.proj_img_s(v_img_s)
        if not self.use_visual or not self.use_solution:
            m_img_s = torch.zeros_like(m_img_s)
        comb_img_s = m_img_s * proj_img_s + (1.0 - m_img_s) * cond_img_s
        tilde_v_img_s = self.ln_img_s(comb_img_s)

        return tilde_v_txt_q, tilde_v_img_q, tilde_v_txt_s, tilde_v_img_s

