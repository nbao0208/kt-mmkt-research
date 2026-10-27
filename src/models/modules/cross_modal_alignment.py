"""
Fine-Grained Cross-Modal Alignment and Context-Aware Hierarchical Gating (Subsystem 1).

Implements Section 3.1 of docs/models/ACD-MMKT.md and Step 2 of
plans/acd_mmkt_implementation_and_training_plan.md:
- Linear projection of token/patch embeddings with LayerNorm.
- Learnable canonical missing modality tokens (E_null_img_q, E_null_img_s, e_null_sol).
- Intra-modal self-attention for text tokens and visual patches with padding mask support.
- Text-guided cross-modal attention generating fine-grained alignment matrix A_cross [L_q, P].
- Masked attention pooling over token length to produce u_txt and u_cross.
- Context-conditioned hierarchical gate conditioned on dynamic learner context c_ctx and
  locked by visual presence mask m_img.
- Solution cross-modal alignment branch with solution presence masking.
- Unique-Question Batch Indexing optimization for high GPU/MPS memory efficiency.
"""

from typing import Dict, Any, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedAttentionPooling(nn.Module):
    """
    Attention-based pooling over variable-length sequence dimension with key masking.
    Computes a learned weighted sum of token representations.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.score_proj = nn.Linear(d_model, 1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Sequence tensor [N, L, d_model].
            mask: Optional boolean validity mask [N, L] (True=valid token, False=PAD).

        Returns:
            Tuple of (pooled representation [N, d_model], attention weights [N, L]).
        """
        # [N, L, d_model] -> [N, L, 1] -> [N, L]
        scores = self.score_proj(torch.tanh(self.proj(x))).squeeze(-1)

        if mask is not None:
            # Apply padding mask: mask False positions with safe large negative value
            scores = scores.masked_fill(~mask, -1e4)

        attn_weights = F.softmax(scores, dim=-1)  # [N, L]
        # In case all tokens in a row were masked out, avoid NaN by replacing zero-sum with uniform
        if mask is not None:
            invalid_rows = (~mask).all(dim=-1, keepdim=True)  # [N, 1]
            attn_weights = torch.where(
                invalid_rows,
                torch.full_like(attn_weights, 1.0 / max(1, x.size(1))),
                attn_weights,
            )

        pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # [N, d_model]
        return pooled, attn_weights


class FineGrainedCrossModalAlignment(nn.Module):
    """
    Subsystem 1: Fine-Grained Cross-Modal Alignment and Context-Aware Gating.
    Aligns text tokens with visual patches and adaptively fuses multimodal representations.
    """

    def __init__(
        self,
        d_raw_txt: int = 768,
        d_raw_img: int = 768,
        d_ctx: int = 21,
        d_model: int = 128,
        num_heads: int = 4,
        patch_count: int = 49,
        dropout: float = 0.20,
        use_visual: bool = True,
        use_solution: bool = True,
        use_fine_grained: bool = True,
    ):
        """
        Initialize FineGrainedCrossModalAlignment module.

        Args:
            d_raw_txt: Dimension of raw frozen text embeddings (default: 768).
            d_raw_img: Dimension of raw frozen vision patch embeddings (default: 768).
            d_ctx: Dimension of dynamic student/question context vector (default: 21).
            d_model: Latent dimension d (default: 128).
            num_heads: Number of attention heads (default: 4).
            patch_count: Number of spatial vision patches P (default: 49 for 7x7 grid).
            dropout: Dropout probability.
            use_visual: Whether visual modalities are enabled.
            use_solution: Whether solution modalities are enabled.
            use_fine_grained: Whether fine-grained token/patch cross-attention is enabled
                              (if False, Ablation 3 uses global vector pooling).
        """
        super().__init__()
        self.d_raw_txt = d_raw_txt
        self.d_raw_img = d_raw_img
        self.d_ctx = d_ctx
        self.d_model = d_model
        self.num_heads = num_heads
        self.patch_count = patch_count
        self.dropout_rate = dropout
        self.use_visual = use_visual
        self.use_solution = use_solution
        self.use_fine_grained = use_fine_grained

        # 1. Projections and LayerNorms for Question Modalities
        self.proj_txt_q = nn.Linear(d_raw_txt, d_model)
        self.ln_txt_q = nn.LayerNorm(d_model)

        self.proj_img_q = nn.Linear(d_raw_img, d_model)
        self.ln_img_q = nn.LayerNorm(d_model)

        # 2. Projections and LayerNorms for Solution Modalities
        self.proj_txt_s = nn.Linear(d_raw_txt, d_model)
        self.ln_txt_s = nn.LayerNorm(d_model)

        self.proj_img_s = nn.Linear(d_raw_img, d_model)
        self.ln_img_s = nn.LayerNorm(d_model)

        # Context vector projection
        self.proj_ctx = nn.Linear(d_ctx, d_model)
        self.ln_ctx = nn.LayerNorm(d_model)

        # 3. Canonical Learnable Missing-Modality Tokens (Standard Normal N(0, 1) per Plan Step 2.2)
        self.e_null_img_q = nn.Parameter(torch.randn(patch_count, d_model))
        self.e_null_img_s = nn.Parameter(torch.randn(patch_count, d_model))
        self.e_null_sol = nn.Parameter(torch.randn(d_model))


        if self.use_fine_grained:
            # 4. Intra-Modal Self-Attention for Question
            self.intra_txt_q = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            self.ln_intra_txt_q = nn.LayerNorm(d_model)

            self.intra_img_q = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            self.ln_intra_img_q = nn.LayerNorm(d_model)

            # 5. Text-Guided Cross-Modal Attention for Question (Text is Q, Image is K, V)
            self.cross_attn_q = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            self.ln_cross_q = nn.LayerNorm(d_model)

            # Masked Pooling over Token Length
            self.pool_txt_q = MaskedAttentionPooling(d_model)
            self.pool_cross_q = MaskedAttentionPooling(d_model)

            # 6. Intra-Modal and Cross-Modal Attention for Solution
            self.intra_txt_s = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            self.ln_intra_txt_s = nn.LayerNorm(d_model)

            self.intra_img_s = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            self.ln_intra_img_s = nn.LayerNorm(d_model)

            self.cross_attn_s = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            self.ln_cross_s = nn.LayerNorm(d_model)

            self.pool_txt_s = MaskedAttentionPooling(d_model)
            self.pool_cross_s = MaskedAttentionPooling(d_model)

        # 7. Context-Aware Hierarchical Gates
        # Question Gate conditioned on: [u_txt_q; u_cross_q; c_ctx] -> [3 * d_model]
        self.w_ctx_q = nn.Linear(3 * d_model, d_model)

        # Solution Gate conditioned on: [u_txt_s; u_cross_s] -> [2 * d_model]
        self.w_ctx_s = nn.Linear(2 * d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def process_unique_items(
        self,
        E_txt_q: torch.Tensor,
        mask_txt_q: torch.Tensor,
        E_img_q: torch.Tensor,
        E_txt_s: torch.Tensor,
        mask_txt_s: torch.Tensor,
        E_img_s: torch.Tensor,
        m_img_q: torch.Tensor,
        m_img_s: torch.Tensor,
        mg_v_txt_q: Optional[torch.Tensor] = None,
        mg_v_img_q: Optional[torch.Tensor] = None,
        mg_v_txt_s: Optional[torch.Tensor] = None,
        mg_v_img_s: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Process static item-level features on unique question indices.
        Computes intra/cross-modal representations independent of sequence history.

        Args:
            E_txt_q: Raw question text tokens [N, L_q, d_raw_txt].
            mask_txt_q: Question text padding mask [N, L_q].
            E_img_q: Raw question image patches [N, P, d_raw_img].
            E_txt_s: Raw solution text tokens [N, L_s, d_raw_txt].
            mask_txt_s: Solution text padding mask [N, L_s].
            E_img_s: Raw solution image patches [N, P, d_raw_img].
            m_img_q: Question image presence flag [N, 1].
            m_img_s: Solution image presence flag [N, 1].
            mg_v_txt_q: Optional pre-extracted global question text vectors [N, d_raw_txt].
            mg_v_img_q: Optional pre-extracted global question image vectors [N, d_raw_img].
            mg_v_txt_s: Optional pre-extracted global solution text vectors [N, d_raw_txt].
            mg_v_img_s: Optional pre-extracted global solution image vectors [N, d_raw_img].

        Returns:
            Dictionary containing item-level pooled vectors:
            - 'u_txt_q': [N, d]
            - 'u_cross_q': [N, d]
            - 'u_txt_s': [N, d]
            - 'u_cross_s': [N, d]
            - 'A_cross_q': [N, L_q, P] fine-grained cross-attention weights.
        """
        N = E_txt_q.size(0)

        # Ensure float32 for computation stability
        E_txt_q = self.ln_txt_q(self.proj_txt_q(E_txt_q.float()))  # [N, L_q, d]
        E_txt_s = self.ln_txt_s(self.proj_txt_s(E_txt_s.float()))  # [N, L_s, d]

        # Project and impute missing images
        H_img_q = self.ln_img_q(self.proj_img_q(E_img_q.float()))  # [N, P, d]
        H_img_s = self.ln_img_s(self.proj_img_s(E_img_s.float()))  # [N, P, d]

        m_q_expand = m_img_q.unsqueeze(-1).float()  # [N, 1, 1]
        m_s_expand = m_img_s.unsqueeze(-1).float()  # [N, 1, 1]

        E_null_q = self.e_null_img_q.unsqueeze(0).expand(N, -1, -1)  # [N, P, d]
        E_null_s = self.e_null_img_s.unsqueeze(0).expand(N, -1, -1)  # [N, P, d]

        if self.use_visual:
            if self.training:
                # Visual Modality Dropout (p=0.15) on question and solution images:
                # Allows gradient flow to canonical learnable tokens e_null_img_q and e_null_img_s (Plan Step 2.3).
                drop_vis_q = (torch.rand_like(m_q_expand) < 0.15)
                eff_m_q = m_q_expand * (~drop_vis_q).float()
                cur_E_img_q = eff_m_q * H_img_q + (1.0 - eff_m_q) * E_null_q

                drop_vis_s = (torch.rand_like(m_s_expand) < 0.15)
                eff_m_s = m_s_expand * (~drop_vis_s).float()
                cur_E_img_s = eff_m_s * H_img_s + (1.0 - eff_m_s) * E_null_s
            else:
                cur_E_img_q = m_q_expand * H_img_q + (1.0 - m_q_expand) * E_null_q
                cur_E_img_s = m_s_expand * H_img_s + (1.0 - m_s_expand) * E_null_s
        else:
            cur_E_img_q = E_null_q
            cur_E_img_s = E_null_s

        if not self.use_fine_grained:
            # Ablation 3 (Plan Step 8.1): Global average pooling or direct use of MG-MMKT global vectors
            if mg_v_txt_q is not None and mg_v_img_q is not None:
                u_txt_q = self.ln_txt_q(self.proj_txt_q(mg_v_txt_q.float()))
                u_cross_q = self.ln_img_q(self.proj_img_q(mg_v_img_q.float()))
                if mg_v_txt_s is not None and mg_v_img_s is not None:
                    u_txt_s = self.ln_txt_s(self.proj_txt_s(mg_v_txt_s.float()))
                    u_cross_s = self.ln_img_s(self.proj_img_s(mg_v_img_s.float()))
                else:
                    u_txt_s = (E_txt_s * mask_txt_s.unsqueeze(-1).float()).sum(dim=1) / torch.clamp(
                        mask_txt_s.unsqueeze(-1).float().sum(dim=1), min=1.0
                    )
                    u_cross_s = cur_E_img_s.mean(dim=1)
            else:
                u_txt_q = (E_txt_q * mask_txt_q.unsqueeze(-1).float()).sum(dim=1) / torch.clamp(
                    mask_txt_q.unsqueeze(-1).float().sum(dim=1), min=1.0
                )
                u_cross_q = cur_E_img_q.mean(dim=1)
                u_txt_s = (E_txt_s * mask_txt_s.unsqueeze(-1).float()).sum(dim=1) / torch.clamp(
                    mask_txt_s.unsqueeze(-1).float().sum(dim=1), min=1.0
                )
                u_cross_s = cur_E_img_s.mean(dim=1)
            A_cross_q = torch.zeros(
                (N, E_txt_q.size(1), cur_E_img_q.size(1)),
                device=E_txt_q.device,
                dtype=E_txt_q.dtype,
            )
            return {
                "u_txt_q": u_txt_q,
                "u_cross_q": u_cross_q,
                "u_txt_s": u_txt_s,
                "u_cross_s": u_cross_s,
                "A_cross_q": A_cross_q,
            }


        # --- Question Processing ---
        # 1. Intra-Modal Self-Attention
        # Key padding mask: True for positions that should be ignored
        txt_q_pad_mask = ~mask_txt_q  # [N, L_q]
        txt_q_intra, _ = self.intra_txt_q(
            E_txt_q, E_txt_q, E_txt_q, key_padding_mask=txt_q_pad_mask
        )
        hat_E_txt_q = self.ln_intra_txt_q(E_txt_q + self.dropout(txt_q_intra))

        img_q_intra, _ = self.intra_img_q(cur_E_img_q, cur_E_img_q, cur_E_img_q)
        hat_E_img_q = self.ln_intra_img_q(cur_E_img_q + self.dropout(img_q_intra))

        # 2. Text-Guided Cross-Modal Attention (Q = text, K, V = image)
        C_cross_q, A_cross_q = self.cross_attn_q(
            query=hat_E_txt_q,
            key=hat_E_img_q,
            value=hat_E_img_q,
            need_weights=True,
            average_attn_weights=True,
        )
        hat_C_cross_q = self.ln_cross_q(hat_E_txt_q + self.dropout(C_cross_q))

        # 3. Masked Attention Pooling
        u_txt_q, _ = self.pool_txt_q(hat_E_txt_q, mask=mask_txt_q)
        u_cross_q, _ = self.pool_cross_q(hat_C_cross_q, mask=mask_txt_q)

        # --- Solution Processing ---
        # 1. Intra-Modal Self-Attention
        txt_s_pad_mask = ~mask_txt_s  # [N, L_s]
        txt_s_intra, _ = self.intra_txt_s(
            E_txt_s, E_txt_s, E_txt_s, key_padding_mask=txt_s_pad_mask
        )
        hat_E_txt_s = self.ln_intra_txt_s(E_txt_s + self.dropout(txt_s_intra))

        img_s_intra, _ = self.intra_img_s(cur_E_img_s, cur_E_img_s, cur_E_img_s)
        hat_E_img_s = self.ln_intra_img_s(cur_E_img_s + self.dropout(img_s_intra))

        # 2. Text-Guided Cross-Modal Attention
        C_cross_s, _ = self.cross_attn_s(
            query=hat_E_txt_s,
            key=hat_E_img_s,
            value=hat_E_img_s,
            need_weights=False,
        )
        hat_C_cross_s = self.ln_cross_s(hat_E_txt_s + self.dropout(C_cross_s))

        # 3. Masked Attention Pooling
        u_txt_s, _ = self.pool_txt_s(hat_E_txt_s, mask=mask_txt_s)
        u_cross_s, _ = self.pool_cross_s(hat_C_cross_s, mask=mask_txt_s)

        return {
            "u_txt_q": u_txt_q,
            "u_cross_q": u_cross_q,
            "u_txt_s": u_txt_s,
            "u_cross_s": u_cross_s,
            "A_cross_q": A_cross_q,  # [N, L_q, P]
        }

    def fuse_with_context(
        self,
        u_txt_q: torch.Tensor,
        u_cross_q: torch.Tensor,
        u_txt_s: torch.Tensor,
        u_cross_s: torch.Tensor,
        m_img_q: torch.Tensor,
        m_txt_s: torch.Tensor,
        m_img_s: torch.Tensor,
        raw_ctx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply context-conditioned hierarchical gating to synthesize v_stem and v_sol.

        Args:
            u_txt_q: Pooled question text representation [..., d].
            u_cross_q: Pooled question cross-modal representation [..., d].
            u_txt_s: Pooled solution text representation [..., d].
            u_cross_s: Pooled solution cross-modal representation [..., d].
            m_img_q: Question image presence flag [..., 1].
            m_txt_s: Solution text presence flag [..., 1].
            m_img_s: Solution image presence flag [..., 1].
            raw_ctx: Raw dynamic context vector [..., d_ctx].

        Returns:
            Tuple of:
            - v_stem: [..., d] gated question stem representation.
            - v_sol: [..., d] gated ground-truth solution representation.
            - lambda_q: [..., d] question gating weights.
            - lambda_s: [..., d] solution gating weights.
        """
        # 1. Project context vector
        c_ctx = self.ln_ctx(self.proj_ctx(raw_ctx))  # [..., d]

        # 2. Question Hierarchical Gate
        q_gate_input = torch.cat([u_txt_q, u_cross_q, c_ctx], dim=-1)
        raw_lambda_q = torch.sigmoid(self.w_ctx_q(q_gate_input))  # [..., d]

        # Lock gate strictly to 0 when visual modality is missing or disabled
        if self.use_visual:
            lambda_q = raw_lambda_q * m_img_q.float()
        else:
            lambda_q = torch.zeros_like(raw_lambda_q)

        v_stem = (1.0 - lambda_q) * u_txt_q + lambda_q * u_cross_q  # [..., d]

        # 3. Solution Hierarchical Gate
        s_gate_input = torch.cat([u_txt_s, u_cross_s], dim=-1)
        raw_lambda_s = torch.sigmoid(self.w_ctx_s(s_gate_input))  # [..., d]

        if self.use_visual:
            lambda_s = raw_lambda_s * m_img_s.float()
        else:
            lambda_s = torch.zeros_like(raw_lambda_s)

        tilde_v_sol = (1.0 - lambda_s) * u_txt_s + lambda_s * u_cross_s  # [..., d]

        # Canonical solution imputation when text solution is missing.
        # Note on dual-gating: By Equation (89) in ACD-MMKT.md, when m_txt_s == 0,
        # the solution acquisition term is explicitly zeroed out in Subsystem 3,
        # so unobserved solutions do not propagate spurious gradients into history.
        # e_null_sol provides a well-defined fallback tensor ensuring numerical stability.
        m_txt_expand = m_txt_s.float()  # [..., 1]
        e_null_expand = self.e_null_sol.expand_as(tilde_v_sol)
        v_sol = m_txt_expand * tilde_v_sol + (1.0 - m_txt_expand) * e_null_expand  # [..., d]

        return v_stem, v_sol, lambda_q, lambda_s

    def forward(
        self,
        E_txt_q: torch.Tensor,
        mask_txt_q: torch.Tensor,
        E_img_q: torch.Tensor,
        E_txt_s: torch.Tensor,
        mask_txt_s: torch.Tensor,
        E_img_s: torch.Tensor,
        modality_masks: torch.Tensor,
        dynamic_ctx: torch.Tensor,
        q_indices: Optional[torch.Tensor] = None,
        static_modality_masks: Optional[torch.Tensor] = None,
        mg_v_txt_q: Optional[torch.Tensor] = None,
        mg_v_img_q: Optional[torch.Tensor] = None,
        mg_v_txt_s: Optional[torch.Tensor] = None,
        mg_v_img_s: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Subsystem 1.
        Supports both direct item forward (Stage 1) and sequence batch forward with
        Unique-Question Batch Indexing (Stage 2).

        Args:
            E_txt_q: Question text tokens [B_or_N, L_q, 768] or [Total_items, L_q, 768].
            mask_txt_q: Question text padding mask [B_or_N, L_q].
            E_img_q: Question image patches [B_or_N, P, 768].
            E_txt_s: Solution text tokens [B_or_N, L_s, 768].
            mask_txt_s: Solution text padding mask [B_or_N, L_s].
            E_img_s: Solution image patches [B_or_N, P, 768].
            modality_masks: Flags [..., 3] where cols are [m_img_q, m_txt_s, m_img_s].
            dynamic_ctx: Context vectors [..., 21].
            q_indices: Optional sequence question indices [B, L] if using static bank lookup.
            static_modality_masks: Optional table of static item modality flags [|Q|, 3].
            mg_v_txt_q: Optional pre-extracted global question text vectors [|Q|, 768].
            mg_v_img_q: Optional pre-extracted global question image vectors [|Q|, 768].
            mg_v_txt_s: Optional pre-extracted global solution text vectors [|Q|, 768].
            mg_v_img_s: Optional pre-extracted global solution image vectors [|Q|, 768].

        Returns:
            Dictionary containing:
            - 'v_stem': Fused question representation [..., d]
            - 'v_sol': Fused solution representation [..., d]
            - 'lambda_q': Question gate weights [..., d]
            - 'lambda_s': Solution gate weights [..., d]
            - 'A_cross_q': Cross-modal alignment matrix.
        """
        m_img_q = modality_masks[..., 0:1]
        m_txt_s = modality_masks[..., 1:2]
        m_img_s = modality_masks[..., 2:3]

        if q_indices is not None and q_indices.ndim == 2:
            # Stage 2 Sequence Mode: Perform Unique-Question Batch Indexing
            B, L = q_indices.shape
            unique_q, inv_idx = torch.unique(q_indices, return_inverse=True)
            N_uniq = unique_q.size(0)

            # Retrieve fine-grained features only for unique questions in batch
            u_E_txt_q = E_txt_q[unique_q]
            u_mask_txt_q = mask_txt_q[unique_q]
            u_E_img_q = E_img_q[unique_q]
            u_E_txt_s = E_txt_s[unique_q]
            u_mask_txt_s = mask_txt_s[unique_q]
            u_E_img_s = E_img_s[unique_q]

            u_mg_v_txt_q = mg_v_txt_q[unique_q] if mg_v_txt_q is not None else None
            u_mg_v_img_q = mg_v_img_q[unique_q] if mg_v_img_q is not None else None
            u_mg_v_txt_s = mg_v_txt_s[unique_q] if mg_v_txt_s is not None else None
            u_mg_v_img_s = mg_v_img_s[unique_q] if mg_v_img_s is not None else None

            # Unique presence flags for static items
            if static_modality_masks is not None:
                u_m_img_q = static_modality_masks[unique_q, 0:1]
                u_m_img_s = static_modality_masks[unique_q, 2:3]
            else:
                flat_masks = modality_masks.reshape(B * L, -1)
                flat_q = q_indices.reshape(B * L)
                u_m_img_q = torch.stack([flat_masks[flat_q == uq][0, 0:1] for uq in unique_q], dim=0)
                u_m_img_s = torch.stack([flat_masks[flat_q == uq][0, 2:3] for uq in unique_q], dim=0)

            item_out = self.process_unique_items(
                E_txt_q=u_E_txt_q,
                mask_txt_q=u_mask_txt_q,
                E_img_q=u_E_img_q,
                E_txt_s=u_E_txt_s,
                mask_txt_s=u_mask_txt_s,
                E_img_s=u_E_img_s,
                m_img_q=u_m_img_q,
                m_img_s=u_m_img_s,
                mg_v_txt_q=u_mg_v_txt_q,
                mg_v_img_q=u_mg_v_img_q,
                mg_v_txt_s=u_mg_v_txt_s,
                mg_v_img_s=u_mg_v_img_s,
            )

            # Scatter back to sequence dimensions [B, L, d]
            seq_u_txt_q = item_out["u_txt_q"][inv_idx].view(B, L, self.d_model)
            seq_u_cross_q = item_out["u_cross_q"][inv_idx].view(B, L, self.d_model)
            seq_u_txt_s = item_out["u_txt_s"][inv_idx].view(B, L, self.d_model)
            seq_u_cross_s = item_out["u_cross_s"][inv_idx].view(B, L, self.d_model)

            v_stem, v_sol, lambda_q, lambda_s = self.fuse_with_context(
                u_txt_q=seq_u_txt_q,
                u_cross_q=seq_u_cross_q,
                u_txt_s=seq_u_txt_s,
                u_cross_s=seq_u_cross_s,
                m_img_q=m_img_q,
                m_txt_s=m_txt_s,
                m_img_s=m_img_s,
                raw_ctx=dynamic_ctx,
            )

            return {
                "v_stem": v_stem,
                "v_sol": v_sol,
                "lambda_q": lambda_q,
                "lambda_s": lambda_s,
                "A_cross_q": item_out["A_cross_q"],
                "uniq_q": unique_q,
                "inv_idx": inv_idx,
                "u_txt_q": item_out["u_txt_q"],
                "u_cross_q": item_out["u_cross_q"],
                "u_txt_s": item_out["u_txt_s"],
                "u_cross_s": item_out["u_cross_s"],
            }
        else:
            # Direct Item Mode (Stage 1 or Flattened Sequences)
            item_out = self.process_unique_items(
                E_txt_q=E_txt_q,
                mask_txt_q=mask_txt_q,
                E_img_q=E_img_q,
                E_txt_s=E_txt_s,
                mask_txt_s=mask_txt_s,
                E_img_s=E_img_s,
                m_img_q=m_img_q,
                m_img_s=m_img_s,
                mg_v_txt_q=mg_v_txt_q,
                mg_v_img_q=mg_v_img_q,
                mg_v_txt_s=mg_v_txt_s,
                mg_v_img_s=mg_v_img_s,
            )


            v_stem, v_sol, lambda_q, lambda_s = self.fuse_with_context(
                u_txt_q=item_out["u_txt_q"],
                u_cross_q=item_out["u_cross_q"],
                u_txt_s=item_out["u_txt_s"],
                u_cross_s=item_out["u_cross_s"],
                m_img_q=m_img_q,
                m_txt_s=m_txt_s,
                m_img_s=m_img_s,
                raw_ctx=dynamic_ctx,
            )

            return {
                "v_stem": v_stem,
                "v_sol": v_sol,
                "lambda_q": lambda_q,
                "lambda_s": lambda_s,
                "A_cross_q": item_out["A_cross_q"],
                "u_txt_q": item_out["u_txt_q"],
                "u_cross_q": item_out["u_cross_q"],
                "u_txt_s": item_out["u_txt_s"],
                "u_cross_s": item_out["u_cross_s"],
            }
