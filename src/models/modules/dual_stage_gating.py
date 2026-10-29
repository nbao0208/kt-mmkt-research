"""
Hierarchical Dual-Stage Gated Fusion and Cross-Dimension Mixing Module (Layer 2 of MG-MMKT v2).

Implements:
1. Stage 1 Modality-Level Softmax Gating with smooth learned missing penalty delta:
   - Question gating: alpha_tau^q in Delta^2 over (id, txt, img).
   - Solution gating: beta_tau^s in Delta^1 over (txt, img).
2. Stage 2 Channel-wise Feature Gating (GLU):
   - Computes element-wise modulation gate vectors G in (0, 1)^d.
3. Post-Fusion Cross-Dimension Residual FFN:
   - Expands d -> 2d -> d with GELU, Dropout, residual shortcut, and LayerNorm.
   - Enables full cross-dimensional non-linear interactions while maintaining 1st-order explainability.
"""

from typing import Dict, Any, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuestionDualStageGating(nn.Module):
    """
    Hierarchical Dual-Stage Gated Fusion for Question Modalities (Layer 2).
    Produces complete question representation x_tau in R^d and explainability weights alpha_tau^q.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_ffn: int = 256,
        dropout: float = 0.20,
        use_glu: bool = True,
        use_post_fusion_ffn: bool = True,
    ):
        """
        Args:
            d_model: Hidden dimension d (default: 128).
            d_ffn: Feed-forward expansion dimension (default: 256).
            dropout: Dropout probability (default: 0.20).
            use_glu: Whether to apply channel-wise GLU gating (default: True).
            use_post_fusion_ffn: Whether to apply post-fusion residual FFN (default: True).
        """
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.use_glu = use_glu
        self.use_post_fusion_ffn = use_post_fusion_ffn

        # Context vector dimension: [e_struct ; v_txt_q ; v_img_q ; m_img_q] -> 3d + 1
        ctx_dim = 3 * d_model + 1

        # 1. Modality-level gate MLP: u^q -> R^d -> R^3
        self.gate_mlp = nn.Sequential(
            nn.Linear(ctx_dim, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 3),
        )

        # Raw penalty parameter for missing image, softplus initialized to ~2.0
        # softplus^-1(2.0) = ln(exp(2.0) - 1) approx 1.841
        init_delta = math.log(math.exp(2.0) - 1.0)
        self.raw_delta_q = nn.Parameter(torch.tensor(init_delta, dtype=torch.float32))

        # 2. Channel-wise GLU projection: u^q -> 3 * d
        if self.use_glu:
            self.glu_proj = nn.Linear(ctx_dim, 3 * d_model)

        # 3. Post-fusion cross-dimension residual FFN: d -> d_ffn -> d
        if self.use_post_fusion_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.GELU(),
                nn.Linear(d_ffn, d_model),
            )
            self.dropout = nn.Dropout(dropout)

        self.ln_out = nn.LayerNorm(d_model)

    def forward(
        self,
        e_struct: torch.Tensor,
        tilde_v_txt_q: torch.Tensor,
        tilde_v_img_q: torch.Tensor,
        m_img_q: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Question Dual-Stage Gating.

        Args:
            e_struct: Structural embeddings [B, L, d].
            tilde_v_txt_q: Normalized question text representations [B, L, d].
            tilde_v_img_q: Normalized question image representations [B, L, d].
            m_img_q: Question image presence flags [B, L, 1].

        Returns:
            Tuple of:
                - x: Complete question representation [B, L, d].
                - alpha_q: Modality attention weights [B, L, 3] in Delta^2 (id, txt, img).
        """
        # Form question context vector: u^q in R^(3d + 1)
        u_q = torch.cat([e_struct, tilde_v_txt_q, tilde_v_img_q, m_img_q], dim=-1)

        # 1. Modality-level logits
        logits = self.gate_mlp(u_q)  # [B, L, 3] -> (id, txt, img)
        g_id = logits[..., 0:1]
        g_txt = logits[..., 1:2]
        g_img = logits[..., 2:3]

        # Bound delta_q > 0 with upper clamp to avoid underflow
        delta_q = torch.clamp(F.softplus(self.raw_delta_q), min=1e-4, max=10.0)
        # Apply smooth penalty when question has no image
        g_img_penalized = g_img - delta_q * (1.0 - m_img_q)

        adjusted_logits = torch.cat([g_id, g_txt, g_img_penalized], dim=-1)
        alpha_q = F.softmax(adjusted_logits, dim=-1)  # [B, L, 3]

        alpha_id = alpha_q[..., 0:1]
        alpha_txt = alpha_q[..., 1:2]
        alpha_img = alpha_q[..., 2:3]

        # 2. Channel-wise feature gating (GLU)
        if self.use_glu:
            glu_gates = torch.sigmoid(self.glu_proj(u_q))  # [B, L, 3d]
            G_id, G_txt, G_img = torch.split(glu_gates, self.d_model, dim=-1)
            v_id_gated = G_id * e_struct
            v_txt_gated = G_txt * tilde_v_txt_q
            v_img_gated = G_img * tilde_v_img_q
        else:
            v_id_gated = e_struct
            v_txt_gated = tilde_v_txt_q
            v_img_gated = tilde_v_img_q

        # Primary weighted combination: bar_x in R^d
        bar_x = alpha_id * v_id_gated + alpha_txt * v_txt_gated + alpha_img * v_img_gated

        # 3. Post-fusion cross-dimension residual FFN
        if self.use_post_fusion_ffn:
            x_res = self.ffn(bar_x)
            x = self.ln_out(bar_x + self.dropout(x_res))
        else:
            x = self.ln_out(bar_x)

        return x, alpha_q


class SolutionDualStageGating(nn.Module):
    """
    Hierarchical Dual-Stage Gated Fusion for Solution Modalities (Layer 2).
    Produces complete solution representation z^s in R^d and explainability weights beta^s.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_ffn: int = 256,
        dropout: float = 0.20,
        use_glu: bool = True,
        use_post_fusion_ffn: bool = True,
    ):
        """
        Args:
            d_model: Hidden dimension d (default: 128).
            d_ffn: Feed-forward expansion dimension (default: 256).
            dropout: Dropout probability (default: 0.20).
            use_glu: Whether to apply channel-wise GLU gating (default: True).
            use_post_fusion_ffn: Whether to apply post-fusion residual FFN (default: True).
        """
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.use_glu = use_glu
        self.use_post_fusion_ffn = use_post_fusion_ffn

        # Context vector dimension: [v_txt_s ; v_img_s ; e_C ; m_txt_s ; m_img_s] -> 3d + 2
        ctx_dim = 3 * d_model + 2

        # 1. Modality-level gate MLP: u^s -> R^d -> R^2
        self.gate_mlp = nn.Sequential(
            nn.Linear(ctx_dim, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 2),
        )

        # Raw penalty parameters for missing solution text and image
        init_delta = math.log(math.exp(2.0) - 1.0)
        self.raw_delta_s_txt = nn.Parameter(torch.tensor(init_delta, dtype=torch.float32))
        self.raw_delta_s_img = nn.Parameter(torch.tensor(init_delta, dtype=torch.float32))

        # 2. Channel-wise GLU projection: u^s -> 2 * d
        if self.use_glu:
            self.glu_proj = nn.Linear(ctx_dim, 2 * d_model)

        # 3. Post-fusion cross-dimension residual FFN: d -> d_ffn -> d
        if self.use_post_fusion_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.GELU(),
                nn.Linear(d_ffn, d_model),
            )
            self.dropout = nn.Dropout(dropout)

        self.ln_out = nn.LayerNorm(d_model)

    def forward(
        self,
        tilde_v_txt_s: torch.Tensor,
        tilde_v_img_s: torch.Tensor,
        e_C: torch.Tensor,
        m_txt_s: torch.Tensor,
        m_img_s: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Solution Dual-Stage Gating.

        Args:
            tilde_v_txt_s: Normalized solution text representations [B, L, d].
            tilde_v_img_s: Normalized solution image representations [B, L, d].
            e_C: Skill concept context [B, L, d].
            m_txt_s: Solution text presence flags [B, L, 1].
            m_img_s: Solution image presence flags [B, L, 1].

        Returns:
            Tuple of:
                - z_s: Complete solution representation [B, L, d].
                - beta_s: Modality attention weights [B, L, 2] in Delta^1 (txt, img).
        """
        # Form solution context vector: u^s in R^(3d + 2)
        u_s = torch.cat([tilde_v_txt_s, tilde_v_img_s, e_C, m_txt_s, m_img_s], dim=-1)

        # 1. Modality-level logits
        logits = self.gate_mlp(u_s)  # [B, L, 2] -> (txt, img)
        g_txt = logits[..., 0:1]
        g_img = logits[..., 1:2]

        delta_txt = torch.clamp(F.softplus(self.raw_delta_s_txt), min=1e-4, max=10.0)
        delta_img = torch.clamp(F.softplus(self.raw_delta_s_img), min=1e-4, max=10.0)

        g_txt_penalized = g_txt - delta_txt * (1.0 - m_txt_s)
        g_img_penalized = g_img - delta_img * (1.0 - m_img_s)

        adjusted_logits = torch.cat([g_txt_penalized, g_img_penalized], dim=-1)
        beta_s = F.softmax(adjusted_logits, dim=-1)  # [B, L, 2]

        beta_txt = beta_s[..., 0:1]
        beta_img = beta_s[..., 1:2]

        # 2. Channel-wise feature gating (GLU)
        if self.use_glu:
            glu_gates = torch.sigmoid(self.glu_proj(u_s))  # [B, L, 2d]
            G_txt, G_img = torch.split(glu_gates, self.d_model, dim=-1)
            v_txt_gated = G_txt * tilde_v_txt_s
            v_img_gated = G_img * tilde_v_img_s
        else:
            v_txt_gated = tilde_v_txt_s
            v_img_gated = tilde_v_img_s

        # Primary weighted combination: bar_z in R^d
        bar_z = beta_txt * v_txt_gated + beta_img * v_img_gated

        # 3. Post-fusion cross-dimension residual FFN
        if self.use_post_fusion_ffn:
            z_res = self.ffn(bar_z)
            z_s = self.ln_out(bar_z + self.dropout(z_res))
        else:
            z_s = self.ln_out(bar_z)

        return z_s, beta_s

