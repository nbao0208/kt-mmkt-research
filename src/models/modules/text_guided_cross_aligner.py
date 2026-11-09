"""
Subsystem 1: Text-Guided Cross-Modal Alignment & Lightweight Cognitive Gating.
Implements the exact mathematical specification for ACD-MMKTv3 (Section 3, Subsystem 1).

Key features:
1. Linear projection without intra-modality self-attention:
   - Text stem and solution: H_txt_q, H_txt_s via shared W_proj_txt + LayerNorm.
   - Vision stem: H_img_q via W_proj_img + LayerNorm (when use_visual=True).
2. Global text vector extraction via Attention Pooling with learnable query q_pool_txt.
   - u_txt_q = (H_txt_q)^T beta_txt_q, u_txt_s = (H_txt_s)^T beta_txt_s.
   - Text-dominant solution stream: v_sol = u_txt_s (with optional gated visual supplement when use_visual=True).
3. Text-guided Cross-Modal Attention with Letterbox Patch Mask M_patch_q:
   - Query: H_txt_q W_cm_Q, Key: H_img_q W_cm_K, Value: H_img_q W_cm_V.
   - Softmax score includes M_patch_q in {0, -1e9}^49 to strictly ignore white letterbox borders.
   - Cross vector: u_cross_q = (C_cross_q)^T beta_txt_q.
   - Active only on items with m_img_q = 1; zero-masked when m_img_q = 0.
4. Intrinsic Cognitive Gate lambda_q:
   - Conditioned on [u_txt_q || u_cross_q || c_C || c_ctx_static].
   - Hard zero gating outside sigmoid when m_img_q = 0, guaranteeing v_stem = u_txt_q cleanly.
5. Support for Modality Ablations:
   - use_multimodal=False (Ablation 1): completely bypassed, zero parameters instantiated.
   - use_solution=False (Ablation 2): solution text and image projections completely disabled, v_sol = 0.
   - use_visual=False (Ablation 3): visual projections, cross-attention and gating disabled; strictly no solution image leakage.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextGuidedCrossAligner(nn.Module):
    """
    Subsystem 1 of ACD-MMKTv3: Text-Guided Cross-Modal Alignment & Lightweight Cognitive Gating.
    """

    def __init__(
        self,
        d_raw_txt: int = 768,
        d_raw_img: int = 768,
        d_model: int = 128,
        d_static_ctx: int = 15,
        dropout: float = 0.15,
        use_visual: bool = True,
        use_solution: bool = True,
        use_multimodal: bool = True,
    ):
        """
        Initialize TextGuidedCrossAligner.

        Args:
            d_raw_txt: Raw text token dimension (768).
            d_raw_img: Raw vision patch dimension (768).
            d_model: Latent projection dimension (128).
            d_static_ctx: Dimension of static item context vector (15).
            dropout: Dropout probability.
            use_visual: Whether visual alignment is enabled (Ablation 3 flag).
            use_solution: Whether solution stream is enabled (Ablation 2 flag).
            use_multimodal: Master switch for multimodal stream (Ablation 1 flag).
        """
        super().__init__()
        self.d_model = d_model
        self.use_visual = use_visual
        self.use_solution = use_solution
        self.use_multimodal = use_multimodal

        if not self.use_multimodal:
            # Ablation 1: pure KC interaction, no Subsystem 1 parameters instantiated
            self.proj_txt = None
            self.norm_txt = None
            self.proj_img = None
            self.norm_img = None
            self.q_pool_txt = None
            self.proj_img_sol = None
            self.norm_img_sol = None
            self.gate_sol_img = None
            self.W_cm_Q = None
            self.W_cm_K = None
            self.W_cm_V = None
            self.proj_ctx = None
            self.norm_ctx = None
            self.W_gate_q = None
            self.dropout = None
            return

        # 1.1 Base text linear projections (shared between stem and solution)
        self.proj_txt = nn.Linear(d_raw_txt, d_model)
        self.norm_txt = nn.LayerNorm(d_model)

        # 1.2 Learnable Attention Pooling query
        self.q_pool_txt = nn.Parameter(torch.randn(d_model))
        nn.init.normal_(self.q_pool_txt, std=0.02)

        self.dropout = nn.Dropout(dropout)

        # 1.3 Visual Stream components (only instantiated when use_visual=True)
        if self.use_visual:
            self.proj_img = nn.Linear(d_raw_img, d_model)
            self.norm_img = nn.LayerNorm(d_model)

            # Solution image projection (optional global visual supplement)
            if self.use_solution:
                self.proj_img_sol = nn.Linear(d_raw_img, d_model)
                self.norm_img_sol = nn.LayerNorm(d_model)
                self.gate_sol_img = nn.Linear(2 * d_model, d_model)
            else:
                self.proj_img_sol = None
                self.norm_img_sol = None
                self.gate_sol_img = None

            # Text-guided cross-modal attention projections
            self.W_cm_Q = nn.Linear(d_model, d_model, bias=False)
            self.W_cm_K = nn.Linear(d_model, d_model, bias=False)
            self.W_cm_V = nn.Linear(d_model, d_model, bias=False)

            # Intrinsic Cognitive Gate lambda_q
            self.proj_ctx = nn.Linear(d_static_ctx, d_model)
            self.norm_ctx = nn.LayerNorm(d_model)

            # Gate input: [u_txt_q || u_cross_q || c_C || c_ctx] -> 4 * d_model
            self.W_gate_q = nn.Linear(4 * d_model, d_model, bias=True)
            nn.init.constant_(self.W_gate_q.bias, -1.0)  # initialize slightly biased towards text
        else:
            self.proj_img = None
            self.norm_img = None
            self.proj_img_sol = None
            self.norm_img_sol = None
            self.gate_sol_img = None
            self.W_cm_Q = None
            self.W_cm_K = None
            self.W_cm_V = None
            self.proj_ctx = None
            self.norm_ctx = None
            self.W_gate_q = None

    def forward(
        self,
        E_txt_q: torch.Tensor,
        mask_txt_q: torch.Tensor,
        E_img_q: torch.Tensor,
        M_patch_q: torch.Tensor,
        E_txt_s: torch.Tensor,
        mask_txt_s: torch.Tensor,
        E_img_s: torch.Tensor,
        modality_masks: torch.Tensor,
        c_C: torch.Tensor,
        c_ctx: torch.Tensor,
        return_cross_weights: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass for Subsystem 1.

        Args:
            E_txt_q: Raw stem token features [M, L_q, d_raw_txt].
            mask_txt_q: Stem token validity mask [M, L_q] (bool).
            E_img_q: Raw stem image patches [M, P, d_raw_img].
            M_patch_q: Letterbox patch additive mask [M, P] (0 for valid, -1e9 for padded).
            E_txt_s: Raw solution token features [M, L_s, d_raw_txt].
            mask_txt_s: Solution token validity mask [M, L_s] (bool).
            E_img_s: Raw solution image global vector [M, d_raw_img].
            modality_masks: Modality availability flags [M, 3] (m_img_q, m_txt_s, m_img_s).
            c_C: Core concept representation from Subsystem 2.1 [M, d_model].
            c_ctx: Static item context features [M, 15].
            return_cross_weights: Whether to retain full A_cross_q map.

        Returns:
            Dictionary containing v_stem, v_sol, lambda_q, bar_lambda_q, beta_txt_q, beta_txt_s, A_cross_q, u_txt_q, u_cross_q.
        """
        M = E_txt_q.size(0)
        device = E_txt_q.device

        # If multimodal stream is entirely disabled (Ablation 1)
        if not self.use_multimodal:
            zeros_vec = torch.zeros((M, self.d_model), device=device, dtype=torch.float32)
            zeros_bar = torch.zeros((M, 1), device=device, dtype=torch.float32)
            len_q = E_txt_q.size(1) if E_txt_q is not None else 128
            len_s = E_txt_s.size(1) if E_txt_s is not None else 160
            beta_q = torch.zeros((M, len_q), device=device, dtype=torch.float32)
            beta_s = torch.zeros((M, len_s), device=device, dtype=torch.float32)
            return {
                "v_stem": zeros_vec,
                "v_sol": zeros_vec,
                "lambda_q": zeros_vec,
                "bar_lambda_q": zeros_bar,
                "beta_txt_q": beta_q,
                "beta_txt_s": beta_s,
                "A_cross_q": None,
                "u_txt_q": zeros_vec,
                "u_cross_q": zeros_vec,
            }

        scale_txt = 1.0 / (self.d_model ** 0.5)

        m_img_q = modality_masks[:, 0:1].float()  # [M, 1]
        m_img_s = modality_masks[:, 2:3].float()  # [M, 1]

        # -------------------------------------------------------------
        # 1.1 Base Feature Projections
        # -------------------------------------------------------------
        H_txt_q = self.dropout(self.norm_txt(self.proj_txt(E_txt_q.float())))  # [M, L_q, d_model]

        # -------------------------------------------------------------
        # 1.2 Attention Pooling for Stem Text
        # -------------------------------------------------------------
        q_scores = (torch.matmul(H_txt_q, self.q_pool_txt) * scale_txt).float()  # [M, L_q] fp32 for safe masking
        q_scores = q_scores.masked_fill(~mask_txt_q.bool(), -1e9)
        beta_txt_q = F.softmax(q_scores, dim=-1)  # [M, L_q]
        beta_txt_q = beta_txt_q * mask_txt_q.float()
        beta_txt_q = beta_txt_q / beta_txt_q.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        u_txt_q = torch.bmm(beta_txt_q.unsqueeze(1), H_txt_q.float()).squeeze(1)  # [M, d_model]

        # -------------------------------------------------------------
        # 1.3 Attention Pooling and Representation for Solution Stream
        # -------------------------------------------------------------
        if self.use_solution:
            H_txt_s = self.dropout(self.norm_txt(self.proj_txt(E_txt_s.float())))  # [M, L_s, d_model]
            s_scores = (torch.matmul(H_txt_s, self.q_pool_txt) * scale_txt).float()  # [M, L_s] fp32 for safe masking
            s_scores = s_scores.masked_fill(~mask_txt_s.bool(), -1e9)
            beta_txt_s = F.softmax(s_scores, dim=-1)  # [M, L_s]
            beta_txt_s = beta_txt_s * mask_txt_s.float()
            beta_txt_s = beta_txt_s / beta_txt_s.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            u_txt_s = torch.bmm(beta_txt_s.unsqueeze(1), H_txt_s.float()).squeeze(1)  # [M, d_model]

            # Solution representation v_sol: STRICTLY check use_visual to prevent visual leakage (Ablation 3)
            if self.use_visual and self.proj_img_sol is not None and (m_img_s > 0.5).any():
                raw_img_s = E_img_s.mean(dim=1) if E_img_s.dim() == 3 else E_img_s
                v_img_s = self.norm_img_sol(self.proj_img_sol(raw_img_s.float()))  # [M, d_model]
                gate_s = torch.sigmoid(self.gate_sol_img(torch.cat([u_txt_s, v_img_s], dim=-1))) * m_img_s
                v_sol = (1.0 - gate_s) * u_txt_s + gate_s * v_img_s
            else:
                v_sol = u_txt_s
        else:
            # Ablation 2: solution stream is entirely bypassed
            v_sol = torch.zeros_like(u_txt_q)
            len_s = E_txt_s.size(1) if E_txt_s is not None else 160
            beta_txt_s = torch.zeros((M, len_s), device=device, dtype=torch.float32)

        # -------------------------------------------------------------
        # 1.4 Text-guided Cross-Modal Attention with Letterbox Mask
        # -------------------------------------------------------------
        u_cross_q = torch.zeros_like(u_txt_q)
        A_cross_q = None

        if self.use_visual:
            has_img_mask = (m_img_q.squeeze(-1) > 0.5)
            if has_img_mask.any():
                img_indices = torch.where(has_img_mask)[0]
                sub_H_txt = H_txt_q[img_indices]        # [K, L_q, d]
                sub_E_img = E_img_q[img_indices]        # [K, P, d_raw]
                sub_M_patch = M_patch_q[img_indices]    # [K, P]
                sub_beta = beta_txt_q[img_indices]      # [K, L_q]
                sub_mask_txt = mask_txt_q[img_indices]  # [K, L_q]

                sub_H_img = self.dropout(self.norm_img(self.proj_img(sub_E_img.float())))  # [K, P, d]

                # Query: text, Key/Value: vision
                Q = self.W_cm_Q(sub_H_txt)  # [K, L_q, d]
                K = self.W_cm_K(sub_H_img)  # [K, P, d]
                V = self.W_cm_V(sub_H_img)  # [K, P, d]

                # Compute attention scores in float32 to prevent half-precision overflow from -1e9 mask
                scores_cm = (torch.bmm(Q.float(), K.float().transpose(1, 2)) * scale_txt) + sub_M_patch.float().unsqueeze(1)  # [K, L_q, P]
                sub_A_cross = F.softmax(scores_cm, dim=-1)  # [K, L_q, P]
                # Clean padding text rows to 0
                sub_A_cross = sub_A_cross * sub_mask_txt.unsqueeze(-1).float()

                sub_C_cross = torch.bmm(sub_A_cross, V.float())  # [K, L_q, d]
                sub_u_cross = torch.bmm(sub_beta.unsqueeze(1), sub_C_cross).squeeze(1)  # [K, d]

                u_cross_q[img_indices] = sub_u_cross

                if return_cross_weights:
                    A_cross_q = torch.zeros((M, E_txt_q.size(1), E_img_q.size(1)), device=device, dtype=torch.float32)
                    A_cross_q[img_indices] = sub_A_cross

            # -------------------------------------------------------------
            # 1.5 Intrinsic Cognitive Gate lambda_q & v_stem
            # -------------------------------------------------------------
            c_ctx_emb = self.norm_ctx(self.proj_ctx(c_ctx.float()))  # [M, d_model]
            gate_in = torch.cat([u_txt_q, u_cross_q, c_C, c_ctx_emb], dim=-1)  # [M, 4*d_model]
            # Hard visual masking outside sigmoid: lambda_q = 0 strictly if m_img_q == 0
            lambda_q = torch.sigmoid(self.W_gate_q(gate_in)) * m_img_q  # [M, d_model]
            bar_lambda_q = lambda_q.mean(dim=-1, keepdim=True)         # [M, 1]
            v_stem = (1.0 - lambda_q) * u_txt_q + lambda_q * u_cross_q   # [M, d_model]
        else:
            # Ablation 3: visual stream disabled, strictly pure text stem
            lambda_q = torch.zeros_like(u_txt_q)
            bar_lambda_q = torch.zeros((M, 1), device=device, dtype=torch.float32)
            v_stem = u_txt_q

        return {
            "v_stem": v_stem,
            "v_sol": v_sol,
            "lambda_q": lambda_q,
            "bar_lambda_q": bar_lambda_q,
            "beta_txt_q": beta_txt_q,
            "beta_txt_s": beta_txt_s,
            "A_cross_q": A_cross_q,
            "u_txt_q": u_txt_q,
            "u_cross_q": u_cross_q,
        }

