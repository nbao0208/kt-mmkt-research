"""
Subsystem 1: Symmetric Fine-Grained Cross-Modal Alignment & Intrinsic Context-Aware Gating.
Implements the exact mathematical specification for ACD-MMKTv2.

Key features:
1. Symmetric dual branches for Question Stem (m = q) and Solution (m = s).
2. Intra-modal text self-attention with padding mask and attention pooling (beta_txt^m).
3. Dynamic sub-indexing for visual modality to compute vision self-attention and text-guided
   cross-attention (A_cross^m) ONLY for items with images present (m_img^m == 1), saving >30x computation.
4. Intrinsic Context-Aware Gating (lambda^m) conditioned on [u_txt^m || u_cross^m || c_C] and
   strictly locked to 0 when m_img^m == 0, guaranteeing Item Parameter Invariance.
5. Ablation 3 support (use_fine_grained=False) replacing token/patch cross-attention with global pooling.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleBranchAligner(nn.Module):
    """
    Fine-grained cross-modal aligner for a single modality branch (either Stem or Solution).
    """

    def __init__(
        self,
        d_raw_txt: int = 768,
        d_raw_img: int = 768,
        d_model: int = 128,
        num_heads: int = 4,
        dropout: float = 0.15,
        use_fine_grained: bool = True,
    ):
        """
        Initialize SingleBranchAligner.

        Args:
            d_raw_txt: Dimensionality of raw text hidden states (default: 768).
            d_raw_img: Dimensionality of raw visual patch features (default: 768).
            d_model: Latent embedding dimension (default: 128).
            num_heads: Number of attention heads (default: 4).
            dropout: Dropout probability.
            use_fine_grained: Whether to use fine-grained token/patch cross-attention (Ablation 3 flag).
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_fine_grained = use_fine_grained

        # Projections from raw feature space to latent space
        self.proj_txt = nn.Linear(d_raw_txt, d_model)
        self.proj_img = nn.Linear(d_raw_img, d_model)

        if self.use_fine_grained:
            # Intra-modal text self-attention
            self.txt_self_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm_txt = nn.LayerNorm(d_model)
            self.q_pool_txt = nn.Parameter(torch.randn(d_model))
            nn.init.normal_(self.q_pool_txt, std=0.02)

            # Intra-modal image self-attention
            self.img_self_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm_img = nn.LayerNorm(d_model)

            # Text-guided cross-modal attention (Query: text, Key/Value: image patches)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm_cross = nn.LayerNorm(d_model)
            self.q_pool_cross = nn.Parameter(torch.randn(d_model))
            nn.init.normal_(self.q_pool_cross, std=0.02)
        else:
            # Ablation 3: Global pooled projection
            self.norm_txt = nn.LayerNorm(d_model)
            self.norm_img = nn.LayerNorm(d_model)

        # Intrinsic Context-Aware Gate conditioned on [u_txt || u_cross || c_C]
        self.gate_fc = nn.Linear(3 * d_model, d_model)

    def forward(
        self,
        E_txt: torch.Tensor,
        mask_txt: torch.Tensor,
        E_img: torch.Tensor,
        m_img: torch.Tensor,
        c_C: torch.Tensor,
        return_cross_weights: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for a single branch.

        Args:
            E_txt: Raw text hidden states [N, L, d_raw_txt].
            mask_txt: Boolean validity mask [N, L] (True for valid tokens, False for padding).
            E_img: Raw visual patch features [N, P, d_raw_img].
            m_img: Binary visual presence indicator [N] or [N, 1].
            c_C: Intrinsic core concept representation [N, d_model] from Subsystem 2.1.
            return_cross_weights: Whether to compute and retain cross-modal attention map.

        Returns:
            Tuple of:
            - v_branch: Fused multimodal item representation [N, d_model].
            - lambda_branch: Intrinsic visual gating vector [N, d_model].
            - beta_txt: Text token attention distribution [N, L].
            - A_cross: Fine-grained token-to-patch cross-modal alignment map [N, L, P] (or None).
        """
        N, L, _ = E_txt.shape
        P = E_img.shape[1]
        device = E_txt.device
        m_img_col = m_img.view(N, 1).float()

        if self.use_fine_grained:
            # 1. Intra-modal Text Self-Attention with padding mask
            txt_emb = self.proj_txt(E_txt.float())  # [N, L, d_model]
            # PyTorch key_padding_mask: True indicates position should be ignored
            key_padding_mask = ~mask_txt.bool()

            attn_txt_out, _ = self.txt_self_attn(
                query=txt_emb,
                key=txt_emb,
                value=txt_emb,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            hat_E_txt = self.norm_txt(txt_emb + attn_txt_out)  # [N, L, d_model]

            # Attention pooling over text tokens with safe padding mask
            scores_txt = torch.matmul(hat_E_txt, self.q_pool_txt)  # [N, L]
            # Use -1e4 for numerical stability across fp16/fp32
            scores_txt = scores_txt.masked_fill(key_padding_mask, -1e4)
            beta_txt = F.softmax(scores_txt, dim=-1)  # [N, L]
            u_txt = torch.bmm(beta_txt.unsqueeze(1), hat_E_txt).squeeze(1)  # [N, d_model]

            # 2. Dynamic Sub-indexing for Visual Modality
            # Only compute image self-attention and cross-attention for items with images
            img_active_indices = (m_img_col.squeeze(-1) > 0.5).nonzero(as_tuple=True)[0]
            num_active = len(img_active_indices)

            u_cross = torch.zeros(N, self.d_model, device=device, dtype=E_txt.dtype)
            A_cross = None

            if num_active > 0:
                # Active subsets
                active_E_img = E_img[img_active_indices].float()               # [M, P, d_raw_img]
                active_hat_E_txt = hat_E_txt[img_active_indices]             # [M, L, d_model]
                active_mask_txt = key_padding_mask[img_active_indices]       # [M, L]

                # Visual self-attention
                active_img_emb = self.proj_img(active_E_img)                 # [M, P, d_model]
                active_attn_img, _ = self.img_self_attn(
                    query=active_img_emb,
                    key=active_img_emb,
                    value=active_img_emb,
                    need_weights=False,
                )
                active_hat_E_img = self.norm_img(active_img_emb + active_attn_img)  # [M, P, d_model]

                # Text-guided cross-attention (Query: text, Key/Value: visual patches)
                if return_cross_weights:
                    active_cross_out, active_cross_weights = self.cross_attn(
                        query=active_hat_E_txt,
                        key=active_hat_E_img,
                        value=active_hat_E_img,
                        average_attn_weights=True,
                    )
                    # Mask padding token rows so padding positions do not emit spurious patch alignments
                    active_cross_weights = active_cross_weights.masked_fill(
                        active_mask_txt.unsqueeze(-1), 0.0
                    )
                    A_cross = torch.zeros(N, L, P, device=device, dtype=torch.float32)
                    A_cross[img_active_indices] = active_cross_weights.to(dtype=torch.float32)
                else:
                    active_cross_out, _ = self.cross_attn(
                        query=active_hat_E_txt,
                        key=active_hat_E_img,
                        value=active_hat_E_img,
                        need_weights=False,
                    )

                active_C_cross = self.norm_cross(active_cross_out)           # [M, L, d_model]

                # Attention pooling on cross-modal representation
                scores_cross = torch.matmul(active_C_cross, self.q_pool_cross)  # [M, L]
                scores_cross = scores_cross.masked_fill(active_mask_txt, -1e4)
                beta_cross = F.softmax(scores_cross, dim=-1)                   # [M, L]
                active_u_cross = torch.bmm(beta_cross.unsqueeze(1), active_C_cross).squeeze(1)  # [M, d_model]

                # Scatter back to full batch tensors
                u_cross[img_active_indices] = active_u_cross.to(dtype=E_txt.dtype)

        else:
            # Ablation 3: Global pooled representations
            txt_emb = self.proj_txt(E_txt.float())  # [N, L, d_model]
            mask_float = mask_txt.unsqueeze(-1).float()
            u_txt = (txt_emb * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-8)
            u_txt = self.norm_txt(u_txt)
            beta_txt = mask_float.squeeze(-1) / mask_float.sum(dim=1, keepdim=True).clamp(min=1e-8)

            img_emb = self.proj_img(E_img.float())  # [N, P, d_model]
            u_cross = self.norm_img(img_emb.mean(dim=1)) * m_img_col
            A_cross = torch.zeros(N, L, P, device=device, dtype=torch.float32) if return_cross_weights else None

        # 3. Intrinsic Context-Aware Gating conditioned on [u_txt || u_cross || c_C]
        gate_input = torch.cat([u_txt, u_cross, c_C], dim=-1)  # [N, 3 * d_model]
        lambda_branch = torch.sigmoid(self.gate_fc(gate_input)) * m_img_col  # [N, d_model]

        # Fused multimodal item representation (clean text when m_img == 0)
        v_branch = (1.0 - lambda_branch) * u_txt + lambda_branch * u_cross  # [N, d_model]

        return v_branch, lambda_branch, beta_txt, A_cross


class SymmetricCrossModalAligner(nn.Module):
    """
    Subsystem 1 of ACD-MMKTv2: Full Symmetric Cross-Modal Alignment & Intrinsic Gating.
    Processes both Question Stem (q) and Solution (s) branches symmetrically.
    """

    def __init__(
        self,
        d_raw_txt: int = 768,
        d_raw_img: int = 768,
        d_model: int = 128,
        num_heads: int = 4,
        dropout: float = 0.15,
        use_fine_grained: bool = True,
    ):
        """
        Initialize SymmetricCrossModalAligner.

        Args:
            d_raw_txt: Dimensionality of raw text hidden states (768).
            d_raw_img: Dimensionality of raw visual patch features (768).
            d_model: Latent embedding dimension (128).
            num_heads: Number of attention heads (4).
            dropout: Dropout probability.
            use_fine_grained: Whether to use fine-grained token/patch cross-attention (Ablation 3 flag).
        """
        super().__init__()
        self.stem_aligner = SingleBranchAligner(
            d_raw_txt=d_raw_txt,
            d_raw_img=d_raw_img,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_fine_grained=use_fine_grained,
        )
        self.sol_aligner = SingleBranchAligner(
            d_raw_txt=d_raw_txt,
            d_raw_img=d_raw_img,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_fine_grained=use_fine_grained,
        )

    def forward(
        self,
        E_txt_q: torch.Tensor,
        mask_txt_q: torch.Tensor,
        E_img_q: torch.Tensor,
        E_txt_s: torch.Tensor,
        mask_txt_s: torch.Tensor,
        E_img_s: torch.Tensor,
        modality_masks: torch.Tensor,
        c_C: torch.Tensor,
        return_cross_weights: bool = True,
    ) -> Dict[str, Any]:
        """
        Symmetric forward pass across Question Stem and Solution.

        Args:
            E_txt_q: Stem text hidden states [N, L_q, 768].
            mask_txt_q: Stem text padding mask [N, L_q].
            E_img_q: Stem visual patch features [N, P_q, 768].
            E_txt_s: Solution text hidden states [N, L_s, 768].
            mask_txt_s: Solution text padding mask [N, L_s].
            E_img_s: Solution visual patch features [N, P_s, 768].
            modality_masks: Binary flags [N, 3] for (m_img_q, m_txt_s, m_img_s).
            c_C: Intrinsic core concept representations [N, d_model] from Subsystem 2.1.
            return_cross_weights: Whether to compute and retain cross-modal attention maps.

        Returns:
            Dictionary containing:
            - 'v_stem': Fused multimodal stem representation [N, d_model].
            - 'v_sol': Fused multimodal solution representation [N, d_model].
            - 'lambda_q': Visual gate for stem [N, d_model].
            - 'lambda_s': Visual gate for solution [N, d_model].
            - 'beta_txt_q': Token attention distribution for stem [N, L_q].
            - 'beta_txt_s': Token attention distribution for solution [N, L_s].
            - 'A_cross_q': Stem cross-modal alignment map [N, L_q, P_q] (or None).
            - 'A_cross_s': Solution cross-modal alignment map [N, L_s, P_s] (or None).
        """
        m_img_q = modality_masks[:, 0]  # [N]
        m_img_s = modality_masks[:, 2]  # [N]

        v_stem, lambda_q, beta_txt_q, A_cross_q = self.stem_aligner(
            E_txt=E_txt_q,
            mask_txt=mask_txt_q,
            E_img=E_img_q,
            m_img=m_img_q,
            c_C=c_C,
            return_cross_weights=return_cross_weights,
        )

        v_sol, lambda_s, beta_txt_s, A_cross_s = self.sol_aligner(
            E_txt=E_txt_s,
            mask_txt=mask_txt_s,
            E_img=E_img_s,
            m_img=m_img_s,
            c_C=c_C,
            return_cross_weights=return_cross_weights,
        )

        return {
            "v_stem": v_stem,
            "v_sol": v_sol,
            "lambda_q": lambda_q,
            "lambda_s": lambda_s,
            "beta_txt_q": beta_txt_q,
            "beta_txt_s": beta_txt_s,
            "A_cross_q": A_cross_q,
            "A_cross_s": A_cross_s,
        }
