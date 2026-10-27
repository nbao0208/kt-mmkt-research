"""
Layer 1: Frozen Feature Projection, LayerNorm, and Learnable Modality Imputation.

Implements the mathematical formulation specified in docs/models/MG-MMKT.md:
- Independent linear projection and LayerNorm for each modality stream.
- Learnable canonical missing-modality tokens (e_null_img_q, e_null_txt_s, e_null_img_s).
- Explicit binary gating to avoid distribution collapse and guarantee non-zero gradient flow.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn


class ModalityImputationLayer(nn.Module):
    """
    Projects raw pre-extracted frozen multimodal features to hidden dimension d,
    normalizes with LayerNorm, and imputes missing modalities using learnable tokens.
    """

    def __init__(
        self,
        d_raw_txt: int = 768,
        d_raw_img: int = 768,
        d_model: int = 128,
        use_learnable_imputation: bool = True,
    ):
        """
        Initialize ModalityImputationLayer.

        Args:
            d_raw_txt: Dimension of raw text feature vectors (default: 768).
            d_raw_img: Dimension of raw image feature vectors (default: 768).
            d_model: Unified projection dimension d (default: 128).
            use_learnable_imputation: Whether to use learnable e_null tokens (True) or zero-padding (False).
        """
        super().__init__()
        self.d_model = d_model
        self.use_learnable_imputation = use_learnable_imputation

        # Linear projections for question text and image
        self.proj_txt_q = nn.Linear(d_raw_txt, d_model)
        self.ln_txt_q = nn.LayerNorm(d_model)

        self.proj_img_q = nn.Linear(d_raw_img, d_model)
        self.ln_img_q = nn.LayerNorm(d_model)

        # Linear projections for solution text and image
        self.proj_txt_s = nn.Linear(d_raw_txt, d_model)
        self.ln_txt_s = nn.LayerNorm(d_model)

        self.proj_img_s = nn.Linear(d_raw_img, d_model)
        self.ln_img_s = nn.LayerNorm(d_model)

        # Learnable canonical missing-modality tokens in R^d
        # Initialized with standard normal distribution N(0, 1) so element variance is 1
        # and expected L2 norm matches LayerNorm output (~sqrt(d_model)), satisfying Plan Step 2.2.
        if self.use_learnable_imputation:
            self.e_null_img_q = nn.Parameter(torch.randn(d_model))
            self.e_null_txt_s = nn.Parameter(torch.randn(d_model))
            self.e_null_img_s = nn.Parameter(torch.randn(d_model))
        else:
            self.register_buffer("e_null_img_q", torch.zeros(d_model))
            self.register_buffer("e_null_txt_s", torch.zeros(d_model))
            self.register_buffer("e_null_img_s", torch.zeros(d_model))

    def forward(
        self,
        v_txt_q: torch.Tensor,
        v_img_q: torch.Tensor,
        v_txt_s: torch.Tensor,
        v_img_s: torch.Tensor,
        modality_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for Layer 1.

        Args:
            v_txt_q: Question text features [B, L, d_raw_txt].
            v_img_q: Question image features [B, L, d_raw_img].
            v_txt_s: Solution text features [B, L, d_raw_txt].
            v_img_s: Solution image features [B, L, d_raw_img].
            modality_masks: Binary presence flags [B, L, 3] where:
                modality_masks[:, :, 0] is m_img_q
                modality_masks[:, :, 1] is m_txt_s
                modality_masks[:, :, 2] is m_img_s

        Returns:
            Tuple of imputed feature tensors:
            (tilde_v_txt_q, tilde_v_img_q, tilde_v_txt_s, tilde_v_img_s), each of shape [B, L, d_model].
        """
        # Ensure float32 for numerical stability in projection and LayerNorm
        v_txt_q = v_txt_q.float()
        v_img_q = v_img_q.float()
        v_txt_s = v_txt_s.float()
        v_img_s = v_img_s.float()

        # Step 1: Linear projections and LayerNorm
        h_txt_q = self.ln_txt_q(self.proj_txt_q(v_txt_q))
        h_img_q = self.ln_img_q(self.proj_img_q(v_img_q))
        h_txt_s = self.ln_txt_s(self.proj_txt_s(v_txt_s))
        h_img_s = self.ln_img_s(self.proj_img_s(v_img_s))

        # Step 2: Extract binary masks and expand to [B, L, 1] for broadcasting
        m_img_q = modality_masks[:, :, 0:1]  # [B, L, 1]
        m_txt_s = modality_masks[:, :, 1:2]  # [B, L, 1]
        m_img_s = modality_masks[:, :, 2:3]  # [B, L, 1]

        # Step 3: Modality imputation switch
        # Question text is always present
        tilde_v_txt_q = h_txt_q

        # Question image: m * h + (1 - m) * e_null
        tilde_v_img_q = m_img_q * h_img_q + (1.0 - m_img_q) * self.e_null_img_q

        # Solution text: m * h + (1 - m) * e_null
        tilde_v_txt_s = m_txt_s * h_txt_s + (1.0 - m_txt_s) * self.e_null_txt_s

        # Solution image: m * h + (1 - m) * e_null
        tilde_v_img_s = m_img_s * h_img_s + (1.0 - m_img_s) * self.e_null_img_s

        return tilde_v_txt_q, tilde_v_img_q, tilde_v_txt_s, tilde_v_img_s
