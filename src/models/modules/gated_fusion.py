"""
Layer 2: Modality-Aware Gated Fusion for Question and Solution Modalities.

Implements the mathematical formulation specified in docs/models/MG-MMKT.md:
- Question Masked Softmax Gate fusing ID/KC structure, question text, and question image into x_tau.
- Solution Masked Softmax Gate fusing solution text and solution image into z_tau^s.
- Mask-guided soft penalty delta > 0 to prevent gradient starvation and avoid NaNs under extreme missingness.
- Support for ablation studies (visual ablation and un-gated baseline).
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuestionMaskedGatedFusion(nn.Module):
    """
    Computes adaptive contribution weights alpha_tau^q = (alpha_id, alpha_txt, alpha_img)
    and fuses question structural ID, text, and image representations into x_tau.
    Supports gated fusion, equal weighting, and naive concatenation (Ablation 4).
    """

    def __init__(
        self,
        d_model: int = 128,
        use_visual: bool = True,
        use_gated_fusion: bool = True,
        fusion_mode: Optional[str] = None,
        max_delta: float = 10.0,
    ):
        """
        Initialize QuestionMaskedGatedFusion.

        Args:
            d_model: Hidden dimension d (default: 128).
            use_visual: Whether to include the visual modality branch (default: True).
            use_gated_fusion: Whether to use adaptive gated fusion (default: True).
            fusion_mode: Explicit fusion mode ('gated', 'equal_weight', or 'naive_concat').
            max_delta: Upper bound for learnable soft penalty delta to prevent gradient starvation (default: 10.0).
        """
        super().__init__()
        self.d_model = d_model
        self.use_visual = use_visual
        self.max_delta = max_delta

        # Determine effective fusion mode
        if fusion_mode is not None:
            self.fusion_mode = fusion_mode
        elif use_gated_fusion:
            self.fusion_mode = "gated"
        else:
            self.fusion_mode = "equal_weight"

        self.use_gated_fusion = (self.fusion_mode == "gated")

        if self.fusion_mode == "gated":
            # Context vector u^q dimension:
            # if use_visual: [e_q + e_C] (d) + tilde_v_txt (d) + tilde_v_img (d) + m_img_q (1) -> 4d + 1
            # if not use_visual: [e_q + e_C] (d) + tilde_v_txt (d) -> 2d
            in_dim = (4 * d_model + 1) if use_visual else (2 * d_model)
            self.gate_mlp = nn.Sequential(
                nn.Linear(in_dim, d_model),
                nn.Tanh(),
            )
            out_dim = 3 if use_visual else 2
            self.gate_out = nn.Linear(d_model, out_dim)

            # Learnable non-negative penalty delta for missing image modality
            if use_visual:
                self.raw_delta_img_q = nn.Parameter(torch.tensor(2.0))

        elif self.fusion_mode == "naive_concat":
            # Configuration 4: Naive linear projection of concatenated modalities
            cat_dim = (3 * d_model) if use_visual else (2 * d_model)
            self.proj_naive_concat = nn.Linear(cat_dim, d_model)

    def get_deltas(self) -> Dict[str, float]:
        """Return the current clamped penalty delta value(s)."""
        if self.fusion_mode == "gated" and self.use_visual and hasattr(self, "raw_delta_img_q"):
            delta = torch.clamp(F.softplus(self.raw_delta_img_q), min=0.0, max=self.max_delta)
            return {"delta_img_q": float(delta.detach().item())}
        return {}

    def forward(
        self,
        e_q: torch.Tensor,
        e_c: torch.Tensor,
        tilde_v_txt_q: torch.Tensor,
        tilde_v_img_q: torch.Tensor,
        m_img_q: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Question Gated Fusion.

        Args:
            e_q: Question ID embedding [B, L, d_model].
            e_c: Pooled Knowledge Component embedding [B, L, d_model].
            tilde_v_txt_q: Imputed question text representation [B, L, d_model].
            tilde_v_img_q: Imputed question image representation [B, L, d_model].
            m_img_q: Binary indicator for question image presence [B, L, 1].

        Returns:
            Tuple of:
            - x_q: Fused question representation [B, L, d_model].
            - alpha_q: Modality attribution weights [B, L, num_branches] summing to 1.
        """
        e_struct = e_q + e_c  # Combined structural ID and KC representation

        # 1. Ablation Mode: Naive Concat (Ablation 4)
        if self.fusion_mode == "naive_concat":
            if self.use_visual:
                # With zero-imputation, missing image branch has zeroes
                # Concatenate [e_struct, text, image] -> Linear(3d -> d)
                concat_in = torch.cat([e_struct, tilde_v_txt_q, tilde_v_img_q], dim=-1)
                x_q = self.proj_naive_concat(concat_in)
                alpha_q = torch.full(
                    (e_q.shape[0], e_q.shape[1], 3),
                    1.0 / 3.0,
                    device=e_q.device,
                    dtype=e_q.dtype,
                )
            else:
                concat_in = torch.cat([e_struct, tilde_v_txt_q], dim=-1)
                x_q = self.proj_naive_concat(concat_in)
                alpha_q = torch.full(
                    (e_q.shape[0], e_q.shape[1], 2),
                    0.5,
                    device=e_q.device,
                    dtype=e_q.dtype,
                )
            return x_q, alpha_q

        # 2. Baseline Mode: Equal weighting
        if self.fusion_mode == "equal_weight":
            if self.use_visual:
                alpha_q = torch.full(
                    (e_q.shape[0], e_q.shape[1], 3),
                    1.0 / 3.0,
                    device=e_q.device,
                    dtype=e_q.dtype,
                )
                x_q = (1.0 / 3.0) * (e_struct + tilde_v_txt_q + tilde_v_img_q)
            else:
                alpha_q = torch.full(
                    (e_q.shape[0], e_q.shape[1], 2),
                    0.5,
                    device=e_q.device,
                    dtype=e_q.dtype,
                )
                x_q = 0.5 * (e_struct + tilde_v_txt_q)
            return x_q, alpha_q

        # 3. Gated Mode: Adaptive Masked Softmax Gate
        if self.use_visual:
            # Concatenate context vector u^q in R^(4d + 1): [e_q, e_c, tilde_v_txt, tilde_v_img, m_img_q]
            u_q = torch.cat([e_q, e_c, tilde_v_txt_q, tilde_v_img_q, m_img_q], dim=-1)
            hidden = self.gate_mlp(u_q)
            raw_logits = self.gate_out(hidden)  # [B, L, 3]: (id, txt, img)

            # Apply mask-guided penalty on missing image branch, clamped to max_delta
            # Clamping prevents delta from growing unbounded (preventing gradient starvation on e_null)
            delta = torch.clamp(F.softplus(self.raw_delta_img_q), min=0.0, max=self.max_delta)
            penalty = (1.0 - m_img_q) * delta  # [B, L, 1]

            adj_logits = torch.cat(
                [
                    raw_logits[:, :, 0:1],
                    raw_logits[:, :, 1:2],
                    raw_logits[:, :, 2:3] - penalty,
                ],
                dim=-1,
            )

            alpha_q = F.softmax(adj_logits, dim=-1)  # [B, L, 3]

            # Convex combination
            alpha_id = alpha_q[:, :, 0:1]
            alpha_txt = alpha_q[:, :, 1:2]
            alpha_img = alpha_q[:, :, 2:3]

            x_q = alpha_id * e_struct + alpha_txt * tilde_v_txt_q + alpha_img * tilde_v_img_q

        else:
            # Visual modality excluded
            u_q = torch.cat([e_struct, tilde_v_txt_q], dim=-1)
            hidden = self.gate_mlp(u_q)
            raw_logits = self.gate_out(hidden)  # [B, L, 2]: (id, txt)
            alpha_q = F.softmax(raw_logits, dim=-1)  # [B, L, 2]

            alpha_id = alpha_q[:, :, 0:1]
            alpha_txt = alpha_q[:, :, 1:2]
            x_q = alpha_id * e_struct + alpha_txt * tilde_v_txt_q

        return x_q, alpha_q


class SolutionMaskedGatedFusion(nn.Module):
    """
    Computes adaptive contribution weights beta_tau^s = (beta_txt, beta_img)
    and fuses solution text and solution image representations into z_tau^s.
    Supports gated fusion, equal weighting, and naive concatenation (Ablation 4).
    """

    def __init__(
        self,
        d_model: int = 128,
        use_visual: bool = True,
        use_gated_fusion: bool = True,
        fusion_mode: Optional[str] = None,
        max_delta: float = 10.0,
    ):
        """
        Initialize SolutionMaskedGatedFusion.

        Args:
            d_model: Hidden dimension d (default: 128).
            use_visual: Whether to include the visual solution modality branch (default: True).
            use_gated_fusion: Whether to use adaptive gated fusion (default: True).
            fusion_mode: Explicit fusion mode ('gated', 'equal_weight', or 'naive_concat').
            max_delta: Upper bound for learnable soft penalties delta (default: 10.0).
        """
        super().__init__()
        self.d_model = d_model
        self.use_visual = use_visual
        self.max_delta = max_delta

        # Determine effective fusion mode
        if fusion_mode is not None:
            self.fusion_mode = fusion_mode
        elif use_gated_fusion:
            self.fusion_mode = "gated"
        else:
            self.fusion_mode = "equal_weight"

        self.use_gated_fusion = (self.fusion_mode == "gated")

        if self.fusion_mode == "gated" and self.use_visual:
            # Context vector u^s dimension: tilde_v_txt (d) + tilde_v_img (d) + m_txt_s (1) + m_img_s (1) -> 2d + 2
            in_dim = 2 * d_model + 2
            self.gate_mlp = nn.Sequential(
                nn.Linear(in_dim, d_model),
                nn.Tanh(),
            )
            self.gate_out = nn.Linear(d_model, 2)

            # Learnable penalties for missing solution modalities
            self.raw_delta_txt_s = nn.Parameter(torch.tensor(2.0))
            self.raw_delta_img_s = nn.Parameter(torch.tensor(2.0))

        elif self.fusion_mode == "naive_concat" and self.use_visual:
            # Configuration 4: Naive linear projection of concatenated solution modalities
            self.proj_naive_concat = nn.Linear(2 * d_model, d_model)

    def get_deltas(self) -> Dict[str, float]:
        """Return the current clamped penalty delta values."""
        deltas = {}
        if self.fusion_mode == "gated" and self.use_visual:
            if hasattr(self, "raw_delta_txt_s"):
                d_txt = torch.clamp(F.softplus(self.raw_delta_txt_s), min=0.0, max=self.max_delta)
                deltas["delta_txt_s"] = float(d_txt.detach().item())
            if hasattr(self, "raw_delta_img_s"):
                d_img = torch.clamp(F.softplus(self.raw_delta_img_s), min=0.0, max=self.max_delta)
                deltas["delta_img_s"] = float(d_img.detach().item())
        return deltas

    def forward(
        self,
        tilde_v_txt_s: torch.Tensor,
        tilde_v_img_s: torch.Tensor,
        m_txt_s: torch.Tensor,
        m_img_s: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Solution Gated Fusion.

        Args:
            tilde_v_txt_s: Imputed solution text representation [B, L, d_model].
            tilde_v_img_s: Imputed solution image representation [B, L, d_model].
            m_txt_s: Binary indicator for solution text presence [B, L, 1].
            m_img_s: Binary indicator for solution image presence [B, L, 1].

        Returns:
            Tuple of:
            - z_s: Fused solution representation [B, L, d_model].
            - beta_s: Modality attribution weights [B, L, 2] summing to 1.
        """
        if not self.use_visual:
            # Visual modality excluded: solution is pure text
            beta_s = torch.zeros(
                (tilde_v_txt_s.shape[0], tilde_v_txt_s.shape[1], 2),
                device=tilde_v_txt_s.device,
                dtype=tilde_v_txt_s.dtype,
            )
            beta_s[:, :, 0] = 1.0
            return tilde_v_txt_s, beta_s

        # 1. Ablation Mode: Naive Concat (Ablation 4)
        if self.fusion_mode == "naive_concat":
            concat_in = torch.cat([tilde_v_txt_s, tilde_v_img_s], dim=-1)
            z_s = self.proj_naive_concat(concat_in)
            beta_s = torch.full(
                (tilde_v_txt_s.shape[0], tilde_v_txt_s.shape[1], 2),
                0.5,
                device=tilde_v_txt_s.device,
                dtype=tilde_v_txt_s.dtype,
            )
            return z_s, beta_s

        # 2. Baseline Mode: Equal weighting
        if self.fusion_mode == "equal_weight":
            beta_s = torch.full(
                (tilde_v_txt_s.shape[0], tilde_v_txt_s.shape[1], 2),
                0.5,
                device=tilde_v_txt_s.device,
                dtype=tilde_v_txt_s.dtype,
            )
            z_s = 0.5 * (tilde_v_txt_s + tilde_v_img_s)
            return z_s, beta_s

        # 3. Gated Mode: Adaptive Masked Softmax Gate
        u_s = torch.cat([tilde_v_txt_s, tilde_v_img_s, m_txt_s, m_img_s], dim=-1)
        hidden = self.gate_mlp(u_s)
        raw_logits = self.gate_out(hidden)  # [B, L, 2]: (txt, img)

        # Apply soft penalties for missing branches, clamped to max_delta
        delta_txt = torch.clamp(F.softplus(self.raw_delta_txt_s), min=0.0, max=self.max_delta)
        delta_img = torch.clamp(F.softplus(self.raw_delta_img_s), min=0.0, max=self.max_delta)

        penalty_txt = (1.0 - m_txt_s) * delta_txt
        penalty_img = (1.0 - m_img_s) * delta_img

        adj_logits = torch.cat(
            [
                raw_logits[:, :, 0:1] - penalty_txt,
                raw_logits[:, :, 1:2] - penalty_img,
            ],
            dim=-1,
        )

        beta_s = F.softmax(adj_logits, dim=-1)  # [B, L, 2]

        beta_txt = beta_s[:, :, 0:1]
        beta_img = beta_s[:, :, 1:2]
        z_s = beta_txt * tilde_v_txt_s + beta_img * tilde_v_img_s

        return z_s, beta_s
