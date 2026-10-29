"""
Response-Conditioned Historical Encoding and Implicit Cognitive Demand Estimator (Layer 3 of MG-MMKT v2).

Implements:
1. Dual-Pathway Historical Interaction Fusion (for tau < t):
   - Mastery Demonstration Pathway (Delta y^(1)) activated when r_tau = 1.
   - Remedial Intervention Pathway (Delta y^(0)) activated when r_tau = 0.
   - Response gating: r_tau * Delta y^(1) + (1 - r_tau) * Delta y^(0).
   - Historical modulation gate G_hist and residual LayerNorm -> y_tau.
2. Implicit Cognitive Demand Estimator f_cog(x_t) (for target step t):
   - Two-layer MLP predicting implicit cognitive solution complexity hat_z_t^s solely from question x_t.
   - Evaluated without actual solution at inference time to prevent causal leakage.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CognitiveHistoryFusionLayer(nn.Module):
    """
    Response-Conditioned Historical Encoding (tau < t) and Implicit Cognitive Estimator (t).
    """

    def __init__(
        self,
        d_model: int = 128,
        use_dual_pathway_history: bool = True,
        use_lupi: bool = True,
    ):
        """
        Args:
            d_model: Hidden dimension d (default: 128).
            use_dual_pathway_history: Whether to separate Mastery and Remedial pathways (default: True).
            use_lupi: Whether to include implicit cognitive estimator f_cog(x_t) (default: True).
        """
        super().__init__()
        self.d_model = d_model
        self.use_dual_pathway_history = use_dual_pathway_history
        self.use_lupi = use_lupi

        # Context vector dimension: [x ; z^s ; (x * z^s) ; m_txt_s ; m_img_s] -> 3d + 2
        u_sol_dim = 3 * d_model + 2

        if self.use_dual_pathway_history:
            # 1a. Mastery Demonstration Pathway (activated when r = 1)
            self.mastery_mlp = nn.Sequential(
                nn.Linear(u_sol_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            # 1b. Remedial Intervention Pathway (activated when r = 0)
            self.remedial_mlp = nn.Sequential(
                nn.Linear(u_sol_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            # Ablation 3: Single shared MLP for all responses
            self.shared_mlp = nn.Sequential(
                nn.Linear(u_sol_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

        # Historical modulation gate: W_hg [x ; e_r ; z^s] + b_hg -> (0, 1)^d
        self.w_hg = nn.Linear(3 * d_model, d_model)
        self.ln_hist = nn.LayerNorm(d_model)

        # 2. Implicit Cognitive Demand Estimator: f_cog(x_t) -> R^d
        if self.use_lupi:
            self.cog_mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.ln_cog = nn.LayerNorm(d_model)

    def forward_history(
        self,
        x: torch.Tensor,
        z_s: torch.Tensor,
        e_r: torch.Tensor,
        r: torch.Tensor,
        m_txt_s: torch.Tensor,
        m_img_s: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute historical interaction representations y_tau for the sequence.

        Args:
            x: Question representations [B, L, d].
            z_s: Solution representations [B, L, d].
            e_r: Response embeddings [B, L, d].
            r: Binary student correctness responses [B, L] in {0, 1}.
            m_txt_s: Solution text presence flags [B, L, 1].
            m_img_s: Solution image presence flags [B, L, 1].

        Returns:
            y: Historical interaction representations [B, L, d].
        """
        # Question-solution interaction context with Hadamard alignment
        hadamard_xz = x * z_s
        u_sol = torch.cat([x, z_s, hadamard_xz, m_txt_s, m_img_s], dim=-1)  # [B, L, 3d + 2]

        r_exp = r.unsqueeze(-1).float()  # [B, L, 1]

        if self.use_dual_pathway_history:
            delta_y_mastery = self.mastery_mlp(u_sol)   # [B, L, d]
            delta_y_remedial = self.remedial_mlp(u_sol)  # [B, L, d]
            # Exact binary convex combination: ensures zero gradient to inactive branch
            delta_y = r_exp * delta_y_mastery + (1.0 - r_exp) * delta_y_remedial
        else:
            delta_y = self.shared_mlp(u_sol)

        # Historical modulation gate
        concat_xrz = torch.cat([x, e_r, z_s], dim=-1)  # [B, L, 3d]
        g_hist = torch.sigmoid(self.w_hg(concat_xrz))  # [B, L, d]

        # y_tau = LN_hist(x_tau + e_r,tau + G_hist,tau * Delta y_tau)
        y = self.ln_hist(x + e_r + g_hist * delta_y)  # [B, L, d]
        return y

    def forward_cognitive_demand(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Estimate implicit cognitive demand hat_z_t^s solely from question x_t.

        Args:
            x: Question representation [B, L, d].

        Returns:
            hat_z_s: Estimated cognitive solution vector [B, L, d] or None if LUPI disabled.
        """
        if not self.use_lupi:
            return None
        cog_raw = self.cog_mlp(x)
        hat_z_s = self.ln_cog(cog_raw)
        return hat_z_s

