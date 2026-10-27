"""
Neural Cognitive Diagnosis and DeepIRT Interaction Readout (Subsystem 4).

Implements Section 3.4 of docs/models/ACD-MMKT.md and Step 5 of
plans/acd_mmkt_implementation_and_training_plan.md:
- Student Proficiency theta_{u, t} in (0, 1) derived from knowledge state h_t.
- Item Discrimination a_{q, t} > 0 derived from exercise representation tilde_x_t.
- Item Difficulty mu_{q, t} in (0, 1) estimated in Subsystem 2.
- Scaled 2-parameter IRT logit: ell_irt = kappa * a_{q, t} * (theta_{u, t} - mu_{q, t}) with kappa = 4.0.
- Regularized Deep Residual Branch ell_res initialized with small variance to prevent IRT distortion.
- Ante-hoc psychometric monotonicity: d(ell)/d(theta) > 0 and d(ell)/d(mu) < 0 strictly.
- Ablation 5 support: black-box MLP readout when use_deep_irt is False.
"""

from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepIRTReadout(nn.Module):
    """
    Subsystem 4: Neural Cognitive Diagnosis Readout combining IRT interpretability
    with deep residual learning.
    """

    def __init__(
        self,
        d_model: int = 128,
        irt_scale: float = 4.0,
        dropout: float = 0.20,
        use_deep_irt: bool = True,
    ):
        """
        Initialize DeepIRTReadout.

        Args:
            d_model: Hidden dimension d (default: 128).
            irt_scale: Logit scaling constant kappa (default: 4.0).
            dropout: Dropout probability.
            use_deep_irt: Whether to use DeepIRT formulation (True) or black-box MLP (False, Ablation 5).
        """
        super().__init__()
        self.d_model = d_model
        self.irt_scale = irt_scale
        self.use_deep_irt = use_deep_irt

        if self.use_deep_irt:
            # 1. Student Proficiency Readout: h_t -> theta_{u, t} in (0, 1)
            self.fc_theta = nn.Linear(d_model, 1)

            # 2. Item Discrimination Readout: tilde_x_t -> a_{q, t} > 0
            self.fc_a = nn.Linear(d_model, 1)

            # 3. Residual Deep Term: [h_t; tilde_x_t] -> [2 * d_model] -> ell_res
            self.res_mlp = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.fc_res = nn.Linear(d_model, 1)
            # Initialize with small variance so IRT branch dominates early learning (Plan Step 5.3)
            nn.init.normal_(self.fc_res.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.fc_res.bias)
        else:
            # Ablation 5: Black-Box MLP Readout
            self.blackbox_mlp = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

    def forward(
        self,
        h: torch.Tensor,
        tilde_x: torch.Tensor,
        mu_q: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Subsystem 4.

        Args:
            h: Latent historical knowledge states [B, L, d].
            tilde_x: Context-aware exercise representations [B, L, d].
            mu_q: Item difficulty probabilities [B, L, 1] or [B, L].

        Returns:
            Dictionary containing:
            - 'logits': [B, L] unnormalized logits hat_ell_t.
            - 'preds': [B, L] predicted probabilities in (0, 1).
            - 'theta': [B, L] student proficiency theta_{u, t} in (0, 1).
            - 'a_q': [B, L] item discrimination a_{q, t} > 0.
            - 'mu_q': [B, L] item difficulty mu_{q, t} in (0, 1).
            - 'ell_irt': [B, L] IRT logit component.
            - 'ell_res': [B, L] residual deep logit component.
            - 'psy_gap': [B, L] psychometric gap (theta - mu).
        """
        if mu_q.ndim == 2:
            mu_q_2d = mu_q
        else:
            mu_q_2d = mu_q.squeeze(-1)

        if not self.use_deep_irt:
            # Ablation 5: Standard Black-Box MLP Readout
            combined = torch.cat([h, tilde_x], dim=-1)
            logits = self.blackbox_mlp(combined).squeeze(-1)  # [B, L]
            preds = torch.sigmoid(logits)
            return {
                "logits": logits,
                "preds": preds,
                "theta": torch.zeros_like(logits),
                "a_q": torch.ones_like(logits),
                "mu_q": mu_q_2d,
                "ell_irt": logits,
                "ell_res": torch.zeros_like(logits),
                "psy_gap": torch.zeros_like(logits),
            }

        # 1. Student Proficiency theta in (0, 1)
        theta = torch.sigmoid(self.fc_theta(h)).squeeze(-1)  # [B, L]

        # 2. Item Discrimination a > 0 via Softplus
        a_q = (F.softplus(self.fc_a(tilde_x)) + 1e-4).squeeze(-1)  # [B, L]

        # 3. Psychometric Gap
        psy_gap = theta - mu_q_2d  # [B, L]

        # 4. IRT Logit
        ell_irt = self.irt_scale * a_q * psy_gap  # [B, L]

        # 5. Deep Residual Term
        res_hidden = self.res_mlp(torch.cat([h, tilde_x], dim=-1))
        ell_res = self.fc_res(res_hidden).squeeze(-1)  # [B, L]

        # 6. Combined Prediction
        logits = ell_irt + ell_res  # [B, L]
        preds = torch.sigmoid(logits)  # [B, L]

        return {
            "logits": logits,
            "preds": preds,
            "theta": theta,
            "a_q": a_q,
            "mu_q": mu_q_2d,
            "ell_irt": ell_irt,
            "ell_res": ell_res,
            "psy_gap": psy_gap,
        }
