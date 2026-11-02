r"""
Subsystem 4.1: Multi-Trait Hadamard Psychometric Interaction & Monotonic 2PL-IRT Readout.
Implements the exact mathematical specification for ACD-MMKTv2.

Key features:
1. Multi-dimensional Hadamard interaction between student state h_t and exercise representation tilde_x_t:
   z_{u,q,t} = tanh(W_h h_t \odot W_x tilde_x_t + b_z).
2. Student proficiency theta_{u,t} = w_theta^T z_{u,q,t} + b_theta on real logit scale R.
3. Item discrimination a_{q,t} = Softplus(w_a^T tilde_x_t + b_a) + 1e-4 > 0 strictly positive.
4. Exact 2PL-IRT prediction equation:
   ell_t = a_{q,t} * (theta_{u,t} - tilde_mu_{q,t}),
   p_hat_t = sigmoid(ell_t).
   CRITICAL: Subtracts the real logit difficulty tilde_mu_{q,t} (NOT the probability mu_q),
   guaranteeing 100% faithfulness: p_hat_t >= 0.5 <=> theta_{u,t} >= tilde_mu_{q,t}.
5. Support for Ablation 5 (use_deep_irt=False) replacing 2PL-IRT with a standard black-box MLP.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Monotonic2PLIRT(nn.Module):
    """
    Subsystem 4.1 of ACD-MMKTv2: Monotonic 2PL-IRT Readout Layer.
    """

    def __init__(
        self,
        d_model: int = 128,
        use_deep_irt: bool = True,
        initial_bias_a: float = 0.5413,  # Softplus(0.5413) ~ 1.0 (standard 2PL discrimination baseline)
    ):
        """
        Initialize Monotonic2PLIRT.

        Args:
            d_model: Latent dimension (default: 128).
            use_deep_irt: Whether to use monotonic 2PL-IRT formulation (Ablation 5 flag).
            initial_bias_a: Initial bias for item discrimination to start near 1.0.
        """
        super().__init__()
        self.d_model = d_model
        self.use_deep_irt = use_deep_irt

        if self.use_deep_irt:
            # Multi-dimensional Hadamard interaction projections
            self.W_h = nn.Linear(d_model, d_model, bias=False)
            self.W_x = nn.Linear(d_model, d_model, bias=False)
            self.b_z = nn.Parameter(torch.zeros(d_model))

            # Student proficiency readout theta_{u,t}
            self.fc_theta = nn.Linear(d_model, 1, bias=True)
            nn.init.normal_(self.fc_theta.weight, std=0.02)
            nn.init.constant_(self.fc_theta.bias, 0.0)

            # Item discrimination readout a_{q,t} > 0
            self.fc_a = nn.Linear(d_model, 1, bias=True)
            nn.init.normal_(self.fc_a.weight, std=0.02)
            nn.init.constant_(self.fc_a.bias, initial_bias_a)
        else:
            # Ablation 5: Standard Black-box MLP readout [h_t || tilde_x_t]
            self.mlp_blackbox = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )

    def forward(
        self,
        h_t: torch.Tensor,
        tilde_x_t: torch.Tensor,
        tilde_mu_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for 2PL-IRT prediction.

        Args:
            h_t: Student historical knowledge state [..., d_model].
            tilde_x_t: Contextual exercise representation [..., d_model].
            tilde_mu_t: Additive total difficulty on real logit scale [...] or [..., 1].

        Returns:
            Tuple of:
            - logits: Unnormalized prediction logits [...].
            - probs: Predicted probabilities of correct response [...] in (0, 1).
            - theta: Student proficiency [...] on real logit scale.
            - a_disc: Item discrimination [...] strictly > 0.
        """
        if self.use_deep_irt:
            # 1. Multi-trait Hadamard Interaction
            proj_h = self.W_h(h_t)
            proj_x = self.W_x(tilde_x_t)
            z_int = torch.tanh(proj_h * proj_x + self.b_z)  # [..., d_model]

            # 2. Student Proficiency theta_{u,t} in R (enforce fp32 for numerical stability)
            theta = self.fc_theta(z_int).squeeze(-1).float()         # [...]

            # 3. Item Discrimination a_{q,t} > 0
            raw_a = self.fc_a(tilde_x_t).squeeze(-1).float()         # [...]
            a_disc = F.softplus(raw_a) + 1e-4                        # [...] strictly > 0

            # 4. Strict 2PL-IRT prediction: ell = a * (theta - tilde_mu)
            diff_logit = tilde_mu_t.squeeze(-1) if tilde_mu_t.dim() > theta.dim() else tilde_mu_t
            diff_logit = diff_logit.float()
            logits = a_disc * (theta - diff_logit)                   # [...] float32
            probs = torch.sigmoid(logits)                            # [...] in (0, 1) float32

            return logits, probs, theta, a_disc

        else:
            # Ablation 5: Black-box MLP readout
            concat_input = torch.cat([h_t, tilde_x_t], dim=-1)
            logits = self.mlp_blackbox(concat_input).squeeze(-1).float()
            probs = torch.sigmoid(logits)
            dummy_theta = torch.zeros_like(logits)
            dummy_a = torch.ones_like(logits)

            return logits, probs, dummy_theta, dummy_a
