r"""
Subsystem 4.1: Identifiable Multi-Trait Psychometric Interaction & Monotonic 2PL-IRT Readout.
Implements the exact mathematical specification for ACD-MMKTv3 (Section 3, Subsystem 4).

Key features:
1. Item Discrimination a_{q_t} > 0:
   - Evaluated strictly from target exercise intrinsic attributes [c_C,t || v_stem,t || v_sol,t] in R^{3d}:
     a_{q_t} = Softplus(w_a^T [c_C,t || v_stem,t || v_sol,t] + b_a) + delta_a > 0.
   - Enforces delta_a = 0.1 positive discrimination floor.
   - Strictly independent of student history and item difficulty: d(a_{q_t}) / d(tilde_mu_{q_t}) = 0.
2. Learner Ability theta_{u,t} in R:
   - Evaluated via Hadamard interaction between knowledge state h_t and difficulty-free item attributes
     k_target,t = [c_C,t || d_t^q] in R^{2d}:
     z_{u,q,t} = tanh(W_h h_t \odot W_k [c_C,t || d_t^q] + b_z) in R^d
     theta_{u,t} = w_theta^T z_{u,q,t} + b_theta in R.
   - Strictly non-identifiable coupling is broken: d(theta_{u,t}) / d(tilde_mu_{q_t}) = 0 analytically.
3. Monotonic 2PL-IRT Prediction:
   - ell_t = a_{q_t} * (theta_{u,t} - tilde_mu_{q_t}),
   - p_hat_t = sigmoid(ell_t) in (0, 1).
   - Guarantees 100% mathematical decision faithfulness: (p_hat_t >= 0.5) <=> (theta_{u,t} >= tilde_mu_{q_t}).
   - Global strict monotonicity: d(p_hat)/d(theta) > 0 and d(p_hat)/d(tilde_mu) < 0.
4. Support for Ablation 6 (use_deep_irt=False):
   - Replaces 2PL-IRT with standard black-box MLP on [h_t || tilde_x_t].
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Identifiable2PLIRT(nn.Module):
    """
    Subsystem 4.1 of ACD-MMKTv3: Identifiable Monotonic 2PL-IRT Readout Layer.
    """

    def __init__(
        self,
        d_model: int = 128,
        delta_a: float = 0.1,
        use_deep_irt: bool = True,
        use_solution: bool = True,
        initial_bias_a: float = 0.5413,  # Softplus(0.5413) ~ 1.0 (standard 2PL baseline)
    ):
        """
        Initialize Identifiable2PLIRT.

        Args:
            d_model: Latent embedding dimension (128).
            delta_a: Positive discrimination floor (default: 0.1).
            use_deep_irt: Whether to use monotonic 2PL-IRT readout (Ablation 6 flag).
            use_solution: Whether solution stream is available (Ablation 2 flag).
            initial_bias_a: Initial bias for item discrimination.
        """
        super().__init__()
        self.d_model = d_model
        self.delta_a = delta_a
        self.use_deep_irt = use_deep_irt
        self.use_solution = use_solution

        if self.use_deep_irt:
            # Multi-trait Hadamard interaction: W_h h_t \odot W_k [c_C || d^q]
            self.W_h = nn.Linear(d_model, d_model, bias=False)
            self.W_k = nn.Linear(2 * d_model, d_model, bias=False)
            self.b_z = nn.Parameter(torch.zeros(d_model))

            # Learner ability theta readout
            self.fc_theta = nn.Linear(d_model, 1, bias=True)
            nn.init.normal_(self.fc_theta.weight, std=0.02)
            nn.init.constant_(self.fc_theta.bias, 0.0)

            # Item discrimination a_q readout from [c_C || v_stem || v_sol]
            disc_in_dim = 3 * d_model if self.use_solution else 2 * d_model
            self.fc_a = nn.Linear(disc_in_dim, 1, bias=True)
            nn.init.normal_(self.fc_a.weight, std=0.02)
            nn.init.constant_(self.fc_a.bias, initial_bias_a)
        else:
            # Ablation 6: Standard Black-box MLP readout [h_t || tilde_x_t]
            self.mlp_blackbox = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )

    def forward(
        self,
        h_t: torch.Tensor,
        c_C_t: torch.Tensor,
        d_q_t: torch.Tensor,
        v_stem_t: torch.Tensor,
        v_sol_t: torch.Tensor,
        tilde_mu_t: torch.Tensor,
        tilde_x_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for 2PL-IRT prediction.

        Args:
            h_t: Historical knowledge state [..., d_model].
            c_C_t: Core concept representation [..., d_model].
            d_q_t: Orthogonal item variation [..., d_model].
            v_stem_t: Stem multimodal representation [..., d_model].
            v_sol_t: Solution representation [..., d_model].
            tilde_mu_t: Total difficulty on real logit scale [...] or [..., 1].
            tilde_x_t: Optional contextual exercise representation for ablation [..., d_model].

        Returns:
            Tuple of:
            - logits: Unnormalized prediction logits [...].
            - probs: Predicted probabilities in (0, 1) [...].
            - theta: Student ability on real logit scale [...].
            - a_disc: Item discrimination strictly >= delta_a [...].
        """
        tilde_mu_flat = tilde_mu_t.squeeze(-1).float()  # [...]

        if self.use_deep_irt:
            # 1. Learner Ability theta_{u,t} via Hadamard interaction
            proj_h = self.W_h(h_t)                                    # [..., d_model]
            k_target = torch.cat([c_C_t, d_q_t], dim=-1)             # [..., 2*d_model]
            proj_k = self.W_k(k_target)                              # [..., d_model]
            z_int = torch.tanh(proj_h * proj_k + self.b_z)           # [..., d_model]
            theta = self.fc_theta(z_int).squeeze(-1).float()         # [...]

            # 2. Item Discrimination a_{q_t} strictly positive
            if self.use_solution:
                disc_in = torch.cat([c_C_t, v_stem_t, v_sol_t], dim=-1)  # [..., 3*d_model]
            else:
                disc_in = torch.cat([c_C_t, v_stem_t], dim=-1)          # [..., 2*d_model]

            raw_a = self.fc_a(disc_in).squeeze(-1).float()           # [...]
            a_disc = F.softplus(raw_a) + self.delta_a                # [...] strictly >= delta_a

            # 3. Exact 2PL-IRT prediction
            logits = a_disc * (theta - tilde_mu_flat)                 # [...]
            probs = torch.sigmoid(logits)                            # [...] in (0, 1)

            return logits, probs, theta, a_disc
        else:
            # Ablation 6: Blackbox MLP on [h_t || tilde_x_t]
            if tilde_x_t is None:
                tilde_x_t = c_C_t
            bb_in = torch.cat([h_t, tilde_x_t], dim=-1)
            logits = self.mlp_blackbox(bb_in).squeeze(-1).float()
            probs = torch.sigmoid(logits)
            theta = logits
            a_disc = torch.ones_like(logits)
            return logits, probs, theta, a_disc
