"""
Disentangled Causal Attention with Relative Forgetting and Multi-Interactive Readout (Layer 4 of MG-MMKT v2).

Implements:
1. Query-Key Disentangled Causal Attention:
   - Queries Q_t and Keys K_tau projected purely from question representations x.
   - Values V_tau projected from full historical interaction representations y_tau.
   - Slices (L-1) x (L-1) matrix to guarantee non-empty causal triangular masking (zero NaNs at t = 1).
2. Per-Head Monotonic Temporal Forgetting Decay:
   - Learnable decay rates theta_h = Softplus(tilde_theta_h) > 0 with multi-scale initializations.
   - Decays attention scores by -theta_h * log(1 + (t - tau)).
3. Boundary condition at t = 1:
   - Assigns learnable canonical prior knowledge state h_1 = h_0.
4. Multi-Interactive Readout Layer (5d):
   - Combines [h_t ; x_t ; hat_z_t^s ; (h_t * x_t) ; (h_t * hat_z_t^s)] -> logit ell_t and prob p_hat_t.
"""

from typing import Dict, Any, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DisentangledCausalAttention(nn.Module):
    """
    Disentangled Causal Attention with Head-Specific Monotonic Relative Forgetting (Layer 4).
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        max_seq_len: int = 200,
        dropout: float = 0.20,
        use_disentangled_qk: bool = True,
        use_relative_forgetting: bool = True,
    ):
        """
        Args:
            d_model: Hidden dimension d (default: 128).
            num_heads: Number of attention heads H (default: 4).
            max_seq_len: Maximum sequence window length L (default: 200).
            dropout: Dropout probability (default: 0.20).
            use_disentangled_qk: Whether Q, K come purely from x (default: True).
            use_relative_forgetting: Whether to apply monotonic temporal decay (default: True).
        """
        super().__init__()
        assert d_model % num_heads == 0, f"d_model {d_model} must be divisible by num_heads {num_heads}"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_h = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.use_disentangled_qk = use_disentangled_qk
        self.use_relative_forgetting = use_relative_forgetting

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # Multi-scale initializations for forgetting rates theta_h: [0.10, 0.30, 0.70, 1.50]
        init_thetas = [0.10, 0.30, 0.70, 1.50]
        if num_heads != 4:
            # Linear interpolation in log space if num_heads is different
            init_thetas = [float(val) for val in torch.logspace(-1.0, 0.2, num_heads)]
        else:
            init_thetas = init_thetas[:num_heads]

        # Invert softplus to initialize raw parameters: softplus^-1(theta) = ln(exp(theta) - 1)
        raw_thetas = [math.log(math.exp(th) - 1.0) for th in init_thetas]
        self.raw_theta = nn.Parameter(torch.tensor(raw_thetas, dtype=torch.float32))

        # Positional embedding fallback for ablation mode (w/o relative forgetting)
        if not self.use_relative_forgetting:
            self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Post-attention FFN and LayerNorm
        self.ffn_attn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.ln_attn = nn.LayerNorm(d_model)

        # Canonical prior knowledge state h_0 for boundary step t = 1
        self.h_0 = nn.Parameter(torch.randn(d_model))

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass using (L-1) x (L-1) causal history slicing.

        Args:
            x: Question representations [B, L, d].
            y: Historical interaction representations [B, L, d].
            seq_mask: Interaction validity mask [B, L] (bool or float).

        Returns:
            Tuple of:
                - h: Retrieved student knowledge states [B, L, d].
                - gamma_bar: Head-averaged attention evidence matrix [B, L, L].
        """
        B, L, _ = x.shape
        device = x.device

        # If sequence length is 1 (boundary corner case), return prior state h_0
        if L == 1:
            h = self.h_0.view(1, 1, self.d_model).expand(B, 1, self.d_model)
            gamma_bar = torch.zeros(B, 1, 1, device=device)
            return h, gamma_bar

        # 1. Projections
        # Queries for target steps t in {2, ..., L} -> indices 1 to L-1 (length L-1)
        x_target = x[:, 1:, :]  # [B, L-1, d]
        x_hist = x[:, :-1, :]    # [B, L-1, d]
        y_hist = y[:, :-1, :]    # [B, L-1, d]

        # Inject absolute positional embeddings if relative forgetting is disabled (Ablation 5)
        if not self.use_relative_forgetting:
            target_pos = torch.arange(1, L, device=device).unsqueeze(0)  # [1, L-1]
            hist_pos = torch.arange(0, L - 1, device=device).unsqueeze(0)  # [1, L-1]
            x_target = x_target + self.pos_emb(target_pos)
            x_hist = x_hist + self.pos_emb(hist_pos)
            y_hist = y_hist + self.pos_emb(hist_pos)

        Q = self.w_q(x_target).view(B, L - 1, self.num_heads, self.d_h).transpose(1, 2)  # [B, H, L-1, d_h]

        # Keys and Values for historical steps tau in {1, ..., L-1} -> indices 0 to L-2 (length L-1)
        if self.use_disentangled_qk:
            K = self.w_k(x_hist).view(B, L - 1, self.num_heads, self.d_h).transpose(1, 2)  # [B, H, L-1, d_h]
        else:
            # Ablation mode: Key projected from history y instead of question x
            K = self.w_k(y_hist).view(B, L - 1, self.num_heads, self.d_h).transpose(1, 2)

        V = self.w_v(y_hist).view(B, L - 1, self.num_heads, self.d_h).transpose(1, 2)  # [B, H, L-1, d_h]

        # 2. Scaled dot-product attention scores on (L-1) x (L-1)
        # scores: [B, H, L-1, L-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_h)

        # 3. Monotonic Relative Forgetting Decay: -theta_h * log(1 + (t - tau))
        if self.use_relative_forgetting:
            # Row index i corresponds to t = i + 2; Column index j corresponds to tau = j + 1
            # Step difference: t - tau = (i + 2) - (j + 1) = i - j + 1
            # log(1 + (t - tau)) = log(2 + i - j)
            i_idx = torch.arange(L - 1, device=device).unsqueeze(1)  # [L-1, 1]
            j_idx = torch.arange(L - 1, device=device).unsqueeze(0)  # [1, L-1]
            diff = i_idx - j_idx                                     # [L-1, L-1]
            dist_matrix = torch.log(2.0 + diff.clamp(min=0).float()) # [L-1, L-1]

            theta = F.softplus(self.raw_theta)  # [H] > 0
            decay_term = theta.view(1, self.num_heads, 1, 1) * dist_matrix.unsqueeze(0).unsqueeze(0)
            scores = scores - decay_term

        # 4. Strict causal lower-triangular mask: j <= i (never all -inf, guarantees no NaNs)
        causal_mask = torch.tril(torch.ones(L - 1, L - 1, device=device, dtype=torch.bool))  # [L-1, L-1]
        # Mask out padding key positions if any (column 0 is always valid in real student sequences)
        key_valid = seq_mask[:, :-1].unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L-1]
        combined_mask = causal_mask.unsqueeze(0).unsqueeze(0) & key_valid.bool()

        scores_masked = scores.masked_fill(~combined_mask, float("-inf"))
        attn_weights = F.softmax(scores_masked, dim=-1)  # [B, H, L-1, L-1]

        # Dropout on attention weights
        attn_weights_dropped = self.dropout(attn_weights)

        # 5. Output combination: [B, H, L-1, d_h] -> [B, L-1, d]
        attn_out = torch.matmul(attn_weights_dropped, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L - 1, self.d_model)
        tilde_h_2_to_L = self.w_o(attn_out)

        # Post-attention Residual FFN: h_t = LN_attn(tilde_h + Dropout(FFN_attn(tilde_h)))
        # Note: Do NOT add x_t into h_t to preserve pure student latent knowledge state
        h_2_to_L = self.ln_attn(tilde_h_2_to_L + self.dropout(self.ffn_attn(tilde_h_2_to_L)))

        # 6. Boundary at t = 1: Prepend prior knowledge state h_0
        h_1 = self.h_0.view(1, 1, self.d_model).expand(B, 1, self.d_model)
        h = torch.cat([h_1, h_2_to_L], dim=1)  # [B, L, d]

        # 7. Construct head-averaged attention evidence matrix gamma_bar: [B, L, L]
        # Row 0 is 0; Rows 1..L-1 and Cols 0..L-2 contain the attention weights
        mean_attn = attn_weights.mean(dim=1)  # [B, L-1, L-1]
        gamma_bar = torch.zeros(B, L, L, device=device, dtype=mean_attn.dtype)
        gamma_bar[:, 1:, :-1] = mean_attn

        return h, gamma_bar


class MultiInteractiveReadout(nn.Module):
    """
    Multi-Interactive Readout Layer (Layer 4).
    Computes [h_t ; x_t ; hat_z_t^s ; (h_t * x_t) ; (h_t * hat_z_t^s)] -> logit and prob.
    """

    def __init__(
        self,
        d_model: int = 128,
        dropout: float = 0.20,
        use_lupi: bool = True,
    ):
        """
        Args:
            d_model: Hidden dimension d (default: 128).
            dropout: Dropout probability (default: 0.20).
            use_lupi: Whether to include implicit cognitive demand hat_z_t^s in readout.
        """
        super().__init__()
        self.d_model = d_model
        self.use_lupi = use_lupi

        # Input dimension: 5d with LUPI, 3d without LUPI
        in_dim = 5 * d_model if use_lupi else 3 * d_model

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        hat_z_s: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Multi-Interactive Readout.

        Args:
            h: Retrieved knowledge states [B, L, d].
            x: Question representations [B, L, d].
            hat_z_s: Estimated cognitive demand [B, L, d] (optional if use_lupi=False).

        Returns:
            Tuple of:
                - logits: Unnormalized prediction logits [B, L].
                - probs: Predicted probabilities [B, L] in (0, 1).
        """
        # Concept alignment interaction
        hadamard_hx = h * x

        if self.use_lupi and hat_z_s is not None:
            hadamard_hz = h * hat_z_s
            u_read = torch.cat([h, x, hat_z_s, hadamard_hx, hadamard_hz], dim=-1)  # [B, L, 5d]
        else:
            u_read = torch.cat([h, x, hadamard_hx], dim=-1)  # [B, L, 3d]

        logits = self.mlp(u_read).squeeze(-1)  # [B, L]
        probs = torch.sigmoid(logits)           # [B, L]
        return logits, probs

