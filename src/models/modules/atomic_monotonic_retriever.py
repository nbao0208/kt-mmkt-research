"""
Subsystem 3: Dual-Stream Asymmetric Encoder & Unpolluted Monotonic Knowledge Retriever.
Implements the exact mathematical specification for ACD-MMKTv3 (Section 3, Subsystem 3).

Key features:
1. Contextual Exercise Stream:
   - Causal Self-Attention with lower-triangular mask (tau <= t) over Rasch representations {x_1, ..., x_t}
     to generate contextual exercises {tilde_x_1, ..., tilde_x_t}.
2. Unpolluted Atomic Empirical Experience Stream:
   - Encodes observed student responses r_tau in {0, 1} modulated by item difficulty mu_{q,tau} and variation direction d_tau^q:
     e_(C_tau, r_tau) = c_C,tau + r_tau * v_correct + (1 - r_tau) * v_wrong
     y_tau = e_(C_tau, r_tau) + mu_{q,tau} * (r_tau * (f_1 + W_exp^(1) d_tau^q) + (1 - r_tau) * (f_0 + W_exp^(0) d_tau^q)) + W_post * v_post,tau^diag.
   - Pointwise LayerNorm norm_y is applied.
   - CRITICAL: y_tau is kept strictly ATOMIC per step without causal self-attention or recurrent mixing,
     preserving 100% historical attribution purity.
   - Ablation 5 flag (polluted_experience_attn): passes y_tau through causal self-attention if enabled.
3. Context-Aware Monotonic Knowledge Retriever:
   - Query: q_t = W_Q tilde_x_t, Key: k_tau = W_K tilde_x_tau, Value: v_tau = W_V y_tau.
   - Distance matrix d(t, tau) = |t - tau| + gamma_time * delta_t_lag + gamma_kc * delta_t_kc > 0 for tau < t.
   - Monotonic exponential score decay with learnable positive rate theta_decay = Softplus(omega_decay) > 0.
   - Strict lower-triangular causal mask (tau < t).
   - Safe boundary handling at t = 1 with learnable initial state h_init to completely eliminate NaN gradients.
   - Target exercise tilde_x_t is strictly NOT added to h_t as a shortcut connection, preserving d(theta)/d(tilde_mu) = 0.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AtomicMonotonicRetriever(nn.Module):
    """
    Subsystem 3 of ACD-MMKTv3: Dual-Stream Asymmetric Encoder & Unpolluted Monotonic Knowledge Retriever.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        max_seq_len: int = 200,
        dropout: float = 0.15,
        polluted_experience_attn: bool = False,
        d_post_diag: int = 0,
    ):
        """
        Initialize AtomicMonotonicRetriever.

        Args:
            d_model: Latent embedding dimension (128).
            num_heads: Number of attention heads (4).
            max_seq_len: Maximum sequence window length (200).
            dropout: Dropout probability.
            polluted_experience_attn: Ablation 5 flag (w/ causal self-attention over experience).
            d_post_diag: Dimension of optional diagnostic post-response vector (0 = none/zero).
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.polluted_experience_attn = polluted_experience_attn
        self.head_dim = d_model // num_heads

        # 3.1 Exercise Stream Causal Encoder
        self.exercise_causal_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_exercise = nn.LayerNorm(d_model)

        # 3.1 Empirical Knowledge Experience Stream
        self.v_correct = nn.Parameter(torch.randn(d_model))
        self.v_wrong = nn.Parameter(torch.randn(d_model))
        self.f_correct = nn.Parameter(torch.randn(d_model))
        self.f_wrong = nn.Parameter(torch.randn(d_model))
        nn.init.normal_(self.v_correct, std=0.02)
        nn.init.normal_(self.v_wrong, std=0.02)
        nn.init.normal_(self.f_correct, std=0.02)
        nn.init.normal_(self.f_wrong, std=0.02)

        self.W_exp_1 = nn.Linear(d_model, d_model, bias=False)
        self.W_exp_0 = nn.Linear(d_model, d_model, bias=False)
        self.norm_y = nn.LayerNorm(d_model)

        # Optional post-diagnostic projection
        if d_post_diag > 0:
            self.W_post = nn.Linear(d_post_diag, d_model, bias=False)
        else:
            self.W_post = None

        # Ablation 5: Polluted Experience Stream Attention
        if self.polluted_experience_attn:
            self.experience_causal_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm_experience = nn.LayerNorm(d_model)

        # 3.2 Monotonic Knowledge Retriever
        self.W_Q_ret = nn.Linear(d_model, d_model, bias=False)
        self.W_K_ret = nn.Linear(d_model, d_model, bias=False)
        self.W_V_ret = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Positive decay rates: theta_decay = Softplus(omega_decay) > 0 (per head)
        # Initialized around ~0.10: Softplus(-2.25) ~ 0.10
        self.omega_decay = nn.Parameter(torch.full((num_heads,), -2.25))

        # Positive distance multipliers: gamma_time >= 0, gamma_kc >= 0
        self.omega_time = nn.Parameter(torch.tensor([-1.0]))
        self.omega_kc = nn.Parameter(torch.tensor([-1.0]))

        # Learnable initial prior knowledge state for t = 1 (empty history)
        self.h_init = nn.Parameter(torch.randn(d_model))
        nn.init.normal_(self.h_init, std=0.02)

        self.norm_h = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def compute_experience_stream(
        self,
        c_C_seq: torch.Tensor,
        mu_q_seq: torch.Tensor,
        d_q_seq: torch.Tensor,
        r_seq: torch.Tensor,
        v_post_diag: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute atomic empirical experience representations [B, L, d_model].
        Under the default architecture, each step tau depends strictly on inputs at step tau
        without cross-step pollution (d(y_tau) / d(r_{tau'}) = 0 for tau' != tau).
        """
        B, L, _ = c_C_seq.shape
        device = c_C_seq.device

        r_float = r_seq.unsqueeze(-1).float()          # [B, L, 1]
        mu_float = mu_q_seq.view(B, L, 1).float()       # [B, L, 1]

        # Concept-response base: e_(C, r) = c_C + r * v_correct + (1 - r) * v_wrong
        base_exp = c_C_seq + r_float * self.v_correct + (1.0 - r_float) * self.v_wrong  # [B, L, d_model]

        # Difficulty-modulated variation
        mod_1 = self.f_correct + self.W_exp_1(d_q_seq)  # [B, L, d_model]
        mod_0 = self.f_wrong + self.W_exp_0(d_q_seq)    # [B, L, d_model]
        mod_term = mu_float * (r_float * mod_1 + (1.0 - r_float) * mod_0)  # [B, L, d_model]

        y_atomic = base_exp + mod_term

        # Add optional post-diagnostic representation
        if self.W_post is not None and v_post_diag is not None:
            y_atomic = y_atomic + self.W_post(v_post_diag.float())

        # Pointwise LayerNorm on atomic steps
        y_atomic = self.norm_y(y_atomic)  # [B, L, d_model]

        # Ablation 5 check: if polluted_experience_attn is enabled, pass through causal self-attention
        if self.polluted_experience_attn:
            causal_mask_exp = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
            try:
                exp_out, _ = self.experience_causal_attn(
                    y_atomic, y_atomic, y_atomic,
                    attn_mask=causal_mask_exp,
                    need_weights=False,
                )
            except NotImplementedError:
                # Fallback for MPS / environments where scaled_dot_product_attention lacks dropout support
                exp_out, _ = self.experience_causal_attn(
                    y_atomic, y_atomic, y_atomic,
                    attn_mask=causal_mask_exp,
                    need_weights=True,
                )
            y_seq = self.norm_experience(y_atomic + self.dropout(exp_out))
        else:
            y_seq = y_atomic  # Strict atomic experience preservation

        return y_seq

    def forward(
        self,
        x_seq: torch.Tensor,
        c_C_seq: torch.Tensor,
        mu_q_seq: torch.Tensor,
        d_q_seq: torch.Tensor,
        r_seq: torch.Tensor,
        delta_t_lag: Optional[torch.Tensor] = None,
        delta_t_kc: Optional[torch.Tensor] = None,
        v_post_diag: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for Subsystem 3.

        Args:
            x_seq: Rasch item representations [B, L, d_model].
            c_C_seq: Core concept sequence [B, L, d_model].
            mu_q_seq: Item difficulty probabilities [B, L] or [B, L, 1].
            d_q_seq: Orthogonal item variations [B, L, d_model].
            r_seq: Student binary responses [B, L] in {0, 1}.
            delta_t_lag: Normalized lag time [B, L] in [0, 1].
            delta_t_kc: Normalized KC spacing [B, L] in [0, 1].
            v_post_diag: Optional diagnostic post-response features.

        Returns:
            Tuple of:
            - h_seq: Knowledge state sequence [B, L, d_model].
            - tilde_x_seq: Contextual exercise sequence [B, L, d_model].
            - alpha_ret: Retrieval attention weights [B, L, L].
        """
        B, L, D = x_seq.shape
        device = x_seq.device

        # -------------------------------------------------------------
        # 3.1 Exercise Stream: Causal Self-Attention (tau <= t)
        # -------------------------------------------------------------
        # Causal mask: True for positions to IGNORE (strictly upper triangle: tau > t)
        causal_mask_ex = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
        try:
            ex_out, _ = self.exercise_causal_attn(
                x_seq, x_seq, x_seq,
                attn_mask=causal_mask_ex,
                need_weights=False,
            )
        except NotImplementedError:
            # Fallback for MPS / environments where scaled_dot_product_attention lacks dropout support
            ex_out, _ = self.exercise_causal_attn(
                x_seq, x_seq, x_seq,
                attn_mask=causal_mask_ex,
                need_weights=True,
            )
        tilde_x = self.norm_exercise(x_seq + self.dropout(ex_out))  # [B, L, d_model]

        # -------------------------------------------------------------
        # 3.1 Empirical Experience Stream
        # -------------------------------------------------------------
        y_seq = self.compute_experience_stream(
            c_C_seq=c_C_seq,
            mu_q_seq=mu_q_seq,
            d_q_seq=d_q_seq,
            r_seq=r_seq,
            v_post_diag=v_post_diag,
        )

        # -------------------------------------------------------------
        # 3.2 Monotonic Knowledge Retriever
        # -------------------------------------------------------------
        # Project Q, K, V
        Q = self.W_Q_ret(tilde_x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d_h]
        K = self.W_K_ret(tilde_x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d_h]
        V = self.W_V_ret(y_seq).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)    # [B, H, L, d_h]

        # Dot product similarity
        scale_ret = 1.0 / (self.head_dim ** 0.5)
        scores_content = torch.matmul(Q, K.transpose(-2, -1)) * scale_ret  # [B, H, L, L]

        # Cognitive distance matrix d(t, tau) on strict lower triangle (tau < t)
        # Position index distance: |t - tau|
        step_indices = torch.arange(L, device=device)
        pos_dist = torch.clamp(step_indices.unsqueeze(1) - step_indices.unsqueeze(0), min=0).float()  # [L, L]

        gamma_time = F.softplus(self.omega_time)  # >= 0
        gamma_kc = F.softplus(self.omega_kc)      # >= 0

        if delta_t_lag is not None:
            # delta_t_lag is [B, L], broadcast over tau dimension at historical position: [B, 1, 1, L]
            lag_term = (gamma_time * delta_t_lag).unsqueeze(1).unsqueeze(2)
        else:
            lag_term = 0.0

        if delta_t_kc is not None:
            # delta_t_kc is [B, L], broadcast over tau dimension: [B, 1, 1, L]
            kc_term = (gamma_kc * delta_t_kc).unsqueeze(1).unsqueeze(2)
        else:
            kc_term = 0.0

        # Full distance: [B, 1, L, L]
        d_mat = pos_dist.unsqueeze(0).unsqueeze(0) + lag_term + kc_term

        # Exponential decay: theta_decay = Softplus(omega_decay) > 0
        theta_decay = F.softplus(self.omega_decay).view(1, self.num_heads, 1, 1)  # [1, H, 1, 1]
        decay_penalty = theta_decay * d_mat  # [B, H, L, L]

        # Total unmasked attention scores
        ret_scores = scores_content - decay_penalty  # [B, H, L, L]

        # Strict lower-triangular causal mask: tau < t
        # Mask out diagonal and upper triangle (tau >= t)
        strict_causal_mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=0)  # [L, L]
        ret_scores_masked = ret_scores.masked_fill(strict_causal_mask.unsqueeze(0).unsqueeze(0), -1e9)

        # For t = 0 (first row), history is empty: replace with safe 0s before softmax to avoid NaN
        ret_scores_safe = ret_scores_masked.clone()
        ret_scores_safe[:, :, 0, :] = 0.0

        attn_weights = F.softmax(ret_scores_safe.float(), dim=-1).type_as(ret_scores)  # [B, H, L, L]

        # Zero out invalid upper triangle and row 0 strictly
        attn_weights = attn_weights.masked_fill(strict_causal_mask.unsqueeze(0).unsqueeze(0), 0.0)
        attn_weights[:, :, 0, :] = 0.0

        # Aggregate values
        h_multihead = torch.matmul(attn_weights, V)  # [B, H, L, d_h]
        h_multihead = h_multihead.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, d_model]
        h_ret = self.norm_h(self.out_proj(h_multihead))  # [B, L, d_model]

        # Safe boundary overwrite: at t = 0 (first interaction, 1-based t = 1), set h_1 = h_init
        h_seq = h_ret.clone()
        h_seq[:, 0, :] = self.h_init.unsqueeze(0).expand(B, D)

        # Average attention weights across heads for explainability
        alpha_ret = attn_weights.mean(dim=1)  # [B, L, L]

        return h_seq, tilde_x, alpha_ret

