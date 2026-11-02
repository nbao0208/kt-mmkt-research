"""
Subsystem 3: Dual-Stream Asymmetric Encoder & Context-Aware Monotonic Knowledge Retriever.
Implements the exact mathematical specification for ACD-MMKTv2.

Key features:
1. Exercise Stream Encoder:
   Causal self-attention over Rasch representations {x_1, ..., x_t} to generate {tilde_x_1, ..., tilde_x_t}.
2. Empirical Knowledge Experience Encoder:
   Encodes observed student responses r_tau in {0, 1} modulated by item difficulty mu_{q,tau} and
   variation direction d_tau^q:
   y_tau = LayerNorm(e_(C_tau, r_tau) + mu_{q,tau} * (f_{r_tau} + W_exp^(r_tau) d_tau^q)).
   Strictly complies with Constraint 2: NEVER adds solution representations v_sol directly into student state!
   Followed by causal self-attention to generate {tilde_y_1, ..., tilde_y_{t-1}}.
3. Context-Aware Monotonic Knowledge Retriever:
   Target exercise tilde_x_t queries historical exercises tilde_x_tau and experiences tilde_y_tau for tau < t.
   Cognitive distance d(t, tau) = |t - tau| * sum_{t'=tau}^{t-1} gamma_{t, t'}.
   Monotonic score decay with learnable positive rate theta_decay = Softplus(omega_decay) > 0.
   Outputs pure historical knowledge state h_t (WITHOUT shortcutting target exercise tilde_x_t).
   Safe boundary handling at t = 1 with learnable initial state h_init to prevent NaN gradients.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricMonotonicRetriever(nn.Module):
    """
    Subsystem 3 of ACD-MMKTv2.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        max_seq_len: int = 200,
        dropout: float = 0.15,
    ):
        """
        Initialize AsymmetricMonotonicRetriever.

        Args:
            d_model: Latent dimension (default: 128).
            num_heads: Number of attention heads (default: 4).
            max_seq_len: Maximum sequence window length (default: 200).
            dropout: Dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # 3.1 Exercise Stream Causal Encoder
        self.exercise_causal_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_exercise = nn.LayerNorm(d_model)

        # 3.1 Empirical Knowledge Experience Encoder
        # Parameters for response modulation (r_tau = 0 vs r_tau = 1)
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

        # Causal attention over student experience sequence
        self.experience_causal_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_experience = nn.LayerNorm(d_model)

        # 3.2 Monotonic Knowledge Retriever projections
        self.W_Q_ret = nn.Linear(d_model, d_model)
        self.W_K_ret = nn.Linear(d_model, d_model)
        self.W_V_ret = nn.Linear(d_model, d_model)

        # Positive decay parameter: theta_decay = Softplus(omega_decay) > 0
        # Initialize omega_decay so that Softplus(omega_decay) starts around 0.10
        self.omega_decay = nn.Parameter(torch.tensor([-2.25]))

        # Learnable initial prior knowledge state for t = 1 (empty history)
        self.h_init = nn.Parameter(torch.randn(d_model))
        nn.init.normal_(self.h_init, std=0.02)

        self.norm_h = nn.LayerNorm(d_model)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate upper triangular boolean causal mask (True = masked out)."""
        # Position (i, j) is True if j > i (cannot attend to future)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        return mask

    def forward(
        self,
        x_seq: torch.Tensor,
        c_C_seq: torch.Tensor,
        d_q_seq: torch.Tensor,
        mu_q_seq: torch.Tensor,
        r_seq: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for Subsystem 3.

        Args:
            x_seq: Rasch exercise representations [B, L, d_model].
            c_C_seq: Core concept representations [B, L, d_model].
            d_q_seq: Orthogonal item variation vectors [B, L, d_model].
            mu_q_seq: Item difficulty probabilities [B, L, 1] in (0, 1).
            r_seq: Student binary response labels [B, L] in {0, 1}.
            seq_mask: Sequence validity mask [B, L] (True = active interaction).

        Returns:
            Tuple of:
            - h_seq: Pure historical knowledge state sequence [B, L, d_model].
            - tilde_x_seq: Contextual exercise representation sequence [B, L, d_model].
            - alpha_ret: Historical retrieval attention weights [B, L, L].
        """
        B, L, d = x_seq.shape
        device = x_seq.device

        # Causal mask for self-attention
        causal_mask = self._generate_causal_mask(L, device)
        key_padding_mask = ~seq_mask.bool()  # True = masked out

        # -------------------------------------------------------------
        # 1. Exercise Stream Encoder: {x_tau} -> {tilde_x_tau}
        # -------------------------------------------------------------
        attn_x_out, _ = self.exercise_causal_attn(
            query=x_seq,
            key=x_seq,
            value=x_seq,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
        )
        tilde_x_seq = self.norm_exercise(x_seq + attn_x_out)  # [B, L, d_model]

        # -------------------------------------------------------------
        # 2. Empirical Knowledge Experience Encoder: {r_tau} -> {y_tau} -> {tilde_y_tau}
        # -------------------------------------------------------------
        r_col = r_seq.unsqueeze(-1).float()  # [B, L, 1]

        # e_(C_tau, r_tau) = c_C + r * v_correct + (1 - r) * v_wrong
        base_resp = r_col * self.v_correct + (1.0 - r_col) * self.v_wrong  # [B, L, d]
        e_resp = c_C_seq + base_resp  # [B, L, d]

        # g_exp = r * (f1 + W_exp_1 d^q) + (1 - r) * (f0 + W_exp_0 d^q)
        g_correct = self.f_correct + self.W_exp_1(d_q_seq)  # [B, L, d]
        g_wrong = self.f_wrong + self.W_exp_0(d_q_seq)      # [B, L, d]
        g_exp = r_col * g_correct + (1.0 - r_col) * g_wrong # [B, L, d]

        # y_tau = LayerNorm(e_resp + mu_q * g_exp)
        y_seq = self.norm_y(e_resp + mu_q_seq * g_exp)      # [B, L, d]

        # Causal attention over student experiences
        attn_y_out, _ = self.experience_causal_attn(
            query=y_seq,
            key=y_seq,
            value=y_seq,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
        )
        tilde_y_seq = self.norm_experience(y_seq + attn_y_out)  # [B, L, d_model]

        # -------------------------------------------------------------
        # 3. Context-Aware Monotonic Knowledge Retriever
        # -------------------------------------------------------------
        Q_ret = self.W_Q_ret(tilde_x_seq)  # [B, L, d] (target exercise at step t)
        K_ret = self.W_K_ret(tilde_x_seq)  # [B, L, d] (historical exercises at tau)
        V_ret = self.W_V_ret(tilde_y_seq)  # [B, L, d] (historical experiences at tau)

        # Raw query-key dot product: [B, L, L]
        scale = 1.0 / (d ** 0.5)
        sim_matrix = torch.bmm(Q_ret, K_ret.transpose(1, 2)) * scale  # [B, L, L]

        # Strict past attention mask: tau must be strictly less than t (< t) and seq_mask[tau] == True
        # For t = 0 (step 1), no past steps exist.
        past_mask = torch.tril(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=-1)
        valid_history_mask = past_mask.unsqueeze(0) & seq_mask.unsqueeze(1).bool()  # [B, L, L]

        # Step 3.1: Calculate concept similarity distribution gamma_{t, t'} over tau' < t
        # For numerical stability and avoiding NaN on row 0, fill invalid positions with -1e4
        safe_sim = sim_matrix.masked_fill(~valid_history_mask, -1e4)
        gamma = F.softmax(safe_sim.float(), dim=-1)  # [B, L, L]
        # Zero out completely masked positions
        gamma = gamma * valid_history_mask.float()

        # Step 3.2: Reverse cumulative sum Gamma_{t, tau} = sum_{t'=tau}^{t-1} gamma_{t, t'}
        # Reverse along last dimension, cumsum, then reverse back
        gamma_reversed = torch.flip(gamma, dims=[-1])
        Gamma_reversed = torch.cumsum(gamma_reversed, dim=-1)
        Gamma = torch.flip(Gamma_reversed, dims=[-1])  # [B, L, L]
        Gamma = Gamma * valid_history_mask.float()

        # Step 3.3: Cognitive distance d(t, tau) = |t - tau| * Gamma_{t, tau}
        steps_t = torch.arange(L, device=device).unsqueeze(1)    # [L, 1]
        steps_tau = torch.arange(L, device=device).unsqueeze(0)  # [1, L]
        step_diff = (steps_t - steps_tau).abs().float()          # [L, L]
        cognitive_dist = step_diff.unsqueeze(0) * Gamma          # [B, L, L]

        # Step 3.4: Monotonic decay penalty theta_decay * d(t, tau)
        theta_decay = F.softplus(self.omega_decay) + 1e-6         # strictly > 0
        decayed_scores = (sim_matrix.float() - theta_decay * cognitive_dist)  # [B, L, L]
        decayed_scores = decayed_scores.masked_fill(~valid_history_mask, -1e4)

        # Softmax to obtain historical retrieval weights alpha_ret
        alpha_ret = F.softmax(decayed_scores, dim=-1)  # [B, L, L]
        alpha_ret = alpha_ret * valid_history_mask.float()

        # Step 3.5: Synthesize historical knowledge state h_t = sum_{tau=1}^{t-1} alpha_{t, tau} V_ret[tau]
        h_history = torch.bmm(alpha_ret.to(dtype=V_ret.dtype), V_ret)  # [B, L, d]
        h_normed = self.norm_h(h_history)

        # Boundary condition: at t = 0 (first interaction), student has no history
        # Override with learnable prior h_init
        h_init_expanded = self.h_init.unsqueeze(0).unsqueeze(1).expand(B, 1, d)
        h_seq = torch.cat([h_init_expanded, h_normed[:, 1:, :]], dim=1)  # [B, L, d]

        return h_seq, tilde_x_seq, alpha_ret

