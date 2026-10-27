"""
Asymmetric Dual-Stream Encoder and Decay-Aware Monotonic Attention Retriever (Subsystem 3).

Implements Section 3.3 of docs/models/ACD-MMKT.md and Step 4 of
plans/acd_mmkt_implementation_and_training_plan.md:
- Stream 1: Exercise Encoder (Causal self-attention over Rasch representations x_t -> tilde_x_t).
- Stream 2: Solution-Augmented Knowledge Acquisition Encoder (y_tau combining response embeddings,
  dynamic temporal features, difficulty scaling, and learned solution absorption W_learn * v_sol).
- Decay-Aware Monotonic Knowledge Retriever:
  - Right-shifted alignment with learnable BOS tokens (tilde_x_0, tilde_y_0) preventing causal leakage.
  - Semantic interference distance d(t, tau') = |t - tau'| * Gamma_{t, tau'} via vectorized cumsum.
  - Strictly positive exponential decay rate theta = Softplus(tilde_theta) > 0 simulating Ebbinghaus forgetting.
  - Synthesizes latent historical knowledge state h_t [B, L, d].
"""

from typing import Dict, Any, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Standard Position-wise Feed-Forward Network with GELU and LayerNorm.
    """

    def __init__(self, d_model: int, d_ff: int = 256, dropout: float = 0.20):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class CausalTransformerEncoder(nn.Module):
    """
    Single-layer Causal Transformer Encoder with multi-head self-attention and FFN.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.20):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff=2 * d_model, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Sequence tensor [B, L, d_model].
            causal_mask: Boolean or float additive attention mask [L, L].
            padding_mask: Optional key padding mask [B, L].

        Returns:
            Encoded sequence [B, L, d_model].
        """
        attn_out, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = self.ln1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_out))
        return x


class AsymmetricMonotonicAttention(nn.Module):
    """
    Subsystem 3: Asymmetric Dual-Stream Encoder and Decay-Aware Monotonic Attention Retriever.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        max_seq_len: int = 200,
        dropout: float = 0.20,
        use_solution: bool = True,
    ):
        """
        Initialize AsymmetricMonotonicAttention module.

        Args:
            d_model: Hidden representation dimension d (default: 128).
            num_heads: Number of attention heads (default: 4).
            max_seq_len: Maximum sequence length (default: 200).
            dropout: Dropout probability.
            use_solution: Whether solution representations are integrated in acquisition stream.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout
        self.use_solution = use_solution

        # 1. Stream 1: Exercise Encoder
        self.exercise_encoder = CausalTransformerEncoder(
            d_model=d_model, num_heads=num_heads, dropout=dropout
        )

        # 2. Stream 2: Solution-Augmented Knowledge Acquisition Components
        # Response Embeddings (r in {0, 1})
        self.r_base_emb = nn.Embedding(2, d_model)
        self.r_var_emb = nn.Embedding(2, d_model)

        # Temporal dynamics projection: [delta_t_lag; delta_t_kc; batch_flag] -> [3] -> [d_model]
        self.time_proj = nn.Linear(3, d_model)

        # Interaction variation MLP: [d_q; e_r_var] -> [2 * d_model] -> [d_model]
        self.mlp_f = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Solution absorption gating parameter: eta_r in (0, 1) conditioned on r in {0, 1}
        self.raw_eta = nn.Parameter(torch.zeros(2, d_model))  # Sigmoid(0) = 0.5 initially

        # Learned solution linear transition
        self.w_learn = nn.Linear(d_model, d_model, bias=False)
        self.ln_y = nn.LayerNorm(d_model)

        self.acquisition_encoder = CausalTransformerEncoder(
            d_model=d_model, num_heads=num_heads, dropout=dropout
        )

        # 3. Monotonic Knowledge Retriever
        # Learnable BOS (Beginning-of-Sequence) tokens for right-shifted alignment
        self.bos_x = nn.Parameter(torch.randn(d_model) / math.sqrt(d_model))
        self.bos_y = nn.Parameter(torch.randn(d_model) / math.sqrt(d_model))

        # Retriever linear projections
        self.w_q_ret = nn.Linear(d_model, d_model)
        self.w_k_ret = nn.Linear(d_model, d_model)
        self.w_v_ret = nn.Linear(d_model, d_model)

        # Learnable decay rate parameter: theta = Softplus(raw_theta) > 0
        # Initialized such that Softplus(0.54) ~ 1.0 (smooth forgetting rate)
        self.raw_theta = nn.Parameter(torch.tensor(0.54, dtype=torch.float32))

        self.ffn_ret = FeedForward(d_model, d_ff=2 * d_model, dropout=dropout)
        self.ln_ret = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _get_causal_mask(
        self, seq_len: int, device: torch.device, is_additive: bool = False
    ) -> torch.Tensor:
        """
        Generate upper-triangular causal mask.
        - If is_additive=False: returns boolean mask (True for masked positions, False for allowed).
          Matches key_padding_mask dtype (bool) to avoid PyTorch mismatched mask warning and enable fast SDPA.
        - If is_additive=True: returns additive float mask (0.0 for allowed, -1e4 for masked positions).
        """
        if is_additive:
            return torch.triu(torch.full((seq_len, seq_len), -1e4, device=device), diagonal=1)
        return torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)

    def build_acquisition_vectors(
        self,
        c_C: torch.Tensor,
        mu_q: torch.Tensor,
        d_q: torch.Tensor,
        r_seq: torch.Tensor,
        delta_t_lag: torch.Tensor,
        delta_t_kc: torch.Tensor,
        batch_flag: torch.Tensor,
        v_sol: Optional[torch.Tensor] = None,
        m_txt_s: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Synthesizes historical knowledge acquisition vector y_tau.

        Args:
            c_C: Core concepts [B, L, d].
            mu_q: Difficulty scalar [B, L, 1].
            d_q: Item variation [B, L, d].
            r_seq: Student binary correctness responses [B, L].
            delta_t_lag: Time lag feature [B, L].
            delta_t_kc: KC repeat distance feature [B, L].
            batch_flag: Batch submission flag [B, L].
            v_sol: Ground truth solution representation [B, L, d].
            m_txt_s: Solution text presence flag [B, L, 1].

        Returns:
            Knowledge acquisition tensor y [B, L, d].
        """
        # 1. Base response interaction with time dynamics
        e_r_base = self.r_base_emb(r_seq)  # [B, L, d]
        time_feats = torch.stack([delta_t_lag, delta_t_kc, batch_flag], dim=-1)  # [B, L, 3]
        e_time = self.time_proj(time_feats)  # [B, L, d]
        e_cr = c_C + e_r_base + e_time  # [B, L, d]

        # 2. Item-difficulty scaled variation interaction
        e_r_var = self.r_var_emb(r_seq)  # [B, L, d]
        f_input = torch.cat([d_q, e_r_var], dim=-1)  # [B, L, 2d]
        f_cr = self.mlp_f(f_input)  # [B, L, d]
        scaled_f = mu_q * f_cr  # [B, L, d]

        # 3. Solution knowledge augmentation term (Equation 89 in ACD-MMKT.md)
        if self.use_solution and v_sol is not None:
            # Gather response-dependent absorption gate eta_r in (0, 1)
            eta = torch.sigmoid(self.raw_eta)[r_seq]  # [B, L, d]
            sol_learned = self.w_learn(v_sol)         # [B, L, d]
            if m_txt_s is not None:
                if m_txt_s.dim() == 2:
                    m_txt_s = m_txt_s.unsqueeze(-1)
                # If m_txt_s == 0, term zeroes out completely (student receives only binary feedback)
                sol_term = m_txt_s.float() * eta * sol_learned  # [B, L, d]
            else:
                sol_term = eta * sol_learned
        else:
            sol_term = torch.zeros_like(e_cr)

        y = self.ln_y(e_cr + scaled_f + sol_term)
        return y


    def retrieve_knowledge_states(
        self,
        tilde_x: torch.Tensor,
        tilde_y: torch.Tensor,
        batch_flag: Optional[torch.Tensor] = None,
        delta_t_lag: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decay-Aware Monotonic Knowledge Retrieval with Right-Shifted Alignment.

        Args:
            tilde_x: Exercise encoder sequence [B, L, d].
            tilde_y: Knowledge acquisition sequence [B, L, d].
            batch_flag: Optional batch submission indicator [B, L].
            delta_t_lag: Optional elapsed time lag [B, L].
            seq_mask: Optional boolean sequence validity mask [B, L].

        Returns:
            Tuple of:
            - h: Latent knowledge states [B, L, d].
            - alpha_ret: Monotonic retrieval attention weights [B, L, L].
        """
        B, L, d = tilde_x.shape
        device = tilde_x.device

        # 1. Right-Shifted Alignment: Prepend BOS tokens at step 0
        # Historical keys and values at step t strictly observe past interactions 0..t-1
        bos_x_batch = self.bos_x.unsqueeze(0).unsqueeze(1).expand(B, 1, d)  # [B, 1, d]
        bos_y_batch = self.bos_y.unsqueeze(0).unsqueeze(1).expand(B, 1, d)  # [B, 1, d]

        # Shifted sequences of length L: position tau' in [0..L-1]
        # tau'=0 is BOS, tau'=1 is interaction 0, ..., tau'=t is interaction t-1
        x_hist = torch.cat([bos_x_batch, tilde_x[:, :-1, :]], dim=1)  # [B, L, d]
        y_hist = torch.cat([bos_y_batch, tilde_y[:, :-1, :]], dim=1)  # [B, L, d]

        # 2. Linear projections
        Q = self.w_q_ret(tilde_x)  # [B, L, d]
        K = self.w_k_ret(x_hist)   # [B, L, d]
        V = self.w_v_ret(y_hist)   # [B, L, d]

        # 3. Scaled dot-product similarity
        scale = 1.0 / math.sqrt(d)
        scores = torch.bmm(Q, K.transpose(1, 2)) * scale  # [B, L, L]

        # Causal mask: query at step t can only attend to key tau' <= t
        # (which maps to history items 0..t-1)
        causal_mask = self._get_causal_mask(L, device=device, is_additive=True)  # [L, L] with -1e4 above diagonal
        scores_masked = scores + causal_mask.unsqueeze(0)      # [B, L, L]

        # 4. Intermediate semantic similarity distribution gamma_{t, tau'}
        gamma = F.softmax(scores_masked, dim=-1)  # [B, L, L]

        # 5. Reverse cumulative interference Gamma_{t, tau'} = sum_{m=tau'+1}^t gamma_{t, m}
        # Vectorized implementation: S_{t, k} = cumsum(gamma, dim=-1)
        # Gamma_{t, tau'} = clamp(S_{t, t} - S_{t, tau'}, min=0.0)
        s_cumsum = torch.cumsum(gamma, dim=-1)  # [B, L, L]
        # Extract diagonal elements S_{t, t} representing cumulative sum up to step t
        s_diag = torch.diagonal(s_cumsum, dim1=1, dim2=2).unsqueeze(-1)  # [B, L, 1]
        gamma_rev = torch.clamp(s_diag - s_cumsum, min=0.0)             # [B, L, L]

        # 6. Cumulative effective temporal distance matrix Delta_tau_eff(t, tau') (Plan Step 4.2)
        # Prevents artificial distance penalties for batch submissions (batch_flag == 1)
        if batch_flag is not None:
            delta_step = torch.ones((B, L), device=device, dtype=torch.float32)
            delta_step[:, 0] = 0.0
            delta_step[:, 1:] = 1.0 - 0.5 * batch_flag[:, 1:].float()
            cum_eff = torch.cumsum(delta_step, dim=-1)  # [B, L], cum_eff[:, 0] = 0.0
            dist_eff = F.relu(cum_eff.unsqueeze(2) - cum_eff.unsqueeze(1))  # [B, L, L]
        else:
            step_idx = torch.arange(L, device=device).unsqueeze(0)          # [1, L]
            dist_eff = F.relu(step_idx.t() - step_idx).unsqueeze(0).float()  # [1, L, L]

        # Cognitive distance: d(t, tau') = Delta_tau_eff(t, tau') * Gamma_{t, tau'}
        d_cognitive = dist_eff * gamma_rev                              # [B, L, L]

        # 7. Decayed attention scores with Softplus strictly positive decay rate
        theta = F.softplus(self.raw_theta) + 1e-4                       # scalar > 0
        decayed_scores = scores_masked - theta * d_cognitive            # [B, L, L]

        # Monotonic retrieval attention weights
        alpha_ret = F.softmax(decayed_scores, dim=-1)                   # [B, L, L]

        # 8. Retrieve context and update knowledge state
        context = torch.bmm(alpha_ret, V)                               # [B, L, d]
        h = self.ln_ret(tilde_x + self.dropout(self.ffn_ret(context)))   # [B, L, d]

        return h, alpha_ret

    def forward(
        self,
        x: torch.Tensor,
        c_C: torch.Tensor,
        mu_q: torch.Tensor,
        d_q: torch.Tensor,
        r_seq: torch.Tensor,
        delta_t_lag: torch.Tensor,
        delta_t_kc: torch.Tensor,
        batch_flag: torch.Tensor,
        v_sol: Optional[torch.Tensor] = None,
        m_txt_s: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Subsystem 3.

        Args:
            x: Rasch representations from Subsystem 2 [B, L, d].
            c_C: Core concepts [B, L, d].
            mu_q: Difficulty scalar [B, L, 1].
            d_q: Item variation [B, L, d].
            r_seq: Student binary correctness responses [B, L].
            delta_t_lag: Time lag feature [B, L].
            delta_t_kc: KC repeat distance feature [B, L].
            batch_flag: Batch submission flag [B, L].
            v_sol: Ground truth solution representation [B, L, d].
            m_txt_s: Solution text presence flag [B, L, 1].
            seq_mask: Sequence validity mask [B, L].

        Returns:
            Dictionary containing:
            - 'h': Latent knowledge states [B, L, d].
            - 'tilde_x': Context-aware exercise representations [B, L, d].
            - 'tilde_y': Context-aware knowledge acquisition representations [B, L, d].
            - 'alpha_ret': Monotonic retrieval attention weights [B, L, L].
            - 'theta_decay': Effective learned decay rate scalar.
        """
        B, L, _ = x.shape
        device = x.device
        causal_mask = self._get_causal_mask(L, device=device)
        padding_mask = ~seq_mask if seq_mask is not None else None

        # 1. Stream 1: Exercise Encoder
        tilde_x = self.exercise_encoder(x, causal_mask=causal_mask, padding_mask=padding_mask)

        # 2. Stream 2: Solution-Augmented Knowledge Acquisition Encoder
        y = self.build_acquisition_vectors(
            c_C=c_C,
            mu_q=mu_q,
            d_q=d_q,
            r_seq=r_seq,
            delta_t_lag=delta_t_lag,
            delta_t_kc=delta_t_kc,
            batch_flag=batch_flag,
            v_sol=v_sol,
            m_txt_s=m_txt_s,
        )
        tilde_y = self.acquisition_encoder(y, causal_mask=causal_mask, padding_mask=padding_mask)

        # 3. Decay-Aware Monotonic Retrieval
        h, alpha_ret = self.retrieve_knowledge_states(
            tilde_x,
            tilde_y,
            batch_flag=batch_flag,
            delta_t_lag=delta_t_lag,
            seq_mask=seq_mask,
        )

        theta_decay = F.softplus(self.raw_theta) + 1e-4


        return {
            "h": h,
            "tilde_x": tilde_x,
            "tilde_y": tilde_y,
            "alpha_ret": alpha_ret,
            "theta_decay": theta_decay,
        }
