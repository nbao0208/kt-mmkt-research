"""
Layer 3: Causal Interaction Modeling and SAKT Sequence Prediction.

Implements the mathematical formulation specified in docs/models/MG-MMKT.md:
- Posterior historical interaction encoding y_tau from question x_tau, response e_r, and solution z_tau^s.
- Right-shifted historical alignment with learnable BOS token y_0 to mathematically prevent causal leakage.
- Self-Attentive Knowledge Tracing (SAKT) with causal lower-triangular attention mask, positional encoding,
  residual connections, LayerNorm, and position-wise FFN.
- Readout MLP predicting correctness logit ell_t and probability p_hat_t.
- Top-K historical attention extraction for first-order explainability.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSAKTLayer(nn.Module):
    """
    Self-Attentive Knowledge Tracing (SAKT) module with strict causal masking
    and right-shifted historical interaction alignment.
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
        Initialize CausalSAKTLayer.

        Args:
            d_model: Hidden representation dimension d (default: 128).
            num_heads: Number of attention heads (default: 4).
            max_seq_len: Maximum sequence length (default: 200).
            dropout: Dropout probability across attention and FFN (default: 0.20).
            use_solution: Whether to incorporate solution representation z^s in history (default: True).
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = d_model // num_heads
        self.use_solution = use_solution

        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        # 1. Historical Interaction Projection W_y
        # Inputs: x_tau (d) + e_r (d) + [z_tau^s (d) + m_txt_s (1) if use_solution]
        in_dim_y = (3 * d_model + 1) if use_solution else (2 * d_model)
        self.proj_y = nn.Sequential(
            nn.Linear(in_dim_y, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 2. Learnable BOS History Token y_0 for step 0 (ensures non-degenerate attention at t=1)
        # Initialized with standard normal distribution N(0, 1) to match hidden state norm scale (~sqrt(d_model))
        self.y_0 = nn.Parameter(torch.randn(d_model))

        # 3. Learnable Positional Embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # 4. Multi-Head Attention Projections
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.proj_out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # 5. SAKT Residual & LayerNorm Blocks
        self.ln_attn = nn.LayerNorm(d_model)

        # 6. Position-wise Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln_ffn = nn.LayerNorm(d_model)

        # 7. Readout Prediction Layer: [h_t + x_t] -> logit
        self.readout = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # Precompute causal mask: lower triangular is 0, strictly upper triangular is -inf
        causal_mask = torch.triu(torch.full((max_seq_len, max_seq_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)

    def forward(
        self,
        x_q: torch.Tensor,
        e_r: torch.Tensor,
        z_s: Optional[torch.Tensor] = None,
        m_txt_s: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for Causal SAKT Layer.

        Args:
            x_q: Fused question representation [B, L, d_model].
            e_r: Student response embedding [B, L, d_model].
            z_s: Fused solution representation [B, L, d_model] (optional if use_solution=False).
            m_txt_s: Solution text presence flag [B, L, 1] (optional if use_solution=False).

        Returns:
            Tuple of:
            - logits: Prediction logits [B, L].
            - preds: Predicted probabilities [B, L].
            - attn_weights: SAKT causal attention weights [B, L, L] averaged across heads.
        """
        batch_size, seq_len, _ = x_q.shape

        # Step 1: Compute historical interaction representations y_tau
        if self.use_solution and z_s is not None and m_txt_s is not None:
            y_in = torch.cat([x_q, e_r, z_s, m_txt_s], dim=-1)
        else:
            y_in = torch.cat([x_q, e_r], dim=-1)

        y = self.proj_y(y_in)  # [B, L, d_model]

        # Step 2: Right-Shift Historical Alignment with y_0 BOS token
        # Insert y_0 at index 0 and shift y[:, :L-1] to positions 1..L-1
        # This guarantees row t in attention ONLY attends to {y_0, y_1, ..., y_{t-1}}!
        y_0_expanded = self.y_0.view(1, 1, self.d_model).expand(batch_size, 1, self.d_model)
        y_shifted = torch.cat([y_0_expanded, y[:, :-1, :]], dim=1)  # [B, L, d_model]

        # Step 3: Add Positional Embeddings to shifted historical sequence
        positions = torch.arange(seq_len, device=x_q.device)
        pos_emb = self.pos_embedding(positions).unsqueeze(0)  # [1, L, d_model]
        k_in = y_shifted + pos_emb
        v_in = y_shifted + pos_emb

        # Step 4: Multi-Head Projections
        # Queries come from current question x_q; Keys & Values come from historical k_in, v_in
        q = self.proj_q(x_q)  # [B, L, d_model]
        k = self.proj_k(k_in)  # [B, L, d_model]
        v = self.proj_v(v_in)  # [B, L, d_model]

        # Reshape for multi-head attention: [B, H, L, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Step 5: Scaled Dot-Product Attention with Causal Masking
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, L, L]

        # Apply causal mask up to current seq_len
        mask = self.causal_mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
        scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)  # [B, H, L, L]
        attn_weights_drop = self.attn_dropout(attn_weights)

        # Context aggregation
        h = torch.matmul(attn_weights_drop, v)  # [B, H, L, head_dim]
        h = h.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        h = self.proj_out(h)

        # Step 6: SAKT Residual Connection & LayerNorm (connects Query with context)
        h_1 = self.ln_attn(h + x_q)

        # Step 7: Position-wise Feed-Forward Network & Residual LayerNorm
        h_out = self.ln_ffn(h_1 + self.ffn(h_1))  # [B, L, d_model]

        # Step 8: Readout Prediction Layer: [h_out, x_q] -> logit
        readout_in = torch.cat([h_out, x_q], dim=-1)  # [B, L, 2 * d_model]
        logits = self.readout(readout_in).squeeze(-1)  # [B, L]
        preds = torch.sigmoid(logits)

        # Average attention weights across heads for interpretability
        avg_attn_weights = attn_weights.mean(dim=1)  # [B, L, L]

        return logits, preds, avg_attn_weights
