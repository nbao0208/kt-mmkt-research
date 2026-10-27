"""
ID-Only Self-Attentive Knowledge Tracing (SAKT) Baseline Model.

Implements the standard non-multimodal SAKT baseline for comparison with MG-MMKT:
- Structural ID and Knowledge Component (KC) representations without multimodal inputs.
- Historical interaction modeling from question ID, KC, and response.
- Strict right-shifted historical alignment with BOS token y_0 to avoid causal leakage.
- Self-attention with causal lower-triangular masking, LayerNorm, and FFN.
- Standard readout MLP predicting correctness logit and probability.
"""

from typing import Dict, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class IDOnlySAKT(nn.Module):
    """
    ID-Only SAKT Baseline model for Knowledge Tracing.
    Utilizes only question IDs, pooled KCs, and binary responses without auxiliary multimodal data.
    """

    def __init__(
        self,
        num_questions: int = 7652,
        num_kcs: int = 1175,
        d_model: int = 128,
        num_heads: int = 4,
        max_seq_len: int = 200,
        dropout: float = 0.20,
        static_item_bank: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Initialize IDOnlySAKT baseline model.

        Args:
            num_questions: Total number of unique questions |Q| (default: 7652).
            num_kcs: Total number of unique knowledge components |K| (default: 1175).
            d_model: Hidden dimension d (default: 128).
            num_heads: Number of attention heads (default: 4).
            max_seq_len: Maximum sequence length (default: 200).
            dropout: Dropout probability across embeddings and attention (default: 0.20).
            static_item_bank: Optional pre-loaded item bank (accepted for pipeline compatibility).
        """
        super().__init__()
        self.num_questions = num_questions
        self.num_kcs = num_kcs
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = d_model // num_heads
        self.dropout_rate = dropout

        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        # Persist architecture configuration dictionary for checkpoint reconstruction
        self.config = {
            "model_name": "sakt",
            "num_questions": num_questions,
            "num_kcs": num_kcs,
            "d_model": d_model,
            "num_heads": num_heads,
            "max_seq_len": max_seq_len,
            "dropout": dropout,
        }

        # 1. Structural Embeddings (Question ID, Knowledge Component, Response)
        self.q_emb = nn.Embedding(num_questions, d_model)
        self.kc_emb = nn.Embedding(num_kcs, d_model)
        self.r_emb = nn.Embedding(2, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # 2. Historical Interaction Projection W_y: [x_tau (d) + e_r (d)] -> d
        self.proj_y = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 3. Learnable BOS History Token y_0 for step 0
        self.y_0 = nn.Parameter(torch.randn(d_model))

        # 4. Positional Embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # 5. Multi-Head Attention Projections
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.proj_out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # 6. Residual & LayerNorm Blocks
        self.ln_attn = nn.LayerNorm(d_model)

        # 7. Position-wise Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln_ffn = nn.LayerNorm(d_model)

        # 8. Readout Prediction Layer: [h_t + x_t] -> logit
        self.readout = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # Precompute causal lower-triangular mask
        causal_mask = torch.triu(torch.full((max_seq_len, max_seq_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)

    def _pool_kc_embeddings(
        self,
        leaf_kcs: torch.Tensor,
        leaf_kc_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform vectorized masked average pooling across leaf Knowledge Components.
        """
        kc_vectors = self.kc_emb(leaf_kcs)  # [B, L, max_leaves, d_model]
        mask_expanded = leaf_kc_mask.unsqueeze(-1).float()  # [B, L, max_leaves, 1]
        summed = (kc_vectors * mask_expanded).sum(dim=2)  # [B, L, d_model]
        counts = torch.clamp(mask_expanded.sum(dim=2), min=1.0)  # [B, L, 1]
        return summed / counts

    def forward(
        self,
        batch_or_q_seq: Union[Dict[str, torch.Tensor], torch.Tensor],
        r_seq: Optional[torch.Tensor] = None,
        leaf_kcs: Optional[torch.Tensor] = None,
        leaf_kc_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for ID-only SAKT baseline.

        Args:
            batch_or_q_seq: Dictionary batch from DataLoader or question sequence tensor [B, L].
            r_seq: Optional response sequence tensor [B, L].
            leaf_kcs: Optional leaf KC indices [B, L, max_leaves].
            leaf_kc_mask: Optional leaf KC mask [B, L, max_leaves].

        Returns:
            Dictionary containing:
            - 'logits': [B, L] prediction logits.
            - 'preds': [B, L] predicted probabilities.
            - 'attn_weights': [B, L, L] causal attention weights.
        """
        if isinstance(batch_or_q_seq, dict):
            q_seq = batch_or_q_seq["q_seq"]
            r_seq = batch_or_q_seq["r_seq"]
            leaf_kcs = batch_or_q_seq["leaf_kcs"]
            leaf_kc_mask = batch_or_q_seq["leaf_kc_mask"]
        else:
            q_seq = batch_or_q_seq

        batch_size, seq_len = q_seq.shape

        # Step 1: Structural Question Representation x_t = e_q + e_C
        e_q = self.q_emb(q_seq)  # [B, L, d]
        e_c = self._pool_kc_embeddings(leaf_kcs, leaf_kc_mask)  # [B, L, d]
        e_r = self.r_emb(r_seq)  # [B, L, d]

        x_q = self.emb_dropout(e_q + e_c)  # [B, L, d]

        # Step 2: Historical interaction y_tau
        y_in = torch.cat([x_q, e_r], dim=-1)  # [B, L, 2d]
        y = self.proj_y(y_in)  # [B, L, d]

        # Step 3: Right-shifted historical alignment with learnable BOS y_0
        y_0_expanded = self.y_0.view(1, 1, self.d_model).expand(batch_size, 1, self.d_model)
        y_shifted = torch.cat([y_0_expanded, y[:, :-1, :]], dim=1)  # [B, L, d]

        # Step 4: Add positional encodings to history
        positions = torch.arange(seq_len, device=q_seq.device)
        pos_emb = self.pos_embedding(positions).unsqueeze(0)  # [1, L, d]
        k_in = y_shifted + pos_emb
        v_in = y_shifted + pos_emb

        # Step 5: Multi-Head Causal Attention
        q = self.proj_q(x_q)  # [B, L, d]
        k = self.proj_k(k_in)  # [B, L, d]
        v = self.proj_v(v_in)  # [B, L, d]

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, L, L]
        mask = self.causal_mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0)
        scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_drop = self.attn_dropout(attn_weights)

        h = torch.matmul(attn_weights_drop, v)  # [B, H, L, head_dim]
        h = h.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        h = self.proj_out(h)

        # Step 6: Residual + LayerNorm
        h_1 = self.ln_attn(h + x_q)

        # Step 7: FFN + Residual + LayerNorm
        h_out = self.ln_ffn(h_1 + self.ffn(h_1))

        # Step 8: Readout
        readout_in = torch.cat([h_out, x_q], dim=-1)
        logits = self.readout(readout_in).squeeze(-1)
        preds = torch.sigmoid(logits)

        avg_attn_weights = attn_weights.mean(dim=1)

        return {
            "logits": logits,
            "preds": preds,
            "attn_weights": avg_attn_weights,
        }
