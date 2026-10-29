"""
Masked-Gated Multimodal Knowledge Tracing (MG-MMKT) Model.

Implements the complete end-to-end architecture specified in docs/models/MG-MMKT.md:
- Tier 1 static item bank registration and fast GPU indexing.
- Layer 1: Frozen multimodal projection and learnable modality imputation.
- Layer 2: Modality-aware masked softmax gating for question and solution modalities.
- Layer 3: Causal historical interaction modeling with SAKT sequence prediction.
- Simultaneous prediction logits, probabilities, and first-order explainability tensors.
"""

from typing import Dict, Any, Optional, Union, Tuple
import torch
import torch.nn as nn

from src.models.modules.modality_imputation import ModalityImputationLayer
from src.models.modules.gated_fusion import (
    QuestionMaskedGatedFusion,
    SolutionMaskedGatedFusion,
)
from src.models.modules.causal_attention import CausalSAKTLayer


class MGMMKT(nn.Module):
    """
    End-to-End Masked-Gated Multimodal Knowledge Tracing (MG-MMKT) model.
    """

    def __init__(
        self,
        num_questions: int = 7652,
        num_kcs: int = 1175,
        d_raw_txt: int = 768,
        d_raw_img: int = 768,
        d_model: int = 128,
        num_heads: int = 4,
        max_seq_len: int = 200,
        dropout: float = 0.20,
        use_solution: bool = True,
        use_visual: bool = True,
        use_learnable_imputation: bool = True,
        use_gated_fusion: bool = True,
        fusion_mode: Optional[str] = None,
        static_item_bank: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Initialize MGMMKT model.

        Args:
            num_questions: Total number of unique questions |Q| (default: 7652).
            num_kcs: Total number of unique knowledge components |K| (default: 1175).
            d_raw_txt: Dimension of raw frozen text embeddings (default: 768).
            d_raw_img: Dimension of raw frozen vision embeddings (default: 768).
            d_model: Hidden dimension d (default: 128).
            num_heads: Number of attention heads (default: 4).
            max_seq_len: Maximum sequence length (default: 200).
            dropout: Dropout probability (default: 0.20).
            use_solution: Whether to incorporate solution representations (default: True).
            use_visual: Whether to incorporate image modalities (default: True).
            use_learnable_imputation: Whether to use learnable e_null tokens (default: True).
            use_gated_fusion: Whether to use adaptive gated fusion (default: True).
            fusion_mode: Explicit fusion mode ('gated', 'equal_weight', or 'naive_concat').
            static_item_bank: Optional pre-loaded item bank to register as model buffers.
        """
        super().__init__()
        self.num_questions = num_questions
        self.num_kcs = num_kcs
        self.d_raw_txt = d_raw_txt
        self.d_raw_img = d_raw_img
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout
        self.use_solution = use_solution
        self.use_visual = use_visual

        # Determine effective fusion mode
        if fusion_mode is not None:
            self.fusion_mode = fusion_mode
        elif use_gated_fusion:
            self.fusion_mode = "gated"
        else:
            self.fusion_mode = "equal_weight"

        self.use_gated_fusion = (self.fusion_mode == "gated")

        # In naive concat mode (Ablation 4), learnable imputation is disabled (zero-padding is used)
        if self.fusion_mode == "naive_concat":
            self.use_learnable_imputation = False
        else:
            self.use_learnable_imputation = use_learnable_imputation

        # Save architecture configuration dictionary for checkpoint reconstruction
        self.config = {
            "model_name": "mg_mmkt",
            "num_questions": num_questions,
            "num_kcs": num_kcs,
            "d_raw_txt": d_raw_txt,
            "d_raw_img": d_raw_img,
            "d_model": d_model,
            "num_heads": num_heads,
            "max_seq_len": max_seq_len,
            "dropout": dropout,
            "use_solution": self.use_solution,
            "use_visual": self.use_visual,
            "use_learnable_imputation": self.use_learnable_imputation,
            "use_gated_fusion": self.use_gated_fusion,
            "fusion_mode": self.fusion_mode,
        }

        # 1. Structural Embeddings (ID, KC, Response)
        self.q_emb = nn.Embedding(num_questions, d_model)
        self.kc_emb = nn.Embedding(num_kcs, d_model)
        self.r_emb = nn.Embedding(2, d_model)

        # Input branch regularization: applies dropout uniformly across structural and multimodal inputs
        # (prevents question ID co-adaptation / overfitting as emphasized in Plan Step 5.3)
        self.input_dropout = nn.Dropout(dropout)

        # 2. Layer 1: Feature Projection and Modality Imputation
        self.imputation = ModalityImputationLayer(
            d_raw_txt=d_raw_txt,
            d_raw_img=d_raw_img,
            d_model=d_model,
            use_learnable_imputation=self.use_learnable_imputation,
        )

        # 3. Layer 2: Modality-Aware Gated Fusion (supports 'gated', 'equal_weight', and 'naive_concat')
        self.q_fusion = QuestionMaskedGatedFusion(
            d_model=d_model,
            use_visual=self.use_visual,
            use_gated_fusion=self.use_gated_fusion,
            fusion_mode=self.fusion_mode,
        )
        self.s_fusion = SolutionMaskedGatedFusion(
            d_model=d_model,
            use_visual=self.use_visual,
            use_gated_fusion=self.use_gated_fusion,
            fusion_mode=self.fusion_mode,
        )

        # 4. Layer 3: Causal Sequence Modeling and SAKT Prediction
        self.causal_sakt = CausalSAKTLayer(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            use_solution=use_solution,
        )

        # 5. Register frozen static item bank buffers if provided
        if static_item_bank is not None:
            self.init_static_buffers(static_item_bank)

    def init_static_buffers(self, static_item_bank: Dict[str, torch.Tensor]) -> None:
        """
        Register frozen feature tensors as non-trainable non-persistent model buffers.
        Ensures automatic device transfer (CPU -> GPU/MPS) when model.to(device) is called,
        while preventing frozen static features (~90 MB) from bloating checkpoint files.
        """
        self.register_buffer("v_txt_q_bank", static_item_bank["mg_v_txt_q"].float(), persistent=False)
        self.register_buffer("v_img_q_bank", static_item_bank["mg_v_img_q"].float(), persistent=False)
        self.register_buffer("v_txt_s_bank", static_item_bank["mg_v_txt_s"].float(), persistent=False)
        self.register_buffer("v_img_s_bank", static_item_bank["mg_v_img_s"].float(), persistent=False)

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """
        Load state dict with backward compatibility for legacy checkpoints containing
        frozen static item bank buffers.
        """
        static_keys = {"v_txt_q_bank", "v_img_q_bank", "v_txt_s_bank", "v_img_s_bank"}
        cleaned_state_dict = {k: v for k, v in state_dict.items() if k not in static_keys}
        return super().load_state_dict(cleaned_state_dict, strict=strict)

    def _pool_kc_embeddings(
        self,
        leaf_kcs: torch.Tensor,
        leaf_kc_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform vectorized masked average pooling across leaf Knowledge Components.

        Args:
            leaf_kcs: Leaf KC indices [B, L, max_leaves].
            leaf_kc_mask: Binary mask for valid KCs [B, L, max_leaves].

        Returns:
            Pooled KC embedding tensor e_C [B, L, d_model].
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
        modality_masks: Optional[torch.Tensor] = None,
        leaf_kcs: Optional[torch.Tensor] = None,
        leaf_kc_mask: Optional[torch.Tensor] = None,
        v_txt_q: Optional[torch.Tensor] = None,
        v_img_q: Optional[torch.Tensor] = None,
        v_txt_s: Optional[torch.Tensor] = None,
        v_img_s: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for MG-MMKT.

        Accepts either a collated batch dictionary or individual tensors.

        Returns:
            Dictionary containing:
            - 'logits': [B, L] prediction logits.
            - 'preds': [B, L] predicted probabilities.
            - 'alpha_q': [B, L, 3] question modality attribution weights.
            - 'beta_s': [B, L, 2] solution modality attribution weights.
            - 'attn_weights': [B, L, L] SAKT causal attention weights.
        """
        if isinstance(batch_or_q_seq, dict):
            q_seq = batch_or_q_seq["q_seq"]
            r_seq = batch_or_q_seq["r_seq"]
            modality_masks = batch_or_q_seq["modality_masks"]
            leaf_kcs = batch_or_q_seq["leaf_kcs"]
            leaf_kc_mask = batch_or_q_seq["leaf_kc_mask"]
        else:
            q_seq = batch_or_q_seq

        # Step 0: Lookup frozen static features if not explicitly provided
        if v_txt_q is None:
            assert hasattr(self, "v_txt_q_bank"), "Static buffers not initialized. Call init_static_buffers first."
            v_txt_q = self.v_txt_q_bank[q_seq]
            v_img_q = self.v_img_q_bank[q_seq]
            v_txt_s = self.v_txt_s_bank[q_seq]
            v_img_s = self.v_img_s_bank[q_seq]

        # Step 1: Structural Embeddings
        e_q = self.q_emb(q_seq)  # [B, L, d]
        e_c = self._pool_kc_embeddings(leaf_kcs, leaf_kc_mask)  # [B, L, d]
        e_r = self.r_emb(r_seq)  # [B, L, d]

        # Step 2: Layer 1 - Modality Imputation
        tilde_txt_q, tilde_img_q, tilde_txt_s, tilde_img_s = self.imputation(
            v_txt_q=v_txt_q,
            v_img_q=v_img_q,
            v_txt_s=v_txt_s,
            v_img_s=v_img_s,
            modality_masks=modality_masks,
        )

        # Apply input branch dropout uniformly across all branches entering fusion
        # (fulfills Plan Step 5.3 to prevent question ID co-adaptation / overfitting)
        e_q = self.input_dropout(e_q)
        e_c = self.input_dropout(e_c)
        tilde_txt_q = self.input_dropout(tilde_txt_q)
        tilde_img_q = self.input_dropout(tilde_img_q)
        tilde_txt_s = self.input_dropout(tilde_txt_s)
        tilde_img_s = self.input_dropout(tilde_img_s)

        # Step 3: Layer 2 - Modality-Aware Gated Fusion
        m_img_q = modality_masks[:, :, 0:1]
        m_txt_s = modality_masks[:, :, 1:2]
        m_img_s = modality_masks[:, :, 2:3]

        x_q, alpha_q = self.q_fusion(
            e_q=e_q,
            e_c=e_c,
            tilde_v_txt_q=tilde_txt_q,
            tilde_v_img_q=tilde_img_q,
            m_img_q=m_img_q,
        )

        z_s, beta_s = self.s_fusion(
            tilde_v_txt_s=tilde_txt_s,
            tilde_v_img_s=tilde_img_s,
            m_txt_s=m_txt_s,
            m_img_s=m_img_s,
        )

        # Step 4: Layer 3 - Causal Sequence Modeling and Prediction
        logits, preds, attn_weights = self.causal_sakt(
            x_q=x_q,
            e_r=e_r,
            z_s=z_s if self.use_solution else None,
            m_txt_s=m_txt_s if self.use_solution else None,
        )

        return {
            "logits": logits,
            "preds": preds,
            "alpha_q": alpha_q,
            "beta_s": beta_s,
            "attn_weights": attn_weights,
        }

    def get_gate_deltas(self) -> Dict[str, float]:
        """
        Aggregate learned soft penalty deltas from question and solution fusion modules.
        Enables monitoring delta values during training to ensure bounded convergence.
        """
        deltas = {}
        if hasattr(self, "q_fusion") and hasattr(self.q_fusion, "get_deltas"):
            deltas.update(self.q_fusion.get_deltas())
        if hasattr(self, "s_fusion") and hasattr(self.s_fusion, "get_deltas"):
            deltas.update(self.s_fusion.get_deltas())
        return deltas
