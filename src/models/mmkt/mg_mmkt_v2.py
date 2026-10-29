"""
Masked-Gated Multimodal Knowledge Tracing v2 (MG-MMKT v2) Model.

Implements the end-to-end architecture specified in docs/models/MG-MMKTv2.md:
- Tier 1 static item bank registration and fast on-device indexing.
- Layer 1: Structural ID-KC decoupling with frequency gating and KC-conditioned imputation inside LayerNorm.
- Layer 2: Hierarchical dual-stage gated fusion with Softmax gating, channel GLU, and post-fusion residual FFN.
- Layer 3: Response-conditioned dual-pathway history (Mastery vs Remedial) & implicit cognitive demand estimator (LUPI).
- Layer 4: Query-Key disentangled causal attention with head-specific monotonic relative forgetting & 5d multi-interactive readout.
- Full 3-level explainability tensor output.
"""

from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules.kc_conditioned_imputation import (
    StructuralDecouplingLayer,
    KCConditionedImputationLayer,
)
from src.models.modules.dual_stage_gating import (
    QuestionDualStageGating,
    SolutionDualStageGating,
)
from src.models.modules.cognitive_history_fusion import (
    CognitiveHistoryFusionLayer,
)
from src.models.modules.disentangled_causal_attention import (
    DisentangledCausalAttention,
    MultiInteractiveReadout,
)


class MGMMKTv2(nn.Module):
    """
    Complete MG-MMKT v2 Model Architecture.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        num_questions: int = 7652,
        num_kcs: int = 1175,
        d_raw_txt: int = 768,
        d_raw_img: int = 768,
        d_model: int = 128,
        d_ffn: int = 256,
        num_heads: int = 4,
        max_seq_len: int = 200,
        p_id_drop: float = 0.20,
        dropout: float = 0.20,
        use_lupi: bool = True,
        use_dual_pathway_history: bool = True,
        use_glu: bool = True,
        use_post_fusion_ffn: bool = True,
        use_disentangled_qk: bool = True,
        use_relative_forgetting: bool = True,
        use_structural_decoupling: bool = True,
        use_kc_imputation: bool = True,
        use_solution: bool = True,
        use_visual: bool = True,
        static_item_bank: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Initialize MGMMKTv2 model.

        Args:
            num_questions: Total number of unique questions |Q| (default: 7652).
            num_kcs: Total number of unique knowledge components |K| (default: 1175).
            d_raw_txt: Dimension of raw frozen text embeddings (default: 768).
            d_raw_img: Dimension of raw frozen vision embeddings (default: 768).
            d_model: Latent dimension d (default: 128).
            d_ffn: Dimension of feed-forward expansion (default: 256).
            num_heads: Number of causal attention heads (default: 4).
            max_seq_len: Maximum sequence length (default: 200).
            p_id_drop: Question ID Dropout probability (default: 0.20).
            dropout: Model dropout probability (default: 0.20).
            use_lupi: Enable privileged cognitive demand distillation f_cog(x_t).
            use_dual_pathway_history: Enable separate Mastery & Remedial pathways.
            use_glu: Enable channel-wise GLU feature gating.
            use_post_fusion_ffn: Enable post-fusion residual FFN.
            use_disentangled_qk: Enable Query-Key disentangled from history Values.
            use_relative_forgetting: Enable monotonic distance decay -theta * log(1 + t - tau).
            use_structural_decoupling: Enable frequency gating & step ID dropout.
            use_kc_imputation: Enable KC-conditioning in missing modality imputation.
            use_solution: Master switch for solution representations.
            use_visual: Master switch for visual representations.
            static_item_bank: Optional pre-loaded item bank to register as model buffers.
        """
        super().__init__()
        self.num_questions = num_questions
        self.num_kcs = num_kcs
        self.d_raw_txt = d_raw_txt
        self.d_raw_img = d_raw_img
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.p_id_drop = p_id_drop
        self.dropout_rate = dropout
        self.use_lupi = use_lupi
        self.use_dual_pathway_history = use_dual_pathway_history
        self.use_glu = use_glu
        self.use_post_fusion_ffn = use_post_fusion_ffn
        self.use_disentangled_qk = use_disentangled_qk
        self.use_relative_forgetting = use_relative_forgetting
        self.use_structural_decoupling = use_structural_decoupling
        self.use_kc_imputation = use_kc_imputation
        self.use_solution = use_solution
        self.use_visual = use_visual

        # Save architecture configuration dictionary for checkpoint reconstruction
        self.config = {
            "model_name": "mg_mmkt_v2",
            "num_questions": num_questions,
            "num_kcs": num_kcs,
            "d_raw_txt": d_raw_txt,
            "d_raw_img": d_raw_img,
            "d_model": d_model,
            "d_ffn": d_ffn,
            "num_heads": num_heads,
            "max_seq_len": max_seq_len,
            "p_id_drop": p_id_drop,
            "dropout": dropout,
            "use_lupi": self.use_lupi,
            "use_dual_pathway_history": self.use_dual_pathway_history,
            "use_glu": self.use_glu,
            "use_post_fusion_ffn": self.use_post_fusion_ffn,
            "use_disentangled_qk": self.use_disentangled_qk,
            "use_relative_forgetting": self.use_relative_forgetting,
            "use_structural_decoupling": self.use_structural_decoupling,
            "use_kc_imputation": self.use_kc_imputation,
            "use_solution": self.use_solution,
            "use_visual": self.use_visual,
        }

        # 1. Structural Embedding Tables (ID, KC, Response)
        self.q_emb = nn.Embedding(num_questions, d_model)
        self.kc_emb = nn.Embedding(num_kcs, d_model)
        self.r_emb = nn.Embedding(2, d_model)

        # 2. Layer 1: Structural Decoupling & KC-Conditioned Imputation
        self.structural_decoupling = StructuralDecouplingLayer(
            d_model=d_model,
            p_id_drop=p_id_drop,
            use_structural_decoupling=self.use_structural_decoupling,
        )
        self.imputation = KCConditionedImputationLayer(
            d_raw_txt=d_raw_txt,
            d_raw_img=d_raw_img,
            d_model=d_model,
            use_kc_imputation=self.use_kc_imputation,
            use_visual=self.use_visual,
            use_solution=self.use_solution,
        )

        # 3. Layer 2: Hierarchical Dual-Stage Gated Fusion
        self.q_gating = QuestionDualStageGating(
            d_model=d_model,
            d_ffn=d_ffn,
            dropout=dropout,
            use_glu=self.use_glu,
            use_post_fusion_ffn=self.use_post_fusion_ffn,
        )
        self.s_gating = SolutionDualStageGating(
            d_model=d_model,
            d_ffn=d_ffn,
            dropout=dropout,
            use_glu=self.use_glu,
            use_post_fusion_ffn=self.use_post_fusion_ffn,
        )

        # 4. Layer 3: Cognitive History Fusion & LUPI Estimator
        self.history_fusion = CognitiveHistoryFusionLayer(
            d_model=d_model,
            use_dual_pathway_history=self.use_dual_pathway_history,
            use_lupi=self.use_lupi,
        )

        # 5. Layer 4: Disentangled Causal Attention & Readout
        self.causal_attention = DisentangledCausalAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            use_disentangled_qk=self.use_disentangled_qk,
            use_relative_forgetting=self.use_relative_forgetting,
        )
        self.readout = MultiInteractiveReadout(
            d_model=d_model,
            dropout=dropout,
            use_lupi=self.use_lupi,
        )

        # Register Item Bank Buffers if provided
        if static_item_bank is not None:
            self.register_item_bank_buffers(static_item_bank)

    def register_item_bank_buffers(self, static_item_bank: Dict[str, torch.Tensor]) -> None:
        """
        Register static item bank properties as non-persistent buffers.
        Converts float16 tensors to float32 for stable on-device operations.
        """
        for key, tensor in static_item_bank.items():
            if key in ["mg_v_txt_q", "mg_v_img_q", "mg_v_txt_s", "mg_v_img_s"]:
                tensor_fp32 = tensor.float() if tensor.dtype != torch.float32 else tensor
                self.register_buffer(key, tensor_fp32, persistent=False)
            elif key in ["question_leaf_kcs", "question_kc_mask", "modality_masks"]:
                self.register_buffer(key, tensor, persistent=False)

    def forward(
        self,
        q_seq: torch.Tensor,
        r_seq: torch.Tensor,
        seq_mask: torch.Tensor,
        modality_masks: Optional[torch.Tensor] = None,
        raw_modality_masks: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of MG-MMKT v2.

        Args:
            q_seq: Question index sequences [B, L].
            r_seq: Student response sequences [B, L] in {0, 1}.
            seq_mask: Sequence validity mask [B, L].
            modality_masks: Dynamic modality flags [B, L, 3] (with dropout during train).
            raw_modality_masks: Clean ground-truth modality flags [B, L, 3] without dropout.

        Returns:
            Dictionary containing prediction logits, probabilities, intermediate representations,
            clean teacher targets for distillation, and explainability weights.
        """
        device = q_seq.device

        # Lookup static item features from registered buffers
        v_txt_q = self.mg_v_txt_q[q_seq]  # [B, L, d_raw_txt]
        v_img_q = self.mg_v_img_q[q_seq]  # [B, L, d_raw_img]
        v_txt_s = self.mg_v_txt_s[q_seq]  # [B, L, d_raw_txt]
        v_img_s = self.mg_v_img_s[q_seq]  # [B, L, d_raw_img]
        leaf_kcs = self.question_leaf_kcs[q_seq]   # [B, L, 6]
        leaf_kc_mask = self.question_kc_mask[q_seq]  # [B, L, 6]

        if modality_masks is None:
            modality_masks = self.modality_masks[q_seq].float()
        if raw_modality_masks is None:
            raw_modality_masks = modality_masks.clone()

        # 1. Structural Embeddings
        e_q = self.q_emb(q_seq)             # [B, L, d]
        e_kcs = self.kc_emb(leaf_kcs)       # [B, L, 6, d]
        e_r = self.r_emb(r_seq)             # [B, L, d]

        # 2. Layer 1: Structural Decoupling & KC-Conditioned Imputation
        e_struct, e_C = self.structural_decoupling(e_q, e_kcs, leaf_kc_mask)
        tilde_v_txt_q, tilde_v_img_q, tilde_v_txt_s, tilde_v_img_s = self.imputation(
            v_txt_q=v_txt_q,
            v_img_q=v_img_q,
            v_txt_s=v_txt_s,
            v_img_s=v_img_s,
            modality_masks=modality_masks,
            e_C=e_C,
        )

        # 3. Layer 2: Hierarchical Dual-Stage Gated Fusion
        m_img_q = modality_masks[..., 0:1]
        m_txt_s = modality_masks[..., 1:2]
        m_img_s = modality_masks[..., 2:3]

        x, alpha_q = self.q_gating(e_struct, tilde_v_txt_q, tilde_v_img_q, m_img_q)
        z_s, beta_s = self.s_gating(tilde_v_txt_s, tilde_v_img_s, e_C, m_txt_s, m_img_s)

        # 4. Layer 3: Cognitive History Fusion & LUPI Estimator
        y = self.history_fusion.forward_history(x, z_s, e_r, r_seq, m_txt_s, m_img_s)
        hat_z_s = self.history_fusion.forward_cognitive_demand(x)

        # Teacher Representation z_s_clean for Privileged Distillation Loss (LUPI)
        # Computed using clean raw modality masks (without dropout), with Dropout temporarily disabled
        # on solution gating to prevent noise corruption on the teacher target, and detached from autograd
        if self.use_lupi:
            was_training = self.s_gating.training
            self.s_gating.eval()
            with torch.no_grad():
                _, _, t_txt_clean, t_img_clean = self.imputation(
                    v_txt_q=v_txt_q,
                    v_img_q=v_img_q,
                    v_txt_s=v_txt_s,
                    v_img_s=v_img_s,
                    modality_masks=raw_modality_masks,
                    e_C=e_C,
                )
                z_s_clean, _ = self.s_gating(
                    t_txt_clean,
                    t_img_clean,
                    e_C,
                    raw_modality_masks[..., 1:2],
                    raw_modality_masks[..., 2:3],
                )
                z_s_clean = z_s_clean.detach()
            if was_training:
                self.s_gating.train()
        else:
            z_s_clean = z_s.detach()

        # 5. Layer 4: Disentangled Causal Attention & Readout
        h, gamma_bar = self.causal_attention(x, y, seq_mask)
        logits, probs = self.readout(h, x, hat_z_s)

        return {
            "logits": logits,
            "probs": probs,
            "x": x,
            "z_s": z_s,
            "z_s_clean": z_s_clean,
            "hat_z_s": hat_z_s,
            "h": h,
            "y": y,
            "e_struct": e_struct,
            "e_C": e_C,
            "alpha_q": alpha_q,
            "beta_s": beta_s,
            "gamma_bar": gamma_bar,
            "modality_masks": modality_masks,
            "raw_modality_masks": raw_modality_masks,
        }

    def get_gating_and_decay_rates(self) -> Dict[str, Any]:
        """
        Extract current values of soft penalty parameters delta and forgetting decay rates theta_h.
        """
        delta_q = float(torch.clamp(F.softplus(self.q_gating.raw_delta_q), min=1e-4, max=10.0).detach().cpu().item())
        delta_s_txt = float(torch.clamp(F.softplus(self.s_gating.raw_delta_s_txt), min=1e-4, max=10.0).detach().cpu().item())
        delta_s_img = float(torch.clamp(F.softplus(self.s_gating.raw_delta_s_img), min=1e-4, max=10.0).detach().cpu().item())

        thetas = []
        if self.use_relative_forgetting and hasattr(self.causal_attention, "raw_theta"):
            thetas = [float(th) for th in F.softplus(self.causal_attention.raw_theta).detach().cpu().numpy()]

        return {
            "delta_q": delta_q,
            "delta_s_txt": delta_s_txt,
            "delta_s_img": delta_s_img,
            "thetas": thetas,
        }
