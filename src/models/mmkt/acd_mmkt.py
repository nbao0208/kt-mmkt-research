"""
Asymmetric Causal-Disentangled Multimodal Knowledge Tracing (ACD-MMKT) Model.

Implements the complete unified end-to-end architecture specified in:
- docs/models/ACD-MMKT.md
- plans/acd_mmkt_implementation_and_training_plan.md

Integrates:
- Subsystem 1: Fine-Grained Cross-Modal Alignment & Context-Aware Hierarchical Gating
  (cross_modal_alignment.py) with Unique-Question Batch Indexing.
- Subsystem 2: Hypergraph Concept Convolution, Latent Solution Complexity Distillation,
  and Rasch Psychometric Disentanglement (hypergraph_rasch.py).
- Subsystem 3: Asymmetric Dual-Stream Encoder and Decay-Aware Monotonic Attention Retriever
  (monotonic_attention.py).
- Subsystem 4: Neural Cognitive Diagnosis Readout (DeepIRT) with Ante-hoc Monotonicity
  (deep_irt_readout.py).
- Tier 1 Static Item Bank Registration with FP16 buffering.
- Dual-mode forward: Stage 1 (Item-Bank Pre-training) and Stage 2 (Sequential KT Fine-tuning).
- Three-tier Ante-hoc Explainability Extraction.
"""

from typing import Dict, Any, Optional, Tuple, Union, List
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules.cross_modal_alignment import FineGrainedCrossModalAlignment
from src.models.modules.hypergraph_rasch import HypergraphRaschDisentanglement
from src.models.modules.monotonic_attention import AsymmetricMonotonicAttention
from src.models.modules.deep_irt_readout import DeepIRTReadout


class ACDMMKT(nn.Module):
    """
    End-to-End ACD-MMKT Model.
    """

    def __init__(
        self,
        num_questions: int = 7652,
        num_kcs: int = 1175,
        d_raw_txt: int = 768,
        d_raw_img: int = 768,
        d_ctx: int = 21,
        d_model: int = 128,
        num_heads: int = 4,
        max_seq_len: int = 200,
        dropout: float = 0.20,
        hgnn_layers: int = 2,
        tau_temp: float = 0.10,
        beta_distill: float = 0.10,
        beta_ctt: float = 0.20,
        beta_disen: float = 0.05,
        lambda_res: float = 1e-3,
        irt_scale: float = 4.0,
        use_solution: bool = True,
        use_visual: bool = True,
        use_distillation: bool = True,
        use_fine_grained: bool = True,
        use_hypergraph_rasch: bool = True,
        use_deep_irt: bool = True,
        static_item_bank: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Initialize ACD-MMKT model.

        Args:
            num_questions: Total number of unique items |Q| (default: 7652).
            num_kcs: Total number of unique skills |K| (default: 1175).
            d_raw_txt: Raw text token embedding dimension (default: 768).
            d_raw_img: Raw vision patch embedding dimension (default: 768).
            d_ctx: Dynamic context dimension (default: 21).
            d_model: Hidden dimension d (default: 128).
            num_heads: Number of attention heads (default: 4).
            max_seq_len: Maximum sequence length (default: 200).
            dropout: Dropout probability (default: 0.20).
            hgnn_layers: Number of HGNN convolution layers (default: 2).
            tau_temp: Temperature for InfoNCE contrastive distillation (default: 0.10).
            beta_distill: Multi-task weight for distillation loss (default: 0.10).
            beta_ctt: Multi-task weight for CTT difficulty loss (default: 0.20).
            beta_disen: Multi-task weight for Rasch orthogonality loss (default: 0.05).
            lambda_res: Regularization coefficient on DeepIRT residual branch (default: 1e-3).
            irt_scale: Scaling parameter kappa in DeepIRT logit (default: 4.0).
            use_solution: Whether to incorporate solution modalities.
            use_visual: Whether to incorporate image modalities.
            use_distillation: Whether to train Student latent solution complexity predictor (Ablation 2).
            use_fine_grained: Whether to use token/patch cross-modal attention (Ablation 3).
            use_hypergraph_rasch: Whether to use HGNN and Rasch decomposition (Ablation 4).
            use_deep_irt: Whether to use DeepIRT readout (Ablation 5).
            static_item_bank: Optional preloaded static item bank to register as model buffers.
        """
        super().__init__()
        self.num_questions = num_questions
        self.num_kcs = num_kcs
        self.d_raw_txt = d_raw_txt
        self.d_raw_img = d_raw_img
        self.d_ctx = d_ctx
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout
        self.hgnn_layers = hgnn_layers
        self.tau_temp = tau_temp
        self.beta_distill = beta_distill
        self.beta_ctt = beta_ctt
        self.beta_disen = beta_disen
        self.lambda_res = lambda_res
        self.irt_scale = irt_scale

        # Ablation Flags
        self.use_solution = use_solution
        self.use_visual = use_visual
        self.use_distillation = use_distillation
        self.use_fine_grained = use_fine_grained
        self.use_hypergraph_rasch = use_hypergraph_rasch
        self.use_deep_irt = use_deep_irt

        # Save architecture configuration dictionary for checkpoint persistence
        self.config = {
            "model_name": "acd_mmkt",
            "num_questions": num_questions,
            "num_kcs": num_kcs,
            "d_raw_txt": d_raw_txt,
            "d_raw_img": d_raw_img,
            "d_ctx": d_ctx,
            "d_model": d_model,
            "num_heads": num_heads,
            "max_seq_len": max_seq_len,
            "dropout": dropout,
            "hgnn_layers": hgnn_layers,
            "tau_temp": tau_temp,
            "beta_distill": beta_distill,
            "beta_ctt": beta_ctt,
            "beta_disen": beta_disen,
            "lambda_res": lambda_res,
            "irt_scale": irt_scale,
            "use_solution": use_solution,
            "use_visual": use_visual,
            "use_distillation": use_distillation,
            "use_fine_grained": use_fine_grained,
            "use_hypergraph_rasch": use_hypergraph_rasch,
            "use_deep_irt": use_deep_irt,
        }

        # Subsystem 1: Fine-Grained Cross-Modal Alignment & Context-Aware Gating
        self.subsystem1 = FineGrainedCrossModalAlignment(
            d_raw_txt=d_raw_txt,
            d_raw_img=d_raw_img,
            d_ctx=d_ctx,
            d_model=d_model,
            num_heads=num_heads,
            patch_count=49,
            dropout=dropout,
            use_visual=use_visual,
            use_solution=use_solution,
            use_fine_grained=use_fine_grained,
        )

        # Subsystem 2: Hypergraph Concept Convolution, Distillation, and Rasch Disentanglement
        self.subsystem2 = HypergraphRaschDisentanglement(
            num_kcs=num_kcs,
            d_raw_kc=d_raw_txt,
            d_model=d_model,
            hgnn_layers=hgnn_layers,
            dropout=dropout,
            use_distillation=use_distillation,
            use_hypergraph_rasch=use_hypergraph_rasch,
        )

        # Subsystem 3: Asymmetric Dual-Stream Encoder & Decay-Aware Monotonic Attention
        self.subsystem3 = AsymmetricMonotonicAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            use_solution=use_solution,
        )

        # Subsystem 4: Neural Cognitive Diagnosis Readout (DeepIRT)
        self.subsystem4 = DeepIRTReadout(
            d_model=d_model,
            irt_scale=irt_scale,
            dropout=dropout,
            use_deep_irt=use_deep_irt,
        )

        # Register static item bank if provided
        if static_item_bank is not None:
            self.init_static_buffers(static_item_bank)

    def init_static_buffers(self, static_item_bank: Dict[str, torch.Tensor]) -> None:
        """
        Register Tier 1 static item bank tensors as persistent model buffers.
        Preserves FP16 precision for 4.3 GB fine-grained token/patch tensors to
        minimize GPU/MPS RAM consumption.
        """
        # Fine-grained token/patch tensors stored in float16
        self.register_buffer("acd_E_txt_q", static_item_bank["acd_E_txt_q"])          # [7652, 128, 768] fp16
        self.register_buffer("acd_mask_txt_q", static_item_bank["acd_mask_txt_q"])    # [7652, 128] bool
        self.register_buffer("acd_E_img_q", static_item_bank["acd_E_img_q"])          # [7652, 49, 768] fp16
        self.register_buffer("acd_E_txt_s", static_item_bank["acd_E_txt_s"])          # [7652, 160, 768] fp16
        self.register_buffer("acd_mask_txt_s", static_item_bank["acd_mask_txt_s"])    # [7652, 160] bool
        self.register_buffer("acd_E_img_s", static_item_bank["acd_E_img_s"])          # [7652, 49, 768] fp16

        # Hypergraph structures in float32
        self.register_buffer("acd_kc_init_emb", static_item_bank["acd_kc_init_emb"].float())      # [1175, 768]
        self.register_buffer("acd_hypergraph_G", static_item_bank["acd_hypergraph_G"].float())    # [1175, 1175]

        # Metadata and discrete references
        self.register_buffer("modality_masks", static_item_bank["modality_masks"].float())        # [7652, 3]
        self.register_buffer("question_leaf_kcs", static_item_bank["question_leaf_kcs"])          # [7652, 6]
        self.register_buffer("question_kc_mask", static_item_bank["question_kc_mask"])            # [7652, 6]
        if "question_static_ctx" in static_item_bank:
            self.register_buffer("question_static_ctx", static_item_bank["question_static_ctx"].float())  # [7652, 15]


        # Global pooled vectors for Ablation 3 (w/o Fine-Grained Alignment)
        if "mg_v_txt_q" in static_item_bank:
            self.register_buffer("mg_v_txt_q", static_item_bank["mg_v_txt_q"])
            self.register_buffer("mg_v_txt_s", static_item_bank["mg_v_txt_s"])
            self.register_buffer("mg_v_img_q", static_item_bank["mg_v_img_q"])
            self.register_buffer("mg_v_img_s", static_item_bank["mg_v_img_s"])


    def forward_item_bank(
        self,
        item_indices: Optional[torch.Tensor] = None,
        dynamic_ctx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Stage 1 Forward Pass: Multimodal Item-Bank Pre-training (no student sequences).
        Optimizes fine-grained cross-modal alignment, 2-layer HGNN, student solution
        complexity distillation, and Rasch difficulty calibration directly on items.

        Args:
            item_indices: 1D tensor of question indices [N] (if None, all 7652 items).
            dynamic_ctx: Optional context [N, 21]. If None, built from static item context.

        Returns:
            Dictionary containing item-level representations and predictions for Stage 1 losses.
        """
        assert hasattr(self, "acd_E_txt_q"), "Static buffers not initialized. Call init_static_buffers first."

        if item_indices is None:
            N = self.num_questions
            q_idx = torch.arange(N, device=self.acd_E_txt_q.device)
        else:
            q_idx = item_indices
            N = q_idx.size(0)

        # Slice fine-grained tensors for the requested items
        E_txt_q = self.acd_E_txt_q[q_idx]
        mask_txt_q = self.acd_mask_txt_q[q_idx]
        E_img_q = self.acd_E_img_q[q_idx]
        E_txt_s = self.acd_E_txt_s[q_idx]
        mask_txt_s = self.acd_mask_txt_s[q_idx]
        E_img_s = self.acd_E_img_s[q_idx]
        mod_masks = self.modality_masks[q_idx]
        leaf_kcs = self.question_leaf_kcs[q_idx]
        leaf_kc_mask = self.question_kc_mask[q_idx]

        if dynamic_ctx is None:
            # Build default 21-dim context: [15 static context, 0, 0, 0, 0, m_txt_s, m_img_s]
            if hasattr(self, "question_static_ctx"):
                static_ctx = self.question_static_ctx[q_idx]
            else:
                static_ctx = torch.zeros((N, 15), device=q_idx.device)
            zeros_4 = torch.zeros((N, 4), device=q_idx.device)
            sol_flags = mod_masks[:, 1:3]
            dynamic_ctx = torch.cat([static_ctx, zeros_4, sol_flags], dim=-1)  # [N, 21]

        # 1. Subsystem 1: Item-level fine-grained alignment and gating
        mg_kwargs = {}
        if not self.use_fine_grained and hasattr(self, "mg_v_txt_q"):
            mg_kwargs = {
                "mg_v_txt_q": self.mg_v_txt_q[q_idx],
                "mg_v_img_q": self.mg_v_img_q[q_idx],
                "mg_v_txt_s": self.mg_v_txt_s[q_idx],
                "mg_v_img_s": self.mg_v_img_s[q_idx],
            }
        sub1_out = self.subsystem1(
            E_txt_q=E_txt_q,
            mask_txt_q=mask_txt_q,
            E_img_q=E_img_q,
            E_txt_s=E_txt_s,
            mask_txt_s=mask_txt_s,
            E_img_s=E_img_s,
            modality_masks=mod_masks,
            dynamic_ctx=dynamic_ctx,
            q_indices=None,
            **mg_kwargs,
        )
        v_stem = sub1_out["v_stem"]  # [N, d]
        v_sol = sub1_out["v_sol"]    # [N, d]


        # 2. Subsystem 2: Hypergraph propagation and Rasch estimation
        sub2_out = self.subsystem2(
            v_stem=v_stem,
            leaf_kcs=leaf_kcs,
            leaf_kc_mask=leaf_kc_mask,
            kc_init_emb=self.acd_kc_init_emb,
            hypergraph_G=self.acd_hypergraph_G,
        )

        return {
            "v_stem": v_stem,
            "v_sol": v_sol,
            "hat_v_sol": sub2_out["hat_v_sol"],
            "mu_q": sub2_out["mu_q"],
            "logit_mu": sub2_out["logit_mu"],
            "d_q": sub2_out["d_q"],
            "c_C": sub2_out["c_C"],
            "m_txt_s": mod_masks[:, 1:2],
            "item_indices": q_idx,
        }

    def forward(
        self,
        batch_or_q_seq: Union[Dict[str, torch.Tensor], torch.Tensor],
        r_seq: Optional[torch.Tensor] = None,
        dynamic_ctx: Optional[torch.Tensor] = None,
        modality_masks: Optional[torch.Tensor] = None,
        leaf_kcs: Optional[torch.Tensor] = None,
        leaf_kc_mask: Optional[torch.Tensor] = None,
        delta_t_lag: Optional[torch.Tensor] = None,
        delta_t_kc: Optional[torch.Tensor] = None,
        batch_flag: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Stage 2 Forward Pass: Sequential Interaction Modeling.

        Accepts either a collated batch dictionary or individual tensors.

        Returns:
            Dictionary containing:
            - 'logits': [B, L] predicted logits.
            - 'preds': [B, L] predicted probabilities.
            - 'theta': [B, L] student proficiency theta_{u, t}.
            - 'a_q': [B, L] item discrimination a_{q, t}.
            - 'mu_q': [B, L] item difficulty mu_{q, t}.
            - 'psy_gap': [B, L] psychometric gap (theta - mu).
            - 'ell_irt': [B, L] IRT logit component.
            - 'ell_res': [B, L] deep residual logit component.
            - 'alpha_ret': [B, L, L] monotonic retrieval attention weights.
            - 'lambda_q': [B, L, d] question gating weights.
            - 'hat_v_sol': [B, L, d] predicted latent solution complexity.
            - 'v_sol': [B, L, d] ground-truth solution representation.
            - 'uniq_hat_v_sol', 'uniq_v_sol', 'uniq_m_txt_s': Unique item tensors for distillation loss.
            - 'c_C': [B, L, d] core concept representations.
            - 'd_q': [B, L, d] item variation representations.
            - 'logit_mu': [B, L] difficulty logits.
        """
        assert hasattr(self, "acd_E_txt_q"), "Static buffers not initialized. Call init_static_buffers first."

        if isinstance(batch_or_q_seq, dict):
            batch = batch_or_q_seq
            q_seq = batch["q_seq"]
            r_seq = batch["r_seq"]
            dynamic_ctx = batch["dynamic_ctx"]
            modality_masks = batch["modality_masks"]
            leaf_kcs = batch["leaf_kcs"]
            leaf_kc_mask = batch["leaf_kc_mask"]
            delta_t_lag = batch.get("delta_t_lag", torch.zeros_like(q_seq, dtype=torch.float32))
            delta_t_kc = batch.get("delta_t_kc", torch.zeros_like(q_seq, dtype=torch.float32))
            batch_flag = batch.get("batch_flag", torch.zeros_like(q_seq, dtype=torch.float32))
            seq_mask = batch.get("seq_mask", torch.ones_like(q_seq, dtype=torch.bool))
        else:
            q_seq = batch_or_q_seq
            assert r_seq is not None and dynamic_ctx is not None, "Missing required arguments"
            if modality_masks is None:
                modality_masks = self.modality_masks[q_seq]
            if leaf_kcs is None:
                leaf_kcs = self.question_leaf_kcs[q_seq]
            if leaf_kc_mask is None:
                leaf_kc_mask = self.question_kc_mask[q_seq]
            if delta_t_lag is None:
                delta_t_lag = torch.zeros_like(q_seq, dtype=torch.float32)
            if delta_t_kc is None:
                delta_t_kc = torch.zeros_like(q_seq, dtype=torch.float32)
            if batch_flag is None:
                batch_flag = torch.zeros_like(q_seq, dtype=torch.float32)
            if seq_mask is None:
                seq_mask = torch.ones_like(q_seq, dtype=torch.bool)

        B, L = q_seq.shape

        # Step 1: Subsystem 1 (Unique-Question Batch Indexing)
        mg_kwargs = {}
        if not self.use_fine_grained and hasattr(self, "mg_v_txt_q"):
            mg_kwargs = {
                "mg_v_txt_q": self.mg_v_txt_q,
                "mg_v_img_q": self.mg_v_img_q,
                "mg_v_txt_s": self.mg_v_txt_s,
                "mg_v_img_s": self.mg_v_img_s,
            }
        sub1_out = self.subsystem1(
            E_txt_q=self.acd_E_txt_q,
            mask_txt_q=self.acd_mask_txt_q,
            E_img_q=self.acd_E_img_q,
            E_txt_s=self.acd_E_txt_s,
            mask_txt_s=self.acd_mask_txt_s,
            E_img_s=self.acd_E_img_s,
            modality_masks=modality_masks,
            dynamic_ctx=dynamic_ctx,
            q_indices=q_seq,
            static_modality_masks=self.modality_masks,
            **mg_kwargs,
        )
        v_stem = sub1_out["v_stem"]    # [B, L, d]
        v_sol = sub1_out["v_sol"]      # [B, L, d]
        lambda_q = sub1_out["lambda_q"]# [B, L, d]

        # Step 2: Subsystem 2 (Hypergraph Concept Propagation & Rasch Decomposition)
        sub2_out = self.subsystem2(
            v_stem=v_stem,
            leaf_kcs=leaf_kcs,
            leaf_kc_mask=leaf_kc_mask,
            kc_init_emb=self.acd_kc_init_emb,
            hypergraph_G=self.acd_hypergraph_G,
        )
        x = sub2_out["x"]                    # [B, L, d]
        c_C = sub2_out["c_C"]                # [B, L, d]
        hat_v_sol = sub2_out["hat_v_sol"]    # [B, L, d]
        mu_q = sub2_out["mu_q"]              # [B, L, 1]
        logit_mu = sub2_out["logit_mu"]      # [B, L, 1]
        d_q = sub2_out["d_q"]                # [B, L, d]

        # Step 3: Subsystem 3 (Asymmetric Dual-Stream & Decay-Aware Monotonic Retriever)
        m_txt_s = modality_masks[..., 1:2]   # [B, L, 1]
        sub3_out = self.subsystem3(
            x=x,
            c_C=c_C,
            mu_q=mu_q,
            d_q=d_q,
            r_seq=r_seq,
            delta_t_lag=delta_t_lag,
            delta_t_kc=delta_t_kc,
            batch_flag=batch_flag,
            v_sol=v_sol,
            m_txt_s=m_txt_s,
            seq_mask=seq_mask,
        )
        h = sub3_out["h"]                    # [B, L, d]
        tilde_x = sub3_out["tilde_x"]        # [B, L, d]
        alpha_ret = sub3_out["alpha_ret"]    # [B, L, L]

        # Step 4: Subsystem 4 (Neural Cognitive Diagnosis / DeepIRT Readout)
        sub4_out = self.subsystem4(
            h=h,
            tilde_x=tilde_x,
            mu_q=mu_q,
        )

        # Unique item tensors for efficient distillation loss computation
        uniq_q = sub1_out["uniq_q"]
        uniq_m_txt_s = self.modality_masks[uniq_q, 1:2]  # [N_uniq, 1]
        # In batch indexing, compute hat_v_sol for unique items directly
        uniq_leaf_kcs = self.question_leaf_kcs[uniq_q]
        uniq_leaf_kc_mask = self.question_kc_mask[uniq_q]
        uniq_c_C = self.subsystem2.pool_concepts(sub2_out["E_K"], uniq_leaf_kcs, uniq_leaf_kc_mask)

        # Build consistent item-level context for unique items: [15 static, 4 zeros (neutral dynamic), 2 solution flags]
        # Matching Stage 1 forward_item_bank distribution instead of zeroing all 21 dimensions (Issue H2)
        if hasattr(self, "question_static_ctx"):
            uniq_static_ctx = self.question_static_ctx[uniq_q]
        else:
            uniq_static_ctx = torch.zeros((uniq_q.size(0), 15), device=q_seq.device)
        zeros_4 = torch.zeros((uniq_q.size(0), 4), device=q_seq.device)
        uniq_sol_flags = self.modality_masks[uniq_q, 1:3]
        uniq_ctx = torch.cat([uniq_static_ctx, zeros_4, uniq_sol_flags], dim=-1)  # [N_uniq, 21]

        u_m_img_q = self.modality_masks[uniq_q, 0:1]
        u_m_img_s = self.modality_masks[uniq_q, 2:3]
        uniq_v_stem, uniq_v_sol, _, _ = self.subsystem1.fuse_with_context(
            u_txt_q=sub1_out["u_txt_q"],
            u_cross_q=sub1_out["u_cross_q"],
            u_txt_s=sub1_out["u_txt_s"],
            u_cross_s=sub1_out["u_cross_s"],
            m_img_q=u_m_img_q,
            m_txt_s=uniq_m_txt_s,
            m_img_s=u_m_img_s,
            raw_ctx=uniq_ctx,
        )
        uniq_hat_v_sol = self.subsystem2.predict_solution_complexity(uniq_v_stem, uniq_c_C)


        return {
            "logits": sub4_out["logits"],          # [B, L]
            "preds": sub4_out["preds"],            # [B, L]
            "theta": sub4_out["theta"],            # [B, L]
            "a_q": sub4_out["a_q"],                # [B, L]
            "mu_q": sub4_out["mu_q"],              # [B, L]
            "psy_gap": sub4_out["psy_gap"],        # [B, L]
            "ell_irt": sub4_out["ell_irt"],        # [B, L]
            "ell_res": sub4_out["ell_res"],        # [B, L]
            "alpha_ret": alpha_ret,                # [B, L, L]
            "lambda_q": lambda_q,                  # [B, L, d]
            "A_cross_q": sub1_out["A_cross_q"],    # [N_uniq, L_q, P]
            "v_stem": v_stem,                      # [B, L, d]
            "v_sol": v_sol,                        # [B, L, d]
            "hat_v_sol": hat_v_sol,                # [B, L, d]
            "c_C": c_C,                            # [B, L, d]
            "d_q": d_q,                            # [B, L, d]
            "logit_mu": logit_mu.squeeze(-1),      # [B, L]
            "uniq_q": uniq_q,
            "uniq_hat_v_sol": uniq_hat_v_sol,      # [N_uniq, d]
            "uniq_v_sol": uniq_v_sol,              # [N_uniq, d]
            "uniq_m_txt_s": uniq_m_txt_s,          # [N_uniq, 1]
            "theta_decay": sub3_out["theta_decay"],
        }

    @torch.no_grad()
    def extract_explanations(
        self,
        batch: Dict[str, torch.Tensor],
        top_k_history: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Extract Three-Tier Ante-hoc Explainability Reports for a sequence batch.

        Args:
            batch: Collated batch dictionary.
            top_k_history: Number of top-attending historical steps to report (default: 5).

        Returns:
            List of explanation report dictionaries, one per sequence in the batch.
        """
        self.eval()
        outputs = self.forward(batch)

        B, L = batch["q_seq"].shape
        reports = []

        q_seq = batch["q_seq"].cpu()
        r_seq = batch["r_seq"].cpu()
        eval_mask = batch.get("eval_mask", batch.get("loss_mask")).cpu()
        context_starts = batch.get("context_start", torch.zeros(B, dtype=torch.int64)).cpu()

        preds = outputs["preds"].cpu()
        thetas = outputs["theta"].cpu()
        mu_qs = outputs["mu_q"].cpu()
        a_qs = outputs["a_q"].cpu()
        psy_gaps = outputs["psy_gap"].cpu()
        alpha_rets = outputs["alpha_ret"].cpu()
        lambda_qs = outputs["lambda_q"].cpu().mean(dim=-1)  # scalar gate intensity [B, L]
        modality_masks = batch["modality_masks"].cpu()

        # Compute stem-only difficulty for delta_mu_sol
        c_C = outputs["c_C"]
        v_stem = outputs["v_stem"]
        mu_stem_only = self.subsystem2.estimate_difficulty_stem_only(v_stem, c_C).squeeze(-1).cpu()
        delta_mu_sols = (mu_qs - mu_stem_only)

        # Top keyword-to-patch cross-modal alignments from A_cross_q (Tier 2)
        uniq_q_ref = outputs.get("uniq_q")
        A_cross_ref = outputs.get("A_cross_q")
        if uniq_q_ref is not None:
            uniq_q_ref = uniq_q_ref.cpu()
        if A_cross_ref is not None:
            A_cross_ref = A_cross_ref.cpu()

        for b in range(B):
            seq_reports = []
            c_start = int(context_starts[b].item())

            for t in range(L):
                if not eval_mask[b, t]:
                    continue

                q_id = int(q_seq[b, t].item())
                r_gt = int(r_seq[b, t].item())
                pred_prob = float(preds[b, t].item())
                theta_val = float(thetas[b, t].item())
                mu_val = float(mu_qs[b, t].item())
                a_val = float(a_qs[b, t].item())
                gap_val = float(psy_gaps[b, t].item())
                gate_val = float(lambda_qs[b, t].item())
                delta_sol_val = float(delta_mu_sols[b, t].item())

                # Top-K historical retrieval steps
                # alpha_rets[b, t] has length L: index 0 is BOS, index 1..t maps to past steps 0..t-1
                attn_row = alpha_rets[b, t, 1: t + 1]  # attention on past steps 0..t-1
                top_k = min(top_k_history, len(attn_row))
                top_history = []
                if top_k > 0:
                    top_vals, top_indices = torch.topk(attn_row, k=top_k)
                    for val, idx in zip(top_vals.tolist(), top_indices.tolist()):
                        hist_step = idx
                        top_history.append({
                            "window_step": hist_step,
                            "absolute_interaction_index": c_start + hist_step,
                            "question_id": int(q_seq[b, hist_step].item()),
                            "response": int(r_seq[b, hist_step].item()),
                            "attention_weight": round(val, 4),
                            "learned_solution": bool(modality_masks[b, hist_step, 1] > 0.5),
                        })

                # Top keyword-to-patch cross-modal alignments (Tier 2, Plan Step 5.1)
                top_alignments = []
                has_img = bool(modality_masks[b, t, 0] > 0.5)
                if has_img and uniq_q_ref is not None and A_cross_ref is not None:
                    matches = (uniq_q_ref == q_id).nonzero(as_tuple=True)[0]
                    if len(matches) > 0:
                        a_mat = A_cross_ref[matches[0]].float()  # [L_q, P]
                        # Filter out [PAD] tokens using item bank question text mask
                        # to ensure Top-K alignments map strictly to valid question words
                        if hasattr(self, "acd_mask_txt_q") and q_id < len(self.acd_mask_txt_q):
                            txt_mask = self.acd_mask_txt_q[q_id].float().unsqueeze(-1).cpu()  # [L_q, 1]
                            a_mat = a_mat * txt_mask

                        flat_a = a_mat.view(-1)
                        # Determine number of valid pairs to retrieve (at most 5)
                        valid_pairs = int((flat_a > 0.0).sum().item())
                        k_pairs = min(5, valid_pairs) if valid_pairs > 0 else min(5, flat_a.size(0))
                        if k_pairs > 0:
                            top_weights, top_flat_indices = torch.topk(flat_a, k=k_pairs)
                            for w, flat_idx in zip(top_weights.tolist(), top_flat_indices.tolist()):
                                t_idx = flat_idx // a_mat.size(1)
                                p_idx = flat_idx % a_mat.size(1)
                                top_alignments.append({
                                    "token_index": int(t_idx),
                                    "patch_index": int(p_idx),
                                    "patch_coord": [int(p_idx // 7), int(p_idx % 7)],
                                    "attention_weight": round(float(w), 4),
                                })

                step_report = {
                    "step": t,
                    "absolute_step": c_start + t,
                    "question_id": q_id,
                    "ground_truth": r_gt,
                    "prediction_probability": round(pred_prob, 4),
                    "tier1_psychometrics": {
                        "proficiency_theta": round(theta_val, 4),
                        "difficulty_mu": round(mu_val, 4),
                        "discrimination_a": round(a_val, 4),
                        "gap_theta_minus_mu": round(gap_val, 4),
                    },
                    "tier2_attribution": {
                        "has_visual_modality": has_img,
                        "visual_gate_intensity_lambda": round(gate_val, 4),
                        "solution_complexity_shift_delta_mu": round(delta_sol_val, 4),
                        "top_influential_history": top_history,
                        "top_token_patch_alignments": top_alignments,
                    },
                }
                seq_reports.append(step_report)


            reports.append(seq_reports)

        return reports
