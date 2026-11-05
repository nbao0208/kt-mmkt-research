"""
End-to-End ACD-MMKTv3 Model Architecture.
Asymmetric Causal-Disentangled Multimodal Knowledge Tracing v3.

Integrates:
- Subsystem 1: Text-Guided Cross-Modal Alignment & Lightweight Cognitive Gating (v_stem, v_sol, lambda_q).
- Subsystem 2: Hypergraph Concept Representation, Identifiable Difficulty Decomposition with Signed Visual Load & Gram-Schmidt Disentanglement (c_C, mu_concept, Delta_mu_stem, Delta_mu_sol, tilde_mu, d^q, x_q).
- Subsystem 3: Dual-Stream Asymmetric Encoder & Unpolluted Monotonic Knowledge Retriever (tilde_x, y_atomic, h, alpha_ret).
- Subsystem 4.1: Identifiable Multi-Trait Psychometric Interaction & Monotonic 2PL-IRT Readout (theta, a_q, p_hat).

Key performance and architectural optimizations:
1. Item Parameter Invariance: item representations depend purely on question content and skill hypergraph.
2. Unique-Item Batch Indexing: Runs Subsystem 1 and Subsystem 2 only on unique active items in the mini-batch.
3. Offline Item-Bank Evaluation Cache: Precomputes all 7,652 static representations once during validation/test inference.
4. One-Stage End-to-End Training: Directly optimizes L_ACD-MMKTv3 without disjoint two-stage pre-training.
5. Multi-Granular Evidence Extraction: extract_evidence for 100% faithful 4-tier pedagogical diagnostics.
"""

from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules.text_guided_cross_aligner import TextGuidedCrossAligner
from src.models.modules.identifiable_rasch_hypergraph import IdentifiableRaschHypergraph
from src.models.modules.atomic_monotonic_retriever import AtomicMonotonicRetriever
from src.models.modules.identifiable_2pl_irt import Identifiable2PLIRT


class ACDMMKTv3(nn.Module):
    """
    Asymmetric Causal-Disentangled Multimodal Knowledge Tracing Model v3.
    """

    def __init__(
        self,
        static_item_bank: Dict[str, torch.Tensor],
        d_raw_txt: int = 768,
        d_raw_img: int = 768,
        d_model: int = 128,
        num_heads: int = 4,
        max_seq_len: int = 200,
        dropout: float = 0.15,
        p_drop: float = 0.15,
        hgnn_layers: int = 2,
        gamma_vis: float = 1.5,
        delta_a: float = 0.1,
        use_solution: bool = True,
        use_visual: bool = True,
        unanchored_concept: bool = False,
        polluted_experience_attn: bool = False,
        use_deep_irt: bool = True,
        use_hypergraph: bool = True,
        use_gram_schmidt: bool = True,
        post_diag_mode: str = "none",
    ):
        """
        Initialize ACDMMKTv3.

        Args:
            static_item_bank: Dictionary of pre-extracted multimodal features and graph matrices.
            d_raw_txt: Dimensionality of raw text hidden states (768).
            d_raw_img: Dimensionality of raw visual features (768).
            d_model: Latent embedding dimension (128).
            num_heads: Number of attention heads (4).
            max_seq_len: Maximum sequence window length (200).
            dropout: Dropout probability (0.15).
            p_drop: Visual modality dropout probability (0.15).
            hgnn_layers: Number of hypergraph convolution layers (2).
            gamma_vis: Maximum visual cognitive load logit amplitude (1.5).
            delta_a: Minimum discrimination floor (0.1).
            use_solution: Ablation 2 flag (w/o solution algorithmic stream).
            use_visual: Ablation 3 flag (w/o text-guided visual stream).
            unanchored_concept: Ablation 4 flag (unanchored learned concept difficulty).
            polluted_experience_attn: Ablation 5 flag (polluted experience stream via self-attention).
            use_deep_irt: Ablation 6 flag (w/o monotonic 2PL-IRT readout).
            use_hypergraph: Whether to use HGNN propagation (default: True).
            use_gram_schmidt: Whether to use Gram-Schmidt projection (default: True).
            post_diag_mode: Post-diagnostic mode ("none" or "behavioral").
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout
        self.p_drop = p_drop
        self.gamma_vis = gamma_vis
        self.delta_a = delta_a
        self.use_solution = use_solution
        self.use_visual = use_visual
        self.unanchored_concept = unanchored_concept
        self.polluted_experience_attn = polluted_experience_attn
        self.use_deep_irt = use_deep_irt
        self.use_hypergraph = use_hypergraph
        self.use_gram_schmidt = use_gram_schmidt
        self.post_diag_mode = str(post_diag_mode).lower()

        # Register Tier 1 Static Item Bank tensors as non-persistent buffers
        self.register_buffer("acd_E_txt_q", static_item_bank["acd_E_txt_q"].half(), persistent=False)
        self.register_buffer("acd_mask_txt_q", static_item_bank["acd_mask_txt_q"].bool(), persistent=False)
        self.register_buffer("acd_E_img_q", static_item_bank["acd_E_img_q"].half(), persistent=False)
        self.register_buffer("acd_E_txt_s", static_item_bank["acd_E_txt_s"].half(), persistent=False)
        self.register_buffer("acd_mask_txt_s", static_item_bank["acd_mask_txt_s"].bool(), persistent=False)
        self.register_buffer("acd_E_img_s", static_item_bank["acd_E_img_s"].half(), persistent=False)
        self.register_buffer("acd_kc_init_emb", static_item_bank["acd_kc_init_emb"].float(), persistent=False)
        self.register_buffer("acd_hypergraph_G", static_item_bank["acd_hypergraph_G"].float(), persistent=False)
        self.register_buffer("bank_modality_masks", static_item_bank["modality_masks"].float(), persistent=False)
        self.register_buffer("bank_leaf_kcs", static_item_bank["question_leaf_kcs"].long(), persistent=False)
        self.register_buffer("bank_kc_mask", static_item_bank["question_kc_mask"].bool(), persistent=False)
        self.register_buffer("bank_static_ctx", static_item_bank.get("question_static_ctx", torch.zeros((7652, 15), dtype=torch.float32)).float(), persistent=False)

        # Letterbox patch mask M_patch_q
        if "M_patch_q" in static_item_bank and static_item_bank["M_patch_q"] is not None:
            self.register_buffer("bank_M_patch_q", static_item_bank["M_patch_q"].float(), persistent=False)
        else:
            self.register_buffer("bank_M_patch_q", torch.zeros((static_item_bank["modality_masks"].size(0), 49), dtype=torch.float32), persistent=False)

        self.num_questions = self.bank_modality_masks.size(0)
        self.num_kcs = self.acd_kc_init_emb.size(0)

        # -------------------------------------------------------------
        # Subsystems Initialization
        # -------------------------------------------------------------
        # Subsystem 2: Hypergraph Concept Representation, Difficulty Decomposition & Gram-Schmidt
        self.sub2 = IdentifiableRaschHypergraph(
            num_kcs=self.num_kcs,
            d_raw_kc=d_raw_txt,
            d_model=d_model,
            hgnn_layers=hgnn_layers,
            gamma_vis=gamma_vis,
            dropout=dropout,
            use_solution=use_solution,
            use_visual=use_visual,
            use_hypergraph=use_hypergraph,
            use_gram_schmidt=use_gram_schmidt,
            unanchored_concept=unanchored_concept,
        )

        # Subsystem 1: Text-Guided Alignment & Cognitive Gating
        self.sub1 = TextGuidedCrossAligner(
            d_raw_txt=d_raw_txt,
            d_raw_img=d_raw_img,
            d_model=d_model,
            d_static_ctx=15,
            dropout=dropout,
            use_visual=use_visual,
            use_solution=use_solution,
        )

        # Subsystem 3: Dual-Stream Asymmetric Encoder & Unpolluted Monotonic Knowledge Retriever
        d_post_diag = 4 if self.post_diag_mode == "behavioral" else 0
        self.sub3 = AtomicMonotonicRetriever(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            d_post_diag=d_post_diag,
            polluted_experience_attn=polluted_experience_attn,
        )

        # Subsystem 4.1: Identifiable Multi-Trait Psychometric Interaction & Monotonic 2PL-IRT
        self.sub4 = Identifiable2PLIRT(
            d_model=d_model,
            delta_a=delta_a,
            use_deep_irt=use_deep_irt,
            use_solution=use_solution,
        )

        # Evaluation item representation cache
        self.cached_item_properties: Optional[Dict[str, torch.Tensor]] = None

    def compute_all_kc_embeddings(self) -> torch.Tensor:
        """Propagate hypergraph once across all 1,175 KCs."""
        return self.sub2.concept_encoder.compute_all_kc_embeddings(
            raw_kc_emb=self.acd_kc_init_emb,
            hypergraph_G=self.acd_hypergraph_G,
        )

    def _forward_items_chunk(
        self,
        q_ids: torch.Tensor,
        all_kc_emb: torch.Tensor,
        mu_concept_prior: torch.Tensor,
        modality_masks: Optional[torch.Tensor] = None,
        return_cross_weights: bool = False,
    ) -> Dict[str, Any]:
        """Compute item representations for a single micro-chunk of questions."""
        q_ids = q_ids.long()
        M = q_ids.size(0)

        E_txt_q = self.acd_E_txt_q[q_ids]
        mask_txt_q = self.acd_mask_txt_q[q_ids]
        E_img_q = self.acd_E_img_q[q_ids]
        M_patch_q = self.bank_M_patch_q[q_ids]

        E_txt_s = self.acd_E_txt_s[q_ids]
        mask_txt_s = self.acd_mask_txt_s[q_ids]
        E_img_s = self.acd_E_img_s[q_ids]

        leaf_kcs = self.bank_leaf_kcs[q_ids]
        leaf_kc_mask = self.bank_kc_mask[q_ids]
        static_ctx = self.bank_static_ctx[q_ids]

        if modality_masks is None:
            mod_masks = self.bank_modality_masks[q_ids]
        else:
            mod_masks = modality_masks

        # 1. Concept Representation c_C (Subsystem 2.1)
        c_C = self.sub2.concept_encoder(
            all_kc_emb=all_kc_emb,
            leaf_kcs=leaf_kcs,
            leaf_kc_mask=leaf_kc_mask,
        )

        # 2. Text-Guided Cross-Modal Alignment & Cognitive Gating (Subsystem 1)
        sub1_out = self.sub1(
            E_txt_q=E_txt_q,
            mask_txt_q=mask_txt_q,
            E_img_q=E_img_q,
            M_patch_q=M_patch_q,
            E_txt_s=E_txt_s,
            mask_txt_s=mask_txt_s,
            E_img_s=E_img_s,
            modality_masks=mod_masks,
            c_C=c_C,
            c_ctx=static_ctx,
            return_cross_weights=return_cross_weights,
        )

        # 3. Difficulty Decomposition & Gram-Schmidt Orthogonalization (Subsystem 2.2 & 2.3)
        sub2_out = self.sub2(
            v_stem=sub1_out["v_stem"],
            v_sol=sub1_out["v_sol"],
            u_txt_q=sub1_out["u_txt_q"],
            u_cross_q=sub1_out["u_cross_q"],
            bar_lambda_q=sub1_out["bar_lambda_q"],
            m_img_q=mod_masks[:, 0:1],
            c_C=c_C,
            mu_concept_prior=mu_concept_prior,
        )

        # 4. Item Discrimination a_q (Subsystem 4.1 branch)
        if self.use_solution:
            disc_in = torch.cat([c_C, sub1_out["v_stem"], sub1_out["v_sol"]], dim=-1)
        else:
            disc_in = torch.cat([c_C, sub1_out["v_stem"]], dim=-1)

        raw_a = self.sub4.fc_a(disc_in).squeeze(-1).float() if self.use_deep_irt else torch.ones_like(sub2_out["tilde_mu"])
        a_q = F.softplus(raw_a) + self.delta_a if self.use_deep_irt else torch.ones_like(sub2_out["tilde_mu"])

        item_props = {
            "q_ids": q_ids,
            "c_C": c_C,
            "v_stem": sub1_out["v_stem"],
            "v_sol": sub1_out["v_sol"],
            "lambda_q": sub1_out["lambda_q"],
            "bar_lambda_q": sub1_out["bar_lambda_q"],
            "beta_txt_q": sub1_out["beta_txt_q"],
            "beta_txt_s": sub1_out["beta_txt_s"],
            "A_cross_q": sub1_out["A_cross_q"],
            "u_txt_q": sub1_out["u_txt_q"],
            "u_cross_q": sub1_out["u_cross_q"],
            "mu_concept": sub2_out["mu_concept"],
            "delta_mu_txt": sub2_out["delta_mu_txt"],
            "delta_mu_vis": sub2_out["delta_mu_vis"],
            "delta_mu_stem": sub2_out["delta_mu_stem"],
            "delta_mu_sol": sub2_out["delta_mu_sol"],
            "tilde_mu": sub2_out["tilde_mu"],
            "mu_q": sub2_out["mu_q"],
            "d_q": sub2_out["d_q"],
            "x_q": sub2_out["x_q"],
            "a_q": a_q,
        }
        return item_props

    def forward_items(
        self,
        q_ids: torch.Tensor,
        all_kc_emb: torch.Tensor,
        mu_concept_prior: torch.Tensor,
        modality_masks: Optional[torch.Tensor] = None,
        chunk_size: int = 256,
        return_cross_weights: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute item representations for a set of question IDs using micro-chunking.
        Guarantees Item Parameter Invariance: independent of student response history.
        """
        q_ids = q_ids.long()
        M = q_ids.size(0)

        if M <= chunk_size:
            return self._forward_items_chunk(
                q_ids=q_ids,
                all_kc_emb=all_kc_emb,
                mu_concept_prior=mu_concept_prior,
                modality_masks=modality_masks,
                return_cross_weights=return_cross_weights,
            )

        accumulated = {}
        for start_idx in range(0, M, chunk_size):
            end_idx = min(start_idx + chunk_size, M)
            sub_q = q_ids[start_idx:end_idx]
            sub_prior = mu_concept_prior[start_idx:end_idx]
            sub_masks = modality_masks[start_idx:end_idx] if modality_masks is not None else None

            chunk_out = self._forward_items_chunk(
                q_ids=sub_q,
                all_kc_emb=all_kc_emb,
                mu_concept_prior=sub_prior,
                modality_masks=sub_masks,
                return_cross_weights=return_cross_weights,
            )
            for k, v in chunk_out.items():
                if v is None:
                    continue
                if k not in accumulated:
                    accumulated[k] = []
                accumulated[k].append(v)

        merged = {}
        for k, v_list in accumulated.items():
            merged[k] = torch.cat(v_list, dim=0)
        return merged

    def precompute_item_bank_cache(
        self,
        mu_concept_prior: Optional[torch.Tensor] = None,
        chunk_size: int = 256,
    ) -> None:
        """
        Precompute all 7,652 static question representations once for fast inference.
        """
        self.eval()
        with torch.no_grad():
            all_kc_emb = self.compute_all_kc_embeddings()
            all_q_ids = torch.arange(self.num_questions, device=self.bank_modality_masks.device)
            if mu_concept_prior is not None:
                prior_tensor = mu_concept_prior.to(all_q_ids.device)
            else:
                prior_tensor = torch.full((self.num_questions,), -1.37, device=all_q_ids.device, dtype=torch.float32)

            self.cached_item_properties = self.forward_items(
                q_ids=all_q_ids,
                all_kc_emb=all_kc_emb,
                mu_concept_prior=prior_tensor,
                modality_masks=self.bank_modality_masks,
                chunk_size=chunk_size,
                return_cross_weights=False,
            )

    def clear_item_bank_cache(self) -> None:
        """Clear cached item bank representations."""
        self.cached_item_properties = None

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        mu_concept_prior: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Sequential KT Forward Pass for ACD-MMKTv3.

        Args:
            batch: Dictionary containing q_seq, r_seq, seq_mask, loss_mask,
                   delta_t_lag, delta_t_kc, modality_masks, raw_modality_masks, etc.
            mu_concept_prior: Optional fold concept difficulty prior tensor [7652].

        Returns:
            Dictionary containing prediction logits, probabilities, IRT parameters, and attention maps.
        """
        q_seq = batch["q_seq"]      # [B, L]
        r_seq = batch["r_seq"]      # [B, L]
        seq_mask = batch["seq_mask"]  # [B, L]
        B, L = q_seq.shape
        device = q_seq.device

        norm_delta_t_lag = batch.get("norm_delta_t_lag", batch.get("delta_t_lag"))
        norm_delta_t_kc = batch.get("norm_delta_t_kc", batch.get("delta_t_kc"))

        # Resolve mu_concept_prior tensor
        if mu_concept_prior is None:
            if "mu_concept" in batch:
                # If sequence batch has mu_concept, we can extract from it or lookup
                batch_mu_concept = batch["mu_concept"]  # [B, L]
            else:
                batch_mu_concept = torch.full((B, L), -1.37, device=device, dtype=torch.float32)
        else:
            batch_mu_concept = mu_concept_prior[q_seq].to(device)

        # Check if batch specifies custom modality masks different from bank masks
        custom_masks = batch.get("modality_masks")
        has_custom_masks = False
        if custom_masks is not None:
            active_q = q_seq[seq_mask.bool()] if seq_mask.bool().any() else q_seq.view(-1)
            if active_q.numel() > 0:
                bank_masks_active = self.bank_modality_masks[active_q]
                batch_masks_active = custom_masks[seq_mask.bool()] if seq_mask.bool().any() else custom_masks.view(-1, 3)
                if not torch.allclose(bank_masks_active, batch_masks_active):
                    has_custom_masks = True

        # -------------------------------------------------------------
        # Path A: Fast Inference using Cached Item Representations
        # -------------------------------------------------------------
        if use_cache and not self.training and self.cached_item_properties is not None and not has_custom_masks:
            cache = self.cached_item_properties
            c_C_seq = cache["c_C"][q_seq]                  # [B, L, d_model]
            v_stem_seq = cache["v_stem"][q_seq]            # [B, L, d_model]
            v_sol_seq = cache["v_sol"][q_seq]              # [B, L, d_model]
            x_seq = cache["x_q"][q_seq]                    # [B, L, d_model]
            d_q_seq = cache["d_q"][q_seq]                  # [B, L, d_model]
            mu_q_seq = cache["mu_q"][q_seq]                # [B, L]
            tilde_mu_seq = cache["tilde_mu"][q_seq]        # [B, L]
            a_seq = cache["a_q"][q_seq]                    # [B, L]
            mu_concept_seq = cache["mu_concept"][q_seq]    # [B, L]
            delta_mu_stem_seq = cache["delta_mu_stem"][q_seq]  # [B, L]
            delta_mu_sol_seq = cache["delta_mu_sol"][q_seq]    # [B, L]
            delta_mu_txt_seq = cache["delta_mu_txt"][q_seq]    # [B, L]
            delta_mu_vis_seq = cache["delta_mu_vis"][q_seq]    # [B, L]
            bar_lambda_seq = cache["bar_lambda_q"][q_seq]      # [B, L, 1]
            lambda_seq = cache["lambda_q"][q_seq]              # [B, L, d_model]

            clean_item_props = None
            unique_q = None

        # -------------------------------------------------------------
        # Path B: Dynamic / Training Mode via Unique Item Batch Indexing
        # -------------------------------------------------------------
        else:
            all_kc_emb = self.compute_all_kc_embeddings()

            # Filter valid questions (strictly ignore padding question 0 from sequence mask)
            valid_mask = seq_mask.bool()
            if valid_mask.any():
                unique_q = torch.unique(q_seq[valid_mask])
            else:
                unique_q = torch.unique(q_seq)

            if mu_concept_prior is not None:
                uniq_prior = mu_concept_prior[unique_q].to(device)
            else:
                # Fallback to mean batch prior for active questions
                uniq_prior = torch.full((len(unique_q),), -1.37, device=device, dtype=torch.float32)

            uniq_clean_masks = self.bank_modality_masks[unique_q]

            # 1. Forward unique items once on clean modalities (for L_D^weighted and base representation)
            clean_item_props = self.forward_items(
                q_ids=unique_q,
                all_kc_emb=all_kc_emb,
                mu_concept_prior=uniq_prior,
                modality_masks=uniq_clean_masks,
                chunk_size=256,
                return_cross_weights=False,
            )

            # 2. Dynamic modality dropout: only re-forward questions that actually dropped a channel
            if self.training and self.p_drop > 0.0 and self.use_visual:
                uniq_mod_masks = uniq_clean_masks.clone()
                drop_q = (torch.rand(len(unique_q), device=device) >= self.p_drop).float()
                uniq_mod_masks[:, 0] = uniq_mod_masks[:, 0] * drop_q
                drop_s = (torch.rand(len(unique_q), device=device) >= self.p_drop).float()
                uniq_mod_masks[:, 2] = uniq_mod_masks[:, 2] * drop_s

                dropped_sub_indices = torch.where((uniq_mod_masks != uniq_clean_masks).any(dim=-1))[0]
                if len(dropped_sub_indices) > 0:
                    dyn_item_props = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in clean_item_props.items()}
                    dropped_q_ids = unique_q[dropped_sub_indices]
                    dropped_prior = uniq_prior[dropped_sub_indices]
                    dropped_masks = uniq_mod_masks[dropped_sub_indices]

                    dropped_props = self.forward_items(
                        q_ids=dropped_q_ids,
                        all_kc_emb=all_kc_emb,
                        mu_concept_prior=dropped_prior,
                        modality_masks=dropped_masks,
                        chunk_size=256,
                        return_cross_weights=False,
                    )
                    for k, v in dropped_props.items():
                        if isinstance(v, torch.Tensor) and k in dyn_item_props:
                            dyn_item_props[k][dropped_sub_indices] = v
                else:
                    dyn_item_props = clean_item_props
            elif has_custom_masks and custom_masks is not None:
                uniq_mod_masks = uniq_clean_masks.clone()
                for i_q, q_val in enumerate(unique_q):
                    pos = torch.where((q_seq == q_val) & valid_mask)
                    if len(pos[0]) > 0:
                        uniq_mod_masks[i_q] = custom_masks[pos[0][0], pos[1][0]]
                    else:
                        pos_fallback = torch.where(q_seq == q_val)
                        if len(pos_fallback[0]) > 0:
                            uniq_mod_masks[i_q] = custom_masks[pos_fallback[0][0], pos_fallback[1][0]]
                dyn_item_props = self.forward_items(
                    q_ids=unique_q,
                    all_kc_emb=all_kc_emb,
                    mu_concept_prior=uniq_prior,
                    modality_masks=uniq_mod_masks,
                    chunk_size=256,
                    return_cross_weights=False,
                )
            else:
                dyn_item_props = clean_item_props

            # Create fast mapping from question ID to unique index
            max_q_id = self.num_questions
            q_to_uniq = torch.full((max_q_id,), -1, dtype=torch.long, device=device)
            q_to_uniq[unique_q] = torch.arange(len(unique_q), device=device)

            # Map unique representations to sequence grid [B, L, ...]
            # For padding positions where q_seq might not be in unique_q, clamp to index 0
            safe_q_seq = q_seq.clone()
            if (~valid_mask).any():
                safe_q_seq[~valid_mask] = unique_q[0]
            indices_in_uniq = q_to_uniq[safe_q_seq]

            c_C_seq = dyn_item_props["c_C"][indices_in_uniq]
            v_stem_seq = dyn_item_props["v_stem"][indices_in_uniq]
            v_sol_seq = dyn_item_props["v_sol"][indices_in_uniq]
            x_seq = dyn_item_props["x_q"][indices_in_uniq]
            d_q_seq = dyn_item_props["d_q"][indices_in_uniq]
            mu_q_seq = dyn_item_props["mu_q"][indices_in_uniq]
            tilde_mu_seq = dyn_item_props["tilde_mu"][indices_in_uniq]
            a_seq = dyn_item_props["a_q"][indices_in_uniq]
            mu_concept_seq = dyn_item_props["mu_concept"][indices_in_uniq]
            delta_mu_stem_seq = dyn_item_props["delta_mu_stem"][indices_in_uniq]
            delta_mu_sol_seq = dyn_item_props["delta_mu_sol"][indices_in_uniq]
            delta_mu_txt_seq = dyn_item_props["delta_mu_txt"][indices_in_uniq]
            delta_mu_vis_seq = dyn_item_props["delta_mu_vis"][indices_in_uniq]
            bar_lambda_seq = dyn_item_props["bar_lambda_q"][indices_in_uniq]
            lambda_seq = dyn_item_props["lambda_q"][indices_in_uniq]

        # -------------------------------------------------------------
        # Subsystem 3: Dual-Stream Asymmetric Encoder & Retriever
        # -------------------------------------------------------------
        v_post_diag = None
        if self.post_diag_mode == "behavioral":
            batch_flag = batch.get("batch_flag", torch.zeros_like(r_seq, dtype=torch.float32)).unsqueeze(-1)
            norm_lag = (norm_delta_t_lag if norm_delta_t_lag is not None else torch.zeros_like(r_seq, dtype=torch.float32)).unsqueeze(-1)
            q_fmt = self.bank_static_ctx[q_seq, 0:1]  # question format flag from static context
            v_post_diag = torch.cat([r_seq.unsqueeze(-1).float(), batch_flag.float(), norm_lag.float(), q_fmt.float()], dim=-1)

        h_seq, tilde_x_seq, alpha_ret = self.sub3(
            x_seq=x_seq,
            c_C_seq=c_C_seq,
            mu_q_seq=mu_q_seq,
            d_q_seq=d_q_seq,
            r_seq=r_seq,
            delta_t_lag=norm_delta_t_lag,
            delta_t_kc=norm_delta_t_kc,
            v_post_diag=v_post_diag,
        )

        # -------------------------------------------------------------
        # Subsystem 4.1: Monotonic 2PL-IRT Prediction
        # -------------------------------------------------------------
        logits, probs, theta, a_disc = self.sub4(
            h_t=h_seq,
            c_C_t=c_C_seq,
            d_q_t=d_q_seq,
            v_stem_t=v_stem_seq,
            v_sol_t=v_sol_seq,
            tilde_mu_t=tilde_mu_seq,
            tilde_x_t=tilde_x_seq,
        )

        return {
            "logits": logits,
            "probs": probs,
            "theta": theta,
            "a_disc": a_disc,
            "tilde_mu": tilde_mu_seq,
            "mu_q": mu_q_seq,
            "mu_concept": mu_concept_seq,
            "delta_mu_stem": delta_mu_stem_seq,
            "delta_mu_sol": delta_mu_sol_seq,
            "delta_mu_txt": delta_mu_txt_seq,
            "delta_mu_vis": delta_mu_vis_seq,
            "bar_lambda_q": bar_lambda_seq,
            "lambda_q": lambda_seq,
            "alpha_ret": alpha_ret,
            "unique_q": unique_q,
            "clean_item_props": clean_item_props,
        }

    @torch.no_grad()
    def extract_evidence(
        self,
        batch: Dict[str, torch.Tensor],
        sample_idx: int = 0,
        target_step: int = -1,
        top_k_history: int = 5,
        mu_concept_prior: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Extract the 100% faithful 4-Tier Evidence Package E_t for a single interaction.

        Returns:
            Dictionary containing 4 tiers of psychometric and multimodal evidence.
        """
        self.eval()
        q_seq = batch["q_seq"]
        seq_mask = batch["seq_mask"]
        eval_mask = batch.get("eval_mask", seq_mask)
        device = q_seq.device

        # Determine target step
        if target_step == -1:
            valid_steps = torch.where(eval_mask[sample_idx])[0]
            if len(valid_steps) > 0:
                t = int(valid_steps[-1].item())
            else:
                t = int(torch.where(seq_mask[sample_idx])[0][-1].item())
        else:
            t = target_step

        target_q_id = int(q_seq[sample_idx, t].item())

        # Forward pass on the full batch to get historical states
        outputs = self.forward(batch, mu_concept_prior=mu_concept_prior)

        # Compute detailed item properties with attention weights for the target question
        all_kc_emb = self.compute_all_kc_embeddings()
        single_q = torch.tensor([target_q_id], device=device, dtype=torch.long)
        single_prior = (
            mu_concept_prior[single_q]
            if mu_concept_prior is not None
            else torch.tensor([-1.37], device=device)
        )

        if "modality_masks" in batch and batch["modality_masks"] is not None:
            single_mask = batch["modality_masks"][sample_idx, t : t + 1].to(device)
        else:
            single_mask = self.bank_modality_masks[single_q]

        item_detail = self.forward_items(
            q_ids=single_q,
            all_kc_emb=all_kc_emb,
            mu_concept_prior=single_prior,
            modality_masks=single_mask,
            chunk_size=1,
            return_cross_weights=True,
        )

        # -------------------------------------------------------------
        # Tier 1: Psychometric Equilibrium Evidence
        # -------------------------------------------------------------
        theta_val = float(outputs["theta"][sample_idx, t].item())
        tilde_mu_val = float(outputs["tilde_mu"][sample_idx, t].item())
        mu_q_val = float(outputs["mu_q"][sample_idx, t].item())
        a_val = float(outputs["a_disc"][sample_idx, t].item())
        p_hat_val = float(outputs["probs"][sample_idx, t].item())
        delta_gap_val = theta_val - tilde_mu_val

        tier1 = {
            "theta": theta_val,
            "tilde_mu": tilde_mu_val,
            "mu_q": mu_q_val,
            "a_disc": a_val,
            "p_hat": p_hat_val,
            "delta_gap": delta_gap_val,
            "prediction_faithful": bool((p_hat_val >= 0.5) == (delta_gap_val >= 0.0)),
        }

        # -------------------------------------------------------------
        # Tier 2: Difficulty Source Anatomy
        # -------------------------------------------------------------
        mu_concept_val = float(item_detail["mu_concept"][0].item())
        delta_stem_val = float(item_detail["delta_mu_stem"][0].item())
        delta_sol_val = float(item_detail["delta_mu_sol"][0].item())

        tier2 = {
            "mu_concept": mu_concept_val,
            "delta_mu_stem": delta_stem_val,
            "delta_mu_sol": delta_sol_val,
            "tilde_mu_reconstructed": mu_concept_val + delta_stem_val + delta_sol_val,
        }

        # -------------------------------------------------------------
        # Tier 3: Fine-Grained Multimodal Attribution
        # -------------------------------------------------------------
        delta_txt_val = float(item_detail["delta_mu_txt"][0].item())
        delta_vis_val = float(item_detail["delta_mu_vis"][0].item())
        bar_lambda_val = float(item_detail["bar_lambda_q"][0].item())
        beta_txt_q_np = item_detail["beta_txt_q"][0].cpu().numpy()
        beta_txt_s_np = item_detail["beta_txt_s"][0].cpu().numpy()

        A_cross_q_np = (
            item_detail["A_cross_q"][0].cpu().numpy()
            if item_detail["A_cross_q"] is not None
            else None
        )

        tier3 = {
            "delta_mu_txt": delta_txt_val,
            "delta_mu_vis": delta_vis_val,
            "bar_lambda_q": bar_lambda_val,
            "visual_scaffold_flag": bool(delta_vis_val < 0.0),
            "visual_distractor_flag": bool(delta_vis_val > 0.0),
            "beta_txt_q": beta_txt_q_np,
            "beta_txt_s": beta_txt_s_np,
            "A_cross_q": A_cross_q_np,
        }

        # -------------------------------------------------------------
        # Tier 4: Empirical History Tracing
        # -------------------------------------------------------------
        alpha_weights = outputs["alpha_ret"][sample_idx, t, :t].cpu().numpy()
        top_k = min(top_k_history, len(alpha_weights))
        top_indices = np.argsort(alpha_weights)[::-1][:top_k]

        context_offset = int(
            batch.get(
                "context_start",
                torch.zeros(batch["q_seq"].size(0), dtype=torch.int64),
            )[sample_idx].item()
        )

        history_items = []
        for idx in top_indices:
            past_q = int(q_seq[sample_idx, idx].item())
            past_r = int(batch["r_seq"][sample_idx, idx].item())
            past_mu = float(outputs["mu_q"][sample_idx, idx].item())
            weight = float(alpha_weights[idx])
            b_flag = int(batch["batch_flag"][sample_idx, idx].item()) if "batch_flag" in batch else 0
            t_lag = float(batch["delta_t_lag"][sample_idx, idx].item()) if "delta_t_lag" in batch else 0.0

            history_items.append({
                "local_step": int(idx),
                "global_step": int(idx + context_offset),
                "q_id": past_q,
                "r": past_r,
                "mu_q": past_mu,
                "alpha_weight": weight,
                "batch_flag": b_flag,
                "delta_t_lag": t_lag,
            })

        tier4 = {
            "target_step": t,
            "global_target_step": t + context_offset,
            "top_interactions": history_items,
        }

        return {
            "sample_idx": sample_idx,
            "target_step": t,
            "target_q_id": target_q_id,
            "tier1_psychometrics": tier1,
            "tier2_difficulty": tier2,
            "tier3_multimodal": tier3,
            "tier4_history": tier4,
        }
