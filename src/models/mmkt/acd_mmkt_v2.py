"""
End-to-End ACD-MMKTv2 Model Architecture.
Asymmetric Causal-Disentangled Multimodal Knowledge Tracing v2.

Integrates:
- Subsystem 1: Symmetric Fine-Grained Cross-Modal Alignment & Intrinsic Gating (v_stem, v_sol).
- Subsystem 2: Additive Multi-Source Rasch Modeling & Gram-Schmidt Disentanglement (c_C, tilde_mu, d^q, x_q).
- Subsystem 3: Dual-Stream Asymmetric Encoder & Monotonic Knowledge Retriever (tilde_x, tilde_y, h).
- Subsystem 4.1: Multi-Trait Hadamard Interaction & Monotonic 2PL-IRT Readout (theta, a, p_hat).

Key performance optimizations:
1. Item Parameter Invariance: item representations (x_q, d_q, tilde_mu_q) depend purely on question content.
2. Unique-Item Batch Indexing: Computes Subsystem 1 and Subsystem 2 only on unique active items in the mini-batch,
   slashing attention and vision computation by up to 30x.
3. Offline Item-Bank Evaluation: Precomputes all 7,652 static representations once during validation/test inference.
4. Two-Stage Curriculum Support: forward_stage1 for item-bank pretraining, forward for sequential KT.
5. Multi-Granular Evidence Extraction: extract_evidence for 100% faithful 4-tier pedagogical diagnostics.
"""

from typing import Dict, Any, Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules.symmetric_cross_modal_aligner import SymmetricCrossModalAligner
from src.models.modules.additive_rasch_hypergraph import AdditiveRaschHypergraph
from src.models.modules.asymmetric_monotonic_retriever import AsymmetricMonotonicRetriever
from src.models.modules.monotonic_2pl_irt import Monotonic2PLIRT


class ACDMMKTv2(nn.Module):
    """
    Asymmetric Causal-Disentangled Multimodal Knowledge Tracing Model v2.
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
        use_solution: bool = True,
        use_fine_grained: bool = True,
        use_hypergraph: bool = True,
        use_gram_schmidt: bool = True,
        use_deep_irt: bool = True,
    ):
        """
        Initialize ACDMMKTv2.

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
            use_solution: Ablation 2 flag (w/o solution cognitive blueprint).
            use_fine_grained: Ablation 3 flag (w/o fine-grained cross-modal attention).
            use_hypergraph: Ablation 4 flag (w/o hypergraph convolution).
            use_gram_schmidt: Ablation 4 flag (w/o Gram-Schmidt orthogonal projection).
            use_deep_irt: Ablation 5 flag (w/o monotonic 2PL-IRT readout).
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout
        self.p_drop = p_drop
        self.use_solution = use_solution
        self.use_fine_grained = use_fine_grained
        self.use_hypergraph = use_hypergraph
        self.use_gram_schmidt = use_gram_schmidt
        self.use_deep_irt = use_deep_irt

        # Register Tier 1 Static Item Bank tensors as non-persistent buffers (avoids duplicating in checkpoint)
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

        self.num_questions = self.bank_modality_masks.size(0)
        self.num_kcs = self.acd_kc_init_emb.size(0)

        # -------------------------------------------------------------
        # Subsystems Initialization
        # -------------------------------------------------------------
        # Subsystem 2: Hypergraph Concept Embedding & Additive Rasch Decomposition
        self.sub2 = AdditiveRaschHypergraph(
            num_kcs=self.num_kcs,
            d_raw_kc=d_raw_txt,
            d_model=d_model,
            hgnn_layers=hgnn_layers,
            dropout=dropout,
            use_solution=use_solution,
            use_hypergraph=use_hypergraph,
            use_gram_schmidt=use_gram_schmidt,
        )

        # Subsystem 1: Symmetric Cross-Modal Alignment & Intrinsic Gating
        self.sub1 = SymmetricCrossModalAligner(
            d_raw_txt=d_raw_txt,
            d_raw_img=d_raw_img,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_fine_grained=use_fine_grained,
        )

        # Subsystem 3: Dual-Stream Asymmetric Encoder & Monotonic Knowledge Retriever
        self.sub3 = AsymmetricMonotonicRetriever(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        # Subsystem 4.1: Multi-Trait Hadamard Psychometric Interaction & Monotonic 2PL-IRT
        self.sub4 = Monotonic2PLIRT(
            d_model=d_model,
            use_deep_irt=use_deep_irt,
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
        modality_masks: Optional[torch.Tensor] = None,
        return_cross_weights: bool = False,
    ) -> Dict[str, Any]:
        """Compute item representations for a single micro-chunk of questions."""
        q_ids = q_ids.long()
        M = q_ids.size(0)

        # Retrieve static item features
        E_txt_q = self.acd_E_txt_q[q_ids]
        mask_txt_q = self.acd_mask_txt_q[q_ids]
        E_img_q = self.acd_E_img_q[q_ids]
        E_txt_s = self.acd_E_txt_s[q_ids]
        mask_txt_s = self.acd_mask_txt_s[q_ids]
        E_img_s = self.acd_E_img_s[q_ids]

        leaf_kcs = self.bank_leaf_kcs[q_ids]
        leaf_kc_mask = self.bank_kc_mask[q_ids]

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

        # 2. Symmetric Cross-Modal Alignment & Intrinsic Gating (Subsystem 1)
        sub1_out = self.sub1(
            E_txt_q=E_txt_q,
            mask_txt_q=mask_txt_q,
            E_img_q=E_img_q,
            E_txt_s=E_txt_s,
            mask_txt_s=mask_txt_s,
            E_img_s=E_img_s,
            modality_masks=mod_masks,
            c_C=c_C,
            return_cross_weights=return_cross_weights,
        )

        # 3. Additive Difficulty Decomposition & Gram-Schmidt Orthogonalization (Subsystem 2.2 & 2.3)
        sub2_out = self.sub2(
            all_kc_emb=all_kc_emb,
            leaf_kcs=leaf_kcs,
            leaf_kc_mask=leaf_kc_mask,
            v_stem=sub1_out["v_stem"],
            v_sol=sub1_out["v_sol"],
            c_C=c_C,
        )

        item_props = {
            "q_ids": q_ids,
            "c_C": c_C,
            "v_stem": sub1_out["v_stem"],
            "v_sol": sub1_out["v_sol"],
            "lambda_q": sub1_out["lambda_q"],
            "lambda_s": sub1_out["lambda_s"],
            "beta_txt_q": sub1_out["beta_txt_q"],
            "beta_txt_s": sub1_out["beta_txt_s"],
            "A_cross_q": sub1_out["A_cross_q"],
            "A_cross_s": sub1_out["A_cross_s"],
            "mu_concept": sub2_out["mu_concept"],
            "delta_mu_stem": sub2_out["delta_mu_stem"],
            "delta_mu_sol": sub2_out["delta_mu_sol"],
            "tilde_mu": sub2_out["tilde_mu"],
            "mu_q": sub2_out["mu_q"],
            "d_q": sub2_out["d_q"],
            "x_q": sub2_out["x_q"],
        }
        return item_props

    def forward_items(
        self,
        q_ids: torch.Tensor,
        all_kc_emb: torch.Tensor,
        modality_masks: Optional[torch.Tensor] = None,
        chunk_size: int = 256,
        return_cross_weights: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute intrinsic multimodal item representations for a set of question IDs.
        Uses micro-chunking to cap peak attention memory under 50 MB per chunk.

        Guarantees Item Parameter Invariance: depends purely on question content and concept graph.

        Args:
            q_ids: 1D tensor of question indices [M].
            all_kc_emb: Enriched KC embeddings [1175, d_model].
            modality_masks: Optional overridden modality masks [M, 3].
            chunk_size: Micro-chunk size to avoid OOM on unified memory (default: 256).
            return_cross_weights: Whether to compute and retain cross-modal attention maps.

        Returns:
            Dictionary of item representations.
        """
        q_ids = q_ids.long()
        M = q_ids.size(0)
        if M <= chunk_size:
            return self._forward_items_chunk(q_ids, all_kc_emb, modality_masks, return_cross_weights=return_cross_weights)

        chunk_results = []
        for i in range(0, M, chunk_size):
            chunk_q = q_ids[i : i + chunk_size]
            chunk_masks = modality_masks[i : i + chunk_size] if modality_masks is not None else None
            res = self._forward_items_chunk(chunk_q, all_kc_emb, chunk_masks, return_cross_weights=return_cross_weights)
            chunk_results.append(res)

        merged = {}
        for k in chunk_results[0].keys():
            if chunk_results[0][k] is not None:
                merged[k] = torch.cat([cr[k] for cr in chunk_results], dim=0)
            else:
                merged[k] = None
        return merged

    def precompute_item_cache(self, batch_size: int = 512) -> None:
        """
        Precompute and cache static item properties for all 7,652 questions.
        Used to achieve ultra-fast inference during validation and test evaluations.
        """
        self.eval()
        with torch.no_grad():
            all_kc_emb = self.compute_all_kc_embeddings()
            q_all = torch.arange(self.num_questions, device=self.bank_modality_masks.device)

            cached_x_q = []
            cached_c_C = []
            cached_d_q = []
            cached_tilde_mu = []
            cached_mu_q = []
            cached_v_stem = []
            cached_v_sol = []
            cached_mu_concept = []
            cached_delta_mu_stem = []
            cached_delta_mu_sol = []
            cached_lambda_q = []
            cached_lambda_s = []

            for i in range(0, self.num_questions, batch_size):
                q_batch = q_all[i : i + batch_size]
                out = self.forward_items(q_batch, all_kc_emb=all_kc_emb)
                cached_x_q.append(out["x_q"])
                cached_c_C.append(out["c_C"])
                cached_d_q.append(out["d_q"])
                cached_tilde_mu.append(out["tilde_mu"])
                cached_mu_q.append(out["mu_q"])
                cached_v_stem.append(out["v_stem"])
                cached_v_sol.append(out["v_sol"])
                cached_mu_concept.append(out["mu_concept"])
                cached_delta_mu_stem.append(out["delta_mu_stem"])
                cached_delta_mu_sol.append(out["delta_mu_sol"])
                cached_lambda_q.append(out["lambda_q"])
                cached_lambda_s.append(out["lambda_s"])

            self.cached_item_properties = {
                "all_kc_emb": all_kc_emb,
                "x_q": torch.cat(cached_x_q, dim=0),
                "c_C": torch.cat(cached_c_C, dim=0),
                "d_q": torch.cat(cached_d_q, dim=0),
                "tilde_mu": torch.cat(cached_tilde_mu, dim=0),
                "mu_q": torch.cat(cached_mu_q, dim=0),
                "v_stem": torch.cat(cached_v_stem, dim=0),
                "v_sol": torch.cat(cached_v_sol, dim=0),
                "mu_concept": torch.cat(cached_mu_concept, dim=0),
                "delta_mu_stem": torch.cat(cached_delta_mu_stem, dim=0),
                "delta_mu_sol": torch.cat(cached_delta_mu_sol, dim=0),
                "lambda_q": torch.cat(cached_lambda_q, dim=0),
                "lambda_s": torch.cat(cached_lambda_s, dim=0),
            }

    def clear_item_cache(self) -> None:
        """Clear precomputed static item cache."""
        self.cached_item_properties = None

    def forward_stage1(
        self,
        q_ids: torch.Tensor,
        modality_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Stage 1: Item-Bank Multimodal Pre-training forward pass.

        Computes Subsystem 1 and Subsystem 2 directly on question items without student history.
        Optimized by L_D^weighted with Bayesian sample size weights.

        Args:
            q_ids: Question indices [M].
            modality_masks: Optional modality masks [M, 3].

        Returns:
            Dictionary containing item difficulty outputs (tilde_mu, mu_q, mu_concept, etc.).
        """
        all_kc_emb = self.compute_all_kc_embeddings()
        return self.forward_items(q_ids, all_kc_emb=all_kc_emb, modality_masks=modality_masks)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        use_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Stage 2 & Inference: End-to-End Sequential Multimodal Knowledge Tracing.

        Args:
            batch: Dictionary from DataLoader containing:
                   - 'q_seq': [B, L]
                   - 'r_seq': [B, L]
                   - 'seq_mask': [B, L]
                   - 'modality_masks': [B, L, 3] (dropped masks during train, un-dropped during eval)
                   - 'raw_modality_masks': [B, L, 3] (un-dropped ground-truth masks)
            use_cache: Whether to reuse precomputed static item properties (for fast test evaluation).

        Returns:
            Dictionary containing:
            - 'logits': Prediction logits [B, L].
            - 'probs': Response probabilities [B, L].
            - 'theta': Student proficiency [B, L].
            - 'a_disc': Item discrimination [B, L].
            - 'tilde_mu_seq': Additive total logit difficulty [B, L, 1].
            - 'mu_q_seq': Probability scale difficulty [B, L, 1].
            - 'alpha_ret': Historical retrieval weights [B, L, L].
            - 'active_unique_q': Tensor of unique active question IDs in the batch.
            - 'active_unique_tilde_mu': Logit difficulty of unique active questions (with clean un-dropped masks).
        """
        q_seq = batch["q_seq"]          # [B, L]
        r_seq = batch["r_seq"]          # [B, L]
        seq_mask = batch["seq_mask"]    # [B, L]
        B, L = q_seq.shape
        device = q_seq.device

        # Check if precomputed evaluation cache can be utilized
        if use_cache and (not self.training) and (self.cached_item_properties is not None):
            cache = self.cached_item_properties
            x_seq = cache["x_q"][q_seq]             # [B, L, d_model]
            c_C_seq = cache["c_C"][q_seq]           # [B, L, d_model]
            d_q_seq = cache["d_q"][q_seq]           # [B, L, d_model]
            tilde_mu_seq = cache["tilde_mu"][q_seq] # [B, L, 1]
            mu_q_seq = cache["mu_q"][q_seq]         # [B, L, 1]
            active_unique_q = torch.empty(0, device=device, dtype=torch.long)
            active_unique_tilde_mu = torch.empty(0, device=device, dtype=torch.float32)

        else:
            # Training / uncached mode: Execute Unique-Item Batch Indexing
            all_kc_emb = self.compute_all_kc_embeddings()

            # Extract unique questions in the batch and inverse indexing mapping back to [B, L]
            unique_q, inv_idx = torch.unique(q_seq, return_inverse=True)

            # Ground-truth modality masks for unique questions
            clean_masks = self.bank_modality_masks[unique_q]
            dropped_masks = clean_masks.clone()

            # Respect custom batch modality masks if provided during eval
            if (not self.training) and ("modality_masks" in batch) and (batch["modality_masks"] is not None):
                flat_masks = batch["modality_masks"].view(-1, 3).to(device)
                flat_inv = inv_idx.view(-1)
                # Scatter in reverse order so that earliest occurrence in sequence takes precedence (temporal causality)
                dropped_masks[flat_inv.flip(0)] = flat_masks.flip(0)
                clean_masks = dropped_masks.clone()

            # In training with Visual Modality Dropout (p_drop in [0.10, 0.15]):
            # Sample dropout directly on unique_q visual flags (Plan Step 3.3).
            # Text channels are 100% kept intact (Constraint 1).
            if self.training and self.p_drop > 0:
                stem_has_img = clean_masks[:, 0] > 0.5
                stem_drop = (torch.rand(len(unique_q), device=device) < self.p_drop) & stem_has_img
                dropped_masks[stem_drop, 0] = 0.0

                sol_has_img = clean_masks[:, 2] > 0.5
                sol_drop = (torch.rand(len(unique_q), device=device) < self.p_drop) & sol_has_img
                dropped_masks[sol_drop, 2] = 0.0
            else:
                stem_drop = torch.zeros(len(unique_q), device=device, dtype=torch.bool)
                sol_drop = torch.zeros(len(unique_q), device=device, dtype=torch.bool)

            item_out = self.forward_items(
                q_ids=unique_q,
                all_kc_emb=all_kc_emb,
                modality_masks=dropped_masks,
                chunk_size=256,
                return_cross_weights=False,
            )

            # Map representations directly back to sequence layout [B, L, ...] via inverse indices
            x_seq = item_out["x_q"][inv_idx]             # [B, L, d_model]
            c_C_seq = item_out["c_C"][inv_idx]           # [B, L, d_model]
            d_q_seq = item_out["d_q"][inv_idx]           # [B, L, d_model]
            tilde_mu_seq = item_out["tilde_mu"][inv_idx] # [B, L, 1]
            mu_q_seq = item_out["mu_q"][inv_idx]         # [B, L, 1]

            # In Stage 2, compute L_D^weighted on unique questions with CLEAN un-dropped masks (Plan Step 7.2)
            # Only include genuine active questions (exclude padding tokens where seq_mask == 0)
            valid_q_flat = q_seq[seq_mask.bool()].long()
            valid_unique_mask = torch.isin(unique_q, valid_q_flat)

            if self.training:
                # Only recompute clean forward for items that actually had images dropped (typically ~4%)
                dropped_idx = (stem_drop | sol_drop).nonzero(as_tuple=True)[0]
                full_unique_tilde_mu = item_out["tilde_mu"].squeeze(-1).clone()
                if len(dropped_idx) > 0:
                    dropped_q = unique_q[dropped_idx]
                    clean_patch = self.forward_items(
                        q_ids=dropped_q,
                        all_kc_emb=all_kc_emb,
                        modality_masks=clean_masks[dropped_idx],
                        chunk_size=256,
                        return_cross_weights=False,
                    )
                    full_unique_tilde_mu[dropped_idx] = clean_patch["tilde_mu"].squeeze(-1)
                active_unique_q = unique_q[valid_unique_mask]
                active_unique_tilde_mu = full_unique_tilde_mu[valid_unique_mask]
            else:
                active_unique_q = unique_q[valid_unique_mask]
                active_unique_tilde_mu = item_out["tilde_mu"].squeeze(-1)[valid_unique_mask]

        # -------------------------------------------------------------
        # 3. Subsystem 3: Dual-Stream Asymmetric Encoder & Retriever
        # -------------------------------------------------------------
        h_seq, tilde_x_seq, alpha_ret = self.sub3(
            x_seq=x_seq,
            c_C_seq=c_C_seq,
            d_q_seq=d_q_seq,
            mu_q_seq=mu_q_seq,
            r_seq=r_seq,
            seq_mask=seq_mask,
        )

        # -------------------------------------------------------------
        # 4. Subsystem 4.1: Monotonic 2PL-IRT Readout
        # -------------------------------------------------------------
        logits, probs, theta, a_disc = self.sub4(
            h_t=h_seq,
            tilde_x_t=tilde_x_seq,
            tilde_mu_t=tilde_mu_seq,
        )

        return {
            "logits": logits,
            "probs": probs,
            "theta": theta,
            "a_disc": a_disc,
            "tilde_mu_seq": tilde_mu_seq,
            "mu_q_seq": mu_q_seq,
            "alpha_ret": alpha_ret,
            "active_unique_q": active_unique_q,
            "active_unique_tilde_mu": active_unique_tilde_mu,
        }

    def extract_evidence(
        self,
        batch: Dict[str, torch.Tensor],
        sample_idx: int = 0,
        target_step: int = -1,
        top_k_history: int = 5,
    ) -> Dict[str, Any]:
        """
        Extract the 4-Tier Multi-Granular Evidence Package (E_t) for a specific interaction.

        Guarantees 100% mathematical consistency: (p_hat >= 0.5) <=> (theta >= tilde_mu).

        Args:
            batch: Single-item or mini-batch sequence dictionary.
            sample_idx: Index of sequence in batch (default: 0).
            target_step: Target interaction step t to explain (-1 = last valid step).
            top_k_history: Number of top historical evidence items to extract (default: 5).

        Returns:
            Hierarchical dictionary containing Tier 1, 2, 3, 4 evidence components.
        """
        self.eval()
        with torch.no_grad():
            q_seq = batch["q_seq"][sample_idx : sample_idx + 1]  # [1, L]
            r_seq = batch["r_seq"][sample_idx : sample_idx + 1]  # [1, L]
            seq_mask = batch["seq_mask"][sample_idx : sample_idx + 1]  # [1, L]
            L = q_seq.size(1)

            if target_step < 0 or target_step >= L:
                valid_indices = seq_mask[0].nonzero(as_tuple=True)[0]
                target_step = int(valid_indices[-1].item()) if len(valid_indices) > 0 else 0

            # Forward pass through model
            fwd_batch = {
                "q_seq": q_seq,
                "r_seq": r_seq,
                "seq_mask": seq_mask,
                "modality_masks": self.bank_modality_masks[q_seq],
            }
            out = self.forward(fwd_batch, use_cache=False)

            p_hat = float(out["probs"][0, target_step].item())
            theta_val = float(out["theta"][0, target_step].item())
            a_val = float(out["a_disc"][0, target_step].item())
            target_q_id = int(q_seq[0, target_step].item())

            # Forward detailed items for target question with Tier 3 fine-grained cross-attention maps
            all_kc_emb = self.compute_all_kc_embeddings()
            item_props = self.forward_items(
                q_ids=torch.tensor([target_q_id], device=q_seq.device),
                all_kc_emb=all_kc_emb,
                return_cross_weights=True,
            )

            mu_concept_val = float(item_props["mu_concept"][0, 0].item())
            delta_mu_stem_val = float(item_props["delta_mu_stem"][0, 0].item())
            delta_mu_sol_val = float(item_props["delta_mu_sol"][0, 0].item())
            tilde_mu_val = float(item_props["tilde_mu"][0, 0].item())
            mu_q_val = float(item_props["mu_q"][0, 0].item())

            # Tier 1: Psychometric Equilibrium
            delta_gap = theta_val - tilde_mu_val
            math_consistent = ((p_hat >= 0.5) == (delta_gap >= 0.0))
            tier1 = {
                "p_hat": p_hat,
                "theta": theta_val,
                "tilde_mu": tilde_mu_val,
                "mu_prob": mu_q_val,
                "a_disc": a_val,
                "delta_gap": delta_gap,
                "is_faithful_100": math_consistent,
            }

            # Tier 2: Difficulty Source Anatomy
            abs_sum = abs(mu_concept_val) + abs(delta_mu_stem_val) + abs(delta_mu_sol_val) + 1e-8
            tier2 = {
                "mu_concept": mu_concept_val,
                "delta_mu_stem": delta_mu_stem_val,
                "delta_mu_sol": delta_mu_sol_val,
                "pct_concept": (abs(mu_concept_val) / abs_sum) * 100.0,
                "pct_stem": (abs(delta_mu_stem_val) / abs_sum) * 100.0,
                "pct_sol": (abs(delta_mu_sol_val) / abs_sum) * 100.0,
            }

            # Tier 3: Multimodal Attribution
            lambda_q_scalar = float(item_props["lambda_q"][0].mean().item())
            lambda_s_scalar = float(item_props["lambda_s"][0].mean().item())
            beta_txt_q = item_props["beta_txt_q"][0].cpu().numpy()
            beta_txt_s = item_props["beta_txt_s"][0].cpu().numpy()
            A_cross_q = item_props["A_cross_q"][0].cpu().numpy()
            A_cross_s = item_props["A_cross_s"][0].cpu().numpy()

            tier3 = {
                "lambda_q_mean": lambda_q_scalar,
                "lambda_s_mean": lambda_s_scalar,
                "has_img_stem": bool(self.bank_modality_masks[target_q_id, 0] > 0.5),
                "has_img_sol": bool(self.bank_modality_masks[target_q_id, 2] > 0.5),
                "beta_txt_q": beta_txt_q,
                "beta_txt_s": beta_txt_s,
                "A_cross_q": A_cross_q,
                "A_cross_s": A_cross_s,
            }

            # Tier 4: Empirical History Tracing
            context_start = int(batch["context_start"][sample_idx].item()) if "context_start" in batch else 0
            global_target_step = context_start + target_step

            alpha_history = out["alpha_ret"][0, target_step].cpu().numpy()  # [L]
            valid_past_steps = [
                tau for tau in range(target_step)
                if bool(seq_mask[0, tau].item())
            ]

            past_attentions = [(tau, float(alpha_history[tau])) for tau in valid_past_steps]
            past_attentions.sort(key=lambda x: x[1], reverse=True)
            top_history = []
            for tau, attn_wt in past_attentions[:top_k_history]:
                past_q = int(q_seq[0, tau].item())
                past_r = int(r_seq[0, tau].item())
                past_mu_prob = float(out["mu_q_seq"][0, tau, 0].item())
                past_tilde_mu = float(out["tilde_mu_seq"][0, tau, 0].item())
                top_history.append({
                    "step_tau": tau,
                    "global_step": context_start + tau,
                    "q_id": past_q,
                    "response_r": past_r,
                    "mu_q": past_mu_prob,
                    "tilde_mu": past_tilde_mu,
                    "retrieval_weight": attn_wt,
                })

            tier4 = {
                "top_history_interactions": top_history,
                "total_past_valid_steps": len(valid_past_steps),
                "context_start": context_start,
                "global_target_step": global_target_step,
            }

            evidence = {
                "target_q_id": target_q_id,
                "target_step": target_step,
                "tier1_psychometrics": tier1,
                "tier2_difficulty": tier2,
                "tier3_multimodal": tier3,
                "tier4_history": tier4,
            }
            return evidence
