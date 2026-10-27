"""
Hypergraph Concept Convolution, Solution Distillation, and Rasch Disentanglement (Subsystem 2).

Implements Section 3.2 of docs/models/ACD-MMKT.md and Step 3 of
plans/acd_mmkt_implementation_and_training_plan.md:
- 2-Layer Residual Hypergraph Convolutional Network (HGNN) over KC correlation matrix G.
- Pooled core concept representation c_C via masked average pooling over leaf KCs.
- Latent solution complexity predictor (Student network f_pred_sol) distilling from Teacher sg(v_sol).
- Multimodal psychometric difficulty estimator mu_q in (0, 1) supervised by CTT labels D_q.
- Item-specific variation representation d_q orthogonal to concept representation c_C.
- Rasch question representation x = c_C + mu_q * d_q.
- Strict Causal Invariance: mu_q and d_q unconditionally use hat_v_sol (never true v_sol).
"""

from typing import Dict, Any, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HypergraphConvLayer(nn.Module):
    """
    Single-layer Hypergraph Convolution with residual connection and LayerNorm:
    H^(l+1) = LayerNorm(H^(l) + ELU(G * H^(l) * Theta))
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.theta = nn.Linear(d_in, d_out, bias=False)
        self.ln = nn.LayerNorm(d_out)
        self.act = nn.ELU()

    def forward(self, h: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: KC node representations [|K|, d_in].
            G: Precomputed normalized hypergraph adjacency matrix [|K|, |K|].

        Returns:
            Updated KC node representations [|K|, d_out].
        """
        # Graph message passing: G * H * Theta
        gh = torch.matmul(G, h)  # [|K|, d_in]
        out = self.theta(gh)     # [|K|, d_out]
        return self.ln(h + self.act(out))


class HypergraphRaschDisentanglement(nn.Module):
    """
    Subsystem 2: Hypergraph Concept Convolution, Latent Solution Distillation,
    and Rasch Psychometric Disentanglement.
    """

    def __init__(
        self,
        num_kcs: int = 1175,
        d_raw_kc: int = 768,
        d_model: int = 128,
        hgnn_layers: int = 2,
        dropout: float = 0.20,
        use_distillation: bool = True,
        use_hypergraph_rasch: bool = True,
    ):
        """
        Initialize HypergraphRaschDisentanglement module.

        Args:
            num_kcs: Number of knowledge components |K| (default: 1175).
            d_raw_kc: Initial semantic KC embedding dimension (default: 768).
            d_model: Latent dimension d (default: 128).
            hgnn_layers: Number of HGNN convolution layers (default: 2).
            dropout: Dropout probability.
            use_distillation: Whether to use latent solution complexity predictor (Ablation 2).
            use_hypergraph_rasch: Whether to use HGNN and Rasch decomposition (Ablation 4).
        """
        super().__init__()
        self.num_kcs = num_kcs
        self.d_raw_kc = d_raw_kc
        self.d_model = d_model
        self.hgnn_layers = hgnn_layers
        self.dropout_rate = dropout
        self.use_distillation = use_distillation
        self.use_hypergraph_rasch = use_hypergraph_rasch

        # 1. Hypergraph Concept Embedding Branch
        if self.use_hypergraph_rasch:
            self.kc_init_proj = nn.Linear(d_raw_kc, d_model)
            self.ln_kc_init = nn.LayerNorm(d_model)

            self.hgnn_layer1 = HypergraphConvLayer(d_model, d_model)
            self.hgnn_layer2 = HypergraphConvLayer(d_model, d_model)
        else:
            # Ablation 4: Standard learned embedding table without hypergraph propagation
            self.standard_kc_emb = nn.Embedding(num_kcs, d_model)

        # 2. Latent Solution Complexity Predictor (Student Network f_pred_sol)
        # Input: [v_stem; c_C] -> [2 * d_model]
        if self.use_distillation:
            self.pred_sol_net = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )

        # 3. Multimodal Psychometric Difficulty Estimator mu_q
        # Input: [v_stem; hat_v_sol; c_C] -> [3 * d_model] (or [2 * d_model] if w/o distillation)
        mu_in_dim = 3 * d_model if self.use_distillation else 2 * d_model
        self.fc_mu1 = nn.Linear(mu_in_dim, d_model)
        self.fc_mu2 = nn.Linear(d_model, 1)

        # 4. Item-Specific Variation MLP d_q
        # Input: [v_stem; hat_v_sol] -> [2 * d_model] (or [d_model] if w/o distillation)
        d_in_dim = 2 * d_model if self.use_distillation else d_model
        self.fc_d1 = nn.Linear(d_in_dim, d_model)
        self.fc_d2 = nn.Linear(d_model, d_model)
        self.ln_d = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def compute_kc_embeddings(
        self,
        kc_init_emb: Optional[torch.Tensor],
        hypergraph_G: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Execute 2-layer HGNN to compute high-order semantic representations of all 1,175 KCs.

        Args:
            kc_init_emb: Initial KC embeddings [|K|, 768].
            hypergraph_G: Precomputed normalized hypergraph matrix [|K|, |K|].

        Returns:
            Propagated KC embeddings E_K_final [|K|, d_model].
        """
        if not self.use_hypergraph_rasch:
            return self.standard_kc_emb.weight  # [|K|, d_model]

        assert kc_init_emb is not None and hypergraph_G is not None, (
            "kc_init_emb and hypergraph_G are required for HGNN"
        )
        h0 = self.ln_kc_init(self.kc_init_proj(kc_init_emb.float()))
        h1 = self.hgnn_layer1(h0, hypergraph_G)
        h2 = self.hgnn_layer2(h1, hypergraph_G)
        return h2  # [|K|, d_model]

    def pool_concepts(
        self,
        E_K: torch.Tensor,
        leaf_kcs: torch.Tensor,
        leaf_kc_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform masked average pooling across leaf KCs of questions.

        Args:
            E_K: All KC embeddings [|K|, d_model].
            leaf_kcs: Question leaf KC indices [..., max_kcs].
            leaf_kc_mask: Boolean mask indicating valid KCs [..., max_kcs].

        Returns:
            Core concept representation c_C [..., d_model].
        """
        kc_vectors = F.embedding(leaf_kcs, E_K)  # [..., max_kcs, d_model]
        mask_expanded = leaf_kc_mask.unsqueeze(-1).float()  # [..., max_kcs, 1]
        summed = (kc_vectors * mask_expanded).sum(dim=-2)  # [..., d_model]
        counts = torch.clamp(mask_expanded.sum(dim=-2), min=1.0)  # [..., 1]
        return summed / counts

    def predict_solution_complexity(
        self,
        v_stem: torch.Tensor,
        c_C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Student network: Predicts latent solution cognitive complexity from question stem and concepts.

        Args:
            v_stem: Question stem representation [..., d_model].
            c_C: Core concept representation [..., d_model].

        Returns:
            Predicted solution complexity representation hat_v_sol [..., d_model].
        """
        if not self.use_distillation:
            return torch.zeros_like(v_stem)

        inputs = torch.cat([v_stem, c_C], dim=-1)
        return self.pred_sol_net(inputs)

    def estimate_difficulty(
        self,
        v_stem: torch.Tensor,
        hat_v_sol: torch.Tensor,
        c_C: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimates multimodal scalar item difficulty mu_q in (0, 1).

        Returns:
            Tuple of:
            - mu_q: [..., 1] difficulty probability in (0, 1).
            - logit_mu: [..., 1] unnormalized logit for BCE loss calculation.
        """
        if self.use_distillation:
            inputs = torch.cat([v_stem, hat_v_sol, c_C], dim=-1)
        else:
            inputs = torch.cat([v_stem, c_C], dim=-1)

        hidden = F.gelu(self.fc_mu1(inputs))
        hidden = self.dropout(hidden)
        logit_mu = self.fc_mu2(hidden)
        mu_q = torch.sigmoid(logit_mu)
        return mu_q, logit_mu

    def estimate_difficulty_stem_only(
        self,
        v_stem: torch.Tensor,
        c_C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimates difficulty using only stem (hat_v_sol set to 0).
        Enables Level 2 explanation: Delta mu_sol = mu_q - mu_stem_only.
        """
        zero_sol = torch.zeros_like(v_stem)
        mu_q_stem, _ = self.estimate_difficulty(v_stem, zero_sol, c_C)
        return mu_q_stem

    def compute_item_variation(
        self,
        v_stem: torch.Tensor,
        hat_v_sol: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes item-specific variation vector d_q [..., d_model].
        """
        if self.use_distillation:
            inputs = torch.cat([v_stem, hat_v_sol], dim=-1)
        else:
            inputs = v_stem

        hidden = F.gelu(self.fc_d1(inputs))
        hidden = self.dropout(hidden)
        out = self.fc_d2(hidden)
        return self.ln_d(out)

    def forward(
        self,
        v_stem: torch.Tensor,
        leaf_kcs: torch.Tensor,
        leaf_kc_mask: torch.Tensor,
        kc_init_emb: Optional[torch.Tensor] = None,
        hypergraph_G: Optional[torch.Tensor] = None,
        precomputed_E_K: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Subsystem 2.

        Args:
            v_stem: Question stem representation [..., d_model].
            leaf_kcs: Leaf KC indices [..., max_kcs].
            leaf_kc_mask: Leaf KC validity mask [..., max_kcs].
            kc_init_emb: Initial KC embeddings [|K|, 768] (if precomputed_E_K is None).
            hypergraph_G: Normalized hypergraph matrix [|K|, |K|].
            precomputed_E_K: Optional precomputed propagated KC embeddings [|K|, d_model].

        Returns:
            Dictionary containing:
            - 'x': Rasch question representation [..., d_model].
            - 'c_C': Core concept representation [..., d_model].
            - 'hat_v_sol': Predicted solution complexity [..., d_model].
            - 'mu_q': Estimated scalar difficulty [..., 1].
            - 'logit_mu': Difficulty logit [..., 1].
            - 'd_q': Item-specific variation [..., d_model].
            - 'E_K': Propagated KC table [|K|, d_model].
        """
        # 1. Propagate or reuse KC embeddings
        if precomputed_E_K is not None:
            E_K = precomputed_E_K
        else:
            E_K = self.compute_kc_embeddings(kc_init_emb, hypergraph_G)

        # 2. Pool core concepts
        c_C = self.pool_concepts(E_K, leaf_kcs, leaf_kc_mask)  # [..., d_model]

        # 3. Student Network: Predict latent solution complexity
        # STRICT CAUSAL INVARIANCE: Always uses v_stem and c_C (never ground truth v_sol)
        hat_v_sol = self.predict_solution_complexity(v_stem, c_C)

        # 4. Multimodal difficulty estimation and item variation
        mu_q, logit_mu = self.estimate_difficulty(v_stem, hat_v_sol, c_C)
        d_q = self.compute_item_variation(v_stem, hat_v_sol)

        # 5. Rasch decomposition
        if self.use_hypergraph_rasch:
            # x = c_C + mu_q * d_q
            x = c_C + mu_q * d_q
        else:
            # Ablation 4: Standard additive combination without Rasch scaling
            x = c_C + v_stem

        return {
            "x": x,
            "c_C": c_C,
            "hat_v_sol": hat_v_sol,
            "mu_q": mu_q,
            "logit_mu": logit_mu,
            "d_q": d_q,
            "E_K": E_K,
        }
