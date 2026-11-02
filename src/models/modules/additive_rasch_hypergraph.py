"""
Subsystem 2: Additive Multi-Source Rasch Modeling & Hypergraph Concept Disentanglement.
Implements the exact mathematical specification for ACD-MMKTv2.

Key features:
1. Subsystem 2.1: Hypergraph Convolution over the 1,175 KCs using precomputed symmetric normalized
   transition matrix G and LeakyReLU activation (negative_slope=0.2) with residual LayerNorm connections.
   Produces intrinsic core concept representation c_C for each item.
2. Subsystem 2.2: Additive Multi-Source Difficulty Decomposition on real logit scale:
   tilde_mu_q = mu_concept + Delta_mu_stem + Delta_mu_sol, where b_c is initialized to ln(p_err / (1 - p_err)) = -1.29
   and stem/sol deviation heads have no external bias to guarantee zero-centered identifiable drift.
   mu_q = sigmoid(tilde_mu_q) in (0, 1).
3. Subsystem 2.3: Closed-form Gram-Schmidt Orthogonal Projection:
   LayerNorm is strictly placed BEFORE the projection on tilde_d^q.
   d^q = tilde_d^q - (<tilde_d^q, c_C> / (||c_C||_2^2 + eps)) * c_C, ensuring <d^q, c_C> = 0 analytically.
   Rasch representation: x_q = c_C + mu_q * d^q.
4. Support for Ablation 2 (use_solution=False) and Ablation 4 (use_hypergraph=False, use_gram_schmidt=False).
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class HypergraphConceptEncoder(nn.Module):
    """
    Subsystem 2.1: Hypergraph Neural Network for Knowledge Component Representations.
    Propagates structural prerequisites and co-occurrence patterns across all 1,175 KCs.
    """

    def __init__(
        self,
        num_kcs: int = 1175,
        d_raw_kc: int = 768,
        d_model: int = 128,
        hgnn_layers: int = 2,
        dropout: float = 0.15,
        use_hypergraph: bool = True,
    ):
        """
        Initialize HypergraphConceptEncoder.

        Args:
            num_kcs: Total number of knowledge components (1175).
            d_raw_kc: Dimension of raw KC semantic embeddings (768).
            d_model: Target latent embedding dimension (128).
            hgnn_layers: Number of hypergraph convolution layers (2).
            dropout: Dropout probability.
            use_hypergraph: Whether to use HGNN propagation (Ablation 4 flag).
        """
        super().__init__()
        self.num_kcs = num_kcs
        self.d_model = d_model
        self.hgnn_layers = hgnn_layers
        self.use_hypergraph = use_hypergraph

        if self.use_hypergraph:
            # Projection of raw KC semantic vectors to latent space
            self.kc_proj = nn.Linear(d_raw_kc, d_model)
            self.norm_kc_init = nn.LayerNorm(d_model)

            # HGNN convolution weight matrices and LayerNorm layers
            self.hgnn_weights = nn.ParameterList([
                nn.Parameter(torch.empty(d_model, d_model)) for _ in range(hgnn_layers)
            ])
            self.hgnn_norms = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(hgnn_layers)
            ])
            self.dropout = nn.Dropout(dropout)

            for w in self.hgnn_weights:
                nn.init.xavier_uniform_(w)
        else:
            # Ablation 4: Standard discrete KC embedding lookup without hypergraph structure
            self.kc_embedding = nn.Embedding(num_kcs, d_model)
            nn.init.normal_(self.kc_embedding.weight, std=0.02)
            self.norm_kc = nn.LayerNorm(d_model)

    def compute_all_kc_embeddings(
        self,
        raw_kc_emb: torch.Tensor,
        hypergraph_G: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute enriched KC representations for all 1,175 KCs once per batch update.

        Args:
            raw_kc_emb: Raw semantic KC embeddings [1175, 768].
            hypergraph_G: Precomputed symmetric normalized transition matrix [1175, 1175].

        Returns:
            Enriched KC embedding matrix [1175, d_model].
        """
        if not self.use_hypergraph:
            return self.norm_kc(self.kc_embedding.weight)

        # Initial projection + LayerNorm
        E_K = self.norm_kc_init(self.kc_proj(raw_kc_emb.float()))  # [1175, d_model]

        # Multi-layer hypergraph propagation with LeakyReLU (negative slope = 0.2)
        G = hypergraph_G.float()  # [1175, 1175]
        for l in range(self.hgnn_layers):
            W = self.hgnn_weights[l]
            # Matrix multiplication: G @ E_K @ W
            prop = torch.matmul(G, E_K)
            prop = torch.matmul(prop, W)
            prop = F.leaky_relu(prop, negative_slope=0.2)
            prop = self.dropout(prop)
            E_K = self.hgnn_norms[l](E_K + prop)

        return E_K  # [1175, d_model]

    def forward(
        self,
        all_kc_emb: torch.Tensor,
        leaf_kcs: torch.Tensor,
        leaf_kc_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate core concept representation c_C for given items.

        Args:
            all_kc_emb: Enriched KC embedding table [1175, d_model].
            leaf_kcs: Leaf KC indices for questions [N, 6].
            leaf_kc_mask: Boolean validity mask [N, 6].

        Returns:
            Core concept representation c_C [N, d_model].
        """
        # Lookup leaf KC embeddings: [N, 6, d_model]
        kc_vecs = F.embedding(leaf_kcs, all_kc_emb)
        mask_float = leaf_kc_mask.unsqueeze(-1).float()  # [N, 6, 1]

        # Masked mean pooling across valid leaf KCs
        sum_vecs = (kc_vecs * mask_float).sum(dim=1)  # [N, d_model]
        sum_mask = mask_float.sum(dim=1).clamp(min=1e-8)  # [N, 1]
        c_C = sum_vecs / sum_mask

        return c_C


class AdditiveRaschHypergraph(nn.Module):
    """
    Subsystem 2 of ACD-MMKTv2: Additive Multi-Source Rasch Modeling & Gram-Schmidt Orthogonalization.
    Combines:
    - 2.1: Hypergraph Concept Embedding (c_C).
    - 2.2: Additive Multi-Source Difficulty Decomposition (tilde_mu_q = mu_concept + Delta_mu_stem + Delta_mu_sol).
    - 2.3: Closed-form Gram-Schmidt Orthogonal Projection (d^q perp c_C) and Rasch representation x_q = c_C + mu_q * d^q.
    """

    def __init__(
        self,
        num_kcs: int = 1175,
        d_raw_kc: int = 768,
        d_model: int = 128,
        hgnn_layers: int = 2,
        dropout: float = 0.15,
        use_solution: bool = True,
        use_hypergraph: bool = True,
        use_gram_schmidt: bool = True,
        initial_bias_c: float = -1.29,
    ):
        """
        Initialize AdditiveRaschHypergraph.

        Args:
            num_kcs: Total number of knowledge components (1175).
            d_raw_kc: Raw KC dimension (768).
            d_model: Latent dimension (128).
            hgnn_layers: HGNN layer count (2).
            dropout: Dropout rate.
            use_solution: Whether to use solution cognitive blueprint (Ablation 2 flag).
            use_hypergraph: Whether to use HGNN (Ablation 4 flag).
            use_gram_schmidt: Whether to use Gram-Schmidt projection (Ablation 4 flag).
            initial_bias_c: Initial psychometric bias for b_c, corresponding to ln(p_err / (1 - p_err)) = -1.29.
        """
        super().__init__()
        self.d_model = d_model
        self.use_solution = use_solution
        self.use_hypergraph = use_hypergraph
        self.use_gram_schmidt = use_gram_schmidt

        # 2.1 Hypergraph Concept Encoder
        self.concept_encoder = HypergraphConceptEncoder(
            num_kcs=num_kcs,
            d_raw_kc=d_raw_kc,
            d_model=d_model,
            hgnn_layers=hgnn_layers,
            dropout=dropout,
            use_hypergraph=use_hypergraph,
        )

        # 2.2 Additive Multi-Source Difficulty Decomposition
        # Head 1: Concept baseline difficulty mu_concept = w_c^T c_C + b_c
        self.fc_mu_concept = nn.Linear(d_model, 1, bias=True)
        # Initialize b_c to psychometric logit prior (~ -1.29)
        nn.init.constant_(self.fc_mu_concept.bias, initial_bias_c)
        nn.init.normal_(self.fc_mu_concept.weight, std=0.01)

        # Head 2: Stem difficulty deviation Delta_mu_stem = w_stem^T GELU(W_stem v_stem + b_stem)
        # Crucial: NO bias on outer linear layer to guarantee zero-centered deviation!
        self.fc_stem_hidden = nn.Linear(d_model, d_model, bias=True)
        self.fc_stem_out = nn.Linear(d_model, 1, bias=False)
        nn.init.normal_(self.fc_stem_out.weight, std=0.01)

        # Head 3: Solution algorithmic difficulty deviation Delta_mu_sol = w_sol^T GELU(W_sol v_sol + b_sol)
        # Crucial: NO bias on outer linear layer to guarantee zero-centered deviation!
        if self.use_solution:
            self.fc_sol_hidden = nn.Linear(d_model, d_model, bias=True)
            self.fc_sol_out = nn.Linear(d_model, 1, bias=False)
            nn.init.normal_(self.fc_sol_out.weight, std=0.01)

        # 2.3 Gram-Schmidt Orthogonalization MLP
        var_in_dim = 2 * d_model if self.use_solution else d_model
        self.mlp_variation = nn.Sequential(
            nn.Linear(var_in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            # CRITICAL: LayerNorm is placed on tilde_d^q BEFORE Gram-Schmidt projection!
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        all_kc_emb: torch.Tensor,
        leaf_kcs: torch.Tensor,
        leaf_kc_mask: torch.Tensor,
        v_stem: torch.Tensor,
        v_sol: torch.Tensor,
        c_C: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Subsystem 2.

        Args:
            all_kc_emb: Enriched KC embeddings [1175, d_model].
            leaf_kcs: Leaf KC indices [N, 6].
            leaf_kc_mask: Boolean leaf KC mask [N, 6].
            v_stem: Multimodal stem representation [N, d_model].
            v_sol: Multimodal solution representation [N, d_model].
            c_C: Optional precomputed core concept representation [N, d_model] to eliminate duplicate calculation.

        Returns:
            Dictionary containing:
            - 'c_C': Core concept representation [N, d_model].
            - 'mu_concept': Baseline concept difficulty [N, 1].
            - 'delta_mu_stem': Stem deviation difficulty [N, 1].
            - 'delta_mu_sol': Solution deviation difficulty [N, 1].
            - 'tilde_mu': Total logit difficulty [N, 1].
            - 'mu_q': Probability scale difficulty [N, 1] in (0, 1).
            - 'd_q': Item-specific variation vector [N, d_model] orthogonal to c_C.
            - 'x_q': Rasch item representation [N, d_model].
        """
        N = v_stem.size(0)

        # 1. Concept Representation c_C (reuse if provided, avoiding duplicate forward pass)
        if c_C is None:
            c_C = self.concept_encoder(
                all_kc_emb=all_kc_emb,
                leaf_kcs=leaf_kcs,
                leaf_kc_mask=leaf_kc_mask,
            )  # [N, d_model]

        # 2. Additive Multi-Source Difficulty Decomposition (maintain in fp32 for numerical stability)
        mu_concept = self.fc_mu_concept(c_C).float()  # [N, 1]

        stem_h = F.gelu(self.fc_stem_hidden(v_stem))
        delta_mu_stem = self.fc_stem_out(stem_h).float()  # [N, 1]

        if self.use_solution:
            sol_h = F.gelu(self.fc_sol_hidden(v_sol))
            delta_mu_sol = self.fc_sol_out(sol_h).float()  # [N, 1]
        else:
            delta_mu_sol = torch.zeros_like(mu_concept)

        tilde_mu = mu_concept + delta_mu_stem + delta_mu_sol  # [N, 1] logit scale (fp32)
        mu_q = torch.sigmoid(tilde_mu)  # [N, 1] probability scale (fp32)

        # 3. Variation Vector and Analytical Gram-Schmidt Orthogonalization
        if self.use_solution:
            var_input = torch.cat([v_stem, v_sol], dim=-1)  # [N, 2 * d_model]
        else:
            var_input = v_stem  # [N, d_model]

        tilde_d_q = self.mlp_variation(var_input)  # [N, d_model] with LayerNorm applied

        if self.use_gram_schmidt:
            # Analytical Gram-Schmidt projection in fp32 for numerical stability
            tilde_d_f32 = tilde_d_q.float()
            c_C_f32 = c_C.float()

            dot_product = (tilde_d_f32 * c_C_f32).sum(dim=-1, keepdim=True)  # [N, 1]
            norm_sq = (c_C_f32 * c_C_f32).sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [N, 1]
            proj = (dot_product / norm_sq) * c_C_f32  # [N, d_model]

            d_q = tilde_d_f32 - proj  # [N, d_model] perp c_C (fp32)
        else:
            # Ablation 4: Skip orthogonal projection
            d_q = tilde_d_q.float()

        # Rasch Model Item Representation: x_q = c_C + mu_q * d_q
        x_q = c_C + mu_q.to(dtype=c_C.dtype) * d_q.to(dtype=c_C.dtype)  # [N, d_model]

        return {
            "c_C": c_C,
            "mu_concept": mu_concept,
            "delta_mu_stem": delta_mu_stem,
            "delta_mu_sol": delta_mu_sol,
            "tilde_mu": tilde_mu,
            "mu_q": mu_q,
            "d_q": d_q,
            "x_q": x_q,
        }

