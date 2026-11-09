"""
Subsystem 2: Hypergraph Concept Representation, Identifiable Difficulty Decomposition & Gram-Schmidt Disentanglement.
Implements the exact mathematical specification for ACD-MMKTv3 (Section 3, Subsystem 2).

Key features:
1. Subsystem 2.1: Hypergraph Convolution over the 1,175 KCs using precomputed symmetric normalized
   transition matrix G and LeakyReLU activation (negative_slope=0.2) with residual LayerNorm connections.
   Produces intrinsic core concept representation c_C for each item.
2. Subsystem 2.2: Identifiable Multi-Source Difficulty Decomposition with Signed Visual Load:
   - mu_concept = logit(mu_q^(prior, f)) anchored to Empirical Bayes skill prior (no free bias/parameters).
   - Delta_mu_txt^q = w_txt^T GELU(W_txt u_txt^q) (no bias in w_txt).
   - Delta_mu_vis^q = gamma_vis * tanh(w_vis^T GELU(W_vis u_cross^q)) in [-gamma_vis, gamma_vis] (no bias in w_vis).
     Signed interpretation: < 0 represents visual scaffolding/aid, > 0 represents visual cognitive overload.
   - Delta_mu_stem = Delta_mu_txt^q + m_img^q * bar_lambda_q * Delta_mu_vis^q.
   - Delta_mu_sol = w_sol^T GELU(W_sol v_sol) (no bias in w_sol).
   - Total difficulty logit: tilde_mu = mu_concept + Delta_mu_stem + Delta_mu_sol.
   - Difficulty probability: mu_q = sigmoid(tilde_mu) in (0, 1).
3. Subsystem 2.3: Closed-form Gram-Schmidt Orthogonal Projection:
   - LayerNorm is placed strictly on tilde_d^q BEFORE Gram-Schmidt.
   - d^q = tilde_d^q - (<tilde_d^q, c_C> / (||c_C||_2^2 + eps)) * c_C, ensuring <d^q, c_C> = 0 analytically.
   - Rasch representation: x_q = c_C + mu_q * d^q.
4. Support for Modality Ablations:
   - use_multimodal=False (Ablation 1): Pure KC interaction, KC init via one-hot/index embedding, no multimodal difficulty heads, d^q = 0, x_q = c_C.
   - use_solution=False (Ablation 2): Delta_mu_sol = 0, no v_sol in d^q (mlp_d takes d_model).
   - use_visual=False (Ablation 3): Delta_mu_vis = 0, no visual projection heads instantiated.
   - unanchored_concept=True (Ablation 4): replaces fixed logit(mu_prior) with learned projection from c_C.
   - use_hypergraph=False: standard KC embedding without HGNN.
   - use_gram_schmidt=False: d^q = tilde_d^q without orthogonalization.
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
        use_multimodal: bool = True,
    ):
        super().__init__()
        self.num_kcs = num_kcs
        self.d_model = d_model
        self.hgnn_layers = hgnn_layers
        self.use_hypergraph = use_hypergraph
        self.use_multimodal = use_multimodal

        if not self.use_multimodal:
            # Ablation 1: pure KC structural identification (one-hot indicator projection without text semantics)
            self.kc_proj = None
            self.kc_embedding = nn.Embedding(num_kcs, d_model)
            nn.init.normal_(self.kc_embedding.weight, std=0.02)
            self.norm_kc_init = nn.LayerNorm(d_model)

            if self.use_hypergraph:
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
                self.hgnn_weights = None
                self.hgnn_norms = None
                self.dropout = nn.Dropout(dropout)
        elif self.use_hypergraph:
            self.kc_embedding = None
            self.kc_proj = nn.Linear(d_raw_kc, d_model)
            self.norm_kc_init = nn.LayerNorm(d_model)

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
            self.kc_proj = None
            self.kc_embedding = nn.Embedding(num_kcs, d_model)
            nn.init.normal_(self.kc_embedding.weight, std=0.02)
            self.norm_kc = nn.LayerNorm(d_model)

    def compute_all_kc_embeddings(
        self,
        raw_kc_emb: Optional[torch.Tensor],
        hypergraph_G: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute enriched KC representations for all 1,175 KCs once per batch update.
        """
        if not self.use_multimodal:
            # Ablation 1: strictly project from discrete KC indices without text token embeddings
            all_kc_indices = torch.arange(self.num_kcs, device=hypergraph_G.device)
            E_K = self.norm_kc_init(self.kc_embedding(all_kc_indices))
            if not self.use_hypergraph:
                return E_K
        elif not self.use_hypergraph:
            return self.norm_kc(self.kc_embedding.weight)
        else:
            E_K = self.norm_kc_init(self.kc_proj(raw_kc_emb.float()))  # [1175, d_model]

        G = hypergraph_G.float()  # [1175, 1175]

        for l in range(self.hgnn_layers):
            W = self.hgnn_weights[l]
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
        """
        kc_vecs = F.embedding(leaf_kcs, all_kc_emb)  # [N, 6, d_model]
        mask_float = leaf_kc_mask.unsqueeze(-1).float()  # [N, 6, 1]

        sum_vecs = (kc_vecs * mask_float).sum(dim=1)  # [N, d_model]
        sum_mask = mask_float.sum(dim=1).clamp(min=1e-8)  # [N, 1]
        c_C = sum_vecs / sum_mask
        return c_C


class IdentifiableRaschHypergraph(nn.Module):
    """
    Subsystem 2 of ACD-MMKTv3: Identifiable Difficulty Decomposition & Gram-Schmidt Disentanglement.
    """

    def __init__(
        self,
        num_kcs: int = 1175,
        d_raw_kc: int = 768,
        d_model: int = 128,
        hgnn_layers: int = 2,
        gamma_vis: float = 1.5,
        dropout: float = 0.15,
        use_solution: bool = True,
        use_visual: bool = True,
        use_hypergraph: bool = True,
        use_gram_schmidt: bool = True,
        unanchored_concept: bool = False,
        use_multimodal: bool = True,
    ):
        """
        Initialize IdentifiableRaschHypergraph.

        Args:
            num_kcs: Total number of knowledge components (1175).
            d_raw_kc: Raw KC semantic embedding dimension (768).
            d_model: Latent embedding dimension (128).
            hgnn_layers: Number of HGNN convolution layers (2).
            gamma_vis: Maximum visual cognitive load logit amplitude (1.5).
            dropout: Dropout probability.
            use_solution: Whether to use solution blueprint (Ablation 2).
            use_visual: Whether to use visual channel (Ablation 3).
            use_hypergraph: Whether to use HGNN propagation (Ablation 4).
            use_gram_schmidt: Whether to use Gram-Schmidt projection (Ablation 4).
            unanchored_concept: Whether to learn unanchored concept difficulty (Ablation 4).
            use_multimodal: Master switch for multimodal stream (Ablation 1).
        """
        super().__init__()
        self.d_model = d_model
        self.gamma_vis = gamma_vis
        self.use_solution = use_solution
        self.use_visual = use_visual
        self.use_hypergraph = use_hypergraph
        self.use_gram_schmidt = use_gram_schmidt
        self.unanchored_concept = unanchored_concept
        self.use_multimodal = use_multimodal

        # 2.1 Hypergraph Concept Encoder
        self.concept_encoder = HypergraphConceptEncoder(
            num_kcs=num_kcs,
            d_raw_kc=d_raw_kc,
            d_model=d_model,
            hgnn_layers=hgnn_layers,
            dropout=dropout,
            use_hypergraph=use_hypergraph,
            use_multimodal=use_multimodal,
        )

        # 2.2 Identifiable Multi-Source Difficulty Decomposition
        # If unanchored_concept (Ablation 4): learned head with bias
        if self.unanchored_concept:
            self.fc_unanchored_concept = nn.Linear(d_model, 1, bias=True)
            nn.init.constant_(self.fc_unanchored_concept.bias, -1.37)
            nn.init.normal_(self.fc_unanchored_concept.weight, std=0.01)
        else:
            self.fc_unanchored_concept = None

        if not self.use_multimodal:
            # Ablation 1: pure KC interaction, completely eliminate difficulty deviation heads and MLP_d
            self.fc_txt_hidden = None
            self.fc_txt_out = None
            self.fc_vis_hidden = None
            self.fc_vis_out = None
            self.fc_sol_hidden = None
            self.fc_sol_out = None
            self.mlp_d = None
            self.norm_d_raw = None
            return

        # Stem text difficulty head: Delta_mu_txt = w_txt^T GELU(W_txt u_txt)
        # Strictly NO BIAS on either layer to eliminate drift and ensure zero input yields zero output
        self.fc_txt_hidden = nn.Linear(d_model, d_model, bias=False)
        self.fc_txt_out = nn.Linear(d_model, 1, bias=False)
        nn.init.normal_(self.fc_txt_out.weight, std=0.01)

        # Stem visual difficulty head: Delta_mu_vis = gamma_vis * tanh(w_vis^T GELU(W_vis u_cross))
        if self.use_visual:
            self.fc_vis_hidden = nn.Linear(d_model, d_model, bias=False)
            self.fc_vis_out = nn.Linear(d_model, 1, bias=False)
            nn.init.normal_(self.fc_vis_out.weight, std=0.01)
        else:
            self.fc_vis_hidden = None
            self.fc_vis_out = None

        # Solution algorithmic difficulty head: Delta_mu_sol = w_sol^T GELU(W_sol v_sol)
        if self.use_solution:
            self.fc_sol_hidden = nn.Linear(d_model, d_model, bias=False)
            self.fc_sol_out = nn.Linear(d_model, 1, bias=False)
            nn.init.normal_(self.fc_sol_out.weight, std=0.01)
        else:
            self.fc_sol_hidden = None
            self.fc_sol_out = None

        # 2.3 Gram-Schmidt Orthogonal Variation Projector MLP_d
        in_dim = 2 * d_model if self.use_solution else d_model
        self.mlp_d = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        # LayerNorm strictly BEFORE Gram-Schmidt orthogonal projection
        self.norm_d_raw = nn.LayerNorm(d_model)

    def forward(
        self,
        v_stem: torch.Tensor,
        v_sol: torch.Tensor,
        u_txt_q: torch.Tensor,
        u_cross_q: torch.Tensor,
        bar_lambda_q: torch.Tensor,
        m_img_q: torch.Tensor,
        c_C: torch.Tensor,
        mu_concept_prior: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Subsystem 2.

        Args:
            v_stem: Multimodal stem representation [M, d_model].
            v_sol: Solution representation [M, d_model].
            u_txt_q: Pooled stem text representation [M, d_model].
            u_cross_q: Cross-attended visual representation [M, d_model].
            bar_lambda_q: Average cognitive visual gate [M, 1].
            m_img_q: Question stem image flag [M, 1].
            c_C: Core concept representation [M, d_model].
            mu_concept_prior: Precomputed fold empirical Bayes logit anchor [M] or [M, 1].

        Returns:
            Dictionary containing mu_concept, delta_mu_txt, delta_mu_vis, delta_mu_stem,
            delta_mu_sol, tilde_mu, mu_q, d_q, x_q.
        """
        M = c_C.size(0)
        device = c_C.device

        # -------------------------------------------------------------
        # 2.2 Difficulty Decomposition
        # -------------------------------------------------------------
        # Concept baseline difficulty
        if self.unanchored_concept and self.fc_unanchored_concept is not None:
            mu_concept = self.fc_unanchored_concept(c_C).squeeze(-1)  # [M]
        else:
            # Anchored to Empirical Bayes prior: mu_concept = logit(mu_prior) in R
            mu_concept = mu_concept_prior.view(-1).float()  # [M]

        # If multimodal stream is entirely disabled (Ablation 1)
        if not self.use_multimodal:
            zero_delta = torch.zeros(M, device=device, dtype=torch.float32)
            tilde_mu = mu_concept.float()
            mu_q = torch.sigmoid(tilde_mu)
            d_q = torch.zeros((M, self.d_model), device=device, dtype=torch.float32)
            x_q = c_C.float()
            return {
                "mu_concept": mu_concept,
                "delta_mu_txt": zero_delta,
                "delta_mu_vis": zero_delta,
                "delta_mu_stem": zero_delta,
                "delta_mu_sol": zero_delta,
                "tilde_mu": tilde_mu,
                "mu_q": mu_q,
                "d_q": d_q,
                "x_q": x_q,
            }

        # Stem text load: Delta_mu_txt^q
        h_txt = F.gelu(self.fc_txt_hidden(u_txt_q.float()))
        delta_mu_txt = self.fc_txt_out(h_txt).squeeze(-1).float()  # [M] fp32

        # Stem visual load: Delta_mu_vis^q in [-gamma_vis, gamma_vis]
        # Strictly zero if m_img_q == 0 to prevent ghost attributions on text-only items
        if self.use_visual and self.fc_vis_hidden is not None:
            h_vis = F.gelu(self.fc_vis_hidden(u_cross_q.float()))
            raw_vis = self.fc_vis_out(h_vis).squeeze(-1).float()  # [M] fp32
            delta_mu_vis = self.gamma_vis * torch.tanh(raw_vis) * m_img_q.view(-1).float()  # [M]
        else:
            delta_mu_vis = torch.zeros_like(delta_mu_txt)

        # Combined stem load: Delta_mu_stem = Delta_mu_txt + m_img_q * bar_lambda_q * Delta_mu_vis
        stem_vis_term = m_img_q.view(-1).float() * bar_lambda_q.view(-1).float() * delta_mu_vis
        delta_mu_stem = delta_mu_txt + stem_vis_term  # [M] fp32

        # Solution algorithmic load: Delta_mu_sol
        if self.use_solution and self.fc_sol_hidden is not None:
            h_sol = F.gelu(self.fc_sol_hidden(v_sol.float()))
            delta_mu_sol = self.fc_sol_out(h_sol).squeeze(-1).float()  # [M] fp32
        else:
            delta_mu_sol = torch.zeros_like(delta_mu_stem)

        # Total psychometric difficulty on real logit scale
        tilde_mu = (mu_concept + delta_mu_stem + delta_mu_sol).float()  # [M] fp32
        mu_q = torch.sigmoid(tilde_mu)                                  # [M] in (0, 1) fp32

        # -------------------------------------------------------------
        # 2.3 Gram-Schmidt Orthogonalization
        # -------------------------------------------------------------
        if self.use_solution:
            mlp_in = torch.cat([v_stem, v_sol], dim=-1)
        else:
            mlp_in = v_stem

        # LayerNorm strictly placed BEFORE Gram-Schmidt projection
        tilde_d = self.norm_d_raw(self.mlp_d(mlp_in.float()))  # [M, d_model]

        if self.use_gram_schmidt:
            # Enforce fp32 for dot product precision and numerical stability
            c_fp32 = c_C.float()
            d_fp32 = tilde_d.float()

            dot_prod = (d_fp32 * c_fp32).sum(dim=-1, keepdim=True)        # [M, 1]
            c_norm_sq = (c_fp32 * c_fp32).sum(dim=-1, keepdim=True) + 1e-8  # [M, 1]
            projection = (dot_prod / c_norm_sq) * c_fp32                  # [M, d_model]
            d_q = d_fp32 - projection                                     # [M, d_model] strictly fp32
        else:
            d_q = tilde_d.float()

        # Rasch Item Representation: x_q = c_C + mu_q * d^q
        x_q = c_C.float() + mu_q.unsqueeze(-1) * d_q  # [M, d_model] fp32

        return {
            "mu_concept": mu_concept,
            "delta_mu_txt": delta_mu_txt,
            "delta_mu_vis": delta_mu_vis,
            "delta_mu_stem": delta_mu_stem,
            "delta_mu_sol": delta_mu_sol,
            "tilde_mu": tilde_mu,
            "mu_q": mu_q,
            "d_q": d_q,
            "x_q": x_q,
        }

