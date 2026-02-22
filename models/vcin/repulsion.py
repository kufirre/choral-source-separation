"""
Cross-source repulsion module for VCIN.

Applies reverse attention between latent sources to encourage diversity
and prevent multiple sources from collapsing to the same content.

Placeholder: identity (returns input unchanged).
"""

import torch
import torch.nn as nn


class RepulsionModule(nn.Module):
    """
    Cross-source reverse attention (placeholder).

    Full implementation: computes pairwise similarity between latent source
    representations and applies a repulsive loss/transformation that pushes
    similar sources apart in representation space.
    Placeholder: identity function.
    """

    def __init__(self, dim: int, num_latent_sources: int = 10):
        super().__init__()
        self.dim = dim
        self.num_latent_sources = num_latent_sources

        # Projection for computing pairwise similarity
        self.proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(
        self,
        sources: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sources: (batch, M, dim) latent source representations

        Returns:
            Repulsion-adjusted sources of same shape.
        """
        # Placeholder: identity
        return sources

    def repulsion_loss(
        self,
        sources: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute repulsion penalty between source representations.

        Args:
            sources: (batch, M, dim)

        Returns:
            Scalar repulsion loss (higher = more similar sources).
        """
        proj = self.proj(sources)  # (B, M, D)
        # Cosine similarity matrix
        proj_norm = torch.nn.functional.normalize(proj, dim=-1)
        sim = torch.bmm(proj_norm, proj_norm.transpose(1, 2))  # (B, M, M)

        # Mask diagonal
        mask = ~torch.eye(
            sim.shape[1], dtype=torch.bool, device=sim.device
        ).unsqueeze(0)
        off_diag = sim.masked_select(mask)

        return off_diag.abs().mean()
