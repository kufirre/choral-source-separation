"""
Cross-source repulsion module for VCIN.

Applies reverse attention between latent sources to encourage diversity
and prevent multiple sources from collapsing to the same content.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RepulsionModule(nn.Module):
    """Cross-source reverse-attention repulsion transform."""

    def __init__(self, dim: int, num_latent_sources: int = 10):
        super().__init__()
        self.dim = dim
        self.num_latent_sources = num_latent_sources

        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.repulsion_strength = nn.Parameter(torch.tensor(0.0))
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
        if sources.ndim != 3:
            raise ValueError(f"RepulsionModule expects (B, M, D), got {tuple(sources.shape)}")
        if sources.shape[1] <= 1:
            return sources

        q = self.query_proj(sources)
        k = self.key_proj(sources)
        v = self.value_proj(sources)

        attn_logits = torch.bmm(q, k.transpose(1, 2)) * self.scale
        eye = torch.eye(attn_logits.shape[1], device=attn_logits.device, dtype=torch.bool).unsqueeze(0)
        attn_logits = attn_logits.masked_fill(eye, float("-inf"))
        attn = torch.softmax(attn_logits, dim=-1)

        repulsive_message = torch.bmm(attn, v)
        repulsive_message = self.out_proj(repulsive_message)
        lam = 0.5 * torch.sigmoid(self.repulsion_strength)
        return sources - lam * repulsive_message

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
        if sources.ndim != 3:
            raise ValueError(f"RepulsionModule expects (B, M, D), got {tuple(sources.shape)}")
        if sources.shape[1] <= 1:
            return sources.new_zeros(())

        proj = self.value_proj(sources.float())  # (B, M, D)
        proj_norm = F.normalize(proj, dim=-1)
        sim = torch.bmm(proj_norm, proj_norm.transpose(1, 2))  # (B, M, M)

        eye = torch.eye(sim.shape[1], dtype=torch.bool, device=sim.device).unsqueeze(0)
        off_diag_sim = sim.masked_select(~eye)
        sim_penalty = F.relu(off_diag_sim).pow(2).mean()

        q = self.query_proj(sources.float())
        k = self.key_proj(sources.float())
        attn_logits = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn_logits = attn_logits.masked_fill(eye, float("-inf"))
        attn = torch.softmax(attn_logits, dim=-1)
        attn_penalty = attn.masked_select(~eye).pow(2).mean()

        return sim_penalty + 0.1 * attn_penalty
