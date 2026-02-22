"""
Soft assignment module for VCIN.

Computes a row-stochastic assignment matrix A of shape (M, V) that maps
M latent over-separated sources to V target voice parts (SATB).

Each entry A[m, v] represents the probability that latent source m belongs
to voice part v. The assignment is informed by pitch embeddings and
source embeddings.

Placeholder: returns uniform assignment (each latent source equally
assigned to all voice parts).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftAssignment(nn.Module):
    """
    Soft assignment from M latent sources to V voice parts (placeholder).

    Full implementation: bilinear attention between source embeddings and
    pitch-conditioned voice queries, followed by row-wise softmax.
    Placeholder: uniform 1/V assignment.
    """

    def __init__(
        self,
        num_latent_sources: int = 10,
        num_voices: int = 4,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.num_latent_sources = num_latent_sources
        self.num_voices = num_voices
        self.embed_dim = embed_dim

        # Voice query embeddings (learnable)
        self.voice_queries = nn.Parameter(
            torch.randn(num_voices, embed_dim) * 0.02
        )
        # Source key projection
        self.source_key_proj = nn.Linear(embed_dim, embed_dim)
        # Temperature for softmax
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        source_embeddings: torch.Tensor | None = None,
        pitch_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            source_embeddings: (batch, M, embed_dim) or None
            pitch_embeddings: (batch, V, embed_dim) or None

        Returns:
            assignment: (batch, M, V) row-stochastic assignment matrix
        """
        if source_embeddings is None:
            # Uniform fallback
            batch_size = pitch_embeddings.shape[0] if pitch_embeddings is not None else 1
            uniform = torch.ones(
                batch_size, self.num_latent_sources, self.num_voices,
                device=self.voice_queries.device,
            ) / self.num_voices
            return uniform

        batch_size = source_embeddings.shape[0]

        # Project source embeddings to keys
        keys = self.source_key_proj(source_embeddings)  # (B, M, D)

        # Use voice queries as Q and optionally condition them by pitch priors.
        queries = self.voice_queries.unsqueeze(0).expand(batch_size, -1, -1)  # (B, V, D)
        if pitch_embeddings is not None:
            queries = queries + pitch_embeddings

        # Attention: (B, M, D) @ (B, D, V) -> (B, M, V)
        logits = torch.bmm(keys, queries.transpose(1, 2))
        logits = logits / (self.embed_dim ** 0.5 * self.temperature.clamp(min=0.1))

        # Row-stochastic: softmax over voices for each source
        assignment = F.softmax(logits, dim=-1)  # (B, M, V)

        return assignment
