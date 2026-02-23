"""
Pitch estimator module for VCIN.

Estimates a Mixture-of-Gaussians (MoG) pitch posterior for each SATB voice part.
The pitch embedding is used downstream by the assignment and FiLM modules to
guide source grouping and conditioning.

Placeholder: returns a learned constant pitch embedding per voice part.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PitchEstimator(nn.Module):
    """
    Per-voice pitch estimator (placeholder).

    In the full implementation, this produces a MoG posterior over fundamental
    frequency for each voice part at each time frame. The placeholder returns
    a fixed learned embedding.
    """

    def __init__(self, num_voices: int = 4, embed_dim: int = 64):
        super().__init__()
        self.num_voices = num_voices
        self.embed_dim = embed_dim

        # Placeholder: learned constant per voice
        self.pitch_embed = nn.Parameter(
            torch.randn(num_voices, embed_dim) * 0.02
        )
        self.hidden_proj = nn.LazyLinear(embed_dim)
        self.readout_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        mixture: torch.Tensor | None = None,
        shared_hidden: torch.Tensor | None = None,
        voice_readouts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            mixture: (batch, channels, time) raw mixture waveform

        Returns:
            pitch_embedding: (batch, num_voices, embed_dim)
        """
        if mixture is not None:
            batch_size = mixture.shape[0]
            device = mixture.device
            dtype = mixture.dtype
        elif voice_readouts is not None:
            batch_size = voice_readouts.shape[0]
            device = voice_readouts.device
            dtype = voice_readouts.dtype
        elif shared_hidden is not None:
            batch_size = shared_hidden.shape[0]
            device = shared_hidden.device
            dtype = shared_hidden.dtype
        else:
            batch_size = 1
            device = self.pitch_embed.device
            dtype = self.pitch_embed.dtype

        base = self.pitch_embed.unsqueeze(0).expand(batch_size, -1, -1).to(device=device, dtype=dtype)

        if voice_readouts is not None:
            base = base + self.readout_proj(voice_readouts)

        if shared_hidden is not None:
            # shared_hidden: (B, C, D, T) -> (B, D)
            hidden_ctx = shared_hidden.mean(dim=1).mean(dim=-1)
            hidden_ctx = self.hidden_proj(hidden_ctx)
            base = base + hidden_ctx.unsqueeze(1)

        return base
