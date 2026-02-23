"""
Articulatory intent module for VCIN.

Models the articulatory intent of each voice part (e.g., vowel identity,
consonant type, vibrato) as a latent variable. The intent embedding is
used to condition the separation and assignment modules.

Placeholder: returns a learned constant embedding per voice part.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class IntentModule(nn.Module):
    """
    Articulatory intent latent (placeholder).

    Full implementation: infers a posterior over articulatory states
    from the mixture spectrogram and current voice estimates.
    Placeholder: returns a fixed learned embedding.
    """

    def __init__(self, num_voices: int = 4, embed_dim: int = 64):
        super().__init__()
        self.num_voices = num_voices
        self.embed_dim = embed_dim

        # Placeholder: fixed learned embedding per voice
        self.intent_embed = nn.Parameter(
            torch.randn(num_voices, embed_dim) * 0.02
        )
        self.hidden_proj = nn.LazyLinear(embed_dim)
        self.readout_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        voice_estimates: torch.Tensor | None = None,
        shared_hidden: torch.Tensor | None = None,
        voice_readouts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            voice_estimates: (batch, V, channels, time) or None

        Returns:
            intent_embedding: (batch, V, embed_dim)
        """
        if voice_estimates is not None:
            batch_size = voice_estimates.shape[0]
            device = voice_estimates.device
            dtype = voice_estimates.dtype
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
            device = self.intent_embed.device
            dtype = self.intent_embed.dtype

        base = self.intent_embed.unsqueeze(0).expand(batch_size, -1, -1).to(device=device, dtype=dtype)

        if voice_readouts is not None:
            base = base + self.readout_proj(voice_readouts)

        if shared_hidden is not None:
            hidden_ctx = shared_hidden.mean(dim=1).mean(dim=-1)
            hidden_ctx = self.hidden_proj(hidden_ctx)
            base = base + hidden_ctx.unsqueeze(1)

        return base
