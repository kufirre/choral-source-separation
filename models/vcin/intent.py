"""
Articulatory intent module for VCIN.

Models the articulatory intent of each voice part (e.g., vowel identity,
consonant type, vibrato) as a latent variable. The intent embedding is
used to condition the separation and assignment modules.

Placeholder: returns a learned constant embedding per voice part.
"""

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

    def forward(
        self,
        voice_estimates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            voice_estimates: (batch, V, channels, time) or None

        Returns:
            intent_embedding: (batch, V, embed_dim)
        """
        if voice_estimates is not None:
            batch_size = voice_estimates.shape[0]
        else:
            batch_size = 1

        return self.intent_embed.unsqueeze(0).expand(batch_size, -1, -1)
