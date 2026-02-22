"""
Pitch estimator module for VCIN.

Estimates a Mixture-of-Gaussians (MoG) pitch posterior for each SATB voice part.
The pitch embedding is used downstream by the assignment and FiLM modules to
guide source grouping and conditioning.

Placeholder: returns a learned constant pitch embedding per voice part.
"""

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

    def forward(
        self,
        mixture: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            mixture: (batch, channels, time) raw mixture waveform

        Returns:
            pitch_embedding: (batch, num_voices, embed_dim)
        """
        batch_size = mixture.shape[0]
        # Broadcast placeholder embedding across batch
        return self.pitch_embed.unsqueeze(0).expand(batch_size, -1, -1)
