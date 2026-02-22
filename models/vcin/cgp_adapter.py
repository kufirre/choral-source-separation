"""
CGP (Choral Gestural Profile) adapter for VCIN.

Wraps a frozen pre-trained CGP encoder with a lightweight trainable
adapter layer. The CGP provides gestural/phonetic features that are
used to condition the separation via FiLM.

Placeholder: returns zeros (disabled by default).
"""

import torch
import torch.nn as nn


class CGPAdapter(nn.Module):
    """
    Frozen CGP + trainable adapter (placeholder).

    Full implementation: loads a pre-trained choral gestural profile
    encoder (frozen), passes audio through it, and applies a small
    trainable adapter MLP on top.
    Placeholder: returns zero embedding (no conditioning).
    """

    def __init__(self, output_dim: int = 64):
        super().__init__()
        self.output_dim = output_dim

        # Placeholder adapter (input dim would match frozen CGP output)
        self.adapter = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            audio: (batch, channels, time) mixture waveform

        Returns:
            cgp_features: (batch, output_dim) gestural profile features
        """
        batch_size = audio.shape[0]
        # Placeholder: return zeros (CGP disabled by default)
        return torch.zeros(
            batch_size, self.output_dim,
            device=audio.device, dtype=audio.dtype,
        )
