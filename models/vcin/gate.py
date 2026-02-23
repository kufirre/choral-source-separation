"""
Texture / synchrony gate module for VCIN.

Produces a scalar gate value per voice part that modulates between
texture-based separation (for homophonic passages where voices move
together) and pitch-based separation (for polyphonic passages).

Placeholder: returns 0.5 (equal weight to both pathways).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TextureGate(nn.Module):
    """
    Texture/synchrony gate (placeholder).

    Full implementation: analyzes inter-voice onset correlation and
    rhythmic similarity to determine if voices are moving together
    (homophonic) or independently (polyphonic).
    Placeholder: returns constant 0.5.
    """

    def __init__(self, num_voices: int = 4, input_dim: int = 64):
        super().__init__()
        self.num_voices = num_voices
        self.input_dim = input_dim

        # Gate predictor
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.hidden_proj = nn.LazyLinear(input_dim)

    def forward(
        self,
        voice_features: torch.Tensor | None = None,
        shared_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            voice_features: (batch, V, input_dim) or None

        Returns:
            gate: (batch, V, 1) values in [0, 1]
        """
        if voice_features is None:
            batch_size = 1
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            voice_features = torch.zeros(batch_size, self.num_voices, self.input_dim, device=device, dtype=dtype)
        else:
            batch_size = voice_features.shape[0]
            device = voice_features.device
            dtype = voice_features.dtype

        if shared_hidden is not None:
            hidden_ctx = shared_hidden.mean(dim=1).mean(dim=-1)
            hidden_ctx = self.hidden_proj(hidden_ctx).to(dtype=dtype)
            voice_features = voice_features + hidden_ctx.unsqueeze(1)

        return self.gate_net(voice_features)  # (B, V, 1)
