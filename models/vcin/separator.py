"""
CRM (Complex Ratio Mask) separator head for VCIN.

The TS-BSMamba2 backbone already performs main STFT-mask-iSTFT separation.
This module applies a lightweight, learnable waveform refinement so VCIN has
an explicit separator hook point that is active (not a pass-through).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CRMSeparator(nn.Module):
    """Lightweight waveform-domain refinement head after backbone separation."""

    def __init__(self, num_latent_sources: int = 10):
        super().__init__()
        self.num_latent_sources = num_latent_sources
        # Small temporal smoothing residual for stability.
        self.smoothing_alpha = nn.Parameter(torch.tensor(0.0))
        # Per-source affine calibration in waveform domain.
        self.source_gain = nn.Parameter(torch.ones(num_latent_sources))
        self.source_bias = nn.Parameter(torch.zeros(num_latent_sources))

    def forward(
        self,
        backbone_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            backbone_output: (batch, M, channels, time) separated sources
                             from the backbone.

        Returns:
            Refined tensor with same shape.
        """
        if backbone_output.ndim != 4:
            raise ValueError(
                f"CRMSeparator expects (B, M, C, T), got {tuple(backbone_output.shape)}"
            )

        bsz, num_sources, channels, num_frames = backbone_output.shape
        x = backbone_output

        # Temporal smoothing branch.
        smooth = F.avg_pool1d(
            x.reshape(bsz * num_sources * channels, 1, num_frames),
            kernel_size=5,
            stride=1,
            padding=2,
        ).reshape(bsz, num_sources, channels, num_frames)
        alpha = 0.5 * torch.sigmoid(self.smoothing_alpha)
        refined = x + alpha * (smooth - x)

        # Source-wise affine calibration. If runtime source count differs from
        # configured count, use neutral affine for out-of-range sources.
        if num_sources == self.num_latent_sources:
            gain = self.source_gain
            bias = self.source_bias
        elif num_sources < self.num_latent_sources:
            gain = self.source_gain[:num_sources]
            bias = self.source_bias[:num_sources]
        else:
            gain = torch.ones(num_sources, device=x.device, dtype=x.dtype)
            bias = torch.zeros(num_sources, device=x.device, dtype=x.dtype)
            gain[:self.num_latent_sources] = self.source_gain.to(dtype=x.dtype)
            bias[:self.num_latent_sources] = self.source_bias.to(dtype=x.dtype)

        gain = gain.view(1, num_sources, 1, 1).to(dtype=x.dtype)
        bias = bias.view(1, num_sources, 1, 1).to(dtype=x.dtype)
        return refined * gain + bias
