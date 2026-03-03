"""
Voicing / aperiodicity head for VCIN.

Provides per-latent voiced/noise probabilities used by assignment and tail-inertia
updates. This implements the paper's "voicing/aperiodicity" signal path without
adding a redundant encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VoicingAperiodicityHead(nn.Module):
    """Estimate per-source voiced/noise probabilities from latent waveforms."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        # 4 waveform descriptors: RMS, abs-mean, ZCR, diff-energy
        self.source_stats_proj = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.shared_hidden_proj = nn.LazyLinear(self.hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 2),  # voiced_logit, noise_logit
        )

    def forward(
        self,
        latent_sources: torch.Tensor,
        shared_hidden: torch.Tensor | None = None,
        return_details: bool = False,
    ):
        """
        Args:
            latent_sources: (B, M, C, T)
            shared_hidden: optional backbone tensor (B, C, D, T')

        Returns:
            p_voiced: (B, M, 1)
            p_noise: (B, M, 1)
            tail_strength: (B, M, 1)
            optional details dict
        """
        if latent_sources.ndim != 4:
            raise ValueError(f"Expected latent_sources shape (B, M, C, T), got {tuple(latent_sources.shape)}")

        x = latent_sources.float().mean(dim=2)  # (B, M, T)
        abs_mean = x.abs().mean(dim=-1)
        rms = x.pow(2).mean(dim=-1).sqrt()
        zcr = (x[..., 1:] * x[..., :-1] < 0).float().mean(dim=-1)
        diff_energy = (x[..., 1:] - x[..., :-1]).pow(2).mean(dim=-1).sqrt()

        stats = torch.stack([rms, abs_mean, zcr, diff_energy], dim=-1)  # (B, M, 4)
        feat = self.source_stats_proj(stats.to(dtype=latent_sources.dtype))

        if shared_hidden is not None:
            # Use global backbone context without introducing another encoder.
            hidden_ctx = shared_hidden.mean(dim=1).mean(dim=-1)  # (B, D)
            hidden_ctx = self.shared_hidden_proj(hidden_ctx).to(dtype=latent_sources.dtype)
            feat = feat + hidden_ctx.unsqueeze(1)

        logits = self.out(feat)
        p_voiced = torch.sigmoid(logits[..., :1])
        p_noise_raw = torch.sigmoid(logits[..., 1:])
        # Tie p_noise to voicing while still allowing learnable correction.
        p_noise = 0.5 * p_noise_raw + 0.5 * (1.0 - p_voiced)

        # Tail proxy used in assignment inertia.
        tail_strength = torch.sigmoid(2.0 * p_noise - p_voiced + 0.5 * zcr.unsqueeze(-1))

        p_voiced = p_voiced.to(dtype=latent_sources.dtype)
        p_noise = p_noise.to(dtype=latent_sources.dtype)
        tail_strength = tail_strength.to(dtype=latent_sources.dtype)

        if not return_details:
            return p_voiced, p_noise, tail_strength

        eps = 1e-8
        voicing_entropy = -(
            p_voiced.float() * torch.log(p_voiced.float().clamp_min(eps))
            + (1.0 - p_voiced.float()) * torch.log((1.0 - p_voiced.float()).clamp_min(eps))
        )
        details = {
            "voicing_entropy": voicing_entropy.mean(dim=-1),
            "rms": rms,
            "zcr": zcr,
            "diff_energy": diff_energy,
        }
        return p_voiced, p_noise, tail_strength, details
