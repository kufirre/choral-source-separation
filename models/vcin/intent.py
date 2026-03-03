"""
Articulatory intent module for VCIN.

Models the articulatory intent of each voice part (e.g., vowel identity,
consonant type, vibrato) as a latent variable. The intent embedding is
used to condition the separation and assignment modules.

Implements anti-pitch-leakage via gradient reversal (architecture.md §10.1):
a pitch adversary head predicts expected pitch from intent embeddings with
reversed gradients, forcing intent to learn pitch-orthogonal features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradientReversal(torch.autograd.Function):
    """Reverse gradients during backward pass (Ganin & Lempitsky, 2015)."""

    @staticmethod
    def forward(ctx, x, lambda_: float):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class IntentModule(nn.Module):
    """Articulatory-intent proxy from voice statistics and shared context.

    Includes a pitch adversary with gradient reversal to prevent pitch
    information from leaking into the intent representation.
    """

    def __init__(self, num_voices: int = 4, embed_dim: int = 64, gradient_reversal_lambda: float = 0.1):
        super().__init__()
        self.num_voices = num_voices
        self.embed_dim = embed_dim
        self.gradient_reversal_lambda = float(gradient_reversal_lambda)

        self.voice_priors = nn.Parameter(
            torch.randn(num_voices, embed_dim) * 0.02
        )
        self.hidden_proj = nn.LazyLinear(embed_dim)
        self.readout_proj = nn.Linear(embed_dim, embed_dim)
        self.voice_stats_proj = nn.Linear(4, embed_dim)
        self.fuse = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

        # Pitch adversary: predicts expected pitch from intent embeddings.
        # Gradient reversal forces intent to NOT encode pitch information.
        self.pitch_adversary = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),  # output in [0, 1] (normalized pitch space)
        )

    def forward(
        self,
        voice_estimates: torch.Tensor | None = None,
        shared_hidden: torch.Tensor | None = None,
        voice_readouts: torch.Tensor | None = None,
        return_details: bool = False,
    ):
        """
        Args:
            voice_estimates: (batch, V, channels, time) or None

        Returns:
            intent_embedding: (batch, V, embed_dim)
            if return_details=True, also returns pairwise intent affinity.
        """
        batch_size = None
        device = None
        dtype = None
        if voice_estimates is not None:
            batch_size = voice_estimates.shape[0]
            device = voice_estimates.device
            dtype = voice_estimates.dtype
        if batch_size is None and voice_readouts is not None:
            batch_size = voice_readouts.shape[0]
            device = voice_readouts.device
            dtype = voice_readouts.dtype
        if batch_size is None and shared_hidden is not None:
            batch_size = shared_hidden.shape[0]
            device = shared_hidden.device
            dtype = shared_hidden.dtype
        if batch_size is None:
            batch_size = 1
            device = self.voice_priors.device
            dtype = self.voice_priors.dtype

        base = self.voice_priors.unsqueeze(0).expand(batch_size, -1, -1).to(device=device, dtype=dtype)

        if voice_readouts is not None:
            base = base + self.readout_proj(voice_readouts)

        if shared_hidden is not None:
            hidden_ctx = shared_hidden.mean(dim=1).mean(dim=-1)
            hidden_ctx = self.hidden_proj(hidden_ctx)
            hidden_ctx = hidden_ctx.unsqueeze(1).expand(-1, self.num_voices, -1).to(dtype=dtype)
        else:
            hidden_ctx = torch.zeros_like(base)

        if voice_estimates is not None:
            mono = voice_estimates.float().mean(dim=2)
            abs_mean = mono.abs().mean(dim=-1)
            std = mono.std(dim=-1)
            zcr = (mono[..., 1:] * mono[..., :-1] < 0).float().mean(dim=-1)
            crest = mono.abs().amax(dim=-1) / abs_mean.clamp_min(1e-6)
            stats = torch.stack([abs_mean, std, zcr, crest], dim=-1)
        elif voice_readouts is not None:
            vr = voice_readouts.float()
            abs_mean = vr.abs().mean(dim=-1)
            std = vr.std(dim=-1)
            peak = vr.abs().amax(dim=-1)
            trough = vr.abs().amin(dim=-1)
            stats = torch.stack([abs_mean, std, peak, trough], dim=-1)
        else:
            stats = torch.zeros(batch_size, self.num_voices, 4, device=device, dtype=torch.float32)
        stats_embed = self.voice_stats_proj(stats.to(dtype=dtype))

        fused = torch.cat([base, hidden_ctx, stats_embed], dim=-1)
        intent_embed = self.fuse(fused)

        # Pitch adversary with gradient reversal (architecture.md §10.1).
        # In forward: identity pass-through. In backward: reversed gradients
        # push intent embeddings to be pitch-orthogonal.
        reversed_embed = _GradientReversal.apply(
            intent_embed, self.gradient_reversal_lambda,
        )
        adversary_pitch_pred = self.pitch_adversary(reversed_embed.float()).squeeze(-1)  # (B, V)

        if not return_details:
            return intent_embed

        normed = F.normalize(intent_embed.float(), dim=-1)
        pairwise_cos = torch.bmm(normed, normed.transpose(1, 2))
        details = {
            "pairwise_cosine": pairwise_cos,
            "mean_intent_norm": intent_embed.float().norm(dim=-1).mean(dim=-1),
            "adversary_pitch_pred": adversary_pitch_pred,
        }
        return intent_embed, details
