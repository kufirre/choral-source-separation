"""
Pitch estimator module for VCIN.

Estimates a Mixture-of-Gaussians (MoG) pitch posterior for each SATB voice part.
The pitch embedding is used downstream by the assignment and FiLM modules to
guide source grouping and conditioning.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PitchEstimator(nn.Module):
    """Per-voice pitch-aware embedding + MoG posterior estimator."""

    def __init__(self, num_voices: int = 4, embed_dim: int = 64, num_components: int = 3):
        super().__init__()
        self.num_voices = num_voices
        self.embed_dim = embed_dim
        self.num_components = num_components

        self.voice_queries = nn.Parameter(
            torch.randn(num_voices, embed_dim) * 0.02
        )
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.hidden_token_proj = nn.LazyLinear(embed_dim)
        self.readout_proj = nn.Linear(embed_dim, embed_dim)
        self.mix_stats_proj = nn.Linear(4, embed_dim)
        self.out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        self.mog_weight_head = nn.Linear(embed_dim, num_components)
        self.mog_mean_head = nn.Linear(embed_dim, num_components)
        self.mog_log_scale_head = nn.Linear(embed_dim, num_components)
        self.rest_logit_head = nn.Linear(embed_dim, 1)

    def forward(
        self,
        mixture: torch.Tensor | None = None,
        shared_hidden: torch.Tensor | None = None,
        voice_readouts: torch.Tensor | None = None,
        return_details: bool = False,
    ):
        """
        Args:
            mixture: (batch, channels, time) raw mixture waveform

        Returns:
            pitch_embedding: (batch, num_voices, embed_dim)
            if return_details=True, also returns details dict with MoG params.
        """
        batch_size = None
        device = None
        dtype = None
        if mixture is not None:
            batch_size = mixture.shape[0]
            device = mixture.device
            dtype = mixture.dtype
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
            device = self.voice_queries.device
            dtype = self.voice_queries.dtype

        queries = self.voice_queries.unsqueeze(0).expand(batch_size, -1, -1).to(device=device, dtype=dtype)

        if voice_readouts is not None:
            queries = queries + self.readout_proj(voice_readouts)
        queries = self.query_proj(queries)

        if mixture is not None:
            mix_mono = mixture.float().mean(dim=1)
            abs_mean = mix_mono.abs().mean(dim=-1)
            rms = mix_mono.pow(2).mean(dim=-1).sqrt()
            std = mix_mono.std(dim=-1)
            crest = mix_mono.abs().amax(dim=-1) / abs_mean.clamp_min(1e-6)
            mix_stats = torch.stack([abs_mean, rms, std, crest], dim=-1).to(dtype=dtype)
            queries = queries + self.mix_stats_proj(mix_stats).unsqueeze(1)

        if shared_hidden is not None:
            # shared_hidden: (B, C, D, T) -> token sequence (B, T, D)
            hidden_tokens = shared_hidden.mean(dim=1).transpose(1, 2).contiguous()
            hidden_tokens = self.hidden_token_proj(hidden_tokens.to(dtype=dtype))
            keys = self.key_proj(hidden_tokens)
            values = self.value_proj(hidden_tokens)

            attn_logits = torch.bmm(queries, keys.transpose(1, 2)) / (self.embed_dim ** 0.5)
            attn = torch.softmax(attn_logits, dim=-1)
            ctx = torch.bmm(attn, values)
            queries = queries + ctx

        pitch_embed = self.out(queries)

        # MoG posterior parameters in normalized pitch space [0, 1].
        mog_logits = self.mog_weight_head(pitch_embed)
        mog_weights = torch.softmax(mog_logits, dim=-1)
        mog_means = torch.sigmoid(self.mog_mean_head(pitch_embed))
        mog_log_scales = torch.clamp(self.mog_log_scale_head(pitch_embed), min=-6.0, max=2.0)
        rest_logit = self.rest_logit_head(pitch_embed)
        rest_prob = torch.sigmoid(rest_logit).squeeze(-1)

        nonrest_mass = (1.0 - rest_prob).clamp_min(1e-6)
        expected_pitch = (mog_weights * mog_means).sum(dim=-1) * nonrest_mass
        entropy = -(mog_weights * (mog_weights.clamp_min(1e-8)).log()).sum(dim=-1)
        confidence = nonrest_mass * (1.0 - entropy / float(max(self.num_components, 2)))
        confidence = confidence.clamp(0.0, 1.0)

        if not return_details:
            return pitch_embed

        details = {
            "mog_weights": mog_weights,
            "mog_means": mog_means,
            "mog_log_scales": mog_log_scales,
            "rest_prob": rest_prob,
            "expected_pitch": expected_pitch,
            "pitch_confidence": confidence,
        }
        return pitch_embed, details
