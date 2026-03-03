"""
Soft assignment module for VCIN.

Computes a row-stochastic assignment matrix A of shape (M, V) that maps
M latent over-separated sources to V target voice parts (SATB).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftAssignment(nn.Module):
    """Bilinear source-to-voice assignment with pitch-conditioned queries
    and harmonic affinity under pitch uncertainty (architecture.md §11.3)."""

    def __init__(
        self,
        num_latent_sources: int = 10,
        num_voices: int = 4,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.num_latent_sources = num_latent_sources
        self.num_voices = num_voices
        self.embed_dim = embed_dim

        # Voice query embeddings (learnable).
        self.voice_queries = nn.Parameter(
            torch.randn(num_voices, embed_dim) * 0.02
        )
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.source_key_proj = nn.Linear(embed_dim, embed_dim)
        self.source_logit_proj = nn.Linear(embed_dim, num_voices)
        self.pitch_proj = nn.Linear(embed_dim, embed_dim)
        self.intent_proj = nn.Linear(embed_dim, embed_dim)
        self.source_pitch_proj = nn.Linear(embed_dim, 1)
        self.row_bias = nn.Parameter(torch.zeros(num_latent_sources, num_voices))
        # Voice-range priors in normalized pitch space [0, 1].
        # We store logit-space values; sigmoid is applied in forward().
        # logit(x) = log(x / (1-x)) so sigmoid(logit(x)) == x.
        default_centers = torch.logit(torch.linspace(0.2, 0.8, num_voices))
        self.voice_range_centers = nn.Parameter(default_centers)
        self.voice_range_log_scales = nn.Parameter(torch.zeros(num_voices))
        # Learned affinity scales.
        self.pitch_affinity_scale = nn.Parameter(torch.tensor(1.0))
        self.intent_affinity_scale = nn.Parameter(torch.tensor(0.5))
        self.range_affinity_scale = nn.Parameter(torch.tensor(0.5))
        self.continuity_scale = nn.Parameter(torch.tensor(0.5))
        self.voicing_affinity_scale = nn.Parameter(torch.tensor(0.4))
        self.noise_affinity_scale = nn.Parameter(torch.tensor(0.2))
        self.harmonic_affinity_scale = nn.Parameter(torch.tensor(0.5))
        self.query_norm = nn.LayerNorm(embed_dim)
        self.key_norm = nn.LayerNorm(embed_dim)
        # Temperature for softmax. Softplus keeps it strictly positive.
        self.temperature = nn.Parameter(torch.ones(1))

    def _compute_harmonic_affinity(
        self,
        source_pitch_proxy: torch.Tensor,
        pitch_details: dict,
    ) -> torch.Tensor:
        """Harmonic affinity: evaluate each source's pitch under each voice's
        MoG pitch posterior (architecture.md §11.3).

        For source m and voice v:
            harm(m, v) = log Σ_j w_{v,j} * N(p_m; μ_{v,j}, σ_{v,j}²)

        where p_m is the source's pitch proxy in [0, 1] and (w, μ, σ) are
        voice v's MoG posterior parameters.  This naturally accounts for
        pitch uncertainty: voices with broad posteriors accept more sources.

        Args:
            source_pitch_proxy: (B, M) in [0, 1]
            pitch_details: dict with 'mog_weights' (B, V, J),
                'mog_means' (B, V, J), 'mog_log_scales' (B, V, J)

        Returns:
            harmonic_affinity: (B, M, V) log-likelihood scores
        """
        mog_weights = pitch_details["mog_weights"].float()        # (B, V, J)
        mog_means = pitch_details["mog_means"].float()            # (B, V, J)
        mog_log_scales = pitch_details["mog_log_scales"].float()  # (B, V, J)

        scales = mog_log_scales.exp().clamp_min(1e-6)  # (B, V, J)

        # Expand source pitch: (B, M) -> (B, M, 1, 1) for broadcasting
        p = source_pitch_proxy.float().unsqueeze(-1).unsqueeze(-1)  # (B, M, 1, 1)

        # Expand voice MoG params: (B, V, J) -> (B, 1, V, J)
        mu = mog_means.unsqueeze(1)      # (B, 1, V, J)
        sigma = scales.unsqueeze(1)       # (B, 1, V, J)
        w = mog_weights.unsqueeze(1)      # (B, 1, V, J)

        # log N(p_m; μ_{v,j}, σ_{v,j}²)
        log_gauss = -0.5 * ((p - mu) / sigma).pow(2) - sigma.log() - 0.5 * math.log(2 * math.pi)
        # (B, M, V, J)

        # log Σ_j w_j * N(...)
        log_mixture = torch.logsumexp(
            w.clamp_min(1e-8).log() + log_gauss, dim=-1
        )  # (B, M, V)

        return log_mixture

    def forward(
        self,
        source_embeddings: torch.Tensor | None = None,
        pitch_embeddings: torch.Tensor | None = None,
        intent_embeddings: torch.Tensor | None = None,
        gate_values: torch.Tensor | None = None,
        prev_assignment: torch.Tensor | None = None,
        source_tail: torch.Tensor | None = None,
        source_voicing_probs: torch.Tensor | None = None,
        source_noise_probs: torch.Tensor | None = None,
        voice_voicing_prior: torch.Tensor | None = None,
        voice_noise_prior: torch.Tensor | None = None,
        pitch_details: dict | None = None,
        return_details: bool = False,
    ):
        """
        Args:
            source_embeddings: (batch, M, embed_dim) or None
            pitch_embeddings: (batch, V, embed_dim) or None

        Returns:
            assignment: (batch, M, V) row-stochastic assignment matrix
        """
        if source_embeddings is None:
            raise ValueError("source_embeddings is required for SoftAssignment")
        if source_embeddings.ndim != 3:
            raise ValueError(
                f"Expected source_embeddings rank=3, got shape {tuple(source_embeddings.shape)}"
            )

        batch_size, num_sources, embed_dim = source_embeddings.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Expected source embedding dim {self.embed_dim}, got {embed_dim}"
            )

        # Project source embeddings to keys.
        keys = self.key_norm(self.source_key_proj(source_embeddings))  # (B, M, D)

        # Build voice queries and optionally condition them by pitch/intent priors.
        queries = self.voice_queries.unsqueeze(0).expand(batch_size, -1, -1)  # (B, V, D)
        pitch_affinity = None
        if pitch_embeddings is not None:
            if pitch_embeddings.shape[:2] != (batch_size, self.num_voices):
                raise ValueError(
                    "pitch_embeddings must have shape (B, V, D); got "
                    f"{tuple(pitch_embeddings.shape)}"
                )
            pitch_queries = self.pitch_proj(pitch_embeddings)
            queries = queries + pitch_queries
            pitch_affinity = torch.bmm(keys, pitch_queries.transpose(1, 2)) / math.sqrt(self.embed_dim)

        intent_affinity = None
        if intent_embeddings is not None:
            if intent_embeddings.shape[:2] != (batch_size, self.num_voices):
                raise ValueError(
                    "intent_embeddings must have shape (B, V, D); got "
                    f"{tuple(intent_embeddings.shape)}"
                )
            intent_queries = self.intent_proj(intent_embeddings)
            queries = queries + intent_queries
            intent_affinity = torch.bmm(keys, intent_queries.transpose(1, 2)) / math.sqrt(self.embed_dim)

        queries = self.query_norm(self.query_proj(queries))

        # Bilinear attention + source-specific logits.
        logits = torch.bmm(keys, queries.transpose(1, 2)) / math.sqrt(self.embed_dim)
        logits = logits + self.source_logit_proj(source_embeddings)

        if pitch_affinity is not None:
            logits = logits + self.pitch_affinity_scale * pitch_affinity

        if intent_affinity is not None:
            gate_scale = 1.0
            if gate_values is not None:
                if gate_values.ndim == 3 and gate_values.shape[-1] == 1:
                    gate_scale = gate_values.transpose(1, 2)
                elif gate_values.ndim == 2:
                    gate_scale = gate_values.unsqueeze(1)
            logits = logits + self.intent_affinity_scale * intent_affinity * gate_scale

        voicing_affinity = None
        if source_voicing_probs is not None and voice_voicing_prior is not None:
            if source_voicing_probs.ndim == 2:
                source_voicing_probs = source_voicing_probs.unsqueeze(-1)
            if source_voicing_probs.shape[:2] != (batch_size, num_sources):
                raise ValueError(
                    "source_voicing_probs must have shape (B, M, 1); got "
                    f"{tuple(source_voicing_probs.shape)}"
                )
            if voice_voicing_prior.ndim == 2:
                voice_voicing_prior = voice_voicing_prior.unsqueeze(-1)
            if voice_voicing_prior.shape[:2] != (batch_size, self.num_voices):
                raise ValueError(
                    "voice_voicing_prior must have shape (B, V, 1); got "
                    f"{tuple(voice_voicing_prior.shape)}"
                )
            voicing_affinity = -torch.abs(
                source_voicing_probs.to(dtype=logits.dtype)
                - voice_voicing_prior.transpose(1, 2).to(dtype=logits.dtype)
            )
            logits = logits + self.voicing_affinity_scale * voicing_affinity

        noise_affinity = None
        if source_noise_probs is not None and voice_noise_prior is not None:
            if source_noise_probs.ndim == 2:
                source_noise_probs = source_noise_probs.unsqueeze(-1)
            if source_noise_probs.shape[:2] != (batch_size, num_sources):
                raise ValueError(
                    "source_noise_probs must have shape (B, M, 1); got "
                    f"{tuple(source_noise_probs.shape)}"
                )
            if voice_noise_prior.ndim == 2:
                voice_noise_prior = voice_noise_prior.unsqueeze(-1)
            if voice_noise_prior.shape[:2] != (batch_size, self.num_voices):
                raise ValueError(
                    "voice_noise_prior must have shape (B, V, 1); got "
                    f"{tuple(voice_noise_prior.shape)}"
                )
            noise_affinity = -torch.abs(
                source_noise_probs.to(dtype=logits.dtype)
                - voice_noise_prior.transpose(1, 2).to(dtype=logits.dtype)
            )
            logits = logits + self.noise_affinity_scale * noise_affinity

        # Voice-range affinity in normalized pitch space.
        source_pitch_proxy = torch.sigmoid(self.source_pitch_proj(source_embeddings)).squeeze(-1)  # (B, M)
        centers = torch.sigmoid(self.voice_range_centers).view(1, 1, self.num_voices)
        scales = torch.exp(self.voice_range_log_scales).view(1, 1, self.num_voices).clamp_min(1e-3)
        range_affinity = -((source_pitch_proxy.unsqueeze(-1) - centers) ** 2) / (2.0 * scales ** 2)
        logits = logits + self.range_affinity_scale * range_affinity

        # Harmonic affinity under pitch uncertainty (§11.3).
        # Evaluates each source's pitch proxy under each voice's MoG posterior.
        harmonic_affinity = None
        if pitch_details is not None and all(
            k in pitch_details for k in ("mog_weights", "mog_means", "mog_log_scales")
        ):
            harmonic_affinity = self._compute_harmonic_affinity(
                source_pitch_proxy, pitch_details,
            ).to(dtype=logits.dtype)
            logits = logits + self.harmonic_affinity_scale * harmonic_affinity

        if num_sources == self.num_latent_sources:
            row_bias = self.row_bias
        elif num_sources < self.num_latent_sources:
            row_bias = self.row_bias[:num_sources]
        else:
            row_bias = torch.zeros(
                num_sources,
                self.num_voices,
                device=logits.device,
                dtype=logits.dtype,
            )
            row_bias[:self.num_latent_sources] = self.row_bias.to(dtype=logits.dtype)
        logits = logits + row_bias.unsqueeze(0)

        # Tail inertia / continuity prior: prefer staying near previous assignment.
        if prev_assignment is not None:
            if prev_assignment.shape != logits.shape:
                raise ValueError(
                    f"prev_assignment shape {tuple(prev_assignment.shape)} must match logits {tuple(logits.shape)}"
                )
            prev_log = torch.log(prev_assignment.clamp_min(1e-8))
            if source_tail is not None:
                if source_tail.ndim == 2:
                    source_tail = source_tail.unsqueeze(-1)
                if source_tail.shape[:2] != (batch_size, num_sources):
                    raise ValueError(
                        f"source_tail must have shape (B, M, 1), got {tuple(source_tail.shape)}"
                    )
                prev_log = prev_log * source_tail
            logits = logits + self.continuity_scale * prev_log

        temp = F.softplus(self.temperature).clamp_min(1e-3)
        logits = logits / temp

        # Row-stochastic: softmax over voices for each source.
        assignment = F.softmax(logits, dim=-1)  # (B, M, V)

        if not return_details:
            return assignment

        details = {
            "source_pitch_proxy": source_pitch_proxy,
            "range_affinity": range_affinity,
            "temperature": temp.detach(),
        }
        if pitch_affinity is not None:
            details["pitch_affinity"] = pitch_affinity
        if intent_affinity is not None:
            details["intent_affinity"] = intent_affinity
        if voicing_affinity is not None:
            details["voicing_affinity"] = voicing_affinity
        if noise_affinity is not None:
            details["noise_affinity"] = noise_affinity
        if harmonic_affinity is not None:
            details["harmonic_affinity"] = harmonic_affinity
        return assignment, details
