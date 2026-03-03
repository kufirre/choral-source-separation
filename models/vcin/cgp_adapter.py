"""
SATB CGP adapter for VCIN.

Implements:
  1) trainable adapter over soft symbolic inputs,
  2) frozen SATB prior proxy over interval/order constraints and pairwise affinity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CGPAdapter(nn.Module):
    """
    Trainable adapter + frozen SATB prior proxy.

    The adapter transforms soft symbolic/acoustic tokens into a space used by a
    frozen prior scorer. The scorer combines interval plausibility, SATB ordering,
    and adjacent-voice affinity.
    """

    def __init__(
        self,
        output_dim: int = 64,
        num_voices: int = 4,
        nonadj_penalty: float = 0.5,
        interval_scale: float = 0.05,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.num_voices = int(num_voices)
        if self.num_voices < 2:
            raise ValueError("CGPAdapter requires num_voices >= 2")

        self.nonadj_penalty = float(nonadj_penalty)
        self.interval_scale = float(interval_scale)

        self.adapter = nn.Linear(self.output_dim, self.output_dim)
        self.norm = nn.LayerNorm(self.output_dim)
        self.pitch_proxy = nn.Linear(self.output_dim, 1)

        adjacent_pairs = [[i, i + 1] for i in range(self.num_voices - 1)]
        nonadj_pairs = []
        for i in range(self.num_voices):
            for j in range(i + 1, self.num_voices):
                if j != i + 1:
                    nonadj_pairs.append([i, j])

        self.register_buffer(
            "adjacent_pairs",
            torch.tensor(adjacent_pairs, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "nonadjacent_pairs",
            torch.tensor(nonadj_pairs, dtype=torch.long) if nonadj_pairs else torch.empty((0, 2), dtype=torch.long),
            persistent=False,
        )

        # Frozen interval priors for (S-A, A-T, T-B) in token-index units.
        # S should be higher than A, A higher than T, T higher than B.
        default_centers = torch.full((self.num_voices - 1,), 8.0, dtype=torch.float32)
        default_scales = torch.full((self.num_voices - 1,), 6.0, dtype=torch.float32)
        self.register_buffer("interval_centers", default_centers, persistent=False)
        self.register_buffer("interval_scales", default_scales, persistent=False)

    def _compute_pitch_positions(
        self,
        adapted_tokens: torch.Tensor,
        token_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mu: (B, V) expected pitch/token position
            var: (B, V) uncertainty proxy
        """
        if token_probs is not None:
            if token_probs.ndim != 3:
                raise ValueError(f"token_probs must have shape (B, V, N), got {tuple(token_probs.shape)}")
            num_tokens = token_probs.shape[-1]
            idx = torch.arange(num_tokens, device=token_probs.device, dtype=token_probs.dtype).view(1, 1, -1)
            mu = (token_probs * idx).sum(dim=-1)
            var = (token_probs * (idx - mu.unsqueeze(-1)).pow(2)).sum(dim=-1)
            return mu.float(), var.float()

        mu = torch.sigmoid(self.pitch_proxy(adapted_tokens.float())).squeeze(-1) * 127.0
        var = torch.ones_like(mu) * 16.0
        return mu, var

    def forward(
        self,
        voice_tokens: torch.Tensor,
        token_probs: torch.Tensor | None = None,
        return_details: bool = False,
    ):
        """
        Args:
            voice_tokens: (B, V, D)
            token_probs: optional (B, V, N) from soft tokenization

        Returns:
            adapted_tokens: (B, V, D)
            log_p_cgp: (B,)
            optional details
        """
        if voice_tokens.ndim != 3:
            raise ValueError(f"Expected voice_tokens rank=3, got shape={tuple(voice_tokens.shape)}")
        if voice_tokens.shape[1] != self.num_voices:
            raise ValueError(f"Expected num_voices={self.num_voices}, got {voice_tokens.shape[1]}")

        adapted_tokens = self.norm(self.adapter(voice_tokens))
        tokens = F.normalize(adapted_tokens.float(), dim=-1)
        sim = torch.bmm(tokens, tokens.transpose(1, 2))

        adj_sim = sim[:, self.adjacent_pairs[:, 0], self.adjacent_pairs[:, 1]].mean(dim=-1)
        if self.nonadjacent_pairs.numel() > 0:
            nonadj_sim = sim[:, self.nonadjacent_pairs[:, 0], self.nonadjacent_pairs[:, 1]].mean(dim=-1)
        else:
            nonadj_sim = torch.zeros_like(adj_sim)
        pairwise_prior = adj_sim - self.nonadj_penalty * nonadj_sim

        mu, var = self._compute_pitch_positions(adapted_tokens, token_probs=token_probs)
        if self.num_voices > 1:
            intervals = mu[:, :-1] - mu[:, 1:]  # should be positive for SATB ordering
            centers = self.interval_centers.to(device=intervals.device, dtype=intervals.dtype).view(1, -1)
            scales = self.interval_scales.to(device=intervals.device, dtype=intervals.dtype).view(1, -1).clamp_min(1e-3)
            interval_prior = -0.5 * ((intervals - centers) / scales).pow(2)
            interval_prior = interval_prior.sum(dim=-1)
            order_penalty = F.relu(mu[:, 1:] - mu[:, :-1]).sum(dim=-1)
        else:
            interval_prior = torch.zeros(mu.shape[0], device=mu.device, dtype=mu.dtype)
            order_penalty = torch.zeros_like(interval_prior)

        uncertainty_penalty = 0.01 * var.mean(dim=-1)
        log_p_cgp = pairwise_prior + self.interval_scale * interval_prior - 0.1 * order_penalty - uncertainty_penalty
        log_p_cgp = log_p_cgp.to(dtype=voice_tokens.dtype)

        if not return_details:
            return adapted_tokens.to(dtype=voice_tokens.dtype), log_p_cgp

        details = {
            "pairwise_prior": pairwise_prior.to(dtype=voice_tokens.dtype),
            "interval_prior": interval_prior.to(dtype=voice_tokens.dtype),
            "order_penalty": order_penalty.to(dtype=voice_tokens.dtype),
            "pitch_position_mu": mu.to(dtype=voice_tokens.dtype),
            "pitch_position_var": var.to(dtype=voice_tokens.dtype),
        }
        return adapted_tokens.to(dtype=voice_tokens.dtype), log_p_cgp, details
