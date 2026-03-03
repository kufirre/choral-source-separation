"""
Acoustic-to-symbolic soft tokenization for VCIN.

Converts continuous MoG pitch posteriors into soft token probabilities and
expected token embeddings for CGP conditioning.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftPitchTokenizer(nn.Module):
    """RBF-style soft quantizer from pitch posterior to symbolic token space."""

    def __init__(
        self,
        *,
        num_tokens: int = 128,
        embed_dim: int = 64,
        temperature_start: float = 1.0,
        temperature_end: float = 0.25,
        anneal_steps: int = 20000,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.embed_dim = int(embed_dim)
        self.temperature_start = float(temperature_start)
        self.temperature_end = float(temperature_end)
        self.anneal_steps = int(anneal_steps)

        self.register_buffer(
            "token_centers",
            torch.linspace(0.0, 1.0, self.num_tokens, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "token_indices",
            torch.arange(self.num_tokens, dtype=torch.float32),
            persistent=False,
        )
        self.token_embedding = nn.Embedding(self.num_tokens, self.embed_dim)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

    def _temperature(self, train_step: Optional[int]) -> float:
        if train_step is None or self.anneal_steps <= 0:
            return self.temperature_start
        progress = min(max(float(train_step) / float(self.anneal_steps), 0.0), 1.0)
        return self.temperature_start + progress * (self.temperature_end - self.temperature_start)

    def _mog_to_token_probs(
        self,
        mog_weights: torch.Tensor,
        mog_means: torch.Tensor,
        mog_log_scales: torch.Tensor,
        rest_prob: Optional[torch.Tensor],
        temperature: float,
    ) -> torch.Tensor:
        # Shapes:
        #   mog_*: (B, V, J)
        #   token_centers: (N,)
        centers = self.token_centers.to(device=mog_weights.device, dtype=mog_weights.dtype)
        centers = centers.view(1, 1, 1, self.num_tokens)

        means = mog_means.unsqueeze(-1)
        scales = torch.exp(mog_log_scales).clamp_min(1e-3).unsqueeze(-1)
        gauss = torch.exp(-0.5 * ((centers - means) / scales).pow(2)) / (
            scales * math.sqrt(2.0 * math.pi)
        )

        token_scores = (mog_weights.unsqueeze(-1) * gauss).sum(dim=2)  # (B, V, N)
        token_scores = token_scores.clamp_min(1e-12)
        token_probs = F.softmax(torch.log(token_scores) / max(float(temperature), 1e-3), dim=-1)

        if rest_prob is not None:
            rest = rest_prob.unsqueeze(-1).to(dtype=token_probs.dtype)
            uniform = torch.full_like(token_probs, 1.0 / float(self.num_tokens))
            token_probs = (1.0 - rest) * token_probs + rest * uniform
            token_probs = token_probs / token_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        return token_probs

    def forward(
        self,
        pitch_details: Dict[str, torch.Tensor],
        *,
        train_step: Optional[int] = None,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            pitch_details: output dict from PitchEstimator(return_details=True)
            train_step: used for temperature annealing

        Returns:
            token_embeddings: (B, V, D)
            token_probs: (B, V, N)
            optional details dict
        """
        required = ("mog_weights", "mog_means", "mog_log_scales")
        for key in required:
            if key not in pitch_details:
                raise KeyError(f"Missing '{key}' in pitch_details for soft tokenization")

        mog_weights = pitch_details["mog_weights"]
        mog_means = pitch_details["mog_means"]
        mog_log_scales = pitch_details["mog_log_scales"]
        rest_prob = pitch_details.get("rest_prob")

        temp = self._temperature(train_step)
        token_probs = self._mog_to_token_probs(
            mog_weights=mog_weights,
            mog_means=mog_means,
            mog_log_scales=mog_log_scales,
            rest_prob=rest_prob,
            temperature=temp,
        )

        token_embed = torch.matmul(token_probs, self.token_embedding.weight.to(dtype=token_probs.dtype))

        if not return_details:
            return token_embed, token_probs

        token_idx = self.token_indices.to(device=token_probs.device, dtype=token_probs.dtype).view(1, 1, -1)
        expected_index = (token_probs * token_idx).sum(dim=-1)
        entropy = -(token_probs * torch.log(token_probs.clamp_min(1e-8))).sum(dim=-1)
        details = {
            "token_temperature": token_probs.new_tensor(float(temp)),
            "token_expected_index": expected_index,
            "token_entropy": entropy,
        }
        return token_embed, token_probs, details
