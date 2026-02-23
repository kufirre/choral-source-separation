"""
CGP (Choral Grammar Prior) adapter for VCIN.

Implements a trainable adapter over voice-token embeddings and a frozen
SATB prior scorer. The scorer is intentionally lightweight so it can be
used without external CGP checkpoints while keeping gradients stable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CGPAdapter(nn.Module):
    """
    Trainable adapter + frozen SATB prior scorer.

    Inputs are soft voice tokens with shape (B, V, D), where V is SATB.
    The adapter produces transformed tokens and a frozen prior log-probability
    proxy that can be optimized via L_cgp = -log P_CGP.
    """

    def __init__(
        self,
        output_dim: int = 64,
        num_voices: int = 4,
        nonadj_penalty: float = 0.5,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.num_voices = int(num_voices)
        if self.num_voices < 2:
            raise ValueError("CGPAdapter requires num_voices >= 2")

        self.nonadj_penalty = float(nonadj_penalty)

        # Trainable adapter; the downstream prior scorer remains frozen.
        self.adapter = nn.Linear(self.output_dim, self.output_dim)
        self.norm = nn.LayerNorm(self.output_dim)

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

    def forward(self, voice_tokens: torch.Tensor):
        """
        Args:
            voice_tokens: (batch, num_voices, output_dim)

        Returns:
            adapted_tokens: (batch, num_voices, output_dim)
            log_p_cgp: (batch,) frozen-prior compatibility score
        """
        if voice_tokens.ndim != 3:
            raise ValueError(f"Expected voice_tokens rank=3, got shape={tuple(voice_tokens.shape)}")
        if voice_tokens.shape[1] != self.num_voices:
            raise ValueError(
                f"Expected num_voices={self.num_voices}, got {voice_tokens.shape[1]}"
            )

        adapted_tokens = self.norm(self.adapter(voice_tokens))

        # Frozen SATB prior proxy over pairwise relationships.
        tokens = F.normalize(adapted_tokens.float(), dim=-1)
        sim = torch.bmm(tokens, tokens.transpose(1, 2))

        adj_sim = sim[:, self.adjacent_pairs[:, 0], self.adjacent_pairs[:, 1]].mean(dim=-1)
        if self.nonadjacent_pairs.numel() > 0:
            nonadj_sim = sim[:, self.nonadjacent_pairs[:, 0], self.nonadjacent_pairs[:, 1]].mean(dim=-1)
        else:
            nonadj_sim = torch.zeros_like(adj_sim)

        log_p_cgp = adj_sim - self.nonadj_penalty * nonadj_sim
        return adapted_tokens.to(dtype=voice_tokens.dtype), log_p_cgp
