"""
FiLM (Feature-wise Linear Modulation) conditioning.

Applies affine transformation gamma * x + beta, where gamma and beta
are derived from a conditioning signal. Used to inject pitch/intent
information into the separation pathway.

Placeholder: passes input through unchanged.
"""

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: gamma * x + beta."""

    def __init__(self, feature_dim: int, conditioning_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.conditioning_dim = conditioning_dim

        # Conditioning -> (gamma, beta)
        self.fc = nn.Linear(conditioning_dim, feature_dim * 2)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        # Initialize gamma bias to 1 (identity), beta bias to 0
        with torch.no_grad():
            self.fc.bias[:feature_dim] = 1.0

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, ..., feature_dim) input features
            conditioning: (batch, conditioning_dim) conditioning vector

        Returns:
            Modulated features of same shape as x.
        """
        gamma_beta = self.fc(conditioning)  # (batch, feature_dim * 2)
        gamma = gamma_beta[..., : self.feature_dim]
        beta = gamma_beta[..., self.feature_dim :]

        # Broadcast gamma/beta over spatial dimensions
        while gamma.ndim < x.ndim:
            gamma = gamma.unsqueeze(-2)
            beta = beta.unsqueeze(-2)

        return gamma * x + beta
