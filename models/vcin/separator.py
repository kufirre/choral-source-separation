"""
CRM (Complex Ratio Mask) separator head for VCIN.

In the full VCIN pipeline, this module applies complex ratio masking
to the STFT of the mixture using the backbone's output. In the skeleton,
this is a thin wrapper that delegates separation to the BSMamba2 backbone.

The backbone itself already produces separated waveforms via its internal
BandSplit -> Mamba2/Transformer -> MaskEstimator pipeline.
"""

import torch
import torch.nn as nn


class CRMSeparator(nn.Module):
    """
    Complex Ratio Mask separator (placeholder).

    In the skeleton, the BSMamba2 backbone already handles STFT, masking,
    and iSTFT internally. This module exists as a hook point for future
    enhancements like iterative mask refinement.
    """

    def __init__(self, num_latent_sources: int = 10):
        super().__init__()
        self.num_latent_sources = num_latent_sources

    def forward(
        self,
        backbone_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            backbone_output: (batch, M, channels, time) separated sources
                             from the backbone.

        Returns:
            Same tensor (pass-through in skeleton).
        """
        return backbone_output
