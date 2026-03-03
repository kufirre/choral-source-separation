import logging
from typing import Optional

import torch
import torch.nn as nn

from models.ts_bs_mamba2 import Separator

logger = logging.getLogger(__name__)


class TSBSMamba2Backbone(nn.Module):
    """Thin adapter around TS-BSMamba2 Separator for VCIN."""

    def __init__(
        self,
        *,
        num_latent_sources: int,
        sr: int = 44100,
        win: int = 2048,
        stride: int = 512,
        feature_dim: int = 128,
        num_repeat_mask: int = 8,
        num_repeat_map: int = 4,
        film_conditioning_dim: int = 0,
    ):
        super().__init__()
        self.separator = Separator(
            sr=sr,
            win=win,
            stride=stride,
            feature_dim=feature_dim,
            num_repeat_mask=num_repeat_mask,
            num_repeat_map=num_repeat_map,
            num_output=num_latent_sources,
            film_conditioning_dim=film_conditioning_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
        film_conditioning: Optional[torch.Tensor] = None,
    ):
        return self.separator(x, return_aux=return_aux, film_conditioning=film_conditioning)

    def forward_heads_only(
        self,
        encoder_cache: dict,
        film_conditioning: Optional[torch.Tensor] = None,
    ):
        """Re-run mask/map heads with updated FiLM conditioning.

        Reuses cached encoder features (STFT, BN, separator blocks) and only
        re-runs the lightweight Conv1d mask/map heads + iSTFT.
        """
        return self.separator.forward_heads_only(encoder_cache, film_conditioning=film_conditioning)

    def load_ts_state_dict(self, state_dict: dict, partial_ok: bool = True):
        """Load checkpoint with support for num_output mismatch.

        When partial_ok=True and the checkpoint was trained with a different
        num_output, only the shared encoder weights (BN, separator blocks,
        in_conv, etc.) are loaded. The mask/map heads — which scale with
        num_output — are left at random init.

        This enables:
          - 4-output ckpt -> 4-output model: full transfer, zero mismatches.
          - 4-output ckpt -> 10-output model: ~195M shared encoder loaded,
            mask/map heads (~7.4M) train from scratch.
        """
        diagnostic_key = 'mask.0.1.weight'
        if diagnostic_key in state_dict and partial_ok:
            ckpt_out_dim = state_dict[diagnostic_key].shape[0]
            ckpt_num_output = ckpt_out_dim // self.separator.feature_dim
            model_num_output = self.separator.num_output

            if ckpt_num_output != model_num_output:
                logger.info(
                    "Partial transfer: checkpoint num_output=%d != model num_output=%d. "
                    "Loading shared encoder only (skipping mask.*/map.* heads).",
                    ckpt_num_output, model_num_output,
                )
                filtered = {
                    k: v for k, v in state_dict.items()
                    if not k.startswith('mask.') and not k.startswith('map.')
                }
                missing, unexpected = self.separator.load_state_dict(filtered, strict=False)
                logger.info(
                    "Partial load complete: %d keys loaded, %d missing (expected — mask/map heads), "
                    "%d unexpected.",
                    len(filtered), len(missing), len(unexpected),
                )
                return missing, unexpected

        missing, unexpected = self.separator.load_state_dict(state_dict, strict=False)
        if missing:
            logger.info("Full load: %d missing keys, %d unexpected.", len(missing), len(unexpected))
        return missing, unexpected
