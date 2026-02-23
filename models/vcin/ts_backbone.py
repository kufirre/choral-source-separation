import torch
import torch.nn as nn

from models.ts_bs_mamba2 import Separator


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
        )

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        return self.separator(x, return_aux=return_aux)

    def load_ts_state_dict(self, state_dict: dict):
        missing, unexpected = self.separator.load_state_dict(state_dict, strict=False)
        return missing, unexpected

