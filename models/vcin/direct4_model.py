"""
Direct-4 SATB baseline model.

This is a first-class production baseline path: TS-BSMamba2 backbone with
4 direct outputs (S/A/T/B), no over-separation grouping loop.
"""

from __future__ import annotations

import contextlib
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.vcin.ts_backbone import TSBSMamba2Backbone


class VCINDirect4Model(nn.Module):
    """Direct SATB baseline (identity routing, no assignment loop)."""

    def __init__(
        self,
        dim,
        *,
        depth,
        stereo=False,
        num_stems=4,
        num_latent_sources=4,
        backbone_type='ts_bsmamba2',
        ts_sr=44100,
        ts_win=2048,
        ts_stride=512,
        ts_feature_dim=128,
        ts_num_repeat_mask=8,
        ts_num_repeat_map=4,
        stage1_loss_weight=1.0,
        stage2_loss_weight=1.0,
        stft_n_fft=2048,
        multi_stft_resolution_loss_weight=1.0,
        multi_stft_resolutions_window_sizes=(4096, 2048, 1024, 512, 256),
        multi_stft_hop_size=147,
        multi_stft_normalized=False,
        **kwargs,
    ):
        super().__init__()

        if kwargs:
            unknown_keys = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unsupported VCINDirect4 config keys: {unknown_keys}")

        _ = (dim, depth)
        self.num_stems = int(num_stems)
        self.num_latent_sources = int(num_latent_sources)
        self.stereo = bool(stereo)

        if self.num_latent_sources != self.num_stems:
            raise ValueError(
                "VCINDirect4 requires num_latent_sources == num_stems for strict "
                f"identity routing, got {self.num_latent_sources} vs {self.num_stems}"
            )
        if backbone_type != 'ts_bsmamba2':
            raise ValueError(
                f"VCINDirect4 only supports backbone_type='ts_bsmamba2', got '{backbone_type}'"
            )

        self.backbone = TSBSMamba2Backbone(
            num_latent_sources=self.num_stems,
            sr=int(ts_sr),
            win=int(ts_win),
            stride=int(ts_stride),
            feature_dim=int(ts_feature_dim),
            num_repeat_mask=int(ts_num_repeat_mask),
            num_repeat_map=int(ts_num_repeat_map),
            film_conditioning_dim=0,
        )

        self.stage1_loss_weight = float(stage1_loss_weight)
        self.stage2_loss_weight = float(stage2_loss_weight)
        self.multi_stft_resolution_loss_weight = float(multi_stft_resolution_loss_weight)
        self.multi_stft_resolutions_window_sizes = tuple(int(v) for v in multi_stft_resolutions_window_sizes)
        self.multi_stft_n_fft = int(stft_n_fft)
        self.multi_stft_window_fn = torch.hann_window
        self.multi_stft_kwargs = dict(
            hop_length=int(multi_stft_hop_size),
            normalized=bool(multi_stft_normalized),
        )

    def _normalize_active_stem_ids(self, active_stem_ids) -> Optional[List[int]]:
        if active_stem_ids is None:
            return None
        if torch.is_tensor(active_stem_ids):
            active_stem_ids = active_stem_ids.flatten().tolist()
        return [int(v.item() if torch.is_tensor(v) else v) for v in active_stem_ids]

    @staticmethod
    def _autocast_off(device: torch.device):
        if device.type == "cuda":
            return torch.autocast(device_type="cuda", enabled=False)
        return contextlib.nullcontext()

    def _compute_reconstruction_losses(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred32 = pred.float()
        target32 = target.float()

        l1 = F.l1_loss(pred32, target32)
        multi_stft_loss = pred32.new_zeros(())
        with self._autocast_off(device):
            for window_size in self.multi_stft_resolutions_window_sizes:
                res_stft_kwargs = dict(
                    n_fft=max(int(window_size), self.multi_stft_n_fft),
                    win_length=int(window_size),
                    return_complex=True,
                    window=self.multi_stft_window_fn(int(window_size), device=device, dtype=torch.float32),
                    **self.multi_stft_kwargs,
                )
                recon_y = torch.stft(rearrange(pred32, 'b n s t -> (b n s) t'), **res_stft_kwargs)
                target_y = torch.stft(rearrange(target32, 'b n s t -> (b n s) t'), **res_stft_kwargs)
                multi_stft_loss = multi_stft_loss + F.l1_loss(
                    torch.view_as_real(recon_y),
                    torch.view_as_real(target_y),
                )

        if len(self.multi_stft_resolutions_window_sizes) > 0:
            multi_stft_loss = multi_stft_loss / float(len(self.multi_stft_resolutions_window_sizes))

        weighted = multi_stft_loss * self.multi_stft_resolution_loss_weight
        return l1 + weighted, l1, multi_stft_loss

    def forward(
        self,
        x: torch.Tensor,
        target=None,
        active_stem_ids=None,
        f0_target=None,
        return_loss_breakdown=False,
    ):
        _ = f0_target
        device = x.device

        voice_estimates, backbone_aux = self.backbone(x, return_aux=True)
        stage1_sources = backbone_aux.get('stage1_sources')

        active_ids = self._normalize_active_stem_ids(active_stem_ids)
        if target is None:
            if active_ids is not None:
                return voice_estimates[:, active_ids]
            return voice_estimates

        if target.ndim == 2:
            target = rearrange(target, '... t -> ... 1 t')
        target = target[..., :voice_estimates.shape[-1]]

        if active_ids is not None:
            voice_sel = voice_estimates[:, active_ids]
            target_sel = target[:, active_ids]
        else:
            voice_sel = voice_estimates
            target_sel = target

        stage2_loss, l1_stage2, stft_stage2 = self._compute_reconstruction_losses(voice_sel, target_sel, device)
        total_loss = stage2_loss.new_zeros(())
        if self.stage2_loss_weight > 0:
            total_loss = total_loss + self.stage2_loss_weight * stage2_loss

        stage1_loss = voice_sel.new_zeros(())
        l1_stage1 = voice_sel.new_zeros(())
        stft_stage1 = voice_sel.new_zeros(())
        if stage1_sources is not None and self.stage1_loss_weight > 0:
            if active_ids is not None:
                stage1_sources = stage1_sources[:, active_ids]
            stage1_loss, l1_stage1, stft_stage1 = self._compute_reconstruction_losses(stage1_sources, target_sel, device)
            total_loss = total_loss + self.stage1_loss_weight * stage1_loss

        if not return_loss_breakdown:
            return total_loss

        breakdown: Dict[str, torch.Tensor] = {
            'stage2': stage2_loss.detach(),
            'stage2_l1': l1_stage2.detach(),
            'stage2_stft': stft_stage2.detach(),
            'stage1': stage1_loss.detach(),
            'stage1_l1': l1_stage1.detach(),
            'stage1_stft': stft_stage1.detach(),
            'weighted_stage2': (self.stage2_loss_weight * stage2_loss).detach(),
            'weighted_stage1': (self.stage1_loss_weight * stage1_loss).detach(),
        }
        return total_loss, breakdown
