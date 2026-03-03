"""
Texture / synchrony gate module for VCIN.

Produces a scalar gate value per voice part that modulates between
texture-based separation (for homophonic passages where voices move
together) and pitch-based separation (for polyphonic passages).

Includes formant-band envelope analysis (architecture.md §10.2):
formant-band envelope correlations (200–4000 Hz) with lag tolerance
provide acoustic evidence of voice synchrony/homophony.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FormantBandFeatures(nn.Module):
    """Extract formant-band envelope correlation features from mixture audio.

    Splits the 200–4000 Hz range into sub-bands corresponding to formant
    regions (F1–F4), computes RMS envelopes, and measures pairwise
    cross-correlations with a lag tolerance window.

    High inter-band envelope correlations indicate homophonic texture
    (voices moving together); low correlations indicate polyphonic
    independence.

    Output dimension: n_bands*(n_bands-1)/2 * 2 + n_bands
        = 6 pairs * 2 (zero-lag + peak) + 4 energy ratios = 16 for n_bands=4.
    """

    def __init__(
        self,
        sr: int = 44100,
        n_bands: int = 4,
        n_fft: int = 2048,
        hop_length: int = 512,
        max_lag_ms: float = 50.0,
    ):
        super().__init__()
        self.sr = sr
        self.n_bands = n_bands
        self.n_fft = n_fft
        self.hop_length = hop_length
        # max lag in STFT frames
        self.max_lag_frames = max(1, int(max_lag_ms * sr / 1000 / hop_length))

        # Band edges: 200, 1150, 2100, 3050, 4000 Hz for n_bands=4
        edges = torch.linspace(200.0, 4000.0, n_bands + 1)
        self.register_buffer("band_edges_hz", edges, persistent=False)

        n_pairs = n_bands * (n_bands - 1) // 2
        self.output_dim = n_pairs * 2 + n_bands

    @torch.no_grad()
    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        """Compute formant-band features from mixture audio.

        Args:
            mixture: (B, C, T) or (B, 1, C, T) waveform

        Returns:
            features: (B, output_dim) formant band descriptors
        """
        # Reduce to mono
        if mixture.ndim == 4:
            x = mixture.squeeze(1).mean(dim=1)  # (B, T)
        elif mixture.ndim == 3:
            x = mixture.mean(dim=1)
        else:
            x = mixture

        B, T = x.shape
        device = x.device

        # STFT
        window = torch.hann_window(self.n_fft, device=device, dtype=x.dtype)
        spec = torch.stft(
            x.float(), n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.n_fft, window=window.float(), return_complex=True,
        )
        mag = spec.abs()  # (B, F, T_stft)

        # Frequency axis
        freq_bins = mag.shape[1]
        freqs = torch.linspace(0, self.sr / 2.0, freq_bins, device=device)

        # Per-band RMS envelopes
        band_envs = []
        for i in range(self.n_bands):
            lo = self.band_edges_hz[i]
            hi = self.band_edges_hz[i + 1]
            mask = ((freqs >= lo) & (freqs < hi)).float()  # (F,)
            band_energy = (mag * mask.unsqueeze(0).unsqueeze(-1)).pow(2).sum(dim=1)  # (B, T_stft)
            env = band_energy.sqrt()  # RMS envelope
            band_envs.append(env)

        band_envs = torch.stack(band_envs, dim=1)  # (B, n_bands, T_stft)

        # Normalize each band envelope to zero-mean, unit-std
        env_mean = band_envs.mean(dim=-1, keepdim=True)
        env_std = band_envs.std(dim=-1, keepdim=True).clamp_min(1e-6)
        band_envs_norm = (band_envs - env_mean) / env_std

        # Pairwise cross-correlations (zero-lag + peak within lag tolerance)
        # Using conv1d for efficient lag computation
        T_env = band_envs_norm.shape[-1]
        xcorr_features = []
        for i in range(self.n_bands):
            for j in range(i + 1, self.n_bands):
                a = band_envs_norm[:, i]  # (B, T_env)
                b = band_envs_norm[:, j]  # (B, T_env)

                # Zero-lag correlation
                xcorr_0 = (a * b).mean(dim=-1)  # (B,)

                # Cross-correlation at lags via conv1d:
                # conv1d(a, flip(b)) gives cross-correlation
                # Reshape for conv1d: (B, 1, T)
                a_3d = a.unsqueeze(1)
                # Use each batch item of b as its own kernel
                # More efficient: compute dot products at shifted positions
                peak_xcorr = xcorr_0.clone()
                L = self.max_lag_frames
                for lag in range(1, L + 1):
                    if lag >= T_env:
                        break
                    c_pos = (a[:, lag:] * b[:, :-lag]).mean(dim=-1)
                    c_neg = (a[:, :-lag] * b[:, lag:]).mean(dim=-1)
                    peak_xcorr = torch.maximum(peak_xcorr, torch.maximum(c_pos, c_neg))

                xcorr_features.append(xcorr_0)
                xcorr_features.append(peak_xcorr)

        # Per-band energy ratios (spectral shape descriptor)
        total_energy = band_envs.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-8)
        energy_ratios = (band_envs.pow(2) / total_energy).mean(dim=-1)  # (B, n_bands)

        features = torch.cat([
            torch.stack(xcorr_features, dim=-1),  # (B, n_pairs*2)
            energy_ratios,                          # (B, n_bands)
        ], dim=-1)

        return features


class TextureGate(nn.Module):
    """Texture/synchrony gate from voice similarity, backbone context,
    and formant-band envelope correlations (architecture.md §10.2)."""

    def __init__(
        self,
        num_voices: int = 4,
        input_dim: int = 64,
        formant_feature_dim: int = 0,
    ):
        super().__init__()
        self.num_voices = num_voices
        self.input_dim = input_dim
        self.formant_feature_dim = formant_feature_dim

        # Gate predictor.
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.hidden_proj = nn.LazyLinear(input_dim)
        self.sync_proj = nn.Linear(3, input_dim)

        # Formant-band feature projection (optional)
        if formant_feature_dim > 0:
            self.formant_proj = nn.Sequential(
                nn.Linear(formant_feature_dim, input_dim),
                nn.GELU(),
                nn.LayerNorm(input_dim),
            )

    def forward(
        self,
        voice_features: torch.Tensor | None = None,
        shared_hidden: torch.Tensor | None = None,
        formant_features: torch.Tensor | None = None,
        return_details: bool = False,
    ):
        """
        Args:
            voice_features: (batch, V, input_dim) or None
            shared_hidden: backbone spectral features or None
            formant_features: (batch, formant_feature_dim) global formant
                descriptors from FormantBandFeatures, or None

        Returns:
            gate: (batch, V, 1) values in [0, 1]
            if return_details=True, also returns pairwise gate matrix.
        """
        if voice_features is None:
            batch_size = 1
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            voice_features = torch.zeros(batch_size, self.num_voices, self.input_dim, device=device, dtype=dtype)
        else:
            batch_size = voice_features.shape[0]
            dtype = voice_features.dtype

        gate_features = voice_features

        if shared_hidden is not None:
            hidden_ctx = shared_hidden.mean(dim=1).mean(dim=-1)
            hidden_ctx = self.hidden_proj(hidden_ctx).to(dtype=dtype)
            gate_features = gate_features + hidden_ctx.unsqueeze(1)

        # Formant-band envelope features (§10.2): global texture descriptor
        # broadcast to all voices (same mixture → same formant correlations).
        if formant_features is not None and self.formant_feature_dim > 0:
            formant_embed = self.formant_proj(formant_features.to(dtype=dtype))  # (B, D)
            gate_features = gate_features + formant_embed.unsqueeze(1)  # broadcast to V

        # Voice synchrony descriptors from pairwise similarities.
        if self.num_voices > 1:
            normed = F.normalize(gate_features.float(), dim=-1)
            sim = torch.bmm(normed, normed.transpose(1, 2))  # (B, V, V)
            eye = torch.eye(self.num_voices, device=sim.device, dtype=torch.bool).unsqueeze(0)
            sim_no_diag = sim.masked_fill(eye, 0.0)
            mean_sim = sim_no_diag.sum(dim=-1) / float(self.num_voices - 1)
            var_sim = sim_no_diag.var(dim=-1, unbiased=False)
            peak_sim = sim_no_diag.max(dim=-1).values
            sync_stats = torch.stack([mean_sim, var_sim, peak_sim], dim=-1).to(dtype=dtype)
        else:
            sync_stats = torch.zeros(
                batch_size,
                self.num_voices,
                3,
                device=gate_features.device,
                dtype=dtype,
            )

        gate_features = gate_features + self.sync_proj(sync_stats)

        gate = self.gate_net(gate_features)  # (B, V, 1)
        if not return_details:
            return gate

        gv = gate.squeeze(-1)
        pairwise_gate = torch.minimum(gv.unsqueeze(2), gv.unsqueeze(1))
        details = {
            "pairwise_gate": pairwise_gate,
            "gate_mean": gv.mean(dim=-1),
            "gate_entropy": -(gv.clamp_min(1e-8).log() * gv + (1.0 - gv).clamp_min(1e-8).log() * (1.0 - gv)).mean(dim=-1),
        }
        return gate, details
