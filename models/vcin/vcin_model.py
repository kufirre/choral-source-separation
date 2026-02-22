"""
VCIN (Voice-Coupled Iterative Network) for choral source separation.

Architecture overview:
  1. BSMamba2 backbone produces M=10 over-separated latent sources
  2. Pitch estimator produces per-voice pitch embeddings
  3. Soft assignment maps M latent sources -> V=4 SATB voice parts
  4. Iterative refinement loop (K=3 iterations) progressively improves
  5. Optional modules: repulsion, intent, gate, CGP, FiLM

Skeleton version: backbone -> uniform assignment -> group to SATB -> L1 + multi-STFT loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.vcin.separator import CRMSeparator
from models.vcin.pitch_estimator import PitchEstimator
from models.vcin.assignment import SoftAssignment
from models.vcin.repulsion import RepulsionModule
from models.vcin.intent import IntentModule
from models.vcin.gate import TextureGate
from models.vcin.cgp_adapter import CGPAdapter
from models.vcin.film import FiLMLayer


class VCINModel(nn.Module):
    """
    Voice-Coupled Iterative Network for SATB choral separation.

    Wraps a BSMamba2 backbone with over-separation (M latent sources),
    soft assignment to V voice parts, and iterative refinement.
    """

    def __init__(
        self,
        dim,
        *,
        depth,
        stereo=False,
        num_stems=4,
        num_latent_sources=10,
        num_iterations=3,
        backbone_type='bs_mamba2',
        # BSMamba2 backbone params (passed through)
        time_module_depth=1,
        freq_module_depth=1,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        flash_attn=True,
        stft_n_fft=2048,
        stft_hop_length=512,
        stft_win_length=2048,
        stft_normalized=False,
        mask_estimator_depth=2,
        multi_stft_resolution_loss_weight=1.0,
        multi_stft_resolutions_window_sizes=(4096, 2048, 1024, 512, 256),
        multi_stft_hop_size=147,
        multi_stft_normalized=False,
        module_type='mamba2',
        mamba_gmlp=False,
        # VCIN module enable flags
        enable_repulsion=True,
        enable_pitch=True,
        enable_assignment=True,
        enable_gate=True,
        enable_intent=True,
        enable_cgp=False,
        enable_film=True,
        # VCIN module dimensions
        embed_dim=64,
        **kwargs,
    ):
        super().__init__()

        self.num_stems = num_stems       # V = target voice parts (4 for SATB)
        self.num_latent_sources = num_latent_sources  # M = over-separated sources
        self.num_iterations = num_iterations  # K = refinement iterations
        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.dim = dim
        self.embed_dim = embed_dim

        # Feature flags
        self.enable_repulsion = enable_repulsion
        self.enable_pitch = enable_pitch
        self.enable_assignment = enable_assignment
        self.enable_gate = enable_gate
        self.enable_intent = enable_intent
        self.enable_cgp = enable_cgp
        self.enable_film = enable_film

        # Store multi-STFT loss params (reused from backbone pattern)
        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = torch.hann_window
        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size,
            normalized=multi_stft_normalized,
        )

        # ---- Backbone ----
        # Instantiate BSMamba2 with M latent sources (not V voice parts)
        if backbone_type == 'bs_mamba2':
            from models.bs_mamba2_code.bs_mamba2 import BSMamba2Model
            self.backbone = BSMamba2Model(
                dim=dim,
                depth=depth,
                stereo=stereo,
                num_stems=num_latent_sources,  # M=10, not V=4
                time_module_depth=time_module_depth,
                freq_module_depth=freq_module_depth,
                dim_head=dim_head,
                heads=heads,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                flash_attn=flash_attn,
                stft_n_fft=stft_n_fft,
                stft_hop_length=stft_hop_length,
                stft_win_length=stft_win_length,
                stft_normalized=stft_normalized,
                mask_estimator_depth=mask_estimator_depth,
                multi_stft_resolution_loss_weight=multi_stft_resolution_loss_weight,
                multi_stft_resolutions_window_sizes=multi_stft_resolutions_window_sizes,
                multi_stft_hop_size=multi_stft_hop_size,
                multi_stft_normalized=multi_stft_normalized,
                module_type=module_type,
                mamba_gmlp=mamba_gmlp,
            )
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")

        # ---- CRM Separator (pass-through in skeleton) ----
        self.separator = CRMSeparator(num_latent_sources=num_latent_sources)

        # ---- VCIN Modules ----
        if enable_pitch:
            self.pitch_estimator = PitchEstimator(
                num_voices=num_stems, embed_dim=embed_dim,
            )

        if enable_assignment:
            self.assignment = SoftAssignment(
                num_latent_sources=num_latent_sources,
                num_voices=num_stems,
                embed_dim=embed_dim,
            )
            # Derive latent source embeddings from waveform-domain backbone outputs.
            self.source_embed_proj = nn.Sequential(
                nn.Linear(self.audio_channels, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
            )

        if enable_repulsion:
            self.repulsion = RepulsionModule(
                dim=embed_dim, num_latent_sources=num_latent_sources,
            )

        if enable_intent:
            self.intent = IntentModule(
                num_voices=num_stems, embed_dim=embed_dim,
            )

        if enable_gate:
            self.gate = TextureGate(
                num_voices=num_stems, input_dim=embed_dim,
            )

        if enable_cgp:
            self.cgp_adapter = CGPAdapter(output_dim=embed_dim)

        if enable_film:
            self.film = FiLMLayer(
                feature_dim=embed_dim, conditioning_dim=embed_dim,
            )

    def assign_sources_to_voices(
        self,
        latent_sources: torch.Tensor,
        assignment: torch.Tensor,
    ) -> torch.Tensor:
        """
        Group M latent sources into V voice parts using soft assignment.

        Args:
            latent_sources: (batch, M, channels, time) over-separated sources
            assignment: (batch, M, V) row-stochastic assignment matrix

        Returns:
            voice_estimates: (batch, V, channels, time)
        """
        # assignment: (B, M, V) -> (B, V, M)
        # latent_sources: (B, M, C, T)
        # voice = sum_m A[m,v] * source[m]

        # Reshape for einsum: (B, M, V) x (B, M, C*T) -> (B, V, C*T)
        B, M, C, T = latent_sources.shape
        V = assignment.shape[-1]

        sources_flat = latent_sources.reshape(B, M, C * T)  # (B, M, C*T)
        assignment_t = assignment.transpose(1, 2)  # (B, V, M)
        voices_flat = torch.bmm(assignment_t, sources_flat)  # (B, V, C*T)
        voice_estimates = voices_flat.reshape(B, V, C, T)

        return voice_estimates

    def load_backbone_weights(self, checkpoint_path: str, verbose: bool = True):
        """
        Load pre-trained weights into the backbone (e.g., from a TS-BSMamba2
        vocals checkpoint). Uses partial weight loading for shape mismatches.

        Args:
            checkpoint_path: Path to the checkpoint file.
            verbose: Print loading decisions.
        """
        from utils.model_utils import load_not_compatible_weights

        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        load_not_compatible_weights(self.backbone, ckpt, verbose=verbose)

    def forward(
        self,
        x: torch.Tensor,
        target=None,
        active_stem_ids=None,
        return_loss_breakdown=False,
    ):
        """
        VCIN forward pass.

        Args:
            x: (batch, channels, time) mixture waveform
            target: (batch, V, channels, time) ground truth stems, or None
            active_stem_ids: list of active stem indices (for partial-stem datasets)
            return_loss_breakdown: if True, return (total_loss, (l1, multi_stft))

        Returns:
            If target is None: (batch, V, channels, time) separated voice estimates
            If target is provided: scalar loss (or loss breakdown tuple)
        """
        device = x.device
        raw_audio_length = x.shape[-1]

        # ---- Step 1: Backbone over-separation ----
        # BSMamba2 forward WITHOUT target -> returns separated waveforms
        latent_sources = self.backbone(x)  # (B, M, C, T)

        # ---- Step 2: CRM separator (pass-through in skeleton) ----
        latent_sources = self.separator(latent_sources)

        # ---- Step 3: Pitch estimation (placeholder) ----
        pitch_embed = None
        if self.enable_pitch:
            pitch_embed = self.pitch_estimator(x)  # (B, V, embed_dim)

        # ---- Step 4: Soft assignment ----
        if self.enable_assignment:
            # Build source embeddings from latent sources to avoid uniform-collapse.
            source_embed = latent_sources.mean(dim=-1)  # (B, M, C)
            source_embed = self.source_embed_proj(source_embed)  # (B, M, D)
            assignment = self.assignment(
                source_embeddings=source_embed,
                pitch_embeddings=pitch_embed,
            )  # (B, M, V)
        else:
            # Hard uniform assignment
            B = x.shape[0]
            assignment = torch.ones(
                B, self.num_latent_sources, self.num_stems,
                device=device,
            ) / self.num_stems

        # ---- Step 5: Group latent sources to voice parts ----
        voice_estimates = self.assign_sources_to_voices(
            latent_sources, assignment,
        )  # (B, V, C, T)

        if active_stem_ids is not None and torch.is_tensor(active_stem_ids):
            active_stem_ids = active_stem_ids.flatten().tolist()
        if active_stem_ids is not None:
            active_stem_ids = [
                int(stem_id.item() if torch.is_tensor(stem_id) else stem_id)
                for stem_id in active_stem_ids
            ]

        # If no target, return separated voices
        if target is None:
            if active_stem_ids is not None:
                return voice_estimates[:, active_stem_ids]
            return voice_estimates

        # ---- Loss computation ----
        if target.ndim == 2:
            target = rearrange(target, '... t -> ... 1 t')

        # Trim to match lengths
        target = target[..., :voice_estimates.shape[-1]]

        # Select active stems
        if active_stem_ids is not None:
            voice_sel = voice_estimates[:, active_stem_ids]
            target_sel = target[:, active_stem_ids]
        else:
            voice_sel = voice_estimates
            target_sel = target

        # L1 loss
        loss = F.l1_loss(voice_sel, target_sel)

        # Multi-resolution STFT loss
        multi_stft_resolution_loss = 0.0

        for window_size in self.multi_stft_resolutions_window_sizes:
            res_stft_kwargs = dict(
                n_fft=max(window_size, self.multi_stft_n_fft),
                win_length=window_size,
                return_complex=True,
                window=self.multi_stft_window_fn(window_size, device=device),
                **self.multi_stft_kwargs,
            )

            recon_Y = torch.stft(
                rearrange(voice_sel, 'b n s t -> (b n s) t'),
                **res_stft_kwargs,
            )
            target_Y = torch.stft(
                rearrange(target_sel, 'b n s t -> (b n s) t'),
                **res_stft_kwargs,
            )

            multi_stft_resolution_loss = (
                multi_stft_resolution_loss + F.l1_loss(recon_Y, target_Y)
            )

        weighted_multi_resolution_loss = (
            multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight
        )

        total_loss = loss + weighted_multi_resolution_loss

        # Optional: repulsion loss
        if self.enable_repulsion and hasattr(self, 'repulsion'):
            # In skeleton, repulsion operates on placeholder embeddings
            # This would be source embeddings in the full implementation
            pass

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (loss, multi_stft_resolution_loss)
