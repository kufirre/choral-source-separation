"""
VCIN (Voice-Coupled Iterative Network) for choral source separation.

Architecture-focused implementation:
  - TS-BSMamba2 backbone with over-separation into M latent sources.
  - Exposes shared hidden tensor H from backbone internals.
  - Iterative refinement loop over assignment, pitch, intent, and gate beliefs.
  - Stage-1 (CRM-only) and Stage-2 (residual-refined) separation losses.
"""

from __future__ import annotations

import contextlib
import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)

from models.vcin.separator import CRMSeparator
from models.vcin.pitch_estimator import PitchEstimator
from models.vcin.assignment import SoftAssignment
from models.vcin.repulsion import RepulsionModule
from models.vcin.intent import IntentModule
from models.vcin.gate import TextureGate, FormantBandFeatures
from models.vcin.cgp_adapter import CGPAdapter
from models.vcin.film import FiLMLayer
from models.vcin.soft_tokenizer import SoftPitchTokenizer
from models.vcin.voicing_head import VoicingAperiodicityHead
from models.vcin.ts_backbone import TSBSMamba2Backbone


def _as_list(x: Any) -> List[float]:
    if x is None:
        return []
    if isinstance(x, (tuple, list)):
        return [float(v) for v in x]
    return [float(x)]


class VCINModel(nn.Module):
    """
    Voice-Coupled Iterative Network for SATB choral separation.

    Key points:
      - M latent sources (over-separation) grouped into V SATB outputs.
      - Iterative assignment refinement K times.
      - Auxiliary modules consume shared backbone hidden state H.
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
        backbone_type='ts_bsmamba2',
        # training/backbone args accepted via configs
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
        # TS-BSMamba2-specific params
        ts_sr=44100,
        ts_win=2048,
        ts_stride=512,
        ts_feature_dim=128,
        ts_num_repeat_mask=8,
        ts_num_repeat_map=4,
        # iterative damping schedules
        assignment_damping=(0.3, 0.6, 0.9),
        pitch_damping=(0.5, 0.8, 1.0),
        gate_damping=(0.2, 0.4, 0.7),
        # regularization/loss knobs
        stage1_loss_weight=1.0,
        stage2_loss_weight=1.0,
        min_energy_epsilon=1e-4,
        min_energy_loss_weight=0.0,
        duplicate_cos_threshold=0.95,
        duplicate_loss_weight=0.0,
        repulsion_loss_weight=0.0,
        cgp_loss_weight=0.0,
        magnitude_penalty_loss_weight=0.0,
        magnitude_penalty_epsilon=1e-4,
        magnitude_penalty_n_fft=None,
        magnitude_penalty_hop_length=None,
        magnitude_penalty_win_length=None,
        # training-time ramps (loop-breaking controls)
        cgp_warmup_steps=0,
        repulsion_warmup_steps=0,
        magnitude_penalty_warmup_steps=0,
        film_warmup_steps=0,
        detach_conditioning_before_film=False,
        # multi-task loss balancing
        use_homoscedastic_weighting=True,
        homoscedastic_init=0.0,
        assignment_entropy_weight=0.0,
        assignment_capacity_loss_weight=0.0,
        assignment_capacity_targets=None,
        intent_similarity_loss_weight=0.0,
        intent_adversary_loss_weight=0.0,
        intent_gradient_reversal_lambda=0.1,
        gate_entropy_loss_weight=0.0,
        gate_mean_loss_weight=0.0,
        gate_mean_target=0.5,
        pitch_nll_loss_weight=0.0,
        pitch_nll_f0_min_hz=65.0,
        pitch_nll_f0_max_hz=1047.0,
        pitch_consistency_loss_weight=0.0,
        voicing_consistency_loss_weight=0.0,
        voicing_entropy_loss_weight=0.0,
        tail_inertia_weight=0.0,
        # VCIN module enable flags
        enable_repulsion=True,
        enable_pitch=True,
        enable_assignment=True,
        enable_gate=True,
        enable_intent=True,
        enable_voicing_head=True,
        enable_soft_tokenizer=True,
        enable_cgp=False,
        enable_film=True,
        # VCIN module dimensions
        embed_dim=64,
        pitch_num_components=3,
        soft_token_num_bins=128,
        soft_token_temp_start=1.0,
        soft_token_temp_end=0.25,
        soft_token_anneal_steps=20000,
        **kwargs,
    ):
        super().__init__()

        if kwargs:
            unknown_keys = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unsupported VCIN config keys: {unknown_keys}")

        # keep references for logging/config introspection
        _ = (
            depth,
            time_module_depth,
            freq_module_depth,
            dim_head,
            heads,
            attn_dropout,
            ff_dropout,
            flash_attn,
            mask_estimator_depth,
            module_type,
            mamba_gmlp,
        )

        self.num_stems = int(num_stems)
        self.num_latent_sources = int(num_latent_sources)
        self.num_iterations = int(num_iterations)
        self.stereo = bool(stereo)
        self.audio_channels = 2 if self.stereo else 1
        self.dim = int(dim)
        self.embed_dim = int(embed_dim)
        if self.num_iterations < 1:
            raise ValueError(f"num_iterations must be >= 1 for VCIN, got {self.num_iterations}")
        if self.num_latent_sources < self.num_stems:
            raise ValueError(
                "VCIN requires num_latent_sources >= num_stems, "
                f"got {self.num_latent_sources} < {self.num_stems}"
            )
        self.over_separation = self.num_latent_sources > self.num_stems
        if backbone_type != 'ts_bsmamba2':
            raise ValueError(
                f"VCIN only supports backbone_type='ts_bsmamba2' in this codebase, got '{backbone_type}'"
            )

        # feature flags
        self.enable_repulsion = bool(enable_repulsion)
        self.enable_pitch = bool(enable_pitch)
        self.enable_assignment = bool(enable_assignment)
        self.enable_gate = bool(enable_gate)
        self.enable_intent = bool(enable_intent)
        self.enable_voicing_head = bool(enable_voicing_head)
        self.enable_soft_tokenizer = bool(enable_soft_tokenizer)
        self.enable_cgp = bool(enable_cgp)
        self.enable_film = bool(enable_film)
        if not self.enable_assignment:
            raise ValueError("VCIN requires enable_assignment=true; fixed-assignment compatibility paths were removed.")

        # Warn about dead auxiliary modules when M == V (no iterative refinement)
        if not self.over_separation:
            dead_flags = [
                name for name, enabled in [
                    ('enable_pitch', self.enable_pitch),
                    ('enable_gate', self.enable_gate),
                    ('enable_intent', self.enable_intent),
                    ('enable_voicing_head', self.enable_voicing_head),
                    ('enable_repulsion', self.enable_repulsion),
                    ('enable_cgp', self.enable_cgp),
                    ('enable_film', self.enable_film),
                ] if enabled
            ]
            if dead_flags:
                logger.warning(
                    "VCINModel with num_latent_sources == num_stems (no over-separation): "
                    "refinement loop is skipped, so these enabled modules have no effect "
                    "and consume VRAM: %s. Consider using model_type='vcin_direct4' instead.",
                    ', '.join(dead_flags),
                )

        # schedules
        self.assignment_damping = _as_list(assignment_damping) or [1.0]
        self.pitch_damping = _as_list(pitch_damping) or [1.0]
        self.gate_damping = _as_list(gate_damping) or [1.0]

        # loss knobs
        self.stage1_loss_weight = float(stage1_loss_weight)
        self.stage2_loss_weight = float(stage2_loss_weight)
        self.min_energy_epsilon = float(min_energy_epsilon)
        self.min_energy_loss_weight = float(min_energy_loss_weight)
        self.duplicate_cos_threshold = float(duplicate_cos_threshold)
        self.duplicate_loss_weight = float(duplicate_loss_weight)
        self.repulsion_loss_weight = float(repulsion_loss_weight)
        self.cgp_loss_weight = float(cgp_loss_weight)
        self.magnitude_penalty_loss_weight = float(magnitude_penalty_loss_weight)
        self.magnitude_penalty_epsilon = float(magnitude_penalty_epsilon)

        self.cgp_warmup_steps = int(cgp_warmup_steps)
        self.repulsion_warmup_steps = int(repulsion_warmup_steps)
        self.magnitude_penalty_warmup_steps = int(magnitude_penalty_warmup_steps)
        self.film_warmup_steps = int(film_warmup_steps)
        self.detach_conditioning_before_film = bool(detach_conditioning_before_film)

        self.use_homoscedastic_weighting = bool(use_homoscedastic_weighting)
        self.homoscedastic_init = float(homoscedastic_init)
        self.assignment_entropy_weight = float(assignment_entropy_weight)
        self.assignment_capacity_loss_weight = float(assignment_capacity_loss_weight)
        self.intent_similarity_loss_weight = float(intent_similarity_loss_weight)
        self.intent_adversary_loss_weight = float(intent_adversary_loss_weight)
        self.intent_gradient_reversal_lambda = float(intent_gradient_reversal_lambda)
        self.gate_entropy_loss_weight = float(gate_entropy_loss_weight)
        self.gate_mean_loss_weight = float(gate_mean_loss_weight)
        self.gate_mean_target = float(gate_mean_target)
        self.pitch_nll_loss_weight = float(pitch_nll_loss_weight)
        self.pitch_nll_f0_min_hz = float(pitch_nll_f0_min_hz)
        self.pitch_nll_f0_max_hz = float(pitch_nll_f0_max_hz)
        self.pitch_consistency_loss_weight = float(pitch_consistency_loss_weight)
        self.voicing_consistency_loss_weight = float(voicing_consistency_loss_weight)
        self.voicing_entropy_loss_weight = float(voicing_entropy_loss_weight)
        self.tail_inertia_weight = float(tail_inertia_weight)
        self.soft_token_num_bins = int(soft_token_num_bins)
        self.soft_token_temp_start = float(soft_token_temp_start)
        self.soft_token_temp_end = float(soft_token_temp_end)
        self.soft_token_anneal_steps = int(soft_token_anneal_steps)
        if assignment_capacity_targets is None:
            assignment_capacity_targets = tuple([1.0 / float(self.num_stems)] * self.num_stems)
        self.register_buffer(
            "assignment_capacity_targets",
            torch.tensor(list(assignment_capacity_targets), dtype=torch.float32).view(1, self.num_stems),
            persistent=False,
        )

        self.multi_stft_resolution_loss_weight = float(multi_stft_resolution_loss_weight)
        self.multi_stft_resolutions_window_sizes = tuple(int(v) for v in multi_stft_resolutions_window_sizes)
        self.multi_stft_n_fft = int(stft_n_fft)
        self.multi_stft_window_fn = torch.hann_window
        self.multi_stft_kwargs = dict(
            hop_length=int(multi_stft_hop_size),
            normalized=bool(multi_stft_normalized),
        )

        self.magnitude_penalty_n_fft = int(magnitude_penalty_n_fft or stft_n_fft)
        self.magnitude_penalty_hop_length = int(magnitude_penalty_hop_length or stft_hop_length)
        self.magnitude_penalty_win_length = int(magnitude_penalty_win_length or stft_win_length)

        # ---- Backbone ----
        self.ts_feature_dim = int(ts_feature_dim)
        self.backbone = TSBSMamba2Backbone(
            num_latent_sources=self.num_latent_sources,
            sr=int(ts_sr),
            win=int(ts_win),
            stride=int(ts_stride),
            feature_dim=self.ts_feature_dim,
            num_repeat_mask=int(ts_num_repeat_mask),
            num_repeat_map=int(ts_num_repeat_map),
            film_conditioning_dim=self.embed_dim if self.enable_film else 0,
        )

        # ---- Separator (hook point) ----
        self.separator = CRMSeparator(num_latent_sources=self.num_latent_sources)

        # ---- VCIN modules ----
        if self.enable_pitch:
            self.pitch_estimator = PitchEstimator(
                num_voices=self.num_stems,
                embed_dim=self.embed_dim,
                num_components=int(pitch_num_components),
            )

        # Per-source energy statistics (3 features: mean, std, peak)
        self.source_stats_proj = nn.Sequential(
            nn.Linear(3, self.embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.embed_dim),
        )
        # Global context from backbone spectral features
        self.hidden_to_embed = nn.LazyLinear(self.embed_dim)
        self.assignment = SoftAssignment(
            num_latent_sources=self.num_latent_sources,
            num_voices=self.num_stems,
            embed_dim=self.embed_dim,
        )

        if self.enable_repulsion:
            self.repulsion = RepulsionModule(dim=self.embed_dim, num_latent_sources=self.num_latent_sources)

        if self.enable_voicing_head:
            self.voicing_head = VoicingAperiodicityHead(hidden_dim=self.embed_dim)

        if self.enable_intent:
            self.intent = IntentModule(
                num_voices=self.num_stems,
                embed_dim=self.embed_dim,
                gradient_reversal_lambda=self.intent_gradient_reversal_lambda,
            )

        if self.enable_gate:
            self.formant_analyzer = FormantBandFeatures(
                sr=int(ts_sr),
                n_fft=int(ts_win),
                hop_length=int(ts_stride),
            )
            self.gate = TextureGate(
                num_voices=self.num_stems,
                input_dim=self.embed_dim,
                formant_feature_dim=self.formant_analyzer.output_dim,
            )

        if self.enable_cgp:
            self.cgp_adapter = CGPAdapter(
                output_dim=self.embed_dim,
                num_voices=self.num_stems,
            )
            if self.enable_pitch and self.enable_soft_tokenizer:
                self.soft_tokenizer = SoftPitchTokenizer(
                    num_tokens=self.soft_token_num_bins,
                    embed_dim=self.embed_dim,
                    temperature_start=self.soft_token_temp_start,
                    temperature_end=self.soft_token_temp_end,
                    anneal_steps=self.soft_token_anneal_steps,
                )

        if self.enable_film:
            self.film = FiLMLayer(feature_dim=self.embed_dim, conditioning_dim=self.embed_dim)

        self._loss_term_names = (
            'stage2',
            'stage1',
            'min_energy',
            'duplicate',
            'repulsion',
            'cgp',
            'magnitude_penalty',
            'assignment_entropy',
            'assignment_capacity',
            'intent_similarity',
            'intent_adversary',
            'pitch_nll',
            'gate_entropy',
            'gate_mean',
            'pitch_consistency',
            'voicing_consistency',
            'voicing_entropy',
        )
        if self.use_homoscedastic_weighting:
            self.loss_log_vars = nn.ParameterDict(
                {
                    name: nn.Parameter(torch.tensor(self.homoscedastic_init, dtype=torch.float32))
                    for name in self._loss_term_names
                }
            )

        self.register_buffer('_eps', torch.tensor(1e-8), persistent=False)
        self.register_buffer('_train_step', torch.tensor(0, dtype=torch.long), persistent=False)

    @staticmethod
    def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
        if isinstance(ckpt, dict):
            if 'model_state_dict' in ckpt and isinstance(ckpt['model_state_dict'], dict):
                return ckpt['model_state_dict']
            if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
                return ckpt['state_dict']
            if 'state' in ckpt and isinstance(ckpt['state'], dict):
                return ckpt['state']
            if all(torch.is_tensor(v) for v in ckpt.values()):
                return ckpt
        raise ValueError('Checkpoint does not contain a recognizable state_dict')

    def load_backbone_weights(
        self,
        checkpoint_source: Union[str, Dict[str, Any]],
        checkpoint_path: Optional[str] = None,
        verbose: bool = True,
    ):
        if isinstance(checkpoint_source, str):
            ckpt = torch.load(checkpoint_source, map_location='cpu', weights_only=False)
            checkpoint_label = checkpoint_source
        else:
            ckpt = checkpoint_source
            checkpoint_label = checkpoint_path if checkpoint_path is not None else '<in-memory checkpoint>'

        state = self._extract_state_dict(ckpt)
        # Normalize common wrapper prefixes from DDP/EMA checkpoints.
        normalized_state = {}
        for key, value in state.items():
            norm_key = key
            for prefix in ('module.', 'model.'):
                if norm_key.startswith(prefix):
                    norm_key = norm_key[len(prefix):]
            normalized_state[norm_key] = value
        candidates = [
            (
                'backbone.separator.',
                {
                    k[len('backbone.separator.'):]: v
                    for k, v in normalized_state.items()
                    if k.startswith('backbone.separator.')
                },
            ),
            (
                'separator.',
                {
                    k[len('separator.'):]: v
                    for k, v in normalized_state.items()
                    if k.startswith('separator.')
                },
            ),
            ('raw', normalized_state),
        ]
        chosen = None
        for _, cand in candidates:
            if len(cand) == 0:
                continue
            if any(k.startswith('BN_mask.') or k.startswith('separator_mask.') for k in cand.keys()):
                chosen = cand
                break
        if chosen is None:
            if any(
                k.startswith('band_split.')
                or k.startswith('layers.')
                or k.startswith('mask_estimators.')
                for k in normalized_state.keys()
            ):
                raise ValueError(
                    'Checkpoint appears to contain BSMamba2Model weights, '
                    'which are incompatible with the TS-BSMamba2 Separator backbone.'
                )
            raise ValueError('No TS-BSMamba2 separator keys found in checkpoint')

        missing, unexpected = self.backbone.load_ts_state_dict(chosen)
        loaded_count = len(chosen) - len(unexpected)
        if verbose:
            print(
                f"Loaded TS-BSMamba2 backbone from {checkpoint_label}: "
                f"loaded={loaded_count}, missing={len(missing)}, unexpected={len(unexpected)}"
            )
        if loaded_count <= 0:
            raise ValueError('Backbone load matched zero parameters')

    def _normalize_active_stem_ids(self, active_stem_ids) -> Optional[List[int]]:
        if active_stem_ids is None:
            return None
        if torch.is_tensor(active_stem_ids):
            active_stem_ids = active_stem_ids.flatten().tolist()
        return [int(v.item() if torch.is_tensor(v) else v) for v in active_stem_ids]

    def _prepare_backbone_outputs(self, x: torch.Tensor):
        backbone_aux: Dict[str, torch.Tensor] = {}
        latent_sources, backbone_aux = self.backbone(x, return_aux=True)
        stage1_sources = backbone_aux.get('stage1_sources')
        shared_hidden = backbone_aux.get('shared_hidden')
        encoder_cache = backbone_aux.get('encoder_cache')

        latent_sources = self.separator(latent_sources)
        if stage1_sources is not None:
            stage1_sources = self.separator(stage1_sources)

        return latent_sources, stage1_sources, shared_hidden, encoder_cache

    def _compute_source_embeddings(
        self,
        latent_sources: torch.Tensor,
        shared_hidden: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute per-source embeddings from energy statistics + backbone context.

        Uses RMS energy envelope statistics (mean, std, peak) per source,
        which are more informative than the previous channel-mean approach.
        """
        B, M, C, T = latent_sources.shape

        # Per-source: RMS energy envelope statistics (cheap, informative)
        src_rms = latent_sources.float().pow(2).mean(dim=2)  # (B, M, T)
        src_stats = torch.stack([
            src_rms.mean(dim=-1),              # mean energy
            src_rms.std(dim=-1),               # energy variability
            src_rms.max(dim=-1).values,        # peak energy
        ], dim=-1)  # (B, M, 3)

        source_embed = self.source_stats_proj(src_stats.to(latent_sources.dtype))  # (B, M, embed_dim)

        # Global context from backbone spectral features
        if shared_hidden is not None:
            # shared_hidden: (B, nch, D, T) -> pool to (B, D)
            hidden_ctx = shared_hidden.mean(dim=1).mean(dim=-1)
            hidden_ctx = self.hidden_to_embed(hidden_ctx)  # (B, embed_dim)
            source_embed = source_embed + hidden_ctx.unsqueeze(1)

        return source_embed

    def _get_schedule_value(self, schedule: List[float], iter_idx: int) -> float:
        if not schedule:
            return 1.0
        if iter_idx < len(schedule):
            return float(schedule[iter_idx])
        return float(schedule[-1])

    @staticmethod
    def _linear_ramp_multiplier(step: int, warmup_steps: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return float(min(1.0, step / float(max(warmup_steps, 1))))

    @staticmethod
    def _autocast_off(device: torch.device):
        """Disable AMP autocast so spectral ops run in float32."""
        if device.type == "cuda":
            return torch.autocast(device_type="cuda", enabled=False)
        return contextlib.nullcontext()

    def _check_finite(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """Log a warning if a loss term is non-finite. Returns the value unchanged."""
        if not torch.isfinite(value).all():
            step = int(self._train_step.item())
            value_for_log = value.detach().float()
            if value_for_log.numel() > 1:
                value_for_log = value_for_log.mean()
            logger.warning(
                "Non-finite detected in loss term '%s' at train_step=%d, value=%s",
                name, step, value_for_log.cpu().item(),
            )
        return value

    def _combine_loss_terms(
        self,
        loss_terms: Dict[str, torch.Tensor],
        loss_weights: Optional[Dict[str, float]] = None,
        loss_ramps: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Combine loss terms with optional homoscedastic uncertainty weighting.

        Args:
            loss_terms: dict of **raw** (unscaled) loss values.
            loss_weights: dict of per-term config weights (base_weight * ramp).
                Used in the non-homoscedastic branch.
            loss_ramps: dict of per-term warmup ramp multipliers in [0, 1].
                Used in the homoscedastic branch to gate loss terms during
                warmup.  The learned log-variance handles relative balancing
                among active terms; the ramp controls curriculum.
        """
        if len(loss_terms) == 0:
            raise ValueError("loss_terms must contain at least one loss")

        total_loss = next(iter(loss_terms.values())).new_zeros(())
        weighting_info: Dict[str, torch.Tensor] = {}

        if self.use_homoscedastic_weighting:
            for name, loss_val in loss_terms.items():
                if name not in self.loss_log_vars:
                    raise KeyError(f"Missing homoscedastic log-var parameter for loss term '{name}'")
                raw_log_var = self.loss_log_vars[name]
                log_var = torch.clamp(raw_log_var, min=-6.0, max=6.0)
                # Homoscedastic: ramp * (exp(-s)*L + s)
                # ramp gates the entire contribution (including regularizer)
                # so log_var is not pulled during warmup when the loss is off.
                ramp = loss_ramps.get(name, 1.0) if loss_ramps else 1.0
                weighted_term = ramp * (torch.exp(-log_var) * loss_val + log_var)
                total_loss = total_loss + weighted_term
                weighting_info[f'log_var_{name}'] = raw_log_var.detach()
                weighting_info[f'log_var_{name}_clamped'] = log_var.detach()
                weighting_info[f'eff_weight_{name}'] = (ramp * torch.exp(-log_var)).detach()
                weighting_info[f'weighted_{name}'] = weighted_term.detach()
                # Alert when log_var hits the clamp boundary — a loss term is
                # being silently killed (max=6 → eff_weight≈0.002) or dominated
                # (min=-6 → eff_weight≈403).
                raw_val = raw_log_var.item()
                if raw_val <= -5.9 or raw_val >= 5.9:
                    step = int(self._train_step.item())
                    logger.warning(
                        "Homoscedastic log_var for '%s' near clamp boundary "
                        "(raw=%.2f, eff_weight=%.4f) at step %d",
                        name, raw_val, (ramp * torch.exp(-log_var)).item(), step,
                    )
        else:
            for name, loss_val in loss_terms.items():
                w = loss_weights.get(name, 1.0) if loss_weights else 1.0
                scaled = w * loss_val
                total_loss = total_loss + scaled
                weighting_info[f'weighted_{name}'] = scaled.detach()

        return total_loss, weighting_info

    def assign_sources_to_voices(
        self,
        latent_sources: torch.Tensor,
        assignment: torch.Tensor,
    ) -> torch.Tensor:
        """Group M latent sources into V voice parts using soft assignment."""
        bsz, num_sources, channels, timesteps = latent_sources.shape
        num_voices = assignment.shape[-1]

        sources_flat = latent_sources.reshape(bsz, num_sources, channels * timesteps)
        assignment_t = assignment.transpose(1, 2)
        voices_flat = torch.bmm(assignment_t, sources_flat)
        return voices_flat.reshape(bsz, num_voices, channels, timesteps)

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

    def _compute_magnitude_penalty_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        # L_MP discourages energy leakage from stem v into TF bins owned by u != v.
        pred32 = pred.float()
        target32 = target.float()

        stft_kwargs = dict(
            n_fft=self.magnitude_penalty_n_fft,
            hop_length=self.magnitude_penalty_hop_length,
            win_length=self.magnitude_penalty_win_length,
            return_complex=True,
            normalized=False,
            window=self.multi_stft_window_fn(
                self.magnitude_penalty_win_length,
                device=device,
                dtype=torch.float32,
            ),
        )

        with self._autocast_off(device):
            pred_spec = torch.stft(rearrange(pred32, 'b n s t -> (b n s) t'), **stft_kwargs)
            target_spec = torch.stft(rearrange(target32, 'b n s t -> (b n s) t'), **stft_kwargs)
        pred_mag = pred_spec.abs()
        target_mag = target_spec.abs()

        bsz, num_voices, channels = pred32.shape[:3]
        pred_mag = rearrange(pred_mag, '(b n s) f l -> b n s f l', b=bsz, n=num_voices, s=channels)
        target_mag = rearrange(target_mag, '(b n s) f l -> b n s f l', b=bsz, n=num_voices, s=channels)

        active_mask = (target_mag > self.magnitude_penalty_epsilon).to(dtype=pred_mag.dtype)
        interferer_mask = active_mask.sum(dim=1, keepdim=True) - active_mask
        interferer_mask = interferer_mask.clamp_min(0.0)

        return (pred_mag * interferer_mask).mean()

    def _latent_regularizers(self, latent_sources: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # minimum latent energy penalty
        energy = latent_sources.float().pow(2).mean(dim=(2, 3))
        min_energy_loss = F.relu(self.min_energy_epsilon - energy).mean()

        # duplicate-source cosine penalty
        src = latent_sources.float().reshape(latent_sources.shape[0], latent_sources.shape[1], -1)
        src = F.normalize(src, dim=-1)
        sim = torch.bmm(src, src.transpose(1, 2))
        eye = torch.eye(sim.shape[1], device=sim.device, dtype=torch.bool).unsqueeze(0)
        off_diag = sim.masked_fill(eye, 0.0)
        dup_loss = F.relu(off_diag - self.duplicate_cos_threshold).pow(2).mean()

        return min_energy_loss, dup_loss

    def _compute_source_tail_strength(self, latent_sources: torch.Tensor) -> torch.Tensor:
        """
        Estimate per-source tail/noise strength for assignment inertia.

        Returns:
            tail: (B, M, 1) in [0, 1]
        """
        x = latent_sources.float().mean(dim=2)  # (B, M, T)
        abs_mean = x.abs().mean(dim=-1)
        zcr = (x[..., 1:] * x[..., :-1] < 0).float().mean(dim=-1)
        diff_energy = (x[..., 1:] - x[..., :-1]).pow(2).mean(dim=-1).sqrt()
        tail = torch.sigmoid(2.0 * zcr + diff_energy - abs_mean)
        return tail.unsqueeze(-1).to(dtype=latent_sources.dtype)

    def _compute_assignment_capacity_loss(self, assignment: torch.Tensor) -> torch.Tensor:
        """
        Capacity prior over assignment mass per voice.
        """
        # assignment: (B, M, V), row-stochastic over V
        cap = assignment.sum(dim=1) / float(max(assignment.shape[1], 1))  # (B, V)
        target = self.assignment_capacity_targets.to(device=assignment.device, dtype=assignment.dtype)
        return (cap - target).pow(2).mean()

    def _compute_intent_adversary_loss(
        self,
        intent_details: Optional[Dict[str, torch.Tensor]],
        pitch_details: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Anti-pitch-leakage adversary loss (architecture.md §10.1).

        The adversary tries to predict expected pitch from gradient-reversed
        intent embeddings.  A high adversary loss means intent still encodes
        pitch information.  Because the gradients are reversed through
        _GradientReversal, minimising this loss in the adversary head
        simultaneously *maximises* the pitch prediction error from intent's
        perspective, pushing intent to be pitch-orthogonal.
        """
        if intent_details is None or pitch_details is None:
            return torch.tensor(0.0, device=self._eps.device)

        adversary_pred = intent_details.get("adversary_pitch_pred")
        expected_pitch = pitch_details.get("expected_pitch")
        if adversary_pred is None or expected_pitch is None:
            return torch.tensor(0.0, device=self._eps.device)

        # expected_pitch is (B, V) in [0, 1] from PitchEstimator;
        # adversary_pred is (B, V) in [0, 1] from Sigmoid.
        target = expected_pitch.float().detach()
        pred = adversary_pred.float()
        return F.mse_loss(pred, target)

    def _compute_intent_similarity_loss(
        self,
        intent_embed: Optional[torch.Tensor],
        pairwise_gate: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if intent_embed is None:
            return torch.tensor(0.0, device=self._eps.device)
        normed = F.normalize(intent_embed.float(), dim=-1)
        sim = torch.bmm(normed, normed.transpose(1, 2))  # (B, V, V)
        eye = torch.eye(sim.shape[1], device=sim.device, dtype=torch.bool).unsqueeze(0)
        off_diag = 1.0 - sim
        off_diag = off_diag.masked_fill(eye, 0.0)
        if pairwise_gate is not None:
            off_diag = off_diag * pairwise_gate.float()
        return off_diag.mean().to(dtype=intent_embed.dtype)

    def _compute_gate_regularization(
        self,
        gate: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if gate is None:
            z = torch.tensor(0.0, device=self._eps.device)
            return z, z
        g = gate.float().squeeze(-1)
        eps = 1e-8
        entropy = -(g * torch.log(g.clamp_min(eps)) + (1.0 - g) * torch.log((1.0 - g).clamp_min(eps))).mean()
        mean_penalty = (g.mean(dim=-1) - float(self.gate_mean_target)).pow(2).mean()
        return entropy.to(dtype=gate.dtype), mean_penalty.to(dtype=gate.dtype)

    def _normalize_f0_to_unit(self, f0_hz: torch.Tensor) -> torch.Tensor:
        """Normalize f0 in Hz to [0, 1] to match MoG means (sigmoid space).

        Uses log-frequency scaling: maps [f0_min, f0_max] Hz linearly in
        log-space to [0, 1].  Unvoiced frames (f0 <= 0) are left as-is.
        """
        f0_min = max(self.pitch_nll_f0_min_hz, 1.0)
        f0_max = max(self.pitch_nll_f0_max_hz, f0_min + 1.0)
        log_min = math.log(f0_min)
        log_max = math.log(f0_max)
        # Only transform voiced (f0 > 0)
        voiced = (f0_hz > 0).float()
        safe_f0 = f0_hz.clamp(min=f0_min)
        normed = (safe_f0.log() - log_min) / (log_max - log_min)
        normed = normed.clamp(0.0, 1.0)
        return normed * voiced  # unvoiced stays 0

    def _compute_pitch_nll_loss(
        self,
        pitch_details: Optional[Dict[str, torch.Tensor]],
        f0_target: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """MoG negative log-likelihood supervision (architecture.md §6).

        For voiced frames (f0 > 0):
            P(f0 | voiced) = (1 - rest) * Σ_j w_j * N(f0_norm; μ_j, σ_j²)
        For unvoiced frames (f0 = 0):
            P(unvoiced) = rest

        Loss = -mean(log P) over all voices and batch items.

        Args:
            pitch_details: dict with 'mog_weights', 'mog_means', 'mog_log_scales',
                'rest_prob' from PitchEstimator.
            f0_target: (B, V) per-voice f0 in Hz; 0 means unvoiced.
        """
        if pitch_details is None or f0_target is None:
            return torch.tensor(0.0, device=self._eps.device)

        mog_weights = pitch_details.get("mog_weights")      # (B, V, J)
        mog_means = pitch_details.get("mog_means")          # (B, V, J)
        mog_log_scales = pitch_details.get("mog_log_scales")  # (B, V, J)
        rest_prob = pitch_details.get("rest_prob")           # (B, V)
        if mog_weights is None or mog_means is None or mog_log_scales is None or rest_prob is None:
            return torch.tensor(0.0, device=self._eps.device)

        # Normalize f0 targets to [0, 1] to match MoG mean space
        f0_norm = self._normalize_f0_to_unit(f0_target.float())  # (B, V)
        voiced_mask = (f0_target > 0).float()  # (B, V)

        # Gaussian log-likelihood per component: log N(f0; μ, σ²)
        scales = mog_log_scales.float().exp().clamp_min(1e-6)  # (B, V, J)
        diff = f0_norm.unsqueeze(-1) - mog_means.float()  # (B, V, J)
        log_gauss = -0.5 * (diff / scales).pow(2) - scales.log() - 0.5 * math.log(2 * math.pi)

        # Log mixture: log Σ_j w_j * N(f0; μ_j, σ_j²)
        log_mixture = torch.logsumexp(
            mog_weights.float().clamp_min(1e-8).log() + log_gauss, dim=-1
        )  # (B, V)

        # Full log-likelihood per voice:
        #   voiced: log[(1 - rest) * mixture_density]
        #   unvoiced: log[rest]
        eps = 1e-8
        log_voiced = torch.log((1.0 - rest_prob.float()).clamp_min(eps)) + log_mixture
        log_unvoiced = torch.log(rest_prob.float().clamp_min(eps))

        log_prob = voiced_mask * log_voiced + (1.0 - voiced_mask) * log_unvoiced

        return -log_prob.mean()

    def _compute_pitch_consistency_loss(
        self,
        pitch_details: Optional[Dict[str, torch.Tensor]],
        voice_readouts: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if pitch_details is None or voice_readouts is None:
            return torch.tensor(0.0, device=self._eps.device)
        expected_pitch = pitch_details.get("expected_pitch")
        if expected_pitch is None:
            return torch.tensor(0.0, device=self._eps.device)
        proxy_pitch = torch.sigmoid(voice_readouts.float().mean(dim=-1))
        return F.l1_loss(proxy_pitch, expected_pitch.float()).to(dtype=voice_readouts.dtype)

    def _compute_voicing_consistency_loss(
        self,
        assignment: Optional[torch.Tensor],
        source_voicing: Optional[torch.Tensor],
        pitch_details: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if assignment is None or source_voicing is None or pitch_details is None:
            return torch.tensor(0.0, device=self._eps.device)

        rest_prob = pitch_details.get("rest_prob")
        if rest_prob is None:
            return torch.tensor(0.0, device=self._eps.device)

        assignment_f = assignment.float()
        assign_mass = assignment_f.transpose(1, 2).sum(dim=-1, keepdim=True).clamp_min(1e-6)  # (B, V, 1)
        voiced_from_sources = torch.bmm(assignment_f.transpose(1, 2), source_voicing.float()) / assign_mass
        voiced_prior = (1.0 - rest_prob.float()).unsqueeze(-1)
        return F.l1_loss(voiced_from_sources, voiced_prior).to(dtype=assignment.dtype)

    def forward(
        self,
        x: torch.Tensor,
        target=None,
        active_stem_ids=None,
        f0_target=None,
        return_loss_breakdown=False,
    ):
        device = x.device

        latent_sources, stage1_sources, shared_hidden, encoder_cache = self._prepare_backbone_outputs(x)

        bsz = x.shape[0]
        cgp_losses: List[torch.Tensor] = []
        source_embed = None
        assignment = None
        final_voice_readouts = None
        final_pitch_details: Optional[Dict[str, torch.Tensor]] = None
        final_intent_embed: Optional[torch.Tensor] = None
        final_gate_state: Optional[torch.Tensor] = None
        final_pairwise_gate: Optional[torch.Tensor] = None
        final_source_voicing: Optional[torch.Tensor] = None
        final_source_noise: Optional[torch.Tensor] = None
        final_voicing_entropy: Optional[torch.Tensor] = None
        final_intent_details: Optional[Dict[str, torch.Tensor]] = None

        # Assignment initialization:
        # - over-separation (M>V): uniform row-stochastic prior.
        # - direct routing (M==V): strict identity mapping for baseline purity.
        if self.over_separation:
            assignment = torch.full(
                (bsz, self.num_latent_sources, self.num_stems),
                1.0 / float(self.num_stems),
                device=device,
                dtype=latent_sources.dtype,
            )
        else:
            assignment = torch.eye(
                self.num_stems,
                device=device,
                dtype=latent_sources.dtype,
            ).unsqueeze(0).expand(bsz, -1, -1).contiguous()

        pitch_state = None
        gate_state = None
        source_voicing = None
        source_noise = None
        source_tail = None
        if self.enable_voicing_head:
            voicing_out = self.voicing_head(
                latent_sources=latent_sources,
                shared_hidden=shared_hidden,
                return_details=True,
            )
            source_voicing, source_noise, source_tail, voicing_details = voicing_out
            final_source_voicing = source_voicing
            final_source_noise = source_noise
            final_voicing_entropy = voicing_details.get("voicing_entropy")
        if source_tail is None:
            source_tail = self._compute_source_tail_strength(latent_sources)
        tail_for_assignment = self.tail_inertia_weight * source_tail if self.tail_inertia_weight > 0 else None
        source_embed = self._compute_source_embeddings(latent_sources, shared_hidden)
        token_step = int(self._train_step.item())

        # Formant-band envelope features (§10.2) — computed once from mixture.
        formant_features = None
        if self.enable_gate and hasattr(self, 'formant_analyzer'):
            formant_features = self.formant_analyzer(x)

        refinement_iters = self.num_iterations if self.over_separation else 0
        for iter_idx in range(refinement_iters):
            voice_readouts = torch.bmm(assignment.transpose(1, 2), source_embed)
            final_voice_readouts = voice_readouts

            pitch_embed = None
            voice_voicing_prior = None
            voice_noise_prior = None
            cgp_token_probs = None
            if self.enable_pitch:
                pitch_out = self.pitch_estimator(
                    mixture=x,
                    shared_hidden=shared_hidden,
                    voice_readouts=voice_readouts,
                    return_details=True,
                )
                pitch_prop, pitch_details = pitch_out
                if pitch_state is None:
                    pitch_state = pitch_prop
                else:
                    zeta = self._get_schedule_value(self.pitch_damping, iter_idx)
                    pitch_state = (1.0 - zeta) * pitch_state + zeta * pitch_prop
                pitch_embed = pitch_state
                final_pitch_details = pitch_details
                if "rest_prob" in pitch_details:
                    rest_prob = pitch_details["rest_prob"]
                    voice_voicing_prior = (1.0 - rest_prob).unsqueeze(-1).to(dtype=latent_sources.dtype)
                    voice_noise_prior = rest_prob.unsqueeze(-1).to(dtype=latent_sources.dtype)

            intent_embed = None
            if self.enable_intent:
                intent_out = self.intent(
                    voice_estimates=None,
                    shared_hidden=shared_hidden,
                    voice_readouts=voice_readouts,
                    return_details=True,
                )
                intent_embed, intent_details = intent_out
                final_intent_embed = intent_embed
                final_intent_details = intent_details

            conditioning = voice_readouts
            if pitch_embed is not None:
                conditioning = conditioning + pitch_embed
            if intent_embed is not None:
                conditioning = conditioning + intent_embed

            if self.enable_gate:
                gate_out = self.gate(
                    voice_features=conditioning,
                    shared_hidden=shared_hidden,
                    formant_features=formant_features,
                    return_details=True,
                )
                gate_prop, gate_details = gate_out
                if gate_state is None:
                    gate_state = gate_prop
                else:
                    beta = self._get_schedule_value(self.gate_damping, iter_idx)
                    gate_state = (1.0 - beta) * gate_state + beta * gate_prop
                conditioning = conditioning * gate_state
                final_gate_state = gate_state
                final_pairwise_gate = gate_details.get("pairwise_gate")

            if self.enable_cgp:
                soft_tokens = None
                if (
                    self.enable_soft_tokenizer
                    and self.enable_pitch
                    and final_pitch_details is not None
                    and hasattr(self, "soft_tokenizer")
                ):
                    soft_token_out = self.soft_tokenizer(
                        final_pitch_details,
                        train_step=token_step,
                        return_details=False,
                    )
                    soft_tokens, cgp_token_probs = soft_token_out
                cgp_input = conditioning if soft_tokens is None else (conditioning + soft_tokens)
                cgp_tokens, cgp_log_prob = self.cgp_adapter(
                    cgp_input,
                    token_probs=cgp_token_probs,
                )
                cgp_losses.append(-cgp_log_prob.mean())
                conditioning = conditioning + cgp_tokens

            if self.enable_film:
                # Loop-breaking: optionally delay FiLM activation and detach
                # conditioning to prevent unstable feedback early in training.
                film_ramp = self._linear_ramp_multiplier(
                    int(self._train_step.item()), self.film_warmup_steps
                )
                if film_ramp > 0:
                    film_cond = conditioning
                    if self.detach_conditioning_before_film and self.training:
                        film_cond = film_cond.detach()
                    film_cond_pooled = film_cond.mean(dim=1)  # (B, D)
                    source_embed = self.film(source_embed, film_cond_pooled)

                    # Re-run backbone mask/map heads with updated FiLM conditioning
                    # to get refined latent sources (architecture.md §4.1).
                    if encoder_cache is not None:
                        remasked_output, remasked_stage1 = self.backbone.forward_heads_only(
                            encoder_cache, film_conditioning=film_cond_pooled,
                        )
                        latent_sources = self.separator(remasked_output)
                        if remasked_stage1 is not None:
                            stage1_sources = self.separator(remasked_stage1)
                        # Recompute source embeddings from updated sources.
                        source_embed = self._compute_source_embeddings(latent_sources, shared_hidden)

            if self.enable_repulsion:
                source_embed = self.repulsion(source_embed)

            assignment_out = self.assignment(
                source_embeddings=source_embed,
                pitch_embeddings=pitch_embed,
                intent_embeddings=intent_embed,
                gate_values=gate_state,
                prev_assignment=assignment,
                source_tail=tail_for_assignment,
                source_voicing_probs=source_voicing,
                source_noise_probs=source_noise,
                voice_voicing_prior=voice_voicing_prior,
                voice_noise_prior=voice_noise_prior,
                pitch_details=final_pitch_details,
                return_details=True,
            )
            assignment_prop, assignment_info = assignment_out
            _ = assignment_info

            eta = self._get_schedule_value(self.assignment_damping, iter_idx)
            assignment = (1.0 - eta) * assignment + eta * assignment_prop
            assignment = assignment.clamp_min(float(self._eps))
            assignment = assignment / assignment.sum(dim=-1, keepdim=True).clamp_min(float(self._eps))

        if final_voice_readouts is None:
            final_voice_readouts = torch.bmm(assignment.transpose(1, 2), source_embed)

        voice_estimates = self.assign_sources_to_voices(latent_sources, assignment)

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

        if self.training:
            self._train_step += 1
        train_step = int(self._train_step.item())

        stage2_loss, l1_stage2, stft_stage2 = self._compute_reconstruction_losses(voice_sel, target_sel, device)
        self._check_finite('stage2_l1', l1_stage2)
        self._check_finite('stage2_stft', stft_stage2)
        loss_terms: Dict[str, torch.Tensor] = {}
        loss_weights: Dict[str, float] = {}
        # loss_ramps: warmup multipliers in [0,1] for the homoscedastic branch.
        # Learned log-variance handles relative importance; ramps handle curriculum.
        loss_ramps: Dict[str, float] = {}

        if self.stage2_loss_weight > 0:
            loss_terms['stage2'] = stage2_loss
            loss_weights['stage2'] = self.stage2_loss_weight
            loss_ramps['stage2'] = 1.0

        stage1_loss = voice_sel.new_zeros(())
        l1_stage1 = voice_sel.new_zeros(())
        stft_stage1 = voice_sel.new_zeros(())
        if stage1_sources is not None and self.stage1_loss_weight > 0:
            stage1_voices = self.assign_sources_to_voices(stage1_sources, assignment)
            if active_ids is not None:
                stage1_voices = stage1_voices[:, active_ids]
            stage1_loss, l1_stage1, stft_stage1 = self._compute_reconstruction_losses(stage1_voices, target_sel, device)
            self._check_finite('stage1_l1', l1_stage1)
            self._check_finite('stage1_stft', stft_stage1)
            loss_terms['stage1'] = stage1_loss
            loss_weights['stage1'] = self.stage1_loss_weight
            loss_ramps['stage1'] = 1.0

        # Latent regularizers only apply to over-separation (M > V)
        min_energy_loss = voice_sel.new_zeros(())
        duplicate_loss = voice_sel.new_zeros(())
        if self.over_separation:
            min_energy_loss, duplicate_loss = self._latent_regularizers(latent_sources)
            if self.min_energy_loss_weight > 0:
                loss_terms['min_energy'] = min_energy_loss
                loss_weights['min_energy'] = self.min_energy_loss_weight
                loss_ramps['min_energy'] = 1.0
            if self.duplicate_loss_weight > 0:
                loss_terms['duplicate'] = duplicate_loss
                loss_weights['duplicate'] = self.duplicate_loss_weight
                loss_ramps['duplicate'] = 1.0

        repulsion_loss = voice_sel.new_zeros(())
        repulsion_ramp = self._linear_ramp_multiplier(train_step, self.repulsion_warmup_steps)
        repulsion_scale = self.repulsion_loss_weight * repulsion_ramp
        if self.enable_repulsion and repulsion_ramp > 0 and source_embed is not None:
            repulsion_loss = self.repulsion.repulsion_loss(source_embed)
            self._check_finite('repulsion', repulsion_loss)
            loss_terms['repulsion'] = repulsion_loss
            loss_weights['repulsion'] = repulsion_scale
            loss_ramps['repulsion'] = repulsion_ramp

        cgp_loss = voice_sel.new_zeros(())
        cgp_ramp = self._linear_ramp_multiplier(train_step, self.cgp_warmup_steps)
        cgp_scale = self.cgp_loss_weight * cgp_ramp
        if self.enable_cgp and cgp_ramp > 0 and len(cgp_losses) > 0:
            cgp_loss = torch.stack(cgp_losses).mean()
            loss_terms['cgp'] = cgp_loss
            loss_weights['cgp'] = cgp_scale
            loss_ramps['cgp'] = cgp_ramp

        magnitude_penalty_loss = voice_sel.new_zeros(())
        mp_ramp = self._linear_ramp_multiplier(train_step, self.magnitude_penalty_warmup_steps)
        magnitude_penalty_scale = self.magnitude_penalty_loss_weight * mp_ramp
        if mp_ramp > 0 and self.magnitude_penalty_loss_weight > 0:
            magnitude_penalty_loss = self._compute_magnitude_penalty_loss(voice_sel, target_sel, device)
            self._check_finite('magnitude_penalty', magnitude_penalty_loss)
            loss_terms['magnitude_penalty'] = magnitude_penalty_loss
            loss_weights['magnitude_penalty'] = magnitude_penalty_scale
            loss_ramps['magnitude_penalty'] = mp_ramp

        # Assignment entropy regularization (M > V soft-assignment path)
        assignment_entropy_loss = voice_sel.new_zeros(())
        if self.assignment_entropy_weight > 0 and assignment is not None:
            eps = 1e-8
            # Encourage low-entropy (peaky) assignments per source
            assignment_entropy_loss = -(assignment * (assignment + eps).log()).sum(dim=-1).mean()
            self._check_finite('assignment_entropy', assignment_entropy_loss)
            loss_terms['assignment_entropy'] = assignment_entropy_loss
            loss_weights['assignment_entropy'] = self.assignment_entropy_weight
            loss_ramps['assignment_entropy'] = 1.0

        assignment_capacity_loss = voice_sel.new_zeros(())
        if self.assignment_capacity_loss_weight > 0 and assignment is not None:
            assignment_capacity_loss = self._compute_assignment_capacity_loss(assignment)
            self._check_finite('assignment_capacity', assignment_capacity_loss)
            loss_terms['assignment_capacity'] = assignment_capacity_loss
            loss_weights['assignment_capacity'] = self.assignment_capacity_loss_weight
            loss_ramps['assignment_capacity'] = 1.0

        intent_similarity_loss = voice_sel.new_zeros(())
        if self.intent_similarity_loss_weight > 0 and final_intent_embed is not None:
            intent_similarity_loss = self._compute_intent_similarity_loss(
                intent_embed=final_intent_embed,
                pairwise_gate=final_pairwise_gate,
            )
            self._check_finite('intent_similarity', intent_similarity_loss)
            loss_terms['intent_similarity'] = intent_similarity_loss
            loss_weights['intent_similarity'] = self.intent_similarity_loss_weight
            loss_ramps['intent_similarity'] = 1.0

        intent_adversary_loss = voice_sel.new_zeros(())
        if self.intent_adversary_loss_weight > 0 and final_intent_details is not None:
            intent_adversary_loss = self._compute_intent_adversary_loss(
                intent_details=final_intent_details,
                pitch_details=final_pitch_details,
            )
            self._check_finite('intent_adversary', intent_adversary_loss)
            loss_terms['intent_adversary'] = intent_adversary_loss
            loss_weights['intent_adversary'] = self.intent_adversary_loss_weight
            loss_ramps['intent_adversary'] = 1.0

        gate_entropy_loss = voice_sel.new_zeros(())
        gate_mean_loss = voice_sel.new_zeros(())
        if final_gate_state is not None:
            gate_entropy_loss, gate_mean_loss = self._compute_gate_regularization(final_gate_state)
            if self.gate_entropy_loss_weight > 0:
                self._check_finite('gate_entropy', gate_entropy_loss)
                loss_terms['gate_entropy'] = gate_entropy_loss
                loss_weights['gate_entropy'] = self.gate_entropy_loss_weight
                loss_ramps['gate_entropy'] = 1.0
            if self.gate_mean_loss_weight > 0:
                self._check_finite('gate_mean', gate_mean_loss)
                loss_terms['gate_mean'] = gate_mean_loss
                loss_weights['gate_mean'] = self.gate_mean_loss_weight
                loss_ramps['gate_mean'] = 1.0

        pitch_nll_loss = voice_sel.new_zeros(())
        if self.pitch_nll_loss_weight > 0 and f0_target is not None:
            pitch_nll_loss = self._compute_pitch_nll_loss(
                pitch_details=final_pitch_details,
                f0_target=f0_target,
            )
            self._check_finite('pitch_nll', pitch_nll_loss)
            loss_terms['pitch_nll'] = pitch_nll_loss
            loss_weights['pitch_nll'] = self.pitch_nll_loss_weight
            loss_ramps['pitch_nll'] = 1.0

        pitch_consistency_loss = voice_sel.new_zeros(())
        if self.pitch_consistency_loss_weight > 0:
            pitch_consistency_loss = self._compute_pitch_consistency_loss(
                pitch_details=final_pitch_details,
                voice_readouts=final_voice_readouts,
            )
            self._check_finite('pitch_consistency', pitch_consistency_loss)
            loss_terms['pitch_consistency'] = pitch_consistency_loss
            loss_weights['pitch_consistency'] = self.pitch_consistency_loss_weight
            loss_ramps['pitch_consistency'] = 1.0

        voicing_consistency_loss = voice_sel.new_zeros(())
        if self.voicing_consistency_loss_weight > 0:
            voicing_consistency_loss = self._compute_voicing_consistency_loss(
                assignment=assignment,
                source_voicing=final_source_voicing,
                pitch_details=final_pitch_details,
            )
            self._check_finite('voicing_consistency', voicing_consistency_loss)
            loss_terms['voicing_consistency'] = voicing_consistency_loss
            loss_weights['voicing_consistency'] = self.voicing_consistency_loss_weight
            loss_ramps['voicing_consistency'] = 1.0

        voicing_entropy_loss = voice_sel.new_zeros(())
        if self.voicing_entropy_loss_weight > 0 and final_voicing_entropy is not None:
            voicing_entropy_loss = final_voicing_entropy.mean().to(dtype=voice_sel.dtype)
            self._check_finite('voicing_entropy', voicing_entropy_loss)
            loss_terms['voicing_entropy'] = voicing_entropy_loss
            loss_weights['voicing_entropy'] = self.voicing_entropy_loss_weight
            loss_ramps['voicing_entropy'] = 1.0

        if len(loss_terms) == 0:
            raise ValueError("No active loss terms. Check VCIN loss weights/config.")
        total_loss, weighting_info = self._combine_loss_terms(loss_terms, loss_weights, loss_ramps)
        self._check_finite('total_loss', total_loss)

        if not return_loss_breakdown:
            return total_loss

        breakdown = {
            'stage2_l1': l1_stage2,
            'stage2_stft': stft_stage2,
            'stage1_l1': l1_stage1,
            'stage1_stft': stft_stage1,
            'min_energy': min_energy_loss,
            'duplicate': duplicate_loss,
            'repulsion': repulsion_loss,
            'cgp': cgp_loss,
            'magnitude_penalty': magnitude_penalty_loss,
            'assignment_entropy': assignment_entropy_loss,
            'assignment_capacity': assignment_capacity_loss,
            'intent_similarity': intent_similarity_loss,
            'intent_adversary': intent_adversary_loss,
            'pitch_nll': pitch_nll_loss,
            'gate_entropy': gate_entropy_loss,
            'gate_mean': gate_mean_loss,
            'pitch_consistency': pitch_consistency_loss,
            'voicing_consistency': voicing_consistency_loss,
            'voicing_entropy': voicing_entropy_loss,
        }
        breakdown.update(weighting_info)
        return total_loss, breakdown
