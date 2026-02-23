"""
VCIN (Voice-Coupled Iterative Network) for choral source separation.

Architecture-focused implementation:
  - TS-BSMamba2 backbone with over-separation into M latent sources.
  - Exposes shared hidden tensor H from backbone internals.
  - Iterative refinement loop over assignment, pitch, intent, and gate beliefs.
  - Stage-1 (CRM-only) and Stage-2 (residual-refined) separation losses.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

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
        # legacy BSMamba2-compatible args (kept for config compatibility)
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
        # training-time ramps
        cgp_warmup_steps=0,
        repulsion_warmup_steps=0,
        magnitude_penalty_warmup_steps=0,
        # multi-task loss balancing
        use_homoscedastic_weighting=True,
        homoscedastic_init=0.0,
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
            kwargs,
        )

        self.num_stems = int(num_stems)
        self.num_latent_sources = int(num_latent_sources)
        self.num_iterations = int(num_iterations)
        self.stereo = bool(stereo)
        self.audio_channels = 2 if self.stereo else 1
        self.dim = int(dim)
        self.embed_dim = int(embed_dim)
        self.backbone_type = backbone_type

        # feature flags
        self.enable_repulsion = bool(enable_repulsion)
        self.enable_pitch = bool(enable_pitch)
        self.enable_assignment = bool(enable_assignment)
        self.enable_gate = bool(enable_gate)
        self.enable_intent = bool(enable_intent)
        self.enable_cgp = bool(enable_cgp)
        self.enable_film = bool(enable_film)

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

        self.use_homoscedastic_weighting = bool(use_homoscedastic_weighting)
        self.homoscedastic_init = float(homoscedastic_init)

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
        if backbone_type == 'ts_bsmamba2':
            self.backbone = TSBSMamba2Backbone(
                num_latent_sources=self.num_latent_sources,
                sr=int(ts_sr),
                win=int(ts_win),
                stride=int(ts_stride),
                feature_dim=int(ts_feature_dim),
                num_repeat_mask=int(ts_num_repeat_mask),
                num_repeat_map=int(ts_num_repeat_map),
            )
        elif backbone_type == 'bs_mamba2':
            from models.bs_mamba2_code.bs_mamba2 import BSMamba2Model

            self.backbone = BSMamba2Model(
                dim=dim,
                depth=depth,
                stereo=stereo,
                num_stems=self.num_latent_sources,
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

        # ---- Separator (hook point) ----
        self.separator = CRMSeparator(num_latent_sources=self.num_latent_sources)

        # ---- VCIN modules ----
        if self.enable_pitch:
            self.pitch_estimator = PitchEstimator(num_voices=self.num_stems, embed_dim=self.embed_dim)

        if self.enable_assignment:
            self.assignment = SoftAssignment(
                num_latent_sources=self.num_latent_sources,
                num_voices=self.num_stems,
                embed_dim=self.embed_dim,
            )
            self.source_embed_proj = nn.Sequential(
                nn.Linear(self.audio_channels, self.embed_dim),
                nn.GELU(),
                nn.LayerNorm(self.embed_dim),
            )
            self.hidden_to_embed = nn.LazyLinear(self.embed_dim)

        if self.enable_repulsion:
            self.repulsion = RepulsionModule(dim=self.embed_dim, num_latent_sources=self.num_latent_sources)

        if self.enable_intent:
            self.intent = IntentModule(num_voices=self.num_stems, embed_dim=self.embed_dim)

        if self.enable_gate:
            self.gate = TextureGate(num_voices=self.num_stems, input_dim=self.embed_dim)

        if self.enable_cgp:
            self.cgp_adapter = CGPAdapter(
                output_dim=self.embed_dim,
                num_voices=self.num_stems,
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

        if self.backbone_type == 'ts_bsmamba2':
            candidates = [
                ('backbone.separator.', {k[len('backbone.separator.'):]: v for k, v in state.items() if k.startswith('backbone.separator.')}),
                ('separator.', {k[len('separator.'):]: v for k, v in state.items() if k.startswith('separator.')}),
                ('raw', state),
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
                    for k in state.keys()
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
            return

        # backward-compatible path for bs_mamba2 fallback
        from utils.model_utils import load_not_compatible_weights

        load_not_compatible_weights(self.backbone, ckpt, verbose=verbose)

    def _normalize_active_stem_ids(self, active_stem_ids) -> Optional[List[int]]:
        if active_stem_ids is None:
            return None
        if torch.is_tensor(active_stem_ids):
            active_stem_ids = active_stem_ids.flatten().tolist()
        return [int(v.item() if torch.is_tensor(v) else v) for v in active_stem_ids]

    def _prepare_backbone_outputs(self, x: torch.Tensor):
        backbone_aux: Dict[str, torch.Tensor] = {}

        if self.backbone_type == 'ts_bsmamba2':
            latent_sources, backbone_aux = self.backbone(x, return_aux=True)
            stage1_sources = backbone_aux.get('stage1_sources')
            shared_hidden = backbone_aux.get('shared_hidden')
        else:
            latent_sources = self.backbone(x)
            stage1_sources = None
            shared_hidden = None

        latent_sources = self.separator(latent_sources)
        if stage1_sources is not None:
            stage1_sources = self.separator(stage1_sources)

        return latent_sources, stage1_sources, shared_hidden

    def _compute_source_embeddings(
        self,
        latent_sources: torch.Tensor,
        shared_hidden: Optional[torch.Tensor],
    ) -> torch.Tensor:
        source_embed = latent_sources.mean(dim=-1)
        source_embed = self.source_embed_proj(source_embed)

        if shared_hidden is not None:
            # shared_hidden: (B, C, D, T) -> context (B, D)
            hidden_ctx = shared_hidden.mean(dim=1).mean(dim=-1)
            hidden_ctx = self.hidden_to_embed(hidden_ctx)
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

    def _combine_loss_terms(
        self,
        loss_terms: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if len(loss_terms) == 0:
            raise ValueError("loss_terms must contain at least one loss")

        total_loss = next(iter(loss_terms.values())).new_zeros(())
        weighting_info: Dict[str, torch.Tensor] = {}

        if self.use_homoscedastic_weighting:
            for name, loss_val in loss_terms.items():
                if name not in self.loss_log_vars:
                    raise KeyError(f"Missing homoscedastic log-var parameter for loss term '{name}'")
                log_var = self.loss_log_vars[name]
                weighted_term = torch.exp(-log_var) * loss_val + log_var
                total_loss = total_loss + weighted_term
                weighting_info[f'log_var_{name}'] = log_var.detach()
                weighting_info[f'weighted_{name}'] = weighted_term.detach()
        else:
            for name, loss_val in loss_terms.items():
                total_loss = total_loss + loss_val
                weighting_info[f'weighted_{name}'] = loss_val.detach()

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

    def forward(
        self,
        x: torch.Tensor,
        target=None,
        active_stem_ids=None,
        return_loss_breakdown=False,
    ):
        device = x.device

        latent_sources, stage1_sources, shared_hidden = self._prepare_backbone_outputs(x)

        # Initialize row-stochastic assignment with uniform prior.
        bsz = x.shape[0]
        assignment = torch.full(
            (bsz, self.num_latent_sources, self.num_stems),
            1.0 / float(self.num_stems),
            device=device,
            dtype=latent_sources.dtype,
        )

        pitch_state = None
        gate_state = None
        cgp_losses: List[torch.Tensor] = []
        source_embed = self._compute_source_embeddings(latent_sources, shared_hidden)

        for iter_idx in range(self.num_iterations):
            voice_readouts = torch.bmm(assignment.transpose(1, 2), source_embed)

            pitch_embed = None
            if self.enable_pitch:
                pitch_prop = self.pitch_estimator(
                    mixture=x,
                    shared_hidden=shared_hidden,
                    voice_readouts=voice_readouts,
                )
                if pitch_state is None:
                    pitch_state = pitch_prop
                else:
                    zeta = self._get_schedule_value(self.pitch_damping, iter_idx)
                    pitch_state = (1.0 - zeta) * pitch_state + zeta * pitch_prop
                pitch_embed = pitch_state

            intent_embed = None
            if self.enable_intent:
                intent_embed = self.intent(
                    voice_estimates=None,
                    shared_hidden=shared_hidden,
                    voice_readouts=voice_readouts,
                )

            conditioning = voice_readouts
            if pitch_embed is not None:
                conditioning = conditioning + pitch_embed
            if intent_embed is not None:
                conditioning = conditioning + intent_embed

            if self.enable_gate:
                gate_prop = self.gate(voice_features=conditioning, shared_hidden=shared_hidden)
                if gate_state is None:
                    gate_state = gate_prop
                else:
                    beta = self._get_schedule_value(self.gate_damping, iter_idx)
                    gate_state = (1.0 - beta) * gate_state + beta * gate_prop
                conditioning = conditioning * gate_state

            if self.enable_cgp:
                cgp_tokens, cgp_log_prob = self.cgp_adapter(conditioning)
                cgp_losses.append(-cgp_log_prob.mean())
                conditioning = conditioning + cgp_tokens

            if self.enable_film:
                source_embed = self.film(source_embed, conditioning.mean(dim=1))

            if self.enable_repulsion:
                source_embed = self.repulsion(source_embed)

            if self.enable_assignment:
                assignment_prop = self.assignment(
                    source_embeddings=source_embed,
                    pitch_embeddings=pitch_embed,
                )
            else:
                assignment_prop = assignment

            eta = self._get_schedule_value(self.assignment_damping, iter_idx)
            assignment = (1.0 - eta) * assignment + eta * assignment_prop
            assignment = assignment.clamp_min(float(self._eps))
            assignment = assignment / assignment.sum(dim=-1, keepdim=True).clamp_min(float(self._eps))

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
        loss_terms: Dict[str, torch.Tensor] = {}
        if self.stage2_loss_weight > 0:
            loss_terms['stage2'] = self.stage2_loss_weight * stage2_loss

        stage1_loss = voice_sel.new_zeros(())
        l1_stage1 = voice_sel.new_zeros(())
        stft_stage1 = voice_sel.new_zeros(())
        if stage1_sources is not None and self.stage1_loss_weight > 0:
            stage1_voices = self.assign_sources_to_voices(stage1_sources, assignment)
            if active_ids is not None:
                stage1_voices = stage1_voices[:, active_ids]
            stage1_loss, l1_stage1, stft_stage1 = self._compute_reconstruction_losses(stage1_voices, target_sel, device)
            loss_terms['stage1'] = self.stage1_loss_weight * stage1_loss

        min_energy_loss, duplicate_loss = self._latent_regularizers(latent_sources)
        if self.min_energy_loss_weight > 0:
            loss_terms['min_energy'] = self.min_energy_loss_weight * min_energy_loss
        if self.duplicate_loss_weight > 0:
            loss_terms['duplicate'] = self.duplicate_loss_weight * duplicate_loss

        repulsion_loss = voice_sel.new_zeros(())
        repulsion_scale = self.repulsion_loss_weight * self._linear_ramp_multiplier(
            train_step, self.repulsion_warmup_steps
        )
        if self.enable_repulsion and repulsion_scale > 0:
            repulsion_loss = self.repulsion.repulsion_loss(source_embed)
            loss_terms['repulsion'] = repulsion_scale * repulsion_loss

        cgp_loss = voice_sel.new_zeros(())
        cgp_scale = self.cgp_loss_weight * self._linear_ramp_multiplier(
            train_step, self.cgp_warmup_steps
        )
        if self.enable_cgp and len(cgp_losses) > 0:
            cgp_loss = torch.stack(cgp_losses).mean()
            if cgp_scale > 0:
                loss_terms['cgp'] = cgp_scale * cgp_loss

        magnitude_penalty_loss = voice_sel.new_zeros(())
        magnitude_penalty_scale = self.magnitude_penalty_loss_weight * self._linear_ramp_multiplier(
            train_step, self.magnitude_penalty_warmup_steps
        )
        if magnitude_penalty_scale > 0:
            magnitude_penalty_loss = self._compute_magnitude_penalty_loss(voice_sel, target_sel, device)
            loss_terms['magnitude_penalty'] = magnitude_penalty_scale * magnitude_penalty_loss

        if len(loss_terms) == 0:
            raise ValueError("No active loss terms. Check VCIN loss weights/config.")
        total_loss, weighting_info = self._combine_loss_terms(loss_terms)

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
        }
        breakdown.update(weighting_info)
        return total_loss, breakdown
