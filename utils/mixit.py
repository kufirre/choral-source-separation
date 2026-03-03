"""
MixIT / teacher-student helpers for Phase-B adaptation.

These utilities are intentionally model-agnostic and can be used from training
scripts or custom loops without changing baseline supervised behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass
class MixITResult:
    loss: torch.Tensor
    best_groups: torch.Tensor


def _sample_binary_partitions(
    batch: int,
    num_sources: int,
    num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns:
        masks: (batch, num_samples, num_sources), values in {0,1}
    """
    masks = torch.randint(0, 2, (batch, num_samples, num_sources), device=device, dtype=torch.float32)
    # Avoid degenerate all-0 or all-1 masks.
    sum_m = masks.sum(dim=-1, keepdim=True)
    all_zero = (sum_m == 0).float()
    all_one = (sum_m == float(num_sources)).float()
    masks = masks + all_zero
    masks = masks - all_one
    masks = masks.clamp(0.0, 1.0)
    return masks


def mixit_partition_loss(
    latent_sources: torch.Tensor,
    mixture_a: torch.Tensor,
    mixture_b: torch.Tensor,
    *,
    num_partition_samples: int = 64,
) -> MixITResult:
    """
    Approximate MixIT by sampled binary partitions over latent sources.

    Args:
        latent_sources: (B, M, C, T)
        mixture_a: (B, C, T)
        mixture_b: (B, C, T)
    """
    if latent_sources.ndim != 4:
        raise ValueError(f"Expected latent_sources (B, M, C, T), got {tuple(latent_sources.shape)}")
    if mixture_a.shape != mixture_b.shape:
        raise ValueError("mixture_a and mixture_b must have the same shape")

    bsz, num_sources, channels, time = latent_sources.shape
    if mixture_a.shape != (bsz, channels, time):
        raise ValueError(
            f"Expected mixtures shape {(bsz, channels, time)}, got {tuple(mixture_a.shape)}"
        )

    device = latent_sources.device
    masks = _sample_binary_partitions(
        batch=bsz,
        num_sources=num_sources,
        num_samples=int(num_partition_samples),
        device=device,
    )

    sources = latent_sources.unsqueeze(1)  # (B, 1, M, C, T)
    masks_exp = masks.unsqueeze(-1).unsqueeze(-1)  # (B, K, M, 1, 1)
    est_a = (sources * masks_exp).sum(dim=2)  # (B, K, C, T)
    est_b = (sources * (1.0 - masks_exp)).sum(dim=2)

    target_a = mixture_a.unsqueeze(1)
    target_b = mixture_b.unsqueeze(1)
    err = (
        (est_a - target_a).pow(2).mean(dim=(2, 3))
        + (est_b - target_b).pow(2).mean(dim=(2, 3))
    )  # (B, K)

    best_idx = err.argmin(dim=1)  # (B,)
    best_loss = err.gather(1, best_idx.unsqueeze(1)).mean()
    best_groups = masks[torch.arange(bsz, device=device), best_idx]  # (B, M)
    return MixITResult(loss=best_loss, best_groups=best_groups)


def teacher_student_consistency_loss(
    student_pred: torch.Tensor,
    teacher_pred: torch.Tensor,
    *,
    loss_type: str = "l1",
) -> torch.Tensor:
    """
    Consistency loss for teacher-student adaptation.

    Args:
        student_pred: (B, V, C, T)
        teacher_pred: (B, V, C, T)
    """
    if student_pred.shape != teacher_pred.shape:
        raise ValueError("student_pred and teacher_pred must have identical shapes")
    if loss_type == "l2":
        return F.mse_loss(student_pred.float(), teacher_pred.float())
    return F.l1_loss(student_pred.float(), teacher_pred.float())
