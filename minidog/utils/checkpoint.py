"""Checkpoint save/load utilities for Stage 1 and Stage 2 training."""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger('minidog')


def save_stage2_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> None:
    """Save Stage 2 training checkpoint."""
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.module.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_stage2_checkpoint(
    path: str,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    """Load Stage 2 training checkpoint. Returns (epoch, step)."""
    checkpoint = torch.load(path, map_location="cpu")
    model.module.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


def _drop_absent_projector(state_dict: dict, model: torch.nn.Module, path: str) -> dict:
    """Drop the alignment projector when the target model does not have one.

    Fine-tuning with repa.use_repa = false builds a model with no repa_projector, while
    the pretrained checkpoint still carries one. The projector is only ever used to
    compute the alignment loss, so a run that does not align has no use for those
    weights. We drop exactly those keys rather than loading non-strictly, so every
    other mismatch still raises.
    """
    if any(k.startswith("repa_projector.") for k in model.state_dict()):
        return state_dict
    extra = [k for k in state_dict if k.startswith("repa_projector.")]
    if extra:
        logger.info(
            f"Dropping {len(extra)} alignment-projector tensor(s) from {path}: "
            "this run does not use representation alignment."
        )
        state_dict = {k: v for k, v in state_dict.items() if k not in extra}
    return state_dict


def load_stage2_weights_only(
    path: str,
    model: DDP,
    ema_model: torch.nn.Module,
) -> None:
    """Load only model + EMA weights from a Stage 2 checkpoint.

    Skips optimizer/scheduler/epoch/step, for warm-starting a fine-tune with a fresh
    optimizer/scheduler/epoch-counter — e.g. when the new run's steps-per-epoch is too
    different from the checkpoint's original run for its saved scheduler state to make sense.
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.module.load_state_dict(_drop_absent_projector(checkpoint["model"], model.module, path))
    ema_model.load_state_dict(_drop_absent_projector(checkpoint["ema"], ema_model, path))


__all__ = [
    "save_stage1_checkpoint",
    "load_stage1_checkpoint",
    "save_stage2_checkpoint",
    "load_stage2_checkpoint",
    "load_stage2_weights_only",
]
