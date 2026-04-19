"""Training loop for co-evolutionary neural jump-diffusion models."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    best_loss: float = float("inf")


def mjd_negative_log_likelihood(
    trajectory: Tensor,
    observations: Tensor,
    jump_mask: Tensor,
    intensity: Tensor,
    dt: float,
) -> Tensor:
    """Approximate MJD negative log-likelihood with thinning-based jump term.

    Diffusion part: Gaussian increment log-density under the Euler step.
    Jump part: Poisson point-process log-likelihood over the grid.
    """
    residual = observations - trajectory[:-1]
    diffusion_ll = -0.5 * (residual.pow(2) / dt).sum(dim=-1).mean()
    jump_ll = (jump_mask * intensity.log() - intensity * dt).sum(dim=-1).mean()
    return -(diffusion_ll + jump_ll)


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    optimizer: Optimizer,
    loss_fn: Callable[..., Tensor],
    grad_clip: float | None = 1.0,
) -> float:
    model.train()
    total, count = 0.0, 0
    for z0, obs in loader:
        optimizer.zero_grad()
        trajectory, jump_mask = model.rollout(z0, model.config)  # type: ignore[attr-defined]
        intensity = model.intensity(trajectory[:-1], torch.tensor(0.0))
        loss = loss_fn(trajectory, obs, jump_mask, intensity, dt=model.config.dt)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
        count += 1
    return total / max(count, 1)
