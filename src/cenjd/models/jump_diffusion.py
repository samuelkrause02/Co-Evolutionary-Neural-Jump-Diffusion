"""Coupled jump-diffusion dynamics on a co-evolutionary latent graph."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class SDEConfig:
    dt: float = 0.05
    t_end: float = 10.0


class DriftNet(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, z: Tensor, neighborhood: Tensor, t: Tensor) -> Tensor:
        t_feat = t.expand(z.shape[0], 1)
        return self.net(torch.cat([z, neighborhood, t_feat], dim=-1))


class DiffusionNet(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
        )

    def forward(self, z: Tensor, t: Tensor) -> Tensor:
        t_feat = t.expand(z.shape[0], 1)
        return self.net(torch.cat([z, t_feat], dim=-1))


class IntensityNet(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, z: Tensor, t: Tensor) -> Tensor:
        t_feat = t.expand(z.shape[0], 1)
        return self.net(torch.cat([z, t_feat], dim=-1)).squeeze(-1)


class CoEvolutionaryJumpDiffusion(nn.Module):
    """End-to-end model: co-evolutionary graph + graph-coupled jump-diffusion SDE.

    Implements the Euler-Maruyama discretisation of equation (1) in the
    project proposal, with thinning-based jump sampling at each step.
    """

    def __init__(
        self,
        adjacency: nn.Module,
        hidden_dim: int,
        jump_magnitude: float = 0.5,
    ) -> None:
        super().__init__()
        self.adjacency = adjacency
        self.drift = DriftNet(hidden_dim)
        self.diffusion = DiffusionNet(hidden_dim)
        self.intensity = IntensityNet(hidden_dim)
        self.jump_magnitude = jump_magnitude

    def step(self, z: Tensor, t: Tensor, dt: float) -> tuple[Tensor, Tensor]:
        A = self.adjacency(z)
        neighborhood = A @ z
        drift = self.drift(z, neighborhood, t)
        diffusion = self.diffusion(z, t)
        dW = torch.randn_like(z) * dt**0.5

        lam = self.intensity(z, t)
        jump_mask = (torch.rand_like(lam) < lam * dt).float().unsqueeze(-1)
        jump = jump_mask * self.jump_magnitude * torch.randn_like(z)

        z_next = z + drift * dt + diffusion * dW + jump
        return z_next, jump_mask.squeeze(-1)

    def rollout(self, z0: Tensor, config: SDEConfig) -> tuple[Tensor, Tensor]:
        steps = int(config.t_end / config.dt)
        trajectory = [z0]
        jumps = []
        z = z0
        for i in range(steps):
            t = torch.tensor(i * config.dt, device=z.device)
            z, jumped = self.step(z, t, config.dt)
            trajectory.append(z)
            jumps.append(jumped)
        return torch.stack(trajectory), torch.stack(jumps)
