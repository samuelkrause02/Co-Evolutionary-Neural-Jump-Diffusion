"""Co-evolutionary adjacency modules for neural jump-diffusion systems."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class DeterministicAdjacency(nn.Module):
    """Variant 1: A_{kj}(t) = softmax_j(e_theta(z_k, z_j))."""

    def __init__(self, hidden_dim: int, edge_dim: int = 64) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, 1),
        )

    def forward(self, z: Tensor) -> Tensor:
        k, d = z.shape
        z_i = z.unsqueeze(1).expand(k, k, d)
        z_j = z.unsqueeze(0).expand(k, k, d)
        logits = self.edge_mlp(torch.cat([z_i, z_j], dim=-1)).squeeze(-1)
        return torch.softmax(logits, dim=-1)


class StochasticAdjacency(nn.Module):
    """Variant 2: adds a learned Gaussian perturbation to the edge logits."""

    def __init__(self, hidden_dim: int, edge_dim: int = 64) -> None:
        super().__init__()
        self.mean_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, edge_dim), nn.SiLU(), nn.Linear(edge_dim, 1)
        )
        self.log_std_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, edge_dim), nn.SiLU(), nn.Linear(edge_dim, 1)
        )

    def forward(self, z: Tensor) -> Tensor:
        k, d = z.shape
        z_i = z.unsqueeze(1).expand(k, k, d)
        z_j = z.unsqueeze(0).expand(k, k, d)
        pair = torch.cat([z_i, z_j], dim=-1)
        mu = self.mean_mlp(pair).squeeze(-1)
        log_std = self.log_std_mlp(pair).squeeze(-1)
        eps = torch.randn_like(mu)
        return torch.softmax(mu + log_std.exp() * eps, dim=-1)
