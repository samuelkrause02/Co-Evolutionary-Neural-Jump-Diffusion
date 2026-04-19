"""Smoke tests for adjacency modules."""
import torch

from cenjd.models.co_evolutionary_graph import (
    DeterministicAdjacency,
    StochasticAdjacency,
)


def test_deterministic_adjacency_is_row_stochastic() -> None:
    torch.manual_seed(0)
    module = DeterministicAdjacency(hidden_dim=8)
    z = torch.randn(5, 8)
    A = module(z)
    assert A.shape == (5, 5)
    assert torch.allclose(A.sum(dim=-1), torch.ones(5), atol=1e-5)
    assert (A >= 0).all()


def test_stochastic_adjacency_respects_shape_and_simplex() -> None:
    torch.manual_seed(0)
    module = StochasticAdjacency(hidden_dim=8)
    z = torch.randn(7, 8)
    A = module(z)
    assert A.shape == (7, 7)
    assert torch.allclose(A.sum(dim=-1), torch.ones(7), atol=1e-5)


def test_stochastic_adjacency_is_not_deterministic() -> None:
    torch.manual_seed(0)
    module = StochasticAdjacency(hidden_dim=8)
    z = torch.randn(4, 8)
    A1 = module(z)
    A2 = module(z)
    assert not torch.allclose(A1, A2)
