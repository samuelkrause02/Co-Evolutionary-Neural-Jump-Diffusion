# Co-Evolutionary Latent Graphs for Neural Jump-Diffusion Processes

**A framework for interacting stochastic processes with discrete jumps on time-varying networks whose topology co-evolves with the system state.**

Research project at ETH Zürich, 2026 — Adrian Martens & Samuel Krause.

> This repository accompanies an ongoing research project. The full implementation is under development; this page documents motivation, formulation, and scope.

---

## Motivation

Many real-world systems involve *interacting stochastic processes with sudden discontinuities on time-varying networks*:

- **Power grids:** renewable in-feed fluctuates stochastically at every bus; line trips are discrete jumps that redistribute flows via Kirchhoff's laws and can cascade.
- **Financial markets:** asset prices follow correlated diffusions punctuated by jumps (defaults, flash crashes); cross-asset contagion intensifies during crises.

Both settings combine three features simultaneously:
1. continuous stochastic node dynamics,
2. discrete jump events propagating through a network,
3. a time-varying interaction structure that co-evolves with the state.

No existing model captures all three. Neural MJD models jumps but treats series independently. LNJSDE adds latent graph structure but fixes it in time. Graph Neural SDEs have graphs but no jumps. This project closes that gap.

## Formulation

Each entity $k \in \{1, \dots, K\}$ has hidden state $z_k(t) \in \mathbb{R}^d$ evolving as

$$
dz_k = \underbrace{f_\theta\!\left(z_k, \sum_j A_{kj}(t)\, z_j, t\right) dt}_{\text{graph-coupled drift}} + \underbrace{g_\phi(z_k, t)\, dW_k}_{\text{diffusion}} + \underbrace{h_\psi(z_k, t^-)\, dN_k}_{\text{jump}}
$$

where $W_k$ is a node-specific Wiener process and $N_k$ a point process with state-dependent intensity $\lambda_k(t)$. The **co-evolutionary adjacency** $A(t)$ evolves jointly with the dynamics in three variants:

- **Variant 1 (deterministic):** $A_{kj}(t) = \mathrm{softmax}_j\!\left(e_\theta(z_k, z_j)\right)$, recomputed every integration step.
- **Variant 2 (stochastic):** adds a learned Gaussian perturbation, capturing uncertainty in coupling strength.
- **Variant 3 (jump-induced):** $A(\tau^+) = A(\tau^-) + \Delta A_\theta(z(\tau^-), \text{event})$; for power grids $\Delta A$ is informed by Power Transfer Distribution Factors.

Training uses the standard TPP log-likelihood with stochastic-adjoint backprop and adjoint jump conditions, extending Neural MJD's likelihood to graph-coupled dynamics.

## Illustrative Applications

**Cascading failures in power grids** (IEEE 39-bus + PowerGraph benchmark). A line trip redistributes flows; a static graph cannot express that a previously non-critical line becomes the most loaded post-trip path. Variant 3 adapts $A$ via PTDFs to correctly identify elevated cascade risk under current topology.

**Cross-asset contagion in finance** (S&P 500, $K = 50$). Correlations empirically spike during crises ("correlations go to one"). Variant 2 learns a state-dependent graph that is sparse in calm regimes and densifies as volatility rises — capturing contagion amplification that static-graph models cannot.

## Positioning

|                      | SDE | Jumps | Graph | Dyn. A | Target      |
|----------------------|:---:|:---:|:---:|:---:|-------------|
| NJ-ODE               | –   | ✓   | –   | –   | States      |
| Neural MJD           | ✓   | ✓   | (✓) | –   | States      |
| LNJSDE               | ✓   | ✓   | ✓   | –   | Intensities |
| Graph Neural SDE     | ✓   | –   | ✓   | –   | Repr.       |
| LaGNA                | ✓   | –   | ✓   | –   | States      |
| **This project**     | ✓   | ✓   | ✓   | ✓   | **Both**    |

## Work Plan (7 months)

- **M1–2:** reproduce LNJSDE and Neural MJD baselines; codebase integration.
- **M3–4:** implement and evaluate co-evolutionary graph variants on TPP benchmarks (Stack Overflow, MIMIC-II, Retweet).
- **M5–6:** power-grid cascading-failure experiments (PowerGraph); cross-asset experiments (S&P 500); theoretical analysis (existence/uniqueness).
- **M7:** ablations, manuscript.

## Tech Stack

`Python` · `PyTorch` · `torchsde` · `torch-geometric` · `NumPy` / `SciPy`

Built on top of the LNJSDE codebase, the `torchsde` library, and the Neural MJD solver.

## Selected References

- Gao et al., *Neural MJD: Neural Non-Stationary Merton Jump Diffusion for Time Series Prediction*, NeurIPS 2025.
- Wang et al., *Learning Neural Jump SDEs with Latent Graph for Multivariate TPPs*, IJCAI 2025.
- Jia & Benson, *Neural Jump Stochastic Differential Equations*, NeurIPS 2019.
- Li, Wong, Chen, Duvenaud, *Scalable Gradients for SDEs*, AISTATS 2020.
- Varbella, Gjorgiev, Sansavini, *PowerGraph*, NeurIPS D&B 2024.

A full bibliography accompanies the project proposal.

## Authors

**Adrian Martens** · ETH Zürich
**Samuel Krause** · ETH Zürich · [sakrause@ethz.ch](mailto:sakrause@ethz.ch)

## Status

Early draft — April 2026.
