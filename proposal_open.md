# Autocurriculum-Driven Continual RL in World Models
# 1. Problem we are trying to solve




## 1. Background

Continual-Dreamer (Kessler et al., 2023) manages experience replay via reservoir sampling and a 50:50 batch split, half uniformly sampled past experience, half recency-biased. This embeds an implicit prior that newer data is more valuable, ignoring that many recent trajectories may be trivially mastered while older episodes from prior tasks may lie at the agent's competence frontier.

Separately, Plan2Explore (P2E) uses ensemble disagreement as intrinsic reward. This conflates *epistemic* uncertainty (learnable) with *aleatoric* uncertainty (irreducible), creating exploration traps in stochastic environments—the agent repeatedly visits noisy regions it can never model better.

IMAC (Güzel et al., 2025) offers a principled alternative: TD-error-based scoring as a proxy for learning potential, naturally tracking the evolving boundary of agent competence.

## 2. Completed Work

We have replicated Continual-Dreamer and upgraded its backbone from DreamerV2 to DreamerV3.

## 3. Proposed Research

### Stage 1: Baseline

Evaluate DreamerV3-based Continual-Dreamer on **Craftax(if time allows)** and **NavIx**, both in symbolic observation mode. This establishes baseline CRL metrics (average performance, forgetting, forward transfer).

### Stage 2: TD-Error Prioritised Replay (Ablation 1, planned)

Replace the 50:50 sampling with learning-potential-based prioritisation. For each episode $e_i$ in the buffer, maintain a cached priority $p_i$ from the IMAC scoring function (aggregated positive TD errors). Sample episodes via:

$$
\Lambda(\tau) = \frac{1}{T} \sum_{t=0}^{T-1} \sum_{k=t}^{T-1} (\gamma \lambda)^{k-t} \max(0, \delta_k)
$$

$$
\delta_k = r_k + \gamma v_\psi(z_{k+1}) - v_\psi(z_k)
$$

$$
P_{cur}(\tau_i) = \frac{|\Lambda(\tau_i)|^\alpha}{\sum_{j \in \mathcal{D}} |\Lambda(\tau_j)|^\alpha}
$$

### Stage 3: Intrinsic Rewards (need more research)

![1770380904966](image/proposal_compact/1770380904966.png)

Using variance of K MLP heads latent states predictions does not align with the openendedness concept. For example in Minecraft, movements of chicken are random, location of generate diamonds are not fixed. Learning how to predict chicken's trajectory and where the diamonds is novel, but is not learnable. So instead of motivate agent to explore on unknown area, we should motivate mode to learn something that is actually learnable, for example how to find the diamonds.
