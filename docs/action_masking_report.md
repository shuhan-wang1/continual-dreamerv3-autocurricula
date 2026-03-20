# Action Masking for DreamerV3 in Craftax: Technical Report

## 1. Overview

This report describes the action masking system integrated into DreamerV3 for the Craftax environment. The system constrains the agent's policy to avoid selecting **infeasible actions** — actions whose preconditions are not satisfied in the current game state (e.g., crafting a pickaxe without raw materials).

The masking operates at two levels:

1. **Real environment interaction**: The mask is computed from **ground-truth** game state and applied to the policy before action sampling.
2. **Imagination (dreaming)**: A learned prediction head estimates the game state context from RSSM latent features, and the predicted mask is applied during world-model rollouts.

Two masking modes are supported:

| Mode | Mechanism | Gradient flow |
|------|-----------|---------------|
| **Soft** | Proportional logit penalty $-\lambda \cdot d_a$ | Smooth; deficit-dependent |
| **Hard** | Binary logit penalty $-L \cdot \mathbb{1}[d_a > 0]$ | Near-zero probability for infeasible actions |

where $d_a$ is the deficit for action $a$, $\lambda$ is the penalty coefficient, and $L$ is a large constant ($10^9$).

---

## 2. Context Vector

### 2.1 Definition

The masking system operates on a **context vector** $\mathbf{c} \in \mathbb{R}^{44}$, a compact summary of the Craftax game state extracted at each environment step. The context captures four categories of information:

$$
\mathbf{c} = \left[\underbrace{c_0, \ldots, c_{13}}_{\text{inventory}},\; \underbrace{c_{14}, \ldots, c_{17}}_{\text{armour}},\; \underbrace{c_{18}, \ldots, c_{23}}_{\text{potions}},\; \underbrace{c_{24}, \ldots, c_{28}}_{\text{player state}},\; \underbrace{c_{29}, \ldots, c_{31}}_{\text{attributes}},\; \underbrace{c_{32}, c_{33}}_{\text{spells}},\; \underbrace{c_{34}, \ldots, c_{43}}_{\text{spatial/level}}\right]
$$

### 2.2 Context Indices

| Index | Symbol | Description | Type |
|-------|--------|-------------|------|
| 0 | `WOOD` | Wood count | Integer |
| 1 | `STONE` | Stone count | Integer |
| 2 | `COAL` | Coal count | Integer |
| 3 | `IRON` | Iron ore count | Integer |
| 4 | `DIAMOND` | Diamond count | Integer |
| 5 | `SAPLING` | Sapling count | Integer |
| 6 | `PICKAXE` | Pickaxe tier (0–4) | Integer |
| 7 | `SWORD` | Sword tier (0–4) | Integer |
| 8 | `BOW` | Bow count | Integer |
| 9 | `ARROWS` | Arrow count | Integer |
| 10 | `TORCHES` | Torch count | Integer |
| 11 | `BOOKS` | Book count | Integer |
| 12 | `RUBY` | Ruby count | Integer |
| 13 | `SAPPHIRE` | Sapphire count | Integer |
| 14–17 | `ARMOUR_0`–`ARMOUR_3` | Armour piece tiers | Integer |
| 18–23 | `POTION_0`–`POTION_5` | Potion counts per type | Integer |
| 24 | `HEALTH` | Player health | Float |
| 25 | `ENERGY` | Player energy | Float |
| 26 | `MANA` | Player mana | Float |
| 27 | `XP` | Experience points | Float |
| 28 | `LEVEL` | Current dungeon level | Integer |
| 29 | `DEXTERITY` | Dexterity attribute (1–5) | Integer |
| 30 | `STRENGTH` | Strength attribute (1–5) | Integer |
| 31 | `INTELLIGENCE` | Intelligence attribute (1–5) | Integer |
| 32 | `SPELL_FIREBALL` | Fireball spell learned | Binary |
| 33 | `SPELL_ICEBALL` | Iceball spell learned | Binary |
| 34 | `NEAR_TABLE` | Adjacent to crafting table | Binary |
| 35 | `NEAR_FURNACE` | Adjacent to furnace | Binary |
| 36 | `FACING_PLACEABLE` | Facing empty, non-solid cell | Binary |
| 37 | `FACING_GRASS` | Facing grass block | Binary |
| 38 | `FACING_ENCHANT_TABLE` | Facing enchantment table | Binary |
| 39 | `FACING_TORCH_PLACEABLE` | Facing torch-placeable surface | Binary |
| 40 | `ON_LADDER_DOWN` | Standing on downward ladder | Binary |
| 41 | `ON_LADDER_UP` | Standing on upward ladder | Binary |
| 42 | `LEVEL_CLEARED` | Current level monsters cleared ($\geq 8$ killed) | Binary |
| 43 | `PROJECTILE_SLOTS` | Active projectiles $< 3$ | Binary |

### 2.3 Context Extraction

The context is extracted from the Craftax `EnvState` object at each environment step via `extract_mask_context(state)`. Spatial features (indices 34–43) require map queries:

- **Proximity** (indices 34–35): 8-cell adjacency check (cardinal + diagonal neighbours) for crafting table (block ID 11) and furnace (block ID 12).
- **Facing block** (indices 36–39): The block at position $(p_r + \Delta r, \, p_c + \Delta c)$ is inspected, where $(\Delta r, \Delta c)$ is the player's facing direction.
- **Ladder** (indices 40–41): Item map at the player's position is checked for ladder items.
- **Level cleared** (index 42): `monsters_killed[level] >= 8`.
- **Projectile slots** (index 43): `active_projectiles < 3`.

---

## 3. Feasibility Rules

### 3.1 Rule Structure

Each maskable action $a$ is associated with a **rule** $R_a$ consisting of a set of conditions $\{C_1^{(a)}, \ldots, C_{K_a}^{(a)}\}$. An action is **feasible** if and only if all conditions are satisfied:

$$
\text{feasible}(a) = \bigwedge_{k=1}^{K_a} C_k^{(a)}(\mathbf{c})
$$

### 3.2 Condition Operators

Each condition $C_k$ is defined by an operator and produces a scalar **deficit** $\delta_k \geq 0$, where $\delta_k = 0$ indicates satisfaction:

| Operator | Semantics | Deficit $\delta_k$ |
|----------|-----------|---------------------|
| $\texttt{min}(i, v)$ | $c_i \geq v$ | $\max(0,\; v - c_i)$ |
| $\texttt{below}(i, v)$ | $c_i < v$ | $\mathbb{1}[c_i \geq v]$ |
| $\texttt{bool}(i)$ | $c_i > 0.5$ | $\mathbb{1}[c_i \leq 0.5]$ |
| $\texttt{any\_below}(i_1, i_2, v)$ | $\exists\, j \in [i_1, i_2): c_j < v$ | $\mathbb{1}\!\left[\forall\, j: c_j \geq v\right]$ |
| $\texttt{sum\_pos}(i_1, i_2)$ | $\sum_{j=i_1}^{i_2-1} c_j > 0$ | $\mathbb{1}\!\left[\sum_j c_j \leq 0\right]$ |
| $\texttt{max\_energy\_check}$ | $c_{\text{energy}} < 7 + 2 c_{\text{dex}}$ | $\mathbb{1}[c_{\text{energy}} \geq 7 + 2 c_{\text{dex}}]$ |
| $\texttt{max\_health\_check}$ | $c_{\text{health}} < 8 + c_{\text{str}}$ | $\mathbb{1}[c_{\text{health}} \geq 8 + c_{\text{str}}]$ |
| $\texttt{gem\_check}$ | $c_{\text{ruby}} \geq 1 \lor c_{\text{sapphire}} \geq 1$ | $\mathbb{1}[c_{\text{ruby}} < 1 \land c_{\text{sapphire}} < 1]$ |
| $\texttt{attr\_below\_max}(i)$ | $c_i < 5$ | $\mathbb{1}[c_i \geq 5]$ |

The **total deficit** for action $a$ is the sum over all conditions:

$$
d_a = \sum_{k=1}^{K_a} \delta_k^{(a)}
$$

An action $a$ is **infeasible** when $d_a > 0$.

### 3.3 Complete Action Rule Table

The system defines rules for **37 out of 42** discrete actions (actions 0–5 are never masked: NOOP, LEFT, RIGHT, UP, DOWN, DO).

| Action | ID | Conditions (abbreviated) |
|--------|----|--------------------------|
| SLEEP | 6 | energy $<$ max_energy |
| PLACE_STONE | 7 | stone $\geq 1$, facing placeable |
| PLACE_TABLE | 8 | wood $\geq 2$, facing placeable |
| PLACE_FURNACE | 9 | stone $\geq 1$, facing placeable |
| PLACE_PLANT | 10 | sapling $\geq 1$, facing grass |
| MAKE_WOOD_PICKAXE | 11 | wood $\geq 1$, near table, pickaxe $< 1$ |
| MAKE_STONE_PICKAXE | 12 | wood $\geq 1$, stone $\geq 1$, near table, pickaxe $< 2$ |
| MAKE_IRON_PICKAXE | 13 | wood, stone, iron, coal $\geq 1$, near table+furnace, pickaxe $< 3$ |
| MAKE_WOOD_SWORD | 14 | wood $\geq 1$, near table, sword $< 1$ |
| MAKE_STONE_SWORD | 15 | wood, stone $\geq 1$, near table, sword $< 2$ |
| MAKE_IRON_SWORD | 16 | wood, stone, iron, coal $\geq 1$, near table+furnace, sword $< 3$ |
| REST | 17 | health $<$ max_health |
| DESCEND | 18 | on ladder down, level cleared, level $< 8$ |
| ASCEND | 19 | on ladder up, level $\geq 1$ |
| MAKE_DIAMOND_PICKAXE | 20 | wood $\geq 1$, diamond $\geq 3$, near table, pickaxe $< 4$ |
| MAKE_DIAMOND_SWORD | 21 | wood $\geq 1$, diamond $\geq 2$, near table, sword $< 4$ |
| MAKE_IRON_ARMOUR | 22 | iron $\geq 3$, coal $\geq 3$, near table+furnace, any armour slot $< 1$ |
| MAKE_DIAMOND_ARMOUR | 23 | diamond $\geq 3$, near table, any armour slot $< 2$ |
| SHOOT_ARROW | 24 | bow $\geq 1$, arrows $\geq 1$, projectile slots available |
| MAKE_ARROW | 25 | wood, stone $\geq 1$, near table, arrows $< 99$ |
| CAST_FIREBALL | 26 | mana $\geq 2$, fireball learned, projectile slots available |
| CAST_ICEBALL | 27 | mana $\geq 2$, iceball learned, projectile slots available |
| PLACE_TORCH | 28 | torches $\geq 1$, facing torch-placeable |
| DRINK_POTION_RED | 29 | potion[0] $\geq 1$ |
| DRINK_POTION_GREEN | 30 | potion[1] $\geq 1$ |
| DRINK_POTION_BLUE | 31 | potion[2] $\geq 1$ |
| DRINK_POTION_PINK | 32 | potion[3] $\geq 1$ |
| DRINK_POTION_CYAN | 33 | potion[4] $\geq 1$ |
| DRINK_POTION_YELLOW | 34 | potion[5] $\geq 1$ |
| READ_BOOK | 35 | books $\geq 1$ |
| ENCHANT_SWORD | 36 | facing enchant table, mana $\geq 9$, gem available, sword $\geq 1$ |
| ENCHANT_ARMOUR | 37 | facing enchant table, mana $\geq 9$, gem available, armour sum $> 0$ |
| MAKE_TORCH | 38 | wood, coal $\geq 1$, near table, torches $< 99$ |
| LEVEL_UP_DEX | 39 | xp $\geq 1$, dexterity $< 5$ |
| LEVEL_UP_STR | 40 | xp $\geq 1$, strength $< 5$ |
| LEVEL_UP_INT | 41 | xp $\geq 1$, intelligence $< 5$ |
| ENCHANT_BOW | 42 | facing enchant table, mana $\geq 9$, gem available, bow $\geq 1$ |

---

## 4. Logit Bias Computation

### 4.1 Bias Formula

Given a context vector $\mathbf{c}$ and the policy network's raw logits $\ell_a$ for each action $a \in \{0, \ldots, N-1\}$, the adjusted logits $\tilde{\ell}_a$ are:

$$
\tilde{\ell}_a = \ell_a + b_a(\mathbf{c})
$$

where $b_a$ is the **mask bias** for action $a$. The bias is computed differently depending on the mode:

**Soft mode** ($\text{mode} = \texttt{soft}$):

$$
b_a = -\lambda \cdot d_a
$$

where $\lambda > 0$ is the penalty coefficient (default $\lambda = 5.0$) and $d_a$ is the total deficit.

**Hard mode** ($\text{mode} = \texttt{hard}$):

$$
b_a = -L \cdot \mathbb{1}[d_a > 0]
$$

where $L$ is a large constant (default $L = 10^9$).

**No masking** ($\text{mode} = \texttt{none}$):

$$
b_a = 0
$$

### 4.2 Effect on Action Probabilities

The policy distribution after masking is:

$$
\pi(a \mid s) = \frac{\exp(\tilde{\ell}_a)}{\sum_{a'} \exp(\tilde{\ell}_{a'})} = \frac{\exp(\ell_a + b_a)}{\sum_{a'} \exp(\ell_{a'} + b_{a'})}
$$

**Soft mode analysis**: For a feasible action ($d_a = 0$), $b_a = 0$ and the logit is unchanged. For an infeasible action with deficit $d_a$, the logit is reduced by $\lambda \cdot d_a$, decreasing its probability by a factor of approximately $\exp(-\lambda \cdot d_a)$. With $\lambda = 5$ and $d_a = 2$, this is a factor of $e^{-10} \approx 4.5 \times 10^{-5}$.

**Hard mode analysis**: Infeasible actions receive $b_a = -10^9$, making $\exp(\tilde{\ell}_a) \approx 0$. The resulting probability is numerically zero; the policy can only select feasible actions.

### 4.3 Distribution Reconstruction

After computing $\tilde{\ell}_a$, a new categorical distribution is constructed:

$$
\tilde{\pi} = \text{Categorical}(\tilde{\ell}_1, \ldots, \tilde{\ell}_N;\; \texttt{unimix}=0)
$$

Setting $\texttt{unimix}=0$ ensures no uniform mixing is applied on top of the masked distribution, preserving the intended bias effect.

---

## 5. Integration with DreamerV3

### 5.1 Real Environment Interaction

During real environment interaction, the `policy()` method of the agent applies the mask using ground-truth context:

$$
\mathbf{c}_t = \texttt{extract\_mask\_context}(s_t)
$$
$$
\tilde{\pi}_t = \texttt{apply\_mask}(\pi_\theta(\cdot \mid z_t),\; \mathbf{c}_t)
$$
$$
a_t \sim \tilde{\pi}_t
$$

where $z_t$ is the RSSM latent state and $\pi_\theta$ is the learned policy network.

### 5.2 Mask Context Prediction Head

Since imagination operates on latent states without access to the true game state, a **mask context prediction head** $f_\phi$ is trained to predict the context from RSSM features:

$$
f_\phi: \mathbb{R}^D \rightarrow \mathbb{R}^{44}
$$

where $D = d_{\text{deter}} + d_{\text{stoch}} \times d_{\text{classes}}$ is the dimension of the concatenated RSSM feature vector.

**Architecture**: 2-layer MLP with SiLU activations, RMS normalization, and Gaussian output distribution (mean + log-std).

**Training loss**: Negative log-likelihood under the predicted Gaussian, with ground-truth context from replay:

$$
\mathcal{L}_{\text{mask\_ctx}} = -\frac{1}{BT} \sum_{b,t} \log f_\phi(\mathbf{c}_{b,t} \mid \text{sg}(\mathbf{h}_{b,t}))
$$

where $\mathbf{h}_{b,t} = [\text{deter}_{b,t};\; \text{stoch}_{b,t}]$ is the RSSM feature vector and $\text{sg}(\cdot)$ denotes stop-gradient (the world model representation is not updated through this loss).

### 5.3 Dream Masking

During imagination rollout of horizon $H$, the policy function applied at each imagined step $\tau$ is:

$$
\hat{\mathbf{c}}_\tau = f_\phi(\mathbf{h}_\tau).\text{pred}() \qquad \text{(predicted context mean)}
$$
$$
\hat{b}_{a,\tau} = -\lambda \cdot \hat{d}_{a,\tau}(\hat{\mathbf{c}}_\tau) \qquad \text{(predicted bias)}
$$
$$
\tilde{\pi}_\tau = \texttt{apply\_mask}(\pi_\theta(\cdot \mid \mathbf{h}_\tau),\; \hat{\mathbf{c}}_\tau)
$$
$$
a_\tau \sim \tilde{\pi}_\tau
$$

The same masking is applied to the policy distribution used in the **actor-critic loss** (imagination loss), ensuring consistency between the rollout distribution and the loss computation:

$$
\mathcal{L}_{\text{actor}} = -\sum_{\tau=0}^{H} \left[\text{sg}(A_\tau) \cdot \log \tilde{\pi}_\tau(a_\tau) + \eta \cdot \mathcal{H}(\tilde{\pi}_\tau)\right]
$$

where $A_\tau$ is the advantage estimate and $\mathcal{H}(\tilde{\pi}_\tau)$ is the entropy of the masked policy.

### 5.4 Warmup Schedule

The mask context head trains from the beginning, but its predictions are only used in imagination after a warmup period. This prevents unreliable early predictions from introducing systematic bias into the actor's training signal.

$$
\texttt{dream\_mask\_active} = \begin{cases}
\texttt{False} & \text{if } t < t_{\text{prefill}} + t_{\text{warmup}} \\
\texttt{True} & \text{otherwise}
\end{cases}
$$

with $t_{\text{prefill}} = 10{,}000$ (random policy data collection, no training) and $t_{\text{warmup}} = 40{,}000$ (mask context head training period). The total warmup is **50,000 environment steps** (5% of 1M training budget).

The flag switch is implemented as a Python-level attribute read at JAX trace time. The periodic `jax.clear_caches()` call (every 5,000 steps) ensures re-tracing after activation.

**Rationale**: During the warmup period:
- The RSSM has not yet learned accurate state representations
- The mask context head has insufficient training data
- Incorrect mask predictions could systematically block feasible actions
- Real environment masking (using ground-truth context) remains active throughout

### 5.5 Training Timeline

```
 Step:  0          10k              50k                            1M
        |── prefill ──|── warmup ────────|── dream mask active ──────|

 mask_ctx_head loss:  not computed   ████████████████████████████████
                                     (trains throughout)

 Dream masking:       off            off                on
                                     (predictions not    (predictions
                                      used in dreams)    used in dreams)

 Real env masking:    ██████████████████████████████████████████████████
                      (always on, ground-truth context)
```

---

## 6. Loss Function Summary

The complete loss optimized by the agent is:

$$
\mathcal{L} = \underbrace{\mathcal{L}_{\text{dyn}} + \mathcal{L}_{\text{rep}}}_{\text{RSSM}} + \underbrace{\mathcal{L}_{\text{obs}}}_{\text{decoder}} + \underbrace{\mathcal{L}_{\text{rew}} + \mathcal{L}_{\text{con}}}_{\text{reward \& continuation}} + \underbrace{\mathcal{L}_{\text{mask\_ctx}}}_{\text{mask context}} + \underbrace{\mathcal{L}_{\text{policy}} + \mathcal{L}_{\text{value}}}_{\text{actor-critic}}
$$

The mask context loss $\mathcal{L}_{\text{mask\_ctx}}$ has scale 1.0, consistent with other reconstruction losses. Importantly, it uses $\text{sg}(\mathbf{h})$ to prevent gradients from flowing back into the world model — the mask prediction head is a **read-only consumer** of the learned representation, analogous to the reward head.

---

## 7. Ablation Experiment Design (Group F)

Two experiments isolate the effect of action masking:

| Experiment | Mode | $\lambda$ | $L$ | Other components |
|------------|------|-----------|-----|------------------|
| **F1\_mask\_soft** | Soft | 5.0 | — | FIFO replay, no intrinsic, no P2E |
| **F2\_mask\_hard** | Hard | — | $10^9$ | FIFO replay, no intrinsic, no P2E |

Both use the **baseline configuration** (no NLR/NLU, no intrinsic rewards, no Plan2Explore), isolating the pure effect of action masking on exploration efficiency. Comparison against A1 (baseline without masking) reveals the marginal contribution of feasibility constraints.

Each experiment runs with 3 seeds (1, 4, 42) for statistical reliability.

---

## 8. Diagnostic Metrics

The following metrics are logged per episode for monitoring mask behaviour:

| Metric | Description | Aggregation |
|--------|-------------|-------------|
| `mask_penalty_mean` | Mean $\|b_a\|$ across all actions | Per-step mean |
| `mask_infeasible_frac` | Fraction of actions with $b_a < 0$ | Per-step mean |
| `mask_blocked_frac` | Fraction with $b_a \leq -L/2$ (hard mode) | Per-step mean |
| `mask/invalid_{action}_count` | Whether specific action was infeasible | Per-episode sum |
| `mask/place_table_prob_before` | $\pi(\text{PLACE\_TABLE})$ before masking | Per-step mean |
| `mask/place_table_prob_after` | $\tilde{\pi}(\text{PLACE\_TABLE})$ after masking | Per-step mean |
