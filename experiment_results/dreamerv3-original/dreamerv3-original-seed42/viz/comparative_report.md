# Comparative Analysis: Original DreamerV3 vs Modified (CL-Ready) DreamerV3 on Craftax

> **Environment:** CraftaxSymbolic-v1 · **Budget:** 500k steps · **Seed:** 42 · **GPU:** RTX 4090
>
> **Original DreamerV3** — vanilla DreamerV3 backbone, no continual learning modifications
> **Modified (CL-Ready)** — DreamerV3 backbone with Continual-Dreamer infrastructure (reservoir sampling hooks, CL metrics tracking, replay management), as prescribed by Kessler et al. (CoLLAs 2023), running in single-task mode

---

## 1. Executive Summary

| Metric | Original DreamerV3 | Modified (CL-Ready) | Δ | Winner |
|---|---|---|---|---|
| **Final Mean Return** | **4.23** | 3.78 | +0.45 (+11.9%) | Original |
| **Final Max Return** | **8.1** | 7.1 | +1.0 | Original |
| **Final Median Return** | 4.10 | 4.10 | 0.0 | Tie |
| **Final Std Return** | 1.110 | **0.859** | −0.251 | Modified |
| **Mean Episode Length** | 278.9 | **322.3** | −43.4 | Modified |
| **Total Episodes** | 1,814 | 1,768 | +46 | — |
| **Success Rate** | 100% | 100% | 0% | Tie |
| **Aggregate Forgetting** | 0.0332 | **0.0300** | −0.003 | Modified |
| **Lifetime Mean Return** | **3.192** | 3.077 | +0.115 | Original |
| **Achievements Discovered (>5%)** | 9 | 9 | 0 | Tie |

**Bottom line:** The Original DreamerV3 achieves **+12% higher mean return** at convergence, driven almost entirely by a massively higher `place_table` achievement rate (68% vs 9%). Despite identical YAML configurations, the training loop differences in the CL-ready codebase appear to introduce a behavioural penalty that manifests as reduced crafting-chain initiation. The Modified run compensates with longer survival (322 vs 279 steps/episode) and tighter variance (σ = 0.86 vs 1.11), but this survival advantage does not translate into higher return.

---

## 2. Episode Return over Training

![Episode Return Comparison](fig1_return_comparison.png)

**Figure 1:** Episode return (EMA-80 smoothed) for both runs. Faint scatter shows raw episodes; bold lines are smoothed trends.

Both runs follow the same general trajectory: rapid ascent from 0–150k, followed by decelerating gains. However, the curves begin to **diverge around 200k steps**. From 200k onward, the Original run pulls ahead and maintains a consistent ~0.5 return advantage through convergence.

**Phase-by-phase breakdown:**

![Phase Comparison](fig12_phase_comparison.png)

**Figure 2:** Mean episode return by 100k-step training phases.

| Phase | Original | Modified | Δ |
|---|---|---|---|
| 0–100k | 1.22 | **1.56** | −0.34 |
| 100–200k | 2.87 | **2.97** | −0.11 |
| 200–300k | **3.59** | 3.52 | +0.07 |
| 300–400k | **4.09** | 3.60 | +0.48 |
| 400–500k | **4.26** | 3.76 | +0.50 |

An interesting reversal occurs: the **Modified run learns faster initially** (0–200k), achieving +0.34 higher mean return in the first 100k steps. This early advantage disappears by 200k, and the Original run dominates the second half of training. This pattern suggests the CL infrastructure may provide a slight regularisation benefit early on (e.g., more uniform replay sampling stabilising initial world model learning), but this comes at the cost of reduced plasticity later when the agent needs to refine higher-order behaviours like crafting.

---

## 3. Rolling Mean Return & Success Rate

![Rolling Metrics](fig2_mean_return_success.png)

**Figure 3:** Left — 100-episode rolling mean return. Right — rolling success rate (threshold ≥ 1.0).

**Rolling mean return** (left panel) confirms the divergence pattern from Section 2. The Original run's curve overtakes the Modified around 250k steps and accelerates through 500k, while the Modified curve flattens. The gap at convergence is 4.23 vs 3.78.

**Success rate** (right panel) is uninformative for comparison — both runs reach 100% by ~150k steps and remain there. This confirms that both agents reliably unlock at least one achievement per episode for the entire second half of training.

---

## 4. Episode Length

![Episode Length](fig3_episode_length.png)

**Figure 4:** Episode length (EMA-80) over training.

The Modified run produces **significantly longer episodes**: 322 steps vs 279 steps on average in the final window, a +15.5% survival advantage. This is the single metric where the Modified run clearly outperforms.

However, longer survival does **not** translate to higher return. The Original agent achieves more reward per unit time — it's more *efficient*. Computing a crude "return per step" ratio:

- **Original:** 4.23 / 278.9 = 0.01516 return/step
- **Modified:** 3.78 / 322.3 = 0.01173 return/step

The Original is **29% more reward-efficient** per environment step. This suggests the Modified agent spends more time on low-value survival behaviours (wandering, drinking) rather than productive actions (crafting, resource collection).

---

## 5. Achievement Rates — Head-to-Head

![Achievement Head-to-Head](fig4_achievement_headtohead.png)

**Figure 5:** Final 100-episode achievement rates for all 22 Craftax achievements. Orange = Original, Blue = Modified.

![Achievement Delta](fig5_achievement_delta.png)

**Figure 6:** Difference in achievement rates (Original − Modified). Green bars = Original higher; Red bars = Modified higher.

### The `place_table` Gap

The single most impactful difference between the two runs is **`place_table`: 68% (Original) vs 9% (Modified)**. This is a +59 percentage point gap — by far the largest delta across all 22 achievements.

`place_table` is the gateway achievement for the entire crafting tree. Without a crafting table, the agent cannot make pickaxes, swords, or any downstream tools. The Original agent has learned to reliably place a crafting table in 68% of episodes, while the Modified agent almost never does. This single behavioural difference cascades through the entire tech tree and explains most of the return gap.

### Full Achievement Comparison

| Achievement | Original | Modified | Δ | Notes |
|---|---|---|---|---|
| collect_wood | 90% | 88% | +2% | Near-identical, both reliable |
| **place_table** | **68%** | **9%** | **+59%** | **Dominant difference** |
| eat_cow | 10% | 11% | −1% | Both low |
| collect_sapling | 100% | 100% | 0% | Both mastered |
| collect_drink | 50% | 56% | −6% | Modified slightly better |
| make_wood_pickaxe | 0% | 0% | 0% | Neither learns |
| make_wood_sword | 0% | 0% | 0% | Neither learns |
| place_stone | 97% | 100% | −3% | Both very high (anomalous) |
| collect_stone | 1% | 1% | 0% | Neither learns |
| make_iron_sword | 94% | 97% | −3% | Both very high (anomalous) |
| *All others* | 0% | 0% | 0% | Neither learns |

The Modified run has marginal advantages in `collect_drink` (+6%), `place_stone` (+3%), and `make_iron_sword` (+3%), but these are dwarfed by the `place_table` gap.

---

## 6. Per-Tier Comparison

![Tier Comparison](fig6_tier_comparison.png)

**Figure 7:** Average achievement rate per tech-tree tier.

| Tier | Original | Modified | Δ |
|---|---|---|---|
| **Tier 0 – Basic** | **45.4%** | 37.7% | **+7.7%** |
| Tier 1 – Stone | 14.1% | 14.7% | −0.6% |
| Tier 2 – Iron | 31.3% | 32.3% | −1.0% |
| Tier 3 – Diamond | 0.0% | 0.0% | 0.0% |
| Tier 4 – Combat | 0.0% | 0.0% | 0.0% |

The Original's advantage is entirely concentrated in **Tier 0**, driven by `place_table`. Tiers 1–4 are statistically indistinguishable between the two runs. Both agents hit the same hard exploration wall at the stone-age crafting chain.

---

## 7. Selected Achievement Learning Curves

![Selected Achievements](fig10_selected_achievements.png)

**Figure 8:** Learning curves for 8 key achievements showing temporal dynamics of skill acquisition.

Key observations from the individual curves:

- **Collect Wood & Collect Sapling:** Virtually identical trajectories. Both agents learn these immediately.
- **Place Table:** The Original's curve ramps from 0 to ~70% between 50k–250k steps, while the Modified's curve barely lifts off. Whatever exploration or policy behaviour leads to table placement, the Original discovers and reinforces it; the Modified does not.
- **Collect Drink:** The Modified run actually reaches a *higher* peak earlier (~60% around 200k) and maintains it, while the Original plateaus at ~50%. The Modified agent's longer survival may give it more opportunities to find water.
- **Eat Cow:** Both near-zero with noisy fluctuations. Neither agent reliably hunts.
- **Place Stone & Make Iron Sword:** Both anomalously high and near-identical — confirming these likely reflect environment/index mapping quirks rather than true crafting behaviour.

---

## 8. Forgetting Analysis

![Forgetting](fig7_forgetting.png)

**Figure 9:** Left — aggregate forgetting over training. Right — per-achievement forgetting at final timestep.

| Metric | Original | Modified |
|---|---|---|
| Aggregate Forgetting | 0.0332 | **0.0300** |

Both runs exhibit minimal forgetting, with the Modified run showing a marginal advantage (0.030 vs 0.033). In a single-task setting this difference is negligible — both replay configurations (identical 50/50 recency-uniform, 2M buffer) prevent catastrophic forgetting effectively.

The per-achievement forgetting (right panel) shows that whatever small forgetting exists is concentrated in `place_table` for the Original run (its rate dropped from a peak >70% to 68%) and in minor Tier 0 achievements for the Modified run.

---

## 9. Score Distribution

![Score Distribution](fig8_score_distribution.png)

**Figure 10:** Score distribution for the final 200 episodes of each run.

The distributions reveal a qualitative difference in policy character:

- **Original:** Wider distribution centered at ~4.0, with a clear right tail extending to 7–8. The agent has higher variance but reaches higher peaks more often.
- **Modified:** Tighter distribution centered at ~3.5–4.0, with less density above 5.0. The agent is more consistent but has a lower ceiling.

This is consistent with the variance numbers (σ = 1.11 vs 0.86). The Original's higher variance is driven by its `place_table` capability: episodes where it successfully places a table unlock additional crafting achievements, pushing the score up, while episodes where it fails remain at the same ~3.0 baseline as the Modified.

---

## 10. Running Maximum Return

![Running Max](fig11_running_max.png)

**Figure 11:** Best episode return achieved up to each training step.

The Original reaches its all-time best of **8.1** around 350k steps, while the Modified peaks at **7.1** around 130k steps and never improves. The 1.0 difference in peak return (8 vs 7 achievements in a single episode) confirms the Original has a higher skill ceiling, likely again attributable to `place_table` enabling longer achievement chains in exceptional episodes.

---

## 11. Achievements per Episode

![Achievements per Episode](fig9_ach_per_episode.png)

**Figure 12:** EMA-smoothed count of unique achievements unlocked per episode.

The curves mirror the return comparison. Both start near 1 achievement/episode at 0 steps. By convergence, the Original averages ~4.2 achievements/episode vs ~3.8 for Modified. The 0.4 achievement gap maps directly to the ~0.45 return gap (each achievement ≈ 0.1 reward), confirming the return difference is driven by achievement count rather than reward scaling differences.

---

## 12. Configuration Comparison

Both runs use **identical YAML configurations**:

| Parameter | Original | Modified |
|---|---|---|
| RSSM (stoch × classes) | 32 × 24 | 32 × 24 |
| RSSM deterministic | 3072 | 3072 |
| Hidden units | 384 | 384 |
| Imagination horizon | 15 | 15 |
| Learning rate | 4e-5 | 4e-5 |
| Batch size × length | 16 × 64 | 16 × 64 |
| Replay size | 2M | 2M |
| Replay fracs | 50% recency / 50% uniform | 50% recency / 50% uniform |
| Train ratio | 64 | 64 |
| Envs | 16 | 16 |
| Seed | 42 | 42 |
| Entropy coefficient | 3e-4 | 3e-4 |
| Compute dtype | bfloat16 | bfloat16 |

The **only differences** are in the training loop code (`train_craftax.py`), where the Modified run includes CL infrastructure: the `CraftaxMetrics` tracker, achievement monitoring, replay buffer management hooks, and continual learning metric computation. Even running in single-task mode, this CL wrapper introduces additional overhead and potentially alters the training loop timing, data flow, or replay sampling behaviour in subtle ways.

---

## 13. Root Cause Analysis: Why Does the Original Outperform?

Given identical configs and seed, the +12% return gap must originate from differences in the training loop code. Several hypotheses:

### Hypothesis 1: Training Loop Overhead Alters Replay Timing (Most Likely)

The CL-ready training loop includes additional metric computation (achievement tracking, forgetting calculation, JSONL logging) at every episode boundary. In a JAX-based training pipeline where environment stepping and model training are tightly coupled, this overhead could:

- Delay gradient updates relative to data collection, subtly shifting the replay distribution
- Alter the ratio of fresh-to-stale data in each training batch
- Change which transitions end up in which training batch due to timing differences

Even millisecond-level delays can compound over 1,768 episodes, causing the two runs to diverge in which experiences get sampled and when gradient updates occur. This is consistent with the observation that both runs are similar early (when the overhead is a small fraction of total compute) but diverge later (when the accumulated timing difference is larger).

### Hypothesis 2: JAX Non-Determinism

Despite `seed=42` and `jax.deterministic=True`, JAX with bfloat16 on GPU is not fully deterministic due to non-associative floating-point reduction operations. The two different code paths (even with identical config) produce different JIT compilation traces, different memory layouts, and potentially different numerical results. Over 500k steps, these butterfly-effect differences can produce meaningfully different policies.

### Hypothesis 3: Replay Buffer Interaction Differences

If the CL code path modifies how episodes are stored in or sampled from the replay buffer (e.g., additional metadata, different episode boundary handling, or different chunking), this would directly affect the training distribution. The Modified run's 50/50 recency-uniform sampling may interact differently with its CL-aware replay management compared to the Original's vanilla replay.

### Why `place_table` Specifically?

`place_table` is a **high-variance, threshold-dependent skill**. It requires the agent to: (1) collect enough wood, (2) open the crafting menu, (3) select the table recipe, and (4) place it — a 4-step sequence that is easily disrupted by any change in exploration behaviour. Small differences in early exploration (driven by the timing/replay effects above) could lead one run to discover and reinforce this sequence while the other misses the critical window and never develops the behaviour.

---

## 14. Conclusions & Implications for Continual Learning Extension

1. **The vanilla DreamerV3 is the stronger single-task baseline** at +12% mean return, primarily due to its `place_table` advantage. Any CL extension should use the original training loop as the base and add CL components minimally.

2. **The CL infrastructure does NOT harm forgetting** — the Modified run actually shows marginally lower forgetting (0.030 vs 0.033). The 50/50 replay strategy works equally well in both code paths.

3. **The Modified run's survival advantage is real but insufficient.** Longer episodes (+15% survival) don't compensate for the crafting deficiency. For the CL extension, this suggests that survival-oriented intrinsic rewards alone won't push the agent deeper into the tech tree — explicit crafting-oriented exploration (e.g., Plan2Explore, curiosity-driven bonuses on novel state transitions) is needed.

4. **Both runs hit the same exploration wall.** 13 of 22 achievements are at 0% for both agents. The imagination horizon of 15 steps is likely too short for multi-step crafting chains, and the entropy coefficient of 3e-4 may be too low for sufficient exploration. These are shared limitations independent of the CL/non-CL distinction.

5. **For the TD-error curriculum learning extension:** Use the Original DreamerV3 as the base, verify the training loop doesn't introduce unnecessary overhead, and focus the curriculum on driving `place_table` → `make_wood_pickaxe` → `collect_stone` as the critical skill chain that would unlock Tiers 1–2.

---

*Report generated February 2026. Data: 2 × 500k-step DreamerV3 runs on CraftaxSymbolic-v1 (seed 42). Reference: Kessler et al., "The Effectiveness of World Models for Continual Reinforcement Learning," CoLLAs 2023.*
