## 1.

$$o_t = \begin{bmatrix} v_{vis}^{(t)} \\ v_{inv}^{(t)} \end{bmatrix}$$

$$r_t^{int} = r_{spatial}(v_{vis}^{(t+1)}) + \lambda \cdot r_{craft}(v_{inv}^{(t+1)})$$

1. Define a projection function $\phi: \mathbb{R}^{d_{vis}} \to \mathbb{Z}^k$ that discretizes/hashes the visual state (e.g., downsampling).
2. Maintain an episodic count table $N(\phi(s))$.
3. The reward is:
$$r_{spatial}(s_{t+1}) = \frac{1}{\sqrt{N(\phi(v_{vis}^{(t+1)}))}}$$

As $N \to \infty$ (looping), $r \to 0$.

To prevent the agent from repeatedly crafting and consuming the same item, we reward only the first discovery of a specific inventory state configuration or specific item counts.

Let $\mathcal{H}_{inv}$ be the set of all previously achieved inventory vectors (or a significant subset, such as "max count per item type achieved").

$$r_{craft}(s_{t+1}) = \mathbb{I}\left[ v_{inv}^{(t+1)} \notin \mathcal{H}_{inv} \right]$$

This is a sparse, non-decaying signal that forces the agent to climb the tech tree to receive further rewards.
## 2.
$$\pi_{masked}(a|s) = \frac{\pi_{\theta}(a|s) \cdot M(s)_a}{\sum_{a' \in \mathcal{A}} \pi_{\theta}(a'|s) \cdot M(s)_{a'}}$$

