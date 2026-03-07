项目中的两个 Intrinsic Reward 设计
本项目在 Craftax 环境中设计了两个 episodic intrinsic reward，分别针对空间探索和物品合成新颖性，最终线性组合后与 extrinsic reward 加权求和。

1. Spatial-Counting Reward（空间计数奖励）
状态抽象（哈希函数）
Craftax 的 flat observation $o_t \in \mathbb{R}^{d}$ 的前 $M = R \times C \times K$ 个维度编码了局部 $R \times C$ 的 tile map（$R=9, C=11, K=83$ channels）。定义视觉状态抽象 $\phi: \mathbb{R}^d \to \mathcal{S}_{\text{vis}}$：

$$
\phi(o_t) = \left(\underset{k \in {0,\ldots,K_b-1}}{\arg\max}; o_t^{(r,c,k)}\right)_{r \in [R],, c \in [C]}
$$

其中 $K_b = 37$ 是 one-hot block-type 通道数。即对每个 tile $(r,c)$，取前 $K_b$ 个通道的 argmax 得到该 tile 的 block 类型 ID，最终 $\phi(o_t) \in {0,\ldots, K_b-1}^{R \times C}$ 作为离散键。

Episodic 访问计数
在每个 episode 内维护访问计数表 $N_{\text{ep}}: \mathcal{S}{\text{vis}} \to \mathbb{N}$，episode 开始时重置 $N{\text{ep}} \leftarrow \emptyset$。每步更新：

$$
N_{\text{ep}}(\phi(o_t)) \leftarrow N_{\text{ep}}(\phi(o_t)) + 1
$$

Spatial Reward
$$
r_t^{\text{spatial}} = \frac{1}{\sqrt{N_{\text{ep}}(\phi(o_t))}}
$$

这是经典的 count-based exploration bonus，对频繁访问的视觉状态给予递减奖励，鼓励 agent 探索未见过的地图区域。

2. Craft-Novelty Reward（合成新颖性奖励）
状态抽象（哈希函数）
Observation $o_t$ 的第 $M$ 到第 $M + L - 1$ 维编码了 inventory 向量（$L = 51$）。定义 inventory 状态抽象 $\psi: \mathbb{R}^d \to \mathcal{S}_{\text{inv}}$：

$$
\psi(o_t) = \left(\text{round}(o_t^{(M+j)},, 1)\right)_{j=0}^{L-1}
$$

即将 inventory 向量的每个分量四舍五入到小数点后 1 位，得到离散化的 inventory 签名 $\psi(o_t) \in \mathcal{S}_{\text{inv}}$。

Episodic 历史集合
在每个 episode 内维护已见 inventory 状态集合 $\mathcal{H}{\text{inv}} \subseteq \mathcal{S}{\text{inv}}$，episode 开始时重置 $\mathcal{H}_{\text{inv}} \leftarrow \emptyset$。

Craft-Novelty Reward
$$
r_t^{\text{craft}} = \mathbb{1}\left[\psi(o_t) \notin \mathcal{H}_{\text{inv}}\right]
$$

计算完后更新：$\mathcal{H}{\text{inv}} \leftarrow \mathcal{H}{\text{inv}} \cup {\psi(o_t)}$。

这是一个 二值新颖性信号：当 agent 通过合成/采集等操作到达一个 episode 内从未见过的 inventory 配置时给予 $+1$ 奖励，否则为 $0$。

组合公式
Intrinsic Reward 合成 (Eq.12)
$$
r_t^{\text{intr}} = r_t^{\text{spatial}} + \lambda \cdot r_t^{\text{craft}}
$$

其中 $\lambda$ 为 craft_weight（默认 $\lambda = 1.0$）。

最终混合奖励 (Eq.13)
$$
r_t = \alpha_i \cdot \text{norm}(r_t^{\text{intr}}) + \alpha_e \cdot r_t^{\text{extr}}
$$

其中 $\text{norm}$ 为自适应归一化：通过 cross-episode EMA 追踪 $\bar{r}^{\text{intr}}$ 和 $\bar{|r^{\text{extr}}|}$，令
$$
\text{norm}(r^{\text{intr}}) = r^{\text{intr}} \cdot \frac{\bar{|r^{\text{extr}}|}}{\bar{r}^{\text{intr}}}
$$
使得归一化后 $\mathbb{E}[\text{norm}(r^{\text{intr}})] \approx \mathbb{E}[|r^{\text{extr}}|]$，从而 $\alpha_i / \alpha_e$ 成为真正的相对权重比。前 100 步为 warmup 期，不做归一化。

$\alpha_i$：intrinsic reward 缩放系数（默认 $0.1$，即 intrinsic 贡献为 extrinsic 的 1/10）
$\alpha_e$：extrinsic reward 缩放系数（默认 $1.0$）
$r_t^{\text{extr}}$：环境原始外部奖励
两个 episodic 状态（$N_{\text{ep}}$ 和 $\mathcal{H}_{\text{inv}}$）在每个 episode 的 is_first 信号为 true 时完全重置，因此这是 episodic intrinsic motivation——每个 episode 独立地鼓励探索，不会随训练推进而全局衰减。