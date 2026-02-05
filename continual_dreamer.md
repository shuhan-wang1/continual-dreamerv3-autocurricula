# 1. Abstract and Introduction

## 1.1 核心贡献：
这篇文章讨论了**世界模型**和**持续强化学习**中的有效性。作者提出*Continual-Dreamer*这是一种基于**DreamerV2**的方法，旨在解决智能体在不断变化的环境（Sequential Tasks）时出现的catastrophic forgetting问题

**核心假设**：世界模型天然具备*Generative Replay*的潜力。通过在Dreamer中训练策略，并结合特定的Experience Replay, 模型可以在不依赖任务标识符（Task-Agnostic）的情况下进行高效的持续学习。

## 1.2 数学视角的公式化表述(Formalization)
我们在这里引入必要的数学符号来形式化逻辑：

### 1.2.1 A. 持续强化学习(CRL)的目标：
假设智能体面临一个Sequential Tasks $\mathcal T=(\mathcal{T}_1,...,\mathcal{T}_N)$。在时刻$t$，智能体处于任务$\mathcal{T}_k$中。CRL的目标是找到一组超参数$\theta$，使得在所有已见过的任务上的累计期望汇报最大化：

$$J(\theta) = \sum_{i=1}^{k} \mathbb{E}_{\tau \sim \pi_\theta, \mathcal{T}_i} \left[ \sum_{t=0}^{H} \gamma^t r_t \right]$$

其中$\tau$是轨迹，$\gamma$是折扣因子。难点在于当训练$\mathcal{T}_k$时，无法直接访问$\mathcal{T}_{i < k}$的环境，只能依赖记忆(Replay Buffer)。

### 1.2.2 B. 世界模型的工作流(The world model loop)：
和model free方法(PPO)不同，世界模型引入了一个中间映射$M$。
1. 交互（Interaction）：策略$\pi$和真实环境交互，将数据存入回访缓冲区(Replay Buffer) $\mathcal{D}$:

$$(o_t, a_t, r_t, o_{t+1}) \rightarrow \mathcal{D}$$

2. 模型学习(Model learning): 学习环境的动态$P(s_{t+1},r_t|s_t,a_t)$和观察模型$P(o_t|s_t)$。这实际上是在压缩环境信息。

3. 世界模型训练(Dreamer): 策略$\pi$完全在世界模型的轨迹中训练：

$$\hat{s}_{t+1} \sim M(\hat{s}_t, \hat{a}_t)$$

$$\theta^* \leftarrow \text{maximize } \mathbb{E}_{\text{dream}} [R]$$

## 1.3 QA:
### 1.3.1 Q1:
为什么世界模型 (World Models) 比传统的无模型方法（如 IMPALA 或 DQN 加上 Replay Buffer）更适合持续学习？仅仅是因为“样本效率”吗？如果在 Buffer 有限的情况下，模型难道不会和策略一样发生遗忘吗？

1. Compressed Representation: 
传统的Replay Buffer储存的是高维原始像素$o_t$。世界模型通过学习潜变量状态$s_t$(Latent State)，实际上执行了一种有损压缩（Lossy Compression）。这意味着在相同的储存预算下，世界模型记住的环境动态特征比原始像素更robust.

2. 生成时回放(Generative Replay)的隐式实现：
在监督学习中，生成式回放(训练一个GAN来生成旧数据)是对抗遗忘的标准方式。世界模型本质上就是一个生成模型。当我们在Dreamer中训练策略时，如果模型没有遗忘，这等价于我们在**无限制，无限量**的回放旧任务数据，而不仅仅是受限于Buffer中有限的样本。

3. 解耦（Decoupling）：
遗忘通常发生在策略更新的时候。在世界模型中，我们将“理解环境”（训练模型）和“执行任务”（策略训练）解耦了。只要模型通过Replay Buffer保持对环境动态的记忆，策略就可以在任何时候通过Dreamer来回复性能，不需要真实的重新去交互。

# 2. Preliminaries:
本部分建立严格的数学框架，定义Partially Observable Markov Decision Process(POMDP)和Continuous Reinforcement Learning(CRL)的具体设定。

## 2.1 POMDP的形式化定义
论文将每个任务都建模成一个POMDP，由元组$(\mathcal{S}, \mathcal{A}, P, R, \Omega, \mathcal{O}, \gamma)$定义：
- $\mathcal{S}$: 状态空间 (State Space)。$s_t \in \mathcal{S}$ 是环境在时刻 $t$ 的真实状态。
- $\mathcal{A}$: 动作空间 (Action Space)。$a_t \in \mathcal{A}$。
- $P$: 转移分布 (Transition Distribution)，定义为条件概率 $P(s_{t+1} | s_t, a_t)$ 
- $R$: 奖励函数 (Reward Function)，$R(a_t, s_t, s_{t+1})$
- $\Omega$: 观测空间 (Observation Space)
- $\mathcal{O}$: 观测函数 (Observation Function)，定义为 $\mathcal{O}: \mathcal{S} \times \mathcal{A} \rightarrow P(\Omega)$。这意味着智能体无法直接访问 $s_t$，只能获得观测值 $o_t \sim P(o_t | s_t, a_{t-1})$
- $\gamma$: 折扣因子 (Discount Factor)，$\gamma \in (0, 1)$

**目标函数**：在有限视界$H$中，回报(Return)定义为$R_t=\sum_{i=t}^H\gamma^{(i-t)}r(s_i,a_i)$。RL的目标是最大化期望回报：

$$J = \mathbb{E}_{a_i \sim \pi, s_0 \sim \rho} [R_1 | s_0]$$

其中$\rho(s_0)$是初始状态分布

## 2.2 CRL设定：
在CRL中，智能体面对的是一个任务序列$\mathcal{T}_1,...,\mathcal{T}_N$。每个任务都是新的POMDP

$$\mathcal{T}_\tau = (\mathcal{S}_\tau, \mathcal{A}_\tau, P_\tau, R_\tau, \Omega_\tau, \mathcal{O}_\tau, \gamma_\tau)$$

智能体对每个任务都有$N$次交互预算。

**关键假设(The disjoint assumption)**:论文做了一个强假设：不同任务的状态空间是Disjoint的。

$$\forall (\mathcal{T}_i, \mathcal{T}_j), i \neq j \implies \mathcal{S}_i \cap \mathcal{S}_j = \emptyset$$

## 2.3 QA:
### 2.3.1 
Q: 既然假设的所有任务的状态空间$\mathcal{S}$是互补相交的($\mathcal{S}_i \cap \mathcal{S}_j=\empty$), 这是否意味着智能体总是能通过当前的输入来轻易判断到他在那个任务中？这难道不是把持续学习中最难的“任务推断”问题给简化没了么？

虽然$\mathcal{S}$是disjoint的（例如, Task A的迷宫坐标和Task B的迷宫坐标在数学几何上不重叠），但智能体接收的是**观测值**$o_t \in \Omega$。论文指出“由于我们处理的是POMDP，不同任务的某些观测值可能会重合，尽管不同task的状态是disjoint的，但**不同任务的某些观测值可能会重合**”。
  - 例如：两个不同的迷宫(状态空间很明显不相同)，在其各自的某个角落，智能体看到的某些东西可能是一样的，比如两个迷宫的tasks不同但迷宫的结构，长相可能比较类似(observation, $o_i \approx o_j$)
  - 结论：即使状态空间不相交，由于观测空间的混淆性(Partial Observalibility)，智能体依然要面临严重的**感知混淆(Perceptual Aliasing)**问题。他必须利用历史的信息来推断当前的潜在状态$s$，从而隐式的判断当前的任务环境。这就是为什么用DREAMER里的Recursive State Space Model (RSSM)的核心动机。


# 3. Methodology I - Continual-Dreamer Architecture
该方法建立在DreamerV2的基础之上，并引入了基于Plan2Explore的任务无关探索机制。

## 3.1 RSSM
DreamerV2的核心是一个Recurrent State-Space Model, 他把环境建模成一个非线性动态系统，旨在压缩观测历史并预测未来。

数学定义: RSSM将状态分为两部分：确定性部分$h_t$和随机部分$z_t$。

### 3.1.1 确定性路径(Deterministic Path):
使用GRU更新隐藏状态：

$$h_t = \text{GRU}(h_{t-1}, z_{t-1}, a_{t-1})$$

这部分负责记忆长期的上下文信息。

### 3.1.2 随机潜变量(Stochastic Latent State):
$z_t$是离散的概率潜变量（Categorical Variable）。
- Prior/Predictor: 仅根据历史预测当前状态（用于Dreamer）：

$$\hat{z}_t \sim p_\phi(z_t | h_t)$$

- Posterior/Representation: 结合真实观测$o_t$更新状态（用于模型训练）：

$$z_t \sim q_\phi(z_t | h_t, o_t)$$

### 3.1.3 Reconstruction: 
模型通过潜变量重构观测值和奖励：

$$\hat{o}_t \sim p_\phi(o_t | h_t, z_t)$$
$$\hat{r}_t \sim p_\phi(r_t | h_t, z_t)$$

## 3.2 Policy Learning in World Model
这是Continual-Dreamer对抗以往的关键。策略$\pi_\psi$和value function, $v_\xi$完全是在模型生成的虚拟轨迹里训练的，而不是直接使用真实的环境样本。

**Algorithm 1:**
1. Freeze Model: 在更新策略时，RSSM的参数$\phi$保持不变。

2. Rollout: 从回放缓冲区中采样起始状态，然后使用学习到的先验$p_\phi(z_{t+1}|h_t)$向前预测$H$步：
$$\hat{z}_{t+1}, \hat{h}_{t+1}, \hat{r}_{t+1}, \dots$$

3. 更新目标(objective): 最大化Dreamer中的累计回报：

$$\text{maximize } \mathbb{E}_{q_\phi} \left[ \sum_{\tau=t}^{t+H} \gamma^{\tau-t} \hat{r}_\tau \right]$$

  - **逻辑连接**：因为这些Dream是模型产生的，只要模型记住了旧任务的动态（通过Replay Buffer维持），策略就能在dream中复习旧任务，即使当前真实环境已经是最新任务。

## 3.3 任务无关的探索(Task-Agnostic Exploration)
为了在没有任务边界信号的情况下来适应新环境，论文继承了**Plane2Explore**。这通过引入*Intrinsic Reward*来驱动探索。

**内在奖励计算**：使用一个由$K$个神经网络的Ensemble, $\{q_k\}_{k=1}^K$来预测一步后的潜变量特征。输入$(h_t, a_t)$目标

内在奖励$r_i$定义为该Ensemble的variance(Disagreement in prediction)

$$r_i(t) = \text{Var} \left( \{ q_k(h_t, a_t) \}_{k=1}^K \right)$$

总奖励：策略优化时的总奖励是外在奖励$r_e$（任务奖励）和内在奖励的加权总和：

$$r_{total} = \alpha_e r_e + \alpha_i r_i$$

其中$\alpha_e, \alpha_i\in[0,1]$。

## 3.4 QA
### 3.4.1 Q1
在计算内在奖励时，RSSM本身旧已经输出了分布$p_\phi(z_{t=1}|h_t)$，这本身就包含了Variance。为什么还要额外训练一个包含$K$个神经网络的Ensemble来计算方差？

这是一个经典的不确定性的混淆。我们需要区分这两种不确定性：
1. 随机不确定性(Aleatoric Uncertainty):类似于Bayesian Error，这是数据固有的噪声，例如掷色子的结果。RSSM输出的分布$p_\phi$捕捉的正式这种环境本身的随机性。即使模型训练得完好无损，这种方法依然存在。如果你把他当作奖励，智能体就会被这些本身存在数据本身的高随机，无意义的状态吸引(Noisy TV problem).
2. 认知不确定性(Epistemic Uncertainty): 这是模型**因为没见过**而产生的不确定性。这才是我们需要奖励模型去探索的地方

**集成的作用：**集成模型$\{q_k\}$捕捉的是*Epistemic Uncertaitny*
- 对于已知的随机时间（掷硬币），所有$K$个模型都会一致的预测$50-50$（分布一致，低集成方差）
- 对于从来没见过的状态，$K$个模型的预测会大相径庭（集成方差高）

因此，必须要用集成（Ensemble）的分析（variance）作为内在的奖励驱动，引导智能体去探索未知区域，而不是随机区域


### 3.4.2 Q2：
什么叫K个神经网络组成的Ensemble? 是简单的MLP么？

是的，他们本质上就是K个简单的MLP（2-3层全连接层），不是K个独立的agent.

在Dreamer/Plan2Explore框架下，这个ensemble是为了衡量“我对当前环境的物理规则（动态）有多不确定”而专门设计的auxiliary组件。

**具体实现：**
假设RSSM提供了一个潜在的状态空间$z_t$。这个Ensemble由$K$个独立的predictor  $\{q_{\phi_k}\}_{k=1}^K$组成
1. 输入：当前的潜在状态$s_t$和采取的动作$a_t$
2. 目标：预测下一步的潜在特征$s_{t+1}$（或者$s_{t+1}$的某种嵌入Embedding）
3. 架构：每个Predictor $q_{\phi_k}$ 都是一个轻量的MLP

$$\hat{s}_{t+1}^{(k)} = \text{MLP}_k(s_t, a_t; \phi_k)$$

4. 训练：这$K$个网络使用相同的真实历史数据($s_t,a_t,s_{t+1}$)进行监督训练，但他们的*参数初始化$\phi_k$是随机独立的*

**为什么这样做**？
这基于 Deep Ensembles (Lakshminarayanan et al., 2017) 的原理：
- 对于已访问过的区域（数据充足），所有 $K$ 个网络都会收敛到相似的预测结果（因为 Ground Truth 是一样的）。
- 对于未访问过的区域（数据稀缺），由于初始化的不同，这 $K$ 个网络预测的Variance会很大

### 3.4.3 Q3:
这和预测regret方法有什么不同？

在这里我们计算的不是Regrett,而是Epistemic Uncertainty，具体定义为$N$个Predictor的预测方差(Predictive Variance)。

**核心区别Regret vs disagreement**:
训练$N$个agent计算regret通常指的是*Bootstrapped DQN*或$Posterior Sampling for RL$这类方法，他们衡量的是“Value/Q”的不确定性（即我不确定这个动作是不是最优的）。

而Plan2Explore方法衡量的是Dynamics的不确定性（即：我确定做了这个动作后的后果是什么，不管这个后果好不好）

P2E的Instrinsic Reward计算公式是：
内在奖励$r_i$定义为这$K$个预测器输出的方差:

$$r_i(s_t, a_t) = \text{Var} \left( \{ \hat{s}_{t+1}^{(k)} \}_{k=1}^K \right) = \frac{1}{K-1} \sum_{k=1}^K \left\| \hat{s}_{t+1}^{(k)} - \bar{s}_{t+1} \right\|^2$$

其中 $\bar{s}_{t+1}$ 是所有预测器的均值。

- 直观理解： 如果 $r_i$ 很大，说明“我的 $K$ 个分身对这个action会发生什的variance，这说明这里是未知领域，值得探索。
- 这种方法的优势： 它是 Task-Agnostic 的。无论任务是要吃金币还是要躲避怪物，只要“物理规律”我不懂，我就要去探索。这正是持续学习（Continual Learning）所需要的，因为新任务往往意味着新的物理环境动态。

# 4. Methodology II - Selective Experience Replay
## 4.1 Population the Buffer:
假设buffer的大小为$N$，当前见过的总轨迹数是$T$。

### 4.1.1 Navie way - First in First Out
在缓冲区满的时候，直接丢弃最旧的数据

$$T > N \implies \text{Drop } \{x_1, \dots, x_{T-N}\}$$

缺陷：随着任务序列$\mathcal{T}_1 \to \mathcal{T}_2 \to ...$, Buffer中的$\mathcal{T}_1$的数据会被完全遗忘，导致模型遗忘旧人物的物理规律。

### 4.1.2 Reservoir Sampling
维护一个能代表**所有已经见过历史的**均匀分布样本，不需要知道历史数据总量。当第$t$条新的轨迹$x_t$来时：

1. 如果$t\le N$, 直接放入
2. 如果$t>N$,以概率$P_{odd}$存入：

$$P_{add} = \frac{N}{t}$$

3. 如果决定存入，则随机替换掉$Buffer$中现有的一个样本。

通过这种机制，在任意时刻$t$, Buffer中的每个样本数量都以相同的概率$P_{odd}$来自于历史的任意一条轨迹。这意味着即使到了第100个任务，Buffer里依然会在期望上有$1/100$的数据来自第一个任务

### 4.1.3 覆盖最大化(Covergae Maximization)
目标是最大化状态空间的覆盖率。定义度量$d(x_i,x_j)$。新样本$x_t$的存入概率取决于他与Buffer中现有样本的距离：

$$P_{add} \propto \min_{x_j \in \mathcal{D}} d(x_t, x_j)$$

如果$x_t$与现有记忆非常不同（代表新的状态分布），则有限储存。

## 4.2 Minibatch Construction
即使buffer里有了均匀的数据，训练时候的采样策略也很重要。

**50-50 Stability Plasticity Balance**：为了平衡记住的旧知识和新知识，文章提出了混合采样：

$$B_{batch} = \{ \underbrace{x \sim \mathcal{U}(\mathcal{D})}_{50\% \text{ Random}}, \underbrace{x \sim \text{Recency}(\mathcal{D})}_{50\% \text{ Recent}} \}$$

- 50% 均匀采样： 用于复习旧任务，防止遗忘（利用水库采样的性质）。
- 50% 最近采样： 专门采样最近存入的数据，确保模型能快速适应当前正在进行的任务 。

## 4.3 QA

### 4.3.1 Q and A
Reservoir Sampling看起来很好，他保持了数学上的随机分布。但是，随着任务数量$T\to \infty$，每个任务在Buffer中的样本数量会变成$N/T \to 0$. 如果任务1的样本只剩下$5$个，模型怎么可能还记得住复杂的环境动态？ 这不是系数性遗忘么（虽然比灾难性遗忘好）？

这是Reservoir采样的理论极限。然而，在持续强化学习的实践中，有几个因素缓解了这个问题：

1. 世界模型具有泛化能力：即使只有很少样本，如果这些样本是高质量的（覆盖了关键的状态转移），训练良好的神经网络（世界模型）能够通过泛化能力差值出中间的物理规律。他不需要记住每一帧，只需要记住核心物理规律。

2. 数据的重叠性：许多任务虽然目标不同(state disjoint assumption)，但有些规律还是相同的，比如无论任务怎么变，走路，撞墙的物理反馈始终是intrinsic的。任务$T_{100}$的数据也可能在隐式的帮助模型复习任务$T_1$的物理规律

3. 实际性能对比：实验表明，相较于*FIFO*的灾难性遗忘，水库采样保留的少样本足以保持baseline performance。当然，如果任务差异巨大，且数量极多，确实是需要扩展Buffer大小$N$，或者让世界模型生成旧数据（这是世界模型的终极潜力，尽管本文主要是依赖buffer）

# 5. Experiments:
本部分对Continual-Dreamer在MiniGrid和MiniHack两个基准测试上的吧i凹陷进行评估

## 5.1 Experiment Setup
### 5.1.1 Benchmarks:
1. Minigrid (3 tasks): pixel level, 部分可观测，稀疏奖励。任务包含：`DoorKey`, `LavaCrossing`, `SimpleCrossing`.
2. Minihack: 基于NetHack的更难环境，设计复杂的任务技能（推石头过河）。任务包含`Room-Random`,`River-Narrow`,`River-Monster`

### 5.1.2 Baselines:
1. Impala: 强大的多任务RL算法，但非针对持续学习设计。
2. CLEAR: 任务无关(Task-Agnostic)的持续学习SOTA（2023年的），使用$V-trace$和Replay Buffer

## 5.2 Evaluation Metrics:
设$p_\tau(t) \in [-1, 1]$为任务$\tau$在总训练步数$t$时的归一化性能。总任务数量为$T$, 每个任务训练时长为$N$。总时长$t_f = T\times N$.

### 5.2.1 Average Performance:
衡量在训练结束时，智能体在**所有**任务的最终掌握程度：

$$AP = \frac{1}{T}\sum_{\tau =1}^Tp_{\tau}(t_f)$$

### 5.2.2 Average Forgetting:
衡量任务$\tau$在刚训练完时候的性能$p_\tau(\tau \times N)$和最终性能$p_\tau(t_f)$之差。

$$F = \frac{1}{T} \sum_{\tau=1}^{T} \left( p_\tau(\tau \times N) - p_\tau(t_f) \right)$$

- $F > 0$ 表示发生了遗忘。
- $F < 0$ 表示性能在后续训练中反而提升了（正向后向迁移）。

### 5.2.3 Forward Transfer
衡量持续学习智能体学习新任务时，相对于从零开始单任务训练(Single-Task Reference)的加速程度。基于Area under the curve (AUC)计算

$$FT = \frac{1}{T} \sum_{\tau=1}^{T} \frac{\text{AUC}_\tau^{CL} - \text{AUC}_\tau^{Ref}}{1 - \text{AUC}_\tau^{Ref}}$$

- $FT > 0$ 表示之前的经验帮助了新任务的学习。

## 5.3 Key Results:
### 5.3.1 表现结果
在Minigrid和Minihack上，Continual-Dreamer(DreamerV2 + Reservoir sampling)在平均回报上显著优于Impala和CLEAR。

- Minigrid: DreamerV2能够解决CLEAR需要10倍数据才能解决的任务(样本效率高)。
- Minihack: 在复杂的`River-Monster`任务中，只有Continual-Dreamer+Plan2Explore能有效学习，而CLEAR完全失败。

### 5.3.2 回放策略的影响：
- FIFO: 随着新任务数据的涌入，旧任务性能快速下降（高遗忘）。
- 水库采样：显著降低了遗忘，使得各个任务数据在Buffer中保持相对均匀的分布
- Plan2Explore(p2e): 提高了Forward Transfer和对难探索任务的解决率，证明了在内在奖励在持续学习中的有效性。

## 5.4 QA:
既然Buffer越大，旧数据越多，那性能应该越好。但Appendix D.5 展示了一个反直觉的现象：增加Buffer大小反而降低了Forward Transfer, 甚至导致某些新任务学不会。这是为什么？难道“记忆”会阻碍“学习”么？

这是Stability-Plasticity Dilemma的体现：
1. Dilution: 当Buffer变得特别大的时候，新任务产生的数据(recent experience)在整个Buffer中的占比会变得非常小（旧数据太多）
2. 训练信号淹没：在训练世界模型和策略的时候，如果我们从这个巨大的Buffer中均匀采样， 绝大多数样本都来自旧任务。这意味着模型主要在复习旧知识，而很少看到新任务的样本
3. 结果：模型变得过于稳定（Stability高，forget低），但失去了可塑性（Plasticity）。他对新任务的适应速度变得特别慢，导致在有限的预算内无法学习到新任务，降低了Forward Transfer

这也反向证明了为什么50:50采样策略（强制一半来之最近）的重要性，它人为地打破了这种数量不对称，强制模型关注当下。


# 6. Discussion and Conclusion
## 6.1 Interference and Imbalance
### 6.1.1 Interference
尽管continual-Dreamer在多个任务上表现出色，但并非免疫干扰。在Appendix D6中，`Four Rooms`环境。
- 设定：两个任务的环境结构**完全相同（墙壁、布局）**, 仅仅是**奖励函数或目标位置**改变了
- 现象：使用Replay Buffer会导致严重干扰。旧任务的数据（在这个位置没有奖励）与新任务的数据（在这个位置有奖励）在buffer中共存，导致模型感到疑惑，无法有效学习新的规则。通常只能学习二者中的一个，或者在两者中震荡。
- 结论：需要简单的Experience Replay无法解决这种同状态，不同奖励的冲突。这可能需要更高级的任务感知（Task-Aware）方法或多头输出(Multi-head)策略

### 6.1.2 Task data imbalance
在Appendix D7中讨论了任务时长不一样的情况。
- 设定：任务A很短（0.4M步），任务B很长（2.4M步）
- 结果：即使是水库采样，优于其概率更新机制$N/t$，在长任务$B$结束时，任务$A$在buffer中的样本量也会被压缩的很小
- 数学直觉：在$t$很大的时候，保留早期样本的概率$\approx N/t$趋近于0。导致长任务会“清洗”掉短任务记忆


## 6.2 Conclusion
本文证明了世界模型结合简单的水库采样是一个强大的任务无关的持续学习基线。其性能优于专门设计的CRL方法如CLEAR.

- 关键组件：
  - World Model: 策略在Dreamer中训练，利用了模型的压缩记忆
  - 水库采样：解决FIFO buffer的灾难性遗忘
  - Plan2Explore: 利用认知不确定性（Ensemble variance）解决了新环境的探索问题
- 局限性： 对于完全冲突的奖励函数（Interference）或者极度平衡的任务长度，单纯的Replay Buffer还不够，可能需要meta-learning或者模块化的网络

### 6.3 QA:
整篇文章都在讨论Task-Agnostic(任务无关)，但在实际应用中，任务边界通常是已知的（例如从“训练模式”到“测试模式”，或者用户明确下达了新指令）。那么，这种Task-Agnostic的方法在现实中有优势么？
1. 平滑过渡 (Smooth Transition): 在现实世界（如机器人或自动驾驶）中，环境的变化往往是渐进的（例如：从晴天慢慢变成雨天，路面摩擦力逐渐改变），而不是离散的 Task A -> Task B 切换。Task-Aware 方法在这种渐变边界上会失效（不知道何时切换参数），而 Continual-Dreamer 这种基于流 (Stream) 的方法能自然适应。

2. 未知子任务 (Latent Subtasks): 很多时候，一个大任务内部包含了多个未知的子阶段。Task-Agnostic 的探索机制 (Plan2Explore) 能够自动识别出这些内部的“新奇点”并进行学习，而不需要人工标注每一个子阶段。

3. 无需人工干预: Task-Aware 需要外部信号（Oracle），这限制了智能体的自主性。Task-Agnostic 是通向真正 Open-Ended Learning (开放式学习) 的必经之路。




# 改进:

## 改进1
把$P_{odd}$变成PLR buffer，优先回放那些智能体表现出高学习潜力的关卡。然后看看能不能把Ensemble变成TD-error

为了量化一个关卡的“价值”，作者使用了基于 Temporal Difference (TD) Error 的评分函数。

A. 评分函数 (Score Function)对于一个想象的轨迹，其评分 $score(s_0)$ 定义为整个想象序列中正向TD误差的加权平均：

$$score(s_{0})=\frac{1}{T}\sum_{t=0}^{T}\sum_{k=t}^{T}(\gamma\lambda)^{k-t}\max(0,\delta_{k})$$
- $T$: 想象的片段长度（在探索阶段随机选择）。
- $\delta_{k}$: 时间步 $k$ 的TD误差（TD-error）。
- $\max(0, \delta_{k})$: 关键操作。公式仅累加正向TD误差。正向误差意味着智能体低估了该状态的价值（获得了意外好的结果）。这表明该场景包含了智能体尚未掌握但能够获得高回报的知识。
- $\gamma, \lambda$: 分别为折扣因子（discount factor）和迹衰减参数（trace decay parameter）。

B. 采样概率分布

当算法决定从缓冲区进行回放时，它并非单纯选择分数最高的关卡，而是基于一个混合分布进行采样，平衡了“高分”与“陈旧度”：

$$l_{i}\sim(1-\rho)\cdot P_{S}(l|\Lambda_{seen},S)+\rho\cdot P_{C}(l|\Lambda_{seen},C,c)$$
- $P_{S}$: 基于分数的分布，优先选择具有高学习潜力（高TD误差）的关卡
- $P_{C}$: 基于陈旧度（staleness）的分布，优先选择很久未被采样的关卡，防止过拟合。
- $\rho$: 混合系数（mixture parameter）。

## 改进2：
我们可以让world model自己想办法生成旧数据，这样在多任务的情况下可以保持在不用扩展Buffer的大小的情况下记住之前的数据