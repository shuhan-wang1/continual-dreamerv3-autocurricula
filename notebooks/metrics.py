from typing import Dict, List, Optional, Union
import json
import os
import numpy as np
import pandas as pd

def smooth(
    scalars: list,
    weight: float, # Weight between 0 and 1
) -> list:  
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def performance(
    data: Dict[int, np.array],
    num_tasks: int=8,
    num_steps: int=1e6,
    num_seeds: int=10,
    smoothing_factor: bool=None,
    verbose: bool=False,
) -> np.array:
    """
    Av performance

    N = num_tasks
    f = \sum_i=1^N p_i
    p_i = perf(num_steps*num_tasks)
    """
    keys = list(data.keys())
    # forgetting
    p_T = np.zeros(num_seeds) # num_seeds
    for i in range(num_tasks):
        # performance data per minihack task
        k = keys[i]
        d = data[k]
        # dataframe construction for different numbers of seeds
        df = {'t': d[:, 0]}
        for j in range(d.shape[1] - 1):
            df['s{}'.format(j)] = d[:, j + 1]
        dataset = pd.DataFrame(df).fillna(method='ffill').fillna(method='bfill')

        # final performance on minihack task i
        if smoothing_factor is None:
            p_i_T = dataset.tail(1).to_numpy()[0, 1:]
        else:
            for j in range(d.shape[1] - 1):
                dataset['s{}'.format(j)] = smooth(dataset['s{}'.format(j)].to_numpy(), smoothing_factor)
            p_i_T = dataset.tail(1).to_numpy()[0, 1:]

        # minihack levels can be negative so let's scale up to 0
        p_i_T[p_i_T <= 0] = 0
        p_T += p_i_T
        if verbose:
            print("Task {0}, Performance {1}".format(i + 1, p_i_T))
    return p_T / num_tasks

def forgetting(
    data: Dict[int, np.array],
    num_tasks: int=8,
    num_steps: int=1e6,
    num_seeds: int=10,
    smoothing_factor: float=None,
    verbose: bool=False,
) -> np.array:
    """
    N = num_tasks
    f = \sum_i=1^N f_i
    f_i = perf(i*num_steps) - perf(num_steps*num_tasks)
    """
    keys = list(data.keys())
    # forgetting
    f = np.zeros(num_seeds)  # num_seeds
    for i in range(num_tasks):
        # performance data per minihack task
        key = keys[i]
        d = data[key]
        # dataframe construction for different numbers of seeds
        df = {'t': d[:, 0]}
        for j in range(d.shape[1] - 1):
            df['s{}'.format(j)] = d[:, j + 1]
        dataset = pd.DataFrame(df).fillna(method='ffill').fillna(method='bfill')

        if smoothing_factor is None:
            f_i = dataset[dataset['t'] <= ((i + 1) * num_steps)].tail(1).to_numpy()[0, 1:]
        else:
            for j in range(num_seeds):
                dataset['s{}'.format(j)] = smooth(dataset['s{}'.format(j)].to_numpy(), smoothing_factor)
        
            f_i = dataset[dataset['t'] <= ((i + 1) * num_steps)].tail(1).to_numpy()[0, 1:]

        f_T = dataset.tail(1).to_numpy()[0, 1:]

        # minihack levels can be negative so let's scale up to 0
        f_T[f_T <= 0] = 0
        f_i[f_i <= 0] = 0
        f += (f_i - f_T)
        if verbose:
            print("Task {0}".format(i+1))
            print("Forgetting {0}, Return {1}, Final Return {2}".format((f_i - f_T), f_i, f_T))
    return f / num_tasks

def integrate(
    dataset: np.array,
    num_seeds: int,
    num_steps: int,
    task: int=None,
    aggregate: bool=False,
) -> np.array:
    # dataframe construction for different numbers of seeds
    df = {'t': dataset[:, 0]}
    for i in range(num_seeds):
        df['s{}'.format(i)] = dataset[:, i + 1]
    dataset = pd.DataFrame(df)

    # rectangle rule for integration
    upper = num_steps if task is None else task * num_steps
    lower = 0 if task is None else (task - 1) * num_steps
    dataset = dataset[(dataset['t'] >= lower) & (dataset['t'] < upper)].fillna(method='ffill').to_numpy()
    auc = np.zeros(1) if aggregate else np.zeros(num_seeds)
    for i in range(1, dataset.shape[0]):
        delta_t = dataset[i, 0] - dataset[i - 1, 0]
        f = dataset[i, 1:]
        # let's max 0 the smallest value of the performance
        f[f <= 0] = 0
        # take the mean of the performance across seeds
        if aggregate:
            f = np.mean(f)
        auc += delta_t * f
    auc /= num_steps
    return auc


def fwd_transfer(
    dones: Dict[str, np.array],
    dones_ref: Dict[str, np.array],
    num_tasks: int=8,
    num_seeds: int=10,
    envs: List = [
        "Room-Random-15x15-v0",
        "Room-Monster-15x15-v0",
        "Room-Trap-15x15-v0",
        "Room-Ultimate-15x15-v0",
        "River-Narrow-v0",
        "River-v0",
        "River-Monster-v0",
        "HideNSeek-v0"
    ],
    num_steps: int=1e6,
    full_range: bool=False,
    verbose: bool=False,
    aggregate_ref_auc: bool=False,
) -> np.array:
    """
    N = num_tasks
    ft = \sum_i=1^N ft_i
    ft_i = auc_i - auc / (1 - auc_i)
    auc_i = are under the curve for particular task in cl loop
    auc = area under the curve for the single task
    ft_i = auc_i - auc / (1 - auc_i)
    """

    ft = np.zeros(num_seeds)
    keys = list(dones.keys())
    for i, e in enumerate(envs):
        if e in list(dones_ref.keys()):
            ref_perf = dones_ref[e]
            key = keys[i]
            cl_perf = dones[key]
            ref_auc = integrate(ref_perf, num_seeds, num_steps, task=None, aggregate=aggregate_ref_auc)
            if full_range:
                cl_auc = integrate(cl_perf, num_seeds, num_steps * num_tasks, task=None, aggregate=False)
            else:
                cl_auc = integrate(cl_perf, num_seeds, num_steps, task=i + 1, aggregate=False)
            ft_i = (cl_auc - ref_auc) / (1 - cl_auc)
            ft += ft_i
            if verbose:
                print("{0} ref auc {1} auc {2}".format(e, ref_auc, cl_auc))
                print("per task ft {0}".format(ft_i))
    return ft / num_tasks


def _clip_perf(value: float) -> float:
    return float(value) if value > 0 else 0.0


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_ref_metrics(path: Optional[str]) -> Dict[str, float]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('ref_auc', {})


def load_ref_metrics_from_paths(paths: Optional[List[str]]) -> Dict[str, float]:
    if not paths:
        return {}
    merged: Dict[str, float] = {}
    idx = 0
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ref = data.get('ref_auc', {})
        if ref:
            for _, v in ref.items():
                merged[str(idx)] = float(v)
                idx += 1
        else:
            if 'auc' in data:
                merged[str(idx)] = float(data['auc'])
                idx += 1
    return merged


def load_ref_metrics_from_dir(directory: Optional[str]) -> Dict[str, float]:
    if not directory or not os.path.isdir(directory):
        return {}
    paths = []
    for name in sorted(os.listdir(directory)):
        if name.endswith('.json') and 'ref_metrics' in name:
            paths.append(os.path.join(directory, name))
    return load_ref_metrics_from_paths(paths)


def load_ref_metrics_from_root(root_dir: Optional[str]) -> Dict[str, float]:
    if not root_dir or not os.path.isdir(root_dir):
        return {}
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name == 'ref_metrics.json':
                paths.append(os.path.join(dirpath, name))
    return load_ref_metrics_from_paths(sorted(paths))


def save_ref_metrics(path: str, ref_auc: Dict[str, float], steps_per_task: Union[int, List[int]]) -> None:
    _ensure_dir(os.path.dirname(path))
    payload = {
        'ref_auc': ref_auc,
        'steps_per_task': steps_per_task,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ============== Craftax Achievement Definitions ==============
# Craftax has 22 achievements organized in tiers (depths)
# These are the canonical achievement names from Craftax
CRAFTAX_ACHIEVEMENTS = [
    # Tier 0 - Basic
    'collect_wood',
    'place_table',
    'eat_cow',
    'collect_sapling',
    'collect_drink',
    'make_wood_pickaxe',
    'make_wood_sword',
    # Tier 1 - Stone
    'place_stone',
    'collect_stone',
    'place_furnace',
    'collect_coal',
    'collect_iron',
    'make_stone_pickaxe',
    'make_stone_sword',
    # Tier 2 - Iron
    'make_iron_pickaxe',
    'make_iron_sword',
    'collect_diamond',
    # Tier 3 - Diamond
    'make_diamond_pickaxe',
    'make_diamond_sword',
    # Tier 4 - Combat
    'defeat_zombie',
    'defeat_skeleton',
    'wake_up_boss',
]

# Achievement tiers for depth calculation
ACHIEVEMENT_TIERS = {
    'collect_wood': 0, 'place_table': 0, 'eat_cow': 0, 'collect_sapling': 0,
    'collect_drink': 0, 'make_wood_pickaxe': 0, 'make_wood_sword': 0,
    'place_stone': 1, 'collect_stone': 1, 'place_furnace': 1, 'collect_coal': 1,
    'collect_iron': 1, 'make_stone_pickaxe': 1, 'make_stone_sword': 1,
    'make_iron_pickaxe': 2, 'make_iron_sword': 2, 'collect_diamond': 2,
    'make_diamond_pickaxe': 3, 'make_diamond_sword': 3,
    'defeat_zombie': 4, 'defeat_skeleton': 4, 'wake_up_boss': 4,
}

NUM_CRAFTAX_ACHIEVEMENTS = len(CRAFTAX_ACHIEVEMENTS)
NUM_ACHIEVEMENT_TIERS = 5  # Tiers 0-4


def compute_achievement_depth(achievements: Dict[str, bool]) -> int:
    """Compute the maximum achievement tier reached.

    Args:
        achievements: Dict mapping achievement name to whether it was achieved.

    Returns:
        Maximum tier reached (0-4), or -1 if no achievements.
    """
    max_tier = -1
    for name, achieved in achievements.items():
        if achieved and name in ACHIEVEMENT_TIERS:
            max_tier = max(max_tier, ACHIEVEMENT_TIERS[name])
    return max_tier


def compute_score_distribution(depths: List[int], num_tiers: int = NUM_ACHIEVEMENT_TIERS) -> List[float]:
    """Compute fraction of episodes at each achievement depth tier.

    Args:
        depths: List of achievement depths for episodes.
        num_tiers: Number of tiers to compute distribution over.

    Returns:
        List of fractions for each tier (including -1 for no achievements).
    """
    if not depths:
        return [0.0] * (num_tiers + 1)  # +1 for tier -1 (no achievements)

    counts = [0] * (num_tiers + 1)
    for d in depths:
        idx = d + 1  # Shift so -1 maps to index 0
        if 0 <= idx < len(counts):
            counts[idx] += 1

    total = len(depths)
    return [c / total for c in counts]


def compute_per_achievement_forgetting(
    peak_rates: Dict[str, float],
    current_rates: Dict[str, float],
) -> Dict[str, float]:
    """Compute forgetting for each achievement.

    F_a = max_{t'<t} p_a(t') - p_a(t) for each achievement a.

    Args:
        peak_rates: Dict of peak success rates for each achievement.
        current_rates: Dict of current success rates for each achievement.

    Returns:
        Dict of forgetting values for each achievement.
    """
    forgetting = {}
    for name in peak_rates:
        peak = peak_rates.get(name, 0.0)
        current = current_rates.get(name, 0.0)
        forgetting[name] = max(0.0, peak - current)
    return forgetting


def compute_aggregate_forgetting(per_achievement_forgetting: Dict[str, float]) -> float:
    """Compute mean forgetting across all achievements.

    Args:
        per_achievement_forgetting: Dict of forgetting values per achievement.

    Returns:
        Mean forgetting across all achievements.
    """
    if not per_achievement_forgetting:
        return 0.0
    return float(np.mean(list(per_achievement_forgetting.values())))


def compute_frontier_rate(
    recent_depths: List[int],
    personal_best_depth: int,
) -> float:
    """Compute fraction of recent episodes reaching a new personal-best depth.

    Args:
        recent_depths: List of achievement depths for recent episodes.
        personal_best_depth: Current personal best depth.

    Returns:
        Fraction of episodes that reached or exceeded personal best.
    """
    if not recent_depths or personal_best_depth < 0:
        return 0.0
    frontier_count = sum(1 for d in recent_depths if d >= personal_best_depth)
    return frontier_count / len(recent_depths)


# ============== Replay Buffer Diagnostics ==============

def compute_buffer_depth_distribution(
    episode_depths: List[int],
    num_tiers: int = NUM_ACHIEVEMENT_TIERS,
) -> List[float]:
    """Compute the distribution of episodes by achievement depth in replay buffer.

    Args:
        episode_depths: List of achievement depths for episodes in buffer.
        num_tiers: Number of achievement tiers.

    Returns:
        List of fractions for each tier (including tier -1 for no achievements).
    """
    return compute_score_distribution(episode_depths, num_tiers)


def compute_buffer_mean_age(
    episode_timestamps: List[int],
    current_step: int,
) -> float:
    """Compute mean age of episodes in replay buffer.

    Args:
        episode_timestamps: List of step numbers when episodes were added.
        current_step: Current training step.

    Returns:
        Mean age in steps.
    """
    if not episode_timestamps:
        return 0.0
    ages = [current_step - ts for ts in episode_timestamps]
    return float(np.mean(ages))


def compute_buffer_td_error_stats(
    td_errors: List[float],
) -> Dict[str, float]:
    """Compute statistics on TD-errors in replay buffer.

    Args:
        td_errors: List of TD-error values.

    Returns:
        Dict with mean, max, std of TD-errors.
    """
    if not td_errors:
        return {'mean': 0.0, 'max': 0.0, 'std': 0.0}
    arr = np.array(td_errors)
    return {
        'mean': float(np.mean(arr)),
        'max': float(np.max(arr)),
        'std': float(np.std(arr)),
    }


# ============== Exploration Diagnostics ==============

def compute_dream_accuracy(
    imagined_values: List[float],
    actual_values: List[float],
) -> float:
    """Compute accuracy of imagination rollouts vs actual rollouts.

    Measures how well the world model predicts actual returns.

    Args:
        imagined_values: List of predicted values from imagination.
        actual_values: List of actual values observed.

    Returns:
        Correlation coefficient between imagined and actual values.
    """
    if not imagined_values or not actual_values:
        return 0.0
    if len(imagined_values) != len(actual_values):
        min_len = min(len(imagined_values), len(actual_values))
        imagined_values = imagined_values[:min_len]
        actual_values = actual_values[:min_len]

    imag = np.array(imagined_values)
    actual = np.array(actual_values)

    # Compute correlation
    if np.std(imag) < 1e-8 or np.std(actual) < 1e-8:
        return 0.0
    corr = np.corrcoef(imag, actual)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def compute_intrinsic_extrinsic_ratio(
    intrinsic_rewards: List[float],
    extrinsic_rewards: List[float],
) -> float:
    """Compute ratio of intrinsic to extrinsic rewards over time.

    Args:
        intrinsic_rewards: List of intrinsic reward values.
        extrinsic_rewards: List of extrinsic reward values.

    Returns:
        Mean ratio of intrinsic/extrinsic rewards.
    """
    if not intrinsic_rewards or not extrinsic_rewards:
        return 0.0

    ratios = []
    for intr, extr in zip(intrinsic_rewards, extrinsic_rewards):
        if abs(extr) > 1e-8:
            ratios.append(intr / extr)
        else:
            ratios.append(0.0)

    return float(np.mean(ratios)) if ratios else 0.0


# ============== Training Metrics Extraction ==============

def extract_training_metrics(mets: Dict) -> Dict[str, float]:
    """Extract relevant training metrics from agent.train() output.

    Args:
        mets: Metrics dict from agent.train().

    Returns:
        Dict with standardized metric names.
    """
    return {
        'loss/obs': float(mets.get('loss/embedding', mets.get('loss/image', 0.0))),
        'loss/rew': float(mets.get('loss/rew', 0.0)),
        'loss/con': float(mets.get('loss/con', 0.0)),
        'loss/dyn': float(mets.get('loss/dyn', 0.0)),
        'loss/rep': float(mets.get('loss/rep', 0.0)),
        'loss/policy': float(mets.get('loss/policy', 0.0)),
        'loss/value': float(mets.get('loss/value', 0.0)),
        'loss/disag': float(mets.get('loss/disag', 0.0)),
        'td_error/mean': float(mets.get('adv', 0.0)),
        'td_error/max': float(mets.get('adv_mag', 0.0)),
        'p2e/intr_rew': float(mets.get('p2e/intr_rew', 0.0)),
        'p2e/extr_rew': float(mets.get('p2e/extr_rew', 0.0)),
        'p2e/combined_rew': float(mets.get('p2e/combined_rew', 0.0)),
        'val': float(mets.get('val', 0.0)),
        'ret': float(mets.get('ret', 0.0)),
        'adv': float(mets.get('adv', 0.0)),
        'adv_std': float(mets.get('adv_std', 0.0)),
    }


class OnlineMetrics:
    """Online metrics calculator aligned with Section 5.2.

    Stores JSONL records for plotting and saves a summary JSON at the end.
    """

    def __init__(
        self,
        logdir: str,
        num_tasks: int,
        steps_per_task: Union[int, List[int]],
        ref_metrics_path: Optional[str] = None,
        ref_metrics_dir: Optional[str] = None,
        ref_metrics_root: Optional[str] = None,
        jsonl_name: str = 'online_metrics.jsonl',
        summary_name: str = 'metrics_summary.json',
    ) -> None:
        self.logdir = str(logdir)
        _ensure_dir(self.logdir)
        self.num_tasks = int(num_tasks)
        if isinstance(steps_per_task, (list, tuple)):
            self.steps_per_task = [int(x) for x in steps_per_task]
        else:
            self.steps_per_task = [int(steps_per_task) for _ in range(self.num_tasks)]

        if ref_metrics_path:
            self.ref_auc = load_ref_metrics(ref_metrics_path)
        elif ref_metrics_dir:
            self.ref_auc = load_ref_metrics_from_dir(ref_metrics_dir)
        elif ref_metrics_root:
            self.ref_auc = load_ref_metrics_from_root(ref_metrics_root)
        else:
            self.ref_auc = {}
        self.jsonl_path = os.path.join(self.logdir, jsonl_name)
        self.summary_path = os.path.join(self.logdir, summary_name)

        self.latest = {i: None for i in range(self.num_tasks)}
        self.end_scores = {i: None for i in range(self.num_tasks)}
        self.auc = {i: 0.0 for i in range(self.num_tasks)}
        self.last_step = {i: None for i in range(self.num_tasks)}
        self.last_score = {i: None for i in range(self.num_tasks)}

    def _steps_for(self, task_id: int) -> int:
        return int(self.steps_per_task[task_id])

    def update(self, task_id: int, step: int, score: float, length: Optional[int] = None) -> Dict[str, float]:
        task_id = int(task_id)
        step = int(step)
        score = _clip_perf(score)

        last_step = self.last_step[task_id]
        last_score = self.last_score[task_id]
        if last_step is not None and step > last_step:
            dt = step - last_step
            self.auc[task_id] += dt * (last_score + score) / 2.0

        self.last_step[task_id] = step
        self.last_score[task_id] = score
        self.latest[task_id] = score

        record = {
            'step': step,
            'task': task_id,
            'score': score,
            'length': None if length is None else int(length),
            'ap': float(self.average_performance()),
            'forgetting': float(self.average_forgetting()),
            'ft': float(self.forward_transfer()),
            'per_task_latest': self.latest_list(),
            'per_task_auc': self.auc_norm_list(),
        }
        with open(self.jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        return record

    def mark_task_end(self, task_id: int) -> None:
        task_id = int(task_id)
        if self.end_scores[task_id] is None:
            self.end_scores[task_id] = self.latest.get(task_id, None)

    def latest_list(self) -> List[float]:
        out = []
        for i in range(self.num_tasks):
            val = self.latest.get(i, None)
            out.append(0.0 if val is None else float(val))
        return out

    def auc_norm_list(self) -> List[float]:
        out = []
        for i in range(self.num_tasks):
            denom = max(1, self._steps_for(i))
            out.append(float(self.auc[i] / denom))
        return out

    def average_performance(self) -> float:
        """AP = (1/T) * sum of p_tau(t) for all tasks seen so far.

        Only includes tasks that have received at least one score update.
        Tasks not yet encountered are excluded from the average (not treated as 0).
        """
        seen = [float(v) for v in self.latest.values() if v is not None]
        if not seen:
            return 0.0
        return float(np.mean(seen))

    def average_forgetting(self) -> float:
        """F = (1/K) * sum_{tau with end_score} (end_score_tau - latest_tau).

        Only includes tasks whose training phase has ended (mark_task_end called).
        Tasks still in training or not yet seen are excluded — no forgetting can
        have occurred for them yet.
        """
        diffs = []
        for i in range(self.num_tasks):
            end_val = self.end_scores.get(i, None)
            if end_val is None:
                # Task i hasn't finished its training phase yet — skip
                continue
            end_val = float(_clip_perf(end_val))
            cur_val = self.latest.get(i, None)
            cur_val = 0.0 if cur_val is None else float(_clip_perf(cur_val))
            diffs.append(end_val - cur_val)
        return float(np.mean(diffs)) if diffs else 0.0

    def forward_transfer(self) -> float:
        if not self.ref_auc:
            return float('nan')
        vals = []
        for i in range(self.num_tasks):
            key = str(i)
            if key not in self.ref_auc:
                continue
            denom = max(1, self._steps_for(i))
            auc_cl = self.auc[i] / denom
            auc_ref = float(self.ref_auc[key])
            vals.append((auc_cl - auc_ref) / (1.0 - auc_ref))
        return float(np.mean(vals)) if vals else float('nan')

    def save_summary(self) -> None:
        payload = {
            'ap': self.average_performance(),
            'forgetting': self.average_forgetting(),
            'ft': self.forward_transfer(),
            'per_task_latest': self.latest_list(),
            'per_task_auc': self.auc_norm_list(),
            'num_tasks': self.num_tasks,
        }
        with open(self.summary_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
