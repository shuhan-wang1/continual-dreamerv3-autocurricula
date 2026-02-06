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
