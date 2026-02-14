import argparse
import json
import math
import os
from typing import List, Dict, Any

import matplotlib.pyplot as plt


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _safe_float(x, default=float('nan')):
    try:
        return float(x)
    except Exception:
        return default


def _plot_series(x, y, title, ylabel, out_path):
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, linewidth=1.5)
    plt.title(title)
    plt.xlabel('step')
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_per_task(x, per_task, title, out_path):
    plt.figure(figsize=(8, 4))
    num_tasks = len(per_task[0]) if per_task else 0
    for i in range(num_tasks):
        y = [row[i] for row in per_task]
        plt.plot(x, y, linewidth=1.2, label=f'task_{i}')
    plt.title(title)
    plt.xlabel('step')
    plt.ylabel('score')
    if num_tasks <= 10:
        plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot online continual-learning metrics.')
    parser.add_argument('--logdir', type=str, required=True, help='Log directory containing online_metrics.jsonl')
    parser.add_argument('--jsonl', type=str, default='online_metrics.jsonl', help='JSONL filename')
    parser.add_argument('--outdir', type=str, default=None, help='Output directory for plots')
    args = parser.parse_args()

    jsonl_path = args.jsonl
    if not os.path.isabs(jsonl_path):
        jsonl_path = os.path.join(args.logdir, jsonl_path)

    outdir = args.outdir or os.path.join(args.logdir, 'plots')
    os.makedirs(outdir, exist_ok=True)

    records = _load_jsonl(jsonl_path)
    if not records:
        raise ValueError('No records found in JSONL.')

    steps = [int(r.get('step', 0)) for r in records]
    ap = [_safe_float(r.get('ap', float('nan'))) for r in records]
    forgetting = [_safe_float(r.get('forgetting', float('nan'))) for r in records]
    ft = [_safe_float(r.get('ft', float('nan'))) for r in records]
    per_task_latest = [r.get('per_task_latest', []) for r in records]

    _plot_series(steps, ap, 'Average Performance (AP)', 'AP', os.path.join(outdir, 'ap.png'))
    _plot_series(steps, forgetting, 'Average Forgetting (F)', 'F', os.path.join(outdir, 'forgetting.png'))
    if not all(math.isnan(v) for v in ft):
        _plot_series(steps, ft, 'Forward Transfer (FT)', 'FT', os.path.join(outdir, 'ft.png'))

    if per_task_latest and len(per_task_latest[0]) > 0:
        _plot_per_task(steps, per_task_latest, 'Per-task Episode Score', os.path.join(outdir, 'per_task_latest.png'))

    last = records[-1]
    print('Last step:', last.get('step'))
    print('AP:', last.get('ap'))
    print('Forgetting:', last.get('forgetting'))
    print('FT:', last.get('ft'))
    print('Plots saved to:', outdir)


if __name__ == '__main__':
    main()
