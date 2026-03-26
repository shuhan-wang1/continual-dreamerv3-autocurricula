#!/usr/bin/env python3
"""
Submit all ablation experiments to UCL Myriad via qsub.

Each experiment+seed pair is submitted as a separate GPU job.
Jobs auto-resubmit when hitting the 48h walltime limit, resuming
from checkpoints until 1B steps are complete.

Usage:
  python submit_myriad_ablation.py                    # submit all 42 jobs
  python submit_myriad_ablation.py --dry_run           # print commands only
  python submit_myriad_ablation.py --only A            # submit Group A only
  python submit_myriad_ablation.py --only A0_5050_baseline  # submit one experiment
  python submit_myriad_ablation.py --skip D,F          # skip groups D and F
"""

import argparse
import subprocess
import sys
from collections import OrderedDict

# Import experiment definitions from run_ablation.py
sys.path.insert(0, '.')
from run_ablation import EXPERIMENTS, SEEDS, filter_experiments, print_experiment_table


JOB_SCRIPT = "myriad_ablation_job.sh"


def main():
    parser = argparse.ArgumentParser(
        description="Submit ablation experiments to UCL Myriad",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry_run", action="store_true",
                        help="Print qsub commands without executing.")
    parser.add_argument("--only", type=str, default=None,
                        help="Submit only these experiments (comma-separated IDs or group letters).")
    parser.add_argument("--skip", type=str, default=None,
                        help="Skip these experiments (comma-separated IDs or group letters).")
    parser.add_argument("--seeds", type=str, default="1,4,42",
                        help="Comma-separated seeds (default: 1,4,42).")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    experiments = filter_experiments(EXPERIMENTS, only=args.only, skip=args.skip)

    if not experiments:
        print("ERROR: No experiments selected. Check --only / --skip flags.")
        sys.exit(1)

    print_experiment_table(experiments, seeds)
    total = len(experiments) * len(seeds)

    if args.dry_run:
        print(f"[DRY RUN] Would submit {total} jobs:\n")

    submitted = 0
    for exp_id in experiments:
        for seed in seeds:
            job_name = f"abl-{exp_id}-s{seed}"
            cmd = [
                "qsub",
                "-N", job_name,
                "-v", f"EXP_ID={exp_id},SEED={seed}",
                JOB_SCRIPT,
            ]
            cmd_str = " ".join(cmd)

            if args.dry_run:
                print(f"  [{submitted + 1:02d}/{total}] {cmd_str}")
                submitted += 1
                continue

            print(f"[{submitted + 1:02d}/{total}] Submitting {exp_id} seed={seed}...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                job_id = result.stdout.strip()
                print(f"  Submitted: {job_id}")
                submitted += 1
            else:
                print(f"  FAILED: {result.stderr.strip()}")

    print(f"\n{'=' * 60}")
    if args.dry_run:
        print(f"[DRY RUN] {total} jobs would be submitted. No commands executed.")
    else:
        print(f"Submitted {submitted}/{total} jobs to Myriad.")
        print(f"\nMonitor with:")
        print(f"  qstat                          # list all your jobs")
        print(f"  qstat -j <job_id>              # detailed job info")
        print(f"  tail -f logs/abl-*.<job_id>.out  # follow job output")
        print(f"\nCancel all ablation jobs:")
        print(f"  qstat | grep 'abl-' | awk '{{print $1}}' | xargs qdel")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
