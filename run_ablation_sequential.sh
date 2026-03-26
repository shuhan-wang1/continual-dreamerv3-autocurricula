#!/bin/bash
# =============================================================
# Sequential Ablation Orchestrator (run inside tmux)
#
# Submits ONE experiment (3 seeds) at a time.
# Waits for all 3 seeds to finish, then submits the next experiment.
# Each seed auto-resubmits itself (6h walltime chunks) until 1B steps done.
#
# Usage:
#   tmux new -s ablation
#   bash run_ablation_sequential.sh
#   # Ctrl-b d to detach; tmux attach -t ablation to re-attach
# =============================================================

set -e

PROJECT_DIR="$HOME/Scratch/projects/continual-dreamerv3-autocurricula"
cd "$PROJECT_DIR"

JOB_SCRIPT="myriad_ablation_job.sh"
SEEDS=(1 4 42)
LOG="logs/orchestrator.log"
mkdir -p logs

# Ordered experiment list
EXPERIMENTS=(
    A1_baseline
    A2_p2e
    A3_intrinsic
    A4_p2e_intrinsic
    B1_spatial_only
    B2_craft_only
    D1_nlr
    D2_nlu
    D3_nlr_priv
    D4_nlu_priv
    E1_nlr_intrinsic
    F1_mask_soft
    F2_mask_hard
)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

# ------------------------------------------------------------------
# wait_for_jobs: poll until none of the given job IDs are running/queued
# ------------------------------------------------------------------
wait_for_jobs() {
    local job_ids=("$@")
    while true; do
        local still_active=0
        for jid in "${job_ids[@]}"; do
            # qstat returns non-zero if job doesn't exist (finished)
            if qstat -j "$jid" &>/dev/null; then
                still_active=$((still_active + 1))
            fi
        done
        if [ "$still_active" -eq 0 ]; then
            break
        fi
        log "  ... $still_active / ${#job_ids[@]} jobs still active, checking again in 60s"
        sleep 60
    done
}

# ------------------------------------------------------------------
# check_all_seeds_done: returns 0 if all 3 seeds have completed 1B steps
# (i.e., no auto-resubmitted job is running/queued for this experiment)
# ------------------------------------------------------------------
wait_for_experiment() {
    local exp_id="$1"
    log "  Waiting for $exp_id (all seeds) to finish..."
    while true; do
        # Check if any job for this experiment is still in the queue
        local active
        active=$(qstat 2>/dev/null | grep "abl-${exp_id}" | wc -l)
        if [ "$active" -eq 0 ]; then
            log "  All jobs for $exp_id finished."
            break
        fi
        log "  ... $active job(s) still active for $exp_id, checking in 120s"
        sleep 120
    done
}

# ==================================================================
# Main loop
# ==================================================================
log "=========================================="
log "  Sequential Ablation Orchestrator"
log "  Experiments: ${#EXPERIMENTS[@]}"
log "  Seeds: ${SEEDS[*]}"
log "  Walltime per job: 6h"
log "=========================================="

TOTAL=${#EXPERIMENTS[@]}
IDX=0

for EXP_ID in "${EXPERIMENTS[@]}"; do
    IDX=$((IDX + 1))
    log ""
    log "====== [$IDX/$TOTAL] Submitting: $EXP_ID (seeds: ${SEEDS[*]}) ======"

    JOB_IDS=()
    for SEED in "${SEEDS[@]}"; do
        JOB_NAME="abl-${EXP_ID}-s${SEED}"
        OUTPUT=$(qsub -N "$JOB_NAME" -v "EXP_ID=${EXP_ID},SEED=${SEED}" "$JOB_SCRIPT" 2>&1)
        # Extract job ID from "Your job 123456 (...) has been submitted"
        JID=$(echo "$OUTPUT" | grep -oP 'job \K[0-9]+')
        JOB_IDS+=("$JID")
        log "  Submitted seed=$SEED -> job $JID"
    done

    # Wait for the initial 3 jobs to finish
    log "  Waiting for initial 3 jobs: ${JOB_IDS[*]}"
    wait_for_jobs "${JOB_IDS[@]}"

    # Now wait for any auto-resubmitted jobs for this experiment to finish
    # (auto-resubmit creates new jobs with the same name prefix)
    wait_for_experiment "$EXP_ID"

    log "  $EXP_ID COMPLETE."
done

log ""
log "=========================================="
log "  ALL ${TOTAL} EXPERIMENTS COMPLETE"
log "  Results: experiment_results/ablation/"
log "=========================================="
