#!/bin/bash
# =============================================================================
# SDE vs ODE joint scaling comparison — SLURM array job
#
# Compares two ways of allocating parameters under the same complete_p_sde
# parametrization (same init rules, same forward multipliers):
#
#   SDE ray (tasks 0–4): --scaling=sde, width_mult=W, depth_mult=1.0
#     N→N·W, H→H·W, M→M·W, L fixed (N = H·head_dim always)
#     alpha_mlp = N/(M·L) and alpha_att = N/(H·L) stay CONSTANT
#     Non-embedding params ∝ W²
#
#   ODE ray (tasks 5–9): --scaling=linear, linear_mult=s
#     N→N·s, H→H·s, M→M·s, L→L·s (everything scales together)
#     alpha_mlp, alpha_att fall like 1/s (ODE / drift-only depth limit)
#     Non-embedding params ∝ s³
#
# Iso-parameter pairing:  ODE s = SDE W^(2/3)
# Tasks with the same index (mod 5) have equal approximate parameter count.
#
#   idx │ SDE W   │ ODE s   │ param ratio (non-embed)
#   ────┼─────────┼─────────┼────────────────────────
#    0  │  0.500  │  0.630  │  0.25 × base
#    1  │  1.000  │  1.000  │  1.00 × base
#    2  │  1.250  │  1.157  │  1.56 × base
#    3  │  1.500  │  1.310  │  2.25 × base
#    4  │  2.000  │  1.587  │  4.00 × base
# =============================================================================

#SBATCH --job-name=sde-ode-scale
#SBATCH --array=0-9%5           # 10 jobs, max 5 concurrent
#SBATCH -N 1
#SBATCH --gres=gpu:3
#SBATCH --ntasks-per-node=3
#SBATCH -t 24:00:00
#SBATCH -p gpu-a100             # TACC Lonestar6; adjust for other clusters
#SBATCH -A <your-allocation>    # ← fill in your TACC allocation
#SBATCH -o logs/sde_ode_%A_%a.out
#SBATCH -e logs/sde_ode_%A_%a.err

# ── Multiplier tables ─────────────────────────────────────────────────────────
# SDE ray: vary width_mult; depth_mult fixed at 1.0
SDE_WIDTH_MULTS=(0.500  1.000  1.250  1.500  2.000)

# ODE ray: linear_mult = SDE_width_mult^(2/3) for iso-parameter comparison
ODE_LINEAR_MULTS=(0.630  1.000  1.157  1.310  1.587)

# ── Job selection ─────────────────────────────────────────────────────────────
TASK=${SLURM_ARRAY_TASK_ID}
IDX=$(( TASK % 5 ))   # index into multiplier arrays (0–4)
RAY=$(( TASK / 5 ))   # 0 → SDE ray, 1 → ODE ray

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO=/home/11423/blakebordelon/olmo-tacc-completeP   # ← adjust if needed
SAVE=/scratch/11423/blakebordelon/ICL_proj/experiments/sde-ode-scaling

# ── Environment ───────────────────────────────────────────────────────────────
cd "$REPO"
source activate olmo    # ← replace with your activate command

mkdir -p logs "$SAVE"

# ── Common training args ──────────────────────────────────────────────────────
COMMON=(
    src/scripts/train/OLMo2-190M-C4.py
    --save-folder="$SAVE"
    --param=sde
    --schedule=constant
    --lr=1e-3
    --total-tokens=4e9
)

# ── Launch ────────────────────────────────────────────────────────────────────
if [ "$RAY" -eq 0 ]; then
    W=${SDE_WIDTH_MULTS[$IDX]}
    echo "Task $TASK — SDE ray: width_mult=$W, depth_mult=1.0"
    torchrun --nproc-per-node=3 "${COMMON[@]}" \
        --scaling=sde \
        --width-mult="$W" \
        --depth-mult=1.0
else
    S=${ODE_LINEAR_MULTS[$IDX]}
    echo "Task $TASK — ODE ray: linear_mult=$S"
    torchrun --nproc-per-node=3 "${COMMON[@]}" \
        --scaling=linear \
        --linear-mult="$S"
fi
