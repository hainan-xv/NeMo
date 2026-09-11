#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:eval-nemotron-base
# Eval jobs go to the INTERACTIVE partition: they are single-node and short,
# and the batch blocks queue behind 8-node training for hours. The admin limit
# is ONE interactive job per user at a time, so do not launch two evals at once.
#SBATCH -p interactive
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 02:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# Leaderboard eval of the DONOR nemotron streaming RNN-T.
#
#   sbatch launch/eval_nemotron_baseline.sh          <- no arguments
#
# WHY THIS EXISTS. Every CHAT number in this project is compared against a
# nemotron macro of 5.81 that is not reproducible from this results tree: all 40
# experiment directories were searched, and no run_config.yaml anywhere
# references the donor .nemo. So the figure predates this harness and may have
# come from different code, a different model file or a different dataset list.
#
# This produces a nemotron number from the SAME driver, datasets, shard split
# and scorer that produced the CHAT numbers, so the comparison rests on one
# pipeline rather than on a remembered value.
#
# chunk_size 14 matches the CHAT arms' emission grid; the donor's own .nemo
# already defaults to att_context_size [70, 13], so this changes nothing for it
# and simply makes the setting explicit.
# ============================================================================

export EXP_NAME=nemotron_streaming_0.6b_baseline
export MODE=offline
export CHUNK_SIZE=14

find_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/eval_nemotron.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/eval_nemotron.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/eval_nemotron.sh" ]] && { echo "${here}"; return; }
    echo "ERROR: cannot locate eval_nemotron.sh" >&2; exit 1
}
exec bash "$(find_launch_dir)/eval_nemotron.sh"
