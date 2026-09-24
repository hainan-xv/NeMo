#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:oci-off-script-both
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 06:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --overcommit
#SBATCH --mail-type=FAIL
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# OFFICIAL Open-ASR-Leaderboard eval (kaldialign + merge_compounds) on OCI.
#
#   sbatch launch/oci_eval_official_script_both.sh
#
# SCRIPT banded (both-side), single chunk size 14.
#
# A Lightning .ckpt, not a .nemo: run_eval.py routes it through
# scripts/script_asr_shim.py, which reuses script_leaderboard_eval's own loader
# and generate path, so only the scoring and dataset plumbing differ from our
# internal harness -- the decode itself is unchanged.
#
# pad 0.5 because SCRIPT trains with data.dataset.pad_extra_duration and its
# emission lags the audio; with no trailing silence the final words never flush.
#
# Scores into the SHARED results dir that every oci_eval_official_*.sh writes to;
# run launch/oci_eval_official_score.sh for the combined macro-7 table.
#
# BATCH, not interactive: the interactive QOS enforces QOSMaxJobsPerUserLimit,
# which serialises a fan-out of seven arms into a queue of one.
# ============================================================================
set -uo pipefail
mkdir -p slurm_out

MY=/lustre/fsw/portfolios/nemotron/users/hainanx
HEH=/lustre/fsw/portfolios/llmservice/users/heh

# Under sbatch $0 is a spool copy, so dirname is useless; SLURM_SUBMIT_DIR is
# the directory sbatch was run FROM, which may be the repo root or launch/.
resolve_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/oci_eval_official_backend.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/oci_eval_official_backend.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    echo "${MY}/NeMo_SCRIPT_cc/launch"
}

exec bash "$(resolve_launch_dir)/oci_eval_official_backend.sh" \
    script_both \
    "${MY}/results/SpeechlmOCI/granary2_script_banded1_both/granary2_script_banded1_both/checkpoints/granary2_script_banded1_both-averaged.ckpt" \
    0.5 0 14
