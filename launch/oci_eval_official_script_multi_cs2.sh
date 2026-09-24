#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:oci-off-script-multi-cs2
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 04:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --overcommit
#SBATCH --mail-type=FAIL
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# OFFICIAL Open-ASR-Leaderboard eval (kaldialign + merge_compounds) on OCI.
#
#   sbatch launch/oci_eval_official_script_multi_cs2.sh
#
# SCRIPT multi-lookahead, decoded at CHUNK SIZE 2 -- the lowest-latency operating
# point, from the SAME checkpoint. See oci_eval_official_script_multi_cs14.sh.
#
# Expect the weakest row of the three: on the internal aggregate cs2 scored 9.50
# against 6.75 at cs7 and 6.01 at cs14. That ordering is the point of the arm.
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
        [[ -f "${SLURM_SUBMIT_DIR}/oci_eval_official_backend_sharded.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/oci_eval_official_backend_sharded.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    echo "${MY}/NeMo_SCRIPT_cc/launch"
}

exec bash "$(resolve_launch_dir)/oci_eval_official_backend_sharded.sh" \
    script_multi_cs2 \
    "${MY}/results/SpeechlmOCI/granary2_script_banded1_both_multilookahead/granary2_script_banded1_both_multilookahead/checkpoints/granary2_script_banded1_both_multilookahead-averaged.ckpt" \
    0.5 0 2
