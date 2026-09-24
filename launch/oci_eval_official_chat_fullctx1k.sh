#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:oci-off-chat-fullctx1k
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
#   sbatch launch/oci_eval_official_chat_fullctx1k.sh
#
# CHAT full-context, 1k purpose-built SentencePiece vocabulary. The DFW twin of
# this arm is the best row on that board (4.316 macro-7).
#
# pad 0: a non-causal encoder emits nothing until the audio ends, so there is no
# lagging tail for trailing silence to flush. Padding it would only add duration.
#
# max_symbols 30 IS THE ONE DECODE OVERRIDE HERE, and it is not cosmetic.
# max_symbols is a PER-CHUNK cap on emitted tokens; a 1k vocabulary needs ~23
# tokens on a probe sentence where 16k needs 13, so at the default cap this arm
# would be penalised for TRUNCATION rather than for errors. 30 clears the
# measured 23 with margin. Every other arm keeps its own value (0 = unchanged).
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
    chat_fullctx1k \
    "${MY}/results/SpeechlmOCI/oci_granary2_chat_spe1k_both_fullctx/averaged/top5-averaged.nemo" \
    0 30 14
