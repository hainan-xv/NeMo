#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-chat-rnnt
#SBATCH -p batch
#SBATCH -N 4
#SBATCH --gpus-per-node=8
#SBATCH -t 04:00:00
#SBATCH --time-min 04:00:00
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=8
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j
# pool0-01815 carries an old NVIDIA driver (12020) and fails torch's CUDA init
# outright; it killed job 18686485 in 103s. pool0-00407 is the reference recipe's
# known-bad node.
#SBATCH --exclude=pool0-00407,pool0-01815

# ============================================================================
# BASELINE 1 of 3 on DFW: STANDARD CHAT.
#
#   sbatch launch/dfw_chat_rnnt.sh          <- no arguments
#
# The plain RNN-T objective: the loss marginalises over EVERY alignment path in
# the lattice, so the aligner's word timings are never used as targets. This is
# the reference the two alignment-based losses have to beat -- if forced or
# banded cannot improve on marginalising, the alignment is not carrying
# information worth the constraint.
#
# loss_type=rnnt makes every forced_alignment.* knob inert; they are left at the
# body's defaults rather than being set to misleading values here.
#
# Fixed chunk_size 14 (1.12 s), like every current CHAT arm. The encoder's
# att_context_size right context is derived from it in the config.
# ============================================================================

export LOSS_TYPE=rnnt
export HISTORY_CHUNKS=0
export LR=1e-4
export WARMUP_STEPS=5000
export MAX_STEPS=500000
export EPOCH_STEPS=2000

# 1,024-piece vocabulary, NOT the Qwen 151,669 one the other arms use.
#
# This is forced, not a preference. The full RNN-T loss marginalises over the
# whole lattice and needs a [B, T, U+1, V+1] joint tensor; at V=151,670 that is
# ~2 GB per sample for the joint output alone. Job 18617613 proved it: the model
# built fine (812M params), passed the sanity check, then sat at step 0/2000 for
# ten minutes without completing ONE step -- no OOM, just intractable. The banded
# loss exists precisely because of this, scoring ~3U+T nodes instead of T*U*V.
#
# CONSEQUENCE FOR THE COMPARISON: this arm differs from dfw_chat_forced.sh in
# BOTH vocabulary and objective, so the gap between them is not attributable to
# either alone. Treat it as a loose reference for what marginalising buys, not as
# a controlled A/B. A clean rnnt-vs-forced pair would need forced re-run at 1k.
#
# The tokenizer is EXTRACTED from the donor .nemo into /results/tokenizer by the
# body, since the 1k SentencePiece ships inside that checkpoint rather than as a
# standalone directory.
export CONFIG_NAME=nemotron_chat_transducer_granary2
export TOKENIZER_DIR=/results/tokenizer
export EXP_NAME="${EXP_NAME:-dfw_granary2_chat_rnnt_1k}"
# Default exclude: at 1k the donor's prediction embedding and joint output DO
# have counterparts (same vocabulary), so only the embedding is left random --
# unlike the Qwen arms where both are 148x larger and cannot be warm-started.

find_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/dfw_chat_train.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/dfw_chat_train.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [[ -f "${here}/dfw_chat_train.sh" ]] && { echo "${here}"; return; }
    # ABSOLUTE fallback, and it is not belt-and-braces -- it is the only thing that
    # works on a REQUEUE. Slurm hands the requeued job SLURM_SUBMIT_DIR pointing at
    # the scratch ROOT rather than the directory the job was submitted from, and $0
    # is the spool copy, so both of the lookups above miss. Both CHAT arms died this
    # way every 17 minutes for three hours (exit 127) after their first 4h wall,
    # while the self-contained SCRIPT arms requeued fine.
    local repo="${DFW_CODE_DIR:-/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr/hainanx/NeMo_SCRIPT_cc}"
    [[ -f "${repo}/launch/dfw_chat_train.sh" ]] && { echo "${repo}/launch"; return; }
    echo "ERROR: cannot locate dfw_chat_train.sh (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}, repo=${repo})" >&2
    exit 1
}
exec bash "$(find_launch_dir)/dfw_chat_train.sh" "$@"
