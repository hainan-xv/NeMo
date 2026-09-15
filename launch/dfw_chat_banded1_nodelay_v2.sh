#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-chat-banded1-nodelay-v2
#SBATCH -p batch
#SBATCH -N 2
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
# CHAT, right-band=1, on the UNMODIFIED alignment, warm-started.
#
#   sbatch launch/dfw_chat_banded1_nodelay.sh      <- no arguments
#
# THREE DELIBERATE DIFFERENCES from dfw_chat_banded1.sh, and they are the whole
# point of this arm:
#
#   1. delay = 0, NOT 3.  This is the correction that motivated the arm. The
#      band was never centred on the aligner's own output. assign_words_to_chunks
#      computes `ready = ceil(end / frame_length) + num_delay_frames` and picks
#      the chunk from THAT, so the delay moved each word before banding; the
#      per-chunk token counts handed to band_nodes were already shifted. With
#      band_side=later the two compounded -- delay pushed a word later, the band
#      could only push it later still -- so nothing in that configuration could
#      place a word where the aligner actually put it. At delay=0 the band is
#      centred on the unmodified alignment.
#
#      NOTE this also sets frame_trim to 0: the model derives it from
#      num_delay_frames, so decode and training move together automatically and
#      this arm must be EVALUATED at trim 0, unlike dfw_chat_forced.
#
#   2. Warm start from the standard CHAT arm -- Qwen 151k vocab with the
#      forced-alignment CE loss (dfw_granary2_chat_forced), via its top-5
#      averaged .nemo. Same vocabulary and same architecture, so the encoder,
#      decoder and joint all transfer; only the loss changes. The 1k-vocab rnnt
#      arm is NOT the right seed here despite also being a "standard" arm -- its
#      1,024-piece vocabulary cannot load into a 151k-vocab decoder embedding or
#      joint output layer.
#
#   3. Two nodes, not four, so more arms run concurrently within the 8-node cap.
#      NOTE this scales the global batch with it, so steps here are NOT
#      equivalent to steps in the 4-node arms. Compare wall-clock or
#      samples-seen, not the step counter, across the two groups.
#
# Everything else is held fixed against dfw_chat_banded1.sh -- band_chunks=1,
# band_side=later, Qwen config, partition targets, LR, warmup -- so the delta
# against that arm's 5.76 macro is the delay change plus the warm start.
# ============================================================================

export LOSS_TYPE=banded
export DELAY_FRAMES=0
export RECOVER_WORDS=0
export BAND_CHUNKS=1
export BAND_SIDE=later
export HISTORY_CHUNKS=0
export MAX_DELAY_FRAMES=0
export TARGET_CONSTRUCTION=partition
export DELAY_PUNCT=true
export WARMUP_STEPS=5000
export MAX_STEPS=500000
export EPOCH_STEPS=2000

# --- v2 CHANGES: bigger batch, smaller LR -----------------------------------
#
# BATCH x2, MEASURED not guessed. nvidia-smi on the running v1 arms showed
# ~30 GiB used of 81 GiB per H100 -- 37%. Doubling every bucket puts the
# activation share at roughly 2x while the fixed weight/optimizer share is
# unchanged, which lands near 50 GiB and leaves real headroom for the long-
# utterance tail. This project has OOM'd four times from batch sizes reasoned
# about rather than measured, hence the reading above rather than a ratio.
#
# It also RESTORES THE GLOBAL BATCH the node cut took away: 2 nodes x 2x batch
# equals the 4-node arms' global batch exactly. So these runs are once again
# comparable in samples-per-step to everything measured before the cut, and the
# variance introduced by halving the nodes is removed at the source.
#
# The list is ABSOLUTE and doubles the YAML's
#   [38, 29, 25, 22, 20, 18, 17, 15, 14, 13, 12, 11, 10, 8, 7, 6, 5, 4]
# which is duplicated here deliberately: an absolute override cannot silently
# become wrong if the YAML is retuned, whereas a multiplier could.
export BUCKET_BATCH_SIZE='[76,58,50,44,40,36,34,30,28,26,24,22,20,16,14,12,10,8]'
#
# LR 1e-4 -> 5e-5. Two reasons, and the second is the stronger one:
#   1. it is the conservative direction while the batch change beds in;
#   2. these arms are WARM STARTS from a converged model, but 1e-4 with a
#      5000-step warmup is a FROM-SCRATCH schedule. Applied to trained weights
#      that is large enough to walk the initialisation back before it helps,
#      which is a plausible reading of the flat trajectory we are reacting to.
# NOTE this is deliberately on the conservative side: with the global batch
# restored above, strict linear scaling would justify keeping 1e-4. If these
# arms now look SLOW rather than noisy, raising LR is the first thing to try.
export LR=5e-5
# ---------------------------------------------------------------------------

export CONFIG_NAME=nemotron_chat_transducer_granary2_qwen
export EXP_NAME="${EXP_NAME:-dfw_granary2_chat_banded1_nodelay_v2}"
# dfw_chat_train.sh defaults DESIGN_NODES to 4 and cannot grep it from $0 (it is
# exec'd from this wrapper). Left unset, a 2-node allocation would be treated as
# an undersized smoke test and silently renamed to ..._n2.
export DESIGN_NODES=2
# FULL transfer, unlike the donor-seeded arms. The seed is dfw_granary2_chat_forced
# -- the same nemotron_chat_transducer_granary2_qwen config, so the same 151k
# vocabulary and the same tensor shapes throughout. The usual exclusions exist
# only to survive a vocabulary change and would here discard the 151k embedding
# and the 151k joint output projection, which is most of what we are warm
# starting FOR.
export INIT_INCLUDE='["encoder.","decoder.","joint."]'
export INIT_EXCLUDE='[]'

# The standard CHAT arm's top-5 average, produced by its own leaderboard eval.
# A .nemo (not a .ckpt) because init_from_nemo_model is what the CHAT trainer
# exposes; the eval pipeline already builds exactly this artifact.
DFW=/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr
export INIT_NEMO="${INIT_NEMO:-${DFW}/hainanx/results/SpeechlmDFW/dfw_granary2_chat_forced/averaged/top5-averaged.nemo}"

if [[ ! -f "${INIT_NEMO}" ]]; then
    echo "ERROR: warm-start model not found: ${INIT_NEMO}" >&2
    echo "       Build it by evaluating the standard arm:" >&2
    echo "       ARMS=dfw_granary2_chat_forced sbatch launch/dfw_eval_chat.sh" >&2
    exit 1
fi

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
    # is the spool copy, so both of the lookups above miss.
    local repo="${DFW_CODE_DIR:-${DFW}/hainanx/NeMo_SCRIPT_cc}"
    [[ -f "${repo}/launch/dfw_chat_train.sh" ]] && { echo "${repo}/launch"; return; }
    echo "ERROR: cannot locate dfw_chat_train.sh (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}, repo=${repo})" >&2
    exit 1
}
exec bash "$(find_launch_dir)/dfw_chat_train.sh" "$@"
