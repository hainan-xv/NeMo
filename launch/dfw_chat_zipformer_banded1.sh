#!/bin/bash
#SBATCH -A nemotron_speechprod_asr
#SBATCH -J nemotron_speechprod_asr:dfw-chat-zipformer-banded1
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
# CHAT with a ZIPFORMER encoder, band=1, delay 0. STANDALONE CHAT -- this uses
# the ASR-collection model, NOT the speechlm2 joint framework at 1:0.
#
#   sbatch launch/dfw_chat_zipformer_banded1.sh    <- no arguments
#
# TRAINED FROM SCRATCH. Every strong result in this project warm-starts from a
# Conformer donor (nemotron-0.6b / dfw_granary2_chat_forced), and no Zipformer
# checkpoint exists to start from. The findings doc is explicit that
# initialisation DOMINATES here -- warm-started arms reach ~0.19 in a few
# thousand steps where from-scratch needed 38k-112k. So a direct comparison
# against the 5.47-macro Conformer CHAT is NOT apples-to-apples until this has
# run far longer than that baseline did.
#
# FRAME RATE DIFFERS. Upstream Zipformer asserts output_downsampling_factor==2,
# so total subsampling is 4x -> 40 ms frames, against the Conformer's 8x/80 ms.
# The config compensates in SECONDS, not frames:
#     joint.chunk_size   28 x 0.04 = 1.12 s   (Conformer: 14 x 0.08)
#     left context      280 x 0.02 = 5.60 s   (Conformer: 70 x 0.08)
#     forced_alignment.frame_length_in_secs = 0.04, set EXPLICITLY
# Note this means MORE encoder frames, the opposite direction to the stacking
# experiments -- worth remembering when reading cost numbers.
#
# k2 IS NOT INSTALLED, so the vendored scaling.py falls back to the pure-PyTorch
# SwooshL/SwooshR definitions. Forward maths is identical; backward goes through
# autograd rather than k2's fused kernel, costing memory and a little speed.
#
# NOT PORTED: ScaledAdam. Upstream reports material degradation without it, so
# this run is Zipformer-the-architecture on NeMo's optimizer, not the full
# upstream recipe. Treat a poor result as inconclusive on that axis.
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

export CONFIG_NAME=nemotron_chat_transducer_granary2_qwen_zipformer
export EXP_NAME="${EXP_NAME:-dfw_chat_zipformer_banded1}"
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
export INIT_NEMO="${INIT_NEMO:-none}"

if [[ "${INIT_NEMO}" != "none" && ! -f "${INIT_NEMO}" ]]; then
    echo "ERROR: warm-start model not found: ${INIT_NEMO}" >&2
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
