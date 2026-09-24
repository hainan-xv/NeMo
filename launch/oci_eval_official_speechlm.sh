#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:oci-off-speechlm-interleaved
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
# OFFICIAL Open-ASR-Leaderboard eval of the INTERLEAVED streaming SpeechLM.
#
#   sbatch launch/oci_eval_official_speechlm.sh
#
# The colleague's baseline: one causal audio-text-audio-text stream
# (StreamingSTTModel), against SCRIPT's spine-plus-branches layout. It shares
# our base LLM (Qwen3-1.7B) and encoder (nemotron-speech-streaming-en-0.6b), so
# the delta against our rows isolates the LAYOUT rather than the ingredients.
# That is what makes it worth carrying onto the official board.
#
# This checkpoint was previously only ever scored by our INTERNAL harness. Every
# published comparison against it therefore used a different normalizer, metric
# and dataset list from the official table -- i.e. it was never actually
# comparable. This puts it on the same seven datasets and the same kaldialign
# scorer as every other row.
#
# It is heh's checkpoint and is READ-ONLY to us: the backend hard-links it into
# our own tree before decoding and never writes to his directory.
#
# DECODE DEFAULTS LIVE IN scripts/speechlm_asr_shim.py, not here, and each is
# measured -- FSM decode (6.03 macro vs 6.63 chunked), streaming embeddings
# (5.51 vs 17.79 offline), and the model's OWN system prompt. Getting any of
# them wrong understates the baseline instead of failing loudly, which would
# quietly flatter our own arms.
#
# pad 0.5: it is a streaming model whose emission lags the audio.
# ============================================================================
set -uo pipefail
mkdir -p slurm_out

MY=/lustre/fsw/portfolios/nemotron/users/hainanx
HEH=/lustre/fsw/portfolios/llmservice/users/heh

resolve_launch_dir() {
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        [[ -f "${SLURM_SUBMIT_DIR}/oci_eval_official_backend.sh" ]] && { echo "${SLURM_SUBMIT_DIR}"; return; }
        [[ -f "${SLURM_SUBMIT_DIR}/launch/oci_eval_official_backend.sh" ]] && { echo "${SLURM_SUBMIT_DIR}/launch"; return; }
    fi
    echo "${MY}/NeMo_SCRIPT_cc/launch"
}

HEH_EXP=oci_streaming_stt_granary2_lora_mcs_noblank_v2_lr0.0001_warmup10000_n8_delay3_rnd_compacttrue_r1_t1
CKPT="${CKPT:-${HEH}/results/Streaming_SLM_debug/${HEH_EXP}/${HEH_EXP}/checkpoints/step=200000-last.ckpt}"

exec bash "$(resolve_launch_dir)/oci_eval_official_backend.sh" \
    speechlm_interleaved \
    "${CKPT}" \
    0.5 0 14 speechlm
