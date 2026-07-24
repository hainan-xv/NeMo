#!/bin/bash
# ============================================================================
# Launch oci/eval_leaderboard_slurm.sh for one of our trained models, with the
# SYSTEM_PROMPT copied VERBATIM from that model's training recipe (so decode
# matches training). This picks the prompt/model-class for you -- you don't set
# any variables yourself.
#
# Usage (prints an sbatch command to copy & run from the repo root):
#   bash oci/eval_leaderboard_samples.sh <model> [chunk] [delay]
#     <model> : base | delayprompt | hist1 | contig | interleave
#     [chunk] : encoder frames, default 14  (any trained size: 2 4 7 10 14 28;
#               80ms/frame -> 2=0.16s ... 14=1.12s ... 28=2.24s)
#     [delay] : ONLY for 'delayprompt' -> 0 | 2 | 4   (default 4 = best accuracy)
#
# Examples:
#   bash oci/eval_leaderboard_samples.sh delayprompt 14 4     # delayprompt, chunk 14, delay-4 prompt
#   bash oci/eval_leaderboard_samples.sh delayprompt 7 0      # delayprompt, chunk 7,  delay-0 prompt
#   bash oci/eval_leaderboard_samples.sh base 14              # base chunk-completion model
#   bash oci/eval_leaderboard_samples.sh hist1 14
#   EXP_NAME=granary2_noblank_wer_wlendelay_stochastic \
#     bash oci/eval_leaderboard_samples.sh interleave 14      # interleaving baseline
#
# It PRINTS the ready-to-run sbatch command (it does NOT submit) -- copy the
# printed line and run it from the repo root. Prepend any other
# eval_leaderboard_slurm.sh env to the printed command as needed, e.g.
#   BACKEND=sslm MAX_EVAL_SAMPLES=200 SYSTEM_PROMPT='...' ... sbatch ...
# ============================================================================
set -euo pipefail

MODEL="${1:-}"
CHUNK="${2:-14}"
DELAY="${3:-4}"
if [[ -z "$MODEL" ]]; then
    echo "usage: bash oci/eval_leaderboard_samples.sh <base|delayprompt|hist1|contig|interleave> [chunk] [delay]" >&2
    exit 1
fi

CC=nemo.collections.speechlm2.models.chunk_completion_model.ChunkCompletionSTTModel
INTERLEAVE=nemo.collections.speechlm2.models.StreamingSTTModel

# Prompts VERBATIM from the recipes (examples/speechlm2/conf/...).
P_CC="You are doing streaming speech recognition. Given the transcript so far and the representation of the next audio chunk, output the words spoken in that chunk."
P_DELAY0="You are doing streaming speech recognition. Given the transcript so far and the next audio chunk, output the words spoken in that chunk. Emit the words as soon as possible, minimizing latency."
P_DELAY2="You are doing streaming speech recognition. Given the transcript so far and the next audio chunk, output the words spoken in that chunk. You may wait a little to gather more context before emitting, trading a small delay for better accuracy."
P_DELAY4="You are doing streaming speech recognition. Given the transcript so far and the next audio chunk, output the words spoken in that chunk. Wait until you are confident before emitting, prioritizing accuracy over latency."
P_INTERLEAVE="Transcribe the audio into text."

case "$MODEL" in
    base)   EXP=granary2_chunkcompletion;            MC="$CC"; PROMPT="$P_CC" ;;
    hist1)  EXP=granary2_chunkcompletion_hist1;       MC="$CC"; PROMPT="$P_CC" ;;
    contig) EXP=granary2_chunkcompletion_contig;      MC="$CC"; PROMPT="$P_CC" ;;
    delayprompt)
        EXP=granary2_chunkcompletion_delayprompt;     MC="$CC"
        case "$DELAY" in
            0) PROMPT="$P_DELAY0" ;;
            2) PROMPT="$P_DELAY2" ;;
            4) PROMPT="$P_DELAY4" ;;
            *) echo "ERROR: delay must be 0, 2, or 4 (got '$DELAY')" >&2; exit 1 ;;
        esac ;;
    interleave)
        EXP="${EXP_NAME:-}"; MC="$INTERLEAVE"; PROMPT="$P_INTERLEAVE"
        [[ -n "$EXP" ]] || { echo "ERROR: set EXP_NAME=<exp> for interleave (e.g. granary2_noblank_wer_wlendelay_stochastic)" >&2; exit 1; } ;;
    *) echo "ERROR: unknown model '$MODEL' (base|delayprompt|hist1|contig|interleave)" >&2; exit 1 ;;
esac
# Allow an explicit EXP_NAME override for any model.
EXP="${EXP_NAME:-$EXP}"

# Print ONLY the ready-to-run sbatch command (copy & paste it, from the repo root).
# The prompt is single-quoted so it pastes intact (the prompts contain no single quotes).
echo "SYSTEM_PROMPT='${PROMPT}' MODEL_CLASS=${MC} CHUNK_SIZE=${CHUNK} sbatch oci/eval_leaderboard_slurm.sh ${EXP}"
