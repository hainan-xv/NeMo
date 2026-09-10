#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:probe-val
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 01:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# Score both RNN-T arms on the TRAINING validation manifest, through the same
# transcribe() path the leaderboard uses.
#
#   sbatch launch/probe_val_wer.sh
#
# val_wer as logged in training and the leaderboard macro disagree for these two
# models, and they come from different code paths -- the Lightning validation
# loop versus transcribe() with an explicit frame_trim. Running the same
# manifest through the leaderboard's path settles whether the near-identical
# val_wer is real or an artefact of how it was measured.
# ============================================================================
set -uo pipefail

OUTPUT_PREFIX=/lustre/fsw/portfolios/nemotron/users/hainanx
B=${OUTPUT_PREFIX}/results/SpeechlmScriptCC
VAL=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/users/dongjig/aligned_amos/steve_val_mmlpc_mcv11_2k/mcv11_dev_clean_pcstrip_en_2k_qwen_aligned.json
CONTAINER=/lustre/fsw/portfolios/llmservice/users/heh/containers/nemo-26.02-streaming-speechlm.sqsh
CODE_DIR=${CODE_DIR:-/lustre/fsw/portfolios/nemotron/users/hainanx/NeMo_SCRIPT_cc}
DATA_DIR=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data
MOUNTS="--container-mounts=${CODE_DIR}:/code,${OUTPUT_PREFIX}:${OUTPUT_PREFIX},/lustre/fsw/portfolios/llmservice:/lustre/fsw/portfolios/llmservice,${DATA_DIR}:/data"

srun --ntasks=1 --container-image="$CONTAINER" $MOUNTS bash -c "
cd /code && export PYTHONPATH=/code:\$PYTHONPATH
LS=/lustre/fsw/portfolios/llmservice/users/hainanx/leaderboard_cache/librispeech/test.clean/_cache_manifest.jsonl
echo '################ librispeech test.clean -- PLAIN model, swept over trim'
echo '### it was never trained with trimming; if trimming still helps, the last'
echo '### frame of each chunk is harmful and this is a chunking bug, not a skill'
python scripts/chat_val_probe.py --nemo ${B}/granary2_chat_rnnt_lr1e4_wu5k/averaged/top5-averaged.nemo \
    --manifest \$LS --trims 0,1,2,3 --flush 0 --normalize
echo
echo '################ librispeech test.clean -- plain RNN-T, no flush'
python scripts/chat_val_probe.py --nemo ${B}/granary2_chat_rnnt_lr1e4_wu5k/averaged/top5-averaged.nemo \
    --manifest \$LS --trims 0 --flush 0 --normalize
echo
echo '################ librispeech test.clean -- plain RNN-T, WITH a flush chunk'
python scripts/chat_val_probe.py --nemo ${B}/granary2_chat_rnnt_lr1e4_wu5k/averaged/top5-averaged.nemo \
    --manifest \$LS --trims 0 --flush 1 --normalize
echo
echo '################ librispeech test.clean -- plain RNN-T, two flush chunks'
python scripts/chat_val_probe.py --nemo ${B}/granary2_chat_rnnt_lr1e4_wu5k/averaged/top5-averaged.nemo \
    --manifest \$LS --trims 0 --flush 2 --normalize
echo
echo '################ librispeech test.clean -- flexible delay'
python scripts/chat_val_probe.py --nemo ${B}/granary2_chat_rnnt_flexdelay4_lr1e4/averaged/top5-averaged.nemo \
    --manifest \$LS --trims 0,1 --normalize
echo
echo '################ plain RNN-T (history_chunks=0, never trims)'
python scripts/chat_val_probe.py \
    --nemo ${B}/granary2_chat_rnnt_lr1e4_wu5k/averaged/top5-averaged.nemo \
    --manifest ${VAL} --trims 0
echo
echo '################ flexible-delay RNN-T (history_chunks=1, trained d~U{0..4})'
python scripts/chat_val_probe.py \
    --nemo ${B}/granary2_chat_rnnt_flexdelay4_lr1e4/averaged/top5-averaged.nemo \
    --manifest ${VAL} --trims 0,1,2,4
"
