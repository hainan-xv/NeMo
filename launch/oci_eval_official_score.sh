#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:oci-eval-official-score
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 00:30:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --overcommit
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# THE OFFICIAL SCORE for every arm evaluated by oci_eval_official_*.sh.
#
#   sbatch launch/oci_eval_official_score.sh      <- final table, any time
#   bash   launch/oci_eval_official_score.sh --inline   <- from inside a job
#
# run_eval.py PRINTS evaluate/jiwer WER; the leaderboard PUBLISHES what
# normalizer/eval_utils.score_results() computes -- kaldialign batch_error_rate
# with merge_compounds=True, which counts a split compound ("white paper" vs
# "whitepaper") as zero errors. That is 0.3-0.5 WER more lenient, so the WER
# lines in the per-dataset logs are systematically PESSIMISTIC and are not
# comparable to any published figure. This is the number that is.
#
# MACRO-7 EXCLUDES plain earnings22, exactly as the DFW table does: it is
# superseded by earnings22_cleaned_aa_chunked, and counting both would weight
# the same corpus twice. Only arms with all 7 datasets present get a row, so a
# still-running arm is omitted rather than averaged over a subset -- a partial
# average would silently look like a real one.
# ============================================================================
set -uo pipefail
mkdir -p slurm_out

LUSTRE=/lustre/fsw/portfolios/nemotron
MY=${LUSTRE}/users/hainanx
HEH=/lustre/fsw/portfolios/llmservice/users/heh
CONTAINER="${CONTAINER:-${HEH}/containers/nemo-26.02-streaming-speechlm.sqsh}"
CODE_DIR="${CODE_DIR:-${MY}/NeMo_SCRIPT_cc}"
OASR="${MY}/open_asr_leaderboard"
PYLIBS="${MY}/pylibs"

srun --overlap -n1 -N1 --container-image="$CONTAINER" \
     --container-mounts="${LUSTRE}:${LUSTRE},${HEH}:${HEH},${CODE_DIR}:/code,${OASR}:/oasr" \
     bash -c "export PYTHONPATH=${PYLIBS}:/oasr:\${PYTHONPATH:-} && python - <<'PYEOF'
import sys
sys.path.insert(0, '/oasr')
from collections import defaultdict
from normalizer import eval_utils

score, results = eval_utils.score_results('/oasr/nemo_asr/results')
per = defaultdict(dict)
for k, v in (results or {}).items():
    m, _, ds = k.partition(' | ')
    per[m][ds] = v.get('wer')

EXCLUDE = 'hf-audio-open-asr-leaderboard_earnings22_test'
SEVEN = [d for d in sorted({d for m in per for d in per[m]}) if d != EXCLUDE]
print()
print('datasets in macro-7:')
for d in SEVEN:
    print('   ', d)
print()
rows, partial = [], []
for m, dd in per.items():
    got = [dd[d] for d in SEVEN if d in dd]
    if len(got) == len(SEVEN):
        rows.append((sum(got) / len(got), got, m))
    else:
        partial.append((len(got), m))
hdr = ' '.join('%5s' % d.split('_')[-2][:5] for d in SEVEN)
print('  MACRO7 | %s | model' % hdr)
for avg, got, m in sorted(rows):
    print('ROW %6.3f | %s | %s' % (avg, ' '.join('%5.2f' % g for g in got), m[-60:]))
for n, m in sorted(partial):
    print('--- (%d/%d datasets, no row yet) %s' % (n, len(SEVEN), m[-60:]))
PYEOF
" 2>&1 | grep -vE "^srun:|CSV Summary|^\*{4,}|^model,|^$"
