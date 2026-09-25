#!/bin/bash
#SBATCH -A nemotron_speech_asr
#SBATCH -J nemotron_speechprod_asr:oci-off-prereq
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 00:25:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --overcommit
#SBATCH --output=slurm_out/%x=%j --error=slurm_out/%x=%j

# ============================================================================
# PREREQUISITES for the official Open-ASR-Leaderboard harness on OCI.
#
#   sbatch launch/oci_eval_official_prereq.sh      <- rebuild ${MY}/pylibs + verify
#   VERIFY_ONLY=1 sbatch launch/oci_eval_official_prereq.sh
#
# RUN THIS BEFORE THE SEVEN oci_eval_official_*.sh JOBS. Every gap it closes was
# found the expensive way -- as an import error that would have killed seven
# four-hour jobs after they had already burned the GPUs.
#
# THREE THINGS THE CONTAINER GETS WRONG FOR THIS HARNESS:
#
#  1. num2words is absent outright. normalizer/data_utils.py imports it at module
#     scope, so the scorer cannot even be imported without it.
#
#  2. kaldialign is present but TOO OLD -- it has no batch_error_rate, which IS
#     the leaderboard metric. The container copy imports fine and then fails on
#     the attribute, so this looks like a code bug rather than a version skew.
#     0.12.0 is built for cp312-x86_64 and the container is Python 3.12.3, so the
#     wheel loads; the check below verifies it COMPUTES correctly too, because an
#     ABI-mismatched extension can import cleanly and still return garbage.
#
#  3. datasets is 4.5.0, which decodes audio ONLY through torchcodec -- absent
#     here. We pin 3.6.0 instead of installing torchcodec, deliberately: 3.6.0
#     decodes via soundfile, is what runs this harness locally, and is what
#     open_asr_leaderboard itself pins. Adding torchcodec would have swapped BOTH
#     the datasets major version AND the audio decoder underneath a comparison
#     whose entire purpose is to remove confounds from the DFW/OCI gap.
#
# PYTHONPATH precedence is what makes the shadowing work, so pylibs must come
# BEFORE site-packages in every consumer -- oci_eval_official_backend.sh does.
# --no-deps keeps the pinned packages from dragging a dependency tree over the
# container's working torch/pyarrow stack.
# ============================================================================
set -uo pipefail
mkdir -p slurm_out
LUSTRE=/lustre/fsw/portfolios/nemotron
MY=${LUSTRE}/users/hainanx
HEH=/lustre/fsw/portfolios/llmservice/users/heh
CONTAINER="${CONTAINER:-${HEH}/containers/nemo-26.02-streaming-speechlm.sqsh}"
CODE_DIR="${CODE_DIR:-${MY}/NeMo_SCRIPT_cc}"
OASR="${MY}/open_asr_leaderboard"
HF_TOKEN="$(tr -d '\r\n' < "$HOME/.hf_token")"

# kaldialign is a COMPILED extension and cannot be pip-installed blind here: the
# container's own copy would satisfy the requirement. It is staged from the
# workstation (same cp312-x86_64 ABI) by the sync step, not rebuilt.
[[ -d "${MY}/pylibs/kaldialign" ]] || { echo "ERROR: ${MY}/pylibs/kaldialign missing -- scp it from the workstation's site-packages (0.12.0, cp312)"; exit 1; }

if [[ "${VERIFY_ONLY:-0}" != "1" ]]; then
  srun --overlap -n1 -N1 --container-image="$CONTAINER" \
    --container-mounts="${LUSTRE}:${LUSTRE},${HEH}:${HEH}" \
    bash -c "pip install --no-deps --target=${MY}/pylibs 'datasets==3.6.0' num2words docopt 2>&1 | tail -2"

  # WARM THE DATASET CACHE, ONLINE, ONCE.
  #
  # Evals decode with HF_HUB_OFFLINE=1 and make NO HuggingFace calls at all -- no
  # rate limits, no dependency on the Hub being reachable mid-run. That only works
  # if the cache is complete, and this is the one place that fills it. The cache
  # lives on lustre and persists, so it is paid once per new dataset rather than
  # once per job; it used to be an in-job warm-up costing every run ~5-6 minutes
  # before a single GPU started.
  echo "==> warming the dataset cache (online, one-time)"
  srun --overlap -n1 -N1 --container-image="$CONTAINER" \
    --container-mounts="${LUSTRE}:${LUSTRE},${CODE_DIR}:/code,${OASR}:/oasr" \
    bash -c "export PYTHONPATH=/code:/code/scripts:/oasr:${MY}/pylibs:\${PYTHONPATH:-} HF_HOME=${MY}/hf_cache HF_TOKEN=${HF_TOKEN} && python - <<'WARMEOF'
import sys
sys.path.insert(0, '/oasr')
from normalizer import data_utils
for path, name, split in [
    ('hf-audio/open-asr-leaderboard', 'ami_cleaned', 'test'),
    ('hf-audio/open-asr-leaderboard', 'gigaspeech_cleaned', 'test'),
    ('hf-audio/open-asr-leaderboard', 'voxpopuli_cleaned_aa', 'test'),
    ('hf-audio/open-asr-leaderboard', 'earnings22', 'test'),
    ('hf-audio/open-asr-leaderboard', 'librispeech', 'test.clean'),
    ('hf-audio/open-asr-leaderboard', 'librispeech', 'test.other'),
    ('hf-audio/open-asr-leaderboard', 'spgispeech', 'test'),
    ('ArtificialAnalysis/Earnings22-Cleaned-AA-chunked', 'earnings22_cleaned_aa_chunked', 'test'),
]:
    class A:
        dataset_path = path
        dataset = name
        split = split
        max_eval_samples = 1
        streaming = False
    data_utils.load_data(A())
    print(f'  cached: {name} {split}', flush=True)
WARMEOF
"
fi

srun --overlap -n1 -N1 --container-image="$CONTAINER" \
  --container-mounts="${LUSTRE}:${LUSTRE},${HEH}:${HEH},${CODE_DIR}:/code,${OASR}:/oasr" \
  bash -c "export PYTHONPATH=/code:/code/scripts:/oasr:${MY}/pylibs:\${PYTHONPATH:-} HF_HOME=${MY}/hf_cache HF_TOKEN=${HF_TOKEN} && cd /oasr/nemo_asr && python - <<'PYEOF'
import sys, traceback
ok = True
def chk(label, fn):
    global ok
    try:
        d = fn()
        print(f'  [PASS] {label}  {d}', flush=True)
    except Exception as e:
        ok = False
        print(f'  [FAIL] {label}  {type(e).__name__}: {e}', flush=True)
        traceback.print_exc()

print('== python', sys.version.split()[0])
chk('import kaldialign', lambda: __import__('kaldialign').__file__)

def ds_pin():
    import datasets
    v, f = datasets.__version__, datasets.__file__
    assert v.startswith('3.6.'), f'expected the pinned 3.6.0, got {v} ({f})'
    assert '/pylibs/' in f, f'pylibs copy is NOT shadowing the container 4.5.0: {f}'
    return f'{v} from pylibs (container ships 4.5.0, which decodes only via torchcodec)'
chk('datasets pinned to 3.6.0, shadowing container', ds_pin)

def kald_shadow():
    import kaldialign
    f = kaldialign.__file__
    assert '/pylibs/' in f, f'pylibs copy is NOT shadowing the container one: {f}'
    return f
chk('kaldialign resolves to pylibs, not container', kald_shadow)

def kald_math():
    # An ABI-loaded extension can import cleanly and still compute garbage, so
    # check a hand-verified number: 1 substitution over a 3-token reference.
    from kaldialign import batch_error_rate
    r = batch_error_rate([['a','b','c']], [['a','x','c']])
    assert abs(r['err_rate'] - 1/3) < 1e-9, r
    # merge_compounds=True is what the leaderboard actually uses: a split
    # compound must score ZERO errors, not two.
    m = batch_error_rate([['whitepaper']], [['white','paper']], merge_compounds=True)
    assert m['err_rate'] == 0.0, m
    return f\"err_rate={r['err_rate']:.4f} (exp 0.3333); merge_compounds split-compound={m['err_rate']}\"
chk('batch_error_rate computes correctly', kald_math)
chk('import num2words (pylibs)',     lambda: __import__('num2words').__file__)
chk('import normalizer.eval_utils',  lambda: __import__('normalizer.eval_utils', fromlist=['x']).__file__)
chk('score_results is callable',     lambda: __import__('normalizer.eval_utils', fromlist=['x']).score_results.__name__)
chk('import script_asr_shim',        lambda: __import__('script_asr_shim').load_script_shim.__name__)

def chunked():
    from normalizer import data_utils
    p='ArtificialAnalysis/Earnings22-Cleaned-AA-chunked'
    assert data_utils.is_chunked_dataset(p), 'not recognised as chunked'
    class A: dataset_path=p; dataset='earnings22_cleaned_aa_chunked'; split='test'; max_eval_samples=4; streaming=False
    ds = data_utils.load_chunked_data(A())
    r = next(iter(ds))
    assert r.get('text','').strip(), 'parent transcript did NOT attach'
    return f\"cols={sorted(r.keys())[:6]} text[:50]={r['text'][:50]!r}\"
chk('chunked earnings22 + parent-join over HF', chunked)

import argparse, importlib.util
def cs_flag():
    src=open('run_eval.py').read()
    assert '--chunk_size' in src and 'chunk_size=args.chunk_size' in src
    assert \"endswith('.ckpt') and args.chunk_size != 14\" in src.replace('\"',\"'\")
    return 'flag + shim passthrough + manifest suffix all present'
chk('run_eval.py --chunk_size patch', cs_flag)

print()
print('SMOKE RESULT:', 'ALL PASS' if ok else 'FAILURES ABOVE')
sys.exit(0 if ok else 1)
PYEOF"
