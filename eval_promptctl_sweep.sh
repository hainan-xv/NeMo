#!/bin/bash
# ============================================================================
# Sweep a PROMPT-CONTROLLED SCRIPT model over its four control dimensions.
#
#   chunk   emission granularity, in encoder frames
#   delay   emission delay, in encoder frames
#   cap     capitalization on/off
#   punct   punctuation on/off
#
# You choose which dimensions to sweep. Named dimensions expand to the values the
# model was actually TRAINED on -- read out of its own exp_config.yaml, not
# hardcoded here -- and every other dimension is pinned to a baseline so the
# sweep stays one-factor-at-a-time and the numbers remain interpretable.
#
# One Slurm job per grid point, all independent, all in parallel.
#
# USAGE
#   ./eval_promptctl_sweep.sh <exp> --sweep delay
#   ./eval_promptctl_sweep.sh <exp> --sweep chunk,delay
#   ./eval_promptctl_sweep.sh <exp> --sweep all              # the full cross product
#   ./eval_promptctl_sweep.sh <exp> --delays "0 3 6" --chunks 14
#   ./eval_promptctl_sweep.sh <exp> --sweep delay --chunks 2 # sweep delay AT chunk 2
#   ./eval_promptctl_sweep.sh <exp> --sweep all --dry-run
#
#   An explicit --chunks/--delays/--cap/--punct always wins over --sweep, so
#   "--sweep delay --chunks 2" means "sweep delay, pinned at chunk 2".
#
# PINNED BASELINES  chunk 14, delay 3, cap on, punct on -- the model's val_*
#   operating point, so a one-dimension sweep passes through the same setting the
#   training run reported val_wer at.
#
# JOB COUNT is the product of the swept dimensions and grows fast: `--sweep all`
#   on the default recipe is 6 x 7 x 2 x 2 = 168 jobs. The script refuses to
#   submit more than MAX_JOBS (default 32) unless you raise it explicitly. It
#   never silently trims the grid.
#
# CAP/PUNCT DO NOT MOVE LEADERBOARD WER. The Open-ASR-Leaderboard normalizer
#   lowercases and strips punctuation before scoring, so those two dimensions
#   will produce near-identical WER by construction. Sweeping them is still
#   useful -- the decoded hypotheses genuinely differ and each run's
#   shards/*.generations.jsonl is kept, so a case- and punctuation-preserving
#   scorer can be run over them afterwards. Just do not read the WER column as
#   evidence that the controls do nothing.
#
# RESULTS are tagged by operating point, e.g.
#   <exp>/leaderboard_eval_chunk14_d3_c1_p0_<ts>_<jobid>/aggregate.log
#   so grid points never overwrite each other and the settings are readable from
#   the path.
# ============================================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source ./oci_env.sh

PROJECT="${PROJECT:-SpeechlmScriptCC}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/lustre/fsw/portfolios/nemotron/users/hainanx}"
RESULTS_ROOT="${OUTPUT_PREFIX}/results/${PROJECT}"

EXP_NAME=""
SWEEP=""
CHUNKS_SET=""; DELAYS_SET=""; CAP_SET=""; PUNCT_SET=""
MAX_JOBS="${MAX_JOBS:-32}"
DRY_RUN=0
SKIP_PRECHECK=0

ENV_ASSIGNMENTS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sweep)          SWEEP="$2"; shift 2 ;;
        --chunks)         CHUNKS_SET="$2"; shift 2 ;;
        --delays)         DELAYS_SET="$2"; shift 2 ;;
        --cap)            CAP_SET="$2"; shift 2 ;;
        --punct)          PUNCT_SET="$2"; shift 2 ;;
        --max-jobs)       MAX_JOBS="$2"; shift 2 ;;
        --skip-precheck)  SKIP_PRECHECK=1; shift ;;
        -n|--dry-run)     DRY_RUN=1; shift ;;
        -h|--help)        sed -n '2,/^# ===*$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        [A-Za-z_]*=*)     ENV_ASSIGNMENTS+=("$1"); shift ;;
        -*)               echo "ERROR: unknown option: $1" >&2; exit 1 ;;
        *)                if [[ -z "$EXP_NAME" ]]; then EXP_NAME="$1"; shift
                          else echo "ERROR: unexpected argument: $1" >&2; exit 1; fi ;;
    esac
done

if [[ -z "$EXP_NAME" ]]; then
    echo "ERROR: no experiment name given." >&2
    echo "usage: ./eval_promptctl_sweep.sh <exp> [--sweep chunk,delay,cap,punct|all] [...]" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Ask the grid what this model is and what it was trained on.
#
# Worth one round trip: a model without prompt_control REJECTS the control
# arguments in generate(), so without this every job in the sweep would start,
# load the checkpoint, and then die identically.
# ---------------------------------------------------------------------------
INFO=$(oci_ssh bash -s -- "$RESULTS_ROOT" "$EXP_NAME" <<'REMOTE'
set -u
ROOT="$1"; EXP="$2"
CFG="$ROOT/$EXP/$EXP/exp_config.yaml"
CKDIR="$ROOT/$EXP/$EXP/checkpoints"

[ -d "$ROOT/$EXP" ] || { echo "MISSING"; exit 0; }
if [ ! -f "$CFG" ]; then echo "NOCONFIG"; exit 0; fi

# First list under `key:`; handles YAML block form and inline [a, b].
extract() {
  awk -v key="$2" '
    done { next }
    {
      if (match($0, "^[[:space:]]*" key ":[[:space:]]*$")) { inblk=1; next }
      if (match($0, "^[[:space:]]*" key ":[[:space:]]*\\[")) {
        line=$0; sub(/^[^[]*\[/, "", line); sub(/\].*$/, "", line);
        gsub(/,/, " ", line); gsub(/[[:space:]]+/, " ", line);
        print line; done=1; next
      }
      if (inblk) {
        if (match($0, "^[[:space:]]*-[[:space:]]*")) { sub(/^[[:space:]]*-[[:space:]]*/, ""); out = out $0 " "; next }
        else { if (out != "") { print out; done=1 }; inblk=0 }
      }
    }
    END { if (!done && out != "") print out }' "$1"
}

pc=false
grep -qE '^[[:space:]]*prompt_control:[[:space:]]*true' "$CFG" && pc=true
n=$(ls -t "$CKDIR"/*.ckpt 2>/dev/null | grep -v -- '-last\.ckpt$' | grep -v -- '-averaged\.ckpt$' | wc -l)

echo "OK"
echo "prompt_control=${pc}"
echo "chunks=$(extract "$CFG" chunk_size)"
echo "delays=$(extract "$CFG" delay_candidates)"
echo "ckpts=${n}"
REMOTE
)

case "$INFO" in
    MISSING*)  echo "ERROR: no such experiment: ${RESULTS_ROOT}/${EXP_NAME}" >&2; exit 1 ;;
    NOCONFIG*) echo "ERROR: ${EXP_NAME} has no exp_config.yaml; cannot tell what it was trained on." >&2
               echo "       Pass --skip-precheck plus explicit --chunks/--delays to sweep anyway." >&2
               [[ "$SKIP_PRECHECK" -eq 1 ]] || exit 1 ;;
esac

IS_PROMPTCTL=$(sed -n 's/^prompt_control=//p' <<< "$INFO")
TRAINED_CHUNKS=$(sed -n 's/^chunks=//p' <<< "$INFO" | xargs || true)
TRAINED_DELAYS=$(sed -n 's/^delays=//p' <<< "$INFO" | xargs || true)
N_CKPTS=$(sed -n 's/^ckpts=//p' <<< "$INFO" | xargs || echo 0)

if [[ "$IS_PROMPTCTL" != "true" && "$SKIP_PRECHECK" -eq 0 ]]; then
    echo "ERROR: ${EXP_NAME} was NOT trained with prompt_control." >&2
    echo "       Its generate() rejects delay/cap/punct arguments rather than ignoring them," >&2
    echo "       so every job in this sweep would fail. Use launch/eval_script.sh for a" >&2
    echo "       plain eval, or sweep a prompt-controlled model instead." >&2
    exit 1
fi
if [[ "${N_CKPTS:-0}" == "0" ]]; then
    echo "ERROR: ${EXP_NAME} has no averageable checkpoints yet." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build the grid: swept dimensions expand, the rest pin to the baseline.
# ---------------------------------------------------------------------------
PIN_CHUNK=14; PIN_DELAY=3; PIN_CAP=1; PIN_PUNCT=1

CHUNKS="$PIN_CHUNK"; DELAYS="$PIN_DELAY"; CAPS="$PIN_CAP"; PUNCTS="$PIN_PUNCT"
if [[ -n "$SWEEP" ]]; then
    IFS=',' read -ra dims <<< "$SWEEP"
    for d in "${dims[@]}"; do
        case "${d// /}" in
            chunk|chunks) CHUNKS="${TRAINED_CHUNKS:-2 4 7 10 14 28}" ;;
            delay|delays) DELAYS="${TRAINED_DELAYS:-0 1 2 3 4 6 8}" ;;
            cap)          CAPS="1 0" ;;
            punct)        PUNCTS="1 0" ;;
            all)          CHUNKS="${TRAINED_CHUNKS:-2 4 7 10 14 28}"
                          DELAYS="${TRAINED_DELAYS:-0 1 2 3 4 6 8}"
                          CAPS="1 0"; PUNCTS="1 0" ;;
            *) echo "ERROR: unknown sweep dimension '${d}' (want chunk, delay, cap, punct or all)" >&2; exit 1 ;;
        esac
    done
fi
# Explicit lists override --sweep.
[[ -n "$CHUNKS_SET" ]] && CHUNKS="$CHUNKS_SET"
[[ -n "$DELAYS_SET" ]] && DELAYS="$DELAYS_SET"
[[ -n "$CAP_SET"    ]] && CAPS="$CAP_SET"
[[ -n "$PUNCT_SET"  ]] && PUNCTS="$PUNCT_SET"

# Warn about out-of-distribution requests rather than silently decoding garbage.
warn_untrained() {  # name, requested, trained
    [[ -z "$3" ]] && return
    local bad=""
    for v in $2; do grep -qw -- "$v" <<< "$3" || bad+=" $v"; done
    [[ -n "$bad" ]] && echo "    !! ${1}:${bad} not in the trained set (${3}) -- out of distribution"
    return 0
}

n_jobs=$(( $(wc -w <<< "$CHUNKS") * $(wc -w <<< "$DELAYS") * $(wc -w <<< "$CAPS") * $(wc -w <<< "$PUNCTS") ))

echo "==> prompt-control sweep: ${EXP_NAME}"
echo "    checkpoints:  ${N_CKPTS} (averaged fresh for the first job, cached after)"
echo "    chunk:        ${CHUNKS}"
echo "    delay:        ${DELAYS}"
echo "    cap:          ${CAPS}"
echo "    punct:        ${PUNCTS}"
warn_untrained chunk "$CHUNKS" "$TRAINED_CHUNKS"
warn_untrained delay "$DELAYS" "$TRAINED_DELAYS"
echo "    grid:         ${n_jobs} job(s)"
if [[ $(wc -w <<< "$CAPS") -gt 1 || $(wc -w <<< "$PUNCTS") -gt 1 ]]; then
    echo ""
    echo "    NOTE: the leaderboard normalizer lowercases and strips punctuation before"
    echo "          scoring, so cap/punct will NOT move the reported WER. The decoded"
    echo "          hypotheses do differ -- re-score shards/*.generations.jsonl with a"
    echo "          case/punct-preserving scorer to measure these two dimensions."
fi
echo ""

if [[ "$n_jobs" -gt "$MAX_JOBS" ]]; then
    echo "REFUSING: ${n_jobs} jobs exceeds MAX_JOBS=${MAX_JOBS}." >&2
    echo "          Narrow the sweep, or raise it deliberately: --max-jobs ${n_jobs}" >&2
    echo "          (nothing was trimmed -- the whole grid is either submitted or not)" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------
DRY_FLAG=""; [[ "$DRY_RUN" -eq 1 ]] && DRY_FLAG="--dry-run"

submitted=()
first=1
for c in $CHUNKS; do
  for d in $DELAYS; do
    for cap in $CAPS; do
      for pun in $PUNCTS; do
        # Averaging: every grid point shares ONE averaged checkpoint. Only the
        # first job rebuilds it (FORCE_AVERAGE=1); the rest are made to WAIT on
        # that job via a Slurm dependency. Without the dependency they would race
        # -- reading the averaged file while the first job is still writing it.
        sync_flag="--no-sync"
        avg_flag="FORCE_AVERAGE=0"
        dep_opt=""
        if [[ "$first" -eq 1 ]]; then
            sync_flag=""                 # verify the grid checkout once
            avg_flag="FORCE_AVERAGE=1"
        elif [[ -n "${AVG_JOB_ID:-}" ]]; then
            dep_opt="SBATCH_OPTS=--dependency=afterok:${AVG_JOB_ID}"
        fi

        tag="d${d}_c${cap}_p${pun}"
        label="chunk=${c} delay=${d} cap=${cap} punct=${pun}"
        echo "--- ${label} ---"
        if ! out="$(./oci_launch.sh $sync_flag $DRY_FLAG \
                        "$avg_flag" "EVAL_TAG=${tag}" \
                        ${dep_opt:+"$dep_opt"} \
                        "NUM_DELAY_FRAMES=${d}" "CAPITALIZATION=${cap}" "PUNCTUATION=${pun}" \
                        ${ENV_ASSIGNMENTS[@]+"${ENV_ASSIGNMENTS[@]}"} \
                        launch/eval_script.sh "$EXP_NAME" "$c" 2>&1)"; then
            echo "$out"
            echo "    !! FAILED to submit" >&2
            submitted+=("${label}: FAILED")
            continue
        fi
        echo "$out" | grep -E '^(==>|    )' || true
        jid="$(sed -n 's/^==> Job \([0-9]*\) submitted.*/\1/p' <<< "$out" | tail -1)"
        if [[ "$first" -eq 1 ]]; then
            AVG_JOB_ID="$jid"            # everything else waits on this one
            first=0
            [[ -n "$AVG_JOB_ID" ]] && echo "    (rebuilds the averaged checkpoint; later jobs wait on ${AVG_JOB_ID})"
        fi
        submitted+=("${label}: ${jid:-dry-run}")
        echo ""
      done
    done
  done
done

echo "===================== submitted ====================="
printf '  %s\n' "${submitted[@]}"
echo ""
if [[ "$DRY_RUN" -eq 0 ]]; then
    echo "Watch:   ssh ${OCI_USER}@${OCI_HOST} squeue -u \$USER"
    echo "Results: ${RESULTS_ROOT}/${EXP_NAME}/eval_<ckpt_ts>/chunk<C>_d<D>_c<0|1>_p<0|1>/aggregate.log"
    echo "Compare: python scripts/analyze_eval_errors.py <label>=<dir> ..."
fi
