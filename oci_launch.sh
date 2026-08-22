#!/bin/bash
# ============================================================================
# Submit a launch script on the OCI grid from your local machine.
#
# Checks that the grid checkout actually matches your local HEAD first, runs
# ./sync_to_oci.sh automatically if it does not, then SSHes in, cds to the grid
# repo, and sbatch-es the script you named.
#
# The sync check is the point of this script: submitting against a stale /code
# checkout is silent — the job runs happily on old code and you only find out
# from the results.
#
# USAGE
#   ./oci_launch.sh [FLAGS] [VAR=VALUE ...] <script> [script args ...]
#
#   ./oci_launch.sh launch/script_baseline.sh
#   ./oci_launch.sh launch/script_baseline.sh 123            # seed 123
#   ./oci_launch.sh DELAY=6 AUDIO_HISTORY_CHUNKS=1 launch/script_baseline.sh
#   ./oci_launch.sh --follow launch/script_baseline.sh
#
# VAR=VALUE pairs before the script name are exported around the remote sbatch,
# so they reach the job (sbatch defaults to --export=ALL). That is how the
# launcher's env knobs (DELAY, AUDIO_HISTORY_CHUNKS, CHUNK_SIZES, MAX_STEPS,
# EXP_NAME, INIT_EXP, ...) are set. Arguments AFTER the script name are passed
# to the script itself (script_baseline.sh takes the Lhotse seed as $1).
#
# FLAGS
#   --follow, -f     tail the job's stdout after submitting (Ctrl-C stops the
#                    tail, NOT the job)
#   --no-sync        skip the sync check and submit against whatever is on the grid
#   --force-sync     always run sync_to_oci.sh, even if HEAD already matches
#   --dry-run, -n    print what would happen; touch nothing
#   -h, --help       this text
#
# ENV
#   SBATCH_OPTS      extra sbatch options, e.g. SBATCH_OPTS="--time=01:00:00"
#   OCI_PARTITION    override the script's #SBATCH -p
#   OCI_NODES        override the script's #SBATCH -N
#   plus everything in oci_env.sh (OCI_REPO, OCI_HOST, SSH_KEY, BRANCH, ...)
# ============================================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source ./oci_env.sh

usage() { sed -n '2,/^# ===*$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

FOLLOW="${FOLLOW:-0}"
DO_SYNC=1
FORCE_SYNC=0
DRY_RUN=0

# --- flags ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--follow)   FOLLOW=1; shift ;;
        --no-sync)     DO_SYNC=0; shift ;;
        --force-sync)  FORCE_SYNC=1; shift ;;
        -n|--dry-run)  DRY_RUN=1; shift ;;
        -h|--help)     usage 0 ;;
        --)            shift; break ;;
        -*)            echo "ERROR: unknown flag: $1" >&2; usage 1 ;;
        *)             break ;;
    esac
done

# --- leading VAR=VALUE assignments (the familiar `FOO=bar cmd` idiom) ---
ENV_ASSIGNMENTS=()
while [[ $# -gt 0 && "$1" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; do
    ENV_ASSIGNMENTS+=("$1")
    shift
done

if [[ $# -eq 0 ]]; then
    echo "ERROR: no script given." >&2
    usage 1
fi
SCRIPT="$1"; shift
SCRIPT_ARGS=("$@")

# Also forward knobs that were exported in the CALLING shell rather than passed as
# arguments. `INIT_CKPT=none ./oci_launch.sh launch/x.sh` is the idiom everyone
# reaches for first, and without this it silently does nothing: the assignment
# lands in THIS process's environment and never reaches the remote sbatch.
#
# The knob names are discovered from the target script itself (every `${NAME:-...}`
# it reads), so this stays correct as launchers gain options. System and Slurm
# variables are excluded -- forwarding those would be actively harmful.
forward_inherited_env() {
    local script="$1" name
    local -a found=()
    while read -r name; do
        case "$name" in
            PATH|HOME|USER|SHELL|PWD|TERM|LANG|LC_*|SHLVL|_|HOSTNAME|SSH_*|SLURM_*|BASH_*)
                continue ;;
            # NEVER forward credentials. The launch scripts read these from
            # chmod-600 files on the grid; putting one on a command line would
            # expose it in the job's argv, in squeue output and in the logs.
            *TOKEN*|*SECRET*|*PASSWORD*|*PASSWD*|*API_KEY*|*_KEY|KEY_*)
                continue ;;
        esac
        # Only forward if actually set here AND not already given as an argument.
        [[ -z "${!name+x}" ]] && continue
        for a in ${ENV_ASSIGNMENTS[@]+"${ENV_ASSIGNMENTS[@]}"}; do
            [[ "$a" == "${name}="* ]] && continue 2
        done
        found+=("${name}=${!name}")
    done < <(grep -oE '\$\{[A-Za-z_][A-Za-z0-9_]*(:-|\})' "$script" \
             | sed -E 's/^\$\{//; s/(:-|\})$//' | sort -u)
    if [[ ${#found[@]} -gt 0 ]]; then
        # Print names only -- a forwarded value could still be sensitive.
        echo "==> Forwarding inherited env: ${found[*]%%=*}"
        ENV_ASSIGNMENTS+=("${found[@]}")
    fi
}

# Catch typos locally rather than after an SSH round trip. The grid runs its own
# checkout, but the file must exist here too or it was never synced.
if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: '$SCRIPT' does not exist in $(pwd)." >&2
    echo "       Available launch scripts:" >&2
    ls launch/*.sh 2>/dev/null | sed 's/^/         /' >&2 || true
    exit 1
fi

forward_inherited_env "$SCRIPT"

# ---------------------------------------------------------------------------
# 1) Is the grid checkout already up to date?
# ---------------------------------------------------------------------------
local_head="$(git rev-parse HEAD)"

# Uncommitted work that sync_to_oci.sh would pick up: any tracked modification,
# or an untracked file under a directory the allowlist covers. (Scratch files
# elsewhere in the tree are ignored, so they do not force a pointless sync.)
dirty_reason=""
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    dirty_reason="uncommitted changes to tracked files"
else
    untracked_relevant="$(git ls-files --others --exclude-standard \
        -- launch nemo/collections/speechlm2 examples/speechlm2 tests/collections/speechlm2 \
           sync_to_oci.sh oci_env.sh oci_launch.sh oci_launch_interactive.sh 2>/dev/null || true)"
    if [[ -n "$untracked_relevant" ]]; then
        dirty_reason="new untracked files: $(echo $untracked_relevant | cut -c1-120)"
    fi
fi

need_sync=0
if [[ "$DO_SYNC" -eq 0 ]]; then
    echo "==> --no-sync: submitting against whatever is on the grid (NOT verified)"
elif [[ "$FORCE_SYNC" -eq 1 ]]; then
    echo "==> --force-sync requested"
    need_sync=1
elif [[ -n "$dirty_reason" ]]; then
    echo "==> Local tree has $dirty_reason -> sync needed"
    need_sync=1
else
    echo "==> Checking grid checkout at ${OCI_REPO} ..."
    remote_head="$(oci_ssh "git -C '${OCI_REPO}' rev-parse HEAD 2>/dev/null || echo MISSING" | tr -d '\r\n')"
    if [[ "$remote_head" == "MISSING" ]]; then
        echo "    grid checkout not present -> sync needed"
        need_sync=1
    elif [[ "$remote_head" != "$local_head" ]]; then
        echo "    grid is at ${remote_head:0:9}, local is at ${local_head:0:9} -> sync needed"
        need_sync=1
    else
        echo "    grid is up to date at ${local_head:0:9}"
    fi
fi

if [[ "$need_sync" -eq 1 ]]; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "    [dry-run] would run: ./sync_to_oci.sh"
    else
        echo "==> Running ./sync_to_oci.sh"
        ./sync_to_oci.sh
        echo ""
    fi
fi

# ---------------------------------------------------------------------------
# 2) Build and run the remote sbatch
# ---------------------------------------------------------------------------
SBATCH_ARGS=()
[[ -n "${OCI_PARTITION:-}" ]] && SBATCH_ARGS+=("--partition=${OCI_PARTITION}")
[[ -n "${OCI_NODES:-}" ]] && SBATCH_ARGS+=("--nodes=${OCI_NODES}")
# shellcheck disable=SC2206  # intentional word-splitting: SBATCH_OPTS is a flag list
[[ -n "${SBATCH_OPTS:-}" ]] && SBATCH_ARGS+=(${SBATCH_OPTS})

# printf %q keeps quoting intact through ssh (prompts and lists contain spaces,
# brackets and commas that the remote shell would otherwise re-split).
remote_cmd="cd $(printf '%q' "$OCI_REPO") && mkdir -p slurm_out && "
for a in ${ENV_ASSIGNMENTS[@]+"${ENV_ASSIGNMENTS[@]}"}; do
    remote_cmd+="$(printf '%q' "$a") "
done
remote_cmd+="sbatch"
for o in ${SBATCH_ARGS[@]+"${SBATCH_ARGS[@]}"}; do
    remote_cmd+=" $(printf '%q' "$o")"
done
remote_cmd+=" $(printf '%q' "$SCRIPT")"
for a in ${SCRIPT_ARGS[@]+"${SCRIPT_ARGS[@]}"}; do
    remote_cmd+=" $(printf '%q' "$a")"
done

echo "==> Submitting on ${OCI_USER}@${OCI_HOST}"
echo "    $remote_cmd"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "    [dry-run] not submitting"
    exit 0
fi

submit_out="$(oci_ssh "$remote_cmd")"
echo "$submit_out"

job_id="$(echo "$submit_out" | grep -oE '[0-9]+' | tail -1)"
if [[ -z "$job_id" ]]; then
    echo "WARNING: could not parse a job id from sbatch output." >&2
    exit 0
fi

echo ""
echo "==> Job ${job_id} submitted."
echo "    status: ssh ${OCI_USER}@${OCI_HOST} squeue -j ${job_id}"
echo "    cancel: ssh ${OCI_USER}@${OCI_HOST} scancel ${job_id}"
echo "    follow: $0 --follow ...   (or re-run with -f)"

if [[ "$FOLLOW" -ne 1 ]]; then
    exit 0
fi

# --- follow: ask Slurm where stdout went, wait for it, then tail ---
echo ""
echo "==> Waiting for job ${job_id} to start (Ctrl-C stops the tail, not the job)"
oci_ssh bash -s -- "$job_id" <<'REMOTE'
set -uo pipefail
job_id="$1"

# StdOut carries %x/%j already expanded by Slurm, so no guessing at the path.
for _ in $(seq 1 120); do
    out="$(scontrol show job "$job_id" 2>/dev/null | tr ' ' '\n' | sed -n 's/^StdOut=//p' | head -1)"
    state="$(scontrol show job "$job_id" 2>/dev/null | tr ' ' '\n' | sed -n 's/^JobState=//p' | head -1)"
    [[ -z "$state" ]] && { echo "job $job_id is no longer in the queue"; exit 0; }
    if [[ -n "$out" && -f "$out" ]]; then
        echo "==> tailing $out  (state=$state)"
        exec tail -n +1 -F "$out"
    fi
    sleep 5
done
echo "timed out waiting for the job to produce output; check: squeue -j $job_id"
REMOTE
