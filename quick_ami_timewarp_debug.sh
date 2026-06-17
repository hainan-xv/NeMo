#!/bin/bash
set -e
set -o pipefail

# Quick likelihood time-warp debug run on the first N AMI test rows.
#
# This is a thin wrapper around eval_likelihood_timewarp_ord.sh, so it supports
# the same checkpoint resolution modes:
#   ./quick_ami_timewarp_debug.sh <EXP_NAME> [STEP] [DEVICE_ID]
#   MODEL=/abs/path/model.nemo ./quick_ami_timewarp_debug.sh
#   ./quick_ami_timewarp_debug.sh /abs/path/model.nemo
#
# Useful overrides:
#   FACTORS="0.95,1.0,1.05"       # default: 0.97,1.0,1.03
#   METHOD=time_stretch|speed     # default: time_stretch
#   SCORE_NORM=none|token|word|char
#   N=64                          # number of HF rows to take before sorting
#   DEVICE=0
#   BATCH_SIZE=32

NEMO_ROOT="$(cd "$(dirname "$0")" && pwd)"

export FACTORS="${FACTORS:-0.97,1.0,1.03}"
export METHOD="${METHOD:-time_stretch}"
export SCORE_NORM="${SCORE_NORM:-token}"
export MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-${N:-64}}"
export ONLY="ami_test"
export OUT_DIR="${OUT_DIR:-${NEMO_ROOT}/likelihood_debug_results_ami}"

echo "==> Quick AMI time-warp debug"
echo "==> ONLY=${ONLY}  MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES}"
echo "==> FACTORS=${FACTORS}  METHOD=${METHOD}  SCORE_NORM=${SCORE_NORM}"

"${NEMO_ROOT}/eval_likelihood_timewarp_ord.sh" "$@"
