# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Print the official macro-7 table from a results directory.

ONE implementation, used by BOTH the per-arm job (which scores as soon as it has
merged its own manifests) and the standalone whole-table launcher. They used to
be separate copies of this logic, which is how the in-job version kept an
unscoped glob long after the standalone one was fixed.

SCORING RUNS AGAINST A SNAPSHOT, NEVER THE LIVE DIRECTORY. eval_utils.score_results
globs whatever it is given, and arms decode concurrently into a shared results
dir, so a live glob sweeps up other arms' ``-shard<i>of<N>`` files -- which are
partial by definition and get DELETED under the scorer the moment their arm
merges. That produced both bogus one-eighth-sized rows and outright
FileNotFoundError mid-scan. Copying only the merged manifests into a private
directory makes this callable at ANY time, mid-run included.

MACRO-7 EXCLUDES plain earnings22: it is superseded by earnings22_cleaned_aa_chunked,
and counting both would weight one corpus twice. Only arms holding all seven get
a row; a partial arm is listed separately rather than averaged over a subset,
because a partial average looks exactly like a real one.
"""

import argparse
import glob
import os
import re
import shutil
import sys
import tempfile
from collections import defaultdict

SHARD_RE = re.compile(r"-shard\d+of\d+")
EXCLUDE = "hf-audio-open-asr-leaderboard_earnings22_test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument(
        "--highlight", default="", help="Mark rows whose arm key matches this, e.g. the current job's arm."
    )
    ap.add_argument("--oasr", default="/oasr", help="open_asr_leaderboard checkout providing the normalizer.")
    args = ap.parse_args()

    sys.path.insert(0, args.oasr)
    from normalizer import eval_utils

    snap = tempfile.mkdtemp(prefix="score_snapshot_")
    kept = skipped = 0
    try:
        for f in sorted(glob.glob(os.path.join(args.results_dir, "*.jsonl"))):
            if SHARD_RE.search(os.path.basename(f)):
                skipped += 1
                continue
            try:
                shutil.copy2(f, os.path.join(snap, os.path.basename(f)))
                kept += 1
            except FileNotFoundError:
                skipped += 1  # merged away mid-copy; harmless
        print(f"scoring {kept} merged manifests ({skipped} in-flight shard files skipped)")
        _score, results = eval_utils.score_results(snap)
    finally:
        shutil.rmtree(snap, ignore_errors=True)

    per = defaultdict(dict)
    for k, v in (results or {}).items():
        m, _, ds = k.partition(" | ")
        per[m][ds] = v.get("wer")

    seven = [d for d in sorted({d for m in per for d in per[m]}) if d != EXCLUDE]
    if not seven:
        print("no scored datasets yet")
        return 0

    rows, partial = [], []
    for m, dd in per.items():
        got = [dd[d] for d in seven if d in dd]
        arm = m.split("__")[0].replace("MODEL_", "")
        if len(got) == len(seven):
            rows.append((sum(got) / len(got), got, arm))
        else:
            partial.append((len(got), arm))

    print()
    print("  MACRO7 | " + " ".join("%5s" % d.split("_")[-2][:5] for d in seven) + " | arm")
    for avg, got, arm in sorted(rows):
        mark = " <<<" if args.highlight and arm == args.highlight else ""
        print("ROW  %6.3f | %s | %s%s" % (avg, " ".join("%5.2f" % g for g in got), arm[:30], mark))
    for n, arm in sorted(partial):
        print("--- (%d/%d datasets, no row) %s" % (n, len(seven), arm[:40]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
