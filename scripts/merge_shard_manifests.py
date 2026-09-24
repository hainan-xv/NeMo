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
"""Merge per-shard leaderboard manifests back into one manifest per dataset.

run_eval.py --num_shards N writes N files whose model id carries a
``-shard<i>of<N>`` suffix. normalizer/eval_utils.score_results derives the model
id FROM the filename, so left alone those become N separate one-Nth-sized rows
rather than one complete row.

WHY THIS VERIFIES INSTEAD OF JUST CONCATENATING. A sharding bug does not crash;
it silently drops or duplicates utterances and still yields a perfectly
plausible WER -- which is the single most dangerous failure mode in this whole
pipeline, because the number looks fine and is wrong. So a merge only happens
when all N shards are present, and the merged row count is reported; with
--expect it is checked and a mismatch is fatal.

Duplicate audio ids across shards are a hard error: run_eval.py assigns shard
membership by ``idx % N`` over one dataset, so an id in two shards means the
shards did not come from the same dataset or the same run.

    python merge_shard_manifests.py /oasr/nemo_asr/results
    python merge_shard_manifests.py /oasr/nemo_asr/results --expect spgispeech=39341
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

SHARD_RE = re.compile(r"-shard(\d+)of(\d+)")


def read_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="SUBSTR=N",
        help="Assert the merged manifest whose name contains SUBSTR has exactly N rows. Repeatable.",
    )
    ap.add_argument(
        "--tag",
        default="",
        help="Only merge manifests whose name starts with MODEL_<tag>__. REQUIRED when arms run "
        "concurrently: the results dir is shared, so an unscoped merge inspects OTHER arms' "
        "in-flight shards. If it ever caught all N present but one still flushing, it would "
        "merge truncated data AND delete the shards, leaving the owning arm nothing to merge.",
    )
    ap.add_argument("--keep-shards", action="store_true", help="Do not delete shard files after merging.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    groups = defaultdict(dict)  # canonical path -> {shard_index: path}
    totals = {}
    prefix = f"MODEL_{args.tag}__" if args.tag else ""
    for path in sorted(glob.glob(os.path.join(args.results_dir, "*.jsonl"))):
        base = os.path.basename(path)
        if prefix and not base.startswith(prefix):
            continue
        m = SHARD_RE.search(base)
        if not m:
            continue
        idx, n = int(m.group(1)), int(m.group(2))
        canonical = os.path.join(args.results_dir, SHARD_RE.sub("", os.path.basename(path)))
        groups[canonical][idx] = path
        totals[canonical] = n

    if not groups:
        print("no shard manifests found; nothing to merge")
        return 0

    rc = 0
    for canonical, shards in sorted(groups.items()):
        n = totals[canonical]
        name = os.path.basename(canonical)
        missing = [i for i in range(n) if i not in shards]
        if missing:
            # Refuse rather than merge a partial set: a manifest short by one
            # shard scores as ~1/N deletions and looks merely "a bit poor".
            print(f"[SKIP] {name}: {len(shards)}/{n} shards, MISSING {missing}", file=sys.stderr)
            rc = 1
            continue

        rows, seen, dupes = [], set(), 0
        for i in range(n):
            for r in read_jsonl(shards[i]):
                key = r.get("audio_filepath") or r.get("id")
                if key is not None:
                    if key in seen:
                        dupes += 1
                    seen.add(key)
                rows.append(r)
        if dupes:
            print(f"[FAIL] {name}: {dupes} duplicate ids across shards", file=sys.stderr)
            rc = 1
            continue

        if args.dry_run:
            print(f"[dry-run] {name}: would merge {n} shards -> {len(rows)} rows")
            continue

        with open(canonical, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        for p in shards.values():
            if not args.keep_shards:
                os.remove(p)
        print(f"[OK] {name}: {n} shards -> {len(rows)} rows")

    for spec in args.expect:
        substr, _, want = spec.partition("=")
        want = int(want)
        hits = [c for c in groups if substr in os.path.basename(c)]
        if not hits:
            print(f"[FAIL] --expect {spec}: no merged manifest matching {substr!r}", file=sys.stderr)
            rc = 1
            continue
        for c in hits:
            got = len(read_jsonl(c))
            ok = got == want
            print(f"  [{'PASS' if ok else 'FAIL'}] {os.path.basename(c)[:70]}: {got} rows (want {want})")
            if not ok:
                rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
