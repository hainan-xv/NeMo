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
"""Rescore NeMo ``speech_to_text_eval.py`` output manifests with the Open ASR
Leaderboard's OWN scoring (vendored normalizer + kaldialign merge_compounds).

``speech_to_text_eval.py`` decodes with NeMo's canonical, maintained pipeline and
writes one output manifest per dataset containing the ground truth and ``pred_text``.
Its printed WER uses NeMo's plain ``word_error_rate`` (no whisper/leaderboard
normalization), so it does NOT line up with the public board. This tool re-reads
those manifests and recomputes WER with ``leaderboard_wer.WER`` -- the SAME scorer
our grid eval uses -- so the decode comes from NeMo's trusted code path while the
number is board-comparable.

Each manifest is a dataset: the dataset key is taken from the filename, which the
wrapper (launch/eval_parakeet_nemo.sh) writes as ``<name>__<split>.json`` (the
double underscore separates the dataset name -- which itself may contain single
underscores, e.g. ami_cleaned -- from the split).

Usage:
    python scripts/rescore_nemo_eval.py --pred_dir <dir of *__*.json manifests>
"""
import argparse
import glob
import json
import os

# Same-dir imports (run as `python scripts/rescore_nemo_eval.py`).
from leaderboard_wer import WER
from speechlm_leaderboard_eval import _log


def parse_key(path: str) -> str:
    """`.../ami_cleaned__test.json` -> `ami_cleaned/test` (split on the FIRST `__`)."""
    base = os.path.basename(path)
    if base.endswith(".json"):
        base = base[:-5]
    if "__" in base:
        name, split = base.split("__", 1)
    else:
        name, split = base, "test"
    return f"{name}/{split}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred_dir", help="Directory of NeMo output manifests named <name>__<split>.json.")
    p.add_argument(
        "--pred_file",
        action="append",
        default=[],
        help="Score a single <name>__<split>.json manifest (repeatable). Alternative to --pred_dir; "
        "use with --no_summary to print just the per-dataset RESULT line as each dataset finishes.",
    )
    p.add_argument("--gt_field", default="reference", help="Ground-truth field in the manifest (default: reference).")
    p.add_argument("--pred_field", default="pred_text", help="Prediction field in the manifest (default: pred_text).")
    p.add_argument("--no_summary", action="store_true", help="Skip the summary table (just print RESULT lines).")
    p.add_argument("--verbose", action="store_true", help="Print a few normalized ref/hyp pairs per dataset.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.pred_file:
        files = sorted(f for f in args.pred_file if "__" in os.path.basename(f))
    elif args.pred_dir:
        files = sorted(f for f in glob.glob(os.path.join(args.pred_dir, "*.json")) if "__" in os.path.basename(f))
    else:
        _log("rescore: pass either --pred_dir or --pred_file")
        return 1
    if not files:
        _log(f"rescore: no <name>__<split>.json manifests found ({args.pred_dir or args.pred_file})")
        return 1

    results = []
    for fn in files:
        key = parse_key(fn)
        refs, hyps, missing = [], [], 0
        with open(fn, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if args.pred_field not in rec:
                    missing += 1
                    continue
                refs.append(rec.get(args.gt_field, "") or "")
                hyps.append(rec.get(args.pred_field, "") or "")
        if missing:
            _log(f"  WARN: {key}: {missing} lines missing '{args.pred_field}' (skipped)")
        if not refs:
            _log(f"  {key}: no scorable lines; skipping")
            continue
        wer = WER(normalize=True, verbose=args.verbose)
        wer.update(key, refs=refs, hyps=hyps)
        wer_val = float(wer.compute()["wer"]) * 100.0
        _log(f"RESULT\t{key}\t{wer_val:.2f}\t0.0\t{len(refs)}")
        results.append({"key": key, "wer": wer_val, "n": len(refs)})

    if results and not args.no_summary:
        _log("\n  {:<28} {:>8} {:>10}".format("Dataset", "WER(%)", "N"))
        _log("  " + "-" * 48)
        tot = 0.0
        for r in results:
            _log(f"  {r['key']:<28} {r['wer']:>8.2f} {r['n']:>10d}")
            tot += r["wer"]
        _log("  " + "-" * 48)
        _log(f"  {'Average (macro)':<28} {tot / len(results):>8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
