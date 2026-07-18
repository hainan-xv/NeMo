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

"""Scan an ASR dataset's transcripts for whitespace problems (esp. runs of 2+
consecutive spaces), which SentencePiece encodes as spurious word-boundary tokens.

It reads exactly the text training sees: it parses the SAME lhotse ``input_cfg``
(or NeMo ``manifest_filepath``) via NeMo's own ``read_cutset_from_config`` and
inspects each supervision's transcript. Iteration is metadata-only (no audio is
fetched), so it is fast.

Examples
--------
# Lhotse input_cfg (the file the launchers pass as model.train_ds.input_cfg):
python scripts/check_multiple_spaces.py \
    --input_cfg /lustre/.../granary_v2_..._iad_s3_audio.yaml \
    --max 200000 --examples 20

# A plain NeMo manifest (json/jsonl), or a comma-separated list / brace list:
python scripts/check_multiple_spaces.py --manifest /data/.../mcv11_dev.json

# Custom transcript field (default scans supervision.text and cut.custom[text_field]):
python scripts/check_multiple_spaces.py --input_cfg ... --text_field text

Notes
-----
* For shar / tarred data served over AIStore, run where AIS is reachable (e.g. an
  OCI node inside the training container) with AIS env vars set
  (AIS_ENDPOINT, AIS_AUTHN_TOKEN); only small metadata shards are fetched.
* Use --max to cap how many cuts are scanned for a quick spot-check on huge sets.
"""

import argparse
import re
import sys
from collections import Counter


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_cfg", type=str, help="Path/URL to a lhotse input_cfg YAML (as passed to training).")
    src.add_argument("--manifest", type=str, help="Path (or comma/brace list) to NeMo manifest json/jsonl file(s).")
    p.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="Custom transcript field to also check on cut.custom (default: text). supervision.text is always checked.",
    )
    p.add_argument("--max", type=int, default=0, help="Max cuts to scan (0 = all).")
    p.add_argument("--examples", type=int, default=20, help="How many offending examples to print (default: 20).")
    p.add_argument(
        "--report_every", type=int, default=200000, help="Progress log cadence in #cuts (default: 200000)."
    )
    return p.parse_args()


# Whitespace problem detectors. The primary concern is >=2 consecutive spaces.
RE_MULTI_SPACE = re.compile(r" {2,}")          # 2+ ASCII spaces in a row
RE_OTHER_WS = re.compile(r"[\t\n\r\f\v]")       # tabs / newlines / other control whitespace
RE_MULTI_WS = re.compile(r"\s{2,}")             # any run of 2+ whitespace chars


def classify(text: str):
    """Return the set of whitespace-issue tags for a transcript (empty if clean)."""
    tags = set()
    if text is None:
        return tags
    if RE_MULTI_SPACE.search(text):
        tags.add("multi_space")            # "a  b"
    if text != text.strip():
        tags.add("lead_trail_ws")          # leading/trailing whitespace
    if RE_OTHER_WS.search(text):
        tags.add("tab_or_newline")         # \t \n \r ...
    # A 2+ whitespace run that is not plain double-space (e.g. " \t", nbsp runs).
    if RE_MULTI_WS.search(text) and not RE_MULTI_SPACE.search(text):
        tags.add("multi_other_ws")
    return tags


def iter_texts(cut, text_field: str):
    """Yield candidate transcripts for a cut: every supervision.text + cut.custom[text_field]."""
    for sup in (getattr(cut, "supervisions", None) or []):
        t = getattr(sup, "text", None)
        if t is not None:
            yield t
    custom = getattr(cut, "custom", None)
    if isinstance(custom, dict) and text_field in custom and isinstance(custom[text_field], str):
        yield custom[text_field]


def build_cuts(args):
    from nemo.collections.common.data.lhotse.cutset import read_cutset_from_config

    if args.input_cfg is not None:
        config = {"input_cfg": args.input_cfg}
    else:
        # NeMo manifest path(s). read_nemo_manifest accepts a str/list; keep it simple.
        config = {"manifest_filepath": args.manifest}
    cuts, is_tarred = read_cutset_from_config(config)
    return cuts, is_tarred


def main():
    args = parse_args()
    cuts, is_tarred = build_cuts(args)
    print(f"[check_multiple_spaces] source={'input_cfg' if args.input_cfg else 'manifest'} is_tarred={is_tarred}")

    n_cuts = 0
    n_texts = 0
    n_bad_texts = 0
    n_bad_cuts = 0
    tag_counts = Counter()
    examples = []

    for cut in cuts:
        n_cuts += 1
        cut_bad = False
        for text in iter_texts(cut, args.text_field):
            n_texts += 1
            tags = classify(text)
            if tags:
                n_bad_texts += 1
                cut_bad = True
                for t in tags:
                    tag_counts[t] += 1
                if len(examples) < args.examples:
                    cid = getattr(cut, "id", "?")
                    examples.append((cid, sorted(tags), text))
        if cut_bad:
            n_bad_cuts += 1

        if args.report_every and n_cuts % args.report_every == 0:
            print(f"  ... scanned {n_cuts} cuts | bad_texts={n_bad_texts} ({_pct(n_bad_texts, n_texts)})")
        if args.max and n_cuts >= args.max:
            print(f"  reached --max={args.max}; stopping early.")
            break

    print("\n================ whitespace scan summary ================")
    print(f"cuts scanned            : {n_cuts}")
    print(f"transcripts scanned     : {n_texts}")
    print(f"transcripts with issues : {n_bad_texts} ({_pct(n_bad_texts, n_texts)})")
    print(f"cuts with issues        : {n_bad_cuts} ({_pct(n_bad_cuts, n_cuts)})")
    print("by issue type (transcript counts):")
    for tag in ("multi_space", "lead_trail_ws", "tab_or_newline", "multi_other_ws"):
        print(f"    {tag:16s}: {tag_counts.get(tag, 0)}")

    if examples:
        print(f"\n---- up to {args.examples} offending examples (repr shows the exact whitespace) ----")
        for cid, tags, text in examples:
            print(f"  [{','.join(tags)}] id={cid}\n      {text!r}")
    print("========================================================")

    # Non-zero exit if any multi-space (the main concern) was found, for CI/scripts.
    sys.exit(1 if tag_counts.get("multi_space", 0) > 0 else 0)


def _pct(num, den):
    return f"{(100.0 * num / den):.4f}%" if den else "n/a"


if __name__ == "__main__":
    main()
