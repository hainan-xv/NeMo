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

By default it reads the transcripts straight from the NeMo **JSON manifests**
(the ``text`` field), which is:
  * exactly the text training tokenizes, and
  * metadata-only -- it never opens the audio tars / AIStore, so it is fast and
    robust (tarred datasets whose tars live on S3 are scanned from their local
    lustre manifests without fetching a single tar).

For a lhotse ``input_cfg`` (the file the launchers pass as
``model.train_ds.input_cfg``) it walks the config, collects every
``manifest_filepath`` (expanding ``_OP_a..b_CL_`` / brace / glob shard patterns),
and reads those ``.json`` / ``.jsonl`` (optionally ``.gz``) manifests directly.

Examples
--------
# Granary 2.0 lhotse input_cfg (reads its JSON manifests directly, no tars):
python scripts/check_multiple_spaces.py \
    --input_cfg /lustre/.../granary_v2_..._iad_s3_audio.yaml \
    --max 200000 --examples 20

# A plain manifest (or comma-separated list / _OP_.._CL_ / brace / glob pattern):
python scripts/check_multiple_spaces.py --manifest /data/.../mcv11_dev.json

# Custom transcript field (default: text):
python scripts/check_multiple_spaces.py --input_cfg ... --text_field text

# Fall back to the lhotse CutSet iterator (WARNING: opens audio tars for
# nemo_tarred data; only use for non-tarred / lhotse-native manifests):
python scripts/check_multiple_spaces.py --input_cfg ... --use_lhotse

Notes
-----
* --max caps how many transcripts are scanned (0 = all) for a quick spot-check.
"""

import argparse
import glob
import gzip
import json
import re
import sys
from collections import Counter


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_cfg", type=str, help="Path/URL to a lhotse input_cfg YAML (as passed to training).")
    src.add_argument("--manifest", type=str, help="NeMo manifest path(s): comma list / _OP_a..b_CL_ / brace / glob.")
    p.add_argument("--text_field", type=str, default="text", help="Transcript field in the manifest (default: text).")
    p.add_argument("--max", type=int, default=0, help="Max transcripts to scan (0 = all).")
    p.add_argument("--examples", type=int, default=20, help="How many offending examples to print (default: 20).")
    p.add_argument("--report_every", type=int, default=200000, help="Progress log cadence in #transcripts.")
    p.add_argument(
        "--use_lhotse",
        action="store_true",
        help="Iterate the lhotse CutSet instead of reading manifests directly (opens tars for nemo_tarred data).",
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
    if RE_MULTI_WS.search(text) and not RE_MULTI_SPACE.search(text):
        tags.add("multi_other_ws")         # 2+ ws run that isn't plain double-space
    return tags


# --------------------------------------------------------------------------- #
# Manifest discovery + reading (default path: text-only, no audio)
# --------------------------------------------------------------------------- #
def _brace_expand(s: str):
    """Expand ``{a..b}`` numeric ranges and ``{a,b,c}`` lists (self-contained)."""
    try:
        from braceexpand import braceexpand

        return list(braceexpand(s))
    except Exception:
        pass
    # Fallback: expand a single {a..b} range or {a,b,...} list.
    m = re.search(r"\{(\d+)\.\.(\d+)\}", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return [s[: m.start()] + str(i) + s[m.end() :] for i in range(a, b + 1)]
    m = re.search(r"\{([^{}]+)\}", s)
    if m and "," in m.group(1):
        return [s[: m.start()] + opt + s[m.end() :] for opt in m.group(1).split(",")]
    return [s]


def _expand(path_spec: str):
    """Expand a NeMo manifest path spec into concrete files.

    Handles the NeMo ``_OP_a..b_CL_`` shard syntax, brace expansion, plain globs,
    and comma-separated lists -- without importing NeMo (keeps this fast/robust).
    """
    out = []
    for part in str(path_spec).split(","):
        part = part.strip()
        if not part:
            continue
        # NeMo shard syntax: _OP_a..b_CL_  <=>  {a..b}
        norm = part.replace("_OP_", "{").replace("_CL_", "}")
        for cand in _brace_expand(norm):
            if any(c in cand for c in "*?["):
                out.extend(sorted(glob.glob(cand)))
            else:
                out.append(cand)
    return out


def _collect_manifest_filepaths(node):
    """Recursively collect every ``manifest_filepath`` value from a parsed input_cfg."""
    found = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "manifest_filepath" and v is not None:
                if isinstance(v, (list, tuple)):
                    found.extend(str(x) for x in v)
                else:
                    found.append(str(v))
            else:
                found.extend(_collect_manifest_filepaths(v))
    elif isinstance(node, (list, tuple)):
        for item in node:
            found.extend(_collect_manifest_filepaths(item))
    return found


def _open_maybe_gz(path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, "rt", encoding="utf-8")


def iter_texts_from_manifests(manifest_files, text_field):
    """Yield (source_id, text) from NeMo json/jsonl(.gz) manifests, one JSON object per line."""
    for mf in manifest_files:
        try:
            fh = _open_maybe_gz(mf)
        except FileNotFoundError:
            print(f"  WARNING: manifest not found, skipping: {mf}", file=sys.stderr)
            continue
        with fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = entry.get(text_field, entry.get("text"))
                cid = entry.get("audio_filepath") or entry.get("id") or mf
                if isinstance(text, str):
                    yield cid, text


def resolve_manifest_files(args):
    if args.manifest is not None:
        files = _expand(args.manifest)
    else:
        # Import from lhotse directly (light) rather than via NeMo, so reading the
        # YAML doesn't pull in the heavy NeMo/megatron import chain.
        from lhotse.serialization import load_yaml

        cfg = load_yaml(args.input_cfg)
        specs = _collect_manifest_filepaths(cfg)
        if not specs:
            raise SystemExit(
                f"No 'manifest_filepath' entries found in {args.input_cfg}. "
                "If this is a lhotse-native (shar/cuts) config, re-run with --use_lhotse."
            )
        files = []
        for s in specs:
            files.extend(_expand(s))
    return files


# --------------------------------------------------------------------------- #
# Optional lhotse CutSet path (opens tars for nemo_tarred data!)
# --------------------------------------------------------------------------- #
def iter_texts_from_lhotse(args):
    from nemo.collections.common.data.lhotse.cutset import read_cutset_from_config

    config = {"input_cfg": args.input_cfg} if args.input_cfg else {"manifest_filepath": args.manifest}
    cuts, is_tarred = read_cutset_from_config(config)
    print(f"[check_multiple_spaces] lhotse CutSet is_tarred={is_tarred}", file=sys.stderr)
    for cut in cuts:
        cid = getattr(cut, "id", "?")
        for sup in (getattr(cut, "supervisions", None) or []):
            t = getattr(sup, "text", None)
            if isinstance(t, str):
                yield cid, t
        custom = getattr(cut, "custom", None)
        if isinstance(custom, dict) and isinstance(custom.get(args.text_field), str):
            yield cid, custom[args.text_field]


def main():
    args = parse_args()

    if args.use_lhotse:
        source_desc = "lhotse CutSet"
        text_iter = iter_texts_from_lhotse(args)
    else:
        manifest_files = resolve_manifest_files(args)
        source_desc = f"{len(manifest_files)} JSON manifest file(s)"
        print(f"[check_multiple_spaces] scanning {source_desc} (text-only; no audio/tars).")
        if manifest_files[:3]:
            for mf in manifest_files[:3]:
                print(f"    e.g. {mf}")
        text_iter = iter_texts_from_manifests(manifest_files, args.text_field)

    n_texts = 0
    n_bad = 0
    tag_counts = Counter()
    examples = []

    for cid, text in text_iter:
        n_texts += 1
        tags = classify(text)
        if tags:
            n_bad += 1
            for t in tags:
                tag_counts[t] += 1
            if len(examples) < args.examples:
                examples.append((cid, sorted(tags), text))
        if args.report_every and n_texts % args.report_every == 0:
            print(f"  ... scanned {n_texts} transcripts | bad={n_bad} ({_pct(n_bad, n_texts)})")
        if args.max and n_texts >= args.max:
            print(f"  reached --max={args.max}; stopping early.")
            break

    print("\n================ whitespace scan summary ================")
    print(f"source                  : {source_desc}")
    print(f"transcripts scanned     : {n_texts}")
    print(f"transcripts with issues : {n_bad} ({_pct(n_bad, n_texts)})")
    print("by issue type (transcript counts):")
    for tag in ("multi_space", "lead_trail_ws", "tab_or_newline", "multi_other_ws"):
        print(f"    {tag:16s}: {tag_counts.get(tag, 0)}")

    if examples:
        print(f"\n---- up to {args.examples} offending examples (repr shows the exact whitespace) ----")
        for cid, tags, text in examples:
            print(f"  [{','.join(tags)}] id={cid}\n      {text!r}")
    print("========================================================")

    sys.exit(1 if tag_counts.get("multi_space", 0) > 0 else 0)


def _pct(num, den):
    return f"{(100.0 * num / den):.4f}%" if den else "n/a"


if __name__ == "__main__":
    main()
