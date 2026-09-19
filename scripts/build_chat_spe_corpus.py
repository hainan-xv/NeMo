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
"""Extract a training-text corpus for building CHAT SentencePiece vocabularies.

WHY NOT JUST CAT THE MANIFESTS. The Granary v2 config lists 54 corpus entries
expanding to ~61k shards, and the trainer samples them with per-corpus WEIGHTS
that differ by three orders of magnitude (0.0049 for AMI, 0.062 for davidai).
A vocabulary built by reading shards uniformly would be fitted to a distribution
the model never sees -- over-representing small corpora and under-representing
the ones that dominate training. So lines are drawn in proportion to weight.

Text is taken VERBATIM: capitalization and punctuation are part of what these
models must emit, so nothing is lowercased or stripped here. (The one repair is
collapsing whitespace runs, which would otherwise train pieces that encode a
formatting artefact -- see nemo/collections/asr/parts/utils/chat_alignment.py.)

    python scripts/build_chat_spe_corpus.py \\
        --input_cfg granary_v2_en_full_...yaml --out corpus.txt --max_lines 3000000
"""

import argparse
import json
import os
import random
import re
import sys


def _expand(path):
    """``manifest__OP_0..64_CL_.jsonl`` -> the 65 shard paths it stands for."""
    m = re.search(r"(.*)_OP_(\d+)\.\.(\d+)_CL_(.*)", path)
    if not m:
        return [path]
    pre, lo, hi, post = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    return [f"{pre}{i}{post}" for i in range(lo, hi + 1)]


def _entries(cfg):
    """``(manifest_filepath, weight)`` for every corpus in the input config."""
    out = []

    def walk(o):
        if isinstance(o, dict):
            if "manifest_filepath" in o and isinstance(o["manifest_filepath"], str):
                out.append((o["manifest_filepath"], float(o.get("weight", 0.0) or 0.0)))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(cfg)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_cfg", required=True, help="the training input_cfg YAML")
    p.add_argument("--out", required=True, help="output text file, one transcript per line")
    p.add_argument("--max_lines", type=int, default=3_000_000)
    p.add_argument("--max_shards_per_corpus", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import yaml

    cfg = yaml.safe_load(open(args.input_cfg))
    ents = _entries(cfg)
    total_w = sum(w for _, w in ents) or 1.0
    rnd = random.Random(args.seed)
    print(f"corpora: {len(ents)}  total weight: {total_w:.4f}", flush=True)

    written = 0
    per_corpus = []
    with open(args.out, "w") as fout:
        for man, w in ents:
            # Proportional budget, with a floor so a tiny-weight corpus still
            # contributes SOME pieces rather than vanishing from the vocabulary.
            budget = max(2000, int(args.max_lines * (w / total_w)))
            shards = _expand(man)
            rnd.shuffle(shards)
            shards = shards[: args.max_shards_per_corpus]
            n = 0
            for sh in shards:
                if n >= budget:
                    break
                if not os.path.exists(sh):
                    continue
                try:
                    with open(sh) as f:
                        for line in f:
                            if n >= budget:
                                break
                            try:
                                t = (json.loads(line).get("text") or "").strip()
                            except Exception:
                                continue
                            if not t:
                                continue
                            t = " ".join(t.split())  # collapse orphaning whitespace
                            fout.write(t + "\n")
                            n += 1
                except Exception as e:  # a bad shard must not abort the corpus
                    print(f"  [warn] {os.path.basename(sh)}: {type(e).__name__}", flush=True)
            written += n
            per_corpus.append((os.path.basename(os.path.dirname(os.path.dirname(man))) or man, w, n))

    print(f"\nwrote {written:,} lines -> {args.out}")
    print(f"{'corpus':28s} {'weight':>9s} {'lines':>10s}")
    for name, w, n in sorted(per_corpus, key=lambda x: -x[2])[:15]:
        print(f"  {name[:26]:26s} {w:9.5f} {n:10,d}")
    if written == 0:
        print("ERROR: no text extracted", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
