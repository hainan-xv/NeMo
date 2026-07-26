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
"""Sharding + reduce helper for the *heh* Open-ASR-Leaderboard eval on OCI.

The heh decode engine is ``examples/speechlm2/streaming_stt_generate.py`` (a
Hydra CLI that ``from_pretrained``-loads a model and decodes a lhotse/NeMo
manifest). It has no notion of "leaderboard suite" or per-dataset WER, so this
helper wraps it for the balanced multi-GPU OCI job:

  build      Pool EVERY utterance across all datasets from the pre-staged cache
             (``<cache_dir>/<dataset>/<split>/_cache_manifest.jsonl``), shuffle
             with a fixed seed, deal round-robin into ``--num_shards`` shards,
             sort each shard by duration (efficient batching), and write one NeMo
             manifest per shard: ``<out_dir>/shard{k}_of{N}.json`` with lines
             ``{"audio_filepath","duration","text","dataset_key"}``. The
             ``dataset_key`` rides along as ``cut.custom["dataset_key"]`` so it
             survives lhotse resample/pad/sort and lands in the generations.

  aggregate  Read every ``<out_dir>/shard*_of*.generations.jsonl`` produced by
             streaming_stt_generate (each row carries ``dataset_key``,
             ``text`` = normalized ref, ``pred_text`` = normalized hyp), group by
             dataset, and print per-dataset WER + average. WER is additive over
             utterances, so pooling by dataset is identical to a non-sharded run.

Kept dependency-light (stdlib + optional soundfile) for the ``build`` step so it
runs quickly in the SLURM prolog. ``aggregate`` imports NeMo's ``word_error_rate``
to match heh's metric exactly.
"""
import argparse
import glob
import json
import os
import random
from collections import Counter, defaultdict
from typing import List, Tuple


def _parse_entries(datasets_arg: str) -> List[Tuple[str, str]]:
    entries = []
    for e in (x.strip() for x in datasets_arg.replace(" ", ",").split(",")):
        if not e:
            continue
        name, _, split = e.partition(":")
        entries.append((name, split or "test"))
    return entries


def read_cache_manifest(cache_dir: str, dataset: str, split: str, max_samples: int = 0):
    """Read a pre-staged split -> list of {path, ref, dur}. Reconstructs audio
    paths under cache_dir when the manifest's absolute paths don't exist (e.g.
    after rsync'ing the cache to a different root)."""
    path = os.path.join(cache_dir, dataset, split, "_cache_manifest.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No pre-staged manifest for {dataset}/{split} at {path}.")
    ds_dir = os.path.join(cache_dir, dataset, split)
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fp = rec["audio_filepath"]
            if not os.path.exists(fp):
                alt = os.path.join(ds_dir, os.path.basename(fp))
                if os.path.exists(alt):
                    fp = alt
            if not os.path.exists(fp):
                continue
            out.append(
                {
                    "path": fp,
                    "ref": rec.get("reference", rec.get("text", "")),
                    "dur": float(rec.get("duration", 0.0) or 0.0),
                }
            )
            if max_samples and len(out) >= max_samples:
                break
    return out


def _ensure_duration(item: dict) -> float:
    """Return a positive duration, computing it from the wav header if missing
    (NeMo/lhotse needs a real duration to build the cut)."""
    if item["dur"] and item["dur"] > 0:
        return item["dur"]
    try:
        import soundfile

        return float(soundfile.info(item["path"]).duration)
    except Exception:
        return 0.0


def cmd_build(args) -> int:
    entries = _parse_entries(args.datasets)
    items: List[dict] = []
    for dataset, split in entries:
        key = f"{dataset}/{split}"
        recs = read_cache_manifest(args.cache_dir, dataset, split, args.max_eval_samples)
        for r in recs:
            r["key"] = key
            items.append(r)
    if not items:
        raise SystemExit("build: no utterances found across any dataset")

    n = int(args.num_shards)
    order = list(range(len(items)))
    random.Random(args.shuffle_seed).shuffle(order)
    shards: List[List[dict]] = [[] for _ in range(n)]
    for pos, j in enumerate(order):
        shards[pos % n].append(items[j])

    os.makedirs(args.out_dir, exist_ok=True)
    total_by_ds = Counter(it["key"] for it in items)
    print(f"build: {len(items)} utts across {len(entries)} datasets -> {n} shards (seed={args.shuffle_seed})")
    for k in range(n):
        shard = shards[k]
        # Sort each shard by duration so decode batches hold similar-length clips.
        shard.sort(key=_ensure_duration)
        out_path = os.path.join(args.out_dir, f"shard{k}_of{n}.json")
        with open(out_path, "w") as f:
            for it in shard:
                f.write(
                    json.dumps(
                        {
                            "audio_filepath": it["path"],
                            "duration": _ensure_duration(it),
                            "text": it["ref"],
                            "dataset_key": it["key"],
                        }
                    )
                    + "\n"
                )
        print(f"  shard{k}_of{n}.json: {len(shard)} utts")
    print("  per-dataset totals: " + ", ".join(f"{k}={total_by_ds[k]}" for k in sorted(total_by_ds)))
    return 0


def _word_edit_distance(hyp_words: List[str], ref_words: List[str]) -> int:
    """Levenshtein distance between two word sequences (unit costs)."""
    n, m = len(hyp_words), len(ref_words)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        hi = hyp_words[i - 1]
        for j in range(1, m + 1):
            cost = 0 if hi == ref_words[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def _word_error_rate(hyps: List[str], refs: List[str]) -> float:
    """Corpus WER = sum(word edits) / sum(ref words).

    Matches NeMo's ``word_error_rate`` formula exactly; inputs are already
    whisper-normalized by streaming_stt_generate, so this needs no normalizer and
    avoids importing the heavy nemo.collections.asr package just to score."""
    errs, words = 0, 0
    for h, r in zip(hyps, refs):
        rw = r.split()
        errs += _word_edit_distance(h.split(), rw)
        words += len(rw)
    return errs / words if words else 0.0


def cmd_aggregate(args) -> int:
    files = sorted(glob.glob(os.path.join(args.out_dir, "shard*_of*.generations.jsonl")))
    if not files:
        print(f"aggregate: no shard generation files under {args.out_dir}")
        return 1
    # lat_sum/lat_words accumulate the word-weighted emission latency (proxy), when
    # streaming_stt_generate wrote per-utterance word_latency + n_words.
    groups = defaultdict(lambda: {"refs": [], "hyps": [], "lat_sum": 0.0, "lat_words": 0})
    n_rows = 0
    for fn in files:
        with open(fn) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = rec.get("dataset_key") or "unknown"
                groups[key]["refs"].append(rec.get("text", ""))
                groups[key]["hyps"].append(rec.get("pred_text", ""))
                if "word_latency" in rec and "n_words" in rec:
                    groups[key]["lat_sum"] += float(rec["word_latency"]) * int(rec["n_words"])
                    groups[key]["lat_words"] += int(rec["n_words"])
                n_rows += 1
    print(f"aggregate: {len(files)} shard files, {n_rows} utts -> {len(groups)} datasets")

    results = []
    for key in sorted(groups):
        g = groups[key]
        # text/pred_text are already whisper-normalized by streaming_stt_generate,
        # so a plain word_error_rate here matches heh's per-dataset WER exactly.
        wer_val = _word_error_rate(g["hyps"], g["refs"]) * 100.0
        lat = (g["lat_sum"] / g["lat_words"]) if g["lat_words"] else None
        lat_str = f"{lat:.3f}" if lat is not None else "-"
        print(f"RESULT\t{key}\t{wer_val:.2f}\t0.0\t{len(g['refs'])}\t{lat_str}")
        results.append({"key": key, "wer": wer_val, "n": len(g["refs"]), "lat": lat})

    if results:
        has_lat = any(r["lat"] is not None for r in results)
        print("\n  {:<28} {:>8} {:>10} {:>12}".format("Dataset", "WER(%)", "N", "WordLat(s)"))
        print("  " + "-" * 62)
        tot = 0.0
        lat_tot = 0.0
        lat_n = 0
        for r in results:
            ls = f"{r['lat']:.3f}" if r["lat"] is not None else "-"
            print(f"  {r['key']:<28} {r['wer']:>8.2f} {r['n']:>10d} {ls:>12}")
            tot += r["wer"]
            if r["lat"] is not None:
                lat_tot += r["lat"]
                lat_n += 1
        print("  " + "-" * 62)
        avg_lat = f"{lat_tot / lat_n:.3f}" if lat_n else "-"
        print(f"  {'Average':<28} {tot / len(results):>8.2f} {'':>10} {avg_lat:>12}")
        if has_lat:
            print("  (WordLat = mean emission time w.r.t. audio start; latency proxy, off by a constant.)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build pooled/sharded NeMo manifests.")
    b.add_argument("--cache_dir", required=True)
    b.add_argument("--datasets", required=True, help="Comma/space-separated 'name:split' list.")
    b.add_argument("--out_dir", required=True)
    b.add_argument("--num_shards", type=int, required=True)
    b.add_argument("--shuffle_seed", type=int, default=1234)
    b.add_argument("--max_eval_samples", type=int, default=0, help="Cap samples per dataset (0 = all).")
    b.set_defaults(func=cmd_build)

    a = sub.add_parser("aggregate", help="Reduce shard generations to per-dataset WER.")
    a.add_argument("--out_dir", required=True)
    a.set_defaults(func=cmd_aggregate)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
