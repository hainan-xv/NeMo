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


# ------------------------------------------------------------------------------
# Long-form variant of `build`.
#
# The Open-ASR-Leaderboard sets are short utterances staged in a fixed cache
# layout; long-form sets instead ship as a few NeMo manifests each holding a
# handful of VERY long recordings (minutes to ~1h). Differences from `build`:
#   * source = arbitrary manifests discovered under --longform_dir (each *.json /
#     *.jsonl is one dataset; its audio_filepath is resolved RELATIVE to the
#     manifest's own dir), not the <cache>/<ds>/<split> layout.
#   * balancing = greedy longest-processing-time by DURATION (not round-robin by
#     count): with wildly uneven, few clips, count-balancing would leave one GPU
#     with a 1h clip while others idle. LPT keeps wall time ~= total_dur / N.
# Output shard files use the SAME shard{k}_of{n}.json schema + dataset_key, so the
# decode fan-out and `aggregate` are shared with the short-form path unchanged.
# ------------------------------------------------------------------------------
def _longform_key(manifest_path: str) -> str:
    """dataset_key = the manifest's parent-directory name, so a dataset shipped as
    several manifests (e.g. apptek's 14 per-locale files under
    apptek_callcenter_dialogues/) pools into ONE key -> one WER. Gives the clean
    3-way long-form comparison: tedlium_longform / earnings22_longform /
    apptek_callcenter_dialogues. Falls back to the filename stem if the manifest
    sits at the root (no dataset subdir)."""
    parent = os.path.basename(os.path.dirname(os.path.abspath(manifest_path)))
    return parent or os.path.splitext(os.path.basename(manifest_path))[0]


def _is_nemo_manifest(path: str) -> bool:
    """True iff the first non-empty line is a JSON object with 'audio_filepath'.
    Filters out sidecar files like HF 'metadata.jsonl' (schema file_name/text)."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                return isinstance(rec, dict) and "audio_filepath" in rec
    except Exception:
        return False
    return False


def discover_longform_manifests(longform_dir: str) -> List[str]:
    out = []
    for root, _dirs, files in os.walk(longform_dir):
        for fn in files:
            if fn.endswith((".json", ".jsonl")):
                p = os.path.join(root, fn)
                if _is_nemo_manifest(p):
                    out.append(p)
    return sorted(out)


def read_longform_manifest(manifest_path: str, max_samples: int = 0):
    """Read one long-form manifest -> list of {path, ref, dur}. audio_filepath is
    resolved relative to the manifest's directory when not absolute (and falls back
    to <manifest_dir>/<basename> if the recorded path doesn't exist)."""
    base = os.path.dirname(os.path.abspath(manifest_path))
    out = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fp = rec.get("audio_filepath")
            if not fp:
                continue
            if not os.path.isabs(fp):
                fp = os.path.join(base, fp)
            if not os.path.exists(fp):
                alt = os.path.join(base, os.path.basename(rec["audio_filepath"]))
                if os.path.exists(alt):
                    fp = alt
            if not os.path.exists(fp):
                print(f"  WARN missing audio (skipped): {rec['audio_filepath']}")
                continue
            out.append(
                {
                    "path": fp,
                    "ref": rec.get("text", rec.get("reference", "")),
                    "dur": float(rec.get("duration", 0.0) or 0.0),
                }
            )
            if max_samples and len(out) >= max_samples:
                break
    return out


# Duration buckets (minutes) for the range-stratified "small" eval set. The 0-2min
# bucket is intentionally dropped (too little signal); the top edge is capped by
# --max_range_min (default 60), so the default ranges are
# [2-5, 5-10, 10-20, 20-40, 40-60).
_RANGE_EDGES_MIN = [2, 5, 10, 20, 40, 60]


def _range_stratified_select(all_items, per_range, num_shards, max_range_min):
    """Pick the SHORTEST `per_range` utts in each minute bucket up to max_range_min,
    and assign the i-th shortest of every bucket to shard (i % num_shards). With
    per_range == num_shards == 8 that puts one utt per bucket on each GPU, so every
    GPU sees the same duration mix (range-balanced) -- a small, representative eval
    set spanning short..long recordings without the multi-hour clips.

    Returns (items, summary) where each item gets it['_shard'] and summary is a list
    of (lo_min, hi_min, n_selected) for logging.
    """
    edges = [e for e in _RANGE_EDGES_MIN if e < max_range_min] + [max_range_min]
    ranges = list(zip(edges[:-1], edges[1:]))
    picked, summary = [], []
    for lo, hi in ranges:
        inb = sorted(
            (it for it in all_items if lo * 60 <= _ensure_duration(it) < hi * 60),
            key=_ensure_duration,
        )[:per_range]
        for i, it in enumerate(inb):
            it["_shard"] = i % num_shards
            picked.append(it)
        summary.append((lo, hi, len(inb)))
    return picked, summary


def cmd_build_longform(args) -> int:
    manifests = discover_longform_manifests(args.longform_dir)
    if not manifests:
        raise SystemExit(f"build_longform: no *.json/*.jsonl manifests under {args.longform_dir}")

    # Collect ALL utts (tagged with their dataset key), then optionally cap to the
    # globally SHORTEST `max_eval_samples` across every dataset (a pooled cap, NOT
    # per-dataset/per-manifest). This is the quick_run path: e.g. cap=8 selects the
    # 8 shortest recordings anywhere, which the greedy balancer below then spreads
    # one-per-shard (one per GPU) -> a fast end-to-end smoke test. cap=0 keeps all.
    all_items: List[dict] = []
    for mp in manifests:
        key = _longform_key(mp)
        for r in read_longform_manifest(mp, 0):
            r["key"] = key
            all_items.append(r)
    if not all_items:
        raise SystemExit("build_longform: no usable utterances found")

    stratified = bool(getattr(args, "range_stratified", False))
    cap = int(args.max_eval_samples)
    if stratified:
        # Range-stratified "small" set: shortest `per_range` per minute bucket, with
        # a deterministic bucket->shard layout (overrides the greedy balancer below
        # in whole-recording mode).
        per_range = int(args.per_range)
        max_range_min = float(args.max_range_min)
        items, strat_summary = _range_stratified_select(all_items, per_range, int(args.num_shards), max_range_min)
        if not items:
            raise SystemExit("build_longform: range-stratified selection matched no utterances "
                             f"(ranges up to {max_range_min:g} min).")
        print(f"build_longform: range-stratified -> shortest {per_range} per bucket, up to {max_range_min:g} min:")
        for lo, hi, c in strat_summary:
            print("    [%2g-%-3g min): %d utts" % (lo, hi, c))
    elif cap and len(all_items) > cap:
        items = sorted(all_items, key=_ensure_duration)[:cap]  # globally shortest N
        print(f"build_longform: quick cap -> {cap} globally shortest utts "
              f"(of {len(all_items)}); max selected clip {_ensure_duration(max(items, key=_ensure_duration)) / 60.0:.1f} min")
    else:
        items = all_items

    # Give every selected utterance a stable unique id (used to reassemble windows).
    for i, it in enumerate(items):
        it["uid"] = f"u{i:06d}"

    # ---- Optional WINDOWED mode -------------------------------------------------
    # window_sec>0 => split each recording into fixed windows of ~window_sec,
    # snapped to a whole number of chunks (chunk_size * frame_sec), and decode each
    # window as an INDEPENDENT cut (offset/duration into the same file -- no audio
    # is written). Each window row carries utt_id + window_index so the window
    # aggregator can concatenate a recording's hyps back in order and score against
    # its full reference. A sidecar utt_map.json holds the full refs.
    window_sec = float(getattr(args, "window_sec", 0) or 0.0)
    utt_map = {}
    units: List[dict] = []  # decode units (a whole utt, or a single window)
    if window_sec > 0:
        chunk_dur = int(args.chunk_size) * float(args.frame_sec)
        win = max(1, round(window_sec / chunk_dur)) * chunk_dur  # snap to whole chunks
        for it in items:
            dur = max(_ensure_duration(it), 0.0)
            offs, idx = 0.0, 0
            while offs < dur - 1e-3:
                wdur = min(win, dur - offs)
                if wdur < 0.05:
                    break
                units.append({
                    "path": it["path"], "offset": round(offs, 3), "duration": round(wdur, 3),
                    "key": it["key"], "utt_id": it["uid"], "window_index": idx,
                })
                offs += win
                idx += 1
            if idx == 0:  # recording shorter than one window
                units.append({
                    "path": it["path"], "offset": 0.0, "duration": round(dur, 3),
                    "key": it["key"], "utt_id": it["uid"], "window_index": 0,
                })
                idx = 1
            utt_map[it["uid"]] = {"text": it["ref"], "dataset_key": it["key"],
                                  "duration": dur, "n_windows": idx}
        print(f"build_longform: windowed mode | target={window_sec:.2f}s -> "
              f"snapped window={win:.2f}s ({int(round(win / chunk_dur))} x {args.chunk_size}-frame chunks) | "
              f"{len(items)} utts -> {len(units)} windows")
    else:
        for it in items:
            units.append({
                "path": it["path"], "offset": None, "duration": _ensure_duration(it),
                "key": it["key"], "text": it["ref"], "utt_id": it["uid"], "window_index": 0,
                "_shard": it.get("_shard"),
            })

    n = int(args.num_shards)
    shards: List[List[dict]] = [[] for _ in range(n)]
    loads = [0.0] * n
    if stratified and window_sec <= 0 and all(u.get("_shard") is not None for u in units):
        # Honor the bucket->shard layout so each GPU gets the same duration mix.
        for u in units:
            k = int(u["_shard"]) % n
            shards[k].append(u)
            loads[k] += max(float(u["duration"]), 0.0)
    else:
        # Greedy longest-processing-time: assign the longest unit to the currently
        # least-loaded shard so total duration is spread as evenly as possible.
        for u in sorted(units, key=lambda x: float(x["duration"]), reverse=True):
            k = min(range(n), key=lambda i: loads[i])
            shards[k].append(u)
            loads[k] += max(float(u["duration"]), 0.0)

    os.makedirs(args.out_dir, exist_ok=True)
    if window_sec > 0:
        with open(os.path.join(args.out_dir, "longform_utt_map.json"), "w") as f:
            json.dump(utt_map, f)
    total_by_ds = Counter(it["key"] for it in items)
    unit_word = "windows" if window_sec > 0 else "utts"
    print(f"build_longform: {len(units)} {unit_word} across {len(manifests)} manifests -> {n} shards")
    for k in range(n):
        shard = shards[k]
        shard.sort(key=lambda x: float(x["duration"]))  # ascending within shard
        out_path = os.path.join(args.out_dir, f"shard{k}_of{n}.json")
        with open(out_path, "w") as f:
            for u in shard:
                rec = {
                    "audio_filepath": u["path"],
                    "duration": float(u["duration"]),
                    # Per-window rows have no per-window reference; the full ref lives
                    # in utt_map. Non-windowed rows keep the recording's full text so
                    # the plain `aggregate` path works unchanged.
                    "text": u.get("text", ""),
                    "dataset_key": u["key"],
                    "utt_id": u["utt_id"],
                    "window_index": u["window_index"],
                }
                if u.get("offset") is not None:
                    rec["offset"] = u["offset"]
                f.write(json.dumps(rec) + "\n")
        print(f"  shard{k}_of{n}.json: {len(shard)} {unit_word}, {loads[k] / 60.0:.1f} min audio")
    print("  per-dataset totals (utts): " + ", ".join(f"{k}={total_by_ds[k]}" for k in sorted(total_by_ds)))
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


def cmd_aggregate_longform_windows(args) -> int:
    """Reduce WINDOWED long-form shards to per-dataset WER.

    Each generation row is one ~fixed-length window tagged with utt_id +
    window_index. We concatenate a recording's windows back together in order to
    form its full hypothesis, then score it against the recording's full reference
    (from utt_map.json). Both sides are whisper-normalized (English) so the number
    is comparable to the full-streaming `aggregate` path. This is the "restart
    every N seconds, then stitch" long-form method.
    """
    files = sorted(glob.glob(os.path.join(args.out_dir, "shard*_of*.generations.jsonl")))
    if not files:
        print(f"aggregate_longform_windows: no shard generation files under {args.out_dir}")
        return 1
    with open(args.utt_map) as f:
        utt_map = json.load(f)

    # Gather windows per utterance: utt_id -> list of (window_index, pred_text).
    by_utt = defaultdict(list)
    n_rows = 0
    for fn in files:
        with open(fn) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                uid = rec.get("utt_id")
                if uid is None:
                    continue
                by_utt[uid].append((int(rec.get("window_index", 0)), rec.get("pred_text", "")))
                n_rows += 1

    try:
        from whisper_normalizer.english import EnglishTextNormalizer
        normalize = EnglishTextNormalizer()
    except Exception:  # noqa: BLE001 - normalization is best-effort; fall back to identity
        def normalize(x):
            return x

    # Concatenate each recording's windows in order; score vs its full reference.
    groups = defaultdict(lambda: {"refs": [], "hyps": []})
    n_missing = 0
    for uid, meta in utt_map.items():
        wins = sorted(by_utt.get(uid, []), key=lambda t: t[0])
        if not wins:
            n_missing += 1
            continue
        hyp = normalize(" ".join(p for _, p in wins).strip())
        ref = normalize(meta.get("text", ""))
        key = meta.get("dataset_key", "unknown")
        groups[key]["refs"].append(ref)
        groups[key]["hyps"].append(hyp)
    print(f"aggregate_longform_windows: {len(files)} shard files, {n_rows} windows -> "
          f"{sum(len(g['refs']) for g in groups.values())} recordings across {len(groups)} datasets"
          + (f" ({n_missing} recordings had NO decoded windows)" if n_missing else ""))

    results = []
    for key in sorted(groups):
        g = groups[key]
        wer_val = _word_error_rate(g["hyps"], g["refs"]) * 100.0
        print(f"RESULT\t{key}\t{wer_val:.2f}\t0.0\t{len(g['refs'])}\t-")
        results.append({"key": key, "wer": wer_val, "n": len(g["refs"]), "lat": None})

    if results:
        print("\n  {:<28} {:>8} {:>10}".format("Dataset", "WER(%)", "N(recordings)"))
        print("  " + "-" * 50)
        tot = 0.0
        for r in results:
            print(f"  {r['key']:<28} {r['wer']:>8.2f} {r['n']:>10d}")
            tot += r["wer"]
        print("  " + "-" * 50)
        print(f"  {'Average':<28} {tot / len(results):>8.2f}")
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

    bl = sub.add_parser("build_longform", help="Build duration-balanced shards from long-form manifests.")
    bl.add_argument("--longform_dir", required=True, help="Root holding long-form NeMo manifests (*.json).")
    bl.add_argument("--out_dir", required=True)
    bl.add_argument("--num_shards", type=int, required=True)
    bl.add_argument("--max_eval_samples", type=int, default=0,
                    help="Cap to the globally shortest N utts across ALL datasets (quick_run; 0 = all).")
    bl.add_argument("--window_sec", type=float, default=0.0,
                    help="If >0, split each recording into ~this-many-second windows (snapped to a whole "
                         "number of chunks) and decode each independently; 0 = whole-recording streaming.")
    bl.add_argument("--chunk_size", type=int, default=14, help="Encoder frames/chunk (windows snap to a multiple).")
    bl.add_argument("--frame_sec", type=float, default=0.08, help="Seconds per encoder frame (default 0.08 = 80ms).")
    bl.add_argument("--range_stratified", action="store_true",
                    help="Build the small range-stratified set: shortest --per_range utts per minute bucket "
                         "(2-5,5-10,... up to --max_range_min), one per shard.")
    bl.add_argument("--per_range", type=int, default=8, help="Utts per minute bucket for --range_stratified.")
    bl.add_argument("--max_range_min", type=float, default=60.0,
                    help="Top minute-bucket edge for --range_stratified (default 60; excludes longer clips).")
    bl.set_defaults(func=cmd_build_longform)

    aw = sub.add_parser("aggregate_longform_windows",
                        help="Reduce windowed long-form shards: concat per-utt windows, WER vs full ref.")
    aw.add_argument("--out_dir", required=True)
    aw.add_argument("--utt_map", required=True, help="longform_utt_map.json written by build_longform --window_sec.")
    aw.set_defaults(func=cmd_aggregate_longform_windows)

    a = sub.add_parser("aggregate", help="Reduce shard generations to per-dataset WER.")
    a.add_argument("--out_dir", required=True)
    a.set_defaults(func=cmd_aggregate)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
