"""Open-ASR-Leaderboard evaluation for a trained CHAT transducer.

    # everything, one GPU
    PYTHONPATH=. python scripts/speechlm2/chat_leaderboard_eval.py --ckpt <path>

    # one shard of an N-way split (the launcher runs one per GPU), then merge
    PYTHONPATH=. python scripts/speechlm2/chat_leaderboard_eval.py --ckpt <p> \
        --shard 0 --num-shards 8 --output-dir <dir>
    PYTHONPATH=. python scripts/speechlm2/chat_leaderboard_eval.py --aggregate --output-dir <dir>

    # sweep the decode-time retract knob on one checkpoint
    ... --retract 1

Reports per-dataset WER, a MACRO average over datasets (the leaderboard's
headline) and a POOLED WER over all utterances. Both are printed because they
answer different questions and diverge sharply here: spgispeech alone is 39,341
of the 74,842 utterances, so pooling is dominated by it while the macro average
is not.

Scoring matches launch/eval_leaderboard.sh -- Whisper's English normaliser plus
kaldialign edit distance -- so a number here is comparable to the SCRIPT and
nemotron columns.
"""

import argparse
import json
import os
import time

import torch

from nemo.collections.speechlm2.parts.chat_eval import (
    build_chat_tokenizer,
    find_splits,
    load_chat_model,
    read_manifest,
    score_pairs,
    transcribe_manifest,
)

DEFAULT_CACHE = os.path.expanduser("~/leaderboard_cache")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", help="training .ckpt (or an averaged one)")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--datasets", default=None, help="comma-separated dataset:split; default is everything cached")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--retract", type=int, default=None, help="retract-by-k decoding; default = checkpoint's setting")
    ap.add_argument("--max-samples", type=int, default=None, help="cap utterances PER dataset (smoke tests)")
    ap.add_argument("--max-duration", type=float, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--output-dir", default=None, help="where shard results are written / read")
    ap.add_argument("--aggregate", action="store_true", help="merge shard results and print the table")
    ap.add_argument("--pretrained-asr", default=None)
    ap.add_argument("--pretrained-llm", default=None)
    ap.add_argument("--save-hyps", action="store_true", help="also write per-utterance hypotheses")
    return ap.parse_args()


def report(per_ds: dict, tag: str = ""):
    """Print the per-dataset table plus macro and pooled WER."""
    names = sorted(per_ds)
    print(f"\n=== Open ASR Leaderboard{(' -- ' + tag) if tag else ''} ===")
    print(f"{'dataset':<28} {'utts':>7} {'ref words':>10} {'WER':>8}")
    print("-" * 56)
    tot_e = tot_w = 0
    for n in names:
        d = per_ds[n]
        tot_e += d["errors"]
        tot_w += d["ref_words"]
        print(f"{n:<28} {d['utts']:>7} {d['ref_words']:>10} {d['wer']*100:>7.2f}")
    macro = sum(per_ds[n]["wer"] for n in names) / max(len(names), 1)
    pooled = tot_e / tot_w if tot_w else float("nan")
    print("-" * 56)
    print(f"{'MACRO (mean over datasets)':<28} {'':>7} {'':>10} {macro*100:>7.2f}")
    print(f"{'POOLED (all utterances)':<28} {sum(per_ds[n]['utts'] for n in names):>7} {tot_w:>10} {pooled*100:>7.2f}")
    return {"per_dataset": per_ds, "macro_wer": macro, "pooled_wer": pooled}


def main():
    args = parse_args()

    if args.aggregate:
        if not args.output_dir:
            raise SystemExit("--aggregate needs --output-dir")
        per_ds = {}
        shards = sorted(f for f in os.listdir(args.output_dir) if f.startswith("shard") and f.endswith(".json"))
        if not shards:
            raise SystemExit(f"no shard*.json under {args.output_dir}")
        for f in shards:
            with open(os.path.join(args.output_dir, f)) as fh:
                per_ds.update(json.load(fh)["per_dataset"])
        summary = report(per_ds, "aggregated")
        with open(os.path.join(args.output_dir, "results.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nwrote {os.path.join(args.output_dir, 'results.json')}")
        return

    if not args.ckpt:
        raise SystemExit("--ckpt is required unless --aggregate")

    splits = (
        [tuple(x.split(":", 1)) for x in args.datasets.split(",")] if args.datasets else find_splits(args.cache_dir)
    )

    # Shard by dataset. Uneven, but the alternative -- splitting utterances --
    # would make each shard load the 609M model for a fraction of a dataset and
    # complicate pooling. spgispeech is over half the corpus, so give it its own
    # shard by ordering largest-first.
    def _size(ds_split):
        m = os.path.join(args.cache_dir, ds_split[0], ds_split[1], "_cache_manifest.jsonl")
        try:
            return -os.path.getsize(m)
        except OSError:
            return 0

    splits = sorted(splits, key=_size)
    mine = [s for i, s in enumerate(splits) if i % args.num_shards == args.shard]
    if not mine:
        print(f"shard {args.shard}/{args.num_shards}: nothing to do")
        return
    print(f"shard {args.shard}/{args.num_shards}: {', '.join(f'{a}:{b}' for a, b in mine)}")

    model, cfg = load_chat_model(args.ckpt, args.device, args.retract, args.pretrained_asr, args.pretrained_llm)
    model.tokenizer = build_chat_tokenizer(cfg)
    print(
        f"  vocab={cfg.get('vocab_size')}  joint_history_chunks={cfg.get('joint_history_chunks', 0)}  "
        f"retract_words={cfg.get('retract_words', 0)}"
    )

    per_ds, hyp_dump = {}, {}
    for ds, split in mine:
        man = os.path.join(args.cache_dir, ds, split, "_cache_manifest.jsonl")
        rows = read_manifest(man, args.max_samples, args.max_duration)
        t0 = time.perf_counter()
        hyps = transcribe_manifest(model, model.tokenizer, rows, args.batch_size, args.device, progress_every=20)
        dt = time.perf_counter() - t0
        refs = [r["reference"] for r in rows]
        s = score_pairs(refs, hyps)
        s["utts"] = len(rows)
        name = f"{ds}:{split}"
        per_ds[name] = s
        audio = sum(r.get("duration", 0) for r in rows)
        print(f"  {name:<28} {len(rows):>6} utts  WER {s['wer']*100:6.2f}   ({dt:.0f}s, {audio/max(dt,1e-9):.0f}x RT)")
        if args.save_hyps:
            hyp_dump[name] = [
                {"audio_filepath": r["audio_filepath"], "reference": r["reference"], "hypothesis": h}
                for r, h in zip(rows, hyps)
            ]

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f"shard{args.shard}.json"), "w") as fh:
            json.dump({"per_dataset": per_ds}, fh, indent=2)
        if args.save_hyps:
            with open(os.path.join(args.output_dir, f"hyps{args.shard}.json"), "w") as fh:
                json.dump(hyp_dump, fh)
    if args.num_shards == 1:
        report(per_ds)


if __name__ == "__main__":
    main()
