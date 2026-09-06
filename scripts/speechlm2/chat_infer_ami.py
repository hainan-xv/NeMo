"""Run a trained CHAT transducer over the AMI test set and report WER.

    # quick smoke test: 20 utterances
    PYTHONPATH=. python scripts/speechlm2/chat_infer_ami.py --ckpt <path> --quick

    # 20 utterances, with retract-by-1 decoding
    PYTHONPATH=. python scripts/speechlm2/chat_infer_ami.py --ckpt <path> --quick --retract 1

    # the full 7,805-utterance test set
    PYTHONPATH=. python scripts/speechlm2/chat_infer_ami.py --ckpt <path>

WHY A SEPARATE SCRIPT FROM THE LEADERBOARD DRIVER. That one runs seven datasets
under Slurm on eight GPUs and needs the whole cache staged. This is for the loop
you actually want while iterating -- one dataset, one GPU, twenty utterances,
answer in under a minute -- and for sweeping decode-time knobs (--retract) on a
checkpoint without retraining.

WER is computed the same way as the leaderboard driver (Whisper's English
normaliser + kaldialign edit distance), so a number here is comparable to the
AMI column there. It is NOT comparable to val_wer, which runs on a different
(mcv-style) validation set.
"""

import argparse
import json
import os
import time

import torch


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="training .ckpt from the grid")
    ap.add_argument(
        "--manifest",
        default=os.path.expanduser("~/leaderboard_cache/ami_cleaned/test/_cache_manifest.jsonl"),
        help="AMI cache manifest (audio_filepath / duration / reference per line)",
    )
    ap.add_argument("--quick", action="store_true", help="only the first --n utterances")
    ap.add_argument("-n", type=int, default=20, help="utterances to run under --quick (default 20)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--retract",
        type=int,
        default=None,
        help="retract-by-k decoding: hand each chunk's last k words to the next chunk so the model "
        "re-decides them with more right context. Default: whatever the checkpoint was configured with.",
    )
    ap.add_argument("--max-duration", type=float, default=None, help="skip utterances longer than this (seconds)")
    ap.add_argument("--print-worst", type=int, default=5, help="show the N worst utterances at the end")
    ap.add_argument("--pretrained-asr", default=None, help="local .nemo, if the recorded path is unavailable")
    ap.add_argument("--pretrained-llm", default=None, help="local path or hub id for the Qwen-arm tokenizer")
    ap.add_argument("--out", default=None, help="write per-utterance hypotheses to this .jsonl")
    return ap.parse_args()


from nemo.collections.speechlm2.parts.chat_eval import (
    build_chat_tokenizer,
    load_chat_model,
    read_manifest,
    score_pairs,
    transcribe_manifest,
)


def main():
    args = parse_args()

    rows = []
    with open(args.manifest) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.max_duration:
        rows = [r for r in rows if r.get("duration", 0) <= args.max_duration]
    if args.quick:
        rows = rows[: args.n]
    print(f"{len(rows)} utterances, {sum(r.get('duration', 0) for r in rows)/60:.1f} min of audio")

    model, cfg = load_chat_model(args.ckpt, args.device, args.retract, args.pretrained_asr, args.pretrained_llm)
    model.tokenizer = build_chat_tokenizer(cfg)
    print(
        f"  vocab={cfg.get('vocab_size')}  joint_history_chunks={cfg.get('joint_history_chunks', 0)}  "
        f"retract_words={cfg.get('retract_words', 0)}"
    )

    import soundfile as sf

    from nemo.collections.speechlm2.parts.metrics.wer import WER

    hyps, refs, per_utt = [], [], []
    t0 = time.perf_counter()
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i : i + args.batch_size]
        waves = []
        for r in batch:
            w, sr = sf.read(r["audio_filepath"], dtype="float32")
            if w.ndim > 1:
                w = w.mean(axis=1)
            assert sr == 16000, f"expected 16 kHz, got {sr} in {r['audio_filepath']}"
            waves.append(torch.from_numpy(w))
        lens = torch.tensor([len(w) for w in waves])
        padded = torch.zeros(len(waves), int(lens.max()))
        for j, w in enumerate(waves):
            padded[j, : len(w)] = w

        with torch.no_grad():
            ids = model.transcribe_ids(padded.to(args.device), lens.to(args.device))
        for r, seq in zip(batch, ids):
            hyp = model.tokenizer.ids_to_text(list(seq)) if seq else ""
            hyps.append(hyp)
            refs.append(r["reference"])
            per_utt.append({"audio_filepath": r["audio_filepath"], "reference": r["reference"], "hypothesis": hyp})
        done = min(i + args.batch_size, len(rows))
        print(f"  {done}/{len(rows)}", end="\r", flush=True)

    elapsed = time.perf_counter() - t0
    audio_s = sum(r.get("duration", 0) for r in rows)
    print(f"\ndecoded in {elapsed:.1f}s  ({audio_s/max(elapsed,1e-9):.1f}x realtime)")

    wer = WER(normalize=True, verbose=False)
    wer.update("ami", refs=refs, hyps=hyps)
    scores = {k: float(v) for k, v in wer.compute().items()}
    print("\n=== AMI ===")
    for k, v in sorted(scores.items()):
        print(f"  {k}: {v:.4f}")

    # Per-utterance WER, to see WHERE it fails rather than only how much.
    import kaldialign
    from whisper_normalizer.english import EnglishTextNormalizer

    norm = EnglishTextNormalizer()
    scored = []
    for u in per_utt:
        r, h = norm(u["reference"]).split(), norm(u["hypothesis"]).split()
        if not r:
            continue
        ali = kaldialign.align(r, h, "*")
        err = sum(1 for a, b in ali if a != b)
        u["wer"] = err / len(r)
        scored.append(u)
    scored.sort(key=lambda x: -x["wer"])

    if args.print_worst and scored:
        print(f"\n=== {min(args.print_worst, len(scored))} worst utterances ===")
        for u in scored[: args.print_worst]:
            print(f"\n  wer={u['wer']:.3f}  {os.path.basename(u['audio_filepath'])}")
            print(f"    ref: {u['reference'][:220]}")
            print(f"    hyp: {u['hypothesis'][:220]}")

    if args.out:
        with open(args.out, "w") as f:
            for u in scored:
                f.write(json.dumps(u) + "\n")
        print(f"\nwrote {len(scored)} hypotheses to {args.out}")


if __name__ == "__main__":
    main()
