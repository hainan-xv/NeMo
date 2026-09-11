"""Measure the emission latency of a chunked CHAT model.

    python scripts/measure_latency.py --nemo m.nemo --manifest x.jsonl --trims 0,1,2

WHAT IS REPORTED. The mean wall-clock time at which output WORDS are committed.
No oracle word timings are needed: the same audio is decoded by every
configuration, so the mean is offset from true latency by a constant and the
DIFFERENCES between configurations are the quantity of interest.

Two conventions, both of which matter for a chunked model:

  * A word is emitted when its LAST subword is emitted. A word is not knowable
    until it is complete, so an earlier subword does not count.
  * A token is emitted at the END of its chunk, wherever it sits inside it.
    The joint sees the whole chunk before producing anything, so a token's
    position within the chunk carries no timing information.

The reported time is the CHUNK END, ``(t+1)*C``, with no credit for a trim.
That is what the run observes: the audio is fully buffered and ``frame_trim``
only masks frames inside the joint's attention, so nothing is waiting on input.

A streaming deployment that decoded chunk ``t`` as soon as ``(t+1)*C - d``
frames had arrived would be ``d`` frames faster, and ``--credit-trim`` reports
that instead. It is an assumption about deployment, not a measurement, so it is
off by default.
"""

import argparse
import json
import sys

import torch

FRAME_SECONDS = 0.08  # 10 ms hop x 8x subsampling


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nemo", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--trims", default="0")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--label", default="")
    p.add_argument(
        "--credit-trim",
        action="store_true",
        help="subtract d frames, i.e. assume a streaming pipeline in which the withheld frames had "
        "not yet arrived. Off by default: the run itself has the whole utterance buffered",
    )
    args = p.parse_args()

    from nemo.collections.asr.models import ASRModel
    from nemo.utils import model_utils

    cfg = ASRModel.restore_from(restore_path=args.nemo, return_config=True)
    cls = model_utils.import_class_by_path(cfg.target)
    model = cls.restore_from(restore_path=args.nemo, map_location="cuda").eval()

    # A CHAT joint emits per CHUNK, so a token is available once its whole chunk
    # has been read: (t+1)*C frames. A plain RNN-T emits per FRAME: (f+1) frames.
    # Both then wait the same encoder look-ahead (att_context right = 13 here),
    # which is common to the two and so cancels in any comparison between them.
    # WHEN IS A TOKEN AVAILABLE? Not when the decoder emits it, but when the
    # audio it depends on has arrived.
    #
    # Both model families here use chunked_limited encoder attention: every
    # frame inside a chunk attends to that chunk's right boundary, so NO frame
    # of a chunk exists until the whole chunk has been read. A frame-level RNN-T
    # on such an encoder is therefore just as chunk-synchronous as a CHAT joint
    # -- its finer emission granularity buys nothing in latency, because the
    # encoder output it consumes only materialises at chunk ends.
    #
    # So both are quantised to the same chunk grid: the CHAT joint by its own
    # chunk_size, a plain joint by the encoder's right context + 1.
    joint_chunk = int(getattr(model.joint, "chunk_size", 0) or 0)
    att = getattr(model.encoder, "att_context_size", None)
    style = str(getattr(model.encoder, "att_context_style", ""))
    enc_chunk = int(att[1]) + 1 if att is not None and "chunked" in style else 1
    chunked = joint_chunk > 0
    chunk = joint_chunk if chunked else enc_chunk

    paths, durs = [], []
    with open(args.manifest) as f:
        for line in f:
            d = json.loads(line)
            paths.append(d["audio_filepath"])
            durs.append(d.get("duration", 0.0))
    if args.limit:
        paths, durs = paths[: args.limit], durs[: args.limit]

    print(f"{args.label or args.nemo}")
    kind = (
        f"CHAT joint, chunk={chunk}" if chunked
        else f"frame-level joint on a {style} encoder, quantised to its {enc_chunk}-frame grid"
    )
    print(f"  {len(paths)} utts, {sum(durs)/3600:.2f} h audio, {kind} ({chunk*FRAME_SECONDS:.2f}s)")

    for t in [int(x) for x in args.trims.split(",")]:
        if hasattr(model.joint, "frame_trim"):
            model.joint.frame_trim = t
        with torch.inference_mode():
            hyps = model.transcribe(paths, batch_size=args.batch_size, verbose=False, return_hypotheses=True)
        if isinstance(hyps, tuple):
            hyps = hyps[0]

        times, n_words, n_utts_used = [], 0, 0
        for h in hyps:
            if isinstance(h, list):
                h = h[0]
            # y_sequence may be a tensor; `or []` on a tensor raises, so convert
            # before testing emptiness.
            seq = getattr(h, "y_sequence", None)
            stamp = getattr(h, "timestamp", None)
            ids = seq.tolist() if hasattr(seq, "tolist") else list(seq or [])
            stamps = stamp.tolist() if hasattr(stamp, "tolist") else list(stamp or [])
            if not ids or len(ids) != len(stamps):
                continue
            pieces = model.tokenizer.ids_to_tokens([int(i) for i in ids])
            # A word ends at the token BEFORE the next word-start; take that
            # token's chunk. Position inside the chunk is not informative.
            last_of_word = []
            for i, piece in enumerate(pieces):
                is_last = (i == len(pieces) - 1) or str(pieces[i + 1]).startswith(("▁", "Ġ"))
                if is_last:
                    last_of_word.append(int(stamps[i]))
            if not last_of_word:
                continue
            n_utts_used += 1
            n_words += len(last_of_word)
            credit = t if args.credit_trim else 0
            # `c` is a chunk index for a CHAT joint and a FRAME index otherwise;
            # quantise the latter up to its encoder chunk boundary.
            times.extend(
                (((c + 1) if chunked else (c // chunk + 1)) * chunk - credit) * FRAME_SECONDS
                for c in last_of_word
            )

        if not times:
            print(f"    trim={t}: no timestamped output")
            continue
        mean = sum(times) / len(times)
        print(
            f"    trim={t}:  mean word emission {mean:7.3f}s   "
            f"({n_words} words over {n_utts_used} utts"
            f"{', trim credited' if args.credit_trim else ''})"
        )


if __name__ == "__main__":
    sys.exit(main())
