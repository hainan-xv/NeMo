"""Score an averaged .nemo on the training-time validation manifest.

WHY THIS EXISTS. val_wer as logged during training and the leaderboard numbers
disagreed for two models that should be close, and the two are produced by
completely different code paths: val_wer comes from the Lightning validation
loop (with whatever frame_trim the model happened to be left in), while the
leaderboard number comes from transcribe() with frame_trim set explicitly. This
scores the SAME manifest through the SAME path as the leaderboard, at whichever
trims you ask for, so the two can finally be compared like for like.

    python scripts/chat_val_probe.py --nemo <model.nemo> --manifest <val.json> --trims 0,1,2
"""

import argparse
import copy
import json
import os
import sys

import torch


def _transcribe_padded(model, paths, batch_size, pad_seconds):
    """Transcribe with trailing silence appended, matching the leaderboard driver.

    It pads to keep a chunked model from having to commit at the exact end of
    speech; scoring without it measures a different thing, which is why a local
    number and a leaderboard number for the SAME model do not match.
    """
    import numpy as np
    import soundfile as sf
    import torch as _torch

    sr = 16000
    out = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i : i + batch_size]
        sigs = []
        for p_ in batch:
            a, file_sr = sf.read(p_, dtype="float32")
            assert file_sr == sr, f"expected {sr} Hz, got {file_sr} in {p_}"
            sigs.append(np.concatenate([a, np.zeros(int(round(pad_seconds * sr)), dtype="float32")]))
        lens = _torch.tensor([len(x) for x in sigs])
        mx = int(lens.max())
        padded = _torch.zeros(len(sigs), mx)
        for j, x in enumerate(sigs):
            padded[j, : len(x)] = _torch.from_numpy(x)
        dev = next(model.parameters()).device
        hyp = model.transcribe(padded.to(dev), batch_size=len(batch), verbose=False)
        if isinstance(hyp, tuple):
            hyp = hyp[0]
        out.extend(hyp)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nemo", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--trims", default="0,1,2")
    p.add_argument("--flush", type=int, default=0, help="extra all-zero chunks appended at decode")
    # --- decoding configuration -------------------------------------------
    p.add_argument(
        "--strategy",
        default=None,
        choices=["greedy", "greedy_batch", "beam", "tsd", "alsd", "maes"],
        help="RNN-T decoding strategy; default leaves the model's own config alone",
    )
    p.add_argument("--max_symbols", type=int, default=None, help="max symbols emitted per time step")
    p.add_argument(
        "--cuda_graphs",
        default=None,
        choices=["on", "off"],
        help="CUDA-graph decoder (greedy_batch only). CHAT falls back to the PyTorch path anyway "
        "when the joint trims frames, so this mainly matters for the plain model",
    )
    p.add_argument(
        "--loop_labels",
        default=None,
        choices=["on", "off"],
        help="label-looping (default) vs frame-looping greedy_batch implementation",
    )
    p.add_argument("--beam_size", type=int, default=None, help="beam width for the beam-family strategies")
    # --- encoder look-ahead -----------------------------------------------
    p.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="encoder frames per chunk; pins att_context_size to [left, chunk_size-1]. "
        "Needed to compare a MULTI-CONTEXT cache-aware model (nemotron) against a model pinned to "
        "one context, since otherwise each runs at a different look-ahead",
    )
    p.add_argument("--left_context", type=int, default=None, help="left context; default keeps the model's own")
    p.add_argument(
        "--pad_extra_seconds",
        type=float,
        default=0.0,
        help="append this much real silence to every clip, as the leaderboard driver does "
        "(--pad_extra_seconds 0.5 there). Without it a local number is NOT comparable to a "
        "leaderboard number for the same model",
    )
    p.add_argument(
        "--dump",
        default=None,
        help="write per-utterance records here (JSONL). One file per trim value: <dump>.trim<N>.jsonl",
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--scorer",
        default="leaderboard",
        choices=["leaderboard", "whisper"],
        help="leaderboard = the vendored normaliser + kaldialign merge_compounds, exactly what "
        "produces the numbers in the leaderboard tables. whisper = stock whisper_normalizer + plain "
        "word_error_rate, which charges compound splits ('school boys' vs 'schoolboys') that the "
        "leaderboard forgives -- so it can rank two models differently",
    )
    p.add_argument(
        "--normalize",
        action="store_true",
        help="apply Whisper's EnglishTextNormalizer to both sides, as the leaderboard scorer does; "
        "required for the staged leaderboard manifests, whose references are lowercased and unpunctuated",
    )
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()

    from nemo.collections.asr.models import ASRModel
    from nemo.collections.asr.metrics.wer import word_error_rate
    from nemo.utils import model_utils

    cfg = ASRModel.restore_from(restore_path=args.nemo, return_config=True)
    cls = model_utils.import_class_by_path(cfg.target)
    model = cls.restore_from(restore_path=args.nemo, map_location="cuda").eval()
    print(f"class={cls.__name__}  history_chunks={getattr(model.joint, 'history_chunks', '?')}")

    # Decoding config, changed only where asked. change_decoding_strategy rebuilds
    # the decoder object, so anything left as None keeps whatever the .nemo was
    # saved with rather than silently adopting this script's defaults.
    if any(
        v is not None
        for v in (args.strategy, args.max_symbols, args.cuda_graphs, args.loop_labels, args.beam_size)
    ):
        from omegaconf import OmegaConf, open_dict

        dec = copy.deepcopy(model.cfg.decoding)
        with open_dict(dec):
            if args.strategy is not None:
                dec.strategy = args.strategy
            grp = "beam" if (args.strategy or dec.strategy) in ("beam", "tsd", "alsd", "maes") else "greedy"
            if grp not in dec:
                dec[grp] = OmegaConf.create({})
            if args.max_symbols is not None:
                dec.greedy.max_symbols = args.max_symbols
            if args.cuda_graphs is not None:
                dec.greedy.use_cuda_graph_decoder = args.cuda_graphs == "on"
            if args.loop_labels is not None:
                dec.greedy.loop_labels = args.loop_labels == "on"
            if args.beam_size is not None:
                dec.beam.beam_size = args.beam_size
        model.change_decoding_strategy(dec)
        g = dec.get("greedy", {})
        print(
            f"decoding: strategy={dec.strategy} max_symbols={g.get('max_symbols', '-')} "
            f"cuda_graphs={g.get('use_cuda_graph_decoder', '-')} loop_labels={g.get('loop_labels', '-')}"
        )
    else:
        print(f"decoding: strategy={model.cfg.decoding.strategy} (unchanged from the .nemo)")

    # Pin the encoder look-ahead, using the leaderboard driver's own resolver so
    # the two cannot drift. It validates against att_context_size_all and raises
    # on an untrained look-ahead rather than warning, which would otherwise
    # produce a plausible but degraded number.
    if args.chunk_size is not None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from nemotron_leaderboard_eval import resolve_att_context

        att = resolve_att_context(model, args.chunk_size, args.left_context)
        model.encoder.set_default_att_context_size(att)
        print(f"encoder: chunk_size={args.chunk_size} -> att_context_size={att}")
    else:
        cur = getattr(model.encoder, "att_context_size", None)
        print(f"encoder: att_context_size={cur} (unchanged from the .nemo)")

    paths, refs, durs = [], [], []
    with open(args.manifest) as f:
        for line in f:
            d = json.loads(line)
            paths.append(d["audio_filepath"])
            # training manifests use "text"; the staged leaderboard cache uses "reference"
            refs.append(d.get("text", d.get("reference", "")))
            durs.append(d.get("duration", 0.0))
    if args.limit:
        paths, refs, durs = paths[: args.limit], refs[: args.limit], durs[: args.limit]
    print(f"{len(paths)} utterances from {args.manifest}\n")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    norm = (lambda x: x)
    lb_scorer = None
    if args.normalize:
        if args.scorer == "leaderboard":
            from leaderboard_wer import WER as LeaderboardWER

            lb_scorer = LeaderboardWER(normalize=True, verbose=False)
            norm = lb_scorer.normalizer
            print("scoring with the VENDORED leaderboard normaliser + merge_compounds\n")
        else:
            from whisper_normalizer.english import EnglishTextNormalizer

            norm = EnglishTextNormalizer()
            print("scoring with stock whisper_normalizer + plain word_error_rate\n")

    for t in [int(x) for x in args.trims.split(",")]:
        if hasattr(model.joint, "frame_trim"):
            model.joint.frame_trim = t
        if hasattr(model.joint, "decode_flush_chunks"):
            model.joint.decode_flush_chunks = args.flush
        with torch.inference_mode():
            if args.pad_extra_seconds > 0:
                hyps = _transcribe_padded(model, paths, args.batch_size, args.pad_extra_seconds)
            else:
                hyps = model.transcribe(paths, batch_size=args.batch_size, verbose=False)
        if isinstance(hyps, tuple):
            hyps = hyps[0]
        texts = [h if isinstance(h, str) else (getattr(h, "text", "") or "") for h in hyps]
        n_hyp, n_ref = [norm(x) for x in texts], [norm(r) for r in refs]
        if lb_scorer is not None:
            # A fresh scorer per trim: this implementation has no reset(), and
            # reusing one would accumulate the previous trim's utterances.
            fresh = type(lb_scorer)(normalize=True, verbose=False)
            fresh.update("probe", refs=refs, hyps=texts)
            wer = float(fresh.compute()["wer"])
        else:
            wer = word_error_rate(hypotheses=n_hyp, references=n_ref)
        empty = sum(1 for x in texts if not x.strip())
        print(f"  frame_trim={t} flush={args.flush}:  val_wer={wer:.4f}   empty_hyps={empty}")

        if args.dump:
            from kaldialign import edit_distance

            out = f"{args.dump}.trim{t}.jsonl"
            with open(out, "w") as fh:
                for path, raw_ref, raw_hyp, nr, nh, dur in zip(paths, refs, texts, n_ref, n_hyp, durs):
                    r_words, h_words = nr.split(), nh.split()
                    d = edit_distance(r_words, h_words)
                    fh.write(
                        json.dumps(
                            {
                                "audio": path,
                                "duration": dur,
                                "errors": d["total"],
                                "words": len(r_words),
                                "wer": d["total"] / max(len(r_words), 1),
                                "ins": d["ins"],
                                "del": d["del"],
                                "sub": d["sub"],
                                "ref": nr,
                                "hyp": nh,
                                "ref_raw": raw_ref,
                                "hyp_raw": raw_hyp,
                            }
                        )
                        + "\n"
                    )
            print(f"      wrote {out}")


if __name__ == "__main__":
    main()
