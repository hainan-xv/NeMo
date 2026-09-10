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
import json

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nemo", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--trims", default="0,1,2")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()

    from nemo.collections.asr.models import ASRModel
    from nemo.collections.asr.metrics.wer import word_error_rate
    from nemo.utils import model_utils

    cfg = ASRModel.restore_from(restore_path=args.nemo, return_config=True)
    cls = model_utils.import_class_by_path(cfg.target)
    model = cls.restore_from(restore_path=args.nemo, map_location="cuda").eval()
    print(f"class={cls.__name__}  history_chunks={getattr(model.joint, 'history_chunks', '?')}")

    paths, refs = [], []
    with open(args.manifest) as f:
        for line in f:
            d = json.loads(line)
            paths.append(d["audio_filepath"])
            refs.append(d["text"])
    if args.limit:
        paths, refs = paths[: args.limit], refs[: args.limit]
    print(f"{len(paths)} utterances from {args.manifest}\n")

    for t in [int(x) for x in args.trims.split(",")]:
        if hasattr(model.joint, "frame_trim"):
            model.joint.frame_trim = t
        with torch.inference_mode():
            hyps = model.transcribe(paths, batch_size=args.batch_size, verbose=False)
        if isinstance(hyps, tuple):
            hyps = hyps[0]
        texts = [h if isinstance(h, str) else (getattr(h, "text", "") or "") for h in hyps]
        wer = word_error_rate(hypotheses=texts, references=refs)
        empty = sum(1 for x in texts if not x.strip())
        print(f"  frame_trim={t}:  val_wer={wer:.4f}   empty_hyps={empty}")


if __name__ == "__main__":
    main()
