"""Decode a few validation utterances from a CHAT checkpoint and show the ids.

    python scripts/qwen_decode_probe.py --ckpt <path.ckpt> [--n 8]

WHY IDS AND NOT JUST TEXT. The Qwen-vocabulary banded arms drive their training
loss from 15.5 down to ~1.2 while val_wer never improves on its epoch-0 value.
Two very different faults produce that, and printing the hypothesis STRING alone
cannot tell them apart:

  * the model has learned the alignment but decoding emits almost nothing, so
    the WER is ~100% deletions -- a MODEL problem;
  * the model emits plenty of sensible ids and the HuggingFace detokenizer shim
    turns them into garbage or drops them -- a TOKENIZER problem.

So this prints, per utterance: how many tokens greedy decoding emitted, the
first ids, the pieces those ids map to, and the final text -- next to the
reference. Empty ids means the first fault; sensible ids with broken text means
the second.

It also round-trips the REFERENCE through the same tokenizer. If ref -> ids ->
text does not return the reference, the fault is in the tokenizer alone and no
amount of training would ever have shown a good val_wer.
"""

import argparse
import sys

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=8, help="utterances to decode")
    p.add_argument("--max_ids", type=int, default=25, help="ids to print per utterance")
    args = p.parse_args()

    from nemo.collections.asr.models import EncDecCHATBPEModel

    print(f"==> loading {args.ckpt}")
    model = EncDecCHATBPEModel.load_from_checkpoint(args.ckpt, map_location="cuda")
    model.eval()
    if hasattr(model.joint, "frame_trim"):
        model.joint.frame_trim = 0

    tok = model.tokenizer
    blank = model.joint.num_classes_with_blank - 1
    print(f"    vocab_size={tok.vocab_size}  blank={blank}  loss_type={getattr(model, 'loss_type', '?')}")

    # ---- 1. Does the tokenizer round-trip at all? -------------------------
    # This needs no audio and no training, so it isolates the detokenizer from
    # everything the model might be doing wrong.
    print("\n==> tokenizer round-trip (no model involved)")
    for probe in ["hello world", "the quick brown fox", "i think that's right"]:
        ids = tok.text_to_ids(probe)
        back = tok.ids_to_text(ids)
        ok = "OK " if back.strip().lower() == probe.strip().lower() else "MISMATCH"
        print(f"    [{ok}] {probe!r} -> {ids[:12]} -> {back!r}")

    # ---- 2. What does the model actually emit? ---------------------------
    print("\n==> decoding validation utterances")
    model.setup_multiple_validation_data(model.cfg.validation_ds)
    dl = model._validation_dl if not isinstance(model._validation_dl, list) else model._validation_dl[0]

    shown = 0
    with torch.inference_mode():
        for batch in dl:
            signal, signal_len, transcript, transcript_len = batch[0], batch[1], batch[2], batch[3]
            signal, signal_len = signal.cuda(), signal_len.cuda()
            encoded, encoded_len = model.forward(input_signal=signal, input_signal_length=signal_len)
            hyps = model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=True
            )
            if isinstance(hyps, tuple):
                hyps = hyps[0]

            for i, h in enumerate(hyps):
                if shown >= args.n:
                    break
                seq = getattr(h, "y_sequence", None)
                ids = seq.tolist() if hasattr(seq, "tolist") else list(seq or [])
                ref_ids = transcript[i][: transcript_len[i]].tolist()
                pieces = tok.ids_to_tokens([int(x) for x in ids[: args.max_ids]]) if ids else []
                print(f"\n  --- utt {shown}")
                print(f"    REF text : {tok.ids_to_text(ref_ids)!r}")
                print(f"    HYP text : {getattr(h, 'text', None)!r}")
                print(f"    emitted  : {len(ids)} tokens   (ref has {len(ref_ids)})")
                print(f"    HYP ids  : {ids[: args.max_ids]}")
                print(f"    HYP piece: {pieces}")
                shown += 1
            if shown >= args.n:
                break

    print("\n==> read it like this")
    print("    0 tokens emitted            -> model collapsed to blanks (a MODEL fault)")
    print("    sensible ids, broken text   -> detokenizer shim (a TOKENIZER fault)")
    print("    round-trip MISMATCH above   -> tokenizer alone; val_wer could never have been good")


if __name__ == "__main__":
    sys.exit(main())
