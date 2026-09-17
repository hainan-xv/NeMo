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
"""How often is a frozen CHAT model WRONG, per utterance and per chunk?

Sizes the training corpus for a verifier/corrector before any of it is built.
The corrector learns "ACCEPT" or "correct this chunk to Z", so the numbers that
decide whether the idea is viable are:

    utterance exact-match rate  -- utterances yielding ONLY accept labels
    chunk wrong rate            -- the fraction of chunks carrying real signal

THE POINT OF RUNNING IT ON TRAINING DATA. CHAT has seen the training set for
dozens of epochs, so it is far more accurate there than at test time. A corpus
built from its training-set hypotheses would be overwhelmingly ACCEPT and its
errors would not resemble the ones a deployed corrector meets -- it would learn
both the wrong prior and the wrong error distribution. Comparing this probe's
train and held-out numbers measures that gap directly, and the gap decides
whether hypotheses must come from a held-out-fold or deliberately weakened
decode instead.

    python scripts/chat_chunk_error_probe.py --nemo <model.nemo> --source train --batches 40
    python scripts/chat_chunk_error_probe.py --nemo <model.nemo> --manifest <val.json>
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nemo.collections.asr.parts.utils.chunk_error_labels import label_chunks  # noqa: E402


def _ref_chunks_for(model, cut, n_chunks):
    """Reference words grouped by chunk, using the model's OWN assignment rule.

    Reuses ``assign_words_to_chunks`` rather than re-deriving it: the probe must
    partition the reference exactly the way training did, or the chunk counts it
    reports describe a different segmentation from the one a corrector would see.
    """
    from nemo.collections.asr.parts.utils.chat_alignment import assign_words_to_chunks

    aligned = (cut.custom or {}).get("alignments", []) or []
    words = [w["text"] for w in aligned]
    groups = assign_words_to_chunks(
        [w["end_time"] for w in aligned],
        n_chunks,
        model.joint.chunk_size,
        model.frame_length_in_secs,
        model.num_delay_frames,
    )
    return [[words[i] for i in g] for g in groups], words


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--source", choices=["train", "manifest"], default="train")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--dump", default="", help="write per-utterance records here (jsonl)")
    args = ap.parse_args()

    dev = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    from nemo.collections.asr.models.chat_bpe_models import EncDecCHATBPEModel

    print(f"==> restoring {args.nemo}", flush=True)
    model = EncDecCHATBPEModel.restore_from(restore_path=args.nemo, map_location=dev).eval().to(dev)

    cfg = model.cfg.train_ds if args.source == "train" else model.cfg.validation_ds
    from omegaconf import open_dict

    with open_dict(cfg):
        cfg.batch_size = args.batch_size
        cfg.num_workers = 2
        cfg.shuffle = False
        if args.source == "manifest" and args.manifest:
            cfg.manifest_filepath = args.manifest
            cfg.input_cfg = None
        # Bucketing reorders and rebatches; for a probe we want plain batches so
        # the utterance count is exactly batches * batch_size.
        if "use_bucketing" in cfg:
            cfg.use_bucketing = False
    # Build the loader through _setup_dataloader_from_config directly with
    # _want_cuts forced on. setup_training_data enables cuts only for the
    # forced/banded losses and setup_validation_data never does -- but this probe
    # needs the cuts on BOTH sources, because the reference word TIMINGS are what
    # the reference chunk partition is derived from. Without them the held-out
    # half of the comparison cannot be computed at all.
    model._want_cuts = True
    try:
        loader = model._setup_dataloader_from_config(cfg)
    finally:
        model._want_cuts = False

    n_utt = n_exact = n_chunk = n_bad = 0
    dump = open(args.dump, "w") if args.dump else None

    with torch.inference_mode():
        for bi, batch in enumerate(loader):
            if bi >= args.batches:
                break
            sig, sig_len = batch[0].to(dev), batch[1].to(dev)
            # LhotseSpeechToTextBpeDataset(return_cuts=True) appends cuts as a
            # 5th element. Length is the reliable test: a Tensor is iterable, so
            # checking __iter__ on batch[-1] silently accepted token_lens.
            cuts = batch[4] if len(batch) >= 5 else None
            if cuts is None:
                raise RuntimeError(
                    "batch carries no cuts; the probe needs word timings to derive " "the reference chunk partition"
                )

            proc, proc_len = model.preprocessor(input_signal=sig, length=sig_len)
            enc, enc_len = model.encoder(audio_signal=proc, length=proc_len)
            hyps = model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=enc, encoded_lengths=enc_len, return_hypotheses=False
            )
            if isinstance(hyps, tuple):
                hyps = hyps[0]
            texts = [h.text if hasattr(h, "text") else str(h) for h in hyps]

            cs = model.joint.chunk_size
            for u, text in enumerate(texts):
                n_ch = int((enc_len[u].item() + cs - 1) // cs)
                if cuts is None:
                    continue
                ref_chunks, ref_words = _ref_chunks_for(model, cuts[u], n_ch)
                if not ref_words:
                    continue
                labels, bad = label_chunks(text.split(), ref_chunks)
                n_utt += 1
                n_chunk += len(ref_chunks)
                n_bad += bad
                if text.split() == ref_words:
                    n_exact += 1
                if dump:
                    dump.write(
                        json.dumps({"hyp": text, "ref": " ".join(ref_words), "bad": bad, "n_chunks": len(ref_chunks)})
                        + "\n"
                    )
            if (bi + 1) % 5 == 0:
                print(f"   {bi+1}/{args.batches} batches, {n_utt} utts", flush=True)

    if dump:
        dump.close()
    if not n_utt:
        print("no utterances scored -- does the batch carry cuts?", file=sys.stderr)
        return 1

    print(f"\n=== {args.source} ===")
    print(f"  utterances            {n_utt}")
    print(f"  exact-match (all ACCEPT) {100.0*n_exact/n_utt:6.2f}%")
    print(f"  chunks                {n_chunk}")
    print(f"  chunks WRONG          {100.0*n_bad/max(1,n_chunk):6.2f}%   <- the corrector's training signal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
