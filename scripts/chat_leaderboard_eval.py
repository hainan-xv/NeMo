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
"""Open-ASR-Leaderboard driver for the CHAT transducer (ChatSTTModel).

Deliberately a sibling of ``speechlm_leaderboard_eval.py``: the dataset list,
the manifest reader, the shard partition, the audio batching and the scoring all
come from ``leaderboard_common``, so a CHAT number is comparable to a SCRIPT or
nemotron number BY CONSTRUCTION rather than by inspection. A given utterance
lands in the same shard, is padded the same way, and is scored by the same
normaliser and edit distance for every system.

Only two things differ from the SpeechLM driver, and they are the two things
that genuinely differ about the model:

  * it builds a ChatSTTModel from the checkpoint's own hyper_parameters, and
  * it decodes with the chunk-synchronous greedy transducer path
    (``transcribe_ids``) instead of ``generate()``, with the extra decode-time
    knob ``--retract``.

Run through the shared launcher, which handles averaging, GPU fan-out and
aggregation:

    EVAL_DRIVER=chat_leaderboard_eval.py \
    MODEL_CLASS=nemo.collections.speechlm2.models.chat_model.ChatSTTModel \
        ./oci_launch.sh launch/eval_leaderboard.sh <exp_name>

or via the thin wrapper launch/eval_chat.sh, which sets those for you.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from leaderboard_common import (  # noqa: E402
    DEFAULT_DATASETS,
    _log,
    aggregate_results,
    build_global_items,
    load_audio_batch,
    select_shard,
)

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional; the shard logs are still readable without it

    def tqdm(x, **kw):
        return x


def load_model(args, device):
    """Build ChatSTTModel from the checkpoint's own config.

    Not from a YAML on disk: the recipes changed repeatedly during development
    (vocabulary, joint window, emission delay, history recovery), and pairing a
    checkpoint with a drifted config builds a different model -- at best a shape
    error, at worst a quiet mismatch that still produces numbers.
    """
    from nemo.collections.speechlm2.parts.chat_eval import build_chat_tokenizer, load_chat_model

    retract = args.retract if args.retract is not None and args.retract >= 0 else None
    model, cfg = load_chat_model(
        args.ckpt_path, device=device, retract=retract, asr=args.pretrained_asr, llm=args.pretrained_llm
    )
    model.tokenizer = build_chat_tokenizer(cfg)
    _log(
        f"    vocab={cfg.get('vocab_size')} chunk_size={cfg.get('chunk_size')} "
        f"joint_history_chunks={cfg.get('joint_history_chunks', 0)} retract_words={cfg.get('retract_words', 0)}"
    )
    return model


def _decode(model, audios, audio_lens):
    """Greedy chunk-synchronous decode -> text, one string per utterance."""
    with torch.no_grad():
        ids = model.transcribe_ids(audios, audio_lens)
    return [model.tokenizer.ids_to_text(list(seq)) if seq else "" for seq in ids]


def evaluate_shard(model, args, device) -> None:
    """Byte-for-byte the SpeechLM driver's loop, with generate() swapped out.

    Keeping the structure identical is the point: same pooled item list, same
    seeded length-balanced partition, same per-batch failure containment, same
    output record format -- so ``aggregate_results`` scores CHAT exactly as it
    scores every other system.
    """
    items = build_global_items(args)
    shard = select_shard(items, args.num_shards, args.shard_index, args.shuffle_seed)
    suffix = ""
    if args.subshard_count > 1:
        shard = shard[args.subshard_index :: args.subshard_count]
        suffix = f"_sub{args.subshard_index}of{args.subshard_count}"
    _log(f"==> shard {args.shard_index}/{args.num_shards}{suffix}: {len(shard)} of {len(items)} pooled utts")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"shard{args.shard_index}_of{args.num_shards}{suffix}.generations.jsonl")

    done, total = defaultdict(int), defaultdict(int)
    for it in shard:
        total[it["key"]] += 1

    t0 = time.time()
    with open(out_path, "w") as fout:
        bar = tqdm(
            range(0, len(shard), args.batch_size), mininterval=args.progress_interval, desc=f"shard{args.shard_index}"
        )
        for i in bar:
            batch = shard[i : i + args.batch_size]
            try:
                audios, audio_lens = load_audio_batch([b["path"] for b in batch], args.pad_extra_seconds)
                hyps = _decode(model, audios.to(device), audio_lens.to(device))
            except Exception as e:  # a bad batch must not kill the whole shard
                _log(f"    [WARN] batch at {i} failed ({type(e).__name__}: {e}); emitting empty hypotheses")
                hyps = [""] * len(batch)
            if len(hyps) != len(batch):
                _log(f"    [WARN] got {len(hyps)} hypotheses for {len(batch)} utts; padding")
                hyps = (list(hyps) + [""] * len(batch))[: len(batch)]
            for b, hyp in zip(batch, hyps):
                fout.write(json.dumps({"key": b["key"], "reference": b["ref"], "hypothesis": hyp}) + "\n")
                done[b["key"]] += 1
            fout.flush()
            if hasattr(bar, "set_postfix_str"):
                bar.set_postfix_str(" ".join(f"{k.split('/')[0]}:{done[k]}/{total[k]}" for k in sorted(total)))

    _log(f"==> shard {args.shard_index} wrote {out_path} in {time.time() - t0:.1f}s")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aggregate", action="store_true", help="merge shard files and print the WER table")
    p.add_argument("--ckpt_path")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--cache_dir", default="")
    p.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS), help="comma-separated name:split")
    p.add_argument("--max_eval_samples", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--subshard_count", type=int, default=1)
    p.add_argument("--subshard_index", type=int, default=0)
    p.add_argument("--shuffle_seed", type=int, default=1234)
    p.add_argument("--pad_extra_seconds", type=float, default=0.5)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--progress_interval", type=float, default=30.0)
    # aggregate_results() reads this; without it --aggregate raises AttributeError.
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--retract", type=int, default=None, help="retract-by-k decoding; default = checkpoint's setting")
    p.add_argument("--pretrained_asr", default=None)
    p.add_argument("--pretrained_llm", default=None)
    # Accepted and ignored so the shared launcher can pass one flag set to every
    # driver. Silently swallowing them beats a crash on an inapplicable knob,
    # but each is echoed once so a knob that does nothing is visible.
    for ignored in ("--model_class", "--system_prompt", "--chunk_size", "--max_new_tokens", "--dtype"):
        p.add_argument(ignored, default=None)
    args, unknown = p.parse_known_args()
    if unknown:
        _log(f"    [note] ignoring flags that do not apply to CHAT: {' '.join(unknown)}")
    for name in ("system_prompt", "max_new_tokens"):
        if getattr(args, name, None) not in (None, ""):
            _log(f"    [note] --{name} has no effect for CHAT (no LLM prompt, no token budget)")
    return args


def main() -> int:
    args = parse_args()
    if args.aggregate:
        return aggregate_results(args)
    if not args.ckpt_path:
        _log("ERROR: --ckpt_path is required unless --aggregate")
        return 1
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    evaluate_shard(model, args, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
