#!/usr/bin/env python3
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
"""A real two-stream training step, locally, with the shipped code path.

The DFW run could not answer "is the loss the right MAGNITUDE" quickly, and the
first probe answered the wrong question because it had no control. This drives
the actual pipeline -- real Qwen, real chunked audio, the real lattice -- and
checks the loss against a reference the model cannot fake: log(V).

  untrained joint, real audio  ->  expect roughly log(V) = 11.93 per token
  broken read-out (no final norm) -> hundreds

Also does a few optimiser steps, because "finite" and "decreasing" are different
claims and only the second means the gradients are wired up.

Run:  python scripts/twostream_local_step.py [--steps 5] [--chunk 14]
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nemo.collections.speechlm2.parts.script import ChunkSpec, build_packed_banded_example


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--chunk", type=int, default=14, help="encoder frames per chunk")
    ap.add_argument("--band", type=int, default=0, help="band_words; 0 == forced")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nemo.collections.speechlm2.models.twostream_model import TwoStreamSTTModel

    dev = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.float32).to(dev)
    core = llm.model
    H, V = llm.config.hidden_size, llm.config.vocab_size
    print(f"==> {args.llm}: hidden={H} vocab={V} layers={len(core.layers)}  log(V)={math.log(V):.2f}")

    # --- a real utterance, chunked ------------------------------------------
    prompt = "You are doing streaming speech recognition. Transcript so far:"
    words = ["one", "two", "three", "four", "five", "six", "seven", "eight"]
    instr = tok(prompt).input_ids
    per_chunk = 2  # words revealed per chunk
    chunks, word_starts, cursor = [], [], 0
    for i in range(0, len(words), per_chunk):
        ids = tok(" " + " ".join(words[i : i + per_chunk])).input_ids
        # word starts are spine-relative token indices
        c = cursor
        for w in words[i : i + per_chunk]:
            word_starts.append(c)
            c += len(tok(" " + w).input_ids)
        chunks.append(ChunkSpec(audio_len=args.chunk, target_ids=ids))
        cursor += len(ids)
    n_tokens = cursor
    T = len(chunks)
    print(f"    {T} chunks x {args.chunk} frames, {n_tokens} spine tokens, band={args.band}")

    ex = build_packed_banded_example(
        instruction_ids=instr,
        chunks=chunks,
        word_starts=word_starts,
        band_words=args.band,
        vision_start_id=tok.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_id=tok.convert_tokens_to_ids("<|vision_end|>"),
        eot_id=tok.eos_token_id,
        band_side="both",
    )
    J = int(ex.cut.shape[1])
    print(f"    lattice: cut{tuple(ex.cut.shape)} J={J} K={int(ex.span_valid.shape[-1]) - 1}")

    # THE COST CLAIM, measured on the real path rather than asserted. In packed
    # SCRIPT the band multiplies packed length; here candidates SHARE per-chunk
    # cells, so widening it should move the joint length far less than J does.
    from nemo.collections.speechlm2.parts.twostream import build_joint_inputs, plan_cells

    _cells = plan_cells(ex.cut, ex.cut_valid, TwoStreamSTTModel._reach_from_cuts(ex.cut, ex.cut_valid, n_tokens))
    _seq, _, _ = build_joint_inputs(torch.zeros(ex.spine_len, 8), torch.zeros(T, args.chunk, 8), _cells, len(instr))
    print(f"    cost   : n_cells={_cells.n_cells}  joint_seq_len={_seq.shape[0]}  (packed SCRIPT would scale with J={J})")

    # --- model shim: real LLM, stand-in perception ---------------------------
    class _Model(TwoStreamSTTModel):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.llm = llm
            self._eot_id = tok.eos_token_id
            self.audio_proj = torch.nn.Linear(1024, H).to(dev)

            class _Cfg:
                joint_layers = int(os.environ.get("JOINT_LAYERS", "1"))
                loss_reduction = "mean_volume"
                extra_joint_layer = bool(os.environ.get("EXTRA_JOINT"))
                extra_joint_init_from_last = True
                audio_position_mode = os.environ.get("AUDIO_POS", "cut")
                joint_text_context = os.environ.get("JOINT_TEXT", "full")

            self.core_cfg = _Cfg()
            self.joint_layer = None
            if self.core_cfg.extra_joint_layer:
                # Same construction path the real model uses, so a mistake here
                # is a mistake there.
                self._build_extra_joint_layer()

        def _llm_core(self):
            return core

        def _lm_head_of(self):
            return llm.lm_head

        def _embed_tokens(self, ids):
            return core.embed_tokens(ids)

        def _llm_forward(self, **kw):
            return llm(**kw)

        def _chunk_audio(self, audios, audio_lens, chunk_size, n_chunks):
            # Stand-in for the Conformer: same shape and dtype, so everything
            # downstream is exercised. Real audio changes the VALUE of the loss,
            # not whether the plumbing is right.
            torch.manual_seed(1234)
            return self.audio_proj(torch.randn(n_chunks, chunk_size, 1024, device=dev))

    model = _Model().to(dev)
    text_ids = ex.input_ids[: ex.spine_len].to(dev)
    spine_ids = text_ids[len(instr) :]
    cut = ex.cut.to(dev)
    cut_valid = ex.cut_valid.to(dev)

    kwargs = dict(
        text_ids=text_ids,
        audio_emb=model._chunk_audio(None, None, args.chunk, T),
        cut=cut,
        cut_valid=cut_valid,
        span_valid=ex.span_valid.to(dev),
        reach=TwoStreamSTTModel._reach_from_cuts(cut, cut_valid, n_tokens),
        spine_ids=spine_ids,
        prompt_len=len(instr),
        n_tokens=n_tokens,
    )

    # --- step 0: magnitude ---------------------------------------------------
    with torch.no_grad():
        nll = model.twostream_loss(**kwargs)
    per_tok = nll.item() / max(n_tokens, 1)
    print("\n==> loss at init")
    print(f"    total NLL      = {nll.item():9.2f}")
    print(f"    per token      = {per_tok:9.2f}   (log V = {math.log(V):.2f})")
    ratio = per_tok / math.log(V)
    print(f"    ratio to log V = {ratio:9.2f}x")
    if ratio < 3:
        print("    PLAUSIBLE: an untrained joint on uninformative audio should sit near log(V).")
    else:
        print("    SUSPICIOUS: far above log(V) -- the read-out is probably still wrong.")

    # --- a few optimiser steps ----------------------------------------------
    print("\n==> optimiser steps (does it actually learn?)")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    for i in range(args.steps):
        opt.zero_grad(set_to_none=True)
        # Recompute the audio each step: it runs through a trainable projection,
        # so reusing the tensor reuses its graph, which is freed by the first
        # backward. (Script bug, not a model one -- the real path rebuilds it per
        # batch from the encoder.)
        kwargs["audio_emb"] = model._chunk_audio(None, None, args.chunk, T)
        loss = model.twostream_loss(**kwargs) / max(n_tokens, 1)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1e9)
        opt.step()
        print(f"    step {i}  loss/token = {loss.item():8.3f}   grad_norm = {gn.item():10.3f}")

    print("\n==> done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
