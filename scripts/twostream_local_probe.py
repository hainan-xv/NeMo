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
"""Local probe for the two-stream loss magnitude.

THE QUESTION. On DFW the two-stream loss starts near 674 per target token where
an untrained model should sit at log(V) ~ 11.9 -- about 56x too high. The <eot>
denominator gap explains only ~1.5x, so something else is wrong.

THE HYPOTHESIS. In packed SCRIPT the audio embeddings enter at layer 0 and are
transformed by all N decoder layers. In the two-stream model they are handed
straight to layer N alongside layer-(N-1) TEXT hidden states. Those are different
representation spaces -- different scale, different statistics -- so attention
between them is meaningless and the logits are garbage.

WHAT THIS MEASURES, in order:
  1. the scale of text_h (layer N-1 output) vs the audio embeddings
  2. whether the joint's output for audio queries looks like a layer-N output
  3. the resulting per-token NLL, against log(V) as the neutral reference
  4. whether a LayerNorm on the audio closes the gap

Run:  python scripts/twostream_local_probe.py [--layers 1] [--chunk 14]
"""

import argparse
import math
import os
import sys

# This repo, not whatever `nemo` happens to be importable. A second checkout on
# the path resolves to a package without the two-stream module, and the failure
# ("No module named ...twostream_model") reads like a missing file rather than
# the wrong tree.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def stats(name, t):
    t = t.float()
    print(
        f"    {name:24s} mean={t.mean().item():+9.4f}  std={t.std().item():8.4f}  "
        f"absmax={t.abs().max().item():9.3f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--layers", type=int, default=1, help="joint layers")
    ap.add_argument("--chunk", type=int, default=14, help="audio frames per chunk")
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device(args.device)
    print(f"==> loading {args.llm}")
    tok = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, torch_dtype=torch.float32).to(dev).eval()
    core = llm.model
    H = llm.config.hidden_size
    V = llm.config.vocab_size
    print(f"    hidden={H} vocab={V} layers={len(core.layers)}  log(V)={math.log(V):.2f}")

    # --- a realistic text stream -------------------------------------------
    prompt = "You are doing streaming speech recognition. Transcript so far:"
    text = " one two three four five six seven eight"
    ids = torch.tensor([tok(prompt + text).input_ids], device=dev)
    with torch.no_grad():
        out = llm(input_ids=ids, output_hidden_states=True, use_cache=False, return_dict=True)
    text_h = out.hidden_states[-(args.layers + 1)][0]  # (P, H) -- input to the joint layers
    print("\n==> 1. representation scales")
    stats("text_h (layer N-1)", text_h)

    # --- audio as the two-stream model currently supplies it ----------------
    # A perception module projects encoder output to H. Its statistics are those
    # of a projection output, NOT of a deep decoder activation. Simulated here
    # with the same shape and a unit-scale projection, which is the optimistic
    # case: a real encoder output is not guaranteed to be better behaved.
    torch.manual_seed(0)
    enc = torch.randn(args.chunks, args.chunk, 1024, device=dev)
    proj = torch.nn.Linear(1024, H, bias=False).to(dev)
    audio_emb = proj(enc)
    stats("audio_emb (as fed now)", audio_emb)

    ratio = (text_h.float().std() / audio_emb.float().std()).item()
    print(f"\n    std ratio text_h / audio = {ratio:.2f}x")

    # --- what the joint layer does with each --------------------------------
    print("\n==> 2. joint layer output, text queries vs audio queries")
    layer = core.layers[-1]
    rot = getattr(core, "rotary_emb", None)

    def run(seq):
        h = seq.unsqueeze(0)
        pos = torch.arange(h.shape[1], device=dev).unsqueeze(0)
        kw = {}
        if rot is not None:
            kw["position_embeddings"] = rot(h, pos)
        with torch.no_grad():
            r = layer(h, attention_mask=None, position_ids=pos, **kw)
        return (r[0] if isinstance(r, tuple) else r)[0]

    text_only = run(text_h)
    stats("joint(text only)", text_only)
    mixed = run(torch.cat([text_h, audio_emb[0]], dim=0))
    stats("joint(text+audio)[audio]", mixed[text_h.shape[0] :])

    # --- 3. CONTROL FIRST: is the read-out path itself correct? -------------
    #
    # HF runs layers -> model.norm(...) -> lm_head. Applying lm_head to a RAW
    # layer output skips the final RMSNorm, and these activations are huge
    # (absmax above 1e4), so the logits would be garbage regardless of audio.
    # Test that on TEXT queries, where the answer is known: a pretrained LLM
    # predicting its own next token should score a couple of nats, not hundreds.
    print("\n==> 3. CONTROL: text queries, with and without the final norm")
    head = llm.lm_head
    final_norm = core.norm

    def nll_at(h, target_ids, use_norm):
        with torch.no_grad():
            hh = final_norm(h) if use_norm else h
            lp = torch.log_softmax(head(hh).float(), dim=-1)
        return -lp[:-1].gather(1, target_ids[1:].unsqueeze(1)).mean().item()

    tgt_text = ids[0]
    ctrl_no = nll_at(text_only, tgt_text, use_norm=False)
    ctrl_yes = nll_at(text_only, tgt_text, use_norm=True)
    print(f"    text queries, NO  final norm  NLL/token = {ctrl_no:8.2f}")
    print(f"    text queries, WITH final norm NLL/token = {ctrl_yes:8.2f}   (log V = {math.log(V):.2f})")
    if ctrl_no > 10 * ctrl_yes:
        print("    -> the missing final norm ALONE explains a large factor.")

    print("\n==> 4. audio query positions")

    def nll_from(h_audio, use_norm=True):
        with torch.no_grad():
            hh = final_norm(h_audio) if use_norm else h_audio
            lp = torch.log_softmax(head(hh).float(), dim=-1)
        tgt = ids[0, -h_audio.shape[0] :]
        return -lp.gather(1, tgt.unsqueeze(1)).mean().item()

    raw = nll_from(mixed[text_h.shape[0] :], use_norm=False)
    raw_n = nll_from(mixed[text_h.shape[0] :], use_norm=True)
    print(f"    audio RAW, no norm       NLL/token = {raw:8.2f}")
    print(f"    audio RAW, WITH norm     NLL/token = {raw_n:8.2f}")

    # --- 4. does a LayerNorm close the gap? ---------------------------------
    ln = torch.nn.LayerNorm(H).to(dev)
    with torch.no_grad():
        ln.weight.copy_(torch.full((H,), text_h.float().std().item()))
    normed = ln(audio_emb[0])
    stats("audio_emb (LayerNorm'd)", normed)
    mixed2 = run(torch.cat([text_h, normed], dim=0))
    fixed = nll_from(mixed2[text_h.shape[0] :])
    print(f"    audio LayerNorm'd        NLL/token = {fixed:8.2f}")

    # --- 5. the model's OWN projection path, so the probe tests the fix ------
    print("\n==> 5. TwoStreamSTTModel._project_to_vocab (the shipped path)")
    from nemo.collections.speechlm2.models.twostream_model import TwoStreamSTTModel

    class _Shim(TwoStreamSTTModel):
        def __init__(self):
            torch.nn.Module.__init__(self)

        def _llm_core(self):
            return core

        def _lm_head_of(self):
            return head

    shim = _Shim()
    with torch.no_grad():
        lp = torch.log_softmax(shim._project_to_vocab(text_only).float(), dim=-1)
    model_path = -lp[:-1].gather(1, ids[0][1:].unsqueeze(1)).mean().item()
    print(f"    text queries via the model  NLL/token = {model_path:8.2f}")
    ok = abs(model_path - ctrl_yes) < 1e-3
    print(f"    matches the with-norm control: {ok}")

    print("\n==> verdict")
    print(f"    control text no-norm  = {ctrl_no:8.2f}   with-norm = {ctrl_yes:8.2f}")
    print(f"    audio   no-norm       = {raw:8.2f}   with-norm = {raw_n:8.2f}")
    if ctrl_no > 10 * ctrl_yes:
        print("    ROOT CAUSE: lm_head was applied WITHOUT model.norm. That is a bug in")
        print("    twostream_loss, independent of anything to do with audio.")
    if raw_n > 3 * math.log(V):
        print("    Audio queries remain high even normalised -- a second, separate issue")
        print("    (expected here: the probe feeds RANDOM audio, which carries no signal).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
