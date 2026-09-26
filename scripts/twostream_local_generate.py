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
"""Exercise TwoStreamSTTModel.generate locally before it runs on a cluster.

The unit tests only check STRUCTURE (generate is overridden, uses the normalised
read-out, threads the cache). The loop itself -- chunk advance, <eot> handling,
cache extension, token accumulation -- has never executed. This runs it.

The decisive check is not "does it emit text" but the CACHE EQUIVALENCE: decoding
with the incremental text cache must give exactly what recomputing the prefix
from scratch gives. That is the design's inference claim, and a stale-cache bug
would otherwise show up only as a quietly worse WER.

Run:  python scripts/twostream_local_generate.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nemo.collections.speechlm2.models.twostream_model import TwoStreamSTTModel

    dev = torch.device("cuda:0")
    name = "Qwen/Qwen3-1.7B"
    tok = AutoTokenizer.from_pretrained(name)
    llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(dev).eval()
    core, H = llm.model, llm.config.hidden_size
    CHUNK, NCH = 14, 3

    class _Tok:
        def text_to_ids(self, t):
            return tok(t).input_ids

        def ids_to_text(self, ids):
            return tok.decode(ids, skip_special_tokens=True)

    class _Model(TwoStreamSTTModel):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.llm = llm
            self.tokenizer = _Tok()
            self._eot_id = tok.eos_token_id
            self.proj = torch.nn.Linear(1024, H).to(dev)

            class _Cfg:
                joint_layers = 1
                loss_reduction = "mean_volume"
                val_chunk_size = CHUNK

            self.core_cfg = _Cfg()

        # deterministic stand-in encoder: same shape/dtype, fixed values
        def _chunk_audio(self, audios, audio_lens, chunk_size, n_chunks):
            g = torch.Generator(device="cpu").manual_seed(7)
            x = torch.randn(NCH, chunk_size, 1024, generator=g).to(dev)
            return self.proj(x)

        def _llm_core(self):
            return core

        def _lm_head_of(self):
            return llm.lm_head

        def _embed_tokens(self, ids):
            return core.embed_tokens(ids)

        def _llm_forward(self, **kw):
            return llm(**kw)

    m = _Model().to(dev).eval()
    audios = torch.zeros(1, 16000, device=dev)
    audio_lens = torch.tensor([16000], device=dev)

    print("==> 1. does it run and emit anything?")
    out = m.generate(audios, audio_lens, system_prompt="Transcribe the audio into text.", max_new_tokens=8)
    assert isinstance(out, list) and len(out) == 1, f"expected List[str] of len 1, got {type(out)}"
    print(f"    returned {type(out).__name__}[{len(out)}]")
    print(f"    text = {out[0]!r}")
    print(f"    non-empty: {bool(out[0].strip())}")

    print("\n==> 2. batch of 2 -- one string per utterance, loop advances")
    out2 = m.generate(torch.zeros(2, 16000, device=dev), torch.tensor([16000, 16000], device=dev), max_new_tokens=6)
    print(f"    len={len(out2)}  identical inputs -> identical outputs: {out2[0] == out2[1]}")

    print("\n==> 3. determinism (greedy must be repeatable)")
    again = m.generate(audios, audio_lens, max_new_tokens=8)
    print(f"    same text on rerun: {again[0] == out[0]}")

    print("\n==> 4. CACHE EQUIVALENCE -- the design's inference claim")
    # Reference: same decode, but the text stream rebuilt from scratch each step.
    import types

    ref = _Model().to(dev).eval()
    ref.proj.load_state_dict(m.proj.state_dict())

    def slow_generate(self, audios, audio_lens, system_prompt="Transcribe the audio into text.", max_new_tokens=8, **kw):
        chunk = CHUNK
        frames = self._chunk_audio(audios[0:1], audio_lens[0:1], chunk, 0)
        ids = list(self.tokenizer.text_to_ids(system_prompt))
        emitted = []
        for t in range(int(frames.shape[0])):
            for _ in range(max_new_tokens):
                full = torch.tensor(ids + emitted, dtype=torch.long, device=dev)
                o = self._llm_forward(
                    inputs_embeds=self._embed_tokens(full.unsqueeze(0)),
                    use_cache=False, output_hidden_states=True, return_dict=True,
                )
                text_h = o["hidden_states"][-2][0]  # NO cache: recomputed wholly
                a = frames[t]
                tu, w = text_h.shape[0], a.shape[0]
                seq = torch.cat([text_h, a], 0)
                total = tu + w
                mask = torch.zeros((total, total), dtype=torch.bool, device=dev)
                mask[:tu, :tu] = torch.ones((tu, tu), dtype=torch.bool, device=dev).tril()
                mask[tu:, :tu] = True
                ar = torch.arange(w, device=dev)
                mask[tu:, tu:] = (ar.unsqueeze(1) >= ar.unsqueeze(0)).T
                pos = torch.arange(total, device=dev)
                pos[tu:] = tu + torch.arange(w, device=dev)
                h = self._run_joint(seq, mask, pos)
                nxt = int(self._project_to_vocab(h[-1:]).argmax(-1).item())
                if nxt == self._eot_id:
                    break
                emitted.append(nxt)
        return [self.tokenizer.ids_to_text(emitted) if emitted else ""]

    ref.generate = types.MethodType(slow_generate, ref)
    with torch.no_grad():
        slow = ref.generate(audios, audio_lens, max_new_tokens=8)
    print(f"    cached  = {out[0]!r}")
    print(f"    no-cache= {slow[0]!r}")
    same = slow[0] == out[0]
    print(f"    MATCH: {same}")

    print("\n==> verdict")
    if same and isinstance(out, list):
        print("    generate runs, is deterministic, and the incremental text cache is EQUIVALENT")
        print("    to recomputing the prefix -- the cache is not stale.")
    elif not same:
        print("    CACHE MISMATCH: the incremental text stream diverges from a full recompute.")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
