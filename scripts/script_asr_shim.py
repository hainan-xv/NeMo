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
"""Make a SCRIPT speechlm2 model look like an ASRModel to the leaderboard harness.

The official Open-ASR-Leaderboard runner does two things our SCRIPT arms cannot
satisfy directly: it restores through ``ASRModel`` and it calls
``model.transcribe(paths, batch_size=...)`` expecting objects with ``.text``.
``ScriptSTTModel`` is a speechlm2 model -- not an ASRModel subclass, and it
decodes through ``generate(audios=..., audio_lens=...)`` instead.

Without this shim the SCRIPT arms simply cannot be scored by the official
harness, which leaves them on a different normalizer and a different metric from
every other row in the comparison -- i.e. not comparable at all, which is the
whole point of moving to the official code.

The shim adds NO decoding logic of its own: it reuses the exact loader and
generate path from scripts/script_leaderboard_eval.py, so a number produced here
differs from our own harness only in the scoring and the dataset plumbing.
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_SYSTEM_PROMPT = (
    "You are doing streaming speech recognition. Given the transcript so far and "
    "the representation of the next audio chunk, output the words spoken in that chunk."
)


@dataclass
class _Hyp:
    """Minimal stand-in for NeMo's Hypothesis: run_eval.py reads only ``.text``."""

    text: str


class _Greedy:
    strategy = "greedy_batch"


class _DecodingCfg:
    """run_eval.py reads cfg.decoding.strategy and may call change_decoding_strategy.

    Reporting "greedy_batch" makes that block a no-op rather than a crash: SCRIPT
    has one decoding path and nothing to switch to.
    """

    strategy = "greedy_batch"
    greedy = _Greedy()


class _Cfg:
    decoding = _DecodingCfg()


class ScriptASRShim:
    """Adapter exposing the ASRModel surface run_eval.py depends on."""

    def __init__(self, model, device, gen_kwargs: dict, pad_extra_seconds: float = 0.5, min_batch_size: int = 1):
        self._m = model
        self._device = device
        self._gen = gen_kwargs
        # SCRIPT is TRAINED with data.dataset.pad_extra_duration; its emission lags
        # the audio, so decoding without trailing silence drops the final words.
        self._pad = pad_extra_seconds
        self._min_bs = min_batch_size
        self.cfg = _Cfg()

    # -- surface run_eval.py touches ------------------------------------
    def to(self, *a, **k):
        return self

    def eval(self):
        return self

    def parameters(self):
        return self._m.parameters()

    def change_decoding_strategy(self, *a, **k):
        return None

    @torch.no_grad()
    def transcribe(
        self, paths: List[str], batch_size: int = 8, verbose: bool = False, num_workers: int = 1, **kw
    ) -> List[_Hyp]:
        """Decode in batches, preserving INPUT ORDER.

        run_eval.py zips the returned list against its reference list positionally,
        so any reordering silently pairs each hypothesis with the wrong reference
        and produces a plausible-looking but meaningless WER.
        """
        from leaderboard_common import load_audio_batch
        from script_leaderboard_eval import _generate

        out: List[_Hyp] = []
        for i in range(0, len(paths), batch_size):
            chunk = paths[i : i + batch_size]
            audios, audio_lens = load_audio_batch(chunk, self._pad)
            texts = _generate(
                self._m,
                audios.to(self._device),
                audio_lens.to(self._device),
                self._gen,
                self._min_bs,
            )
            if isinstance(texts, tuple):  # --emit_chunk_ids returns (texts, chunks)
                texts = texts[0]
            out.extend(_Hyp(t if isinstance(t, str) else getattr(t, "text", str(t))) for t in texts)
        return out


def load_script_shim(
    ckpt_path: str,
    device,
    dtype,
    model_class_path: Optional[str] = None,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 64,
    chunk_size: int = 14,
    pad_extra_seconds: float = 0.5,
) -> ScriptASRShim:
    """Restore a SCRIPT checkpoint and wrap it."""
    from script_leaderboard_eval import load_model
    from transformers import GenerationConfig

    model_class_path = model_class_path or "nemo.collections.speechlm2.models.script_model.ScriptSTTModel"
    model = load_model(ckpt_path, model_class_path, device, dtype)
    gen_kwargs = dict(
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        max_new_tokens=max_new_tokens,
        generation_config=GenerationConfig(do_sample=False),
        chunk_size_override=chunk_size if chunk_size and chunk_size > 0 else None,
        force_word_start=False,
    )
    return ScriptASRShim(model, device, gen_kwargs, pad_extra_seconds=pad_extra_seconds)
