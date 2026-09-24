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
"""Make the INTERLEAVED streaming SpeechLM look like an ASRModel to the harness.

Companion to script_asr_shim.py, for the other model layout: one causal
audio-text-audio-text stream (``StreamingSTTModel``) rather than SCRIPT's spine
plus branches. Same base LLM and encoder as our arms, so a row produced here
isolates the LAYOUT rather than the ingredients -- which is exactly why it is
worth carrying onto the official board alongside our own models.

Like the SCRIPT shim, this adds NO decoding logic: it reuses the loader and
generate path from scripts/speechlm_leaderboard_eval.py verbatim.

FOUR DEFAULTS HERE ARE LOAD-BEARING, and each was measured -- getting any of
them wrong understates the model rather than failing loudly:

  * FSM (state-machine) decode, not bulk-prefill chunked decode. This is the
    path the model was designed for and the one its paper's numbers use:
    6.03 macro / 10.16 AMI against 6.63 / 10.80 for the chunked path.
  * STREAMING embeddings, not offline. The offline path scored 17.79 macro
    against streaming's 5.51 on this checkpoint -- it drops words at chunk
    starts. offline_embs is a diagnostic, never a reporting mode.
  * ITS OWN system prompt ("Transcribe the audio into text."), NOT the SCRIPT
    prompt. Ours would put it out of distribution.
  * pretrained_llm / pretrained_asr pointed at local snapshots. The checkpoint
    stores bare hub ids that cannot resolve offline on the grid; pointing them
    at the same snapshots our arms load also guarantees a byte-identical base
    LLM and encoder across every row.
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The interleaved model's OWN training instruction. See the module docstring.
DEFAULT_SYSTEM_PROMPT = "Transcribe the audio into text."
DEFAULT_MODEL_CLASS = "nemo.collections.speechlm2.models.streaming_stt_model.StreamingSTTModel"


@dataclass
class _Hyp:
    """Minimal stand-in for NeMo's Hypothesis: run_eval.py reads only ``.text``."""

    text: str


class _Greedy:
    strategy = "greedy_batch"


class _DecodingCfg:
    """run_eval.py reads cfg.decoding.strategy and may call change_decoding_strategy.

    Reporting "greedy_batch" makes that block a no-op rather than a crash: this
    model has one decoding path and nothing to switch to.
    """

    strategy = "greedy_batch"
    greedy = _Greedy()


class _Cfg:
    decoding = _DecodingCfg()


class SpeechLMASRShim:
    """Adapter exposing the ASRModel surface run_eval.py depends on."""

    def __init__(self, model, device, gen_kwargs: dict, pad_extra_seconds: float = 0.5, min_batch_size: int = 1):
        self._m = model
        self._device = device
        self._gen = gen_kwargs
        # Trained with trailing silence, and its emission lags the audio, so
        # decoding without the pad drops the final words.
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

        run_eval.py zips the returned list against its reference list
        positionally, so any reordering silently pairs each hypothesis with the
        wrong reference and yields a plausible-looking but meaningless WER.
        """
        from leaderboard_common import load_audio_batch
        from speechlm_leaderboard_eval import _generate

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
            if isinstance(texts, tuple):  # alignment-returning paths yield (texts, aligns)
                texts = texts[0]
            out.extend(_Hyp(t if isinstance(t, str) else getattr(t, "text", str(t))) for t in texts)
        return out


def load_speechlm_shim(
    ckpt_path: str,
    device,
    dtype,
    model_class_path: Optional[str] = None,
    system_prompt: Optional[str] = None,
    chunk_size: int = 14,
    max_new_tokens: int = 64,
    pad_extra_seconds: float = 0.5,
    pretrained_llm: Optional[str] = None,
    pretrained_asr: Optional[str] = None,
    state_machine: bool = True,
    offline_embs: bool = False,
    emit_delay_frames: int = 0,
) -> SpeechLMASRShim:
    """Restore the interleaved checkpoint and wrap it."""
    from speechlm_leaderboard_eval import load_model
    from transformers import GenerationConfig

    model = load_model(
        ckpt_path,
        model_class_path or DEFAULT_MODEL_CLASS,
        device,
        dtype,
        pretrained_llm,
        pretrained_asr,
    )
    # Key-for-key the same dict speechlm_leaderboard_eval.build_gen_kwargs builds.
    # generation_config is NOT optional: omitting it lets the LLM fall back to its
    # own config, which may SAMPLE -- a non-deterministic WER that would still
    # look plausible. max_new_tokens is the driver's per-chunk decode cap.
    gen_kwargs = dict(
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        max_new_tokens=max_new_tokens,
        generation_config=GenerationConfig(do_sample=False),
        chunk_size_override=chunk_size if chunk_size and chunk_size > 0 else None,
        use_offline_embs=offline_embs,
        use_state_machine_inference=state_machine,
        emit_delay_frames=emit_delay_frames,
        # Both are expensive and unused for WER.
        return_chunk_ids=False,
        return_alignments=False,
        return_debug_logs=False,
    )
    return SpeechLMASRShim(model, device, gen_kwargs, pad_extra_seconds=pad_extra_seconds)
