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
"""Forced-alignment batches for the CHAT (chunk-wise attention) transducer.

WHY THIS EXISTS. A transducer's loss marginalises over every alignment, which
costs a [B, T, U, V] tensor. That is fine at V=1024 and impossible at V=151936 --
and whether a LARGE vocabulary helps a transducer the way it helps the SpeechLM
is exactly the question. Fixing the alignment removes the marginalisation: each
word is emitted at the chunk its last token falls in, so only the pairs on that
single path are ever scored (see ``RNNTAttJoint.joint_on_path``).

WHERE THE ALIGNMENT COMES FROM. Straight from :class:`ScriptSTTDataset`, by
subclassing it rather than reimplementing: the word-to-chunk assignment, the
emission delay and the tokenisation are then IDENTICAL to what the SpeechLM
trains on, so a CHAT-vs-SpeechLM comparison differs only in the model.

THE PATH. For an utterance whose chunks carry ``[[a, b], [], [c]]``::

    chunk 0 -> a, b, <blank>
    chunk 1 -> <blank>
    chunk 2 -> c, <blank>

so ``t_idx = [0,0,0, 1, 2,2]``, ``u_idx = [0,1,2, 2, 2,3]`` and
``labels = [a, b, blank, blank, c, blank]``. Every chunk ends with exactly one
blank -- including silent ones, which is how the model learns to emit nothing --
and ``u`` advances only on real tokens, because the prediction network is
conditioned on emitted labels alone.

Length is therefore ``sum(len(tokens)) + n_chunks``, versus the ``T x U`` lattice
the standard loss would score.
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from nemo.collections.speechlm2.data.script_dataset import ScriptSTTDataConfig, ScriptSTTDataset
from nemo.collections.speechlm2.parts.script_messages import get_llm_messages_for_batch


@dataclass
class ChatAlignedBatch:
    """One batch of forced-alignment transducer training data.

    Attributes:
        audios: (B, n_samples) waveforms.
        audio_lens: (B,) valid sample counts.
        b_idx, t_idx, u_idx: (N,) the path -- which utterance, which chunk, and
            how many labels have been emitted, for each scored step.
        labels: (N,) target at each step; ``blank_id`` terminates every chunk.
        pred_input: (B, U_max) emitted tokens, right-padded. No start symbol --
            RNNTDecoder prepends its own.
        pred_lens: (B,) number of real tokens U in each row (the decoder's own
            SOS makes its output U + 1 long).
        n_chunks: (B,) chunks per utterance, for cross-checking against the
            encoder's own chunking.
        chunk_size: the chunk size these indices were built for.
        text: (B,) reference transcripts. Carried because validation is
            decode-only WER, which needs the references and none of the path.
    """

    audios: torch.Tensor
    audio_lens: torch.Tensor
    b_idx: torch.Tensor
    t_idx: torch.Tensor
    u_idx: torch.Tensor
    labels: torch.Tensor
    pred_input: torch.Tensor
    pred_lens: torch.Tensor
    n_chunks: torch.Tensor
    chunk_size: int
    text: List[str] = field(default_factory=list)

    def to(self, device) -> "ChatAlignedBatch":
        f = lambda x: x.to(device) if torch.is_tensor(x) else x  # noqa: E731
        return ChatAlignedBatch(
            audios=f(self.audios),
            audio_lens=f(self.audio_lens),
            b_idx=f(self.b_idx),
            t_idx=f(self.t_idx),
            u_idx=f(self.u_idx),
            labels=f(self.labels),
            pred_input=f(self.pred_input),
            pred_lens=f(self.pred_lens),
            n_chunks=f(self.n_chunks),
            chunk_size=self.chunk_size,
            text=self.text,
        )


def build_path(chunk_tokens: List[List[int]], blank_id: int):
    """(t_idx, u_idx, labels) for one utterance's forced path.

    ``u`` counts EMITTED LABELS, so it does not advance on a blank -- the
    prediction network only ever sees real tokens. Every chunk contributes
    exactly one blank, silent ones included; that is the signal for "nothing to
    emit here", and dropping it would leave the model no way to learn silence.
    """
    t_idx, u_idx, labels = [], [], []
    u = 0
    for t, toks in enumerate(chunk_tokens):
        for tok in toks:
            t_idx.append(t)
            u_idx.append(u)
            labels.append(int(tok))
            u += 1
        t_idx.append(t)
        u_idx.append(u)
        labels.append(blank_id)
    return t_idx, u_idx, labels


class ChatAlignedDataset(ScriptSTTDataset):
    """SCRIPT's chunk assignment, emitted as a transducer training path.

    Everything about *what* is assigned to *which* chunk -- the forced
    alignment, the emission delay, the tokenizer -- is inherited unchanged; only
    the batch layout differs.
    """

    def __init__(self, cfg, tokenizer, blank_id: Optional[int] = None):
        super().__init__(cfg, tokenizer)
        # Stochastic WORD-level emission delay. 0 disables it.
        self._word_delay_prob = float(getattr(cfg, "word_delay_prob", 0.0) or 0.0)
        self._word_delay_rngs: dict = {}
        # The transducer's blank is an extra class at the END of the vocabulary,
        # the standard NeMo convention (num_classes == vocab_size, blank == V).
        self.blank_id = int(len(self.tokenizer) if blank_id is None else blank_id)

    def _get_word_delay_rng(self) -> random.Random:
        """RNG for the word-delay draw, seeded per dataloader worker.

        Workers must not draw identical sequences, or every worker would perturb
        its batches the same way and the augmentation would be far less diverse
        than it looks -- the same reason the chunk-size and control draws are
        offset by worker id.
        """
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        if wid not in self._word_delay_rngs:
            self._word_delay_rngs[wid] = random.Random(int(getattr(self.cfg, "word_delay_seed", 1234)) + wid)
        return self._word_delay_rngs[wid]

    def get_batch_data(self, cuts, audios, audio_lens, alignments, text) -> ChatAlignedBatch:
        chunk_size = int(self.cfg.chunk_size)
        audio_durations_secs = (audio_lens.float() / self.cfg.sample_rate).tolist()
        system_prompts = [cut.custom.get(self.cfg.prompt_field, self.cfg.system_prompt) for cut in cuts]

        # Identical call to ScriptSTTDataset's, so the word-to-chunk assignment
        # and the emission delay are the same ones the SpeechLM trains on. The
        # prompt/style knobs are irrelevant here -- a transducer has no
        # instruction -- so the configured delay is used directly rather than
        # sampling per-example controls.
        batch_messages = get_llm_messages_for_batch(
            system_role=self.cfg.system_role,
            system_prompt=system_prompts,
            audio_tag=self.cfg.audio_tag,
            blank_token=self.cfg.blank_token,
            chunk_size=chunk_size,
            num_delay_frames=self.cfg.num_delay_frames,
            audio_durations_secs=audio_durations_secs,
            frame_length_in_secs=self.cfg.frame_length_in_secs,
            alignments=alignments,
            transcripts=text,
            capitalization=True,
            punctuation=True,
            word_delay_prob=self._word_delay_prob,
            rng=self._get_word_delay_rng() if self._word_delay_prob > 0.0 else None,
        )

        all_b, all_t, all_u, all_lab = [], [], [], []
        pred_rows, n_chunks = [], []
        for bi, messages in enumerate(batch_messages):
            chunks = self._messages_to_chunks(messages)
            chunk_tokens = [list(c.target_ids) for c in chunks]
            t_idx, u_idx, labels = build_path(chunk_tokens, self.blank_id)

            all_b.extend([bi] * len(t_idx))
            all_t.extend(t_idx)
            all_u.extend(u_idx)
            all_lab.extend(labels)
            n_chunks.append(len(chunk_tokens))
            # Emitted tokens only, WITHOUT a start symbol: RNNTDecoder.forward
            # prepends SOS itself (add_sos=True in training) and returns U+1
            # states, so row u is the state after u labels -- exactly what u_idx
            # indexes. Prefixing SOS here would double it and shift every state
            # by one, which trains the joint against the wrong prediction state
            # while looking perfectly healthy.
            pred_rows.append([t for toks in chunk_tokens for t in toks])

        # At least width 1: an all-silent batch has U=0, and the decoder still
        # needs a tensor to consume.
        u_max = max([len(r) for r in pred_rows] + [1])
        pred_input = torch.full((len(pred_rows), u_max), self.blank_id, dtype=torch.long)
        for i, r in enumerate(pred_rows):
            pred_input[i, : len(r)] = torch.tensor(r, dtype=torch.long)

        return ChatAlignedBatch(
            audios=audios,
            audio_lens=audio_lens,
            b_idx=torch.tensor(all_b, dtype=torch.long),
            t_idx=torch.tensor(all_t, dtype=torch.long),
            u_idx=torch.tensor(all_u, dtype=torch.long),
            labels=torch.tensor(all_lab, dtype=torch.long),
            pred_input=pred_input,
            pred_lens=torch.tensor([len(r) for r in pred_rows], dtype=torch.long),
            n_chunks=torch.tensor(n_chunks, dtype=torch.long),
            chunk_size=chunk_size,
            text=list(text) if text is not None else [],
        )


__all__ = ["ChatAlignedBatch", "ChatAlignedDataset", "ScriptSTTDataConfig", "build_path"]
