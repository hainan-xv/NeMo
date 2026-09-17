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
"""SpeechLM trained to VERIFY a frozen CHAT transducer, chunk by chunk.

Per chunk the model reads CHAT's completed hypothesis and answers ACCEPT or a
correction. That is structurally beyond score fusion, which sees only CHAT's
next-token distribution and therefore cannot react to a phrase it has not
finished reading.

WHAT IS FROZEN, AND WHY IT IS THE WHOLE MODEL. CHAT is frozen end to end --
encoder included -- and the corrector CONSUMES CHAT'S ENCODER OUTPUT rather than
running one of its own. Three consequences, all intended:

  * one encoder forward serves both models instead of two 609M forwards;
  * trainable parameters drop from SCRIPT's 948M to ~340M (LoRA, embed_tokens,
    projection), which is most of the memory that pinned the SCRIPT arms at 86%;
  * the corrector judges CHAT through the representation CHAT actually saw,
    which is the honest input for the question being asked.

The cost is that the warm start is PARTIAL. The LLM, its LoRA and embed_tokens
transfer from a trained SCRIPT checkpoint, but ``perception.proj`` was learned
against SCRIPT's own encoder and must re-adapt to CHAT's. Both descend from the
same nemotron-0.6b donor, so the gap should be small -- an expectation, not a
measurement, and the first thing to watch in the loss curve.

HYPOTHESES ARE GENERATED ONLINE ON THE REFERENCE PREFIX. For chunk k, CHAT is
asked what it would emit given the REFERENCE transcript of chunks 0..k-1. That
matches inference, where earlier chunks have already been corrected, and keeps
each example independent: one early ASR error cannot poison every later label.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from omegaconf import open_dict

from nemo.collections.speechlm2.models.script_model import ScriptSTTModel
from nemo.collections.speechlm2.parts.script_corrector import (
    CorrectorIds,
    collate_corrector_examples,
    corrector_examples_for_utterance,
)
from nemo.utils import logging

__all__ = ["ScriptCorrectorModel", "DEFAULT_CORRECTOR_PROMPT"]

DEFAULT_CORRECTOR_PROMPT = (
    "You are verifying a streaming speech recognizer. Given the transcript so far, the "
    "representation of the next audio chunk, and the recognizer's hypothesis for that chunk, "
    "reply with the accept token if the hypothesis is correct, otherwise output the corrected words."
)


class ScriptCorrectorModel(ScriptSTTModel):
    """SCRIPT's stack, retargeted from generation to verification."""

    def __init__(self, cfg: dict, chat_nemo: Optional[str] = None, **kw) -> None:
        super().__init__(cfg, **kw)
        self.ids = CorrectorIds()
        self.ids.validate(vocab_size=int(self.text_pad_id) + 10**6)

        # The corrector's instruction is DELIBERATELY not SCRIPT's. It describes a
        # different task -- judge a hypothesis, do not produce one -- and the
        # warm-started weights have to be told that. Read from the model config
        # rather than core_cfg: system_prompt is a DATASET field there, which is
        # what the first smoke run tripped over.
        self.system_prompt = (
            cfg.get("system_prompt") or getattr(self.core_cfg, "val_system_prompt", None) or DEFAULT_CORRECTOR_PROMPT
        )

        path = chat_nemo or cfg.get("chat_nemo", "")
        if not path:
            raise ValueError("ScriptCorrectorModel needs chat_nemo: the model it verifies")
        self.chat = self._load_frozen_chat(path)

        # Share CHAT's encoder. Assigning the module (rather than copying weights)
        # means there is exactly one encoder in memory and exactly one forward.
        self.perception.encoder = self.chat.encoder
        for p in self.perception.encoder.parameters():
            p.requires_grad = False

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info("ScriptCorrectorModel: %.0fM trainable (CHAT frozen, encoder shared)", n_train / 1e6)

    # ------------------------------------------------------------------ setup
    def _load_frozen_chat(self, path: str):
        from nemo.collections.asr.models.chat_bpe_models import EncDecCHATBPEModel

        cfg = EncDecCHATBPEModel.restore_from(restore_path=path, return_config=True)
        tok = str(cfg.tokenizer.get("dir", "") or "")
        import os

        if tok and not os.path.isdir(tok):
            # A .nemo records the tokenizer path of the machine that trained it.
            from scripts.script_leaderboard_eval import _hubify  # noqa

            with open_dict(cfg):
                cfg.tokenizer.dir = _hubify(tok)
        m = EncDecCHATBPEModel.restore_from(restore_path=path, override_config_path=cfg, map_location="cpu")
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        return m

    def state_dict(self, *a, **kw):
        """Drop CHAT from checkpoints.

        It is frozen and reloaded from its own .nemo, so persisting it would add
        ~3 GB to every checkpoint and, worse, let a stale copy silently override
        the .nemo the launcher points at.
        """
        sd = super().state_dict(*a, **kw)
        return {k: v for k, v in sd.items() if not k.startswith("chat.")}

    def load_state_dict(self, state_dict, strict: bool = True):
        return super().load_state_dict(state_dict, strict=False)

    # ------------------------------------------------------------- data prep
    @torch.no_grad()
    def _reference_chunks(self, cut, n_chunks: int):
        """Reference words and token ids per chunk, on the model's own grid."""
        from nemo.collections.asr.parts.utils.chat_alignment import assign_words_to_chunks

        aligned = (cut.custom or {}).get("alignments", []) or []
        words = [w["text"] for w in aligned]
        groups = assign_words_to_chunks(
            [w["end_time"] for w in aligned],
            n_chunks,
            self.chat.joint.chunk_size,
            self.chat.frame_length_in_secs,
            self.chat.num_delay_frames,
        )
        w_chunks = [[words[i] for i in g] for g in groups]
        id_chunks = [self.tokenizer.text_to_ids(" ".join(c)) if c else [] for c in w_chunks]
        return w_chunks, id_chunks

    @torch.no_grad()
    def _chat_hypotheses(self, enc, enc_len, ref_id_chunks, n_chunks):
        """Greedy CHAT output per chunk, conditioned on the REFERENCE prefix."""
        from nemo.collections.speechlm2.parts.chat_fusion import ChatFusionScorer, chat_only_chunk

        scorer = ChatFusionScorer(self.chat, enc, enc_len)
        B = len(ref_id_chunks)
        hyp = [[] for _ in range(B)]
        for k in range(max(n_chunks)):
            rows = [b for b in range(B) if k < n_chunks[b]]
            if not rows:
                continue
            prefixes = [[t for c in ref_id_chunks[b][:k] for t in c] for b in rows]
            toks, _ = chat_only_chunk(scorer, rows, k, prefixes, max_new_tokens=32, margin_threshold=0.0)
            for i, b in enumerate(rows):
                hyp[b].append(toks[i])
        return hyp

    # ------------------------------------------------------------- train step
    def training_step(self, batch, batch_idx):
        sig, sig_len = batch[0], batch[1]
        cuts = batch[4] if len(batch) >= 5 else None
        if cuts is None:
            raise RuntimeError("corrector training needs cuts (word timings) on the batch")

        with torch.no_grad():
            proc, proc_len = self.chat.preprocessor(input_signal=sig, length=sig_len)
            enc, enc_len = self.chat.encoder(audio_signal=proc, length=proc_len)
            enc = enc.transpose(1, 2)  # [B, T, D]; joint_on_path chunks it itself

        cs = self.chat.joint.chunk_size
        n_chunks = [int((int(l) + cs - 1) // cs) for l in enc_len]

        ref_w, ref_i = [], []
        for b in range(len(n_chunks)):
            w, i = self._reference_chunks(cuts[b], n_chunks[b])
            ref_w.append(w)
            ref_i.append(i)

        hyp_i = self._chat_hypotheses(enc, enc_len, ref_i, n_chunks)

        instr = self.tokenizer.text_to_ids(self.system_prompt + "\n")
        examples, frame_src = [], []
        for b in range(len(n_chunks)):
            hyp_w = [self.tokenizer.ids_to_text(t).split() if t else [] for t in hyp_i[b]]
            lens = [min(cs, max(0, int(enc_len[b]) - k * cs)) for k in range(n_chunks[b])]
            exs = corrector_examples_for_utterance(instr, ref_w[b], ref_i[b], hyp_i[b], hyp_w, lens, ids=self.ids)
            for k, e in enumerate(exs):
                examples.append(e)
                frame_src.append((b, k * cs))
        if not examples:
            return enc.sum() * 0.0

        cb = collate_corrector_examples(examples, self.text_pad_id, ids=self.ids)
        dev = enc.device
        input_ids = torch.tensor(cb.input_ids, dtype=torch.long, device=dev)
        labels = torch.tensor(cb.labels, dtype=torch.long, device=dev)
        attn = torch.tensor(cb.attention_mask, dtype=torch.long, device=dev)

        embeds = self._embed_tokens(input_ids)
        if cb.audio_slots:
            # Splice CHAT's frames through the projection into the reserved slots.
            proj = self.perception.proj
            rows = torch.tensor([r for r, _, _ in cb.audio_slots], device=dev)
            poss = torch.tensor([p for _, p, _ in cb.audio_slots], device=dev)
            srcb = torch.tensor([frame_src[r][0] for r, _, _ in cb.audio_slots], device=dev)
            srcf = torch.tensor([frame_src[r][1] + k for r, _, k in cb.audio_slots], device=dev).clamp_(
                max=enc.shape[1] - 1
            )
            embeds = embeds.clone()
            embeds[rows, poss] = proj(enc[srcb, srcf].to(embeds.dtype))

        out = self.llm(inputs_embeds=embeds, attention_mask=attn, labels=labels)
        loss = out.loss
        acc = float(sum(cb.is_accept)) / max(1, len(cb.is_accept))
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("accept_frac", acc, prog_bar=True, sync_dist=True)
        self.log("chunks_per_batch", float(len(examples)), sync_dist=True)
        return loss
