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

from nemo.collections.asr.parts.utils.chunk_error_labels import simple_normalize
from nemo.collections.speechlm2.models.script_model import ScriptSTTModel
from nemo.collections.speechlm2.parts.script_corrector import (
    CorrectorIds,
    collate_corrector_examples,
    corrector_examples_for_utterance,
    decision_stats,
    format_sample,
    word_errors,
)
from nemo.utils import logging

__all__ = ["ScriptCorrectorModel", "DEFAULT_CORRECTOR_PROMPT"]

# EVERY rank must log EXACTLY these keys, in this order, on EVERY step.
#
# self.log(..., sync_dist=True) is a collective. A rank that skips one -- because
# its batch had no examples, or a metric was undefined, or a try/except swallowed
# it -- issues fewer collectives than its peers, and the group desyncs. That is
# not a slow-training symptom and it does not look like a logging bug: it
# surfaces ten minutes later as
#   WorkNCCL(SeqNum=..., OpType=ALLREDUCE, NumelIn=1, NumelOut=1) ran for 600013ms
# and kills the job. It killed dfw_corrector_v1 at step ~6654 while the step
# timing was a healthy 0.3-0.8 s.
_METRIC_KEYS = (
    "train_loss",
    "chunks_per_batch",
    "train_pred_accept_frac",
    "train_label_accept_frac",
    "train_reject_precision",
    "train_reject_recall",
    "train_chat_wer_tf",
)

# Validation reports the same quantities, but ACCUMULATED over the whole split
# rather than per batch: corpus WER is total edits over total reference words,
# and averaging per-batch rates would over-weight batches of short utterances.
# Logged once in on_validation_epoch_end, so the same rank-uniformity rule
# applies there -- every rank logs every key, every epoch.
_VAL_METRIC_KEYS = (
    "val_chat_wer_tf",
    "val_corrected_wer",
    # corrected MINUS chat, so NEGATIVE is the corrector winning. Logged as its
    # own series because the effect is a couple of WER points on top of numbers
    # around 0.05 -- a size that two overlaid curves hide and a difference shows.
    "val_wer_delta",
    "val_reject_precision",
    "val_reject_recall",
    "val_pred_accept_frac",
    "val_label_accept_frac",
)


def _norm_words(words):
    """Normalise and DROP anything that normalises away.

    Standalone punctuation becomes "" and would otherwise count as an unmatched
    word on both sides, inflating every WER computed here -- the same defect that
    pushed the label accept rate from ~0.93 to ~0.77.
    """
    return [w for w in (simple_normalize(x) for x in words) if w]


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
        """Reference token ids and words per chunk, via CHAT'S OWN partitioning.

        Delegates to ``chat._chunk_tokens`` rather than re-deriving this. Two
        differences made the hand-rolled version wrong, and both inflated the
        measured WER roughly fivefold (0.40-0.49 against CHAT's true ~0.08):

          * it used the ALIGNER'S word forms, which have punctuation stripped,
            while CHAT is trained on the original transcript. Every punctuated
            sentence then scored as errors. _chunk_tokens' own docstring warns
            about exactly this.
          * it tokenized each chunk's text INDEPENDENTLY. Partitioning the full
            utterance's tokenization is not the same thing -- in Qwen BPE
            " hello" and "hello" are different tokens -- so every chunk after the
            first began with the wrong id, and that wrong prefix was then fed to
            CHAT's prediction network as its history.

        Using the model's own method makes the reference the corrector trains
        against identical to the one CHAT was trained against, by construction.
        Words are derived FROM the ids so the two can never disagree.
        """
        id_chunks = self.chat._chunk_tokens(cut, n_chunks)
        w_chunks = [self.tokenizer.ids_to_text(c).split() if c else [] for c in id_chunks]
        return w_chunks, id_chunks

    @torch.no_grad()
    def _chat_hypotheses(self, enc, enc_len, ref_id_chunks, n_chunks):
        """Greedy CHAT output per chunk, conditioned on CHAT'S OWN history.

        FREE-RUNNING, not teacher-forced, and the reason is not a preference --
        teacher forcing cannot produce a coherent transcript here at all.

        CHAT defers words past a chunk boundary by design; in free-running decode
        they reappear in the next chunk and the transcript is complete. Feed it
        the REFERENCE prefix instead and the deferred words are already present
        in that prefix, so CHAT never re-emits them and they vanish from the
        concatenation. Every boundary leaks a word or two. Measured: a
        deletion-dominated 0.40-0.49 WER against CHAT's true ~0.08, visible in
        the sample dump as a missing tail on nearly every chunk --
        "help you out" -> "help you", "provide you with an" -> "provide you with".

        The labelling rule flattens the hypothesis and aligns it to the
        reference, which presumes a real transcript. So the hypothesis has to be
        one.

        The cost is the property reference-history was chosen for: at inference
        the corrector sees history that has ALREADY been corrected, whereas here
        it sees CHAT's raw output. That mismatch is real, and an early error now
        propagates into later chunks' context -- but it is a second-order effect
        next to a hypothesis that is not a transcript.
        """
        from nemo.collections.speechlm2.parts.chat_fusion import ChatFusionScorer, chat_only_chunk

        scorer = ChatFusionScorer(self.chat, enc, enc_len)
        B = len(ref_id_chunks)
        hyp = [[] for _ in range(B)]
        for k in range(max(n_chunks) if n_chunks else 0):
            rows = [b for b in range(B) if k < n_chunks[b]]
            if not rows:
                continue
            # CHAT's OWN emitted prefix, so deferred words carry forward.
            prefixes = [[t for c in hyp[b] for t in c] for b in rows]
            toks, _ = chat_only_chunk(scorer, rows, k, prefixes, max_new_tokens=32, margin_threshold=0.0)
            for i, b in enumerate(rows):
                hyp[b].append(toks[i])
        return hyp

    # ------------------------------------------------------- shared pipeline
    def _prepare(self, batch):
        """Audio -> CHAT hypotheses -> labels -> a padded, audio-spliced batch.

        Shared by training and validation so the two cannot drift: a corrector
        evaluated on differently-built examples than it trained on would report a
        number about a different task.

        Returns ``None`` when the batch yields no chunks at all, which the caller
        must still treat as a LOGGING step -- see _METRIC_KEYS.
        """
        sig, sig_len = batch[0], batch[1]
        cuts = batch[4] if len(batch) >= 5 else None
        if cuts is None:
            raise RuntimeError("corrector needs cuts (word timings) on the batch")

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

        from nemo.collections.asr.parts.utils.chunk_error_labels import label_chunks

        examples, frame_src, sample = [], [], None
        chat_e = chat_n = 0
        for b in range(len(n_chunks)):
            # ONE detokenization of the whole hypothesis; per-chunk detokenization
            # splits words that straddle a chunk boundary.
            hyp_w = self.tokenizer.ids_to_text([t for c in hyp_i[b] for t in c]).split()
            lens = [min(cs, max(0, int(enc_len[b]) - k * cs)) for k in range(n_chunks[b])]
            chunk_labels, _ = label_chunks(hyp_w, ref_w[b], normalize=simple_normalize)

            rw = [w for c in ref_w[b] for w in c]
            e, n = word_errors(_norm_words(hyp_w), _norm_words(rw))
            chat_e += e
            chat_n += n

            for k, ex in enumerate(
                corrector_examples_for_utterance(
                    instr,
                    ref_w[b],
                    ref_i[b],
                    hyp_i[b],
                    hyp_w,
                    lens,
                    ids=self.ids,
                    normalize=simple_normalize,
                )
            ):
                examples.append(ex)
                frame_src.append((b, k * cs))

            if b == 0:
                sample = (
                    [" ".join(c) for c in ref_w[b]],
                    [self.tokenizer.ids_to_text(t) if t else "" for t in hyp_i[b]],
                    chunk_labels,
                )

        if not examples:
            return None

        cb = collate_corrector_examples(examples, self.text_pad_id, ids=self.ids)
        dev = enc.device
        input_ids = torch.tensor(cb.input_ids, dtype=torch.long, device=dev)
        labels = torch.tensor(cb.labels, dtype=torch.long, device=dev)
        attn = torch.tensor(cb.attention_mask, dtype=torch.long, device=dev)

        embeds = self._embed_tokens(input_ids)
        if cb.audio_slots:
            # Splice CHAT's frames through the projection into the reserved slots.
            rows = torch.tensor([r for r, _, _ in cb.audio_slots], device=dev)
            poss = torch.tensor([p for _, p, _ in cb.audio_slots], device=dev)
            srcb = torch.tensor([frame_src[r][0] for r, _, _ in cb.audio_slots], device=dev)
            srcf = torch.tensor([frame_src[r][1] + k for r, _, k in cb.audio_slots], device=dev).clamp_(
                max=enc.shape[1] - 1
            )
            embeds = embeds.clone()
            embeds[rows, poss] = self.perception.proj(enc[srcb, srcf].to(embeds.dtype))

        return {
            "embeds": embeds,
            "attn": attn,
            "labels": labels,
            "examples": examples,
            "cb": cb,
            "device": dev,
            "sample": sample,
            "chat_e": chat_e,
            "chat_n": chat_n,
            "carry": {
                "enc_len": enc_len,
                "n_chunks": n_chunks,
                "ref_w": ref_w,
                "ref_i": ref_i,
                "hyp_i": hyp_i,
                "instr": instr,
                "device": dev,
            },
        }

    # ------------------------------------------------------------- validation
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Same pipeline as training, ACCUMULATED rather than logged per batch.

        Nothing is logged here on purpose: logging inside the loop would average
        rates (wrong for corpus WER) and make the number of collectives depend on
        how many val batches a rank happened to get -- the desync that killed
        dfw_corrector_v1.

        Overrides the PARENT's validation_step, which expects SCRIPT's own batch
        type and dies with "'list' object has no attribute 'text'" on a plain
        lhotse batch.
        """
        try:
            st = self._prepare(batch)
        except RuntimeError:
            return
        if st is None:
            return

        out = self.llm(inputs_embeds=st["embeds"], attention_mask=st["attn"], labels=st["labels"])
        first = torch.tensor([e.prompt_len - 1 for e in st["examples"]], device=st["device"])
        rows = torch.arange(len(st["examples"]), device=st["device"])
        pred_accept = (out.logits[rows, first].argmax(-1) == self.ids.accept).tolist()

        v = self._val
        v["pred"] += pred_accept
        v["label"] += st["cb"].is_accept
        v["chat_e"] += st["chat_e"]
        v["chat_n"] += st["chat_n"]
        try:
            e, n = self._corrected_wer(st["carry"], return_counts=True)
            v["corr_e"] += e
            v["corr_n"] += n
        except Exception as exc:
            logging.warning("val corrected_wer failed: %s", exc)

    def on_validation_epoch_start(self):
        self._val = {"pred": [], "label": [], "chat_e": 0, "chat_n": 0, "corr_e": 0, "corr_n": 0}

    def on_validation_epoch_end(self):
        v = getattr(self, "_val", None) or {
            "pred": [],
            "label": [],
            "chat_e": 0,
            "chat_n": 0,
            "corr_e": 0,
            "corr_n": 0,
        }
        stats = decision_stats(v["pred"], v["label"]) if v["pred"] else {}
        vals = {
            "val_chat_wer_tf": (v["chat_e"] / v["chat_n"]) if v["chat_n"] else 0.0,
            "val_corrected_wer": (v["corr_e"] / v["corr_n"]) if v["corr_n"] else 0.0,
            # Differenced from the ACCUMULATED counts, not averaged over batches:
            # corpus WER is total edits over total reference words, so a mean of
            # per-batch deltas would weight a short utterance like a long one.
            "val_wer_delta": (
                ((v["corr_e"] / v["corr_n"]) if v["corr_n"] else 0.0)
                - ((v["chat_e"] / v["chat_n"]) if v["chat_n"] else 0.0)
            ),
            **{f"val_{k}": float(x) for k, x in stats.items()},
        }
        # Fixed key set, every rank, every epoch -- same rule as _METRIC_KEYS.
        for k in _VAL_METRIC_KEYS:
            self.log(
                k,
                float(vals.get(k, 0.0)),
                prog_bar=k in ("val_corrected_wer", "val_chat_wer_tf"),
                sync_dist=True,
            )
        self._val = None

    def training_step(self, batch, batch_idx):
        st = self._prepare(batch)
        if st is None:
            # A rank that stays silent here desyncs the logging collectives.
            self._log_all({})
            return torch.zeros((), device=self.device, requires_grad=True)

        out = self.llm(inputs_embeds=st["embeds"], attention_mask=st["attn"], labels=st["labels"])
        loss = out.loss

        with torch.no_grad():
            # The DECISION is the argmax at prompt_len - 1: logits at position i
            # predict token i+1, so that column is the first target token.
            first = torch.tensor([e.prompt_len - 1 for e in st["examples"]], device=st["device"])
            rows = torch.arange(len(st["examples"]), device=st["device"])
            pred_accept = (out.logits[rows, first].argmax(-1) == self.ids.accept).tolist()
            stats = decision_stats(pred_accept, st["cb"].is_accept)

        self._log_all(
            {
                "train_loss": loss,
                "chunks_per_batch": float(len(st["examples"])),
                **{f"train_{k}": float(v) for k, v in stats.items()},
                "train_chat_wer_tf": (st["chat_e"] / st["chat_n"]) if st["chat_n"] else 0.0,
            }
        )

        # Periodic dump of one real example. Metrics say WHETHER the labels look
        # right in aggregate; this says WHAT they are -- and shows the
        # hypothesis's own chunk boundaries, which no metric exposes and which
        # separate a real error from a timing shift.
        every = int(getattr(self.core_cfg, "sample_print_every_n_steps", 0) or 100)
        if every and self.global_step % every == 0 and self.trainer.global_rank == 0 and st["sample"]:
            try:
                logging.info("\n" + format_sample(*st["sample"], step=self.global_step))
            except Exception as e:
                logging.warning("sample print failed: %s", e)

        self._last = st["carry"]
        # Stashed WITH _last, because the delta is only meaningful when both
        # halves come from the same batch: train_corrected_wer is recomputed
        # every corrected_wer_every_n_steps, while train_chat_wer_tf is logged
        # every step, so differencing the two live panels would subtract numbers
        # measured up to 200 steps apart.
        self._last_chat_wer = (st["chat_e"] / st["chat_n"]) if st["chat_n"] else None
        return loss

    @torch.no_grad()
    def _corrected_wer(self, st, return_counts: bool = False):
        """WER after applying the corrector's own accept/reject decisions.

        Accept keeps CHAT's chunk; reject substitutes the reference. That makes
        this an ORACLE-CORRECTION number: it measures the DECISION quality only,
        not the generated text, so it is the ceiling the accept/reject head can
        reach. Reported beside train_chat_wer_tf on the same batch, the pair is an
        A/B on identical audio -- any gap is the decision doing work. Generating
        the corrections instead would fold two skills into one number and make a
        regression impossible to attribute.
        """
        cs = self.chat.joint.chunk_size
        dev = st["device"]
        e_tot = n_tot = 0
        for b in range(len(st["n_chunks"])):
            out_ids: List[int] = []
            history: List[int] = []
            for k in range(st["n_chunks"][b]):
                hyp = st["hyp_i"][b][k]
                alen = min(cs, max(0, int(st["enc_len"][b]) - k * cs))
                ex = corrector_examples_for_utterance(st["instr"], [[]], [[]], [hyp], [], [alen], ids=self.ids)[0]
                tail = ex.input_ids[len(st["instr"]) : ex.prompt_len]
                ids_t = torch.tensor([list(st["instr"]) + history + tail], dtype=torch.long, device=dev)
                logits = self.llm(inputs_embeds=self._embed_tokens(ids_t)).logits[0, -1]
                toks = hyp if int(logits.argmax()) == self.ids.accept else st["ref_i"][b][k]
                out_ids += list(toks)
                history = history + list(st["ref_i"][b][k])
            # One detokenization of the whole output, for the same reason as above:
            # per-chunk detokenization splits words that straddle a boundary.
            out_words = self.tokenizer.ids_to_text(out_ids).split() if out_ids else []
            rw = [w for c in st["ref_w"][b] for w in c]
            e, n = word_errors(_norm_words(out_words), _norm_words(rw))
            e_tot += e
            n_tot += n
        if return_counts:
            # Counts, not a rate: corpus WER is total edits over total reference
            # words, so validation must accumulate and divide ONCE at the end.
            return e_tot, n_tot
        return (e_tot / n_tot) if n_tot else None

    def _log_all(self, values: dict) -> None:
        """Log the whole metric set, filling anything absent with 0.0.

        Fixed key set, fixed order, every rank, every step -- see _METRIC_KEYS.
        """
        for k in _METRIC_KEYS:
            v = values.get(k, 0.0)
            self.log(
                k,
                v if torch.is_tensor(v) else float(v),
                prog_bar=k
                in (
                    "train_loss",
                    "train_reject_recall",
                    "train_chat_wer_tf",
                    "train_label_accept_frac",
                ),
                sync_dist=True,
            )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # global_step is identical across ranks, so the BRANCH is taken by all of
        # them or none -- which is what makes this collective safe. Inside it the
        # log is unconditional: a rank whose computation failed must still log, or
        # it desyncs exactly like the training_step case.
        every = int(getattr(self.core_cfg, "corrected_wer_every_n_steps", 0) or 200)
        if every and self.global_step and self.global_step % every == 0:
            w = 0.0
            ok = False
            last = getattr(self, "_last", None)
            if last:
                try:
                    w = self._corrected_wer(last) or 0.0
                    ok = True
                except Exception as e:  # a metric must never kill a training run
                    logging.warning("corrected_wer failed at step %s: %s", self.global_step, e)
            self.log("train_corrected_wer", float(w), prog_bar=True, sync_dist=True)
            # Both logs sit INSIDE the rank-uniform branch and are unconditional
            # within it -- a rank that skipped either one would desync the
            # sync_dist collective, which is how an earlier version hung.
            # 0.0 on failure rather than w - chat: with w defaulted to 0.0 a
            # failed computation would otherwise post a large spurious GAIN.
            chat = getattr(self, "_last_chat_wer", None)
            delta = (w - chat) if (ok and chat is not None) else 0.0
            self.log("train_wer_delta", float(delta), prog_bar=True, sync_dist=True)
        self._last = None
        self._last_chat_wer = None
