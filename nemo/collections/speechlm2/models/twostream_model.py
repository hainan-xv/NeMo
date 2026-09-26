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
"""Two-stream SCRIPT: text and audio meet only in the LLM's final layer(s).

The text stream is an ordinary causal LM over ``[prompt][transcript]``. Because
it never attends audio, the hidden state after prefix ``p`` is independent of
which chunk or which candidate cut is being scored -- so it is computed ONCE and
every lattice cell indexes into it.

The joint is the LLM's own last layer, run a second time over
``[text_h][audio block per cell]`` with a mask that gives cell ``(t, p)`` the
text keys ``< m + p`` and its own audio block. The distribution is read at each
block's last position.

Contrast with packed SCRIPT, where each candidate is a separate branch segment
carrying its own copy of the chunk's audio: there, widening the band multiplies
the packed length and the ``(B, L, V)`` logit tensor. Here the band selects a
SUBSET of ``(chunk, text position)`` cells, so widening it costs less work.

STATUS: first implementation, single utterance per step, no KV cache reuse
across chunks at inference. It exists to be measured against packed SCRIPT on the
``band_words=0 == forced`` equivalence before being optimised.
"""

from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from nemo.collections.speechlm2.models.script_model import ScriptSTTModel, ScriptSTTModelConfig
from nemo.collections.speechlm2.parts.script_banded import banded_forward, span_scores
from nemo.collections.speechlm2.parts.twostream import (
    build_joint_inputs,
    cell_logprobs,
    gather_span_tensors,
    plan_cells,
)
from nemo.utils import logging


class TwoStreamSTTModelConfig(ScriptSTTModelConfig):
    # How many trailing LLM layers see audio. 1 is the design point; making it a
    # knob is the cheap hedge against last-layer-only fusion being too shallow,
    # since that is a bet rather than a known quantity.
    joint_layers: int = 1
    # How the per-utterance NLLs are combined. "mean_volume" is NeMo's RNN-T
    # convention and what packed SCRIPT uses: SUM of losses over SUM of target
    # lengths, so every token carries equal weight regardless of which utterance
    # it came from. "mean" would average per-utterance ratios instead, which
    # over-weights short utterances.
    loss_reduction: str = "mean_volume"


class TwoStreamSTTModel(ScriptSTTModel):
    """SCRIPT with the band expressed as a lattice instead of as sequence length."""

    # ------------------------------------------------------------------
    # Reaching into the LLM
    # ------------------------------------------------------------------
    def _llm_core(self) -> nn.Module:
        """The decoder stack, unwrapping PEFT if present.

        PEFT inserts ``base_model.model`` between the wrapper and the real model;
        SALM and duplex both unwrap the same way.
        """
        llm = self.llm
        if hasattr(llm, "base_model") and hasattr(llm.base_model, "model"):
            llm = llm.base_model.model
        return llm.model if hasattr(llm, "model") else llm

    def _joint_layers(self) -> List[nn.Module]:
        n = int(getattr(self.core_cfg, "joint_layers", 1) or 1)
        layers = self._llm_core().layers
        if n > len(layers):
            raise ValueError(f"joint_layers={n} exceeds the LLM's {len(layers)} layers")
        return list(layers[-n:])

    def _lm_head_of(self) -> nn.Module:
        llm = self.llm
        if hasattr(llm, "base_model") and hasattr(llm.base_model, "model"):
            llm = llm.base_model.model
        return llm.lm_head

    # ------------------------------------------------------------------
    # Text stream
    # ------------------------------------------------------------------
    def _text_hidden(self, input_ids: Tensor) -> Tensor:
        """Hidden state entering the joint layers, for ``[prompt][transcript]``.

        Implemented by running the full LLM with ``output_hidden_states=True`` and
        taking the activation ``joint_layers`` from the end, rather than slicing
        the layer stack by hand. That recomputes the trailing layers on text whose
        output is unused -- a few layers over ~60 positions, which is negligible
        against what the packing saves, and it keeps this free of assumptions
        about rotary/position plumbing inside the stack.
        """
        n_joint = int(getattr(self.core_cfg, "joint_layers", 1) or 1)
        embeds = self._embed_tokens(input_ids)
        out = self._llm_forward(
            inputs_embeds=embeds,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = out["hidden_states"]
        # hidden_states[0] is the embedding output; [-1] is after the last layer.
        return hs[-(n_joint + 1)]

    # ------------------------------------------------------------------
    # Joint
    # ------------------------------------------------------------------
    def _run_joint(self, seq: Tensor, mask: Tensor, position_ids: Tensor) -> Tensor:
        """Apply the joint layer(s) to the packed ``[text | audio blocks]`` sequence."""
        core = self._llm_core()
        h = seq.unsqueeze(0)  # (1, L_j, H)
        pos = position_ids.unsqueeze(0)

        # Additive 4-D mask: 0 where allowed, large negative where not.
        add = torch.zeros(mask.shape, dtype=h.dtype, device=h.device)
        add = add.masked_fill(~mask, torch.finfo(h.dtype).min).unsqueeze(0).unsqueeze(0)

        rotary = getattr(core, "rotary_emb", None)
        pos_emb = rotary(h, pos) if rotary is not None else None

        for layer in self._joint_layers():
            kw = {"attention_mask": add, "position_ids": pos}
            if pos_emb is not None:
                kw["position_embeddings"] = pos_emb
            res = layer(h, **kw)
            h = res[0] if isinstance(res, tuple) else res
        return h.squeeze(0)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def twostream_loss(
        self,
        text_ids: Tensor,
        audio_emb: Tensor,
        cut: Tensor,
        cut_valid: Tensor,
        span_valid: Tensor,
        reach: Tensor,
        spine_ids: Tensor,
        prompt_len: int,
        n_tokens: int,
    ) -> Tensor:
        """Negative log-likelihood for one utterance, marginalised over the band.

        Args:
            text_ids: ``(m + U,)`` prompt followed by the transcript.
            audio_emb: ``(T, w, H)`` per-chunk audio, already at the LLM width.
            cut / cut_valid / span_valid / reach: the lattice, as built for packed
                SCRIPT -- unchanged, which is the point.
            spine_ids: ``(U,)`` transcript token ids.
        """
        text_h = self._text_hidden(text_ids.unsqueeze(0)).squeeze(0)  # (m+U, H)
        cells = plan_cells(cut, cut_valid, reach)
        seq, mask, read_at = build_joint_inputs(text_h, audio_emb, cells, prompt_len)

        # Positions: text keeps its own index; a cell's audio continues from its
        # text cutoff, mirroring SCRIPT's branch position scheme so the joint sees
        # a contiguous stretch rather than a jump.
        tu = text_h.shape[0]
        w = audio_emb.shape[1]
        pos = torch.arange(seq.shape[0], device=seq.device)
        for c in range(cells.n_cells):
            s = tu + c * w
            base = prompt_len + int(cells.text_pos[c].item())
            pos[s : s + w] = base + torch.arange(w, device=seq.device)

        h = self._run_joint(seq, mask, pos)
        logits = self._lm_head_of()(h[read_at])  # (n_cells, V) -- ONLY at the cells
        tok_lp, stop_lp = cell_logprobs(logits, cells, spine_ids, self._eot_id)

        K = int(span_valid.shape[-1]) - 1
        token_logprob, stop_logprob = gather_span_tensors(tok_lp, stop_lp, cells, cut, K)
        sigma = span_scores(token_logprob.unsqueeze(0), stop_logprob.unsqueeze(0), span_valid.unsqueeze(0))
        nll = banded_forward(
            sigma,
            cut.unsqueeze(0),
            cut_valid.unsqueeze(0),
            torch.tensor([cut.shape[0]], device=cut.device),
            torch.tensor([n_tokens], device=cut.device),
        )
        return nll[0]

    # ------------------------------------------------------------------
    # Adapter: packed ScriptBatch -> two-stream arguments
    # ------------------------------------------------------------------
    def _chunk_audio(self, audios: Tensor, audio_lens: Tensor, chunk_size: int, n_chunks: int) -> Tensor:
        """Encoder frames, reshaped to ``(T, w, H)``.

        The encoder runs ONCE over the utterance; chunk t simply takes frames
        ``[t*C, (t+1)*C)``. Nothing is replicated here -- that is the difference
        from the packed layout, where every candidate carried its own copy.
        """
        embs, _ = self.perception(input_signal=audios, input_signal_length=audio_lens)  # (1, F, H)
        embs = embs[0]
        F, H = embs.shape
        need = n_chunks * chunk_size
        if need > F:
            embs = torch.cat([embs, embs.new_zeros(need - F, H)], dim=0)
        return embs[:need].reshape(n_chunks, chunk_size, H)

    @staticmethod
    def _reach_from_cuts(cut: Tensor, cut_valid: Tensor, n_tokens: int) -> Tensor:
        """``reach[t]`` = furthest spine index chunk t may emit up to.

        Derived rather than stored: it is ``max(valid cuts of chunk t+1)``, and
        ``n_tokens`` for the last chunk -- the same rule the packer uses.
        """
        T = int(cut.shape[0])
        out = []
        for t in range(T):
            if t + 1 < T:
                nxt = cut[t + 1][cut_valid[t + 1]]
                out.append(int(nxt.max().item()) if nxt.numel() else n_tokens)
            else:
                out.append(n_tokens)
        return torch.tensor(out, dtype=torch.long, device=cut.device)

    def _utt_args(self, batch, b: int) -> dict:
        """Pull one utterance's two-stream arguments out of a packed banded batch."""
        lat = batch.banded
        n_tok = int(lat.n_tokens[b].item())
        n_chunks = int(lat.n_chunks[b].item())
        spine_len = int(lat.spine_lens[b].item())
        prompt_len = spine_len - n_tok  # the packer writes prompt then transcript

        cut = lat.cut[b, :n_chunks]
        cut_valid = lat.cut_valid[b, :n_chunks]
        span_valid = lat.span_valid[b, :n_chunks]
        text_ids = batch.input_tokens[b, :spine_len]
        spine_ids = text_ids[prompt_len:]

        return dict(
            text_ids=text_ids,
            audio_emb=self._chunk_audio(
                batch.audios[b : b + 1], batch.audio_lens[b : b + 1], int(batch.chunk_size), n_chunks
            ),
            cut=cut,
            cut_valid=cut_valid,
            span_valid=span_valid,
            reach=self._reach_from_cuts(cut, cut_valid, n_tok),
            spine_ids=spine_ids,
            prompt_len=prompt_len,
            n_tokens=n_tok,
        )

    def _training_step_inner(self, batch, batch_idx: int):
        """One optimisation step.

        Utterances are looped rather than batched: the joint packing is per
        utterance (cells depend on that utterance's lattice), and getting a
        correct number first matters more than the throughput. Batching the joint
        is the obvious next optimisation.
        """
        if batch.banded is None:
            raise ValueError(
                "TwoStreamSTTModel needs the banded lattice on the batch. Set loss_type=banded "
                "(band_words=0 reproduces the forced loss exactly)."
            )
        losses = []
        n_targets = 0
        n_chunks_total = 0
        for b in range(int(batch.input_tokens.shape[0])):
            args = self._utt_args(batch, b)
            nll = self.twostream_loss(**args)
            if torch.isfinite(nll):
                losses.append(nll)
                n_targets += args["n_tokens"]
                n_chunks_total += int(args["cut"].shape[0])
            else:
                # No in-band path completed. Report it rather than letting -inf
                # dominate the mean, which is how a silent data bug would look.
                logging.warning("utterance %d has no completable in-band path; skipping", b)
        if not losses:
            raise RuntimeError("no utterance in this batch produced a finite loss")

        reduction = str(getattr(self.core_cfg, "loss_reduction", "mean_volume"))
        stacked = torch.stack(losses)
        if reduction == "mean_volume":
            # Token-weighted: one long utterance counts for more than one short
            # one, which is what keeps the gradient scale stable as the bucket
            # composition changes.
            loss = stacked.sum() / max(n_targets, 1)
        elif reduction == "mean":
            loss = stacked.mean()
        elif reduction == "sum":
            loss = stacked.sum()
        else:
            raise ValueError(f"loss_reduction must be mean_volume | mean | sum, got {reduction!r}")

        self.log_dict(
            {
                "train_loss": loss.detach(),
                "num_targets": float(n_targets),
                # Emissions = tokens + one <eot> per chunk. The lattice scores both,
                # but mean_volume divides by TOKENS only (matching packed SCRIPT),
                # so this is logged to make the difference visible rather than
                # silently inflating the reported per-token loss.
                "num_emissions": float(n_targets + n_chunks_total),
            },
            on_step=True,
            prog_bar=True,
        )
        return {"loss": loss}

    # ------------------------------------------------------------------
    def on_train_start(self) -> None:  # pragma: no cover - logging only
        n = int(getattr(self.core_cfg, "joint_layers", 1) or 1)
        logging.info(
            "TwoStreamSTTModel: text stream is causal and audio-free; %d trailing layer(s) form the joint. "
            "The band selects lattice cells rather than adding packed segments.",
            n,
        )
        return super().on_train_start() if hasattr(super(), "on_train_start") else None
