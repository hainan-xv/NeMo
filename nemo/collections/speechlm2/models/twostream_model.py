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

import math
from dataclasses import dataclass, fields
from typing import List, Optional, Tuple, Union

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
from nemo.collections.speechlm2.parts.utils.misc import to_dataclass
from nemo.utils import logging


@dataclass
class TwoStreamSTTModelConfig(ScriptSTTModelConfig):
    # @dataclass IS REQUIRED. Without it these annotations stay plain class
    # attributes: dataclasses.fields() returns only the parent's, so to_dataclass
    # drops every field below as "not supported and will be ignored", and
    # is_dataclass() still answers True (inherited) so the omission is invisible.
    # The result is a config knob Hydra accepts, logs, and silently discards.
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
    # Give the joint its OWN layer instead of borrowing the LLM's last one.
    #
    # With joint_layers=1 and no extra layer, layer N-1 is already exclusively the
    # joint (the text stream reads hidden_states[-2], i.e. layer N-1's INPUT), so
    # the text representation is silently one layer short of the full stack AND
    # the joint is frozen apart from ~0.92M of LoRA on q_proj/v_proj. This flag
    # fixes both: text gets all N layers, and the joint becomes a fully trainable
    # ~50M-parameter layer of its own.
    extra_joint_layer: bool = False
    # Warm-start the extra layer from the LLM's last layer rather than random init.
    # A randomly initialised layer sitting directly under the read-out destroys the
    # representation for thousands of steps, which is indistinguishable from the
    # design not working.
    extra_joint_init_from_last: bool = True
    # Where the audio blocks sit in ATTENTION position space (independent of where
    # they sit in the tensor -- the joint takes position_ids explicitly).
    #
    #   "cut"  : base = m + p, audio adjacent to its own cut. Positive, recency-
    #            ordered offsets, but the base depends on the cell, so every cell
    #            of a chunk needs its own copy of identical audio.
    #   "left" : audio occupies a FIXED slot before all text. Cell-independent, so
    #            cells of one chunk become bit-identical and can later share one
    #            block (n_cells*w -> T*w + n_cells). Implemented by shifting text
    #            right by w rather than using negative indices: RoPE depends only
    #            on relative offsets, so [audio 0..w-1][text w..] is identical to
    #            [audio -w..-1][text 0..] while keeping every index >= 0.
    audio_position_mode: str = "cut"
    # How much of the text history the joint may attend to.
    #   "full" : every key before the cut (the original design).
    #   "last" : ONE key, the most recent history token. The text stream is causal
    #            so that vector already encodes the whole prefix -- this turns the
    #            joint into an RNN-T-style joiner f(audio frames, text state) and
    #            makes the text side trivially cacheable.
    joint_text_context: str = "full"


class TwoStreamSTTModel(ScriptSTTModel):
    """SCRIPT with the band expressed as a lattice instead of as sequence length."""

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # REBUILD core_cfg against this subclass's dataclass. ScriptSTTModel hardcodes
        # `to_dataclass(ScriptSTTModelConfig, cfg)`, so without this every field added
        # by TwoStreamSTTModelConfig -- joint_layers, loss_reduction, extra_joint_layer
        # -- is simply absent, and each read falls back to its getattr() default. That
        # fails SILENTLY: ++model.joint_layers=4 was accepted by Hydra, logged in the
        # config, and ignored. Rebuilding is safe because this dataclass is a strict
        # superset; inherited fields resolve identically.
        self.core_cfg = to_dataclass(TwoStreamSTTModelConfig, cfg)
        # Derive the field list from the dataclass instead of hardcoding it, so a
        # field added later is validated and logged automatically. Hardcoding is how
        # extra_joint_layer stayed False through an entire 4-node run while the log
        # looked healthy -- the run was a duplicate of its own baseline.
        _own = [
            f.name
            for f in fields(TwoStreamSTTModelConfig)
            if f.name not in {g.name for g in fields(ScriptSTTModelConfig)}
        ]
        if not _own:
            raise RuntimeError(
                "TwoStreamSTTModelConfig adds no fields over its parent -- @dataclass is probably missing"
            )
        for _f in _own:
            if not hasattr(self.core_cfg, _f):
                raise RuntimeError(f"core_cfg is missing {_f!r} -- config plumbing is broken")
        logging.info("two-stream config | %s", "  ".join(f"{k}={getattr(self.core_cfg, k)}" for k in _own))
        self.joint_layer = None
        # Direct access, no getattr default: the loop above already proved the field
        # exists, so a default here could only hide a future plumbing break.
        if bool(self.core_cfg.extra_joint_layer):
            self._build_extra_joint_layer()

    def _build_extra_joint_layer(self) -> None:
        """Construct a dedicated, fully trainable joint layer.

        Built as a FRESH decoder layer rather than a deepcopy of the LLM's last
        one: under PEFT that last layer is LoRA-wrapped, so a copy would carry
        adapter modules and the wrong parameter names. Instead a clean layer is
        instantiated from the LLM's own config and seeded from the last layer's
        BASE weights (``.base_layer.`` stripped out of the keys).

        It lives at the top level, not inside ``self.llm``, so
        ``freeze_module(self.llm.model)`` never touches it -- it stays trainable
        without needing the prevent_freeze_params machinery, which cannot
        resurrect an already-frozen parameter anyway.
        """
        core = self._llm_core()
        last = core.layers[-1]
        layer_cls = type(last)
        llm_cfg = getattr(core, "config", None) or self.llm.config
        idx = int(getattr(getattr(last, "self_attn", last), "layer_idx", len(core.layers) - 1))
        self.joint_layer = layer_cls(llm_cfg, idx)

        if bool(getattr(self.core_cfg, "extra_joint_init_from_last", True)):
            src = {}
            for k, v in last.state_dict().items():
                if "lora_" in k:
                    continue  # adapter weights have no counterpart in a clean layer
                src[k.replace(".base_layer.", ".")] = v
            missing, unexpected = self.joint_layer.load_state_dict(src, strict=False)
            if missing:
                # Loud on purpose: a silently half-seeded joint looks like a warm
                # start while behaving like a random one.
                logging.warning("extra joint layer: %d params left at random init: %s", len(missing), missing[:8])
            logging.info(
                "extra joint layer: seeded %d/%d tensors from the LLM's last layer (%d unexpected)",
                len(src) - len(unexpected),
                len(self.joint_layer.state_dict()),
                len(unexpected),
            )

        n_p = sum(p.numel() for p in self.joint_layer.parameters())
        logging.info("extra joint layer: %s, %.2fM parameters, all trainable", layer_cls.__name__, n_p / 1e6)

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
        if getattr(self, "joint_layer", None) is not None:
            return [self.joint_layer]
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

    def _project_to_vocab(self, h: Tensor) -> Tensor:
        """Final RMSNorm, THEN the head -- the order HF's own forward uses.

        Applying lm_head to a raw layer output skips ``model.norm`` and produces
        garbage logits. Measured on Qwen3-1.7B predicting its own next token:
        4.43 nats/token with the norm, 175.37 without. That 40x was the bulk of
        the two-stream loss starting near 674 where log(V) is 11.93, and it had
        nothing to do with audio -- the same bug hits text queries identically.

        Late-layer activations here reach absmax > 1e4, so the norm is not a
        nicety; the head's weights were trained against normalised inputs.
        """
        core = self._llm_core()
        norm = getattr(core, "norm", None)
        if norm is not None:
            h = norm(h)
        return self._lm_head_of()(h)

    # ------------------------------------------------------------------
    # Text stream
    # ------------------------------------------------------------------
    def _forward_text(self, inputs_embeds: Tensor, use_cache: bool = False, past_key_values=None):
        """Run the text stream; return ``(text_h, out)``.

        With a dedicated joint layer the text stream uses ALL N LLM layers, and
        the joint's input must be the last layer's PRE-norm output -- the true
        residual stream, which is what a real layer N+1 would receive.

        A forward hook is the only honest way to get it: HF appends its final
        hidden state AFTER ``model.norm``, so ``hidden_states[-1]`` is already
        normalised. Feeding that to the joint and then normalising again inside
        ``_project_to_vocab`` would double-norm -- the same class of bug as the
        missing norm that cost 175 vs 4.43 nats, just inverted. It would also
        break the warm start, since the layer was seeded from one that expects
        un-normalised input.

        Without the extra layer, nothing changes: the joint borrows trailing LLM
        layers and the text stream reads ``hidden_states[-(n_joint+1)]``.
        """
        if getattr(self, "joint_layer", None) is not None:
            cap = {}

            def _hook(_mod, _args, output):
                cap["h"] = output[0] if isinstance(output, tuple) else output

            handle = self._llm_core().layers[-1].register_forward_hook(_hook)
            try:
                out = self._llm_forward(
                    inputs_embeds=inputs_embeds,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    return_dict=True,
                )
            finally:
                handle.remove()
            if "h" not in cap:
                raise RuntimeError("joint layer hook did not fire -- the LLM's last layer never ran")
            return cap["h"], out

        n_joint = int(getattr(self.core_cfg, "joint_layers", 1) or 1)
        if n_joint >= len(self._llm_core().layers):
            # FULL-DEPTH FUSION (the SCRIPT/two-stream hybrid). The joint IS the whole
            # stack, so the text stream's input to it is just the embeddings -- running
            # the LLM here would compute all N layers and discard every one of them.
            #
            # Text still gets all N layers: it sits *inside* the joint sequence under a
            # causal text-only mask. What it never attends to is audio -- which is the
            # point. Audio-dependent text would invalidate the text KV cache on every
            # new frame and force a full prefix recompute per chunk at decode time.
            #
            # past_key_values is meaningless here (no LLM call), so None is returned and
            # callers extend the text stream by concatenating embeddings.
            return inputs_embeds, None
        out = self._llm_forward(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,
            return_dict=True,
        )
        return out["hidden_states"][-(n_joint + 1)], out

    def _text_hidden(self, input_ids: Tensor) -> Tensor:
        """Hidden state entering the joint layers, for ``[prompt][transcript]``.

        Implemented by running the full LLM with ``output_hidden_states=True`` and
        taking the activation ``joint_layers`` from the end, rather than slicing
        the layer stack by hand. That recomputes the trailing layers on text whose
        output is unused -- a few layers over ~60 positions, which is negligible
        against what the packing saves, and it keeps this free of assumptions
        about rotary/position plumbing inside the stack.
        """
        text_h, _ = self._forward_text(self._embed_tokens(input_ids), use_cache=False)
        return text_h

    def _joint_positions(self, tu: int, w: int, bases: List[int], device) -> Tensor:
        """Attention positions for ``[text | one block per cell]``.

        Deliberately ONE function for training and decoding. The two paths build
        different sequences (a lattice vs a single growing hypothesis) but must
        agree on position semantics exactly; when they were written separately the
        agreement was a coincidence waiting to break.
        """
        n = len(bases)
        pos = torch.empty(tu + n * w, dtype=torch.long, device=device)
        ar = torch.arange(w, device=device)
        mode = str(getattr(self.core_cfg, "audio_position_mode", "cut"))
        if mode == "left":
            pos[:tu] = torch.arange(tu, device=device) + w
            for c in range(n):
                pos[tu + c * w : tu + (c + 1) * w] = ar
        elif mode == "cut":
            pos[:tu] = torch.arange(tu, device=device)
            for c, b in enumerate(bases):
                pos[tu + c * w : tu + (c + 1) * w] = int(b) + ar
        else:
            raise ValueError(f"audio_position_mode must be 'cut' or 'left', got {mode!r}")
        return pos

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
        seq, mask, read_at = build_joint_inputs(
            text_h, audio_emb, cells, prompt_len, text_context=str(self.core_cfg.joint_text_context)
        )

        # Positions: text keeps its own index; a cell's audio continues from its
        # text cutoff, mirroring SCRIPT's branch position scheme so the joint sees
        # a contiguous stretch rather than a jump.
        tu = text_h.shape[0]
        w = audio_emb.shape[1]
        bases = [prompt_len + int(cells.text_pos[c].item()) for c in range(cells.n_cells)]
        pos = self._joint_positions(tu, w, bases, seq.device)

        h = self._run_joint(seq, mask, pos)
        logits = self._project_to_vocab(h[read_at])  # (n_cells, V) -- ONLY at the cells
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
        if n_chunks <= 0:
            # Inference cannot know the chunk count in advance -- training reads it
            # off the lattice, decoding has to derive it from the audio. Passing 0
            # used to produce an EMPTY tensor and silently decode nothing.
            n_chunks = max(1, math.ceil(F / chunk_size))
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
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        audios: Tensor,
        audio_lens: Tensor,
        system_prompt: Union[str, List[str]] = "Transcribe the audio into text.",
        max_new_tokens: int = 64,
        chunk_size_override: Optional[int] = None,
        **unused,
    ) -> List[str]:
        """Chunk-synchronous decode through the two-stream path.

        MUST override the inherited one. ScriptSTTModel.generate decodes through
        the PACKED layout -- audio spliced in at layer 0, running all N layers --
        which is a different architecture from the one being trained here. Left
        inherited, validation reports the WER of a packed-SCRIPT reading of these
        weights, which on a warm-started run looks entirely plausible and measures
        nothing about two-stream.

        The loop is where the design pays off at inference:

          * the text stream is causal and audio-free, so emitting a token EXTENDS
            it -- previous positions are untouched and stay cached;
          * a chunk's audio K/V depend only on the encoder output, so they are
            built once;
          * per emission only the joint layer runs, over one audio block against
            the cached text keys.

        Batch entries are looped rather than batched. Correctness first; the
        batched form is the obvious follow-up and does not change the structure.
        """
        chunk = int(chunk_size_override or self.val_chunk_size or 14)
        core = self._llm_core()
        dev = audios.device
        outs: List[str] = []

        for b in range(int(audios.shape[0])):
            frames = self._chunk_audio(audios[b : b + 1], audio_lens[b : b + 1], chunk, 0)
            n_chunks = int(frames.shape[0]) if frames.numel() else 0

            # One prompt, or one PER UTTERANCE -- ScriptSTTModel.generate accepts
            # both and _eval_step passes a list. Taking the list straight to
            # text_to_ids raised "TextEncodeInput must be ...", which reads like a
            # tokenizer problem rather than a signature mismatch.
            prompt = system_prompt[b] if isinstance(system_prompt, (list, tuple)) else system_prompt
            # "\n" to match the packer (ScriptSTTModel.generate line ~1483 and the
            # training packer both tokenize prompt + "\n"). Without it the text
            # stream starts from a prefix training never saw.
            prompt_ids = self.tokenizer.text_to_ids(prompt + "\n")
            ids = torch.tensor(prompt_ids, dtype=torch.long, device=dev)
            # Text stream, primed once. past holds layers 1..N K/V so that
            # appending a token costs one forward over ONE position.
            th, out = self._forward_text(self._embed_tokens(ids.unsqueeze(0)), use_cache=True)
            past = out["past_key_values"] if out is not None else None
            text_h = th[0]  # (P, H)

            emitted: List[int] = []
            for t in range(n_chunks):
                audio_t = frames[t]  # (w, H)
                for _ in range(max_new_tokens):
                    seq = torch.cat([text_h, audio_t], dim=0)
                    tu, w = text_h.shape[0], audio_t.shape[0]
                    total = tu + w
                    mask = torch.zeros((total, total), dtype=torch.bool, device=dev)
                    mask[:tu, :tu] = torch.ones((tu, tu), dtype=torch.bool, device=dev).tril()
                    if str(getattr(self.core_cfg, "joint_text_context", "full")) == "last":
                        mask[tu:, tu - 1] = True  # ONE key: the most recent history token
                    else:
                        mask[tu:, :tu] = True  # audio sees the whole prefix so far
                    ar = torch.arange(w, device=dev)
                    # CAUSAL within the block: query i sees keys j <= i. The former
                    # trailing .T made this ANTI-causal, so read_at (the LAST frame)
                    # saw only itself -- 1 frame of audio instead of w.
                    mask[tu:, tu:] = ar.unsqueeze(1) >= ar.unsqueeze(0)

                    # base = tu is exactly m + (tokens emitted so far), the decode-time
                    # value of the training base m + p.
                    pos = self._joint_positions(tu, w, [tu], dev)
                    h = self._run_joint(seq, mask, pos)
                    nxt = int(self._project_to_vocab(h[-1:]).argmax(-1).item())
                    if nxt == self._eot_id:
                        break

                    emitted.append(nxt)
                    # Extend the text stream by ONE position against the cache.
                    step = torch.tensor([[nxt]], dtype=torch.long, device=dev)
                    th, out = self._forward_text(self._embed_tokens(step), use_cache=True, past_key_values=past)
                    past = out["past_key_values"] if out is not None else None
                    text_h = torch.cat([text_h, th[0]], dim=0)

            outs.append(self.tokenizer.ids_to_text(emitted) if emitted else "")
        return outs

    # ------------------------------------------------------------------
    def on_train_start(self) -> None:  # pragma: no cover - logging only
        n = int(getattr(self.core_cfg, "joint_layers", 1) or 1)
        logging.info(
            "TwoStreamSTTModel: text stream is causal and audio-free; %d trailing layer(s) form the joint. "
            "The band selects lattice cells rather than adding packed segments.",
            n,
        )
        return super().on_train_start() if hasattr(super(), "on_train_start") else None
