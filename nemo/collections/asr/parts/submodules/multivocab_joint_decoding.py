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

"""Chunk-synchronous joint decoding across several vocabularies.

``EncDecMultiVocabCHATBPEModel`` carries one encoder and N (tokenizer, decoder,
joint) heads over DIFFERENT SentencePiece vocabularies. Each head is a complete
RNN-T, so each defines its own distribution over token sequences -- but the
thing they actually agree about is TEXT, not tokens. This module decodes the
text that maximises the summed log-likelihood under all heads at once:

    W* = argmax_W  sum_k  lambda_k * log P_k( tokenizer_k(W) | audio )

which is a log-linear ensemble (product of experts) over a shared word sequence.

WHY THIS IS CHEAP HERE. Ensembling models with different tokenisations is
normally awkward: two models emit different numbers of tokens at different
times, so their scores cannot be added per step, and combining them properly
means marginalising over alignments that do not line up. CHAT sidesteps this
because its TIME AXIS IS CHUNKS. Every token a head emits inside chunk c sits
at the same time index, so the set of RNN-T alignments consistent with "chunk c
emits exactly y" has exactly ONE member. The marginal therefore collapses to a
single path,

    log P_k(y | chunk c, state) = sum_i log P_k(y_i | ...) + log P_k(blank | ...)

computable with |y|+1 joint evaluations. No forward algorithm, no lattice, and
the heads never have to agree on alignment -- only on the text of each chunk.

The heads also share chunk geometry (``_inherit_chunk_geometry`` asserts equal
chunk_size / history_chunks / frame_trim), so ONE chunking of the encoder output
is valid for every head and the expensive part -- the 609M-parameter encoder --
runs once.

EXACTNESS. Step 3 below is an exact argmax over the POOLED CANDIDATE SET, not
over all conceivable word sequences. A true global argmax would need the
intersection of N word-level lattices; pooling each head's n-best is the
standard practical stand-in, and because every head proposes, no single head's
blind spot can remove a candidate on its own.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from nemo.utils import logging

__all__ = ["MultiVocabChunkJointDecoder", "JointDecodeResult"]


@dataclass
class _Head:
    """Everything the decoder needs from one vocabulary's head."""

    decoder: torch.nn.Module
    joint: torch.nn.Module
    tokenizer: object
    blank: int
    weight: float
    name: str
    word_start: List[bool]


@dataclass
class _State:
    """A head's committed position in its own token stream."""

    dec_state: Optional[List[torch.Tensor]]
    last_token: Optional[int]  # None == start of utterance (no token consumed yet)


@dataclass
class JointDecodeResult:
    """One utterance's decode, plus the per-chunk trace used for diagnostics."""

    text: str
    chunk_texts: List[str]
    chunk_scores: List[float]
    per_head_scores: List[Dict[str, float]]
    n_candidates: List[int]


class MultiVocabChunkJointDecoder:
    """Greedy over chunks, jointly optimal over heads within each chunk.

    Args:
        model: an ``EncDecMultiVocabCHATBPEModel``.
        weights: per-head lambda_k. Defaults to uniform. ``[1, 0, 0]`` reduces
            to head 0 alone -- NOT to head 0's greedy output, though. Greedy
            stops at the first chunk-step where blank is the argmax; this
            compares complete chunk-local PATH scores, so the two can differ
            even at beam 1. Single-head here means "head 0's best path", which
            is the better decode, not the identical one.
        beam: within-chunk beam width used to PROPOSE candidates, per head.
        max_symbols: cap on tokens emitted per chunk. Defaults to the model's
            configured cap, falling back to chunk_size.
        max_candidates: ceiling on the pooled candidate set per chunk.
    """

    def __init__(
        self,
        model,
        weights: Optional[Sequence[float]] = None,
        beam: int = 4,
        max_symbols: Optional[int] = None,
        max_candidates: int = 16,
        emit_bonus: float = 0.0,
        strategy: str = "greedy",
        length_norm: bool = False,
    ):
        heads = getattr(model, "_heads", None)
        if not heads:
            raise ValueError("model has no _heads; MultiVocabChunkJointDecoder needs a multi-vocab CHAT model")

        if weights is None:
            weights = [1.0] * len(heads)
        if len(weights) != len(heads):
            raise ValueError(f"weights has {len(weights)} entries but the model has {len(heads)} heads")

        self.model = model
        self.beam = int(beam)
        self.max_candidates = int(max_candidates)
        self.emit_bonus = float(emit_bonus)
        if strategy not in ("greedy", "beam"):
            raise ValueError(f"strategy must be 'greedy' or 'beam', got {strategy!r}")
        self.strategy = strategy
        self.length_norm = bool(length_norm)
        self.heads: List[_Head] = []
        for i, h in enumerate(heads):
            # num_classes_with_blank counts the blank, and RNN-T puts it last.
            blank = int(h.joint.num_classes_with_blank) - 1
            self.heads.append(
                _Head(
                    decoder=h.decoder,
                    joint=h.joint,
                    tokenizer=h.tokenizer,
                    blank=blank,
                    weight=float(weights[i]),
                    name=str(h.name),
                    word_start=self._word_start_mask(h.tokenizer, blank),
                )
            )

        # The emission cap is a property of how much TEXT one chunk may emit,
        # not of how much audio the joint may look at -- the same distinction
        # that made the greedy path's old `x.shape[-1] // encoder_hidden` wrong
        # once history_chunks was turned on.
        if max_symbols is None:
            # Take it from the DECODING STRATEGY the greedy path uses, not from
            # the model or the joint. --max_symbols on the eval harness lands
            # there, and if the joint decoder silently used chunk_size instead
            # the two would cap emissions differently -- which is exactly the
            # confound that made fullctx SPE-1k look 1.19 WER worse than it is.
            inner = getattr(getattr(model, "decoding", None), "decoding", None)
            max_symbols = getattr(inner, "max_symbols", None)
        if max_symbols is None:
            max_symbols = getattr(model, "max_symbols", None)
        if max_symbols is None:
            max_symbols = int(getattr(self.heads[0].joint, "chunk_size", 0) or 0)
        if max_symbols <= 0:
            raise ValueError("could not determine max_symbols; pass it explicitly")
        self.max_symbols = int(max_symbols)

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _word_start_mask(tokenizer, blank: int) -> List[bool]:
        """Which token ids begin a word, so emissions can be counted in WORDS.

        SentencePiece marks a word start with U+2581. Counting words rather than
        tokens keeps the emission bonus comparable across heads: the 1k head
        spends ~23 tokens where the 4k head spends ~13 on the same sentence, so
        a per-TOKEN bonus would silently push the small-vocabulary head harder.
        """
        mask = [False] * (blank + 1)
        try:
            pieces = list(getattr(tokenizer, "vocab", []) or [])
            for i, piece in enumerate(pieces[: blank + 1]):
                mask[i] = str(piece).startswith("\u2581")
        except Exception:  # pragma: no cover - tokenizer without a vocab list
            logging.warning("could not read tokenizer vocab; emission bonus will count TOKENS, not words")
            mask = [True] * (blank + 1)
        return mask

    def _n_words(self, head: _Head, tokens: Sequence[int]) -> int:
        return sum(1 for t in tokens if head.word_start[t])

    def _pred(
        self, head: _Head, tokens: Sequence[int], state: _State
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Prediction-network outputs for every position that scores ``tokens``.

        The RNN-T recurrence is g_u = LSTM(embed(y_{u-1}), h_{u-1}): the output
        that PREDICTS y_u is driven by the token BEFORE it. So scoring
        ``tokens`` = [y_1..y_U] from a committed (last_token, state) needs the
        LSTM run over [last_token, y_1, ..., y_U], giving U+1 outputs -- the
        last of which scores the terminating blank.

        Returns g of shape [1, U+1, D]. One LSTM call, not U+1 of them.
        """
        device = next(head.decoder.parameters()).device
        if state.last_token is None:
            # Start of utterance: predict() with add_sos prepends the zero
            # vector that stands in for SOS, which is what the greedy path's
            # _SOS branch does too.
            if len(tokens) == 0:
                g, new_state = head.decoder.predict(None, state=state.dec_state, add_sos=False)
                return g, new_state
            y = torch.tensor([list(tokens)], dtype=torch.long, device=device)
            g, new_state = head.decoder.predict(y, state=state.dec_state, add_sos=True)
            return g, new_state

        y = torch.tensor([[state.last_token] + list(tokens)], dtype=torch.long, device=device)
        g, new_state = head.decoder.predict(y, state=state.dec_state, add_sos=False)
        return g, new_state

    def _advance(self, head: _Head, ids: Sequence[int], state: _State) -> _State:
        """Commit ``ids`` to this head's state.

        THE INVARIANT, inherited from the greedy decoder: ``dec_state`` has
        consumed everything up to but NOT INCLUDING ``last_token``. The greedy
        loop maintains it by storing ``hidden_prime`` -- the state produced by
        feeding the PREVIOUS token -- alongside the token just emitted.

        Getting this wrong is silent. Storing the state that has already
        consumed ids[-1] while also setting last_token=ids[-1] feeds that token
        twice, so every chunk after the first conditions on a prediction-network
        state that never occurs during training, and the decode degrades in a
        way that looks like a bad model rather than a bad decoder.
        """
        if not ids:
            return state

        if state.last_token is None:
            # START OF UTTERANCE. The label stream the prediction network sees
            # is [SOS, y1, y2, ...], where SOS is a zero EMBEDDING pushed
            # through the LSTM -- not a zero output. predict(add_sos=True)
            # prepends it for us, and _pred/_score therefore get it right, but
            # advancing the state has to consume it explicitly or the first real
            # token of the utterance is fed from a virgin state. That drops one
            # LSTM step for the whole rest of the utterance: position 0 still
            # matches, everything after it drifts.
            _g, base = head.decoder.predict(None, state=state.dec_state, add_sos=False)
            consume = list(ids[:-1])
        else:
            base = state.dec_state
            consume = [state.last_token] + list(ids[:-1])

        if not consume:
            return _State(base, int(ids[-1]))
        device = next(head.decoder.parameters()).device
        y = torch.tensor([consume], dtype=torch.long, device=device)
        _g, new_state = head.decoder.predict(y, state=base, add_sos=False)
        return _State(new_state, int(ids[-1]))

    def _logp(self, head: _Head, f: torch.Tensor, f_len: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Log-probabilities at every (chunk, prediction-position) pair.

        ALWAYS log-normalised. The greedy path leaves logits unnormalised on
        GPU because an argmax does not care; summing scores ACROSS HEADS does,
        so an unnormalised joint here would silently weight the heads by their
        logit scale rather than by lambda_k.
        """
        n = g.shape[0]
        f_rep = f.expand(n, -1, -1) if f.shape[0] == 1 else f
        len_rep = f_len.expand(n, -1) if f_len.shape[0] == 1 else f_len
        logits = head.joint.joint(f_rep, g, len_rep)  # [n, 1, U+1, V+1]
        return logits.float().log_softmax(dim=-1)[:, 0]  # [n, U+1, V+1]

    # -------------------------------------------------------------- proposals

    @torch.no_grad()
    def _propose(self, head: _Head, f: torch.Tensor, f_len: torch.Tensor, state: _State) -> List[List[int]]:
        """Within-chunk beam search; returns token sequences, best first.

        A chunk's emission is "some tokens, then blank", so at every step each
        live hypothesis also spawns a FINISHED one by taking blank here. That is
        the whole of the chunk-local path space.

        Each hypothesis CARRIES ITS OWN prediction-network state and extends it
        one step at a time. The obvious alternative -- re-running the LSTM over
        the whole prefix from the committed state at every step -- is O(U) per
        hypothesis per step, which at beam 4 and max_symbols 14 is ~56 LSTM runs
        per head per chunk instead of 56 single steps. On a full leaderboard set
        that is the difference between a decode that finishes and one that does
        not.
        """
        live: List[Tuple[List[int], float, _State]] = [([], 0.0, state)]
        finished: List[Tuple[List[int], float]] = []

        for _ in range(self.max_symbols):
            if not live:
                break
            # One joint call covers every live hypothesis; each _pred is now a
            # single LSTM step because the state is carried, not rebuilt.
            gs = [self._pred(head, [], st)[0][:, -1:] for _tok, _sc, st in live]
            logp = self._logp(head, f, f_len, torch.cat(gs, dim=0))[:, 0]  # [n_live, V+1]

            expansions: List[Tuple[List[int], float, int, int]] = []
            for i, (tokens, score, _st) in enumerate(live):
                row = logp[i]
                finished.append((tokens, score + float(row[head.blank])))
                row_no_blank = row.clone()
                row_no_blank[head.blank] = float("-inf")
                top_v, top_k = row_no_blank.topk(min(self.beam, row_no_blank.numel()))
                for v, k in zip(top_v.tolist(), top_k.tolist()):
                    if v == float("-inf"):
                        continue
                    expansions.append((tokens + [int(k)], score + v, i, int(k)))

            expansions.sort(key=lambda p: p[1], reverse=True)
            # Advance ONLY the survivors: _advance is an LSTM step, so doing it
            # for every expansion would undo the saving above.
            live = [
                (tokens, score, self._advance(head, [tok], live[parent][2]))
                for tokens, score, parent, tok in expansions[: self.beam]
            ]

        # Hypotheses still alive at the cap never emitted blank. Charge them the
        # blank they would have had to emit, so every returned sequence is a
        # complete chunk-local path and the scores stay comparable.
        for tokens, score, st in live:
            g, _ = self._pred(head, [], st)
            logp = self._logp(head, f, f_len, g[:, -1:])[0, 0]
            finished.append((tokens, score + float(logp[head.blank])))

        finished.sort(key=lambda p: p[1], reverse=True)
        return [tokens for tokens, _ in finished[: self.beam]]

    # ------------------------------------------------- chunk-synchronous greedy

    def _greedy_chunk(self, head: _Head, f, f_len, state: _State):
        """This head's OWN greedy emission for this chunk.

        Emit while the argmax over V+1 is a token; stop at blank. Byte for byte
        the rule ``_greedy_decode_chat`` uses, which is what makes ``weights=[1,
        0, 0]`` reduce to plain head-0 greedy and gives us a hard equivalence
        test.

        The emit-or-stop decision is LOCAL to each step, so nothing here ever
        compares emissions of different length -- which is precisely the trap
        the beam strategy fell into (it scored whole chunk-emissions as
        sequences, so every extra token multiplied in another probability < 1
        and silence won; 16% of words were deleted).

        Returns (tokens, score_tokens, score_with_blank). The first score sums
        only the emitted tokens, as the greedy decoder's own hypothesis.score
        does; the second adds the terminating blank, making it the probability
        of "this chunk emits exactly these tokens".
        """
        tokens: List[int] = []
        score = 0.0
        st = state
        for _ in range(self.max_symbols):
            g, _ = self._pred(head, [], st)
            lp = self._logp(head, f, f_len, g[:, -1:])[0, 0]
            v, k = lp.max(0)
            k = int(k)
            if k == head.blank:
                return tokens, score, score + float(v)
            tokens.append(k)
            score += float(v)
            st = self._advance(head, [k], st)
        # Hit the cap without ever choosing blank. Charge the blank it would
        # have had to emit so the score stays a comparable path probability.
        g, _ = self._pred(head, [], st)
        lp = self._logp(head, f, f_len, g[:, -1:])[0, 0]
        return tokens, score, score + float(lp[head.blank])

    # ---------------------------------------------------------------- scoring

    @torch.no_grad()
    def _score(
        self, head: _Head, candidates: List[List[int]], f: torch.Tensor, f_len: torch.Tensor, state: _State
    ) -> List[float]:
        """log P_k of each candidate token sequence for this chunk.

        Batched across candidates: pad to the longest, run ONE LSTM pass and ONE
        joint call, then gather each candidate's own positions. Padding is
        scored but never read.
        """
        if not candidates:
            return []
        device = next(head.decoder.parameters()).device
        lens = [len(c) for c in candidates]
        u_max = max(lens)

        rows = []
        for c in candidates:
            prefix = [] if state.last_token is None else [state.last_token]
            # Pad with blank; those positions are masked out of the sum below.
            rows.append(prefix + list(c) + [head.blank] * (u_max - len(c)))
        y = torch.tensor(rows, dtype=torch.long, device=device)

        if state.last_token is None:
            g, _ = head.decoder.predict(
                y if u_max else None, state=self._batch_state(head, state, len(candidates)), add_sos=True
            )
            if u_max == 0:
                g = g[:, :1]
        else:
            g, _ = head.decoder.predict(y, state=self._batch_state(head, state, len(candidates)), add_sos=False)

        logp = self._logp(head, f, f_len, g)  # [n_cand, U+1, V+1]

        out = []
        for i, c in enumerate(candidates):
            total = 0.0
            for u, tok in enumerate(c):
                total += float(logp[i, u, tok])
            total += float(logp[i, len(c), head.blank])
            out.append(total)
        return out

    @staticmethod
    def _batch_state(head: _Head, state: _State, n: int) -> Optional[List[torch.Tensor]]:
        """Replicate one committed LSTM state across n candidates."""
        if state.dec_state is None:
            return None
        return [s.expand(-1, n, -1).contiguous() for s in state.dec_state]

    # ----------------------------------------------------------------- decode

    @torch.no_grad()
    def decode(self, encoded: torch.Tensor, encoded_len: torch.Tensor) -> List[JointDecodeResult]:
        """Joint-decode a batch of encoder outputs.

        Args:
            encoded: [B, D, T] as returned by ``model.forward``.
            encoded_len: [B].
        """
        chunk_fn = getattr(self.heads[0].joint, "chunk_encoder_for_decoding", None)
        if chunk_fn is None:
            raise ValueError("head 0's joint has no chunk_encoder_for_decoding; not a CHAT attention joint")

        # Chunk ONCE. Legal for every head only because _inherit_chunk_geometry
        # asserts they share chunk_size / history_chunks / frame_trim.
        #
        # chunk_encoder_for_decoding takes the encoder's own [B, D, T] and
        # returns [B, D', n_chunks] with D' = window_frames * D. The decoding
        # loop wants one chunk vector at a time, so transpose to [B, n_chunks,
        # D'] exactly as rnnt_decoder_predictions_tensor does.
        chunked, _num_chunks, chunk_frame_lengths = chunk_fn(encoded, encoded_len)
        chunked = chunked.transpose(1, 2)

        results = []
        for b in range(chunked.shape[0]):
            results.append(self._decode_one(chunked[b].unsqueeze(1), chunk_frame_lengths[b : b + 1]))
        return results

    def _decode_one(self, x: torch.Tensor, chunk_lens: torch.Tensor) -> JointDecodeResult:
        """x: [n_chunks, 1, D_window];  chunk_lens: [1, n_chunks]."""
        n_chunks_total = chunk_lens.shape[1]
        states = [_State(dec_state=None, last_token=None) for _ in self.heads]

        # DETOKENISE ONCE, AT THE END -- never per chunk. ids_to_text drops the
        # leading word-boundary marker, so gluing per-chunk strings together
        # welds the last word of one chunk to the first word of the next:
        # "If we look into" + "take this forward" -> "If welook intotake this
        # forward". Every word is correct and the WER is still 58-74%, because a
        # ~9-chunk utterance loses ~8 word boundaries. Accumulating ids and
        # decoding once is also exactly what the greedy path does, which keeps
        # the two directly comparable.
        ref_head = next((i for i, h in enumerate(self.heads) if h.weight != 0.0), 0)
        ref_ids: List[int] = []

        chunk_texts: List[str] = []
        chunk_scores: List[float] = []
        per_head: List[Dict[str, float]] = []
        n_cands: List[int] = []

        for t in range(n_chunks_total):
            if int(chunk_lens[0, t]) == 0:
                break
            f = x.narrow(dim=0, start=t, length=1)  # [1, 1, D]
            f_len = chunk_lens[:, t : t + 1]  # [1, 1]

            if self.strategy == "greedy":
                # CHUNK-SYNCHRONOUS GREEDY. Every head decodes this chunk with
                # its OWN greedy rule, then the best-scoring head's TEXT is
                # adopted by all of them. One decision per chunk, N greedy
                # decodes, no pool and no cross-scoring -- so the cost is N x
                # greedy rather than the beam strategy's ~40x.
                best_i, best_sc, best_toks = None, None, None
                head_scores: Dict[str, float] = {}
                for i, (head, state) in enumerate(zip(self.heads, states)):
                    if head.weight == 0.0:
                        continue
                    toks, s_tok, s_full = self._greedy_chunk(head, f, f_len, state)
                    sc = s_full
                    if self.length_norm and toks:
                        # Heads spend different numbers of tokens on the same
                        # words, so the raw path score favours whichever head
                        # segments most coarsely. Optional, off by default.
                        sc = sc / (len(toks) + 1)
                    sc *= head.weight
                    head_scores[head.name] = sc
                    if best_sc is None or sc > best_sc:
                        best_i, best_sc, best_toks = i, sc, toks
                if best_i is None:
                    raise ValueError("every head has weight 0; nothing to decode with")
                won = self.heads[best_i].tokenizer.ids_to_text(best_toks)
                n_cands.append(sum(1 for h in self.heads if h.weight != 0.0))
            else:
                # --- 1/2. propose per head, pool as TEXT -----------------------
                # Text is the only representation the heads share. Pooling on
                # token ids would be meaningless -- id 412 is a different piece
                # in each vocabulary.
                texts: Dict[str, None] = {"": None}
                for head, state in zip(self.heads, states):
                    if head.weight == 0.0:
                        continue
                    for tokens in self._propose(head, f, f_len, state):
                        texts.setdefault(head.tokenizer.ids_to_text(tokens), None)
                        if len(texts) >= self.max_candidates:
                            break
                cands = list(texts.keys())
                n_cands.append(len(cands))

                # --- 3. score every candidate under every head -----------------
                totals = [0.0] * len(cands)
                head_scores = {}
                for head, state in zip(self.heads, states):
                    if head.weight == 0.0:
                        continue
                    retok = [head.tokenizer.text_to_ids(w) for w in cands]
                    scores = self._score(head, retok, f, f_len, state)
                    for i, sc in enumerate(scores):
                        totals[i] += head.weight * sc
                    head_scores[head.name] = max(scores) if scores else 0.0
                best = max(range(len(cands)), key=lambda i: totals[i])
                won = cands[best]
                best_sc = totals[best]
                best_i, best_toks = None, None

            # --- commit: every head re-encodes the winning TEXT ---------------
            chunk_texts.append(won)
            chunk_scores.append(float(best_sc))
            per_head.append(head_scores)

            # THE WINNING HEAD CARRIES ITS OWN TOKEN IDS. Re-encoding its text
            # (text_to_ids(ids_to_text(t))) is NOT the identity -- unknown pieces
            # and repeated subwords come back different, so the head would be fed
            # a token stream it never emitted and drift from the next chunk on.
            # Only the OTHER heads have to re-encode, because the text is the
            # only thing they can consume.
            for i, (head, state) in enumerate(zip(self.heads, states)):
                if self.strategy == "greedy" and i == best_i:
                    ids = best_toks
                else:
                    ids = head.tokenizer.text_to_ids(won)
                if i == ref_head:
                    ref_ids.extend(ids)
                states[i] = self._advance(head, ids, state)

        return JointDecodeResult(
            text=self.heads[ref_head].tokenizer.ids_to_text(ref_ids),
            chunk_texts=chunk_texts,
            chunk_scores=chunk_scores,
            per_head_scores=per_head,
            n_candidates=n_cands,
        )
