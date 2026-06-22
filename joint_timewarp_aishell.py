"""
Joint time-warp decoding for the Mandarin CHAT (chunk-wise attention transducer).

Idea (per user design):
    * Decode several time-warped copies of the SAME utterance simultaneously with
      ONE model and a SHARED prediction-network (decoder) state.
    * The CHAT model is chunk-synchronous: the decode time index is a *chunk*
      index, and blank advances it by one chunk.  We keep one chunk counter per
      warp factor; all start at 0.
    * Each joint step:
        1. For each (active) factor, compute its log-prob distribution at its
           current chunk using the shared decoder output.
        2. While any active factor's argmax is blank, advance that factor's chunk
           counter and recompute (per-factor blank advancement).
        3. When no active factor prefers blank, log-softmax each factor's
           distribution and default to the PRIMARY (x1.0) stream's top token.  Only
           switch to a warped stream's top token if that stream's top log-prob beats
           the primary's top log-prob by at least ``epsilon`` (and is the best such
           stream) -- an epsilon-biased most-confident rule (epsilon=0 reduces to
           pure most-confident with primary preference on ties).  The emitted token
           updates the shared decoder state.  Since each factor's own top class is
           non-blank after step 2, the winner is non-blank by construction (a
           defensive blank warning remains, but never fires).
    * Termination: stop as soon as the PRIMARY (x1.0) stream reaches its end.
      Non-primary factors that exhaust earlier simply drop out of the combination.

References are used only to report CER (and an oracle diagnostic), never for
decoding.  This complements ``likelihood_timewarp_aishell.py`` (which picks one
independent per-factor hypothesis) by instead fusing the streams frame-by-frame.
"""
import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile
import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import eval_aishell_cer as A  # noqa: E402
import oracle_timewarp_aishell as O  # noqa: E402
import run_eval_asr as R  # noqa: E402
from oracle_timewarp_eval import warp_audio  # noqa: E402

from nemo.collections.asr.metrics.wer import word_error_rate  # noqa: E402


# --------------------------------------------------------------------------- #
# Model-side helpers
# --------------------------------------------------------------------------- #
def _blank_id(model):
    decoding = getattr(model, "decoding", None)
    for obj in (decoding, getattr(decoding, "decoding", None)):
        bid = getattr(obj, "blank_id", None)
        if bid is None:
            bid = getattr(obj, "_blank_index", None)
        if bid is not None:
            return int(bid)
    # Fall back to joint vocabulary size (blank is the last class in NeMo RNNT).
    joint = getattr(model, "joint", None)
    n = getattr(joint, "num_classes_with_blank", None)
    if n is not None:
        return int(n) - 1
    raise RuntimeError("could not determine blank id from model")


def _tokens_to_text(model, token_ids):
    if not token_ids:
        return ""
    ids = [int(x) for x in token_ids]
    # Char-based RNNT (this CHAT model) exposes decode_tokens_to_str via decoding;
    # BPE models expose a tokenizer.ids_to_text.  Try both, in that order.
    decoding = getattr(model, "decoding", None)
    if decoding is not None and hasattr(decoding, "decode_tokens_to_str"):
        try:
            return decoding.decode_tokens_to_str(ids).strip()
        except Exception:
            pass
    tok = getattr(model, "tokenizer", None)
    if tok is not None:
        try:
            return tok.ids_to_text(ids).strip()
        except Exception:
            pass
    voc = getattr(getattr(model, "joint", None), "vocabulary", None)
    if voc is not None:
        return "".join(voc[i] for i in ids if 0 <= i < len(voc))
    return "".join(str(x) for x in ids)


@torch.inference_mode()
def encode_warps(model, audio, factors, method, device):
    """Warp ``audio`` by every factor, encode as one batch, return per-factor
    projected chunk encodings + chunk frame-lengths + chunk counts.

    Returns lists indexed like ``factors``:
        enc_proj[f]   : [1, T_chunks_max, H_joint_enc]   (projected encoder output)
        chunk_lens[f] : [1, T_chunks_max]                (valid frames per chunk)
        num_chunks[f] : int                              (valid chunk count)
    """
    signals, lengths = [], []
    for fct in factors:
        y = warp_audio(np.asarray(audio, dtype=np.float32), fct, method)
        signals.append(torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)))
        lengths.append(y.shape[0])
    max_len = max(lengths)
    batch = torch.zeros(len(signals), max_len, dtype=torch.float32)
    for i, sig in enumerate(signals):
        batch[i, : sig.numel()] = sig
    batch = batch.to(device)
    length_t = torch.tensor(lengths, dtype=torch.long, device=device)

    encoded, encoded_len = model.forward(input_signal=batch, input_signal_length=length_t)
    # CHAT chunking: encoded [B, D, T] -> chunked [B, chunk*D, n_chunks]
    chunked, num_chunks, chunk_lens = model.joint.chunk_encoder_for_decoding(encoded, encoded_len)
    chunked = chunked.transpose(1, 2)  # [B, n_chunks, chunk*D]

    enc_proj_list, chunk_lens_list, num_chunks_list = [], [], []
    for i in range(len(factors)):
        enc_f = chunked[i : i + 1]  # [1, n_chunks, chunk*D]
        enc_proj_list.append(model.joint.project_encoder(enc_f))  # [1, n_chunks, H]
        chunk_lens_list.append(chunk_lens[i : i + 1])  # [1, n_chunks]
        num_chunks_list.append(int(num_chunks[i].item()))
    return enc_proj_list, chunk_lens_list, num_chunks_list


# Fusion strategies for combining the per-stream distributions at each step.
# Each receives ``lp`` (dict: factor_idx -> log-softmax vector [V+1]) for the
# active streams and returns the chosen token id (may be blank for the additive
# rules; the caller treats a blank win as a joint-blank step).
FUSION_MODES = ("max", "sum", "mean_prob", "entropy", "wsum_conf", "topk_rescore",
                "agree_gate", "borda", "conf_drop_sum")


def _fuse_step(lp, active, primary_idx, blank_id, fusion, epsilon):
    if fusion == "max":
        # Epsilon-biased most-confident: default to primary's top token; switch to a
        # warped stream only if its top log-prob beats primary's by >= epsilon.
        pval, pidx = lp[primary_idx].max(dim=-1)
        score_1 = float(pval.item())
        best, best_val = int(pidx.item()), score_1
        for f in active:
            if f == primary_idx:
                continue
            val, idx = lp[f].max(dim=-1)
            val = float(val.item())
            if val >= score_1 + epsilon and val > best_val:
                best_val, best = val, int(idx.item())
        return best
    if fusion == "sum":
        # Product-of-experts: sum log-probs across streams, argmax (blank allowed).
        comb = None
        for f in active:
            comb = lp[f] if comb is None else comb + lp[f]
        return int(comb.argmax().item())
    if fusion == "mean_prob":
        # Mixture-of-experts: sum probabilities across streams, argmax.
        comb = None
        for f in active:
            p = lp[f].exp()
            comb = p if comb is None else comb + p
        return int(comb.argmax().item())
    if fusion == "entropy":
        # Trust the whole-distribution most-confident (lowest-entropy) stream.
        best_f, best_ent = None, None
        for f in active:
            ent = float(-(lp[f].exp() * lp[f]).sum().item())
            if best_ent is None or ent < best_ent:
                best_ent, best_f = ent, f
        return int(lp[best_f].argmax().item())
    if fusion == "wsum_conf":
        # Confidence-weighted log-prob sum: peakier streams (higher top prob) weigh more.
        comb = None
        for f in active:
            w = float(lp[f].max().exp().item())
            term = w * lp[f]
            comb = term if comb is None else comb + term
        return int(comb.argmax().item())
    if fusion == "topk_rescore":
        # Primary proposes top-k candidates; rescore them by the summed log-probs
        # across streams. Keeps warped streams from injecting off-list tokens.
        k = min(8, lp[primary_idx].numel())
        topk = lp[primary_idx].topk(k).indices
        comb = None
        for f in active:
            comb = lp[f] if comb is None else comb + lp[f]
        sub = comb[topk]
        return int(topk[int(sub.argmax().item())].item())
    if fusion == "agree_gate":
        # High-precision correction: keep primary's token unless ALL warped streams
        # agree on the same different token AND each is more confident than primary.
        p_idx = int(lp[primary_idx].argmax().item())
        p_top = float(lp[primary_idx].max().item())
        others = [f for f in active if f != primary_idx]
        if len(others) >= 1:
            o_idx = [int(lp[f].argmax().item()) for f in others]
            o_top = [float(lp[f].max().item()) for f in others]
            if all(o == o_idx[0] for o in o_idx) and o_idx[0] != p_idx and min(o_top) > p_top:
                return o_idx[0]
        return p_idx
    if fusion == "borda":
        # Rank fusion over the union of per-stream top-k: each stream awards
        # (k-1..0) points to its top-k tokens; pick the highest total, tie-break by
        # summed log-prob. Robust to per-stream log-prob scale differences.
        k = min(5, lp[primary_idx].numel())
        points, lpsum = {}, {}
        for f in active:
            vals, idxs = lp[f].topk(k)
            for rank, (v, tok) in enumerate(zip(vals.tolist(), idxs.tolist())):
                points[tok] = points.get(tok, 0) + (k - 1 - rank)
                lpsum[tok] = lpsum.get(tok, 0.0) + v
        best_tok = max(points, key=lambda tk: (points[tk], lpsum[tk]))
        return int(best_tok)
    if fusion == "conf_drop_sum":
        # Product-of-experts, but drop streams less confident than primary (their
        # frame is likely misaligned/uncertain), so only trustworthy streams vote.
        p_top = float(lp[primary_idx].max().item())
        comb = lp[primary_idx].clone()
        for f in active:
            if f == primary_idx:
                continue
            if float(lp[f].max().item()) >= p_top:
                comb = comb + lp[f]
        return int(comb.argmax().item())
    raise ValueError(f"unknown fusion mode: {fusion}")


# Hypothesis-level reranking strategies (operate on whole per-factor hypotheses
# rather than per-step distributions). Handled in main(), not _fuse_step.
RERANK_MODES = ("rerank_own", "rerank_cross_sum", "rerank_cross_pertok")
ROVER_MODES = ("rover",)  # confidence-weighted hypothesis voting (can exceed oracle-of-N)
_LEFTOVER_PENALTY = 5.0  # nats subtracted per hyp token a stream cannot place


@torch.inference_mode()
def forced_sequence_score(model, enc_proj, chunk_lens, num_chunks, tokens,
                          blank_id, max_symbols):
    """Greedy forced-alignment score of ``tokens`` under one stream's encoder.

    Follows the model's blank (time-advance) decisions but forces every emission
    to be the next target token, accumulating that token's log-prob. Returns
    ``(total_logprob, placed, leftover)`` where ``placed`` is how many target
    tokens were emitted before frames ran out and ``leftover = len(tokens)-placed``.
    A large ``leftover`` means this stream cannot explain the hypothesis.
    """
    device = enc_proj.device
    joint, decoder = model.joint, model.decoder
    state = decoder.initialize_state(enc_proj)
    label = torch.full((1, 1), blank_id, dtype=torch.long, device=device)
    dec_out, state = decoder.predict(label, state, add_sos=False, batch_size=1)[:2]
    dec_proj = joint.project_prednet(dec_out)

    n = len(tokens)
    t = idx = 0
    score = 0.0
    emitted_since_adv = 0
    while idx < n and t < num_chunks:
        enc = enc_proj[:, t : t + 1]
        flen = chunk_lens[:, t : t + 1]
        logits = joint.joint_after_projection(enc, dec_proj, flen)
        lp = torch.log_softmax(logits[0, 0, 0].float(), dim=-1)
        if int(lp.argmax().item()) == blank_id:
            t += 1
            emitted_since_adv = 0
            continue
        tgt = int(tokens[idx])
        score += float(lp[tgt].item())
        idx += 1
        emitted_since_adv += 1
        label = torch.tensor([[tgt]], dtype=torch.long, device=device)
        dec_out, state = decoder.predict(label, state, add_sos=False, batch_size=1)[:2]
        dec_proj = joint.project_prednet(dec_out)
        if emitted_since_adv >= max_symbols:
            t += 1
            emitted_since_adv = 0
    return score, idx, n - idx


@torch.inference_mode()
def joint_greedy_decode(model, enc_proj_list, chunk_lens_list, num_chunks_list,
                        blank_id, max_symbols, primary_idx, factor_tags, epsilon=0.0,
                        fusion="max"):
    """Shared-state chunk-synchronous joint greedy decode across warp streams.

    ``factor_tags`` is only used for readable warning messages.  Returns
    ``(token_ids, stats, token_logprobs)`` where ``stats`` records joint-blank and
    max-symbol events and ``token_logprobs`` is the per-emitted-token log-prob
    (the winning stream's log-prob of that token) for ROVER confidence weighting.
    """
    device = enc_proj_list[0].device
    n_factors = len(enc_proj_list)
    joint = model.joint
    decoder = model.decoder

    # Shared prediction network: one hypothesis history for all streams.
    state = decoder.initialize_state(enc_proj_list[0])
    label = torch.full((1, 1), blank_id, dtype=torch.long, device=device)
    dec_out, state = decoder.predict(label, state, add_sos=False, batch_size=1)[:2]
    dec_proj = joint.project_prednet(dec_out)  # [1, 1, H]

    t = [0] * n_factors
    tokens = []
    token_logprobs = []
    symbols_since_advance = 0
    stats = {"joint_blank": 0, "max_symbol_forces": 0, "steps": 0}

    def factor_logprob(f):
        ti = t[f]
        enc = enc_proj_list[f][:, ti : ti + 1]              # [1, 1, H]
        flen = chunk_lens_list[f][:, ti : ti + 1]            # [1, 1]
        logits = joint.joint_after_projection(enc, dec_proj, flen)  # [1, 1, 1, V+1] (raw logits)
        return torch.log_softmax(logits[0, 0, 0].float(), dim=-1)    # [V+1]

    while t[primary_idx] < num_chunks_list[primary_idx]:
        stats["steps"] += 1
        # ---- per-factor blank advancement ----
        lp = {}
        for f in range(n_factors):
            while t[f] < num_chunks_list[f]:
                cur = factor_logprob(f)
                if int(cur.argmax().item()) == blank_id:
                    t[f] += 1
                    symbols_since_advance = 0
                else:
                    lp[f] = cur
                    break
        # primary reached the end during blank advancement -> stop
        if t[primary_idx] >= num_chunks_list[primary_idx]:
            break

        active = [f for f in range(n_factors) if f in lp]
        if not active:
            # every active factor exhausted while seeking non-blank
            break

        # ---- fuse the active streams' distributions into one token ----
        best = _fuse_step(lp, active, primary_idx, blank_id, fusion, epsilon)

        if best == blank_id:
            stats["joint_blank"] += 1
            tags = ",".join(factor_tags[f] for f in active)
            print(f"    [warn] joint argmax is BLANK at chunks "
                  f"{{{','.join(f'{factor_tags[f]}:{t[f]}' for f in active)}}} "
                  f"(active={tags}); advancing all active streams.")
            for f in active:
                t[f] += 1
            symbols_since_advance = 0
            continue

        # ---- emit token, update shared decoder state ----
        tokens.append(best)
        token_logprobs.append(max(float(lp[f][best].item()) for f in active))
        symbols_since_advance += 1
        label = torch.tensor([[best]], dtype=torch.long, device=device)
        dec_out, state = decoder.predict(label, state, add_sos=False, batch_size=1)[:2]
        dec_proj = joint.project_prednet(dec_out)

        # ---- max-symbols safeguard (avoid infinite emission at fixed chunks) ----
        if symbols_since_advance >= max_symbols:
            stats["max_symbol_forces"] += 1
            for f in active:
                if t[f] < num_chunks_list[f]:
                    t[f] += 1
            symbols_since_advance = 0

    return tokens, stats, token_logprobs


def rover_fuse(hyps, confs, primary_pos):
    """Confidence-weighted ROVER over per-factor token-id hypotheses.

    ``hyps`` is a list of token-id lists, ``confs`` the matching per-token log-prob
    lists, ``primary_pos`` the index of the x1.0 hypothesis (used as the alignment
    skeleton and as the tie-break/abstain anchor).  Each non-primary hyp is aligned
    to the primary with a simple edit-distance backtrace; at every primary position
    the substituted/matched tokens from all hyps vote with weight ``exp(logprob)``
    (a confidence in [0,1]), and the highest-weight token is emitted.  Insertions
    relative to the primary are ignored (kept conservative); deletions let the other
    streams out-vote a low-confidence primary token, which can drop a spurious char.

    Unlike argmax selection this can beat oracle-of-N by mixing correct characters
    from different streams.  Returns the fused token-id list.
    """
    import math

    prim = hyps[primary_pos]
    prim_conf = confs[primary_pos]
    Lp = len(prim)
    # votes[p] : dict token_id -> summed confidence; del_votes[p] : confidence mass
    # arguing position p should be deleted.
    votes = [dict() for _ in range(Lp)]
    del_votes = [0.0] * Lp
    for p in range(Lp):
        votes[p][prim[p]] = votes[p].get(prim[p], 0.0) + math.exp(prim_conf[p])

    for hi, hyp in enumerate(hyps):
        if hi == primary_pos:
            continue
        c = confs[hi]
        # edit-distance DP between prim (rows) and hyp (cols), then backtrace.
        m, k = Lp, len(hyp)
        dp = [[0] * (k + 1) for _ in range(m + 1)]
        for r in range(m + 1):
            dp[r][0] = r
        for col in range(k + 1):
            dp[0][col] = col
        for r in range(1, m + 1):
            for col in range(1, k + 1):
                cost = 0 if prim[r - 1] == hyp[col - 1] else 1
                dp[r][col] = min(dp[r - 1][col] + 1, dp[r][col - 1] + 1, dp[r - 1][col - 1] + cost)
        r, col = m, k
        while r > 0 and col > 0:
            cost = 0 if prim[r - 1] == hyp[col - 1] else 1
            if dp[r][col] == dp[r - 1][col - 1] + cost:       # match/substitution -> vote at pos r-1
                tok = hyp[col - 1]
                votes[r - 1][tok] = votes[r - 1].get(tok, 0.0) + math.exp(c[col - 1])
                r -= 1
                col -= 1
            elif dp[r][col] == dp[r - 1][col] + 1:            # deletion (prim char absent in hyp)
                del_votes[r - 1] += math.exp(c[col - 1]) if col - 1 < len(c) else 0.5
                r -= 1
            else:                                              # insertion in hyp -> ignored
                col -= 1
        while r > 0:                                           # trailing prim chars unmatched
            del_votes[r - 1] += 0.5
            r -= 1

    out = []
    for p in range(Lp):
        best_tok = max(votes[p], key=votes[p].get)
        if del_votes[p] > votes[p][best_tok]:
            continue  # streams collectively argue this position should be dropped
        out.append(best_tok)
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main(args):
    factors = [float(x) for x in args.factors.split(",") if x.strip()]
    if 1.0 not in factors:
        factors = [1.0] + factors
    factors = sorted(set(factors))
    primary_idx = factors.index(1.0)
    factor_tags = [f"x{f}" for f in factors]

    torch.set_float32_matmul_precision("medium")
    device = torch.device(
        f"cuda:{args.device}"
        if (args.device is not None and args.device >= 0 and torch.cuda.is_available())
        else "cpu"
    )

    model, is_ms = R.load_model(args.model, device, tokenizer_dir=args.tokenizer_dir)
    if is_ms:
        sys.exit("joint_timewarp_aishell.py expects a single-stream CHAT/RNNT model, not a multistream model.")
    if not hasattr(model, "joint") or not hasattr(model.joint, "chunk_encoder_for_decoding"):
        sys.exit("model.joint has no chunk_encoder_for_decoding; this script targets the CHAT (RNNTAttJoint) model.")
    if hasattr(model, "use_cer"):
        model.use_cer = True

    blank_id = _blank_id(model)
    max_symbols = args.max_symbols_per_step
    if max_symbols is None:
        max_symbols = int(getattr(getattr(model.cfg, "decoding", {}), "greedy", {}).get("max_symbols", 10)) \
            if hasattr(model, "cfg") else 10

    # Force raw logits out of the joint so we control the log-softmax ourselves.
    saved_log_softmax = getattr(model.joint, "log_softmax", None)
    model.joint.log_softmax = False

    items = O.load_aishell_manifest(args)
    n = len(items)
    if n == 0:
        print("ERROR: nothing to evaluate (check --audio_src_prefix/--audio_dst_prefix).")
        sys.exit(1)

    refs = [A.normalize(it["ref_raw"], args.keep_spaces) for it in items]
    set_name = args.set_name or os.path.splitext(os.path.basename(args.manifest))[0]

    # Which strategies to evaluate (all in one encode-once pass).
    all_modes = list(FUSION_MODES) + list(RERANK_MODES) + list(ROVER_MODES)
    if args.fusion.strip().lower() == "all":
        requested = all_modes
    else:
        requested = [m.strip() for m in args.fusion.split(",") if m.strip()]
        for m in requested:
            if m not in all_modes:
                sys.exit(f"unknown mode {m!r}; choose from {all_modes} or 'all'.")
    step_modes = [m for m in requested if m in FUSION_MODES]
    rerank_modes = [m for m in requested if m in RERANK_MODES]
    rover_modes = [m for m in requested if m in ROVER_MODES]
    need_forced = bool(rerank_modes)
    need_confs = bool(rover_modes)

    print(
        f"Loaded {n} utterances; joint-decoding warps {factors} (primary=x1.0) "
        f"method={args.method!r}, blank_id={blank_id}, max_symbols={max_symbols}, "
        f"epsilon={args.score_epsilon}, modes={requested}."
    )

    def _pick_factor(scores_by_factor):
        best = max(scores_by_factor.values())
        tied = [f for f, s in scores_by_factor.items() if s == best]
        return 1.0 if 1.0 in tied else min(tied, key=lambda f: abs(f - 1.0))

    joint_preds = {m: [] for m in requested}
    joint_stats = {m: {"joint_blank": 0, "max_symbol_forces": 0} for m in step_modes}
    per_factor_preds = {f: [] for f in factors}
    t0 = time.time()

    try:
        for i in tqdm(range(n), desc="joint time-warp"):
            y, sr = soundfile.read(items[i]["audio"], dtype="float32", always_2d=False)
            if y.ndim == 2:
                y = y.mean(axis=1)
            enc_proj, chunk_lens, num_chunks = encode_warps(model, y, factors, args.method, device)

            # Per-factor independent greedy (single stream) -> baseline + oracle.
            per_factor_tokens = {}
            per_factor_confs = {}
            for fi, fct in enumerate(factors):
                toks, _, tlp = joint_greedy_decode(
                    model, [enc_proj[fi]], [chunk_lens[fi]], [num_chunks[fi]],
                    blank_id, max_symbols, 0, [factor_tags[fi]],
                )
                per_factor_tokens[fct] = toks
                per_factor_confs[fct] = tlp
                per_factor_preds[fct].append(A.normalize(_tokens_to_text(model, toks), args.keep_spaces))

            # ROVER: confidence-weighted character voting across the 3 hypotheses.
            if need_confs:
                fused = rover_fuse(
                    [per_factor_tokens[f] for f in factors],
                    [per_factor_confs[f] for f in factors],
                    primary_idx,
                )
                rover_text = A.normalize(_tokens_to_text(model, fused), args.keep_spaces)
                for m in rover_modes:
                    joint_preds[m].append(rover_text)

            # Hypothesis-level reranking: forced-align each per-factor hyp under
            # every stream, then select a hyp per rerank strategy.
            if need_forced:
                sc = {ff: {} for ff in factors}    # sc[hyp_factor][stream_factor] = (logprob, placed, leftover)
                for ff in factors:                  # hypothesis from factor ff
                    toks = per_factor_tokens[ff]
                    for gi, gg in enumerate(factors):  # scored under stream gg
                        sc[ff][gg] = forced_sequence_score(
                            model, enc_proj[gi], chunk_lens[gi], num_chunks[gi],
                            toks, blank_id, max_symbols,
                        )
                own = {ff: sc[ff][ff][0] for ff in factors}
                cross_sum = {
                    ff: sum(sc[ff][gg][0] - _LEFTOVER_PENALTY * sc[ff][gg][2] for gg in factors)
                    for ff in factors
                }
                cross_pertok = {
                    ff: (sum(sc[ff][gg][0] / max(sc[ff][gg][1], 1) for gg in factors) / len(factors)
                         - _LEFTOVER_PENALTY * sum(sc[ff][gg][2] for gg in factors)
                         / max(len(per_factor_tokens[ff]), 1))
                    for ff in factors
                }
                sel = {
                    "rerank_own": _pick_factor(own),
                    "rerank_cross_sum": _pick_factor(cross_sum),
                    "rerank_cross_pertok": _pick_factor(cross_pertok),
                }
                for m in rerank_modes:
                    joint_preds[m].append(per_factor_preds[sel[m]][i])

            # Joint decode across all streams, once per step fusion mode.
            for m in step_modes:
                toks, stats, _ = joint_greedy_decode(
                    model, enc_proj, chunk_lens, num_chunks,
                    blank_id, max_symbols, primary_idx, factor_tags,
                    epsilon=args.score_epsilon, fusion=m,
                )
                joint_preds[m].append(A.normalize(_tokens_to_text(model, toks), args.keep_spaces))
                joint_stats[m]["joint_blank"] += stats["joint_blank"]
                joint_stats[m]["max_symbol_forces"] += stats["max_symbol_forces"]
    finally:
        if saved_log_softmax is not None or hasattr(model.joint, "log_softmax"):
            model.joint.log_softmax = saved_log_softmax

    elapsed = time.time() - t0

    factor_corpus = {
        f: 100 * word_error_rate(hypotheses=per_factor_preds[f], references=refs, use_cer=True) for f in factors
    }
    best_fixed = min(factor_corpus, key=factor_corpus.get)
    joint_corpus = {m: 100 * word_error_rate(hypotheses=joint_preds[m], references=refs, use_cer=True)
                    for m in requested}

    # Oracle: per-utterance best among per-factor independent hyps.
    oracle_preds = []
    for i in range(n):
        cand = [(O.per_utt_cer(refs[i], per_factor_preds[f][i]), f) for f in factors]
        best_cer = min(c for c, _ in cand)
        tied = [f for c, f in cand if c == best_cer]
        of = 1.0 if 1.0 in tied else min(tied, key=lambda f: abs(f - 1.0))
        oracle_preds.append(per_factor_preds[of][i])
    oracle_corpus = 100 * word_error_rate(hypotheses=oracle_preds, references=refs, use_cer=True)

    best_mode = min(joint_corpus, key=joint_corpus.get)

    # ----------------------------- report ----------------------------------- #
    print()
    print("=" * 84)
    print(f"JOINT TIME-WARP DECODING (CER)  |  set={set_name}  method={args.method}")
    print("=" * 84)
    print(f"model      : {args.model}")
    print(f"utterances : {n}   factors: {factors}   epsilon: {args.score_epsilon}   (decode {elapsed:.1f}s)")
    print("")
    print("Per-factor corpus CER (each warp decoded alone):")
    for f in factors:
        tag = "  (baseline/no-warp)" if f == 1.0 else ""
        print(f"  x{f:<5}: {factor_corpus[f]:6.2f} %{tag}")
    print("")
    print(f"baseline (x1.0)        : {factor_corpus[1.0]:6.2f} %")
    print(f"best single fixed (x{best_fixed}) : {factor_corpus[best_fixed]:6.2f} %")
    print("JOINT strategies (sorted best-first):")
    span = max(factor_corpus[1.0] - oracle_corpus, 1e-9)
    for m in sorted(joint_corpus, key=joint_corpus.get):
        # % of the baseline->oracle gap that this strategy closes.
        closed = 100.0 * (factor_corpus[1.0] - joint_corpus[m]) / span
        star = "  <== best" if m == best_mode else ""
        print(f"  {m:<18}: {joint_corpus[m]:6.2f} %   (gap closed: {closed:5.1f}%){star}")
    print(f"ORACLE best-of-{len(factors)}           : {oracle_corpus:6.2f} %")
    for m in step_modes:
        jb, mf = joint_stats[m]["joint_blank"], joint_stats[m]["max_symbol_forces"]
        if jb or mf:
            print(f"[note] fusion={m}: joint_blank={jb} max_force={mf}")
    print("=" * 84)
    # Machine-readable lines.
    for m in requested:
        print(
            f"JOINT_MODE set={set_name} fusion={m} scored={n} epsilon={args.score_epsilon} "
            f"baseline={factor_corpus[1.0]:.4f} joint={joint_corpus[m]:.4f} oracle={oracle_corpus:.4f}"
        )
    # Backward-compatible summary line (best mode) for the shell wrapper.
    bstats = joint_stats.get(best_mode, {"joint_blank": 0, "max_symbol_forces": 0})
    print(
        f"JOINT_SUMMARY set={set_name} scored={n} epsilon={args.score_epsilon} fusion={best_mode} "
        f"baseline={factor_corpus[1.0]:.4f} best_fixed={factor_corpus[best_fixed]:.4f} "
        f"joint={joint_corpus[best_mode]:.4f} oracle={oracle_corpus:.4f} "
        f"factor_cers={','.join(f'x{f}:{factor_corpus[f]:.2f}' for f in factors)} "
        f"joint_blank={bstats['joint_blank']} max_force={bstats['max_symbol_forces']}"
    )

    if args.output:
        import json
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            for i in range(n):
                fh.write(json.dumps({
                    "audio_filepath": items[i]["audio"],
                    "duration": items[i]["duration"],
                    "text_normalized": refs[i],
                    "joint_preds": {m: joint_preds[m][i] for m in requested},
                    "joint_cers": {m: round(O.per_utt_cer(refs[i], joint_preds[m][i]), 4) for m in requested},
                    "per_factor": {
                        str(f): {
                            "pred": per_factor_preds[f][i],
                            "cer": round(O.per_utt_cer(refs[i], per_factor_preds[f][i]), 4),
                        } for f in factors
                    },
                }, ensure_ascii=False) + "\n")
        print(f"Wrote per-utterance report to {args.output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Path to a .nemo or Lightning .ckpt CHAT model.")
    ap.add_argument("--manifest", required=True, help="Local NeMo manifest (audio_filepath + text per line).")
    ap.add_argument("--set_name", default=None, help="Label for the summary line (default: manifest basename).")
    ap.add_argument("--tokenizer_dir", default=None, help="Override tokenizer dir for .ckpt loads.")
    ap.add_argument("--audio_src_prefix", default="/data/mandarin/aishell2/evaluation/aishell")
    ap.add_argument("--audio_dst_prefix", default="")
    ap.add_argument("--text_key", default="text", help="Reference text field in the manifest.")
    ap.add_argument("--factors", default="0.9,1.0,1.1", help="Comma-separated warp factors; 1.0 is auto-added.")
    ap.add_argument("--score_epsilon", type=float, default=0.0,
                    help="(max fusion) switch to a warped stream's token only if its top log-prob "
                         "beats the primary x1.0 top log-prob by >= epsilon (0 = pure most-confident, "
                         "primary-preferred on ties).")
    ap.add_argument("--fusion", default="max",
                    help=f"Comma-separated fusion strategies, or 'all'. Choices: {','.join(FUSION_MODES)}. "
                         "All are evaluated in a single encode-once pass.")
    ap.add_argument("--method", default="speed", choices=["speed", "time_stretch"])
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--max_eval_samples", type=int, default=None)
    ap.add_argument("--max_symbols_per_step", type=int, default=None,
                    help="Cap on non-blank emissions before forcing a chunk advance (default: model cfg or 10).")
    ap.add_argument("--keep_spaces", action="store_true")
    ap.add_argument("--output", default=None)
    main(ap.parse_args())
