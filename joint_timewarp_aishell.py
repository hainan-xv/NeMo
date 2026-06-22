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
           distribution and take the argmax over the UNION of the active factors'
           distributions -- i.e. emit the token that is most confident (highest
           log-prob) in ANY single stream (a max-of-distributions / most-confident
           rule).  The emitted token updates the shared decoder state.  Since each
           factor's own top class is non-blank after step 2, the winner is non-blank
           by construction (a defensive blank warning remains, but never fires).
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


@torch.inference_mode()
def joint_greedy_decode(model, enc_proj_list, chunk_lens_list, num_chunks_list,
                        blank_id, max_symbols, primary_idx, factor_tags):
    """Shared-state chunk-synchronous joint greedy decode across warp streams.

    ``factor_tags`` is only used for readable warning messages.  Returns
    ``(token_ids, stats)`` where ``stats`` records joint-blank and max-symbol events.
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

        # ---- pick the single most-confident (factor, token) across streams ----
        # Each lp[f] is already a log-softmax distribution; take the argmax over the
        # union of the active factors' distributions (i.e. the token whose log-prob
        # is highest in ANY single stream).  Because every active factor's own argmax
        # is non-blank after the advancement above, the winner is non-blank by
        # construction -- the blank branch below is a defensive safety net only.
        best = None
        best_val = None
        for f in active:
            val, idx = lp[f].max(dim=-1)
            val = float(val.item())
            if best_val is None or val > best_val:
                best_val = val
                best = int(idx.item())

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

    return tokens, stats


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
    print(
        f"Loaded {n} utterances; joint-decoding warps {factors} (primary=x1.0) "
        f"method={args.method!r}, blank_id={blank_id}, max_symbols={max_symbols}."
    )

    joint_preds = []
    per_factor_preds = {f: [] for f in factors}
    total_joint_blank = 0
    total_max_force = 0
    t0 = time.time()

    try:
        for i in tqdm(range(n), desc="joint time-warp"):
            y, sr = soundfile.read(items[i]["audio"], dtype="float32", always_2d=False)
            if y.ndim == 2:
                y = y.mean(axis=1)
            enc_proj, chunk_lens, num_chunks = encode_warps(model, y, factors, args.method, device)

            # Per-factor independent greedy (single stream) -> baseline + oracle.
            for fi, fct in enumerate(factors):
                toks, _ = joint_greedy_decode(
                    model, [enc_proj[fi]], [chunk_lens[fi]], [num_chunks[fi]],
                    blank_id, max_symbols, 0, [factor_tags[fi]],
                )
                per_factor_preds[fct].append(A.normalize(_tokens_to_text(model, toks), args.keep_spaces))

            # Joint decode across all streams.
            toks, stats = joint_greedy_decode(
                model, enc_proj, chunk_lens, num_chunks,
                blank_id, max_symbols, primary_idx, factor_tags,
            )
            joint_preds.append(A.normalize(_tokens_to_text(model, toks), args.keep_spaces))
            total_joint_blank += stats["joint_blank"]
            total_max_force += stats["max_symbol_forces"]
    finally:
        if saved_log_softmax is not None or hasattr(model.joint, "log_softmax"):
            model.joint.log_softmax = saved_log_softmax

    elapsed = time.time() - t0

    factor_corpus = {
        f: 100 * word_error_rate(hypotheses=per_factor_preds[f], references=refs, use_cer=True) for f in factors
    }
    best_fixed = min(factor_corpus, key=factor_corpus.get)
    joint_corpus = 100 * word_error_rate(hypotheses=joint_preds, references=refs, use_cer=True)

    # Oracle: per-utterance best among per-factor independent hyps.
    oracle_preds = []
    for i in range(n):
        cand = [(O.per_utt_cer(refs[i], per_factor_preds[f][i]), f) for f in factors]
        best_cer = min(c for c, _ in cand)
        tied = [f for c, f in cand if c == best_cer]
        of = 1.0 if 1.0 in tied else min(tied, key=lambda f: abs(f - 1.0))
        oracle_preds.append(per_factor_preds[of][i])
    oracle_corpus = 100 * word_error_rate(hypotheses=oracle_preds, references=refs, use_cer=True)

    # ----------------------------- report ----------------------------------- #
    print()
    print("=" * 84)
    print(f"JOINT TIME-WARP DECODING (CER)  |  set={set_name}  method={args.method}")
    print("=" * 84)
    print(f"model      : {args.model}")
    print(f"utterances : {n}   factors: {factors}   (decode {elapsed:.1f}s)")
    print("")
    print("Per-factor corpus CER (each warp decoded alone):")
    for f in factors:
        tag = "  (baseline/no-warp)" if f == 1.0 else ""
        print(f"  x{f:<5}: {factor_corpus[f]:6.2f} %{tag}")
    print("")
    print(f"baseline (x1.0)        : {factor_corpus[1.0]:6.2f} %")
    print(f"best single fixed (x{best_fixed}) : {factor_corpus[best_fixed]:6.2f} %")
    print(f"JOINT decode           : {joint_corpus:6.2f} %")
    print(f"ORACLE best-of-{len(factors)}       : {oracle_corpus:6.2f} %")
    if total_joint_blank:
        print(f"[warn] joint argmax was blank {total_joint_blank} time(s) across the set.")
    if total_max_force:
        print(f"[note] max-symbol safeguard fired {total_max_force} time(s).")
    print("=" * 84)
    # Machine-readable summary line for the shell wrapper.
    print(
        f"JOINT_SUMMARY set={set_name} scored={n} "
        f"baseline={factor_corpus[1.0]:.4f} best_fixed={factor_corpus[best_fixed]:.4f} "
        f"joint={joint_corpus:.4f} oracle={oracle_corpus:.4f} "
        f"factor_cers={','.join(f'x{f}:{factor_corpus[f]:.2f}' for f in factors)} "
        f"joint_blank={total_joint_blank} max_force={total_max_force}"
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
                    "joint_pred": joint_preds[i],
                    "joint_cer": round(O.per_utt_cer(refs[i], joint_preds[i]), 4),
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
    ap.add_argument("--method", default="speed", choices=["speed", "time_stretch"])
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--max_eval_samples", type=int, default=None)
    ap.add_argument("--max_symbols_per_step", type=int, default=None,
                    help="Cap on non-blank emissions before forcing a chunk advance (default: model cfg or 10).")
    ap.add_argument("--keep_spaces", action="store_true")
    ap.add_argument("--output", default=None)
    main(ap.parse_args())
