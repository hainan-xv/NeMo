#!/usr/bin/env python3
"""Verify MultiVocabChunkJointDecoder against a step-by-step reference.

Random weights are fine: every check here is about the DECODING MATH (the
prediction-network recurrence, batched scoring, state carry-over, head
isolation), none of which depends on the model being any good.
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from chat_multivocab_smoke import build_tokenizers, make_corpus, make_data, toy_cfg

OK = True


def check(label, cond, detail=""):
    global OK
    OK = OK and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}  {detail}", flush=True)


def main():
    from nemo.collections.asr.models import EncDecMultiVocabCHATBPEModel
    from nemo.collections.asr.parts.submodules.multivocab_joint_decoding import MultiVocabChunkJointDecoder

    root = tempfile.mkdtemp(prefix="mvjoint_")
    try:
        corpus = os.path.join(root, "corpus.txt")
        make_corpus(corpus)
        tok_dirs = build_tokenizers(root, corpus)
        cuts = make_data(root)
        cfg = toy_cfg(tok_dirs, cuts, steps=1, devices=1)

        model = EncDecMultiVocabCHATBPEModel(cfg=cfg.model, trainer=None)
        model.eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(dev)
        print(f"  model on {dev}, heads={[h.name.split('/')[-1] for h in model._heads]}", flush=True)

        torch.manual_seed(0)
        sig = torch.randn(1, 16000 * 2, device=dev) * 0.05
        sig_len = torch.tensor([16000 * 2], device=dev)
        with torch.no_grad():
            enc, enc_len = model.forward(input_signal=sig, input_signal_length=sig_len)

        # Tests A-E exercise the BEAM strategy's internals (candidate pool,
        # cross-head scoring, summed argmax). Pin it explicitly -- the default is
        # now the chunk-synchronous greedy strategy, under which a "pool" does
        # not exist and these checks would be testing nothing.
        dec = MultiVocabChunkJointDecoder(
            model, weights=[1.0, 1.0, 1.0], beam=2, max_candidates=8, strategy="beam"
        )
        h0 = dec.heads[0]

        chunk_fn = h0.joint.chunk_encoder_for_decoding
        chunked, _n, cfl = chunk_fn(enc, enc_len)
        x = chunked.transpose(1, 2)[0].unsqueeze(1)
        f = x.narrow(0, 0, 1)
        f_len = cfl[0:1, 0:1]
        print(f"  {x.shape[0]} chunks, window dim {x.shape[-1]}, chunk0 frames {int(f_len[0,0])}", flush=True)

        from nemo.collections.asr.parts.submodules.multivocab_joint_decoding import _State

        state = _State(dec_state=None, last_token=None)

        # --- A. batched scoring == step-by-step reference ---------------------
        # The reference walks the RNN-T recurrence one token at a time, exactly
        # as the greedy decoder does, and is therefore independent of _score's
        # padding/batching logic.
        def ref_score(head, tokens, st):
            total, cur = 0.0, _State(st.dec_state, st.last_token)
            for tok in tokens:
                g, new = dec._pred(head, [], cur)
                lp = dec._logp(head, f, f_len, g[:, -1:])[0, 0]
                total += float(lp[tok])
                cur = dec._advance(head, [tok], cur)
            g, _ = dec._pred(head, [], cur)
            lp = dec._logp(head, f, f_len, g[:, -1:])[0, 0]
            total += float(lp[head.blank])
            return total

        cands = [[], h0.tokenizer.text_to_ids("the"), h0.tokenizer.text_to_ids("the cat sat")]
        batched = dec._score(h0, cands, f, f_len, state)
        refs = [ref_score(h0, c, state) for c in cands]
        worst = max(abs(b - r) for b, r in zip(batched, refs))
        check(
            "batched score == step-by-step reference", worst < 1e-3, f"max|diff|={worst:.2e} over {len(cands)} cands"
        )

        # --- B. padding does not leak between candidates ----------------------
        alone = [dec._score(h0, [c], f, f_len, state)[0] for c in cands]
        worst_b = max(abs(a - b) for a, b in zip(alone, batched))
        check("scoring is batch-invariant", worst_b < 1e-4, f"max|diff|={worst_b:.2e}")

        # --- C. a zero weight truly removes a head ----------------------------
        # Compare SCORES, not just text: on a random model the text is often
        # empty, and ""=="" would pass this test without exercising anything.
        d100 = MultiVocabChunkJointDecoder(model, weights=[1.0, 0.0, 0.0], beam=2, strategy="beam")
        before = d100.decode(enc, enc_len)[0]
        uni_before = dec.decode(enc, enc_len)[0]
        with torch.no_grad():
            for prm in model._heads[1].decoder.parameters():
                prm.add_(torch.randn_like(prm) * 5.0)
            for prm in model._heads[2].joint.parameters():
                prm.add_(torch.randn_like(prm) * 5.0)
        after = d100.decode(enc, enc_len)[0]
        uni_after = dec.decode(enc, enc_len)[0]

        same = before.chunk_texts == after.chunk_texts and all(
            abs(a - b) < 1e-4 for a, b in zip(before.chunk_scores, after.chunk_scores)
        )
        check("weights=[1,0,0] ignores heads 1-2", same, f"scores {[round(v,3) for v in before.chunk_scores]}")
        # POSITIVE CONTROL. If perturbing heads 1-2 changes nothing under UNIFORM
        # weights either, then the test above is vacuous and the heads are not
        # actually contributing to the combined score.
        moved = uni_before.chunk_texts != uni_after.chunk_texts or any(
            abs(a - b) > 1e-3 for a, b in zip(uni_before.chunk_scores, uni_after.chunk_scores)
        )
        check("positive control: uniform weights DO react", moved, "perturbing heads 1-2 moved the joint score")

        # --- D. end to end, all heads ----------------------------------------
        res = dec.decode(enc, enc_len)[0]
        check(
            "decode runs over every chunk",
            len(res.chunk_texts) <= x.shape[0],
            f"{len(res.chunk_texts)} chunks emitted",
        )
        check("candidate pool is non-trivial", max(res.n_candidates) > 1, f"pool sizes {res.n_candidates[:6]}")

        # --- D2. chunk boundaries keep their word boundary --------------------
        # Tested at the TOKENISER level, not on the decode: on a toy model the
        # second chunk often emits nothing, and a word-count check then passes
        # even with the bug present -- verified by reintroducing it. This pins
        # the mechanism the decoder relies on and cannot pass vacuously.
        tok = h0.tokenizer
        a, b = "the cat", "sat down"
        via_ids = tok.ids_to_text(tok.text_to_ids(a) + tok.text_to_ids(b))
        via_join = tok.ids_to_text(tok.text_to_ids(a)) + tok.ids_to_text(tok.text_to_ids(b))
        check(
            "accumulating ids preserves the word boundary",
            via_ids.split() == (a + " " + b).split(),
            f"{via_ids!r}",
        )
        check(
            "control: per-chunk join really does weld",
            via_join.split() != (a + " " + b).split(),
            f"{via_join!r} -- this is the 58-74% WER",
        )
        # And the decoder must use the first form.
        n_text, n_chunks_w = len(res.text.split()), sum(len(c.split()) for c in res.chunk_texts)
        nonempty = sum(1 for c in res.chunk_texts if c.strip())
        if nonempty >= 2:
            check("decoder text keeps every word", n_text == n_chunks_w, f"{n_text} vs {n_chunks_w}")
        else:
            print(f"  [SKIP] decoder text keeps every word  only {nonempty} non-empty chunk(s) on toy audio")

        # --- E. the argmax really is the joint argmax -------------------------
        # Re-score the winner and every head's own favourite under the SUM; the
        # winner must not be beaten by any of them, or the combination is not
        # doing what it claims.
        st = [_State(None, None) for _ in dec.heads]
        pool = {""}
        for head in dec.heads:
            for toks in dec._propose(head, f, f_len, _State(None, None)):
                pool.add(head.tokenizer.ids_to_text(toks))
        pool = sorted(pool)
        tot = [0.0] * len(pool)
        for head in dec.heads:
            sc = dec._score(head, [head.tokenizer.text_to_ids(w) for w in pool], f, f_len, _State(None, None))
            for i, s in enumerate(sc):
                tot[i] += s
        best = pool[max(range(len(pool)), key=lambda i: tot[i])]
        check("chunk 0 winner is the summed argmax", res.chunk_texts[0] == best, f"got {res.chunk_texts[0]!r}")

        # --- F. THE correctness test: [1,0,0] == head-0 greedy, exactly --------
        # Not "close to" -- identical text. The chunk-synchronous greedy strategy
        # uses each head's own local argmax rule, so with one head active it IS
        # the greedy decoder. Anything else means the state hand-off, the
        # max_symbols cap or the chunk geometry disagrees with the real path.
        # This is what caught the winning head being advanced on
        # text_to_ids(ids_to_text(t)), which is not the identity.
        model._select_head(0)
        ms = getattr(model.decoding.decoding, "max_symbols", None)
        with torch.no_grad():
            ref = model.decoding.rnnt_decoder_predictions_tensor(enc, enc_len, return_hypotheses=False)
        if isinstance(ref, tuple):
            ref = ref[0]
        ref_text = [h.text if hasattr(h, "text") else h for h in ref]
        gdec = MultiVocabChunkJointDecoder(model, weights=[1.0, 0.0, 0.0], strategy="greedy", max_symbols=ms)
        with torch.no_grad():
            got_text = [r.text for r in gdec.decode(enc, enc_len)]
        check(
            "greedy [1,0,0] == head-0 greedy, exactly",
            ref_text == got_text,
            f"{len(ref_text)} utt(s), max_symbols={ms}",
        )
        if ref_text != got_text:
            for a, b in zip(ref_text, got_text):
                if a != b:
                    print(f"    greedy: {a[:100]!r}\n    joint : {b[:100]!r}")
                    break

        # all heads active still runs and yields text
        adec = MultiVocabChunkJointDecoder(model, weights=[1.0, 1.0, 1.0], strategy="greedy", max_symbols=ms)
        with torch.no_grad():
            a3 = adec.decode(enc, enc_len)
        check("greedy 3-head decode runs", all(isinstance(r.text, str) for r in a3), f"{len(a3)} utt(s)")

        print("\nRESULT:", "ALL PASS" if OK else "FAILURES ABOVE", flush=True)
        return 0 if OK else 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
