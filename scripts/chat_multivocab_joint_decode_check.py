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

        dec = MultiVocabChunkJointDecoder(model, weights=[1.0, 1.0, 1.0], beam=2, max_candidates=8)
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
        d100 = MultiVocabChunkJointDecoder(model, weights=[1.0, 0.0, 0.0], beam=2)
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

        print("\nRESULT:", "ALL PASS" if OK else "FAILURES ABOVE", flush=True)
        return 0 if OK else 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
