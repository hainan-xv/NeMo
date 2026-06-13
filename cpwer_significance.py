#!/usr/bin/env python3
"""Compute cpWER for the 2-speaker Fisher ASR task and run a paired bootstrap
significance test for Causal (baseline) vs. BACON.

cpWER (concatenated minimum-permutation WER) is computed per segment: the
reference and hypothesis are split into per-speaker word streams using the
[SPKk] tags, and for each permutation of hypothesis speakers we sum the
word-level edit distance against the reference speakers, keeping the minimum.
The corpus cpWER is sum(min_errors) / sum(ref_words).

Significance uses a paired bootstrap over segments on the cpWER *difference*
(baseline - BACON), matching the methodology used for the other result tables.
"""
import json
import re
import sys
import numpy as np
from itertools import permutations

BASELINE = "predictions_fisher_2spk_chat_baseline.json"
BACON = "predictions_fisher_2spk_chat_BACON.json"
SPK_RE = re.compile(r"\[SPK\d+\]")
SPK_TOK_RE = re.compile(r"\[SPK(\d+)\]")


def split_speakers(text):
    """Return {spk_id: [words...]} from a string with [SPKk] tags."""
    text = text.strip()
    streams = {}
    # find tag positions
    parts = []
    last_end = 0
    last_spk = None
    for m in SPK_TOK_RE.finditer(text):
        if last_spk is not None:
            parts.append((last_spk, text[last_end:m.start()]))
        last_spk = int(m.group(1))
        last_end = m.end()
    if last_spk is not None:
        parts.append((last_spk, text[last_end:]))
    else:
        # no speaker tag -> treat whole thing as speaker 0
        parts.append((0, text))
    for spk, chunk in parts:
        streams.setdefault(spk, []).extend(chunk.split())
    return streams


def edit_distance(ref, hyp):
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ri = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def cpwer_segment(ref_text, hyp_text):
    """Return (min_errors, ref_words) for one segment."""
    ref = split_speakers(ref_text)
    hyp = split_speakers(hyp_text)
    spk_ids = sorted(set(ref) | set(hyp))
    ref_streams = [ref.get(s, []) for s in spk_ids]
    hyp_streams = [hyp.get(s, []) for s in spk_ids]
    ref_words = sum(len(r) for r in ref_streams)
    best = None
    for perm in permutations(range(len(spk_ids))):
        err = sum(edit_distance(ref_streams[i], hyp_streams[perm[i]])
                  for i in range(len(spk_ids)))
        if best is None or err < best:
            best = err
    return best, ref_words


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    base = load(BASELINE)
    bac = load(BACON)
    assert len(base) == len(bac), f"length mismatch {len(base)} vs {len(bac)}"

    n = len(base)
    err_b = np.zeros(n)
    err_a = np.zeros(n)
    words = np.zeros(n)
    for i, (rb, ra) in enumerate(zip(base, bac)):
        key_b = (rb["audio_filepath"], rb.get("offset"))
        key_a = (ra["audio_filepath"], ra.get("offset"))
        if key_b != key_a:
            sys.exit(f"PAIRING MISMATCH at line {i}: {key_b} vs {key_a}")
        eb, wb = cpwer_segment(rb["text"], rb["pred_text"])
        ea, wa = cpwer_segment(ra["text"], ra["pred_text"])
        assert wb == wa, f"ref word count differs at {i}: {wb} vs {wa}"
        err_b[i], err_a[i], words[i] = eb, ea, wb

    tot_w = words.sum()
    cpwer_b = 100.0 * err_b.sum() / tot_w
    cpwer_a = 100.0 * err_a.sum() / tot_w
    print(f"segments            : {n}")
    print(f"total ref words     : {int(tot_w)}")
    print(f"cpWER Causal (base) : {cpwer_b:.2f}%  (paper 27.41)")
    print(f"cpWER BACON         : {cpwer_a:.2f}%  (paper 27.14)")
    print(f"absolute improvement: {cpwer_b - cpwer_a:.2f}%")

    # paired bootstrap on the cpWER difference (baseline - bacon)
    rng = np.random.default_rng(1234)
    B = 10000
    diffs = np.zeros(B)
    idx = np.arange(n)
    for b in range(B):
        s = rng.choice(idx, size=n, replace=True)
        w = words[s].sum()
        cb = 100.0 * err_b[s].sum() / w
        ca = 100.0 * err_a[s].sum() / w
        diffs[b] = cb - ca
    obs = cpwer_b - cpwer_a
    # two-sided p: fraction of bootstrap diffs that cross zero (no improvement)
    p_boot = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    p_boot = min(1.0, p_boot)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print()
    print(f"paired bootstrap (B={B}) on cpWER diff (base-BACON):")
    print(f"  observed delta cpWER : {obs:.3f}%")
    print(f"  95% CI               : [{lo:.3f}, {hi:.3f}]")
    print(f"  p (two-sided)        : {p_boot:.3f}")

    # Wilcoxon signed-rank on per-segment error differences (secondary)
    try:
        from scipy.stats import wilcoxon
        d = err_b - err_a
        nz = d[d != 0]
        if len(nz) > 0:
            stat, p_w = wilcoxon(err_b, err_a)
            print(f"  Wilcoxon signed-rank p: {p_w:.3f}  (nonzero pairs={len(nz)})")
    except Exception as e:
        print(f"  (wilcoxon skipped: {e})")


if __name__ == "__main__":
    main()
