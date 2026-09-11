"""Read the per-utterance dumps chat_val_probe.py writes, and show what differs.

    # worst utterances for one model
    python scripts/inspect_results.py worst chat.trim0.jsonl -n 20

    # where two models disagree most (the useful one for a diagnosis)
    python scripts/inspect_results.py diff chat.trim0.jsonl nemotron.trim0.jsonl -n 20

    # aggregate error breakdown, and WER against utterance duration
    python scripts/inspect_results.py summary chat.trim0.jsonl nemotron.trim0.jsonl

A corpus WER hides which utterances moved and how. `diff` pairs two runs by
audio path and sorts by the change in error count, so the transcripts that
actually account for a gap come first instead of being averaged away.
"""

import argparse
import json
from collections import defaultdict


def load(path):
    recs = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                recs[r["audio"]] = r
    return recs


def _corpus_wer(recs):
    e = sum(r["errors"] for r in recs.values())
    w = sum(r["words"] for r in recs.values())
    return e / max(w, 1), e, w


def _show(r, label=""):
    print(f"  {label}[{r['errors']}/{r['words']} err, {r['duration']:.1f}s, "
          f"ins {r['ins']} del {r['del']} sub {r['sub']}]")
    print(f"    REF: {r['ref']}")
    print(f"    HYP: {r['hyp']}")


def cmd_worst(args):
    recs = load(args.a)
    wer, e, w = _corpus_wer(recs)
    print(f"{args.a}: {len(recs)} utts, corpus WER {wer:.4f} ({e}/{w})\n")
    ranked = sorted(recs.values(), key=lambda r: (-r["errors"], -r["wer"]))
    for r in ranked[: args.n]:
        _show(r)
        print()


def cmd_diff(args):
    A, B = load(args.a), load(args.b)
    shared = set(A) & set(B)
    wa, ea, wta = _corpus_wer({k: A[k] for k in shared})
    wb, eb, wtb = _corpus_wer({k: B[k] for k in shared})
    print(f"A = {args.a}\n    {len(A)} utts, corpus WER {wa:.4f} ({ea} errors)")
    print(f"B = {args.b}\n    {len(B)} utts, corpus WER {wb:.4f} ({eb} errors)")
    print(f"shared: {len(shared)} utts;  A-B = {ea - eb:+d} errors\n")

    delta = sorted(shared, key=lambda k: A[k]["errors"] - B[k]["errors"])
    worse_in_b, worse_in_a = delta[: args.n], delta[-args.n :][::-1]

    print(f"### {args.n} utterances where B is WORSE than A")
    for k in worse_in_b:
        if A[k]["errors"] - B[k]["errors"] >= 0:
            continue
        print(f"  {k.split('/')[-1]}  A={A[k]['errors']} B={B[k]['errors']} "
              f"({B[k]['errors'] - A[k]['errors']:+d})  {A[k]['duration']:.1f}s")
        print(f"    REF: {A[k]['ref']}")
        print(f"    A:   {A[k]['hyp']}")
        print(f"    B:   {B[k]['hyp']}")
        print()

    print(f"### {args.n} utterances where A is WORSE than B")
    for k in worse_in_a:
        if A[k]["errors"] - B[k]["errors"] <= 0:
            continue
        print(f"  {k.split('/')[-1]}  A={A[k]['errors']} B={B[k]['errors']} "
              f"({B[k]['errors'] - A[k]['errors']:+d})  {A[k]['duration']:.1f}s")
        print(f"    REF: {A[k]['ref']}")
        print(f"    A:   {A[k]['hyp']}")
        print(f"    B:   {B[k]['hyp']}")
        print()


def cmd_summary(args):
    for path in [args.a] + ([args.b] if args.b else []):
        recs = load(path)
        wer, e, w = _corpus_wer(recs)
        ins = sum(r["ins"] for r in recs.values())
        dele = sum(r["del"] for r in recs.values())
        sub = sum(r["sub"] for r in recs.values())
        empty = sum(1 for r in recs.values() if not r["hyp"].strip())
        print(f"=== {path}")
        print(f"    utts {len(recs)}   words {w}   WER {wer:.4f}")
        print(f"    ins {ins} ({ins/max(e,1):.0%})  del {dele} ({dele/max(e,1):.0%})  "
              f"sub {sub} ({sub/max(e,1):.0%})   empty hyps {empty}")

        # WER by duration: a defect at the end of an utterance shows up as a
        # trend here, while a uniform one does not.
        buckets = defaultdict(lambda: [0, 0])
        for r in recs.values():
            b = min(int(r["duration"] // 5) * 5, 30)
            buckets[b][0] += r["errors"]
            buckets[b][1] += r["words"]
        print("    WER by duration:")
        for b in sorted(buckets):
            e_b, w_b = buckets[b]
            if w_b:
                print(f"      {b:>2}-{b+5:<3}s  {e_b/w_b:.4f}   ({w_b} words)")
        print()


def cmd_deletions(args):
    """Where in the reference do deletions fall?

    A model that truncates the tail of an utterance produces deletions bunched
    in the last decile; one that misses onsets bunches them in the first. A
    model that simply recognises badly spreads them evenly. The duration trend
    alone cannot tell these apart, so align and look.
    """
    from kaldialign import align

    EPS = "*"
    for path in [args.a] + ([args.b] if args.b else []):
        recs = load(path)
        bins = [0] * 10
        first_word = last_word = 0
        total_del = 0
        # Deletions in runs at the very edges, which is what truncation looks like.
        tail_run = head_run = 0
        for r in recs.values():
            ref, hyp = r["ref"].split(), r["hyp"].split()
            if not ref:
                continue
            pairs = align(ref, hyp, EPS)
            ref_pos = 0
            dele_positions = []
            for a, b in pairs:
                if a != EPS:
                    if b == EPS:
                        dele_positions.append(ref_pos)
                    ref_pos += 1
            if not dele_positions:
                continue
            n = len(ref)
            total_del += len(dele_positions)
            for d in dele_positions:
                bins[min(int(d / n * 10), 9)] += 1
            if 0 in dele_positions:
                first_word += 1
            if n - 1 in dele_positions:
                last_word += 1
            # trailing run: deletions covering the final k words contiguously
            k = 0
            while n - 1 - k in dele_positions:
                k += 1
            tail_run += k
            k = 0
            while k in dele_positions:
                k += 1
            head_run += k

        print(f"=== {path}")
        print(f"    {total_del} deletions over {len(recs)} utterances")
        print("    position in reference (decile):")
        for i, c in enumerate(bins):
            bar = "#" * int(60 * c / max(max(bins), 1))
            print(f"      {i*10:>3}-{i*10+10:<3}%  {c:>6}  {bar}")
        print(f"    utterances missing the FIRST word: {first_word}")
        print(f"    utterances missing the LAST  word: {last_word}")
        print(f"    words lost in a contiguous HEAD run: {head_run}")
        print(f"    words lost in a contiguous TAIL run: {tail_run}")
        print()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (
        ("worst", cmd_worst),
        ("diff", cmd_diff),
        ("summary", cmd_summary),
        ("deletions", cmd_deletions),
    ):
        q = sub.add_parser(name)
        q.add_argument("a")
        q.add_argument("b", nargs="?" if name != "diff" else None)
        q.add_argument("-n", type=int, default=10)
        q.set_defaults(func=fn)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
