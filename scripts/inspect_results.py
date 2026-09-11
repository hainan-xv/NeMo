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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("worst", cmd_worst), ("diff", cmd_diff), ("summary", cmd_summary)):
        q = sub.add_parser(name)
        q.add_argument("a")
        q.add_argument("b", nargs="?" if name != "diff" else None)
        q.add_argument("-n", type=int, default=10)
        q.set_defaults(func=fn)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
