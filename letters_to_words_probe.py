#!/usr/bin/env python3
"""Training-free probe: can base Qwen assemble a chunked LETTER stream into WORDS?

Motivation
----------
ASR is, loosely, "assemble low-level units (audio frames) into words". This script
isolates the LLM half of that job in the *text* domain, with NO fine-tuning: we
render a sentence as a stream of individual letters (the analog of audio frames),
deliver that stream in randomly-sized chunks through a chunked / interleaving
multi-turn dialog (mirroring the streaming_stt template), and ask base Qwen to
reconstruct the real words chunk-by-chunk.

Stream encoding
---------------
Each sentence becomes a sequence of SYMBOLS:
  * one symbol per letter (case-lowered), and
  * the literal token ``<space>`` marking a word boundary.
We append a trailing ``<space>`` after the final word so *every* word (including
the last) is boundary-terminated. The prompt explicitly explains this convention.

  "the quick"  ->  t h e <space> q u i c k <space>

Chunking
--------
Chunk length ``n = round(Gaussian(mu, sigma))`` clipped to >= 0 (0-length draws are
skipped). ``n`` counts symbols, so a ``<space>`` marker consumes one slot. Chunks
freely split words mid-word -- exactly like a 14-frame audio chunk ignoring word
boundaries.

Protocol (multi-turn streaming)
-------------------------------
A system turn states the task + the ``<space>`` convention. Then each chunk is a
separate USER turn; after each we generate the ASSISTANT turn, which must emit ONLY
the words that just became COMPLETE (their closing ``<space>`` has now been seen),
or ``<none>`` if the chunk finished mid-word. The conversation history carries the
buffered partial word across turns. Concatenating the assistant turns yields the
reconstruction, scored by WER against the reference.

Usage
-----
    python letters_to_words_probe.py                       # built-in tiny set
    python letters_to_words_probe.py --sentences my.txt    # one sentence per line
    python letters_to_words_probe.py --mu 10 --sigma 5 --seed 0
    python letters_to_words_probe.py --model Qwen/Qwen3-1.7B --enable-thinking
    python letters_to_words_probe.py --show-dialog          # print the full dialog
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SPACE = "<space>"

# A small hand-picked set spanning short/long words, digits-free, mixed lengths.
DEFAULT_SENTENCES = [
    "the quick brown fox",
    "she sells sea shells",
    "speech recognition is hard",
    "assemble letters into words",
    "a cat sat on the mat",
    "streaming models emit text in chunks",
    "language understanding without training",
    "boundaries between words matter",
]


# --------------------------------------------------------------------------- #
# Stream construction
# --------------------------------------------------------------------------- #
def normalize(sentence: str) -> list[str]:
    """Lowercase, keep a-z words only, return the list of reference words."""
    words = re.findall(r"[a-z']+", sentence.lower())
    return words


def to_symbol_stream(words: list[str]) -> list[str]:
    """Words -> flat symbol stream: letters + a <space> after every word."""
    stream: list[str] = []
    for w in words:
        stream.extend(list(w))
        stream.append(SPACE)
    return stream


def chunk_stream(stream: list[str], mu: float, sigma: float, rng: random.Random) -> list[list[str]]:
    """Split the symbol stream into random-length chunks (Gaussian, clipped >=0)."""
    chunks: list[list[str]] = []
    i, n = 0, len(stream)
    while i < n:
        length = int(round(rng.gauss(mu, sigma)))
        if length <= 0:
            continue  # clip negatives/zeros -> redraw (never an empty chunk)
        chunk = stream[i : i + length]
        chunks.append(chunk)
        i += length
    return chunks


def render_chunk(chunk: list[str]) -> str:
    """Space-join symbols so each letter / <space> is a visually distinct unit."""
    return " ".join(chunk)


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You reconstruct words from a stream of letters that arrives in chunks.\n"
    "There are TWO different kinds of spacing in the stream, and you must not confuse them:\n"
    "  - A PLAIN space simply separates the individual letters of the SAME word. When you\n"
    "    build a word you DELETE these plain spaces (pure concatenation of the letters).\n"
    f"  - The literal token {SPACE} is a WORD BOUNDARY. It separates one word from the next.\n"
    f"    You NEVER merge letters across a {SPACE}, and {SPACE} is never part of any word.\n"
    "The letters of a single word may be split across several chunks.\n"
    "\n"
    f"So the letters of one word are exactly the letters lying between two {SPACE} tokens\n"
    "(or between the start of the stream and the first boundary), concatenated with the\n"
    "plain spaces removed. It does NOT matter whether the result is a real dictionary word.\n"
    "\n"
    "Study these examples carefully -- especially how the same letters group differently\n"
    f"depending on where the {SPACE} boundaries fall:\n"
    f"  'i a m {SPACE}'      -> ONE completed word: 'iam'      (letters i,a,m are one word)\n"
    f"  'z q x {SPACE}'      -> ONE completed word: 'zqx'\n"
    f"  'i {SPACE} a m {SPACE}'  -> TWO completed words: 'i' then 'am'  (the {SPACE} splits them;\n"
    "                          you must NOT output 'iam')\n"
    f"  'i {SPACE} a m'      -> only 'i' is completed; 'a','m' are the START of the next word\n"
    f"                          and have NO closing {SPACE} yet, so BUFFER them and output just 'i'\n"
    "\n"
    "Your task: after each new chunk, output ONLY the word(s) that have just become\n"
    f"COMPLETE -- i.e. every word whose closing {SPACE} has now appeared -- each written as\n"
    "the concatenation of its letters (no internal spaces). Do NOT output a word until its\n"
    f"{SPACE} has arrived; if the chunk ends in the middle of a word (no trailing {SPACE}),\n"
    "keep those letters buffered and wait for the rest. If no new word completed in this\n"
    "chunk, reply with exactly <none>.\n"
    "\n"
    "Worked streaming example (letters split across chunks, buffer carried between turns):\n"
    f"  chunk 'i a m {SPACE} h a'   -> output 'iam'      (buffer the trailing 'ha')\n"
    f"  chunk 'p p y {SPACE}'       -> output 'happy'    ('ha'+'ppy' = 'happy')\n"
    "\n"
    "Output only the completed word(s), lowercase, one word per completed word, separated by\n"
    "single spaces, each word's letters concatenated. No explanations, no punctuation, no quotes."
)


def build_first_user(chunk_text: str) -> str:
    return f"Chunk: {chunk_text}"


# --------------------------------------------------------------------------- #
# WER
# --------------------------------------------------------------------------- #
def wer(ref: list[str], hyp: list[str]) -> tuple[float, int, int, int]:
    """Word-level Levenshtein -> (wer, sub, del, ins). Pure-python, no deps."""
    R, H = len(ref), len(hyp)
    d = [[0] * (H + 1) for _ in range(R + 1)]
    for i in range(R + 1):
        d[i][0] = i
    for j in range(H + 1):
        d[0][j] = j
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    # backtrack for S/D/I
    i, j, s, dele, ins = R, H, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            if ref[i - 1] != hyp[j - 1]:
                s += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    err = (s + dele + ins) / max(1, R)
    return err, s, dele, ins


def parse_reply(text: str) -> list[str]:
    """Extract emitted words from an assistant turn (drop <none>, punctuation)."""
    text = text.strip()
    # Strip any stray thinking block if the model emitted one.
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    text = text.replace("</think>", " ").replace("<think>", " ")
    out = []
    for tok in re.findall(r"[a-z']+", text.lower()):
        if tok == "none":
            continue
        out.append(tok)
    return out


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def load_model(model_name: str, dtype: str, local_only: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        local_files_only=local_only,
    )
    model.eval()
    return tok, model


def generate_turn(tok, model, messages: list[dict], enable_thinking: bool, max_new_tokens: int) -> str:
    """Append the generation prompt for `messages` and return the raw assistant reply."""
    import torch

    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def run_sentence(tok, model, chunks: list[list[str]], enable_thinking: bool,
                 max_new_tokens: int, show_dialog: bool) -> tuple[list[str], list[tuple[str, str]]]:
    """Stream chunks through a growing chat; return (reconstructed words, dialog)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    recon: list[str] = []
    dialog: list[tuple[str, str]] = []
    for ci, chunk in enumerate(chunks):
        user_msg = build_first_user(render_chunk(chunk))
        messages.append({"role": "user", "content": user_msg})
        reply = generate_turn(tok, model, messages, enable_thinking, max_new_tokens)
        messages.append({"role": "assistant", "content": reply})
        recon.extend(parse_reply(reply))
        dialog.append((user_msg, reply))
        if show_dialog:
            print(f"    [chunk {ci}] USER: {user_msg}")
            print(f"    [chunk {ci}] ASST: {reply!r}")
    return recon, dialog


def interactive(tok, model, enable_thinking: bool, max_new_tokens: int) -> None:
    """REPL: you type one chunk per line; the model responds after each Enter.

    Type letters separated by spaces and use the literal token ``<space>`` for a
    word boundary, e.g.:  t h e <space> q u   then Enter, then:  i c k <space>
    Commands:  /reset (new sentence)   /show (running reconstruction)   /quit
    """
    def _fresh():
        return [{"role": "system", "content": SYSTEM_PROMPT}], []

    messages, recon = _fresh()
    print("\n" + "=" * 68)
    print("INTERACTIVE letters -> words probe.")
    print("  Type ONE chunk per line, then Enter. Letters are space-separated;")
    print(f"  use the token '{SPACE}' for a word boundary. Example:")
    print("      t h e <space> q u")
    print("      i c k <space>")
    print("  Commands:  /reset  (start a new sentence)   /show  (reconstruction)")
    print("             /quit   (exit)")
    print("=" * 68)
    turn = 0
    while True:
        try:
            line = input(f"\nchunk[{turn}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return
        if not line:
            continue
        cmd = line.lower()
        if cmd in ("/quit", "/exit", "/q"):
            print("bye.")
            return
        if cmd == "/reset":
            messages, recon = _fresh()
            turn = 0
            print("--- conversation reset (new sentence) ---")
            continue
        if cmd == "/show":
            print(f"  reconstruction so far: {' '.join(recon) or '(empty)'}")
            continue

        messages.append({"role": "user", "content": build_first_user(line)})
        reply = generate_turn(tok, model, messages, enable_thinking, max_new_tokens)
        messages.append({"role": "assistant", "content": reply})
        new_words = parse_reply(reply)
        recon.extend(new_words)
        print(f"  model: {reply!r}")
        print(f"  new words this chunk: {new_words if new_words else '(none)'}")
        print(f"  reconstruction so far: {' '.join(recon) or '(empty)'}")
        turn += 1


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--sentences", default=None, help="file with one sentence per line (default: built-in set)")
    ap.add_argument("--mu", type=float, default=10.0, help="mean chunk length in symbols")
    ap.add_argument("--sigma", type=float, default=5.0, help="std of chunk length")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--enable-thinking", action="store_true", help="enable Qwen3 thinking (default OFF)")
    ap.add_argument("--local-files-only", action="store_true", help="use only locally cached weights")
    ap.add_argument("--show-dialog", action="store_true", help="print every chunk turn")
    ap.add_argument("--dry-run", action="store_true", help="build streams/chunks only; no model load")
    ap.add_argument("--interactive", action="store_true",
                    help="REPL: you type chunks one Enter at a time and watch it assemble live")
    args = ap.parse_args()

    if args.interactive:
        tok, model = load_model(args.model, args.dtype, args.local_files_only)
        print(f"Loaded {args.model} (thinking={'ON' if args.enable_thinking else 'OFF'})")
        interactive(tok, model, args.enable_thinking, args.max_new_tokens)
        return

    if args.sentences:
        with open(args.sentences) as f:
            sentences = [ln.strip() for ln in f if ln.strip()]
    else:
        sentences = DEFAULT_SENTENCES

    rng = random.Random(args.seed)

    # Build all streams/chunks first (deterministic given seed).
    prepared = []
    for s in sentences:
        words = normalize(s)
        stream = to_symbol_stream(words)
        chunks = chunk_stream(stream, args.mu, args.sigma, rng)
        prepared.append((s, words, stream, chunks))

    if args.dry_run:
        for s, words, stream, chunks in prepared:
            print(f"\nSENT: {s}")
            print(f"  stream ({len(stream)} sym): {' '.join(stream)}")
            for ci, c in enumerate(chunks):
                print(f"  chunk {ci} ({len(c)}): {render_chunk(c)}")
        return

    tok, model = load_model(args.model, args.dtype, args.local_files_only)
    print(f"Loaded {args.model} (thinking={'ON' if args.enable_thinking else 'OFF'}, "
          f"mu={args.mu}, sigma={args.sigma}, seed={args.seed})\n")

    tot_ref = tot_err = 0
    rows = []
    for s, words, stream, chunks in prepared:
        print(f"SENT: {s!r}  ({len(stream)} symbols -> {len(chunks)} chunks)")
        recon, _ = run_sentence(tok, model, chunks, args.enable_thinking, args.max_new_tokens, args.show_dialog)
        e, sub, dele, ins = wer(words, recon)
        tot_ref += len(words)
        tot_err += sub + dele + ins
        rows.append((s, words, recon, e))
        print(f"  ref:   {' '.join(words)}")
        print(f"  hyp:   {' '.join(recon)}")
        print(f"  WER={e*100:.1f}%  (S={sub} D={dele} I={ins})\n")

    overall = tot_err / max(1, tot_ref)
    print("=" * 60)
    print(f"OVERALL WER: {overall*100:.2f}%  over {len(sentences)} sentences, {tot_ref} ref words")


if __name__ == "__main__":
    main()
