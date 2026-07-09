#!/usr/bin/env python3
"""Teach Qwen the chunked letter->word streaming task with a LoRA adapter.

The base model *can't* do this task untrained (see `letters_to_words_probe.py`): it
leaks in-context example words and, more fundamentally, fails to carry the partial-word
buffer across chunks. Here we fix that with a tiny LoRA adapter trained on an *endless*
stream of freshly synthesized examples whose labels are computed deterministically (no
model-in-the-loop labeling needed -- the task is a pure, decidable function of the
symbol stream).

Data (generated fresh every step -- "continuous")
-------------------------------------------------
Each example is a random sentence (mix of real words + random letter strings) rendered
as a symbol stream `l e t t e r s <space> ...` (same convention as the probe), split
into random-length chunks. For every chunk the target is the word(s) whose closing
`<space>` fell inside that chunk (buffer carried across chunks), or `<none>`.

Training
--------
We build one multi-turn ChatML conversation per example and supervise ONLY the assistant
tokens (chunk targets + the `<|im_end|>` stop), masking system/user/prompt tokens with
-100. LoRA on all attention + MLP projections; base weights frozen.

    python letters_to_words_lora.py --steps 1500 --batch-size 8
    python letters_to_words_lora.py --steps 3000 --eval-every 200 --out ./lora_l2w
    python letters_to_words_lora.py --resume ./lora_l2w   # continue from a saved adapter

The saved adapter can be loaded back into the probe REPL via --adapter (see probe).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import string
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Reuse the probe's stream construction / scoring so train & eval share one convention.
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("l2w_probe", os.path.join(_here, "letters_to_words_probe.py"))
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

SPACE = probe.SPACE
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# Concise instruction. With training the model learns the behavior from labels, so we
# keep the prompt short and, crucially, use NO concrete example words (those leak).
SYSTEM_PROMPT = (
    "You reconstruct words from a stream of letters arriving in chunks. Letters are "
    f"space-separated; the token {SPACE} marks a word boundary. A word is the letters "
    f"between two {SPACE} tokens, concatenated with no spaces. After each chunk, output "
    f"only the word(s) whose closing {SPACE} appeared in that chunk (lowercase, space-"
    "separated, letters concatenated). If a chunk ends mid-word, buffer those letters and "
    f"wait. If no word completed, output exactly <none>."
)


# --------------------------------------------------------------------------- #
# Word pool + example synthesis
# --------------------------------------------------------------------------- #
FALLBACK_WORDS = (
    "the quick brown fox jumps over a lazy dog she sells sea shells by shore speech "
    "recognition is hard assemble letters into words cat sat on mat streaming models "
    "emit text in chunks language understanding without training boundaries between "
    "matter time people year way day thing man world life hand part child eye woman "
    "place work week case point government company number group problem fact water "
    "hello happy house music river mountain window yellow orange purple silver golden "
    "morning evening winter summer autumn spring reason answer letter number system "
    "before after under above below inside outside between around through across near"
).split()


def load_word_pool() -> list[str]:
    words = set(w for w in FALLBACK_WORDS if w.isalpha())
    for path in ("/usr/share/dict/words", "/usr/share/dict/american-english"):
        if os.path.exists(path):
            try:
                with open(path) as f:
                    for ln in f:
                        w = ln.strip().lower()
                        if w.isalpha() and w.isascii() and 1 <= len(w) <= 10:
                            words.add(w)
            except OSError:
                pass
            break
    return sorted(words)


def _rand_word(rng: random.Random) -> str:
    n = rng.randint(1, 8)
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))


def sample_sentence(rng: random.Random, pool: list[str], p_random: float = 0.3,
                    min_w: int = 1, max_w: int = 8) -> list[str]:
    n = rng.randint(min_w, max_w)
    out = []
    for _ in range(n):
        out.append(_rand_word(rng) if rng.random() < p_random else rng.choice(pool))
    return out


def chunk_targets(chunks: list[list[str]]) -> list[str]:
    """Deterministic ground truth: word(s) completed within each chunk (buffer carried)."""
    buf: list[str] = []
    targets: list[str] = []
    for chunk in chunks:
        done: list[str] = []
        for sym in chunk:
            if sym == SPACE:
                done.append("".join(buf))
                buf = []
            else:
                buf.append(sym)
        targets.append(" ".join(done) if done else "<none>")
    return targets


def make_example(rng: random.Random, pool: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """Return (reference_words, [(user_chunk_text, target_text), ...])."""
    words = sample_sentence(rng, pool)
    stream = probe.to_symbol_stream(words)
    mu = rng.uniform(2.0, 12.0)  # vary chunking so buffering is exercised at all scales
    sigma = max(1.0, mu / 2.0)
    chunks = probe.chunk_stream(stream, mu, sigma, rng)
    targets = chunk_targets(chunks)
    turns = [(probe.build_first_user(probe.render_chunk(c)), t) for c, t in zip(chunks, targets)]
    return words, turns


# --------------------------------------------------------------------------- #
# ChatML tokenization with assistant-only supervision (manual, template-free)
# --------------------------------------------------------------------------- #
def tokenize_conversation(tok, turns: list[tuple[str, str]]) -> tuple[list[int], list[int]]:
    ids: list[int] = []
    labels: list[int] = []

    def emit(text: str, supervise: bool):
        t = tok.encode(text, add_special_tokens=False)
        ids.extend(t)
        labels.extend(t if supervise else [-100] * len(t))

    emit(f"{IM_START}system\n{SYSTEM_PROMPT}{IM_END}\n", False)
    for user, asst in turns:
        emit(f"{IM_START}user\n{user}{IM_END}\n", False)
        emit(f"{IM_START}assistant\n", False)
        emit(f"{asst}{IM_END}\n", True)  # supervise the reply AND its stop token
    return ids, labels


def build_prompt_ids(tok, completed: list[tuple[str, str]], pending_user: str) -> list[int]:
    """ids ending in the assistant generation prompt, ready for model.generate."""
    parts = [f"{IM_START}system\n{SYSTEM_PROMPT}{IM_END}\n"]
    for user, asst in completed:
        parts.append(f"{IM_START}user\n{user}{IM_END}\n{IM_START}assistant\n{asst}{IM_END}\n")
    parts.append(f"{IM_START}user\n{pending_user}{IM_END}\n{IM_START}assistant\n")
    return tok.encode("".join(parts), add_special_tokens=False)


def make_batch(tok, rng, pool, batch_size, max_len, device):
    import torch

    seqs, labs = [], []
    while len(seqs) < batch_size:
        _, turns = make_example(rng, pool)
        ids, labels = tokenize_conversation(tok, turns)
        if len(ids) > max_len:
            continue  # rare; just resample rather than truncate mid-turn
        seqs.append(ids)
        labs.append(labels)
    L = max(len(s) for s in seqs)
    pad_id = tok.pad_token_id
    input_ids = torch.full((batch_size, L), pad_id, dtype=torch.long)
    attn = torch.zeros((batch_size, L), dtype=torch.long)
    label = torch.full((batch_size, L), -100, dtype=torch.long)
    for i, (s, lb) in enumerate(zip(seqs, labs)):
        input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        attn[i, : len(s)] = 1
        label[i, : len(lb)] = torch.tensor(lb, dtype=torch.long)
    return input_ids.to(device), attn.to(device), label.to(device)


# --------------------------------------------------------------------------- #
# Eval by real multi-turn streaming (mirrors deployment)
# --------------------------------------------------------------------------- #
EDGE_CASES = [
    # (name, [chunk strings], expected reconstruction words)
    ("single word",          ["i a m <space>"],                             ["iam"]),
    ("buffer trailing word", ["i <space> a m"],                             ["i"]),
    ("two words split",      ["i <space> a m <space>"],                     ["i", "am"]),
    ("cross-chunk buffer",   ["h a", "p p y <space>"],                      ["happy"]),
    ("the quick",            ["t h e <space> q u", "i c k <space>"],        ["the", "quick"]),
    ("mid-word split x3",    ["s t r", "e a m", "i n g <space>"],           ["streaming"]),
    ("multi-complete",       ["a <space> c a t <space> s"],                 ["a", "cat"]),
    ("finish buffered s",    ["a <space> c a t <space> s", "a t <space>"],  ["a", "cat", "sat"]),
]


def stream_reconstruct(tok, model, chunks_text: list[str], max_new_tokens: int = 32) -> list[str]:
    import torch

    im_end_id = tok.convert_tokens_to_ids(IM_END)
    completed: list[tuple[str, str]] = []
    recon: list[str] = []
    for user in chunks_text:
        ids = build_prompt_ids(tok, completed, user)
        inp = torch.tensor([ids], device=model.device)
        with torch.no_grad():
            gen = model.generate(
                input_ids=inp,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None, top_p=None, top_k=None,
                pad_token_id=tok.pad_token_id,
                eos_token_id=im_end_id,
            )
        reply = tok.decode(gen[0, inp.shape[1]:], skip_special_tokens=True).strip()
        completed.append((user, reply))
        recon.extend(probe.parse_reply(reply))
    return recon


def evaluate(tok, model, rng, pool, n_random: int = 16, verbose: bool = False):
    model.eval()
    # 1) fixed edge cases (exact-match)
    edge_pass = 0
    for name, chunks, expected in EDGE_CASES:
        got = stream_reconstruct(tok, model, chunks)
        ok = got == expected
        edge_pass += ok
        if verbose:
            print(f"    [edge] {name:22s} got={got} exp={expected} {'OK' if ok else 'X'}")
    # 2) fresh random sentences (WER)
    tot_ref = tot_err = 0
    for _ in range(n_random):
        words, turns = make_example(rng, pool)
        chunks_text = [u for u, _ in turns]
        recon = stream_reconstruct(tok, model, chunks_text)
        e, s, d, i = probe.wer(words, recon)
        tot_ref += len(words)
        tot_err += s + d + i
    wer = tot_err / max(1, tot_ref)
    model.train()
    return edge_pass, len(EDGE_CASES), wer


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=150)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--out", default="./lora_l2w")
    ap.add_argument("--resume", default=None, help="path to an existing adapter to continue from")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print a few generated examples and exit")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    eval_rng = random.Random(args.seed + 10_000)
    pool = load_word_pool()
    print(f"word pool: {len(pool)} words")

    if args.dry_run:
        for k in range(4):
            words, turns = make_example(rng, pool)
            print(f"\nexample {k}: ref = {' '.join(words)}")
            for u, t in turns:
                print(f"  USER {u!r:40s} -> TARGET {t!r}")
        return

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map="cuda",
        local_files_only=args.local_files_only,
    )
    model.config.use_cache = False

    if args.resume:
        model = PeftModel.from_pretrained(model, args.resume, is_trainable=True)
        print(f"resumed adapter from {args.resume}")
    else:
        lcfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM", bias="none",
        )
        model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        return args.lr

    device = model.device
    print(f"\ntraining {args.steps} steps, batch {args.batch_size} ...\n" + "=" * 70)
    # baseline eval (adapter is near-identity at init but check the harness works)
    ep, et, w = evaluate(tok, model, eval_rng, pool, n_random=12)
    print(f"[eval @0] edge {ep}/{et}   random-WER {w*100:.1f}%")

    t0 = time.time()
    run_loss = 0.0
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        input_ids, attn, labels = make_batch(tok, rng, pool, args.batch_size, args.max_len, device)
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        run_loss += loss.item()

        if step % args.log_every == 0:
            avg = run_loss / args.log_every
            run_loss = 0.0
            rate = step / (time.time() - t0)
            print(f"step {step:5d}/{args.steps}  loss {avg:.4f}  lr {lr_at(step):.2e}  {rate:.2f} it/s")

        if step % args.eval_every == 0 or step == args.steps:
            ep, et, w = evaluate(tok, model, eval_rng, pool, n_random=16, verbose=(step == args.steps))
            print(f"[eval @{step}] edge {ep}/{et}   random-WER {w*100:.1f}%")
            model.save_pretrained(args.out)
            print(f"  saved adapter -> {args.out}")

    print("=" * 70 + f"\ndone. adapter at {args.out}")


if __name__ == "__main__":
    main()
