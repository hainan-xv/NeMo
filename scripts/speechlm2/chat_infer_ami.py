"""Run a trained CHAT transducer over the AMI test set and report WER.

    # quick smoke test: 20 utterances
    PYTHONPATH=. python scripts/speechlm2/chat_infer_ami.py --ckpt <path> --quick

    # 20 utterances, with retract-by-1 decoding
    PYTHONPATH=. python scripts/speechlm2/chat_infer_ami.py --ckpt <path> --quick --retract 1

    # the full 7,805-utterance test set
    PYTHONPATH=. python scripts/speechlm2/chat_infer_ami.py --ckpt <path>

WHY A SEPARATE SCRIPT FROM THE LEADERBOARD DRIVER. That one runs seven datasets
under Slurm on eight GPUs and needs the whole cache staged. This is for the loop
you actually want while iterating -- one dataset, one GPU, twenty utterances,
answer in under a minute -- and for sweeping decode-time knobs (--retract) on a
checkpoint without retraining.

WER is computed the same way as the leaderboard driver (Whisper's English
normaliser + kaldialign edit distance), so a number here is comparable to the
AMI column there. It is NOT comparable to val_wer, which runs on a different
(mcv-style) validation set.
"""

import argparse
import json
import os
import time

import torch


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="training .ckpt from the grid")
    ap.add_argument(
        "--manifest",
        default=os.path.expanduser("~/leaderboard_cache/ami_cleaned/test/_cache_manifest.jsonl"),
        help="AMI cache manifest (audio_filepath / duration / reference per line)",
    )
    ap.add_argument("--quick", action="store_true", help="only the first --n utterances")
    ap.add_argument("-n", type=int, default=20, help="utterances to run under --quick (default 20)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--retract",
        type=int,
        default=None,
        help="retract-by-k decoding: hand each chunk's last k words to the next chunk so the model "
        "re-decides them with more right context. Default: whatever the checkpoint was configured with.",
    )
    ap.add_argument("--max-duration", type=float, default=None, help="skip utterances longer than this (seconds)")
    ap.add_argument("--print-worst", type=int, default=5, help="show the N worst utterances at the end")
    ap.add_argument("--pretrained-asr", default=None, help="local .nemo, if the recorded path is unavailable")
    ap.add_argument("--pretrained-llm", default=None, help="local path or hub id for the Qwen-arm tokenizer")
    ap.add_argument("--out", default=None, help="write per-utterance hypotheses to this .jsonl")
    return ap.parse_args()


def _resolve_asr(recorded: str, override: str = None) -> str:
    """Local path to the pretrained ASR .nemo the checkpoint was built on."""
    import glob

    for cand in (override, recorded):
        if cand and os.path.exists(cand):
            return cand
    hits = sorted(
        glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--*/**/*.nemo"), recursive=True)
    )
    if recorded:
        want = os.path.basename(recorded)
        exact = [h for h in hits if os.path.basename(h) == want]
        if exact:
            print(f"  pretrained_asr remapped to local cache: {exact[0]}")
            return exact[0]
    if hits:
        print(f"  pretrained_asr remapped to local cache: {hits[0]}")
        return hits[0]
    raise FileNotFoundError(
        f"cannot find the pretrained ASR model locally (checkpoint recorded {recorded!r}). "
        "Pass --pretrained-asr with a local .nemo."
    )


def _resolve_llm(recorded: str, override: str = None) -> str:
    """Local path or hub id for the LLM whose tokenizer the Qwen arm uses."""
    if override:
        return override
    if recorded and os.path.exists(recorded):
        return recorded
    # A hub id works if the tokenizer is already cached, which it is if this
    # machine has ever built the Qwen arm.
    guess = "Qwen/" + os.path.basename(recorded.rstrip("/")) if recorded else "Qwen/Qwen3-1.7B"
    print(f"  pretrained_llm remapped to hub id: {guess}")
    return guess


def load_model(ckpt_path, device, retract, asr_override=None, llm_override=None):
    """Rebuild ChatSTTModel from the checkpoint's own hyper_parameters.

    The config is read from the checkpoint rather than from a YAML on disk: the
    recipes have changed repeatedly (vocabulary, joint_history_chunks, delay),
    and pairing a checkpoint with a drifted config silently builds the wrong
    model -- usually a shape error, but at worst a quiet mismatch.
    """
    from omegaconf import OmegaConf

    from nemo.collections.speechlm2.models.chat_model import ChatSTTModel

    print(f"loading {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ck.get("hyper_parameters", {})
    cfg = hp.get("cfg", hp)
    cfg = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True) if not isinstance(cfg, dict) else dict(cfg)

    # Warm-starting from the donor RNN-T is a TRAINING-time convenience; here it
    # would just be overwritten by the checkpoint's own weights, at the cost of
    # reading a 2.5 GB .nemo.
    cfg["init_rnnt_from_asr"] = False

    # The checkpoint records the CLUSTER's paths for the pretrained encoder and
    # LLM, which do not exist on a desktop. Remap them to local copies rather
    # than making the caller edit a config: leaving them unresolved fails deep
    # inside HuggingFace with an unhelpful "Repo id must be in the form ..."
    # because NeMo falls back to treating the path as a hub name.
    cfg["pretrained_asr"] = _resolve_asr(cfg.get("pretrained_asr", ""), asr_override)
    if not cfg.get("text_vocab_from_asr", True):
        cfg["pretrained_llm"] = _resolve_llm(cfg.get("pretrained_llm", ""), llm_override)
    if retract is not None:
        cfg["retract_words"] = int(retract)

    model = ChatSTTModel(cfg)
    sd = ck["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if not k.startswith("perception.preprocessor")]
    if real_missing:
        raise RuntimeError(f"checkpoint is missing {len(real_missing)} parameters, e.g. {real_missing[:5]}")
    if unexpected:
        print(f"  ignoring {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")

    model = model.to(device).eval()
    print(
        f"  vocab={cfg.get('vocab_size')}  chunk_size={cfg.get('chunk_size')}  "
        f"joint_history_chunks={cfg.get('joint_history_chunks', 0)}  retract_words={cfg.get('retract_words', 0)}"
    )
    return model, cfg


def build_tokenizer(cfg):
    """The same tokenizer the run trained with -- the arm's defining choice."""
    import tempfile

    if cfg.get("text_vocab_from_asr", True):
        from nemo.collections.speechlm2.data.script_dataset import ScriptSTTDataset
        from nemo.collections.speechlm2.parts.asr_vocab import AsrVocabTokenizer, extract_spm_from_nemo

        spm = extract_spm_from_nemo(cfg["pretrained_asr"], tempfile.mkdtemp(prefix="chat_vocab_"))
        EOT = "<|im_end|>"
        specials = [ScriptSTTDataset.audio_open_token, ScriptSTTDataset.audio_close_token, EOT]
        tok = AsrVocabTokenizer(spm, special_tokens=specials, eos_token=EOT, pad_token=EOT)
        print(f"  tokenizer: ASR SentencePiece, {len(tok)} pieces")
        return tok

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

    tok = AutoTokenizer(cfg["pretrained_llm"], use_fast=True)
    print(f"  tokenizer: {cfg['pretrained_llm']}, {len(tok.tokenizer)} pieces")
    return tok


def main():
    args = parse_args()

    rows = []
    with open(args.manifest) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.max_duration:
        rows = [r for r in rows if r.get("duration", 0) <= args.max_duration]
    if args.quick:
        rows = rows[: args.n]
    print(f"{len(rows)} utterances, {sum(r.get('duration', 0) for r in rows)/60:.1f} min of audio")

    model, cfg = load_model(args.ckpt, args.device, args.retract, args.pretrained_asr, args.pretrained_llm)
    model.tokenizer = build_tokenizer(cfg)

    import soundfile as sf

    from nemo.collections.speechlm2.parts.metrics.wer import WER

    hyps, refs, per_utt = [], [], []
    t0 = time.perf_counter()
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i : i + args.batch_size]
        waves = []
        for r in batch:
            w, sr = sf.read(r["audio_filepath"], dtype="float32")
            if w.ndim > 1:
                w = w.mean(axis=1)
            assert sr == 16000, f"expected 16 kHz, got {sr} in {r['audio_filepath']}"
            waves.append(torch.from_numpy(w))
        lens = torch.tensor([len(w) for w in waves])
        padded = torch.zeros(len(waves), int(lens.max()))
        for j, w in enumerate(waves):
            padded[j, : len(w)] = w

        with torch.no_grad():
            ids = model.transcribe_ids(padded.to(args.device), lens.to(args.device))
        for r, seq in zip(batch, ids):
            hyp = model.tokenizer.ids_to_text(list(seq)) if seq else ""
            hyps.append(hyp)
            refs.append(r["reference"])
            per_utt.append({"audio_filepath": r["audio_filepath"], "reference": r["reference"], "hypothesis": hyp})
        done = min(i + args.batch_size, len(rows))
        print(f"  {done}/{len(rows)}", end="\r", flush=True)

    elapsed = time.perf_counter() - t0
    audio_s = sum(r.get("duration", 0) for r in rows)
    print(f"\ndecoded in {elapsed:.1f}s  ({audio_s/max(elapsed,1e-9):.1f}x realtime)")

    wer = WER(normalize=True, verbose=False)
    wer.update("ami", refs=refs, hyps=hyps)
    scores = {k: float(v) for k, v in wer.compute().items()}
    print("\n=== AMI ===")
    for k, v in sorted(scores.items()):
        print(f"  {k}: {v:.4f}")

    # Per-utterance WER, to see WHERE it fails rather than only how much.
    import kaldialign
    from whisper_normalizer.english import EnglishTextNormalizer

    norm = EnglishTextNormalizer()
    scored = []
    for u in per_utt:
        r, h = norm(u["reference"]).split(), norm(u["hypothesis"]).split()
        if not r:
            continue
        ali = kaldialign.align(r, h, "*")
        err = sum(1 for a, b in ali if a != b)
        u["wer"] = err / len(r)
        scored.append(u)
    scored.sort(key=lambda x: -x["wer"])

    if args.print_worst and scored:
        print(f"\n=== {min(args.print_worst, len(scored))} worst utterances ===")
        for u in scored[: args.print_worst]:
            print(f"\n  wer={u['wer']:.3f}  {os.path.basename(u['audio_filepath'])}")
            print(f"    ref: {u['reference'][:220]}")
            print(f"    hyp: {u['hypothesis'][:220]}")

    if args.out:
        with open(args.out, "w") as f:
            for u in scored:
                f.write(json.dumps(u) + "\n")
        print(f"\nwrote {len(scored)} hypotheses to {args.out}")


if __name__ == "__main__":
    main()
