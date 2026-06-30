#!/usr/bin/env python
"""Run the 4 CHAT/streaming LibriSpeech models on the full Open-ASR-Leaderboard
suite and print a single side-by-side WER comparison table.

These are the models from the most recent LibriSpeech run (rno/avg_and_eval.py /
rno/eval_avg_local.sh), whose checkpoints already live locally under
``$HOME/Workplace/librispeech_runs/<run>/``. Each run is evaluated exactly the
way that driver does it:

  * average the run's epoch ``*.ckpt`` weights (``-last`` / ``-averaged`` excluded),
  * restore the run's ``*.nemo`` with ``loss_type chat_aligner -> rnnt`` (dropping the
    cluster-only ``external_aligner``), and decode with ``greedy_batch``;

but instead of the local LibriSpeech manifests it reuses ``run_eval_asr.py``'s HF
ESB / Open-ASR-Leaderboard loading + caching + whisper-normalized WER, so every
number is leaderboard-comparable. The audio for all suite datasets is already
cached under ``./audio_cache`` from earlier runs, so no re-download is needed.

Usage:
  python eval_leaderboard_chat.py                       # all 4 runs, full suite
  python eval_leaderboard_chat.py --only librispeech,ami
  python eval_leaderboard_chat.py --max_eval_samples 10 # quick smoke test
  python eval_leaderboard_chat.py --device 1 --batch_size 8
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "rno")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# HF dataset loading + caching + (whisper-)normalized WER, reused verbatim.
from run_eval_asr import (  # noqa: E402
    prepare_dataset,
    text_normalizer,
    transcribe_multistream,
    transcribe_tdt,
    wer_metric,
)

# Faithful CHAT-model loading (averaging + loss_type override), reused verbatim.
from avg_and_eval import (  # noqa: E402
    average_state_dicts,
    find_checkpoints,
    load_model_for_eval,
)
from omegaconf import OmegaConf  # noqa: E402

# The 4 models shown in the terminal output (run-folder names == dir under MODELS_ROOT).
DEFAULT_RUNS = [
    "rno_chat_extaligner_precomp_ctx72_5_ls960_fclarge_scratch_warmup10000_n8",
    "rno_chat_extaligner_precomp_ctx72_5_delayk8_d2_g0p8_ls960_fclarge_scratch_warmup10000_n8",
    "rno_chat_extaligner_precomp_ctx72_5_fixeddelay3_ls960_fclarge_scratch_warmup10000_n8",
    "rno_chat_streaming_ctx72_5_ls960_fclarge_scratch_warmup10000_n8",
]
DEFAULT_MODELS_ROOT = os.path.join(os.path.expanduser("~"), "Workplace", "librispeech_runs")

# Open-ASR-Leaderboard suite (dataset, split, hf_revision). TED-LIUM was removed
# from the hub on 2026-05-27; pin to the last commit that still has the parquet.
LEADERBOARD = [
    ("ami", "test", None),
    ("earnings22", "test", None),
    ("gigaspeech", "test", None),
    ("librispeech", "test.clean", None),
    ("librispeech", "test.other", None),
    ("spgispeech", "test", None),
    ("voxpopuli", "test", None),
    ("tedlium", "test", "20a009a"),
]
# Short column labels for the printed table.
COL_LABEL = {
    ("ami", "test"): "ami",
    ("earnings22", "test"): "earn22",
    ("gigaspeech", "test"): "giga",
    ("librispeech", "test.clean"): "ls-clean",
    ("librispeech", "test.other"): "ls-other",
    ("spgispeech", "test"): "spgi",
    ("voxpopuli", "test"): "vox",
    ("tedlium", "test"): "ted",
}
DATASET_PATH = "hf-audio/esb-datasets-test-only-sorted"


def short_run_label(run):
    """A compact, table-friendly label for a (long) run-folder name."""
    s = run
    for pref in ("rno_chat_extaligner_precomp_", "rno_chat_"):
        if s.startswith(pref):
            s = s[len(pref):]
            break
    s = s.replace("_ls960_fclarge_scratch_warmup10000_n8", "")
    return s or run


def resolve_run_paths(run, models_root):
    """Find the run's eval .nemo (architecture/tokenizer) and its checkpoint dir."""
    run_dir = run if os.path.isabs(run) else os.path.join(models_root, run)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run dir not found: {run_dir}")
    nemos = [p for p in sorted(Path(run_dir).glob("*.nemo")) if "-averaged" not in p.name]
    if not nemos:
        raise FileNotFoundError(f"no (non '-averaged') .nemo in {run_dir}")
    return str(nemos[0]), run_dir


def load_chat_model(nemo_path, ckpt_dir, device, decoding="greedy_batch"):
    """Average the run's checkpoints, restore the .nemo for inference, set decoding."""
    ckpts = find_checkpoints(ckpt_dir)
    if not ckpts:
        raise RuntimeError(f"no averageable *.ckpt in {ckpt_dir}")
    print(f"  averaging {len(ckpts)} checkpoint(s)")
    avg_state = average_state_dicts(ckpts)

    model = load_model_for_eval(nemo_path, device)
    missing, unexpected = model.load_state_dict(avg_state, strict=False)
    real_missing = [k for k in missing if not k.startswith(("loss", "_external_aligner"))]
    if real_missing:
        print(f"  [warn] {len(real_missing)} unmatched params (showing 5): {real_missing[:5]}")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected ckpt keys (showing 5): {unexpected[:5]}")
    model = model.to(device).eval()
    model.freeze()

    try:
        dec_cfg = OmegaConf.create(OmegaConf.to_container(model.cfg.decoding, resolve=True))
        dec_cfg.strategy = decoding
        model.change_decoding_strategy(dec_cfg)
    except Exception as e:  # not fatal; saved decoding cfg is already usable
        print(f"  [warn] change_decoding_strategy failed ({e}); using saved decoding cfg.")

    is_multistream = hasattr(model, "_decode_hyp_texts")
    return model, is_multistream, len(ckpts)


def eval_model_on_dataset(model, is_multistream, data, batch_size):
    """Transcribe one prepared dataset and return whisper-normalized WER (%)."""
    if is_multistream:
        transcriptions = transcribe_multistream(model, data["audio_filepaths"], batch_size)
    else:
        transcriptions = transcribe_tdt(model, data["audio_filepaths"], batch_size)
    predictions = [text_normalizer(t.strip()) for t in transcriptions]
    wer = wer_metric.compute(references=data["references"], predictions=predictions)
    return round(100 * wer, 2)


def print_table(runs, run_labels, datasets, results):
    """results[run][(ds,split)] -> WER float | 'N/A' | 'FAIL'."""
    col_keys = [(d, s) for (d, s, _) in datasets]
    headers = [COL_LABEL.get(k, f"{k[0]}/{k[1]}") for k in col_keys] + ["Avg"]
    label_w = max([len("model")] + [len(run_labels[r]) for r in runs])
    col_w = max(8, *(len(h) for h in headers))

    def cell(v):
        return f"{v:>{col_w}.2f}" if isinstance(v, float) else f"{v:>{col_w}}"

    line = f"{'model':<{label_w}}  " + "  ".join(f"{h:>{col_w}}" for h in headers)
    print("\n" + "=" * len(line))
    print("ASR leaderboard WER (%) -- whisper-normalized, lower is better")
    print("=" * len(line))
    print(line)
    print("-" * len(line))
    for r in runs:
        row = results[r]
        vals = [row.get(k, "N/A") for k in col_keys]
        nums = [v for v in vals if isinstance(v, float)]
        avg = round(sum(nums) / len(nums), 2) if nums else "N/A"
        cells = "  ".join(cell(v) for v in vals + [avg])
        print(f"{run_labels[r]:<{label_w}}  {cells}")
    print("=" * len(line) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models_root", default=DEFAULT_MODELS_ROOT, help="Dir holding the run subfolders.")
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS, help="Run-folder names (or absolute dirs).")
    ap.add_argument("--only", default=None, help="Comma-separated dataset-name filter, e.g. 'librispeech,ami'.")
    ap.add_argument("--device", type=int, default=0, help="CUDA device index (<0 for CPU).")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--decoding", default="greedy_batch", choices=["greedy", "greedy_batch"])
    ap.add_argument("--max_eval_samples", type=int, default=None, help="Cap samples per dataset (quick test).")
    ap.add_argument("--output_json", default=None, help="Optional path to dump the full WER matrix.")
    args = ap.parse_args()

    device = f"cuda:{args.device}" if (args.device is not None and args.device >= 0 and torch.cuda.is_available()) else "cpu"
    if device == "cpu":
        print("[warn] CUDA unavailable (or --device<0); running on CPU will be slow.")

    datasets = LEADERBOARD
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        datasets = [d for d in LEADERBOARD if d[0] in wanted]
        if not datasets:
            raise SystemExit(f"--only '{args.only}' matched no datasets in the suite.")

    run_labels = {r: short_run_label(os.path.basename(r.rstrip("/"))) for r in args.runs}

    # Prepare (cache/normalize) each dataset ONCE, reused across all models.
    print("==> Preparing leaderboard datasets (audio cached under ./audio_cache) ...")
    datasets_data = {}
    for (ds, split, rev) in datasets:
        print(f"---- {ds}/{split} ----")
        datasets_data[(ds, split)] = prepare_dataset(
            dataset_path=DATASET_PATH,
            dataset=ds,
            split=split,
            dataset_revision=rev,
            streaming=True,
            max_eval_samples=args.max_eval_samples,
        )

    results = {r: {} for r in args.runs}
    for r in args.runs:
        print("\n" + "=" * 70)
        print(f">> MODEL: {r}")
        print("=" * 70)
        try:
            nemo_path, ckpt_dir = resolve_run_paths(r, args.models_root)
            model, is_multistream, n_avg = load_chat_model(nemo_path, ckpt_dir, device, args.decoding)
            print(f"  loaded ({n_avg} ckpts averaged, multistream={is_multistream})")
        except Exception as e:
            print(f"  !! load FAILED for {r}: {e}")
            for (ds, split, _) in datasets:
                results[r][(ds, split)] = "FAIL"
            continue

        for (ds, split, _) in datasets:
            data = datasets_data[(ds, split)]
            try:
                wer = eval_model_on_dataset(model, is_multistream, data, args.batch_size)
                print(f"  {ds}/{split}: WER {wer:.2f}% (n={len(data['references'])})")
                results[r][(ds, split)] = wer
            except Exception as e:
                print(f"  !! {ds}/{split} FAILED: {e}")
                results[r][(ds, split)] = "N/A"

        # Free GPU memory before loading the next model.
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_table(args.runs, run_labels, datasets, results)

    if args.output_json:
        serializable = {
            run_labels[r]: {f"{ds}/{split}": results[r].get((ds, split)) for (ds, split, _) in datasets}
            for r in args.runs
        }
        with open(args.output_json, "w") as fh:
            json.dump({"dataset_path": DATASET_PATH, "decoding": args.decoding, "results": serializable}, fh, indent=2)
        print(f"WER matrix written to {os.path.abspath(args.output_json)}")


if __name__ == "__main__":
    main()
