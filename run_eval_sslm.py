"""
ASR leaderboard evaluation for StreamingSTTModel.

Loads a Lightning checkpoint, evaluates on HuggingFace ESB datasets,
and reports WER + RTFx.  Results are written in the same format used by
the open_asr_leaderboard so that ``eval_utils.score_results()`` works.

Usage:
    python run_eval_sslm.py \
        --ckpt_path /path/to/step=8000.ckpt \
        --dataset librispeech --split test.clean \
        --device 0 --batch_size 64

    # With explicit model overrides (if auto-resolution fails):
    python run_eval_sslm.py \
        --ckpt_path /path/to/step=8000.ckpt \
        --pretrained_llm Qwen/Qwen3-1.7B \
        --pretrained_asr nvidia/nemotron-speech-streaming-en-0.6b \
        --dataset librispeech --split test.clean --device 0
"""
import argparse
import contextlib
import io
import json
import logging
import os
import time
import warnings

# ---------------------------------------------------------------------------
# Quiet mode: this script only emits an on-the-fly results table. Everything
# else (framework banners, JIT notices, deprecation warnings, progress bars)
# is silenced. Env vars must be set BEFORE the heavy imports below.
#   - HF_AUDIO_DECODER=soundfile avoids torchcodec/FFmpeg issues.
# ---------------------------------------------------------------------------
os.environ["HF_AUDIO_DECODER"] = "soundfile"
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DATASETS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)


@contextlib.contextmanager
def _suppress_output():
    """Silence stdout+stderr at the file-descriptor level.

    This catches not just Python ``print``/``logging`` output but also the
    C-extension banners emitted by NeMo / flashinfer / OneLogger during import
    and model construction. Exceptions still propagate (the fds are restored in
    ``finally``, so any traceback prints normally).
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_out, saved_err = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(devnull)
        os.close(saved_out)
        os.close(saved_err)


import numpy as np
import soundfile
import torch
from tqdm import tqdm

with _suppress_output():
    import lhotse
    import lhotse.dataset
    from datasets import load_dataset, Audio
    from whisper_normalizer.english import EnglishTextNormalizer
    import evaluate

    from nemo.collections.speechlm2.models import StreamingSTTModel

    wer_metric = evaluate.load("wer")
    text_normalizer = EnglishTextNormalizer()

# Standard open-ASR-leaderboard ESB test suite (dataset, split), evaluated when
# --dataset is not given. Matches the list in eval_leaderboard.sh.
LEADERBOARD_DATASETS = [
    ("ami", "test"),
    ("earnings22", "test"),
    ("gigaspeech", "test"),
    ("librispeech", "test.clean"),
    ("librispeech", "test.other"),
    ("spgispeech", "test"),
    ("tedlium", "test"),
    ("voxpopuli", "test"),
]


# ---------------------------------------------------------------------------
# Audio loading for inference (lhotse-based batched dataloader)
# ---------------------------------------------------------------------------

class ToAudio(torch.utils.data.Dataset):
    def __getitem__(self, cuts):
        cuts = lhotse.CutSet(
            [c.to_mono(mono_downmix=True) if isinstance(c, lhotse.MultiCut) else c for c in cuts]
        )
        audios, audio_lens = cuts.load_audio(collate=True)
        return {"cuts": cuts, "audios": audios, "audio_lens": audio_lens}


def setup_dloader(audio_files, batch_size, num_workers=1):
    cuts = lhotse.CutSet([lhotse.Recording.from_file(p).to_cut() for p in audio_files])
    cuts = cuts.resample(16000)
    return torch.utils.data.DataLoader(
        dataset=ToAudio(),
        sampler=lhotse.dataset.DynamicCutSampler(cuts, max_cuts=batch_size),
        num_workers=num_workers,
        batch_size=None,
    )


def _tail_flush_pad_samples(model) -> int:
    """Number of trailing silence samples to append to each utterance.

    Streaming models are trained to emit each word a few frames AFTER its audio
    ends (``num_delay_frames`` + ``random_delay_max_frames``), and in training
    those deferred emissions land in the trailing-silence chunks naturally
    present in the cut. Leaderboard audio is trimmed tight, so at inference the
    last word's emission is deferred into a chunk that is never fed, producing
    end-of-utterance deletions. We append a couple of silent chunks so the model
    has room to flush that tail. Only meaningful for fixed-chunk streaming.
    """
    chunk_size = int(getattr(model.core_cfg, "chunk_size", 0))
    if chunk_size <= 0:
        return 0
    # Models trained with the explicit <flush> token flush their own tail at
    # inference (model._flush_step), so the silence-pad hack is unnecessary.
    if bool(getattr(model.core_cfg, "use_flush", False)):
        return 0
    sr = int(getattr(model.core_cfg, "sample_rate", 16000))
    frame_len = float(getattr(model.core_cfg, "frame_length_in_secs", 0.08))
    flush_chunks = 2
    return int(round(flush_chunks * chunk_size * frame_len * sr))


def transcribe_sslm(model, dloader, system_prompt, max_new_tokens, no_repeat_ngram_size) -> list[str]:
    from transformers import GenerationConfig
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )

    pad_samples = _tail_flush_pad_samples(model)

    hyps = []
    for batch in tqdm(dloader, desc="Transcribing", disable=True):
        audios = batch["audios"]
        audio_lens = batch["audio_lens"]
        if pad_samples > 0:
            audios = torch.nn.functional.pad(audios, (0, pad_samples))
            audio_lens = audio_lens + pad_samples
        with torch.inference_mode():
            batch_hyps = model.generate(
                audios=audios.to(model.device, non_blocking=True),
                audio_lens=audio_lens.to(model.device, non_blocking=True),
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                generation_config=gen_cfg,
            )
        hyps.extend(batch_hyps)
    return hyps


# ---------------------------------------------------------------------------
# Model loading from Lightning checkpoint
# ---------------------------------------------------------------------------

def _resolve_remote_path(path: str) -> str:
    """Convert an OCI/cluster-local path to a Hub ID (org/model).

    Examples:
        /lustre/.../huggingface/Qwen/Qwen3-1.7B           ->  Qwen/Qwen3-1.7B
        /lustre/.../huggingface/nvidia/model/model.nemo     ->  nvidia/model
        nvidia/parakeet-tdt-0.6b-v2                        ->  (unchanged)
    """
    if not path.startswith("/") or os.path.exists(path):
        return path
    parts = path.rstrip("/").split("/")
    try:
        hf_idx = parts.index("huggingface")
        remainder = [p for p in parts[hf_idx + 1:] if p]
        if len(remainder) >= 2:
            return f"{remainder[0]}/{remainder[1]}"
        elif remainder:
            return remainder[0]
    except ValueError:
        pass
    non_file = [p for p in parts if p and not p.endswith((".nemo", ".ckpt", ".bin"))]
    if len(non_file) >= 2:
        return f"{non_file[-2]}/{non_file[-1]}"
    return path


def load_model(ckpt_path, device, override_llm=None, override_asr=None, dtype=torch.bfloat16):
    """Load StreamingSTTModel from a Lightning checkpoint.

    Handles:
      - Resolving OCI-local pretrained paths to Hub IDs (auto-downloads if needed)
      - Skipping LLM pretrained weight loading (only config/tokenizer needed)
      - Adding missing LoRA target_modules for newer PEFT versions
    """
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["hyper_parameters"]["cfg"]
    state_dict = ckpt["state_dict"]

    # --- LLM path: only need AutoConfig + tokenizer (weights come from ckpt) ---
    cfg["load_llm_weights"] = False
    orig_llm = cfg.get("pretrained_llm", "")
    if override_llm:
        cfg["pretrained_llm"] = override_llm
        print(f"  Override pretrained_llm: {orig_llm} -> {override_llm}")
    else:
        resolved = _resolve_remote_path(orig_llm)
        if resolved != orig_llm:
            print(f"  Resolved pretrained_llm: {orig_llm} -> {resolved}")
            cfg["pretrained_llm"] = resolved

    # --- ASR path: must load to populate preprocessor/encoder config ---
    # Weights will be overwritten by checkpoint state_dict afterwards.
    orig_asr = cfg.get("pretrained_asr", "")
    if override_asr:
        cfg["pretrained_asr"] = override_asr
        print(f"  Override pretrained_asr: {orig_asr} -> {override_asr}")
    else:
        resolved = _resolve_remote_path(orig_asr)
        if resolved != orig_asr:
            print(f"  Resolved pretrained_asr: {orig_asr} -> {resolved}")
            cfg["pretrained_asr"] = resolved

    # --- LoRA: newer PEFT requires target_modules ---
    # Auto-detect from checkpoint state_dict keys to match exactly what was trained.
    if "lora" in cfg and "target_modules" not in cfg["lora"]:
        lora_modules = set()
        for key in state_dict:
            if ".lora_A." in key:
                # e.g. "llm.base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
                #  -> extract "q_proj"
                parts = key.split(".")
                lora_idx = parts.index("lora_A")
                if lora_idx > 0:
                    lora_modules.add(parts[lora_idx - 1])
        if lora_modules:
            cfg["lora"]["target_modules"] = sorted(lora_modules)
            print(f"  Auto-detected LoRA target_modules from checkpoint: {cfg['lora']['target_modules']}")
        else:
            cfg["lora"]["target_modules"] = "all-linear"
            print("  No LoRA keys found in checkpoint, using target_modules='all-linear'")

    print("  Constructing model (will download pretrained ASR/LLM configs if needed)...")
    model = StreamingSTTModel(cfg=cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    del ckpt

    model = model.eval().to(dtype).to(device)
    print(f"  Model loaded on {device}, dtype={dtype}")
    return model


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def get_text(sample):
    for key in ("text", "sentence", "normalized_text", "transcript", "transcription"):
        if key in sample:
            return sample[key]
    raise ValueError(f"No transcript column found in sample keys: {list(sample.keys())}")


def write_manifest(references, predictions, model_id, dataset_path, dataset_name, split,
                   audio_length=None, transcription_time=None):
    model_id_safe = model_id.replace("/", "-")
    dataset_path_safe = dataset_path.replace("/", "-")
    dataset_name_safe = dataset_name.replace("/", "-")

    basedir = "./results/"
    os.makedirs(basedir, exist_ok=True)

    manifest_path = os.path.join(
        basedir,
        f"MODEL_{model_id_safe}_DATASET_{dataset_path_safe}_{dataset_name_safe}_{split}.jsonl",
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        for idx in range(len(references)):
            datum = {
                "audio_filepath": f"sample_{idx}",
                "duration": audio_length[idx] if audio_length else None,
                "time": transcription_time[idx] if transcription_time else None,
                "text": references[idx],
                "pred_text": predictions[idx],
            }
            f.write(json.dumps(datum, ensure_ascii=False) + "\n")
    return manifest_path


def dump_nemo_manifest(manifest_path, audio_filepaths, durations, references):
    """Write a NeMo-style manifest (audio_filepath/duration/text) and exit.

    Used by the VLLM fast path in eval_leaderboard_ord.sh: this reuses the exact
    same HF loading / soundfile caching / whisper normalization as the in-process
    eval so the wavs and references match, then hands off decoding to the vLLM
    container instead of model.generate().
    """
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for path, dur, ref in zip(audio_filepaths, durations, references):
            f.write(json.dumps(
                {"audio_filepath": os.path.abspath(path), "duration": dur, "text": ref},
                ensure_ascii=False,
            ) + "\n")
    print("Manifest written to:", os.path.abspath(manifest_path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _hf_token():
    # Prefer an explicit env token (HF_TOKEN / HUGGING_FACE_HUB_TOKEN);
    # `token=True` (cached CLI login) errors if you never ran
    # `huggingface-cli login`, so fall back to None (anonymous) when unset.
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or None
    )


def prepare_samples(args, dataset, split):
    """Load + cache audio for one (dataset, split). Returns the all_data dict
    (audio_filepaths/durations/references), sorted longest-first for batching."""
    CACHE_DIR = os.path.join(os.getcwd(), "audio_cache", dataset, split)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # trust_remote_code is dropped: newer `datasets` rejects it and the ESB
    # test sets are plain Parquet (no loading script needed).
    ds = load_dataset(
        args.dataset_path, dataset, split=split,
        streaming=args.streaming, token=_hf_token(),
    )
    # Disable audio decoding — we decode manually with soundfile to avoid
    # torchcodec/FFmpeg compatibility issues.
    ds = ds.cast_column("audio", Audio(decode=False))

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        ds = ds.take(args.max_eval_samples)

    all_data = {"audio_filepaths": [], "durations": [], "references": []}
    for sample in ds:
        ref = text_normalizer(get_text(sample))
        if not ref.strip() or ref.strip() == "ignore time segment in scoring":
            continue

        raw_audio = sample["audio"]
        sample_id = sample["id"].replace("/", "_").removesuffix(".wav")
        audio_path = os.path.join(CACHE_DIR, f"{sample_id}.wav")

        if not os.path.exists(audio_path):
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            if isinstance(raw_audio, dict):
                if "array" in raw_audio and raw_audio["array"] is not None:
                    audio_array = np.float32(raw_audio["array"])
                    sr = raw_audio.get("sampling_rate", 16000)
                elif "bytes" in raw_audio and raw_audio["bytes"]:
                    with io.BytesIO(raw_audio["bytes"]) as f:
                        audio_array, sr = soundfile.read(f, dtype="float32")
                elif "path" in raw_audio and raw_audio["path"] and os.path.exists(raw_audio["path"]):
                    audio_array, sr = soundfile.read(raw_audio["path"], dtype="float32")
                else:
                    continue
            else:
                continue

            if sr != 16000:
                import torchaudio
                t = torch.from_numpy(audio_array).unsqueeze(0) if audio_array.ndim == 1 else torch.from_numpy(audio_array)
                t = torchaudio.functional.resample(t, sr, 16000)
                audio_array = t.squeeze(0).numpy()

            soundfile.write(audio_path, audio_array, 16000)

        info = soundfile.info(audio_path)
        all_data["audio_filepaths"].append(audio_path)
        all_data["durations"].append(info.duration)
        all_data["references"].append(ref)

    # Sort by duration (longest first) for efficient batching
    sorted_idx = sorted(range(len(all_data["durations"])), key=lambda k: all_data["durations"][k], reverse=True)
    for key in all_data:
        all_data[key] = [all_data[key][i] for i in sorted_idx]
    return all_data


def evaluate_one(model, args, dataset, split):
    """Prepare + transcribe + score one (dataset, split). Returns a result dict
    or None when there are no samples. Caller is responsible for any logging;
    this function is intended to run inside ``_suppress_output()``."""
    all_data = prepare_samples(args, dataset, split)
    n_samples = len(all_data["references"])
    if n_samples == 0:
        return None

    dloader = setup_dloader(all_data["audio_filepaths"], batch_size=args.batch_size)
    start = time.time()
    transcriptions = transcribe_sslm(
        model, dloader,
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    total_time = time.time() - start
    predictions = [text_normalizer(pred.strip()) for pred in transcriptions]

    avg_time = total_time / n_samples
    model_label = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(args.ckpt_path))))
    write_manifest(
        all_data["references"], predictions, model_label,
        args.dataset_path, dataset, split,
        audio_length=all_data["durations"],
        transcription_time=[avg_time] * n_samples,
    )

    wer = round(100 * wer_metric.compute(references=all_data["references"], predictions=predictions), 2)
    rtfx = round(sum(all_data["durations"]) / total_time, 2)
    return {
        "wer": wer, "rtfx": rtfx, "time": total_time, "n": n_samples,
        "refs": all_data["references"], "hyps": predictions,
    }


def main(args):
    torch.set_float32_matmul_precision("medium")

    # ---- Manifest-dump fast path (single dataset; used by the vLLM path) ----
    if args.dump_manifest:
        if args.dataset is None:
            raise SystemExit("--dump_manifest requires --dataset")
        with _suppress_output():
            all_data = prepare_samples(args, args.dataset, args.split)
        dump_nemo_manifest(
            args.dump_manifest,
            all_data["audio_filepaths"],
            all_data["durations"],
            all_data["references"],
        )
        return

    # ---- Datasets to evaluate: one explicit set, or the full ESB suite ----
    datasets = [(args.dataset, args.split)] if args.dataset is not None else list(LEADERBOARD_DATASETS)

    with _suppress_output():
        device = torch.device(f"cuda:{args.device}")
        model = load_model(
            args.ckpt_path, device,
            override_llm=args.pretrained_llm,
            override_asr=args.pretrained_asr,
        )

    print(f"\nckpt: {args.ckpt_path}")
    print(f"{'dataset':<24}{'WER%':>9}{'time(s)':>11}", flush=True)
    print("-" * 44, flush=True)

    results = []
    for dataset, split in datasets:
        name = f"{dataset}/{split}"
        try:
            with _suppress_output():
                res = evaluate_one(model, args, dataset, split)
        except Exception as e:
            print(f"{name:<24}{'ERR':>9}{'':>11}  ({type(e).__name__}: {e})", flush=True)
            continue
        if res is None:
            print(f"{name:<24}{'n/a':>9}{'':>11}  (no samples)", flush=True)
            continue
        print(f"{name:<24}{res['wer']:>9.2f}{res['time']:>11.1f}", flush=True)
        results.append((name, res))
        if args.verbose:
            for i, (ref, hyp) in enumerate(zip(res["refs"], res["hyps"])):
                print(f"    [{i}] REF: {ref}")
                print(f"    [{i}] HYP: {hyp}")

    if len(results) > 1:
        avg = sum(r["wer"] for _, r in results) / len(results)
        print("-" * 44, flush=True)
        print(f"{'AVERAGE':<24}{avg:>9.2f}", flush=True)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR leaderboard eval for StreamingSTTModel")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to Lightning .ckpt file")
    parser.add_argument("--pretrained_llm", type=str, default=None,
                        help="Override pretrained_llm (e.g. Qwen/Qwen3-1.7B). Auto-resolved if not set.")
    parser.add_argument("--pretrained_asr", type=str, default=None,
                        help="Override pretrained_asr (e.g. nvidia/nemotron-speech-streaming-en-0.6b). Auto-resolved if not set.")
    parser.add_argument("--dataset_path", type=str, default="hf-audio/esb-datasets-test-only-sorted")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Single dataset to evaluate. If omitted, runs the full ESB leaderboard suite.")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Max tokens to generate per chunk (streaming) or per utterance (offline).")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=4,
                        help="Disallow repeating n-grams during generation to break loops. Set 0 to disable.")
    parser.add_argument("--system_prompt", type=str, default="Transcribe the audio into text.")
    parser.add_argument("--verbose", action="store_true", help="Print each REF/HYP pair to stdout")
    parser.add_argument("--dump_manifest", type=str, default=None,
                        help="Instead of decoding, cache audio to 16k wav and write a NeMo manifest "
                             "(audio_filepath/duration/text) to this path, then exit. Used by the "
                             "VLLM fast path in eval_leaderboard_ord.sh.")
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.set_defaults(streaming=True)
    args = parser.parse_args()
    main(args)
