"""
ASR leaderboard evaluation for StreamingSTTModel.

Loads a Lightning checkpoint, evaluates on HuggingFace ESB datasets,
and reports WER + RTFx.  Results are written in the same format used by
the open_asr_leaderboard so that ``eval_utils.score_results()`` works.

Usage:
    # By grid EXP_NAME (auto-downloads the best checkpoint from OCI):
    python run_eval_sslm.py imend_flush --device 0

    # By local checkpoint path:
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
import gc
import io
import json
import logging
import os
import time
import warnings
import itertools

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


def transcribe_sslm(model, dloader, system_prompt, max_new_tokens, no_repeat_ngram_size, warmup_decode=False,
                    inference_audio_chunks_per_turn=1) -> list[str]:
    from transformers import GenerationConfig
    if max_new_tokens is None:
        max_new_tokens = int(
            getattr(model.core_cfg, "max_new_tokens_per_chunk", 10)
        )
    # Match training validation decode exactly when no extra decoding constraint is
    # requested: validation calls model.generate(..., generation_config=None).
    # Only construct a GenerationConfig for non-default eval-time constraints.
    gen_cfg = None
    if no_repeat_ngram_size and no_repeat_ngram_size > 0:
        gen_cfg = GenerationConfig(
            do_sample=False,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

    pad_samples = _tail_flush_pad_samples(model)

    def _prepare_batch(batch):
        audios = batch["audios"]
        audio_lens = batch["audio_lens"]
        if pad_samples > 0:
            audios = torch.nn.functional.pad(audios, (0, pad_samples))
            audio_lens = audio_lens + pad_samples
        return audios, audio_lens

    iterator = iter(dloader)
    try:
        first_batch = next(iterator)
    except StopIteration:
        return []

    if warmup_decode:
        audios, audio_lens = _prepare_batch(first_batch)
        with torch.inference_mode():
            _ = model.generate(
                audios=audios.to(model.device, non_blocking=True),
                audio_lens=audio_lens.to(model.device, non_blocking=True),
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                generation_config=gen_cfg,
                inference_audio_chunks_per_turn=inference_audio_chunks_per_turn,
            )

    hyps = []
    for batch in tqdm(itertools.chain([first_batch], iterator), desc="Transcribing", disable=True):
        audios, audio_lens = _prepare_batch(batch)
        with torch.inference_mode():
            batch_hyps = model.generate(
                audios=audios.to(model.device, non_blocking=True),
                audio_lens=audio_lens.to(model.device, non_blocking=True),
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                generation_config=gen_cfg,
                inference_audio_chunks_per_turn=inference_audio_chunks_per_turn,
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
        for idx, (path, dur, ref) in enumerate(zip(audio_filepaths, durations, references)):
            abspath = os.path.abspath(path)
            # b_streaming_infer.py requires a per-utterance "id"; derive a stable
            # one from the audio filename (fall back to the row index).
            utt_id = os.path.splitext(os.path.basename(abspath))[0] or f"utt_{idx}"
            f.write(json.dumps(
                {"id": utt_id, "audio_filepath": abspath, "duration": dur, "text": ref},
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
    this function is intended to run inside ``_suppress_output()``.

    On CUDA out-of-memory the transcription is retried with the batch size
    repeatedly halved (down to ``--min_batch_size``, default 1) after freeing GPU
    memory, so a too-large ``--batch_size`` degrades gracefully instead of
    crashing. The batch size that finally succeeded is returned as
    ``result['batch_size']``."""
    all_data = prepare_samples(args, dataset, split)
    n_samples = len(all_data["references"])
    if n_samples == 0:
        return None

    min_bs = max(1, int(getattr(args, "min_batch_size", 1) or 1))
    bs = max(min_bs, int(args.batch_size))
    start = None
    transcriptions = None
    while True:
        try:
            dloader = setup_dloader(all_data["audio_filepaths"], batch_size=bs)
            start = time.time()
            transcriptions = transcribe_sslm(
                model, dloader,
                system_prompt=args.system_prompt,
                max_new_tokens=args.max_new_tokens,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                warmup_decode=args.warmup_decode,
                inference_audio_chunks_per_turn=args.inference_audio_chunks_per_turn,
            )
            break
        except RuntimeError as e:
            # Retry on CUDA OOM only; re-raise any other RuntimeError (and OOM at
            # the minimum batch size, which we can't shrink further).
            if "out of memory" not in str(e).lower() or bs <= min_bs:
                raise
            transcriptions = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            bs = max(min_bs, bs // 2)

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
        "batch_size": bs,
        "refs": all_data["references"], "hyps": predictions,
    }


# Default to the locally-built CUDA-12.8 image: vLLM 0.21.0 rebuilt from source
# against cudart 12 (the stock gitlab :v2 image is a CUDA-13 build that fails with
# "Error 803 / CUDA driver insufficient" on hosts with driver < 580). This image also
# bakes the patched converter (convert_patched.sh) + Qwen3-1.7B base LLM. Override with
# --vllm_image (e.g. the gitlab image) when running on a new-enough-driver cluster.
DEFAULT_VLLM_IMAGE = "streaming-stt-eval:cu128-vllmsrc"


def _docker_gpu_args(device):
    """Docker GPU-passthrough args (list), reused by the convert + decode steps.

    Honors ``$DOCKER_GPU_ARGS`` verbatim when set. Otherwise auto-detects: if the
    daemon has the legacy ``nvidia`` runtime registered but ``--gpus`` isn't
    usable (the device-driver plugin is only discovered on a full daemon restart,
    not a reload), prefer ``--runtime=nvidia``; else fall back to
    ``--gpus device=<device>``. Callers add ``-e NVIDIA_VISIBLE_DEVICES``.
    """
    env = os.environ.get("DOCKER_GPU_ARGS")
    if env is not None:
        return env.split()
    import subprocess
    try:
        info = subprocess.run(
            ["docker", "info", "--format", "{{json .Runtimes}}"],
            capture_output=True, text=True, timeout=15,
        ).stdout
    except Exception:
        info = ""
    if '"nvidia"' in info:
        return ["--runtime=nvidia"]
    return ["--gpus", f"device={device}"]


def _vllm_preflight(device, vllm_image):
    """Fail fast (with an actionable message) if the vLLM container can't do CUDA
    compute on this host — most commonly a host-driver / container-CUDA mismatch
    (CUDA error 803). This avoids a long, confusing convert.sh traceback.
    """
    import subprocess

    gpu_args = _docker_gpu_args(device)
    probe = (
        "import torch,sys; "
        "ok=torch.cuda.is_available(); "
        "print('cuda_avail', ok); "
        "sys.exit(0 if ok else 42)"
    )
    cmd = [
        "docker", "run", "--rm", *gpu_args,
        "-e", f"NVIDIA_VISIBLE_DEVICES={device}",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        vllm_image, "python", "-c", probe,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        return
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    mismatch = "error 803" in out.lower() or "unsupported display driver" in out.lower()
    msg = [
        "ERROR: the vLLM container cannot run CUDA compute on this host.",
    ]
    if mismatch:
        msg += [
            "  Cause: host NVIDIA driver is older than the container's CUDA build",
            "         (CUDA error 803 = driver / CUDA-runtime mismatch). Forward-compat",
            "         libs in the image do not bridge this gap.",
            "  Fixes: run --vllm on a node with a new-enough driver (e.g. the OCI",
            "         cluster), use a CUDA build of the image matching this host's",
            "         driver, or upgrade the host driver (machine-wide, disruptive).",
            "  On THIS box, use the native path (drop --vllm):",
            "         python run_eval_sslm.py <model> --last --device 0",
        ]
    else:
        msg += ["  Container GPU probe failed; tail of output:", out[-800:]]
    raise SystemExit("\n".join(msg))


def _vllm_convert(ckpt_path, device, vllm_image, force_convert=False):
    """Convert a Lightning .ckpt to a vLLM model dir via the eval container.

    Mirrors the VLLM path in eval_leaderboard.sh: runs ``/workspace/convert.sh``
    inside ``vllm_image`` (ckpt dir mounted as both /ckpt and /out) and caches the
    output next to the ckpt as ``vllm_<ckptname>/``. Returns the vLLM model dir.
    """
    import shutil
    import subprocess

    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    ckpt_base = os.path.basename(ckpt_path)
    # The in-container to_hf.py is a Hydra app, and Hydra parses '=' in an argument
    # as an override separator -- so a checkpoint literally named 'step=44006.ckpt'
    # breaks conversion. Expose an '='-free hardlink (fallback: copy) in the same dir
    # and hand that name to the container instead.
    safe_base = ckpt_base.replace("=", "_")
    if safe_base != ckpt_base:
        safe_src = os.path.join(ckpt_dir, safe_base)
        if not os.path.exists(safe_src):
            try:
                os.link(os.path.join(ckpt_dir, ckpt_base), safe_src)
            except OSError:
                shutil.copy2(os.path.join(ckpt_dir, ckpt_base), safe_src)
        ckpt_base = safe_base
    vllm_name = f"vllm_{ckpt_base[:-5] if ckpt_base.endswith('.ckpt') else ckpt_base}"
    vllm_out = os.path.join(ckpt_dir, vllm_name)
    gpu_args = _docker_gpu_args(device)

    if os.path.isdir(vllm_out) and os.listdir(vllm_out) and not force_convert:
        print(f"==> Reusing converted vLLM model (use --force_convert to rebuild): {vllm_out}", flush=True)
        return vllm_out

    print(f"==> Converting checkpoint -> vLLM model dir via {vllm_image}", flush=True)
    if os.path.isdir(vllm_out):
        shutil.rmtree(vllm_out)
    os.makedirs(vllm_out, exist_ok=True)
    cmd = [
        "docker", "run", "--rm", *gpu_args,
        "-e", f"NVIDIA_VISIBLE_DEVICES={device}",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        "-v", f"{ckpt_dir}:/ckpt",
        "-v", f"{ckpt_dir}:/out",
        vllm_image,
        # Prefer the patched converter (rewrites pretrained_llm /lustre paths -> HF repo
        # ids and disables torch weights_only) baked into the cu128 image; fall back to
        # the stock convert.sh for images that don't ship it.
        "bash", "-c",
        'S=/workspace/convert_patched.sh; [ -f "$S" ] || S=/workspace/convert.sh; '
        'exec bash "$S" "$1" "$2"',
        "_", f"/ckpt/{ckpt_base}", f"/out/{vllm_name}",
    ]
    if subprocess.run(cmd).returncode != 0:
        raise SystemExit("ERROR: vLLM convert failed (see docker output above)")
    print(f"==> Convert complete: {vllm_out}", flush=True)
    return vllm_out


def evaluate_one_vllm(args, dataset, split, vllm_out):
    """Materialize one (dataset, split) to wavs + a NeMo manifest and decode it in
    the vLLM container (``b_streaming_infer.py``). Returns a result dict
    (``wer``/``time``/``n``) or ``None`` when there are no samples.

    Reuses ``prepare_samples`` so the wavs / references / normalization exactly
    match the native path; scoring uses the WER reported by the container (same as
    eval_leaderboard.sh's VLLM path).
    """
    import re
    import subprocess

    all_data = prepare_samples(args, dataset, split)
    n = len(all_data["references"])
    if n == 0:
        return None

    work = os.getcwd()
    man_dir = os.path.join(work, "manifests")
    os.makedirs(man_dir, exist_ok=True)
    man_path = os.path.join(man_dir, f"{dataset}_{split}.json")
    dump_nemo_manifest(
        man_path, all_data["audio_filepaths"], all_data["durations"], all_data["references"]
    )
    audio_cache = os.path.join(work, "audio_cache")
    gpu_args = _docker_gpu_args(args.device)

    cmd = [
        "docker", "run", "--rm", *gpu_args,
        "-e", f"NVIDIA_VISIBLE_DEVICES={args.device}",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        "-v", f"{vllm_out}:/model",
        "-v", f"{man_dir}:/data",
        "-v", f"{audio_cache}:{audio_cache}",
        "-e", "B_MODEL=/model",
        "-e", f"B_MAN=/data/{os.path.basename(man_path)}",
        args.vllm_image,
        "python", "/workspace/b_streaming_infer.py",
    ]
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")

    # b_streaming_infer.py reports the aggregate as "DONE RTFx mean=.. WER (%)=X";
    # older scripts print "WER: X %". Accept both and take the last (final) value.
    matches = re.findall(r"WER\s*\(%\)\s*=\s*([\d.]+)", out) or re.findall(
        r"WER:\s*([\d.]+)\s*%", out
    )
    if not matches:
        raise RuntimeError(
            f"could not parse WER from vLLM container output (exit={proc.returncode}); "
            f"tail:\n{out[-1000:]}"
        )
    return {"wer": float(matches[-1]), "time": elapsed, "n": n,
            "refs": all_data["references"], "hyps": []}


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

    # ---- vLLM backend: convert once, decode each split in the container ----
    if args.vllm:
        _vllm_preflight(args.device, args.vllm_image)
        vllm_out = _vllm_convert(args.ckpt_path, args.device, args.vllm_image, args.force_convert)
        print(f"\nckpt: {args.ckpt_path}")
        print(f"backend: vLLM ({args.vllm_image})", flush=True)
        print(f"{'dataset':<24}{'WER%':>9}{'time(s)':>11}", flush=True)
        print("-" * 44, flush=True)
        results = []
        for dataset, split in datasets:
            name = f"{dataset}/{split}"
            try:
                with _suppress_output():
                    res = evaluate_one_vllm(args, dataset, split, vllm_out)
            except Exception as e:
                print(f"{name:<24}{'ERR':>9}{'':>11}  ({type(e).__name__}: {e})", flush=True)
                continue
            if res is None:
                print(f"{name:<24}{'n/a':>9}{'':>11}  (no samples)", flush=True)
                continue
            print(f"{name:<24}{res['wer']:>9.2f}{res['time']:>11.1f}", flush=True)
            results.append((name, res))
        if len(results) > 1:
            avg = sum(r["wer"] for _, r in results) / len(results)
            print("-" * 44, flush=True)
            print(f"{'AVERAGE':<24}{avg:>9.2f}", flush=True)
        return results

    with _suppress_output():
        device = torch.device(f"cuda:{args.device}")
        eval_dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
        model = load_model(
            args.ckpt_path, device,
            override_llm=args.pretrained_llm,
            override_asr=args.pretrained_asr,
            dtype=eval_dtype,
        )
        # Build the inference token/flush cache now so we can report its status
        # (the internal logging.info is suppressed by this script's quiet mode).
        model._ensure_inference_cache()

    cfg_use_flush = bool(getattr(model.core_cfg, "use_flush", False))
    inf_flush = bool(getattr(model, "_inference_use_flush", False))
    print(f"\nckpt: {args.ckpt_path}")
    print(f"flush: cfg.use_flush={cfg_use_flush}, inference_flush_active={inf_flush}", flush=True)
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
        bs_note = ""
        if res.get("batch_size") and res["batch_size"] != args.batch_size:
            bs_note = f"   bs={res['batch_size']}"  # reduced from the request via OOM backoff
        print(f"{name:<24}{res['wer']:>9.2f}{res['time']:>11.1f}{bs_note}", flush=True)
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


def resolve_ckpt_spec(spec: str, use_last: bool = False) -> str:
    """Resolve a checkpoint spec to a local ``.ckpt`` path.

    If ``spec`` is an existing local file it is returned unchanged. Otherwise it
    is treated as a grid EXP_NAME and the checkpoint is downloaded from OCI
    (draco-oci-iad) into ``<repo>/checkpoints/<EXP_NAME>/``, mirroring
    ``eval_leaderboard.sh``. By default the BEST snapshot (top-1 val_acc — the
    single non ``-last`` ``step=*.ckpt``) is fetched.

    Env knobs (all optional):
        REMOTE_HOST / REMOTE_USER / SSH_KEY / REMOTE_RESULTS_ROOT
        PROJECT           force the grid project dir (else probe known candidates)
        USE_LAST=1        pull the rolling ``-last.ckpt`` instead of the best
        STEP=<N>          pull ``step=<N>.ckpt`` explicitly
        FORCE_DOWNLOAD=1  re-download even if a local copy is already cached
    """
    if os.path.isfile(spec):
        return spec

    import subprocess

    exp = spec
    remote_host = os.environ.get("REMOTE_HOST", "draco-oci-login-01.draco-oci-iad.nvidia.com")
    remote_user = os.environ.get("REMOTE_USER", "hainanx")
    ssh_key = os.environ.get("SSH_KEY", os.path.expanduser("~/.ssh/draco-rno"))
    results_root = os.environ.get(
        "REMOTE_RESULTS_ROOT", "/lustre/fsw/portfolios/llmservice/users/hainanx/results"
    )
    ssh_opts = ["-i", ssh_key, "-o", "StrictHostKeyChecking=no"]
    ssh_target = f"{remote_user}@{remote_host}"

    projects = (
        [os.environ["PROJECT"]]
        if os.environ.get("PROJECT")
        else [
            "Streaming_SLM_Qwen1p7B",  # current active project (Qwen3-1.7B runs)
            "Streaming_SLM_629",
            "Streaming_SLM_624",
            "Streaming_SLM_chunk14",
            "Streaming_SLM",
        ]
    )

    def _ssh_out(cmd: str) -> str:
        return subprocess.run(
            ["ssh", *ssh_opts, ssh_target, cmd], capture_output=True, text=True
        ).stdout.strip()

    remote_ckpt_dir = ""
    for proj in projects:
        candidate = f"{results_root}/{proj}/{exp}/{exp}/checkpoints"
        if subprocess.run(["ssh", *ssh_opts, ssh_target, f"[ -d '{candidate}' ]"]).returncode == 0:
            remote_ckpt_dir = candidate
            print(f"==> Resolved grid project: {proj}", flush=True)
            break
    if not remote_ckpt_dir:
        raise SystemExit(
            f"ERROR: experiment '{exp}' not found on {ssh_target} under any known project "
            f"(tried: {', '.join(projects)}). Pass a local .ckpt path or set PROJECT=<name>."
        )

    step = os.environ.get("STEP", "")
    use_last = use_last or os.environ.get("USE_LAST", "0") == "1"
    if step:
        ckpt_filename = f"step={step}.ckpt"
    else:
        import re

        def _best_by_val_wer(names: str) -> str:
            """Return the checkpoint filename with the lowest val_wer if present."""
            best_name = ""
            best_wer = None
            for name in names.splitlines():
                m = re.search(r"val_wer=([0-9.]+)", name)
                if m is None:
                    continue
                try:
                    wer = float(m.group(1).rstrip("."))
                except ValueError:
                    continue
                if best_wer is None or wer < best_wer:
                    best_name, best_wer = name, wer
            return best_name

        if use_last:
            list_cmd = f"ls -t {remote_ckpt_dir}/*-last.ckpt 2>/dev/null | head -1 | xargs -r basename"
            ckpt_filename = _ssh_out(list_cmd)
        else:
            all_cmd = (
                f"ls -1 {remote_ckpt_dir}/*.ckpt 2>/dev/null | grep -v -- '-last\\.ckpt$' "
                f"| xargs -r -n1 basename"
            )
            all_names = _ssh_out(all_cmd)
            ckpt_filename = _best_by_val_wer(all_names)
            if not ckpt_filename:
                list_cmd = (
                    f"ls -t {remote_ckpt_dir}/*.ckpt 2>/dev/null | grep -v -- '-last\\.ckpt$' "
                    f"| head -1 | xargs -r basename"
                )
                ckpt_filename = _ssh_out(list_cmd)
        if not ckpt_filename and not use_last:
            ckpt_filename = _ssh_out(
                f"ls -t {remote_ckpt_dir}/*-last.ckpt 2>/dev/null | head -1 | xargs -r basename"
            )
        if not ckpt_filename:
            raise SystemExit(f"ERROR: no checkpoints found in {remote_ckpt_dir}")
        print(f"==> Grid checkpoint: {ckpt_filename}", flush=True)

    repo_root = os.path.dirname(os.path.abspath(__file__))
    local_dir = os.path.join(repo_root, "checkpoints", exp)
    local_path = os.path.join(local_dir, ckpt_filename)
    remote_path = f"{remote_ckpt_dir}/{ckpt_filename}"

    if (
        os.path.isfile(local_path)
        and os.path.getsize(local_path) > 0
        and os.environ.get("FORCE_DOWNLOAD", "0") != "1"
    ):
        print(f"==> Reusing cached checkpoint: {local_path} (set FORCE_DOWNLOAD=1 to re-pull)", flush=True)
        return local_path

    os.makedirs(local_dir, exist_ok=True)
    print(f"==> Downloading from grid:\n    {ssh_target}:{remote_path}\n -> {local_path}", flush=True)
    if subprocess.run(["scp", *ssh_opts, f"{ssh_target}:{remote_path}", local_path]).returncode != 0:
        # Don't leave a truncated file behind to be "reused" next time.
        if os.path.isfile(local_path) and os.path.getsize(local_path) == 0:
            os.remove(local_path)
        raise SystemExit(f"ERROR: scp failed for {remote_path}")
    print("==> Download complete", flush=True)
    return local_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR leaderboard eval for StreamingSTTModel")
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="Grid EXP_NAME (auto-downloaded from OCI) OR a local .ckpt path. "
        "Alternatively pass --ckpt_path.",
    )
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to a local Lightning .ckpt file")
    parser.add_argument(
        "--last",
        action="store_true",
        help="When auto-downloading by EXP_NAME, fetch the rolling -last.ckpt instead of the best "
        "(top-1 val_acc) snapshot. Useful for evaluating an in-progress / undertrained run.",
    )
    parser.add_argument("--pretrained_llm", type=str, default=None,
                        help="Override pretrained_llm (e.g. Qwen/Qwen3-1.7B). Auto-resolved if not set.")
    parser.add_argument("--pretrained_asr", type=str, default=None,
                        help="Override pretrained_asr (e.g. nvidia/nemotron-speech-streaming-en-0.6b). Auto-resolved if not set.")
    parser.add_argument("--dataset_path", type=str, default="hf-audio/esb-datasets-test-only-sorted")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Single dataset to evaluate. If omitted, runs the full ESB leaderboard suite.")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Starting batch size. On CUDA OOM it is halved and retried "
                             "(freeing GPU memory each time) down to --min_batch_size.")
    parser.add_argument("--min_batch_size", type=int, default=1,
                        help="Lower bound for the OOM batch-size backoff; OOM at this size re-raises.")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16",
                        help="Native inference dtype. fp32 is slower/heavier but avoids BF16 greedy-decode "
                             "batch/call-order instabilities observed in debugging.")
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Max tokens to generate per chunk (streaming) or per utterance (offline). "
                             "Defaults to the checkpoint's model.max_new_tokens_per_chunk, matching val_wer.")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0,
                        help="Disallow repeating n-grams during generation to break loops. Defaults to 0, "
                             "matching training validation decode.")
    parser.add_argument("--inference_audio_chunks_per_turn", type=int, default=1,
                        help="How many audio chunks the decoder consumes per LLM turn (streaming). "
                             "1 = default single-chunk decode; >1 groups multiple chunks per turn, "
                             "which lowers WER for models trained with max_audio_chunks_per_turn>1 "
                             "(e.g. the variable/multi-chunk 'maxChunks4' models).")
    parser.add_argument("--warmup_decode", action="store_true",
                        help="Run and discard one single-utterance generate() call before scoring. "
                             "Useful for diagnosing first-call/lazy-kernel decode differences.")
    parser.add_argument("--system_prompt", type=str, default="Transcribe the audio into text.")
    parser.add_argument("--verbose", action="store_true", help="Print each REF/HYP pair to stdout")
    parser.add_argument("--dump_manifest", type=str, default=None,
                        help="Instead of decoding, cache audio to 16k wav and write a NeMo manifest "
                             "(audio_filepath/duration/text) to this path, then exit. Used by the "
                             "VLLM fast path in eval_leaderboard_ord.sh.")
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.set_defaults(streaming=True)
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="Decode with the vLLM eval container instead of in-process model.generate(): "
        "convert the ckpt once, then run /workspace/b_streaming_infer.py per split. Requires docker + GPU.",
    )
    parser.add_argument("--vllm_image", type=str, default=DEFAULT_VLLM_IMAGE,
                        help="Container image for the --vllm path.")
    parser.add_argument("--force_convert", action="store_true",
                        help="Re-run the ckpt->vLLM conversion even if a cached vLLM dir exists.")
    args = parser.parse_args()

    # Accept the checkpoint as a positional EXP_NAME / path or via --ckpt_path.
    # A positional model name that isn't a local file is auto-downloaded from the
    # grid (see resolve_ckpt_spec).
    spec = args.model or args.ckpt_path
    if not spec:
        parser.error("provide a model EXP_NAME (auto-downloaded from grid) or --ckpt_path /path/to.ckpt")
    args.ckpt_path = resolve_ckpt_spec(spec, use_last=args.last)

    main(args)
