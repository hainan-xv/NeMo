"""
ASR leaderboard evaluation for (multistream-)TDT transducer models.

Local sibling of ``run_eval_sslm.py`` (the StreamingSTTModel driver): same HF
ESB / Open-ASR-Leaderboard dataset loading, audio caching and whisper-normalized
WER, but it evaluates NeMo transducer ASR checkpoints instead.

Supports BOTH model families, auto-detected from the checkpoint's stored
``target`` field:
  * the regular TDT model (e.g. a fine-tuned ``nvidia/parakeet-tdt-0.6b-v2``,
    ``EncDecRNNTBPEModel``) -- decoded with its own (batched) ``transcribe``; and
  * the 2-stream factorized spelling+capitalization model
    (``EncDecMultiStreamTDTBPEModel``) -- decoded with its non-batched greedy
    multistream decoder (encoder forward -> ``_decode_hyp_texts``).

Use ``eval_asr_ord.sh`` to pull a checkpoint from ORD and run this over the
whole leaderboard suite; this script handles a single (dataset, split).
"""
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile
import torch
from tqdm import tqdm

# Force HF datasets to use soundfile for audio decoding (avoids torchcodec
# issues with incompatible PyTorch/FFmpeg versions). Must be set before
# importing datasets.
os.environ["HF_AUDIO_DECODER"] = "soundfile"

import evaluate
import lhotse
import lhotse.dataset
from datasets import Audio, load_dataset
from whisper_normalizer.english import EnglishTextNormalizer

# Make sure we import the `nemo` that lives next to this script (mirrors how
# run_eval_sslm.py pins STREAMING_STT_MODEL_ROOT) so eval runs the local code.
_EVAL_ROOT = Path(__file__).resolve().parent
if str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT))

from omegaconf import OmegaConf, open_dict  # noqa: E402

from nemo.collections.asr.models import ASRModel  # noqa: E402
from nemo.utils import logging  # noqa: E402
from nemo.utils.model_utils import import_class_by_path  # noqa: E402

wer_metric = evaluate.load("wer")
text_normalizer = EnglishTextNormalizer()


# --------------------------------------------------------------------------- #
# Model loading (auto-detect class; .nemo or .ckpt)
# --------------------------------------------------------------------------- #
def load_model(model_path, device, tokenizer_dir=None, dtype=torch.bfloat16):
    """Load a .nemo or .ckpt into the correct ASR subclass (auto-detected)."""
    print(f"Loading checkpoint: {model_path}")
    if model_path.endswith(".nemo"):
        cfg = ASRModel.restore_from(restore_path=model_path, return_config=True)
        target = cfg.get("target", "nemo.collections.asr.models.ASRModel")
        cls = import_class_by_path(target)
        print(f"  Restoring {cls.__name__} from .nemo")
        model = cls.restore_from(restore_path=model_path, map_location="cpu")
    elif model_path.endswith(".ckpt"):
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        if "hyper_parameters" not in ckpt or "cfg" not in ckpt["hyper_parameters"]:
            raise ValueError(
                f"{model_path} does not embed a model config under hyper_parameters['cfg']; "
                "evaluate a .nemo export instead."
            )
        cfg = OmegaConf.create(ckpt["hyper_parameters"]["cfg"])
        if tokenizer_dir is not None:
            with open_dict(cfg):
                if "tokenizer" in cfg:
                    cfg.tokenizer.dir = tokenizer_dir
                    cfg.tokenizer.update_tokenizer = False
        target = cfg.get("target", None)
        if target is None:
            raise ValueError("Checkpoint config has no `target`; cannot determine the model class.")
        cls = import_class_by_path(target)
        print(f"  Instantiating {cls.__name__} from .ckpt config and loading weights")
        model = cls(cfg=cfg)
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        del ckpt
    else:
        raise ValueError(f"Unsupported checkpoint extension for {model_path} (expected .nemo or .ckpt).")

    # Detect by capability so any multistream variant (2-stream cap, 3-stream cap+punct, ...) works.
    is_multistream = hasattr(model, "_decode_hyp_texts")
    # bf16 on the encoder is fine; keep multistream greedy in fp32-friendly path.
    model = model.eval().to(device)
    if dtype is not None and not is_multistream:
        try:
            model = model.to(dtype)
        except Exception as e:  # pragma: no cover - defensive
            print(f"  (could not cast to {dtype}: {e}; staying in fp32)")
    print(f"  Model loaded on {device}; multistream={is_multistream}")
    return model, is_multistream


# --------------------------------------------------------------------------- #
# Audio dataloader (multistream path) -- identical to run_eval_sslm.py
# --------------------------------------------------------------------------- #
class ToAudio(torch.utils.data.Dataset):
    def __getitem__(self, cuts):
        cuts = lhotse.CutSet([c.to_mono(mono_downmix=True) if isinstance(c, lhotse.MultiCut) else c for c in cuts])
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


@torch.inference_mode()
def transcribe_multistream(model, audio_files, batch_size):
    """Greedy multistream decode: encoder forward -> _decode_hyp_texts.

    DynamicCutSampler preserves input order, so the returned hypotheses align
    with ``audio_files`` (which the caller has already sorted by duration).
    """
    dloader = setup_dloader(audio_files, batch_size=batch_size)
    hyps = []
    for batch in tqdm(dloader, desc="Transcribing (multistream)"):
        audios = batch["audios"].to(model.device, non_blocking=True)
        audio_lens = batch["audio_lens"].to(model.device, non_blocking=True)
        encoded, encoded_len = model.forward(input_signal=audios, input_signal_length=audio_lens)
        hyps.extend(model._decode_hyp_texts(encoded, encoded_len))
    return hyps


def transcribe_tdt(model, audio_files, batch_size):
    """Regular TDT/RNNT decode via the model's own (batched) transcribe()."""
    with torch.inference_mode():
        out = model.transcribe(audio_files, batch_size=batch_size, verbose=True)
    # transcribe() may return list[Hypothesis], list[str], or a tuple of those.
    if isinstance(out, tuple):
        out = out[0]
    texts = []
    for h in out:
        texts.append(h.text if hasattr(h, "text") else str(h))
    return texts


class AudioFileDataset(torch.utils.data.Dataset):
    """Simple soundfile-backed dataset for eval-only transcription.

    This avoids NeMo/Lhotse ``model.transcribe`` dataloader construction for
    aligner / chunked-aligner checkpoints. Some ESB samples are cached as mono
    arrays that hit a Lhotse collation path expecting an explicit channel
    dimension; for these models we only need padded waveforms + lengths.
    """

    def __init__(self, audio_files):
        self.audio_files = audio_files

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio, sr = soundfile.read(self.audio_files[idx], dtype="float32", always_2d=False)
        if sr != 16000:
            raise ValueError(f"Expected cached 16 kHz audio, got sr={sr} for {self.audio_files[idx]}")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        return audio, torch.tensor(audio.numel(), dtype=torch.long)


def _collate_audio(batch):
    audios, audio_lens = zip(*batch)
    audio_lens = torch.stack(audio_lens)
    max_len = int(audio_lens.max().item())
    padded = torch.zeros(len(audios), max_len, dtype=torch.float32)
    for i, audio in enumerate(audios):
        padded[i, : audio.numel()] = audio
    return padded, audio_lens


@torch.inference_mode()
def transcribe_aligner_like(model, audio_files, batch_size):
    """Decode aligner / chunked-aligner models without ``model.transcribe``.

    The model's transcribe dataloader can inherit Lhotse settings from training
    configs and fail on mono cached ESB wavs. This direct path preserves the
    model's own encoder and ``_transcribe_output_processing`` logic, including
    chunked-aligner greedy decoding, but uses a minimal soundfile dataloader.
    """

    dloader = torch.utils.data.DataLoader(
        AudioFileDataset(audio_files),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_audio,
    )
    hyps = []
    for signal, signal_len in tqdm(dloader, desc="Transcribing (aligner/chunked)"):
        signal = signal.to(model.device, non_blocking=True)
        signal_len = signal_len.to(model.device, non_blocking=True)
        outputs = model._transcribe_forward((signal, signal_len), trcfg=None)
        out = model._transcribe_output_processing(outputs, trcfg=None)
        for h in out:
            hyps.append(h.text if hasattr(h, "text") else str(h))
    return hyps


# --------------------------------------------------------------------------- #
# Dataset handling -- mirrors run_eval_sslm.py
# --------------------------------------------------------------------------- #
def get_text(sample):
    for key in ("text", "sentence", "normalized_text", "transcript", "transcription"):
        if key in sample:
            return sample[key]
    raise ValueError(f"No transcript column found in sample keys: {list(sample.keys())}")


def write_manifest(references, predictions, model_id, dataset_path, dataset_name, split,
                   audio_length=None, transcription_time=None,
                   references_formatted=None, predictions_formatted=None):
    """Write a results manifest.

    ``text`` / ``pred_text`` hold the *verbatim* reference and the *formatted* decode output
    (capitalization + punctuation preserved). ``text_normalized`` / ``pred_text_normalized`` hold
    the normalized versions actually used for the (leaderboard-comparable) WER. Decoding always
    keeps casing+punctuation; normalization happens only for scoring.
    """
    model_id_safe = model_id.replace("/", "-")
    dataset_path_safe = dataset_path.replace("/", "-")
    dataset_name_safe = dataset_name.replace("/", "-")
    basedir = "./results/"
    os.makedirs(basedir, exist_ok=True)
    manifest_path = os.path.join(
        basedir, f"MODEL_{model_id_safe}_DATASET_{dataset_path_safe}_{dataset_name_safe}_{split}.jsonl"
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        for idx in range(len(references)):
            ref_fmt = references_formatted[idx] if references_formatted is not None else references[idx]
            pred_fmt = predictions_formatted[idx] if predictions_formatted is not None else predictions[idx]
            datum = {
                "audio_filepath": f"sample_{idx}",
                "duration": audio_length[idx] if audio_length else None,
                "time": transcription_time[idx] if transcription_time else None,
                "text": ref_fmt,
                "pred_text": pred_fmt,
                "text_normalized": references[idx],
                "pred_text_normalized": predictions[idx],
            }
            f.write(json.dumps(datum, ensure_ascii=False) + "\n")
    return manifest_path


def main(args):
    data_cache_dir = os.path.join(os.getcwd(), "audio_cache")
    cache_dir = os.path.join(data_cache_dir, args.dataset, args.split)
    os.makedirs(cache_dir, exist_ok=True)

    torch.set_float32_matmul_precision("medium")
    device = torch.device(f"cuda:{args.device}" if (args.device is not None and args.device >= 0 and torch.cuda.is_available()) else "cpu")

    model, is_multistream = load_model(args.model, device, tokenizer_dir=args.tokenizer_dir)
    if args.use_cer and hasattr(model, "use_cer"):
        model.use_cer = True
    if args.max_symbols_per_step is not None and hasattr(model, "ms_greedy"):
        model.ms_greedy.max_symbols = args.max_symbols_per_step

    rev_str = f"@{args.dataset_revision}" if args.dataset_revision else ""
    print(f"Loading dataset: {args.dataset_path}{rev_str}/{args.dataset} split={args.split}")
    load_kwargs = dict(
        path=args.dataset_path,
        name=args.dataset,
        split=args.split,
        streaming=args.streaming,
        token=True,
        trust_remote_code=True,
    )
    if args.dataset_revision:
        load_kwargs["revision"] = args.dataset_revision
    dataset = load_dataset(**load_kwargs)
    dataset = dataset.cast_column("audio", Audio(decode=False))

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling to first {args.max_eval_samples} samples")
        dataset = dataset.take(args.max_eval_samples)

    print("Downloading and caching audio samples...")
    all_data = {"audio_filepaths": [], "durations": [], "references": [], "references_raw": []}
    for sample in tqdm(dataset, desc="Processing samples"):
        ref_raw = get_text(sample)
        ref = text_normalizer(ref_raw)
        if not ref.strip() or ref.strip() == "ignore time segment in scoring":
            continue
        raw_audio = sample["audio"]
        sample_id = sample["id"].replace("/", "_").removesuffix(".wav")
        audio_path = os.path.join(cache_dir, f"{sample_id}.wav")
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
                    print(f"  WARNING: cannot decode audio for {sample_id}, skipping")
                    continue
            else:
                print(f"  WARNING: unexpected audio format for {sample_id}: {type(raw_audio)}, skipping")
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
        all_data["references_raw"].append(ref_raw)

    # Sort by duration (desc) for efficient batching; both decode paths preserve order.
    sorted_idx = sorted(range(len(all_data["durations"])), key=lambda k: all_data["durations"][k], reverse=True)
    for key in all_data:
        all_data[key] = [all_data[key][i] for i in sorted_idx]

    n_samples = len(all_data["references"])
    print(f"Total samples: {n_samples}")
    if n_samples == 0:
        print("ERROR: No samples to evaluate!")
        return

    loss_type = getattr(model, "loss_type", None)
    is_aligner_like = loss_type in ("aligner", "chunked_aligner")
    decode_kind = "multistream" if is_multistream else ("aligner/chunked" if is_aligner_like else "tdt")
    print(f"Transcribing {n_samples} samples ({decode_kind})...")
    start = time.time()
    if is_multistream:
        transcriptions = transcribe_multistream(model, all_data["audio_filepaths"], args.batch_size)
    elif is_aligner_like:
        transcriptions = transcribe_aligner_like(model, all_data["audio_filepaths"], args.batch_size)
    else:
        transcriptions = transcribe_tdt(model, all_data["audio_filepaths"], args.batch_size)
    total_time = time.time() - start

    # Decoding always keeps casing + punctuation; normalize only for (leaderboard) scoring.
    predictions_formatted = [pred.strip() for pred in transcriptions]
    predictions = [text_normalizer(pred) for pred in predictions_formatted]

    if args.verbose:
        print("\n" + "=" * 70 + "\nREF / HYP pairs (formatted | normalized):\n" + "=" * 70)
        for i in range(len(predictions)):
            print(f"[{i}] REF : {all_data['references_raw'][i]}")
            print(f"[{i}] HYP : {predictions_formatted[i]}")
            print(f"[{i}] HYP*: {predictions[i]}\n")
        print("=" * 70)

    avg_time = total_time / n_samples
    model_label = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(args.model))))
    manifest_path = write_manifest(
        all_data["references"], predictions, model_label, args.dataset_path, args.dataset, args.split,
        audio_length=all_data["durations"], transcription_time=[avg_time] * n_samples,
        references_formatted=all_data["references_raw"], predictions_formatted=predictions_formatted,
    )
    print("Results saved at:", os.path.abspath(manifest_path))

    metric = wer_metric.compute(references=all_data["references"], predictions=predictions)
    metric = round(100 * metric, 2)
    audio_length = sum(all_data["durations"])
    rtfx = round(audio_length / total_time, 2)

    print(f"Dataset: {args.dataset}/{args.split}")
    print(f"RTFX: {rtfx}")
    print(f"WER: {metric} %")
    return metric, rtfx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR leaderboard eval for (multistream-)TDT models")
    parser.add_argument("--model", type=str, required=True, help="Path to a .nemo or Lightning .ckpt file")
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default=None,
        help="Override tokenizer dir for .ckpt loads (needed when the trained-time path is gone).",
    )
    parser.add_argument("--dataset_path", type=str, default="hf-audio/esb-datasets-test-only-sorted")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--dataset_revision",
        type=str,
        default=None,
        help="Optional HF dataset revision (used to pin TED-LIUM to a pre-deletion snapshot).",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--use_cer", action="store_true", help="Report CER instead of WER (multistream).")
    parser.add_argument(
        "--max_symbols_per_step", type=int, default=None, help="Override greedy symbols-per-step (multistream only)."
    )
    parser.add_argument("--verbose", action="store_true", help="Print each REF/HYP pair to stdout")
    parser.add_argument(
        "--use_pass", type=str, default="streaming", help="Accepted for eval-launcher compatibility; ignored here."
    )
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.set_defaults(streaming=True)
    main(parser.parse_args())
