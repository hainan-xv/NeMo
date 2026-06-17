"""
Oracle time-warp ("speed cheat") eval over the Open-ASR-Leaderboard datasets.

Motivation
----------
This follows the same reliable data path as ``run_eval_asr.py``: load the HF
dataset row, read its reference, cache the corresponding wav under
``audio_cache/<dataset>/<split>/``, then score that exact (audio, reference)
pair. No prediction/duration joins or partial "matched" subsets are used.

For every cached utterance we decode x1.0 plus N-1 additional time-WARPED copies and,
in scoring only, pick per utterance the warp with the lowest WER (the oracle
upper bound). Model loading / decode paths / normalization / WER are reused from
run_eval_asr.py so numbers are comparable.

Per-dataset; loop with eval_oracle_timewarp.sh.
"""
import argparse
import io
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile
import torch
from datasets import Audio, load_dataset
from tqdm import tqdm

try:
    import librosa
except ImportError:  # pragma: no cover
    sys.exit("librosa is required: pip install librosa")
try:
    import jiwer
except ImportError:  # pragma: no cover
    sys.exit("jiwer is required: pip install jiwer")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import run_eval_asr as R  # noqa: E402

# dataset tag (as in wrapper output names) -> (HF dataset, split)
DATASET_SPECS = {
    "ami_test": ("ami", "test"),
    "earnings22_test": ("earnings22", "test"),
    "gigaspeech_test": ("gigaspeech", "test"),
    "librispeech_test.clean": ("librispeech", "test.clean"),
    "librispeech_test.other": ("librispeech", "test.other"),
    "spgispeech_test": ("spgispeech", "test"),
    "tedlium_test": ("tedlium", "test"),
    "voxpopuli_test": ("voxpopuli", "test"),
}


def warp_audio(y, factor, method):
    """factor>1 => faster/shorter, <1 => slower/longer, ==1 => copy."""
    if factor == 1.0:
        return np.asarray(y, dtype=np.float32)
    if method == "time_stretch":
        return librosa.effects.time_stretch(np.asarray(y, dtype=np.float32), rate=factor)
    if method == "speed":
        n = max(1, int(round(len(y) / factor)))
        xo = np.linspace(0.0, 1.0, num=len(y), endpoint=False)
        xn = np.linspace(0.0, 1.0, num=n, endpoint=False)
        return np.interp(xn, xo, y).astype(np.float32)
    raise ValueError(f"unknown method {method!r}")


def make_transcriber(model, is_multistream, batch_size):
    loss_type = getattr(model, "loss_type", None)
    is_aligner = loss_type in ("aligner", "chunked_aligner")

    def _t(paths):
        if is_multistream:
            return R.transcribe_multistream(model, paths, batch_size)
        if is_aligner:
            return R.transcribe_aligner_like(model, paths, batch_size)
        return R.transcribe_tdt(model, paths, batch_size)

    return _t, ("multistream" if is_multistream else ("aligner/chunked" if is_aligner else "tdt"))


def sample_id_from_row(sample):
    return sample["id"].replace("/", "_").removesuffix(".wav")


def cache_audio(raw_audio, audio_path, sample_id):
    already_cached = os.path.exists(audio_path) and soundfile.info(audio_path).channels == 1
    if already_cached:
        return True

    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    if not isinstance(raw_audio, dict):
        print(f"  WARNING: unexpected audio format for {sample_id}: {type(raw_audio)}, skipping")
        return False

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
        return False

    if audio_array.ndim == 2:
        audio_array = audio_array.mean(axis=1)
    if sr != 16000:
        import torchaudio

        t = torch.from_numpy(audio_array).unsqueeze(0) if audio_array.ndim == 1 else torch.from_numpy(audio_array)
        t = torchaudio.functional.resample(t, sr, 16000)
        audio_array = t.squeeze(0).numpy()
    soundfile.write(audio_path, audio_array, 16000)
    return True


def load_leaderboard_audio_refs(args):
    if args.dataset not in DATASET_SPECS:
        sys.exit(f"unknown dataset tag {args.dataset!r}; choose one of {sorted(DATASET_SPECS)}")
    dataset_name, split = DATASET_SPECS[args.dataset]
    cache_dir = args.audio_dir or os.path.join("audio_cache", dataset_name, split)
    os.makedirs(cache_dir, exist_ok=True)

    rev_str = f"@{args.dataset_revision}" if args.dataset_revision else ""
    print(f"[{args.dataset}] loading dataset: {args.dataset_path}{rev_str}/{dataset_name} split={split}")
    load_kwargs = dict(
        path=args.dataset_path,
        name=dataset_name,
        split=split,
        streaming=args.streaming,
        token=True,
        trust_remote_code=True,
    )
    if args.dataset_revision:
        load_kwargs["revision"] = args.dataset_revision
    dataset = load_dataset(**load_kwargs)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    if args.max_samples and args.max_samples > 0:
        print(f"[{args.dataset}] subsampling to first {args.max_samples} samples")
        dataset = dataset.take(args.max_samples)

    data = {"wavs": [], "durations": [], "refs": [], "refs_raw": [], "sample_ids": [], "total_rows": 0}
    for sample in tqdm(dataset, desc=f"Processing {dataset_name}/{split}"):
        data["total_rows"] += 1
        ref_raw = R.get_text(sample)
        ref = R.text_normalizer(ref_raw)
        if not ref.strip() or ref.strip() == "ignore time segment in scoring":
            continue

        sample_id = sample_id_from_row(sample)
        audio_path = os.path.join(cache_dir, f"{sample_id}.wav")
        if not cache_audio(sample["audio"], audio_path, sample_id):
            continue
        info = soundfile.info(audio_path)
        data["wavs"].append(audio_path)
        data["durations"].append(info.duration)
        data["refs"].append(ref)
        data["refs_raw"].append(ref_raw)
        data["sample_ids"].append(sample_id)

    # Match run_eval_asr.py's batching order so x1.0 baseline is directly comparable.
    sorted_idx = sorted(range(len(data["durations"])), key=lambda k: data["durations"][k], reverse=True)
    for key in ("wavs", "durations", "refs", "refs_raw", "sample_ids"):
        data[key] = [data[key][i] for i in sorted_idx]
    print(f"[{args.dataset}] cached/scored samples: {len(data['wavs'])}/{data['total_rows']}")
    return data


def decode_paths(transcribe, paths):
    out = transcribe(paths)
    fmt = [h.strip() for h in out]
    return fmt, [R.text_normalizer(h) for h in fmt]


def warp_decode(transcribe, wavs, factor, method, keep=False):
    """Write warped copies of ``wavs`` to a temp dir, decode, return (fmt, norm)."""
    work = tempfile.mkdtemp(prefix=f"warp_{factor}_")
    try:
        paths = []
        for i, w in enumerate(tqdm(wavs, desc=f"warp x{factor}", leave=False)):
            y, sr = soundfile.read(w, dtype="float32", always_2d=False)
            if y.ndim == 2:
                y = y.mean(axis=1)
            op = os.path.join(work, f"{i:07d}.wav")
            soundfile.write(op, warp_audio(y, factor, method), sr)
            paths.append(op)
        return decode_paths(transcribe, paths)
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)


def per_utt_wer(ref, hyp):
    return jiwer.wer(ref, hyp if hyp.strip() else " ")


def main(args):
    factors = [float(x) for x in args.factors.split(",") if x.strip()]
    if 1.0 not in factors:
        factors = [1.0] + factors  # x1.0 is the no-warp baseline.
    factors = sorted(set(factors))
    torch.set_float32_matmul_precision("medium")
    device = torch.device(
        f"cuda:{args.device}" if (args.device is not None and args.device >= 0 and torch.cuda.is_available()) else "cpu"
    )

    data = load_leaderboard_audio_refs(args)
    wavs = data["wavs"]
    if not wavs:
        sys.exit(f"no scorable samples for {args.dataset!r}")
    refs = data["refs"]
    refs_raw = data["refs_raw"]
    sample_ids = data["sample_ids"]
    total_rows = data["total_rows"]

    model, is_ms = R.load_model(args.model, device, tokenizer_dir=args.tokenizer_dir)
    transcribe, kind = make_transcriber(model, is_ms, args.batch_size)
    print(f"[{args.dataset}] {len(wavs)} wavs | decode={kind} | factors={factors}")

    # ---- x1.0 decode on the exact referenced set ----
    t0 = time.time()
    fmt1, norm1 = decode_paths(transcribe, wavs)
    base_time = time.time() - t0
    scored = len(wavs)
    print(f"[{args.dataset}] scored samples: {scored}/{total_rows}")

    kept_wavs = wavs
    hyps_norm = {1.0: norm1}
    hyps_fmt = {1.0: fmt1}
    decode_time = {1.0: base_time}

    # ---- decode warped copies on the same scored dataset rows ----
    for f in factors:
        if f == 1.0:
            continue
        t0 = time.time()
        fmt, norm = warp_decode(transcribe, kept_wavs, f, args.method, keep=args.keep_warped)
        decode_time[f] = time.time() - t0
        hyps_fmt[f] = fmt
        hyps_norm[f] = norm

    # ---- scoring ----
    factor_corpus = {f: 100 * R.wer_metric.compute(references=refs, predictions=hyps_norm[f]) for f in factors}
    chosen, oracle_preds, details = [], [], []
    for j in range(scored):
        cand = [(per_utt_wer(refs[j], hyps_norm[f][j]), f) for f in factors]
        best_w = min(c[0] for c in cand)
        tied = [f for (w, f) in cand if w == best_w]
        best_f = 1.0 if 1.0 in tied else min(tied, key=lambda f: abs(f - 1.0))
        chosen.append(best_f)
        oracle_preds.append(hyps_norm[best_f][j])
        details.append({
            "audio_filepath": kept_wavs[j],
            "sample_id": sample_ids[j],
            "text": refs_raw[j],
            "text_normalized": refs[j],
            "per_factor": {str(f): {"pred_text_normalized": hyps_norm[f][j],
                                    "wer": round(per_utt_wer(refs[j], hyps_norm[f][j]), 4)} for f in factors},
            "chosen_factor": best_f,
            "oracle_wer": round(per_utt_wer(refs[j], oracle_preds[j]), 4),
        })
    oracle_corpus = 100 * R.wer_metric.compute(references=refs, predictions=oracle_preds)
    best_f = min(factor_corpus, key=factor_corpus.get)
    counts = Counter(chosen)

    lines = []
    p = lines.append
    p("=" * 80)
    p(f"ORACLE TIME-WARP  |  dataset={args.dataset}  method={args.method}")
    p("=" * 80)
    p(f"scored utterances   : {scored}/{total_rows}")
    p(f"factors            : {factors}")
    p("")
    p("Per-factor corpus WER (each warp alone):")
    for f in factors:
        tag = "  (baseline/no-warp)" if f == 1.0 else ""
        p(f"  x{f:<5}: {factor_corpus[f]:6.2f} %   [decode {decode_time[f]:.0f}s]{tag}")
    p("")
    p(f"baseline (x1.0)          : {factor_corpus[1.0]:6.2f} %")
    p(f"best single fixed (x{best_f}) : {factor_corpus[best_f]:6.2f} %")
    p(f"ORACLE best-of-{len(factors)}        : {oracle_corpus:6.2f} %")
    p(f"  gain vs baseline       : {factor_corpus[1.0] - oracle_corpus:6.2f} pts "
      f"({100*(factor_corpus[1.0]-oracle_corpus)/max(factor_corpus[1.0],1e-9):.1f}% rel.)")
    p("")
    p("oracle pick distribution:")
    for f in factors:
        c = counts.get(f, 0)
        p(f"  x{f:<5}: {c:>6}  ({100*c/scored:5.1f}%)  {'#'*int(round(40*c/scored))}")
    pick_summary = ",".join(f"x{f}:{100*counts.get(f, 0)/scored:.1f}" for f in factors)
    report = "\n".join(lines)
    print(report)

    # machine-readable summary line for the wrapper to parse
    print(f"ORACLE_SUMMARY {args.dataset} scored={scored} total={total_rows} baseline={factor_corpus[1.0]:.2f} "
          f"best_fixed={factor_corpus[best_f]:.2f} oracle={oracle_corpus:.2f} picks={pick_summary}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write("# " + report.replace("\n", "\n# ") + "\n")
            for d in details:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"[per-utterance report -> {args.output}]", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, help="e.g. librispeech_test.clean (matches results/ filename tag)")
    ap.add_argument("--dataset_path", default="hf-audio/esb-datasets-test-only-sorted")
    ap.add_argument("--dataset_revision", default=None)
    ap.add_argument("--audio_dir", default=None, help="override audio cache dir for this dataset/split")
    ap.add_argument("--tokenizer_dir", default=None)
    ap.add_argument("--factors", default="0.9,1.0,1.1")
    ap.add_argument("--method", default="time_stretch", choices=["time_stretch", "speed"])
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--keep_warped", action="store_true")
    ap.add_argument("--output", default=None)
    ap.add_argument("--no-streaming", dest="streaming", action="store_false")
    ap.set_defaults(streaming=True)
    main(ap.parse_args())
