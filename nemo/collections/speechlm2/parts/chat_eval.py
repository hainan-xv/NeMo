# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared plumbing for evaluating a trained CHAT transducer.

Lives here rather than in a script so the quick AMI loop and the full
leaderboard driver cannot drift apart: a checkpoint that scores X on AMI through
one path must score X through the other, or the two numbers are not comparable
and nobody notices.
"""

import glob
import json
import os
import tempfile
from typing import List, Optional, Tuple

import torch

__all__ = [
    "load_chat_model",
    "build_chat_tokenizer",
    "read_manifest",
    "transcribe_manifest",
    "score_pairs",
    "find_splits",
]


def _resolve_asr(recorded: str, override: Optional[str] = None) -> str:
    """Local path to the pretrained ASR .nemo the checkpoint was built on.

    Checkpoints record the CLUSTER's paths, which do not exist on a desktop.
    Left unresolved, NeMo falls back to treating the path as a HuggingFace hub
    name and fails with "Repo id must be in the form ..." from deep inside
    huggingface_hub, which says nothing about the real problem.
    """
    for cand in (override, recorded):
        if cand and os.path.exists(cand):
            return cand
    # Search caches by EXACT FILENAME only. Returning "some other nvidia .nemo"
    # is not a fallback, it is a different model: a leaderboard job silently
    # built its encoder from canary-1b-flash instead of the nemotron streaming
    # model this checkpoint was trained on, and only failed later on a shape
    # mismatch (1024x4352 vs 1024x4096). Had the shapes happened to agree it
    # would have produced numbers.
    want = os.path.basename(recorded) if recorded else None
    roots = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.environ.get("HF_HOME", ""),
        "/root/.cache/huggingface/hub",
    ]
    if want:
        for root in [r for r in roots if r and os.path.isdir(r)]:
            for h in sorted(glob.glob(os.path.join(root, "**", want), recursive=True)):
                return h
    raise FileNotFoundError(
        f"cannot find the pretrained ASR model this checkpoint was built on: {recorded!r}.\n"
        "On the grid this usually means the portfolio holding it is not mounted into the container.\n"
        "Pass an explicit local path rather than letting a different model be substituted."
    )


def _resolve_llm(recorded: str, override: Optional[str] = None) -> str:
    if override:
        return override
    if recorded and os.path.exists(recorded):
        return recorded
    return "Qwen/" + os.path.basename(recorded.rstrip("/")) if recorded else "Qwen/Qwen3-1.7B"


def load_chat_model(ckpt_path: str, device: str = "cuda", retract: Optional[int] = None, asr=None, llm=None):
    """Rebuild ChatSTTModel from the CHECKPOINT's own hyper_parameters.

    Deliberately not from a YAML on disk: the recipes have changed repeatedly
    (vocabulary, joint window, delay, recovery), and pairing a checkpoint with a
    drifted config builds a different model -- usually a shape error, at worst a
    quiet mismatch that still produces numbers.
    """
    from omegaconf import OmegaConf

    from nemo.collections.speechlm2.models.chat_model import ChatSTTModel

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ck.get("hyper_parameters", {})
    cfg = hp.get("cfg", hp)
    cfg = dict(cfg) if isinstance(cfg, dict) else OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)

    # Warm-starting from the donor RNN-T is a TRAINING convenience; here it would
    # be overwritten by the checkpoint's own weights at the cost of reading a
    # 2.5 GB .nemo.
    cfg["init_rnnt_from_asr"] = False
    cfg["pretrained_asr"] = _resolve_asr(cfg.get("pretrained_asr", ""), asr)
    if not cfg.get("text_vocab_from_asr", True):
        cfg["pretrained_llm"] = _resolve_llm(cfg.get("pretrained_llm", ""), llm)
    if retract is not None:
        cfg["retract_words"] = int(retract)

    model = ChatSTTModel(cfg)
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    real_missing = [k for k in missing if not k.startswith("perception.preprocessor")]
    if real_missing:
        raise RuntimeError(f"checkpoint is missing {len(real_missing)} parameters, e.g. {real_missing[:5]}")
    return model.to(device).eval(), cfg


def build_chat_tokenizer(cfg: dict):
    """The tokenizer the run trained with -- the arm's defining choice.

    Getting this wrong does not crash: WER would simply be scored against text
    detokenised by the wrong vocabulary, which looks like a very bad model.
    """
    if cfg.get("text_vocab_from_asr", True):
        from nemo.collections.speechlm2.data.script_dataset import ScriptSTTDataset
        from nemo.collections.speechlm2.parts.asr_vocab import AsrVocabTokenizer, extract_spm_from_nemo

        spm = extract_spm_from_nemo(cfg["pretrained_asr"], tempfile.mkdtemp(prefix="chat_vocab_"))
        EOT = "<|im_end|>"
        specials = [ScriptSTTDataset.audio_open_token, ScriptSTTDataset.audio_close_token, EOT]
        return AsrVocabTokenizer(spm, special_tokens=specials, eos_token=EOT, pad_token=EOT)

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

    return AutoTokenizer(cfg["pretrained_llm"], use_fast=True)


def find_splits(cache_dir: str) -> List[Tuple[str, str]]:
    """Every ``<dataset>/<split>`` under the cache that has a manifest."""
    out = []
    for m in sorted(glob.glob(os.path.join(cache_dir, "*", "*", "_cache_manifest.jsonl"))):
        rel = os.path.relpath(os.path.dirname(m), cache_dir)
        ds, split = rel.split(os.sep, 1)
        out.append((ds, split))
    return out


def read_manifest(path: str, max_samples: Optional[int] = None, max_duration: Optional[float] = None) -> List[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if max_duration and r.get("duration", 0) > max_duration:
                continue
            rows.append(r)
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def transcribe_manifest(
    model, tokenizer, rows: List[dict], batch_size: int = 8, device: str = "cuda", progress_every: int = 0
) -> List[str]:
    """Greedy CHAT decode over a manifest, longest-first within each batch."""
    import soundfile as sf

    hyps: List[str] = [""] * len(rows)
    # Sort by duration so a batch is not dominated by padding around one long
    # utterance; indices are carried so the output order still matches `rows`.
    order = sorted(range(len(rows)), key=lambda i: -rows[i].get("duration", 0.0))
    for start in range(0, len(order), batch_size):
        idxs = order[start : start + batch_size]
        waves = []
        for i in idxs:
            w, sr = sf.read(rows[i]["audio_filepath"], dtype="float32")
            if w.ndim > 1:
                w = w.mean(axis=1)
            if sr != 16000:
                raise ValueError(f"expected 16 kHz, got {sr} in {rows[i]['audio_filepath']}")
            waves.append(torch.from_numpy(w))
        lens = torch.tensor([len(w) for w in waves])
        padded = torch.zeros(len(waves), int(lens.max()))
        for j, w in enumerate(waves):
            padded[j, : len(w)] = w
        with torch.no_grad():
            ids = model.transcribe_ids(padded.to(device), lens.to(device))
        for i, seq in zip(idxs, ids):
            hyps[i] = tokenizer.ids_to_text(list(seq)) if seq else ""
        if progress_every and (start // batch_size) % progress_every == 0:
            print(f"    {min(start + batch_size, len(order))}/{len(order)}", end="\r", flush=True)
    return hyps


def score_pairs(refs: List[str], hyps: List[str]) -> dict:
    """Whisper-normalised WER with kaldialign, matching the leaderboard driver.

    Returns errors and reference length as well as the ratio, so results from
    several datasets can be POOLED correctly -- averaging per-dataset WERs is a
    different (and, for the leaderboard, wrong) statistic.
    """
    import kaldialign
    from whisper_normalizer.english import EnglishTextNormalizer

    norm = EnglishTextNormalizer()
    errs = nref = 0
    for r, h in zip(refs, hyps):
        rw, hw = norm(r).split(), norm(h).split()
        if not rw:
            continue
        errs += sum(1 for a, b in kaldialign.align(rw, hw, "*") if a != b)
        nref += len(rw)
    return {"errors": errs, "ref_words": nref, "wer": (errs / nref) if nref else float("nan")}
