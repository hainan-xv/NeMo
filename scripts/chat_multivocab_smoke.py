#!/usr/bin/env python3
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

"""Local end-to-end smoke test for the multi-vocab CHAT model.

WHY. Four consecutive grid launches died -- mandatory multivocab section,
state_dict key mismatch on the donor restore, DDP unused parameters, and heads
that silently lost the inferred chunk_size -- at roughly ten minutes of queue
plus startup each. Every one was a construction or first-step failure that toy
data on a local GPU reproduces in under a minute.

WHAT IT COVERS, in the order the grid hit them:
  1. construction from the multivocab config
  2. the DONOR restore path: builds this class from a single-vocab config and
     loads its weights strictly, which is what init_from_nemo_model does
  3. chunk geometry identical across heads (the silent one)
  4. real training steps through the banded loss, per-head lattice and CUDA
     kernel, under DDP if two GPUs are present -- which is the only way the
     unused-parameter failure appears
  5. a validation pass, pinned to head 0

Toy everything: 12 synthetic utterances, a 2-layer encoder, and three小
SentencePiece vocabularies built here. Sizes are small (the corpus cannot
support 1024 pieces) but DIFFERENT, which is all the head machinery cares about.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import sentencepiece as spm
import soundfile as sf

WORDS = ("the quick brown fox jumps over a lazy dog and then runs far away while birds sing "
         "softly in bright morning light near cold water").split()
VOCAB_SIZES = (48, 64, 96)


def make_corpus(path, n=400, rng=None):
    rng = rng or np.random.default_rng(0)
    with open(path, "w") as f:
        for _ in range(n):
            k = int(rng.integers(4, 12))
            f.write(" ".join(rng.choice(WORDS, size=k)) + "\n")


def build_tokenizers(root, corpus):
    dirs = []
    for v in VOCAB_SIZES:
        d = os.path.join(root, f"v{v}")
        os.makedirs(d, exist_ok=True)
        spm.SentencePieceTrainer.train(
            input=corpus, model_prefix=os.path.join(d, "tokenizer"), vocab_size=v,
            model_type="bpe", character_coverage=1.0, bos_id=-1, eos_id=-1, unk_id=0,
add_dummy_prefix=True, minloglevel=2,
        )
        # NeMo's monolingual BPE path registers tokenizer.model AND vocab.txt as
        # artifacts; the real vocabularies get vocab.txt from
        # process_asr_text_tokenizer.py, sentencepiece alone writes .vocab.
        sp = spm.SentencePieceProcessor(model_file=os.path.join(d, "tokenizer.model"))
        with open(os.path.join(d, "vocab.txt"), "w") as vf:
            for i in range(sp.get_piece_size()):
                vf.write(sp.id_to_piece(i) + "\n")
        dirs.append(d)
    return dirs


def make_data(root, n_utt=12, sr=16000, rng=None):
    """Cuts with the two things _chunk_tokens reads: supervision text and
    custom['alignments'] as {text, end_time}."""
    rng = rng or np.random.default_rng(1)
    audio_dir = os.path.join(root, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    lines = []
    for i in range(n_utt):
        nw = int(rng.integers(4, 9))
        words = list(rng.choice(WORDS, size=nw))
        # ~0.35 s per word, so chunk_size 14 frames (1.12 s) spans ~3 words.
        per = 0.35
        dur = round(nw * per + 0.5, 3)
        wav = (0.01 * rng.standard_normal(int(dur * sr))).astype(np.float32)
        p = os.path.join(audio_dir, f"utt{i}.wav")
        sf.write(p, wav, sr)
        aligns = [{"text": w, "end_time": round((j + 1) * per, 3)} for j, w in enumerate(words)]
        lines.append({
            "id": f"utt{i}", "start": 0.0, "duration": dur, "channel": 0,
            "recording": {"id": f"utt{i}", "sources": [{"type": "file", "channels": [0], "source": p}],
                          "sampling_rate": sr, "num_samples": int(dur * sr), "duration": dur},
            "supervisions": [{"id": f"sup{i}", "recording_id": f"utt{i}", "start": 0.0,
                              "duration": dur, "channel": 0, "text": " ".join(words), "language": "en"}],
            "custom": {"alignments": aligns},
            "type": "MonoCut",
        })
    cuts = os.path.join(root, "cuts.jsonl")
    with open(cuts, "w") as f:
        for d in lines:
            f.write(json.dumps(d) + "\n")
    return cuts

def toy_cfg(tok_dirs, cuts, steps, devices):
    """The real multivocab config, shrunk. Only SIZE changes -- the encoder keeps
    att_context_style/att_context_size so chunk_size is still INFERRED, which is
    the path that silently gave heads 1-2 chunk_size=-1 on the grid."""
    from omegaconf import OmegaConf, open_dict

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = OmegaConf.load(os.path.join(
        here, "examples/asr/conf/fastconformer/cache_aware_streaming",
        "nemotron_chat_transducer_granary2_spe_multivocab.yaml"))
    m = cfg.model
    with open_dict(cfg):
        # Sizes flow from model_defaults through ${...} interpolation into BOTH
        # decoder.prednet and joint.jointnet. Overriding the leaves individually
        # shadows some sites and not others -- the joint kept encoder_hidden=1024
        # against a 176-wide encoder and the decode died in project_encoder.
        m.encoder.n_layers = 2
        m.encoder.d_model = 176
        m.encoder.n_heads = 4
        m.model_defaults.enc_hidden = 176
        m.model_defaults.pred_hidden = 160
        m.model_defaults.joint_hidden = 160
        m.tokenizer.dir = tok_dirs[0]
        m.multivocab.tokenizer_dirs = list(tok_dirs)

        for ds in (m.train_ds, m.validation_ds):
            ds.use_lhotse = True
            ds.cuts_path = cuts
            ds.manifest_filepath = None
            if "input_cfg" in ds:
                ds.input_cfg = None
            ds.num_workers = 0
            ds.batch_size = 2
            if "use_bucketing" in ds:
                ds.use_bucketing = False
            for k in ("bucket_batch_size", "bucket_duration_bins"):
                if k in ds:
                    ds[k] = None
            ds.shuffle = False
        m.optim.sched.warmup_steps = 1
        cfg.trainer.devices = devices
        cfg.trainer.num_nodes = 1
        cfg.trainer.max_steps = steps
        # Validate PART WAY THROUGH, not at the end. Training after a validation
        # pass is its own code path -- the modules Lightning put in eval mode
        # have to come back to train, and a head rebound by _select_head may not.
        # With validation only at the end this never runs, and the smoke test
        # passed while the grid died with "cudnn RNN backward can only be called
        # in training mode" at the first step after validation.
        cfg.trainer.val_check_interval = None
        cfg.trainer.check_val_every_n_epoch = 1
        cfg.trainer.limit_val_batches = 2
        # EVERY step, not the default 50. training_step only runs the greedy
        # decode for training_batch_wer on this cadence, and that decode is what
        # flips the decoder into eval mode. At the default a short smoke run
        # never executes it, which is how the grid failure slipped through.
        cfg.trainer.log_every_n_steps = 1
        # MULTIPLE EPOCHS with checkpointing ON. The grid died at "Epoch 1: 0%" --
        # the first training step after an epoch boundary, where Lightning runs
        # validation, saves a checkpoint (which invokes our state_dict override)
        # and then returns to training. With one epoch and checkpointing off,
        # none of that executes and the smoke test cannot see it.
        cfg.trainer.limit_train_batches = max(steps // 3, 2)
        cfg.trainer.accelerator = "gpu"
        cfg.trainer.logger = False
        cfg.trainer.enable_checkpointing = os.environ.get("MVSMOKE_NOCKPT", "0") != "1"
        # bf16-mixed, MATCHING THE GRID. Under autocast the LSTM takes the
        # cuDNN RNN path, which raises "cudnn RNN backward can only be called
        # in training mode" if the forward ran in eval. In fp32 the fallback
        # kernel tolerates it silently, so a 32-bit smoke test cannot see the
        # bug the grid hit.
        cfg.trainer.precision = "bf16-mixed"
        cfg.exp_manager = None
    return cfg


def run_model(cfg, root, devices):
    import lightning.pytorch as pl
    import torch
    from omegaconf import OmegaConf, open_dict

    from nemo.collections.asr.models import EncDecCHATBPEModel, EncDecMultiVocabCHATBPEModel
    from nemo.utils.trainer_utils import resolve_trainer_cfg

    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}  {detail}", flush=True)

    # (2) DONOR: a single-vocab CHAT model saved to .nemo, which is exactly what
    # init_from_nemo_model restores -- through THIS class, with strict loading.
    print("\n=== donor .nemo (single-vocab) ===", flush=True)
    dcfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(dcfg):
        del dcfg.model["multivocab"]
    donor = EncDecCHATBPEModel(cfg=dcfg.model, trainer=None)
    # Keep a reference tensor so we can prove head 0 actually RECEIVED the
    # donor's decoder, rather than merely that the restore raised nothing.
    donor_embed = donor.decoder.prediction.embed.weight.detach().clone().cpu()
    donor_path = os.path.join(root, "donor.nemo")
    donor.save_to(donor_path)
    del donor
    # Lightning re-executes this script in every DDP process, and NeMo's save_to
    # only writes on global rank zero -- so on rank 1 the file legitimately does
    # not exist and asserting it would report a failure that is purely an
    # artifact of the harness.
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        check("donor saved", os.path.exists(donor_path), donor_path)

    with open_dict(cfg):
        # FULL init of head 0, not just the encoder: the donor is a single-vocab
        # model at the SAME vocabulary as head 0, so its decoder and joint
        # transfer exactly. Heads 1-2 have different vocabularies and stay random.
        cfg.init_from_nemo_model = OmegaConf.create(
            {"model0": {"path": donor_path, "include": ["encoder.", "decoder.", "joint."], "exclude": []}})

    print("\n=== construct + warm start + train ===", flush=True)
    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    model = EncDecMultiVocabCHATBPEModel(cfg=cfg.model, trainer=trainer)

    sizes = [int(h.joint.chunk_size) for h in model._heads]
    check("all heads share chunk_size", len(set(sizes)) == 1 and sizes[0] > 0, f"{sizes}")
    widths = [h.joint.window_width() for h in model._heads]
    check("all heads share window_width", len(set(widths)) == 1 and widths[0] > 0, f"{widths}")
    vs = [h.joint.num_classes_with_blank - 1 for h in model._heads]
    check("three distinct vocab sizes", len(set(vs)) == 3, f"{vs}")

    model.maybe_init_from_pretrained_checkpoint(cfg)
    check("donor restore survived", True, "(no exception)")

    import torch as _t

    got = model._heads[0].decoder.prediction.embed.weight.detach().cpu()
    check("head 0 decoder LOADED from donor", _t.allclose(got, donor_embed),
          f"max|diff|={float((got - donor_embed).abs().max()):.3e}")
    for k in (1, 2):
        w = model._heads[k].decoder.prediction.embed.weight.detach().cpu()
        # Different vocabulary sizes, so not even comparable -- assert the shape
        # differs, which is what makes 'heads 1-2 are untouched' meaningful.
        check(f"head {k} decoder NOT from donor", w.shape[0] != donor_embed.shape[0],
              f"{tuple(w.shape)} vs donor {tuple(donor_embed.shape)}")

    trainer.fit(model)
    check("training completed", trainer.global_step >= 1, f"global_step={trainer.global_step}")
    model.train()
    modes = [(h.decoder.training, h.joint.training) for h in model._heads]
    check("every head follows model.train()", all(a and b for a, b in modes), f"{modes}")
    model.eval()
    modes = [(h.decoder.training, h.joint.training) for h in model._heads]
    check("every head follows model.eval()", not any(a or b for a, b in modes), f"{modes}")
    model.train()
    # A TERMINATION checkpoint is written with whatever head was last sampled.
    # Capture the state with a non-zero head active and confirm a fresh model can
    # load it -- the exact failure that killed 19118299 on resume.
    import torch as _t2

    model._select_head(len(model._heads) - 1)
    sd = model.state_dict()
    shape = tuple(sd["decoder.prediction.embed.weight"].shape)
    want = model._heads[0].decoder.prediction.embed.weight.shape
    check("state_dict pins head 0 under decoder.*", shape == tuple(want),
          f"{shape} vs head0 {tuple(want)} (active head was {model._active_head})")
    fresh = EncDecMultiVocabCHATBPEModel(cfg=cfg.model, trainer=None)
    try:
        fresh.load_state_dict(sd, strict=True)
        check("fresh model loads that checkpoint", True, "(strict)")
    except Exception as e:
        check("fresh model loads that checkpoint", False, f"{type(e).__name__}: {str(e)[:110]}")
    counts = model._head_counts
    check("more than one head sampled", sum(1 for c in counts if c > 0) > 1, f"head counts {counts}")
    # Test the PINNING MECHANISM, not the head left over after fit: with more
    # than one epoch, training continues past the last validation, so the
    # post-fit head is whatever was sampled last and says nothing about
    # validation.
    model._select_head(len(model._heads) - 1)
    model.on_validation_epoch_start()
    check("validation pins head 0", model._active_head == model._val_head,
          f"selected {model._active_head} from head {len(model._heads) - 1}; val_head={model._val_head}")
    return ok


def time_steps(cfg, root, multivocab: bool, fup: bool, steps: int, devices: int):
    """Wall-clock for `steps` training steps, single-vocab vs multi-vocab."""
    import time

    import lightning.pytorch as pl
    from omegaconf import OmegaConf, open_dict

    from nemo.collections.asr.models import EncDecCHATBPEModel, EncDecMultiVocabCHATBPEModel
    from nemo.utils.trainer_utils import resolve_trainer_cfg

    c = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(c):
        if not multivocab:
            del c.model["multivocab"]
        c.trainer.strategy.find_unused_parameters = fup
        c.trainer.max_steps = steps
        c.trainer.limit_train_batches = steps
        c.trainer.limit_val_batches = 0
        c.trainer.num_sanity_val_steps = 0
        c.trainer.val_check_interval = None
        c.trainer.check_val_every_n_epoch = 10**6
        c.trainer.devices = devices
    cls = EncDecMultiVocabCHATBPEModel if multivocab else EncDecCHATBPEModel
    trainer = pl.Trainer(**resolve_trainer_cfg(c.trainer))
    model = cls(cfg=c.model, trainer=trainer)
    import torch

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    trainer.fit(model)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    n = max(trainer.global_step, 1)
    return dt / n * 1e3


def run_compare(cfg, root, devices, steps):
    """Decompose the multi-vocab slowdown into DDP cost and model cost.

    The config pins DDPStrategy even for one device, so find_unused_parameters
    cannot simply be turned off to isolate it -- a single-vocab run with the flag
    ON is the control that prices the flag, and multi-vocab against THAT prices
    the model.
    """
    print(f"\n=== ms/step, {devices} GPU(s), {steps} steps ===", flush=True)
    # WARMUP, discarded. The first fit in a process pays CUDA context creation,
    # cuDNN/cuBLAS init and -- the big one -- numba JIT-compiling the banded
    # kernels. Without this the first config measured looks 2.4x SLOWER than the
    # others purely because it went first.
    time_steps(cfg, root, True, True, max(steps // 2, 4), devices)
    print("  (warmup done)", flush=True)
    a = time_steps(cfg, root, False, False, steps, devices)   # production single-vocab
    b = time_steps(cfg, root, False, True, steps, devices)    # same model, flag on
    c = time_steps(cfg, root, True, True, steps, devices)     # production multi-vocab
    print(f"  single-vocab, find_unused=False   {a:8.1f} ms/step   1.00x  (production single)", flush=True)
    print(f"  single-vocab, find_unused=True    {b:8.1f} ms/step   {b/a:5.2f}x  (cost of the DDP flag)", flush=True)
    print(f"  multi-vocab,  find_unused=True    {c:8.1f} ms/step   {c/a:5.2f}x  (production multi)", flush=True)
    print(f"\n  attribution: DDP flag {b/a:5.2f}x, extra heads {c/b:5.2f}x", flush=True)
    return [a, b, c]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", type=int, default=1, help="2 exercises DDP, where unused-parameter bugs appear")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--compare", action="store_true", help="time single- vs multi-vocab instead of smoke-testing")
    args = ap.parse_args()

    root = tempfile.mkdtemp(prefix="mvsmoke_")
    print(f"workdir: {root}  devices={args.devices}  steps={args.steps}", flush=True)
    try:
        corpus = os.path.join(root, "corpus.txt")
        make_corpus(corpus)
        tok_dirs = build_tokenizers(root, corpus)
        cuts = make_data(root)
        print(f"  tokenizers {[os.path.basename(d) for d in tok_dirs]}, {sum(1 for _ in open(cuts))} cuts", flush=True)
        cfg = toy_cfg(tok_dirs, cuts, args.steps, args.devices)
        if args.compare:
            run_compare(cfg, root, args.devices, args.steps)
            return 0
        ok = run_model(cfg, root, args.devices)
        print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE", flush=True)
        return 0 if ok else 1
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
