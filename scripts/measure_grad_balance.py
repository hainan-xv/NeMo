# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Gradient balance between the SCRIPT and CHAT heads at the SHARED encoder.

Both losses are per-target-token means, but that says nothing about how hard each
one pushes on the encoder they share. This measures the thing that matters:

    ||d L_script / d enc||   vs   ||d L_chat / d enc||

at the SAME encoder output, for a trained checkpoint. At init the ratio was
~3000:1 in SCRIPT's favour; the question is whether CHAT catches up as its joint
sharpens, or stays a passenger on a representation SCRIPT alone is shaping.
"""
import argparse, json

import torch
from omegaconf import OmegaConf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--chunks", default="7,14,28")
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--overrides", nargs="*", default=[],
                    help="OmegaConf dotlist, to match what the launcher overrode at train time")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    from nemo.collections.speechlm2 import ScriptSTTDataset
    from nemo.collections.speechlm2.models.joint_chat_script_model import JointChatScriptModel
    from nemo.collections.speechlm2.models.script_model import ScriptSTTModel

    # Activation checkpointing replays the forward inside backward; autograd.grad
    # on an intermediate does not re-enter the model's backward() override, so turn
    # it off. It changes memory, not gradients.
    cfg.model.activation_checkpointing = False
    model = JointChatScriptModel(OmegaConf.to_container(cfg.model, resolve=True)).cuda()

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded: {len(sd)} tensors | missing {len(missing)} | unexpected {len(unexpected)}", flush=True)
    chat_loaded = sum(1 for k in sd if k.startswith("chat."))
    print(f"  chat.* tensors in checkpoint: {chat_loaded}", flush=True)
    if chat_loaded == 0:
        raise SystemExit("checkpoint carries no CHAT head -- the ratio would be meaningless")
    model.train()

    from lhotse import CutSet
    from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoIterator

    cuts = CutSet(LazyNeMoIterator(args.manifest))
    ds_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)

    print(f"\n{'chunk':>6} {'n':>3} {'L_script':>10} {'L_chat':>10} "
          f"{'|g_script|':>12} {'|g_chat|':>12} {'chat/script':>12}", flush=True)
    for chunk in [int(c) for c in args.chunks.split(",")]:
        ds_cfg["chunk_size"] = [chunk]
        ds = ScriptSTTDataset(cfg=OmegaConf.create(ds_cfg), tokenizer=model.tokenizer)
        it = iter(cuts)
        gs_tot = gc_tot = ls_tot = lc_tot = 0.0
        n = 0
        for _ in range(args.batches):
            batch_cuts = CutSet.from_cuts([next(it) for _ in range(args.batch_size)])
            batch = ds[batch_cuts]
            batch = type(batch)(**{k: (v.cuda() if torch.is_tensor(v) else v) for k, v in vars(batch).items()})
            model.zero_grad(set_to_none=True)
            model._cap_enc = model._cap_enc_len = None
            try:
                out = ScriptSTTModel._training_step_inner(model, batch, 0)
                script_loss = out["loss"]
                enc = model._cap_enc
                chat_loss = model._chat_loss(batch)
                gs = torch.autograd.grad(script_loss, enc, retain_graph=True)[0].norm().item()
                gc = torch.autograd.grad(chat_loss, enc, retain_graph=True)[0].norm().item()
            except Exception as e:
                print(f"  (chunk {chunk} batch skipped: {type(e).__name__}: {e})", flush=True)
                continue
            gs_tot += gs; gc_tot += gc; ls_tot += script_loss.item(); lc_tot += chat_loss.item(); n += 1
        if n:
            print(f"{chunk:>6} {n:>3} {ls_tot/n:>10.4f} {lc_tot/n:>10.4f} "
                  f"{gs_tot/n:>12.3e} {gc_tot/n:>12.3e} {gc_tot/max(gs_tot,1e-30):>12.4f}", flush=True)


if __name__ == "__main__":
    main()
