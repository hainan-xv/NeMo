# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Does the seeded CHAT head decode at step 0, BEFORE any training?

If yes, the decode path is sound and training under a mismatched objective is what
collapsed it. If no, the bug is in how the joint model mounts/drives the CHAT head
-- and no amount of config alignment will help.

Also decodes with the DONOR's own encoder context ([70,13]) alongside ours
([84, chunk-1]), to separate "wrong left context" from "wrong wiring".
"""
import argparse

import torch
from omegaconf import OmegaConf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--donor", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=14)
    ap.add_argument("--overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    cfg.model.activation_checkpointing = False

    import sys
    sys.path.insert(0, "/code/examples/speechlm2")
    from script_train import _init_from_chat_nemo

    from nemo.collections.speechlm2.models.joint_chat_script_model import JointChatScriptModel

    model = JointChatScriptModel(OmegaConf.to_container(cfg.model, resolve=True)).cuda().eval()
    _init_from_chat_nemo(model, args.donor)

    from lhotse import CutSet
    from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoIterator

    cuts = CutSet.from_cuts([c for _, c in zip(range(args.n), LazyNeMoIterator(args.manifest))])
    audios, lens = [], []
    for c in cuts:
        a = torch.as_tensor(c.load_audio()).squeeze(0)
        audios.append(a)
        lens.append(a.shape[-1])
    T = max(lens)
    sig = torch.zeros(len(audios), T)
    for i, a in enumerate(audios):
        sig[i, : a.shape[-1]] = a
    sig = sig.cuda()
    slen = torch.tensor(lens, device="cuda")

    for left, label in ((int(cfg.model.att_context_size[0]), "ours"), (70, "donor [70,13]")):
        model.core_cfg.att_context_size = [left, cfg.model.att_context_size[1]]
        model._set_encoder_att_context(args.chunk)
        model._cap_enc = model._cap_enc_len = None
        with torch.no_grad():
            model.perception(input_signal=sig, input_signal_length=slen)
        enc, enc_len = model._cap_enc, model._cap_enc_len
        hyps = model._chat_decode(enc, enc_len, args.chunk)
        print(f"\n=== left_context={left} ({label}), chunk={args.chunk} | enc {tuple(enc.shape)} ===", flush=True)
        for c, h in zip(cuts, hyps):
            ref = " ".join(s.text for s in (c.supervisions or []) if s.text)
            print(f"  REF: {ref[:90]}")
            print(f"  HYP: {h[:90]!r}")


if __name__ == "__main__":
    main()
