#!/usr/bin/env python3
"""Compare banded-loss layouts on a REAL training batch.

The toy harness proves the layouts agree on synthetic chunk/token counts. Real
batches differ in the ways that actually stress the geometry: empty chunks from
pauses, wildly varying T/U per utterance, dynamic bucketing, and real forced
alignments. This runs one real batch through every layout and compares the loss
and the gradients.

    python scripts/chat_band_layout_check.py --nemo <arm>/averaged/top5-averaged.nemo
"""
import argparse
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--batches", type=int, default=2)
    ap.add_argument("--layouts", default="chunk,token,chunk_band")
    args = ap.parse_args()

    from nemo.collections.asr.models import ASRModel

    cfg = ASRModel.restore_from(restore_path=args.nemo, return_config=True)
    cls = ASRModel._get_model_class(cfg) if hasattr(ASRModel, "_get_model_class") else None
    model = ASRModel.restore_from(args.nemo, map_location="cuda")
    print(
        f"model: {type(model).__name__}  loss_type={getattr(model, 'loss_type', '?')} "
        f"band_chunks={getattr(model, 'band_chunks', '?')} band_side={getattr(model, 'band_side', '?')}",
        flush=True,
    )

    # train() for cuDNN's RNN backward; dropout OFF so the layouts are comparable
    # (their attention tensors have different shapes, so a shared seed would not
    # give them the same mask).
    model.train()
    n_dp = 0
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
            n_dp += 1
        if isinstance(mod, torch.nn.LSTM) and getattr(mod, "dropout", 0):
            mod.dropout = 0.0
            n_dp += 1
    print(f"dropout modules disabled: {n_dp}", flush=True)

    # The averaged .nemo is built with `~model.train_ds`, so train_ds is gone.
    # validation_ds survives and points at an ALIGNED manifest, which is what
    # this check actually needs: real audio, real forced alignments, real pauses
    # (empty chunks) and real T/U spread.
    if "train_ds" in model.cfg and model.cfg.train_ds is not None:
        model.setup_training_data(model.cfg.train_ds)
        dl = model._train_dl
        print("data: train_ds", flush=True)
    else:
        model.setup_validation_data(model.cfg.validation_ds)
        dl = model._validation_dl
        print(f"data: validation_ds -> {model.cfg.validation_ds.get('manifest_filepath', '?')}", flush=True)
    layouts = args.layouts.split(",")
    ok_all = True

    for nb, batch in enumerate(dl):
        if nb >= args.batches:
            break
        signal, signal_len, _t, _tl, cuts = batch
        signal, signal_len = signal.cuda(), signal_len.cuda()
        with torch.no_grad():
            enc0, enc_len = model.forward(input_signal=signal, input_signal_length=signal_len)
        nch = torch.div(enc_len + model.joint.chunk_size - 1, model.joint.chunk_size, rounding_mode="floor")
        print(
            f"\n--- batch {nb}: B={signal.shape[0]} enc={tuple(enc0.shape)} "
            f"chunks/utt min={int(nch.min())} max={int(nch.max())}",
            flush=True,
        )

        res = {}
        for layout in layouts:
            model.band_layout = layout
            model.zero_grad(set_to_none=True)
            e = enc0.clone().detach().requires_grad_(True)
            loss = model._banded_loss(e, enc_len, cuts)
            loss.backward()
            pg = {
                n: p.grad.detach().clone()
                for n, p in model.named_parameters()
                if p.grad is not None and ("joint" in n or "decoder" in n)
            }
            res[layout] = (float(loss), e.grad.detach().clone(), pg)
            print(f"    {layout:11s} loss = {float(loss):.8f}", flush=True)

        ref = layouts[0]
        la, ga, pa = res[ref]
        for name in layouts[1:]:
            lb, gb, pb = res[name]
            d_loss = abs(la - lb)
            d_enc = float((ga - gb).abs().max())
            scale = max(float(ga.abs().max()), 1e-12)
            worst = max((float((pa[n] - pb[n]).abs().max()) for n in pa if n in pb), default=0.0)
            good = d_loss <= 1e-4 * max(abs(la), 1.0) and d_enc / scale < 1e-3 and worst < 1e-2
            ok_all = ok_all and good
            print(
                f"    [{'PASS' if good else 'FAIL'}] {name} vs {ref}: dloss={d_loss:.3e} "
                f"dgrad_enc={d_enc:.3e} (rel {d_enc/scale:.2e}) dgrad_param={worst:.3e}",
                flush=True,
            )

    print("\nRESULT:", "ALL PASS" if ok_all else "FAILURES ABOVE", flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
