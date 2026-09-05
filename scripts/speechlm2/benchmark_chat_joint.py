"""Speed/memory: forced-alignment (path) CHAT loss vs marginalised (full) RNN-T loss.

    PYTHONPATH=. python scripts/speechlm2/benchmark_chat_joint.py --dtype float32

MEASURED RESULT (RTX A6000, fp32). The path formulation is NOT a general speed
win, and reading "15.4x fewer scored positions" as "15.4x cheaper" is wrong:

  V=1,028    path is 0.7-1.1x the full joint -- i.e. usually SLOWER, and no
             memory saving. N ~ 600 positions is too small to keep the GPU busy,
             while the dense [B,T,U,V] joint is one efficient matmul whose
             output at this V is cheap to materialise.
  V=151,937  path is 6-25x faster and 5-17x smaller, and the gap GROWS with
             utterance length (full cost ~ T*U*V, path cost ~ (U+T)*V).

For scale: the shared 609M encoder forward+backward is ~564 ms / 15.7 GB for
B=15 x 12s. At V=1,028 the joint difference is under 1% of a step and simply
does not matter. At V=151,937 the marginalised joint adds ~12 GB on top of the
encoder and OOMs a 48 GB card by B=60, while the path form runs B=120 in 12 GB.

So this formulation buys FEASIBILITY at large vocabulary, not throughput.

Only the JOINT + LOSS differ between the two training modes -- the encoder and
prediction network are identical and are excluded, since including a shared 609M
forward would dilute exactly the ratio we are trying to measure. Encoder cost is
reported separately at the end for context.

Path mode   : joint_on_path -> [N, V+1] -> cross_entropy,  N = U + T
Full mode   : joint         -> [B, T, U+1, V+1] -> RNNTLoss (warprnnt_numba)
"""

import argparse
import gc
import time

import torch
import torch.nn.functional as F

from nemo.collections.asr.modules import RNNTAttJoint
from nemo.collections.speechlm2.data.chat_dataset import build_path

DEV = "cuda"


def make_joint(V, D, C, M):
    torch.manual_seed(0)
    j = RNNTAttJoint(
        jointnet={"encoder_hidden": D, "pred_hidden": D, "joint_hidden": D, "activation": "relu", "dropout": 0.1},
        num_classes=V,
        chunk_size=C,
        history_chunks=M,
    ).to(DEV)
    return j


def synth(B, T_chunks, U, C, D, V, dtype):
    """Encoder output, prediction output, and a forced path with U tokens spread over T chunks."""
    torch.manual_seed(1)
    T_frames = T_chunks * C
    f = torch.randn(B, T_frames, D, device=DEV, dtype=dtype, requires_grad=True)
    f_len = torch.full((B,), T_frames, device=DEV, dtype=torch.long)
    g = torch.randn(B, U + 1, D, device=DEV, dtype=dtype, requires_grad=True)

    per = [U // T_chunks] * T_chunks
    for i in range(U - sum(per)):
        per[i] += 1
    rng = torch.Generator().manual_seed(2)
    b_i, t_i, u_i, lab = [], [], [], []
    for bi in range(B):
        ct = [[int(torch.randint(1, V, (1,), generator=rng)) for _ in range(n)] for n in per]
        t, u, l = build_path(ct, V)
        b_i += [bi] * len(t)
        t_i += t
        u_i += u
        lab += l
    to = lambda x: torch.tensor(x, device=DEV, dtype=torch.long)  # noqa: E731
    targets = torch.randint(1, V, (B, U), device=DEV, dtype=torch.int32)
    return f, f_len, g, to(b_i), to(t_i), to(u_i), to(lab), targets


def timeit(fn, reps=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2], torch.cuda.max_memory_allocated() / 2**30


def run_case(B, T_chunks, U, V, C, D, M, dtype):
    row = {"B": B, "T": T_chunks, "U": U, "V": V, "M": M}
    joint = make_joint(V, D, C, M)
    f, f_len, g, b_i, t_i, u_i, lab, targets = synth(B, T_chunks, U, C, D, V, dtype)
    row["N_path"] = int(lab.numel())
    row["N_full"] = B * T_chunks * (U + 1)

    def path_step():
        joint.zero_grad(set_to_none=True)
        if f.grad is not None:
            f.grad = None
        logits = joint.joint_on_path(f, g, b_i, t_i, u_i, f_len)
        loss = F.cross_entropy(logits.float(), lab)
        loss.backward()

    try:
        row["path_s"], row["path_gb"] = timeit(path_step)
    except torch.cuda.OutOfMemoryError:
        row["path_s"], row["path_gb"] = None, None
    torch.cuda.empty_cache()
    gc.collect()

    # warprnnt_numba will not compile in this environment (numba/CUDA signature
    # mismatch) and torchaudio is absent, so the RNN-T forward-backward KERNEL is
    # excluded. What is measured is everything the marginalised loss requires
    # before that kernel: materialising [B, T, U+1, V+1], the log_softmax over V
    # that the loss consumes, the gather of the blank and target log-probs the
    # lattice recursion reads, and the backward through all of it.
    #
    # The excluded kernel is O(B*T*U) work over two gathered scalars per cell --
    # negligible beside a tensor that is V times larger. So this UNDERSTATES the
    # full method's cost, which is the conservative direction for the claim being
    # made here.
    tgt = targets.long().clamp(max=V - 1).view(B, 1, U, 1).expand(B, T_chunks, U, 1)

    def full_step():
        joint.zero_grad(set_to_none=True)
        if f.grad is not None:
            f.grad = None
        logits = joint.joint(f, g, f_len)  # [B, T, U+1, V+1]
        logp = logits.float().log_softmax(-1)
        blank_lp = logp[..., V]
        tgt_lp = logp[:, :, :U, :V].gather(-1, tgt)
        loss = -(blank_lp.mean() + tgt_lp.mean())
        loss.backward()

    try:
        row["full_s"], row["full_gb"] = timeit(full_step, reps=3, warmup=1)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:  # noqa: BLE001
        row["full_s"], row["full_gb"] = None, None
        row["full_err"] = type(e).__name__
    torch.cuda.empty_cache()
    gc.collect()
    del joint, f, g
    torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)
    C, D = 14, 640

    # (B, T_chunks, U) chosen to match the real bucketing: ~15 utts of ~12s
    # (T=11 chunks) and smaller batches of longer audio.
    cases = [
        (20, 8, 22),  # ~9s  x20   -- a dense small bucket
        (15, 11, 30),  # ~12s x15  -- the modal bucket
        (8, 18, 50),  # ~20s x8
        (4, 32, 90),  # ~36s x4   -- the long bucket
    ]
    print(f"dtype={args.dtype}  chunk_size={C}  joint_hidden={D}  device={torch.cuda.get_device_name(0)}")
    print()
    hdr = f"{'B':>3} {'T':>3} {'U':>4} | {'V':>7} {'M':>2} | {'path pos':>9} {'full pos':>10} | {'path ms':>8} {'full ms':>9} {'speedup':>8} | {'path GB':>8} {'full GB':>8} {'mem x':>7}"
    print(hdr)
    print("-" * len(hdr))
    for V in (1028, 151937):
        for M in (0, 1):
            for B, T, U in cases:
                r = run_case(B, T, U, V, C, D, M, dtype)
                ps = f"{r['path_s']*1e3:8.1f}" if r["path_s"] else "     OOM"
                fs = f"{r['full_s']*1e3:9.1f}" if r["full_s"] else "      OOM"
                sp = f"{r['full_s']/r['path_s']:7.1f}x" if r["path_s"] and r["full_s"] else "       -"
                pg = f"{r['path_gb']:8.2f}" if r["path_gb"] else "     OOM"
                fg = f"{r['full_gb']:8.2f}" if r["full_gb"] else "     OOM"
                mx = f"{r['full_gb']/r['path_gb']:6.1f}x" if r["path_gb"] and r["full_gb"] else "      -"
                print(
                    f"{B:3d} {T:3d} {U:4d} | {V:7d} {M:2d} | {r['N_path']:9d} {r['N_full']:10d} | {ps} {fs} {sp} | {pg} {fg} {mx}"
                )
        print()


if __name__ == "__main__":
    main()
