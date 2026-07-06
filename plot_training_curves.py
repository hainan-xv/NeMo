#!/usr/bin/env python3
"""Fetch a model's run logs from the cluster and plot its training curves, W&B-style.

Give it the model's run folder. That can be either:

  * a LOCAL directory, or
  * a REMOTE folder on the cluster -- an absolute path such as
    ``/lustre/fsw/.../results/<PROJECT>/<EXP>`` (fetched from the default host),
    or an explicit ``[user@]host:/abs/path``.

For a remote folder it rsyncs only the log files (``slurm-*.out`` / ``error-*.out``
-- never the multi-GB checkpoints or audio) into a local cache, then parses them.

It extracts the periodic stats we print during training

    [train] step 6500 stats: loss=0.1234  train_wer=0.2000  best_val_wer=0.1500  last_val_wer=0.1620

plus the validation WER recorded at each checkpoint

    ... step=6500-val_wer=0.1620.ckpt ...

and renders a single PNG with several panels (train loss, train WER, val WER,
and a train-vs-val WER overlay). Runs of the same experiment that were resumed
across multiple SLURM jobs are merged automatically (later data wins per step).

Usage:
    # remote (grid absolute path -> fetch + plot, end to end):
    python plot_training_curves.py /lustre/fsw/.../results/Streaming_SLM_Qwen1p7B/imend_loss_14_delay0_wanbd

    # explicit host / local folder / options:
    python plot_training_curves.py user@host:/abs/path -o curves.png
    python plot_training_curves.py /abs/local/folder --no-fetch
    python plot_training_curves.py <folder> --smooth 0.1 --dpi 160

Connection defaults (override via flags or env OCI_HOST / OCI_USER / SSH_KEY):
    host  draco-oci-login-01.draco-oci-iad.nvidia.com
    user  $USER
    key   ~/.ssh/clusters/draco-rno
"""

from __future__ import annotations

import argparse
import getpass
import os

# Headless-safe: never require a display.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
import re
import subprocess
import sys
from glob import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_HOST = os.environ.get("OCI_HOST", "draco-oci-login-01.draco-oci-iad.nvidia.com")
DEFAULT_USER = os.environ.get("OCI_USER", getpass.getuser())
DEFAULT_KEY = os.environ.get("SSH_KEY", os.path.expanduser("~/.ssh/clusters/draco-rno"))
DEFAULT_CACHE = os.path.expanduser("~/.cache/nemo_train_curves")

# ----------------------------------------------------------------------------- parsing

# [NeMo I 2026-07-04 18:08:12 streaming_stt_model:2188] [train] step 6500 stats:
#   loss=0.1234  train_wer=0.2000  best_val_wer=0.1500  last_val_wer=0.1620
_NUM = r"(n/a|nan|[-+0-9.eE]+)"
TRAIN_RE = re.compile(
    r"\[train\] step (\d+) stats:\s*"
    r"loss=" + _NUM + r"\s+"
    r"train_wer=" + _NUM + r"\s+"
    r"best_val_wer=" + _NUM + r"\s+"
    r"last_val_wer=" + _NUM
)
# leading NeMo timestamp on the same line, e.g. "2026-07-04 18:08:12"
TS_RE = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
# validation WER embedded in checkpoint names: step=6500-val_wer=0.1620
CKPT_RE = re.compile(r"step=(\d+)-val_wer=([0-9]*\.?[0-9]+)")
# "... global step 22500: 'val_wer' reached 0.1485 ..."
REACHED_RE = re.compile(r"global step (\d+):\s*'val_wer'\s*reached\s*([0-9]*\.?[0-9]+)")

LOG_GLOBS = ("slurm-*.out", "slurm-*", "*.out", "*.log", "*.txt")


def _to_float(tok: str):
    if tok is None:
        return None
    t = tok.strip().lower()
    if t in ("n/a", "nan", ""):
        return None
    try:
        v = float(t)
    except ValueError:
        return None
    return None if (v != v) else v  # drop NaN


def find_log_files(folder: str) -> list[str]:
    """Collect candidate log files under *folder*, preferring SLURM stdout files."""
    seen: dict[str, None] = {}
    for pat in LOG_GLOBS:
        for path in glob(os.path.join(folder, "**", pat), recursive=True):
            if os.path.isfile(path):
                seen.setdefault(os.path.realpath(path), None)
        # If SLURM logs exist, they contain everything we need; stop early so we
        # don't also slurp unrelated *.txt (e.g. transcripts) that can be huge.
        if pat.startswith("slurm-") and seen:
            break
    return sorted(seen)


def parse_logs(files: list[str]):
    """Return dicts keyed by global step, merged across all files (later wins)."""
    train = {}   # step -> {"loss", "train_wer", "best_val_wer", "last_val_wer", "ts"}
    val = {}     # step -> val_wer

    for path in files:
        try:
            with open(path, "r", errors="replace") as fh:
                for line in fh:
                    m = TRAIN_RE.search(line)
                    if m:
                        step = int(m.group(1))
                        tsm = TS_RE.search(line)
                        train[step] = {
                            "loss": _to_float(m.group(2)),
                            "train_wer": _to_float(m.group(3)),
                            "best_val_wer": _to_float(m.group(4)),
                            "last_val_wer": _to_float(m.group(5)),
                            "ts": tsm.group(1) if tsm else None,
                        }
                    for cm in CKPT_RE.finditer(line):
                        v = _to_float(cm.group(2))
                        if v is not None:
                            val[int(cm.group(1))] = v
                    rm = REACHED_RE.search(line)
                    if rm:
                        v = _to_float(rm.group(2))
                        if v is not None:
                            val[int(rm.group(1))] = v
        except OSError as e:
            print(f"  ! skipping {path}: {e}", file=sys.stderr)
    return train, val


def _series(train, key):
    steps = sorted(s for s, d in train.items() if d.get(key) is not None)
    return np.array(steps), np.array([train[s][key] for s in steps], dtype=float)


def ema(values: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0 or len(values) == 0:
        return values
    out = np.empty_like(values, dtype=float)
    m = values[0]
    for i, v in enumerate(values):
        m = alpha * v + (1 - alpha) * m
        out[i] = m
    return out


# ----------------------------------------------------------------------------- plotting


def plot(train, val, out_path, title, smooth, dpi):
    loss_x, loss_y = _series(train, "loss")
    twer_x, twer_y = _series(train, "train_wer")
    val_steps = np.array(sorted(val))
    val_y = np.array([val[s] for s in val_steps], dtype=float)

    panels = []
    if len(loss_x):
        panels.append("loss")
    if len(twer_x):
        panels.append("train_wer")
    if len(val_steps):
        panels.append("val_wer")
    if len(twer_x) and len(val_steps):
        panels.append("overlay")
    if not panels:
        print("No plottable metrics found in the logs.", file=sys.stderr)
        return False

    ncols = 2 if len(panels) > 1 else 1
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 4.2 * nrows), squeeze=False)
    axes = axes.ravel()

    def _finish(ax):
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    ai = 0
    for p in panels:
        ax = axes[ai]
        ai += 1
        if p == "loss":
            ax.plot(loss_x, loss_y, color="#1f77b4", alpha=0.28, lw=1, label="loss (raw)")
            if smooth > 0 and len(loss_y) > 1:
                ax.plot(loss_x, ema(loss_y, smooth), color="#1f77b4", lw=2,
                        label=f"loss (EMA α={smooth:g})")
            ax.set_title("Training loss")
            ax.set_xlabel("global step")
            ax.set_ylabel("loss")
        elif p == "train_wer":
            ax.plot(twer_x, twer_y, color="#ff7f0e", alpha=0.30, lw=1, label="train WER (raw)")
            if smooth > 0 and len(twer_y) > 1:
                ax.plot(twer_x, ema(twer_y, smooth), color="#ff7f0e", lw=2,
                        label=f"train WER (EMA α={smooth:g})")
            ax.set_title("Train WER")
            ax.set_xlabel("global step")
            ax.set_ylabel("WER")
        elif p == "val_wer":
            ax.plot(val_steps, val_y, "-o", color="#2ca02c", ms=4, lw=1.6, label="val WER")
            best_i = int(np.argmin(val_y))
            ax.plot(val_steps[best_i], val_y[best_i], "*", color="#d62728", ms=15,
                    label=f"best {val_y[best_i]:.4f} @ {val_steps[best_i]}")
            ax.set_title("Validation WER")
            ax.set_xlabel("global step")
            ax.set_ylabel("WER")
        elif p == "overlay":
            if smooth > 0 and len(twer_y) > 1:
                ax.plot(twer_x, ema(twer_y, smooth), color="#ff7f0e", lw=2, label="train WER")
            else:
                ax.plot(twer_x, twer_y, color="#ff7f0e", lw=1.5, label="train WER")
            ax.plot(val_steps, val_y, "-o", color="#2ca02c", ms=4, lw=1.6, label="val WER")
            ax.set_title("Train vs Val WER")
            ax.set_xlabel("global step")
            ax.set_ylabel("WER")
        _finish(ax)

    for j in range(ai, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


# ----------------------------------------------------------------------------- fetch


def _split_remote(path: str):
    """Return (host, remote_path) if *path* is remote, else (None, None).

    Recognises ``[user@]host:/abs/path`` and bare absolute paths that do not
    exist locally (treated as remote on the default host).
    """
    m = re.match(r"^([^/][^:]*):(/.*)$", path)
    if m and not os.path.isdir(path.split(":", 1)[0]):
        return m.group(1), m.group(2)  # host part may include user@
    if path.startswith("/") and not os.path.isdir(path):
        return "", path  # bare absolute path -> default host
    return None, None


def fetch_logs(host_spec: str, remote_path: str, args) -> str:
    """rsync only the log files of *remote_path* into a local cache; return the dir."""
    if "@" in host_spec:
        user, host = host_spec.split("@", 1)
    else:
        user, host = args.user, (host_spec or args.host)

    exp = os.path.basename(remote_path.rstrip("/")) or "run"
    local_dir = os.path.join(os.path.abspath(os.path.expanduser(args.cache_dir)), exp)
    os.makedirs(local_dir, exist_ok=True)

    ssh = f"ssh -i {args.ssh_key} -o StrictHostKeyChecking=no -o BatchMode=yes"
    cmd = [
        "rsync", "-az", "--prune-empty-dirs",
        "--include=slurm-*.out", "--include=slurm-*", "--include=error-*.out",
        "--exclude=*",
        "-e", ssh,
        f"{user}@{host}:{remote_path.rstrip('/')}/",
        f"{local_dir}/",
    ]
    print(f"Fetching logs: {user}@{host}:{remote_path}")
    print(f"  -> {local_dir}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit("error: rsync not found on PATH.")
    except subprocess.CalledProcessError as e:
        sys.exit(f"error: rsync failed (exit {e.returncode}). Check host/key/path and connectivity.")
    return local_dir


def resolve_folder(args) -> str:
    """Return a local folder containing the logs, fetching from the cluster if needed."""
    path = args.path
    host_spec, remote_path = _split_remote(path)

    if args.no_fetch or (host_spec is None):
        folder = os.path.abspath(os.path.expanduser(path))
        if not os.path.isdir(folder):
            sys.exit(f"error: not a local directory: {folder}\n"
                     f"       (pass a cluster path or [user@]host:/path to fetch automatically)")
        return folder

    return fetch_logs(host_spec, remote_path, args)


# ----------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="model run folder: a local dir, a cluster absolute path, or [user@]host:/path")
    ap.add_argument("-o", "--output", default=None,
                    help="output PNG path (default: ./<exp>_training_curves.png)")
    ap.add_argument("--smooth", type=float, default=0.1,
                    help="EMA smoothing factor in (0,1]; 0 disables (default: 0.1)")
    ap.add_argument("--dpi", type=int, default=150, help="PNG resolution (default: 150)")
    ap.add_argument("--no-fetch", action="store_true", help="treat PATH as a local folder; never rsync")
    ap.add_argument("--host", default=DEFAULT_HOST, help=f"ssh host for fetching (default: {DEFAULT_HOST})")
    ap.add_argument("--user", default=DEFAULT_USER, help=f"ssh user for fetching (default: {DEFAULT_USER})")
    ap.add_argument("--ssh-key", default=DEFAULT_KEY, help=f"ssh identity file (default: {DEFAULT_KEY})")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE,
                    help=f"where fetched logs are cached (default: {DEFAULT_CACHE})")
    args = ap.parse_args()

    folder = resolve_folder(args)

    files = find_log_files(folder)
    if not files:
        sys.exit(f"error: no log files found under {folder} (looked for {', '.join(LOG_GLOBS)}).")
    print(f"Scanning {len(files)} log file(s) under {folder}")

    train, val = parse_logs(files)
    print(f"  parsed {len(train)} train-stat step(s), {len(val)} validation point(s)")
    if train:
        last = max(train)
        d = train[last]
        print(f"  last logged step {last}: loss={d['loss']}, train_wer={d['train_wer']}, "
              f"last_val_wer={d['last_val_wer']}, best_val_wer={d['best_val_wer']}")
    if val:
        bstep = min(val, key=val.get)
        print(f"  best val_wer={val[bstep]:.4f} @ step {bstep}")

    exp = os.path.basename(folder.rstrip("/"))
    out_path = args.output or os.path.join(os.getcwd(), f"{exp}_training_curves.png")
    out_path = os.path.abspath(os.path.expanduser(out_path))
    title = f"{exp}  —  training curves"
    if plot(train, val, out_path, title, args.smooth, args.dpi):
        print(f"Wrote {out_path}")
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
