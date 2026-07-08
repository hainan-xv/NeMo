#!/usr/bin/env python3
"""Fetch ALL runs under a project and plot loss/WER curves comparing every model.

This is the multi-model companion to ``plot_training_curves.py``. Point it at a
project (by name, or a cluster/remote path) and it will:

  1. Discover every experiment (run) directory under the project on the cluster,
  2. rsync only each run's log files (``slurm-*.out`` / ``error-*.out`` -- never
     the multi-GB checkpoints or audio) into a per-project local cache,
  3. parse the periodic training stats we print

         [train] step 6500 stats: loss=0.1234  train_wer=0.2000  best_val_wer=0.1500  last_val_wer=0.1620

     plus the validation WER recorded at each checkpoint (``step=6500-val_wer=0.1620``),
  4. render ONE PNG overlaying every model: training loss, train WER, validation
     WER, and a best-val-WER bar chart for a quick ranking.

Runs of the same experiment resumed across multiple SLURM jobs are merged
automatically (later data wins per step), exactly as in the single-model script.

Usage:
    # whole project (default project name), fetch + plot end to end:
    python plot_project_comparison.py

    # explicit project name / path / options:
    python plot_project_comparison.py Streaming_SLM_Qwen1p7B
    python plot_project_comparison.py /lustre/fsw/.../results/Streaming_SLM_Qwen1p7B
    python plot_project_comparison.py user@host:/lustre/fsw/.../results/MyProject
    python plot_project_comparison.py --models imend_loss_14_delay0_wanbd imend_delayer2
    python plot_project_comparison.py --no-fetch          # use whatever is cached
    python plot_project_comparison.py --smooth 0.1 --dpi 160 -o compare.png

Connection defaults (override via flags or env OCI_HOST / OCI_USER / SSH_KEY):
    host  draco-oci-login-01.draco-oci-iad.nvidia.com
    user  $USER
    key   ~/.ssh/clusters/draco-rno
"""

from __future__ import annotations

import argparse
import os

# Headless-safe: never require a display.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reuse the single-model parser/fetch primitives so the two scripts never drift.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_training_curves as ptc  # noqa: E402

DEFAULT_HOST = ptc.DEFAULT_HOST
DEFAULT_USER = ptc.DEFAULT_USER
DEFAULT_KEY = ptc.DEFAULT_KEY
DEFAULT_CACHE = ptc.DEFAULT_CACHE
DEFAULT_RESULTS_ROOT = os.environ.get(
    "REMOTE_RESULTS_ROOT", "/lustre/fsw/portfolios/llmservice/users/hainanx/results"
)
DEFAULT_PROJECT = os.environ.get("PROJECT", "Streaming_SLM_Qwen1p7B")


# ----------------------------------------------------------------------------- resolve


def resolve_project(project_arg: str, results_root: str):
    """Return (host_spec, project_dir, project_name).

    ``host_spec`` is "" for the default host, a ``[user@]host`` string for an
    explicit remote, or None for a purely local path (used with --no-fetch).
    """
    p = project_arg
    m = re.match(r"^([^/][^:]*):(/.*)$", p)
    if m and not os.path.isdir(p.split(":", 1)[0]):
        host_spec, project_dir = m.group(1), m.group(2)
    elif p.startswith("/"):
        host_spec, project_dir = ("" if not os.path.isdir(p) else None), p
    elif os.path.isdir(p):
        host_spec, project_dir = None, os.path.abspath(p)
    else:
        # bare project name -> default host, results_root/<name>
        host_spec, project_dir = "", f"{results_root.rstrip('/')}/{p}"
    return host_spec, project_dir.rstrip("/"), os.path.basename(project_dir.rstrip("/"))


def discover_experiments(host_spec: str, project_dir: str, args) -> list[str]:
    """List run (experiment) directory names under *project_dir* on the cluster."""
    user, host = _user_host(host_spec, args)
    remote_cmd = (
        f"for d in {project_dir}/*/; do e=$(basename \"$d\"); [ -d \"$d\" ] && echo \"$e\"; done"
    )
    cmd = [
        "ssh", "-i", args.ssh_key, "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
        f"{user}@{host}", remote_cmd,
    ]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    except FileNotFoundError:
        sys.exit("error: ssh not found on PATH.")
    except subprocess.CalledProcessError as e:
        sys.exit(f"error: could not list experiments under {user}@{host}:{project_dir}\n{e.stderr}")
    exps = sorted({ln.strip() for ln in out.splitlines() if ln.strip()})
    return exps


def _user_host(host_spec: str, args):
    if host_spec and "@" in host_spec:
        user, host = host_spec.split("@", 1)
    else:
        user, host = args.user, (host_spec or args.host)
    return user, host


def fetch_one(host_spec: str, project_dir: str, exp: str, project_cache: str, args):
    """rsync one run's log files into ``project_cache/<exp>``; return dir or None."""
    user, host = _user_host(host_spec, args)
    local_dir = os.path.join(project_cache, exp)
    os.makedirs(local_dir, exist_ok=True)
    ssh = f"ssh -i {args.ssh_key} -o StrictHostKeyChecking=no -o BatchMode=yes"
    cmd = [
        "rsync", "-az", "--prune-empty-dirs",
        "--include=slurm-*.out", "--include=slurm-*", "--include=error-*.out",
        "--exclude=*",
        "-e", ssh,
        f"{user}@{host}:{project_dir}/{exp}/",
        f"{local_dir}/",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"  ! rsync failed for {exp}: {e.stderr.strip()}", file=sys.stderr)
        return local_dir if os.path.isdir(local_dir) else None
    return local_dir


# ----------------------------------------------------------------------------- plotting


def _color_list(n: int):
    if n <= 10:
        cmap = plt.get_cmap("tab10")
        return [cmap(i) for i in range(n)]
    if n <= 20:
        cmap = plt.get_cmap("tab20")
        return [cmap(i) for i in range(n)]
    cmap = plt.get_cmap("gist_ncar")
    return [cmap(i / max(1, n - 1)) for i in range(n)]


def plot_comparison(models: dict, out_path: str, title: str, smooth: float, dpi: int) -> bool:
    """models: {name: (train_dict, val_dict)} -> overlay every model in 4 panels."""
    names = [n for n in models if models[n][0] or models[n][1]]
    if not names:
        print("No plottable data for any model.", file=sys.stderr)
        return False
    colors = dict(zip(names, _color_list(len(names))))

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5), squeeze=False)
    ax_loss, ax_twer = axes[0]
    ax_vwer, ax_best = axes[1]

    def _line(ax, x, y, name, use_ema):
        if len(x) == 0:
            return False
        c = colors[name]
        if use_ema and smooth > 0 and len(y) > 1:
            ax.plot(x, ptc.ema(y, smooth), color=c, lw=1.9, label=name)
        else:
            ax.plot(x, y, color=c, lw=1.6, label=name)
        return True

    any_loss = any_twer = any_vwer = False
    best_pairs = []  # (name, best_val_wer)

    for name in names:
        train, val = models[name]
        lx, ly = ptc._series(train, "loss")
        tx, ty = ptc._series(train, "train_wer")
        any_loss |= _line(ax_loss, lx, ly, name, use_ema=True)
        any_twer |= _line(ax_twer, tx, ty, name, use_ema=True)
        if val:
            vsteps = np.array(sorted(val))
            vy = np.array([val[s] for s in vsteps], dtype=float)
            ax_vwer.plot(vsteps, vy, "-o", color=colors[name], ms=3.5, lw=1.5, label=name)
            any_vwer = True
            bi = int(np.argmin(vy))
            best_pairs.append((name, float(vy[bi])))
            ax_vwer.plot(vsteps[bi], vy[bi], "*", color=colors[name], ms=13)

    ax_loss.set_title("Training loss" + (f" (EMA α={smooth:g})" if smooth > 0 else ""))
    ax_loss.set_xlabel("global step"); ax_loss.set_ylabel("loss")
    ax_twer.set_title("Train WER" + (f" (EMA α={smooth:g})" if smooth > 0 else ""))
    ax_twer.set_xlabel("global step"); ax_twer.set_ylabel("WER")
    ax_vwer.set_title("Validation WER (★ = best)")
    ax_vwer.set_xlabel("global step"); ax_vwer.set_ylabel("WER")

    for ax, has in ((ax_loss, any_loss), (ax_twer, any_twer), (ax_vwer, any_vwer)):
        ax.grid(True, alpha=0.3)
        if has:
            ax.legend(loc="best", fontsize=8)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

    # --- best-val-WER ranking bar chart (lower is better) ---
    if best_pairs:
        best_pairs.sort(key=lambda kv: kv[1])  # best (lowest) first
        labels = [n for n, _ in best_pairs]
        vals = [v for _, v in best_pairs]
        ypos = np.arange(len(labels))[::-1]  # best at top
        ax_best.barh(ypos, vals, color=[colors[n] for n in labels])
        ax_best.set_yticks(ypos)
        ax_best.set_yticklabels(labels, fontsize=8)
        for yp, v in zip(ypos, vals):
            ax_best.text(v, yp, f" {v:.4f}", va="center", fontsize=8)
        ax_best.set_title("Best validation WER (lower is better)")
        ax_best.set_xlabel("WER")
        ax_best.grid(True, axis="x", alpha=0.3)
        ax_best.margins(x=0.15)
    else:
        ax_best.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


# ----------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project", nargs="?", default=DEFAULT_PROJECT,
                    help=f"project name, a cluster absolute path, or [user@]host:/path (default: {DEFAULT_PROJECT})")
    ap.add_argument("--models", nargs="+", default=None,
                    help="only these experiment names (default: all discovered under the project)")
    ap.add_argument("-o", "--output", default=None,
                    help="output PNG path (default: ./<project>_comparison.png)")
    ap.add_argument("--smooth", type=float, default=0.1,
                    help="EMA smoothing for loss/train-WER in (0,1]; 0 disables (default: 0.1)")
    ap.add_argument("--dpi", type=int, default=150, help="PNG resolution (default: 150)")
    ap.add_argument("--max-step", default=None,
                    help="truncate every curve to global steps <= this value. Pass an int, "
                         "'auto' to cap at the SHORTEST selected run's last step, or "
                         "'auto:<model>' to cap at a SPECIFIC run's last step (handy for "
                         "comparing a warm-started run against its baseline over equal #steps).")
    ap.add_argument("--round-step", type=int, default=0,
                    help="round the effective --max-step cap UP to the nearest multiple of this "
                         "(e.g. 10000). 0 disables (default: 0).")
    ap.add_argument("--no-fetch", action="store_true", help="use the local cache only; never rsync")
    ap.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT,
                    help=f"remote results root for bare project names (default: {DEFAULT_RESULTS_ROOT})")
    ap.add_argument("--host", default=DEFAULT_HOST, help=f"ssh host (default: {DEFAULT_HOST})")
    ap.add_argument("--user", default=DEFAULT_USER, help=f"ssh user (default: {DEFAULT_USER})")
    ap.add_argument("--ssh-key", default=DEFAULT_KEY, help=f"ssh identity file (default: {DEFAULT_KEY})")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE,
                    help=f"where fetched logs are cached (default: {DEFAULT_CACHE})")
    args = ap.parse_args()

    host_spec, project_dir, project_name = resolve_project(args.project, args.results_root)
    project_cache = os.path.join(os.path.abspath(os.path.expanduser(args.cache_dir)), project_name)

    # --- Determine the set of experiments ---
    if args.no_fetch or host_spec is None:
        # Local: list subdirs of the cache (or of a local project dir).
        base = project_dir if (host_spec is None and os.path.isdir(project_dir)) else project_cache
        if not os.path.isdir(base):
            sys.exit(f"error: no local directory to read runs from: {base}\n"
                     f"       (run without --no-fetch to fetch from the cluster first)")
        exps = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
        local_base = base
    else:
        print(f"Discovering runs under {_user_host(host_spec, args)[0]}@"
              f"{_user_host(host_spec, args)[1]}:{project_dir} ...")
        exps = discover_experiments(host_spec, project_dir, args)
        local_base = project_cache

    if args.models:
        wanted = set(args.models)
        exps = [e for e in exps if e in wanted]
        missing = wanted - set(exps)
        if missing:
            print(f"  ! requested models not found under project: {', '.join(sorted(missing))}", file=sys.stderr)
    if not exps:
        sys.exit(f"error: no experiments found for project '{project_name}'.")
    print(f"Models ({len(exps)}): {', '.join(exps)}")

    # --- Fetch + parse each model ---
    models: dict = {}
    for exp in exps:
        if args.no_fetch or host_spec is None:
            folder = os.path.join(local_base, exp)
        else:
            folder = fetch_one(host_spec, project_dir, exp, project_cache, args)
        if not folder or not os.path.isdir(folder):
            print(f"  - {exp}: no local logs; skipping", file=sys.stderr)
            continue
        files = ptc.find_log_files(folder)
        if not files:
            print(f"  - {exp}: no log files found; skipping", file=sys.stderr)
            continue
        train, val = ptc.parse_logs(files)
        n_best = f"best_val_wer={min(val.values()):.4f}" if val else "no val"
        print(f"  - {exp}: {len(train)} train-steps, {len(val)} val-points ({n_best})")
        if train or val:
            models[exp] = (train, val)

    if not models:
        sys.exit("error: parsed no usable data from any model.")

    # --- Optional step cap: truncate all curves to a common step budget ---
    cap = None
    if args.max_step is not None:
        def _max_step(tv):
            tr, va = tv
            steps = list(tr.keys()) + list(va.keys())
            return max(steps) if steps else 0
        ms = str(args.max_step).lower()
        if ms == "auto":
            cap = min(_max_step(tv) for tv in models.values())
        elif ms.startswith("auto:"):
            ref = args.max_step.split(":", 1)[1]
            if ref not in models:
                sys.exit(f"error: --max-step auto:{ref} but '{ref}' is not among plotted models: "
                         f"{', '.join(models)}")
            cap = _max_step(models[ref])
        else:
            cap = int(args.max_step)
        if args.round_step and args.round_step > 0 and cap > 0:
            import math
            cap = int(math.ceil(cap / args.round_step) * args.round_step)
        for name, (tr, va) in list(models.items()):
            tr = {s: v for s, v in tr.items() if s <= cap}
            va = {s: v for s, v in va.items() if s <= cap}
            models[name] = (tr, va)
        print(f"Capped all curves to global step <= {cap}")

    out_path = args.output or os.path.join(os.getcwd(), f"{project_name}_comparison.png")
    out_path = os.path.abspath(os.path.expanduser(out_path))
    title = f"{project_name}  —  model comparison ({len(models)} runs)"
    if cap is not None:
        title += f"  [first {cap} steps]"
    if plot_comparison(models, out_path, title, args.smooth, args.dpi):
        print(f"Wrote {out_path}")
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
