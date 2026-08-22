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
"""Report one leaderboard eval's per-dataset WER to Weights & Biases.

Backend-agnostic: the pooled-shard aggregator
(``scripts/speechlm_leaderboard_eval.py --aggregate``) prints
tab-separated ``RESULT<TAB>key<TAB>wer<TAB>time<TAB>n[<TAB>lat]`` lines, which the
launcher (``launch/eval_leaderboard.sh``) tees to ``aggregate.log``. This
script parses those + the ``run_config.yaml`` the launcher writes, and logs ONE
wandb run -- named by the decode config -- into a SEPARATE project from training.

Per-dataset WER is logged as a training-like STEPPED series: one step per dataset
in a fixed (alphabetical) order, then the macro average as the final step. So the
default ``wer`` line chart shows a single line per run over the datasets (+ avg at
the end), and several runs overlay for a dataset-by-dataset comparison. The same
numbers are also written to the run SUMMARY (``wer/avg`` + per-dataset ``wer/<key>``)
so the project Runs table can sort/compare across runs, plus a ``per_dataset`` table.

It must NEVER fail the eval: a missing wandb install / API key / network error is
caught and turned into a warning with exit code 0.
"""
import argparse
import os
import re
import sys


def _log(msg: str) -> None:
    print(msg, flush=True)


def parse_results(path: str):
    """Parse ``RESULT\\tkey\\twer\\ttime\\tn[\\tlat]`` lines -> list of dicts."""
    rows = []
    if not os.path.isfile(path):
        _log(f"[wandb-report] aggregate log not found: {path}")
        return rows
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("RESULT\t"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            key = parts[1]
            try:
                wer = float(parts[2])
            except ValueError:
                continue  # e.g. a "RESULT\tds\tERR\t..." failure row
            n = None
            lat = None
            if len(parts) >= 5:
                try:
                    n = int(parts[4])
                except ValueError:
                    n = None
            if len(parts) >= 6 and parts[5] not in ("-", ""):
                try:
                    lat = float(parts[5])
                except ValueError:
                    lat = None
            rows.append({"key": key, "wer": wer, "n": n, "lat": lat})
    return rows


def load_config(path: str) -> dict:
    """Load run_config.yaml into a dict (yaml if available, else a line parser)."""
    cfg = {}
    if not os.path.isfile(path):
        return cfg
    try:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:  # noqa: BLE001
        _log(f"[wandb-report] yaml parse of {path} failed ({e}); using line parser.")
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip().startswith("-") or ":" not in line:
                continue
            k, _, v = line.partition(":")
            cfg[k.strip()] = v.strip().strip('"')
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", required=True, help="wandb project (separate from the training project).")
    p.add_argument("--run_name", required=True, help="wandb run name (should encode the decode config).")
    p.add_argument("--results_dir", required=True, help="Eval RESULTS_DIR (holds aggregate.log + run_config.yaml).")
    p.add_argument("--aggregate_log", default=None, help="Override path to aggregate.log.")
    p.add_argument("--run_config", default=None, help="Override path to run_config.yaml.")
    p.add_argument("--group", default=None, help="wandb group (default: exp_name from run_config).")
    p.add_argument("--job_type", default=None, help="wandb job_type (default: backend from run_config).")
    p.add_argument("--tags", default="", help="Comma/space-separated wandb tags.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    agg = args.aggregate_log or os.path.join(args.results_dir, "aggregate.log")
    rcfg = args.run_config or os.path.join(args.results_dir, "run_config.yaml")

    rows = parse_results(agg)
    if not rows:
        _log("[wandb-report] no RESULT rows parsed; skipping wandb (nothing to report).")
        return 0

    # Fix a deterministic dataset order (alphabetical by key) so the stepped WER
    # series below lines up ACROSS runs: step 0 is always the same dataset in every
    # run (e.g. ami/test), step 1 the next, etc. -> overlaying two runs' `wer` line
    # charts compares them dataset-by-dataset. (The aggregators already emit rows in
    # sorted-key order, but we re-sort here to be robust.)
    rows = sorted(rows, key=lambda r: r["key"])
    dataset_order = [r["key"] for r in rows]

    # Macro average == the 'Average' both aggregators print (mean over datasets).
    avg = sum(r["wer"] for r in rows) / len(rows)
    lat_vals = [r["lat"] for r in rows if r["lat"] is not None]
    avg_lat = (sum(lat_vals) / len(lat_vals)) if lat_vals else None

    cfg = load_config(rcfg)
    # Surface the decode knobs as top-level config keys for easy wandb filtering.
    eval_tag = str(cfg.get("eval_tag", "") or "")
    m = re.match(r"^d(\d+)_(.+)$", eval_tag)
    if m:
        cfg.setdefault("delay", int(m.group(1)))
        cfg.setdefault("text_repr", m.group(2))
    cfg["macro_wer"] = avg
    cfg["num_datasets"] = len(rows)
    # step index -> dataset legend (so a reader can map x=step back to a dataset).
    cfg["dataset_order"] = dataset_order

    try:
        import wandb
    except Exception as e:  # noqa: BLE001
        _log(f"[wandb-report] wandb not importable ({e}); skipping (eval unaffected).")
        return 0

    offline = os.environ.get("WANDB_MODE", "").lower() == "offline"
    if not os.environ.get("WANDB_API_KEY") and not offline:
        _log("[wandb-report] no WANDB_API_KEY (and WANDB_MODE!=offline); skipping. "
             "Provide ~/.wandb_token or set WANDB_MODE=offline.")
        return 0

    tags = [t for t in re.split(r"[,\s]+", args.tags) if t]
    try:
        run = wandb.init(
            project=args.project,
            name=args.run_name,
            group=args.group or cfg.get("exp_name"),
            job_type=args.job_type or cfg.get("backend"),
            dir=args.results_dir,
            tags=tags or None,
            config=cfg,
            settings=wandb.Settings(init_timeout=90),
            reinit=True,
        )
    except Exception as e:  # noqa: BLE001
        _log(f"[wandb-report] wandb.init failed ({e}); skipping (eval unaffected).")
        return 0

    try:
        # Training-like stepped series: emit ONE step per dataset (in the fixed
        # dataset_order), then the macro average as the final step. Plotted as
        # `wer` vs step, each run is a single line over the 8 datasets (+ avg at the
        # end), so several runs overlay directly for a dataset-by-dataset compare.
        # We log an explicit, monotonically-increasing `step=` so mixing with the
        # summary/table writes below is safe.
        for i, r in enumerate(rows):
            step_metrics = {"wer": r["wer"]}
            if r["lat"] is not None:
                step_metrics["latency"] = r["lat"]
            if r["n"] is not None:
                step_metrics["n"] = r["n"]
            wandb.log(step_metrics, step=i)
        avg_step = len(rows)  # last time-step == the average
        avg_metrics = {"wer": avg}
        if avg_lat is not None:
            avg_metrics["latency"] = avg_lat
        wandb.log(avg_metrics, step=avg_step)

        # Runs-table columns (summary, step-independent): sortable macro avg + a
        # per-dataset `wer/<key>` so the project Runs table can compare across runs.
        run.summary["wer/avg"] = avg
        if avg_lat is not None:
            run.summary["latency/avg"] = avg_lat
        for r in rows:
            run.summary[f"wer/{r['key']}"] = r["wer"]
            if r["lat"] is not None:
                run.summary[f"latency/{r['key']}"] = r["lat"]

        _log(f"[wandb-report] logged {len(rows)} datasets (avg WER={avg:.2f}%) to "
             f"project='{args.project}' run='{args.run_name}'.")
        _log("[wandb-report] step->dataset: " +
             ", ".join(f"{i}={k}" for i, k in enumerate(dataset_order)) +
             f", {avg_step}=Average")

        # Sortable per-dataset table (also encodes step<->dataset mapping via idx).
        # Isolated: a wandb Table writes an artifact, which can fail on read-only
        # cache dirs; that must not drop the scalar series/summary logged above.
        try:
            table = wandb.Table(columns=["step", "dataset", "wer", "n", "latency"])
            for i, r in enumerate(rows):
                table.add_data(i, r["key"], r["wer"], r["n"], r["lat"])
            wandb.log({"per_dataset": table}, step=avg_step)
        except Exception as e:  # noqa: BLE001
            _log(f"[wandb-report] per_dataset table skipped ({e}); scalars already logged.")
    except Exception as e:  # noqa: BLE001
        _log(f"[wandb-report] logging error ({e}); continuing (eval unaffected).")
    finally:
        try:
            wandb.finish()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
