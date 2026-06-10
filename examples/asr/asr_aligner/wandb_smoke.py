#!/usr/bin/env python3
"""Minimal W&B scalar logging smoke test.

Use this to separate W&B upload/UI issues from NeMo/Lightning/model logging.

Examples:
    python examples/asr/asr_aligner/wandb_smoke.py --project aligner_local
    WANDB_MODE=offline python examples/asr/asr_aligner/wandb_smoke.py
"""

import argparse
import math
import time

import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="aligner_local")
    parser.add_argument("--name", default="wandb_smoke")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument(
        "--explicit-step",
        action="store_true",
        help="Pass step=... to wandb.log. By default, let W&B use its internal step counter.",
    )
    args = parser.parse_args()

    run = wandb.init(project=args.project, name=args.name)
    print(f"wandb run id={run.id} url={run.url} mode={getattr(run, 'mode', '?')}")

    for step in range(args.steps):
        value = math.exp(-step / 10.0)
        metrics = {
            "loss_plain": value,
            "step_float_plain": float(step),
            "smoke/loss": value,
            "smoke/step_float": float(step),
        }
        if args.explicit_step:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)
        print(f"logged step={step} loss_plain={value:.6f}")
        time.sleep(args.sleep)

    run.finish()
    print("finished")


if __name__ == "__main__":
    main()
