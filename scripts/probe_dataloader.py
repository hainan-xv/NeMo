#!/usr/bin/env python3
"""Time ONLY the Lhotse dataloader: how long until the first batch?

Four 4-node training jobs were killed by the idle-GPU reaper, all sitting at
step 0/2000 with their last log line "Creating a Lhotse DynamicBucketingSampler".
The identical launcher trained fine on Oct 7 (1662/2000 steps), so the model and
the config are not at fault; what changed is the shared data config, edited at
Oct 7 18:27 between the last success and the first failure.

This builds the dataloader alone -- no model, no GPUs -- and reports the time to
the first N batches, so we learn whether startup is SLOW (finishes, just past the
reaper's 30 min) or HUNG (never yields), without risking another 4-node job.
"""
import argparse, sys, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_cfg", required=True)
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--max_duration", type=float, default=20.0)
    ap.add_argument("--tokenizer", default="")
    args = ap.parse_args()

    from omegaconf import OmegaConf

    from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config

    cfg = OmegaConf.create({
        "input_cfg": args.input_cfg,
        "sample_rate": 16000,
        "shuffle": True,
        "shard_seed": "randomized",
        "num_workers": args.num_workers,
        "force_iterable_dataset": True,
        "use_bucketing": True,
        "bucket_duration_bins": [4.32, 6.0, 7.04, 7.92, 8.8, 9.6, 10.4, 11.12,
                                 11.89, 12.66, 13.47, 14.8, 16.92, 20.0],
        "bucket_batch_size": [38, 29, 25, 22, 20, 18, 17, 15, 14, 13, 12, 11, 10, 8],
        "max_duration": args.max_duration,
        "skip_missing_manifest_entries": True,
        "seed": 42,
    })

    class Identity(torch.utils.data.Dataset):
        def __getitem__(self, cuts):
            return cuts

    t0 = time.time()
    print(f"[{0:7.1f}s] building dataloader ...", flush=True)
    dl = get_lhotse_dataloader_from_config(cfg, global_rank=0, world_size=1, dataset=Identity())
    print(f"[{time.time()-t0:7.1f}s] dataloader object created; waiting for batches", flush=True)

    it = iter(dl)
    for i in range(args.batches):
        b = next(it)
        n = len(b) if hasattr(b, "__len__") else -1
        print(f"[{time.time()-t0:7.1f}s] batch {i+1}: {n} cuts", flush=True)
    print(f"[{time.time()-t0:7.1f}s] DONE -- dataloader is functional", flush=True)


if __name__ == "__main__":
    import torch  # noqa: E402  (imported late so the timer starts clean)
    main()
