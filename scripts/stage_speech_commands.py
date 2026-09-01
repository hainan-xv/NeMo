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
"""Stage Google Speech Commands into the leaderboard cache our eval already reads.

WHY THIS DATASET. Every clip is ONE spoken word, ~1 second. That is the
short-utterance regime in pure form: the leaderboard as a whole yields only 1,589
utterances of 1-2 words (2.1%), where our streaming models score ~32-40 WER
against ~6 corpus-wide, and the failure is dominated by INSERTIONS -- the LLM
completing a plausible turn. Speech Commands gives ~4.9k such utterances (v0.02
test), so the effect can be measured at scale instead of in a 2% tail.

It also tests the end-of-utterance hypothesis from the full-context study
directly: a streaming model cannot observe that the utterance has ended, so it
should over-generate on 1-second clips, and a full-context model -- which sees
the trailing silence -- should not.

WHAT THIS IS NOT. Speech Commands is a KEYWORD-SPOTTING benchmark: a closed
35-word vocabulary of isolated words with no linguistic context, far outside the
conversational speech these models are trained on. WER here is a DIAGNOSTIC of
over-generation, not a quality number, and it is not comparable to any
leaderboard figure. Do not put it in a results table next to AMI.

SCORING IS SAFE. All 35 labels survive the leaderboard normalizer: none map to
empty, none collide, and the digit words are canonicalised consistently on both
sides ("zero"->"0" for reference and hypothesis alike; "1"->"one" and
"one"->"one" both land on "one"). So a class cannot be silently unscorable.

STDLIB ONLY. This runs on the login node -- the only host here with both internet
access and the lustre cache mounted -- and that Python is 3.9 with no numpy and
no soundfile. Hence `wave` and `shutil` rather than the usual audio stack. The
clips are already 16 kHz mono WAV, exactly what the harness wants, so staging is
a copy plus a manifest; no resampling is needed and none is done.

Usage (on the login node):
    python3 scripts/stage_speech_commands.py \
        --cache_dir /lustre/fsw/portfolios/llmservice/users/hainanx/leaderboard_cache \
        --work_dir  /lustre/fsw/portfolios/nemotron/users/hainanx/speech_commands

Then evaluate exactly like any other set:
    DATASETS="speech_commands:test" ./oci_launch.sh launch/eval_script.sh <exp> <chunk>
"""

import argparse
import json
import os
import shutil
import sys
import tarfile
import urllib.request
import wave

URLS = {
    1: "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz",
    2: "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
}
SR = 16000
# Not a word: the tarball ships long noise recordings under this directory, and
# they are not in any split list. Excluded so "all classes" means all WORDS.
NOISE_DIR = "_background_noise_"


def log(msg):
    print(msg, flush=True)


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        log(f"    already downloaded: {dest} ({os.path.getsize(dest) / 1e9:.2f} GB)")
        return dest
    tmp = dest + ".part"
    log(f"    downloading {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    os.replace(tmp, dest)
    log(f"    -> {dest} ({os.path.getsize(dest) / 1e9:.2f} GB)")
    return dest


def extract(tar_path, out_dir):
    marker = os.path.join(out_dir, ".extracted")
    if os.path.exists(marker):
        log(f"    already extracted: {out_dir}")
        return
    os.makedirs(out_dir, exist_ok=True)
    log(f"    extracting -> {out_dir}")
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf.getmembers():
            # Refuse absolute paths and traversal: this archive is trusted, but a
            # staging script that writes outside its target is worth preventing.
            p = os.path.normpath(m.name)
            if os.path.isabs(p) or p.startswith(".."):
                raise ValueError(f"unsafe path in archive: {m.name}")
            tf.extract(m, out_dir)
    open(marker, "w").close()


def read_split_list(root, split):
    """The OFFICIAL split, as shipped in the tarball.

    ``testing_list.txt`` / ``validation_list.txt`` hold "<label>/<file>.wav" lines;
    train is everything else. Using these keeps the split speaker-disjoint, which
    is the whole point of how the dataset was built -- deriving our own split
    would leak speakers and quietly flatter the model.
    """
    listed = {}
    for name, tag in (("testing_list.txt", "test"), ("validation_list.txt", "validation")):
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing {path} -- is this a Speech Commands tarball?")
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    listed[line] = tag
    if split in ("test", "validation"):
        return sorted(k for k, v in listed.items() if v == split)
    if split != "train":
        raise ValueError(f"unknown split {split!r}")
    out = []
    for label in sorted(os.listdir(root)):
        d = os.path.join(root, label)
        if not os.path.isdir(d) or label == NOISE_DIR:
            continue
        for fn in sorted(os.listdir(d)):
            rel = "{}/{}".format(label, fn)
            if fn.endswith(".wav") and rel not in listed:
                out.append(rel)
    return out


def wav_info(path):
    """(duration_seconds, framerate, channels) via stdlib -- no soundfile here."""
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate()), w.getframerate(), w.getnchannels()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", required=True, help="leaderboard cache root the eval reads")
    ap.add_argument("--work_dir", required=True, help="scratch for the tarball and extraction")
    ap.add_argument("--version", type=int, default=2, choices=(1, 2))
    ap.add_argument("--split", default="test", choices=("test", "validation", "train"))
    ap.add_argument("--dataset_name", default="speech_commands", help="name used in DATASETS=<name>:<split>")
    ap.add_argument("--limit", type=int, default=0, help="stage at most N clips (0 = all)")
    ap.add_argument("--refresh", action="store_true", help="re-stage even if the .done marker matches")
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    out_dir = os.path.join(args.cache_dir, args.dataset_name, args.split)
    done = os.path.join(out_dir, ".done")
    manifest_path = os.path.join(out_dir, "_cache_manifest.jsonl")

    log("==> Google Speech Commands v0.0{} [{}]".format(args.version, args.split))
    if os.path.exists(done) and not args.refresh:
        with open(done) as f:
            log("    already staged ({}). Use --refresh to redo.".format(f.read().strip()))
        return 0

    tar_path = download(
        URLS[args.version], os.path.join(args.work_dir, "speech_commands_v0.0{}.tar.gz".format(args.version))
    )
    root = os.path.join(args.work_dir, "v{}".format(args.version))
    extract(tar_path, root)

    rels = read_split_list(root, args.split)
    if args.limit:
        rels = rels[: args.limit]
    log("    {} clips in the official {} split".format(len(rels), args.split))

    os.makedirs(out_dir, exist_ok=True)
    records, skipped, durs = [], [], []
    labels = {}
    for i, rel in enumerate(rels):
        src = os.path.join(root, rel)
        label = rel.split("/")[0]
        try:
            dur, sr, ch = wav_info(src)
        except Exception as e:  # a truncated clip should not kill the whole staging run
            skipped.append((rel, str(e)))
            continue
        if sr != SR or ch != 1:
            skipped.append((rel, "sr={} ch={}".format(sr, ch)))
            continue
        dst_name = "{:06d}.wav".format(i)
        shutil.copyfile(src, os.path.join(out_dir, dst_name))
        durs.append(dur)
        labels[label] = labels.get(label, 0) + 1
        records.append(
            {
                "audio_filepath": os.path.join(out_dir, dst_name),
                "duration": round(dur, 4),
                # The spoken word IS the transcript. Lowercase, no punctuation --
                # the normalizer would strip both anyway, on reference and
                # hypothesis alike.
                "reference": label,
                # Extra keys the harness ignores, kept for per-class analysis and
                # so a staged clip can be traced back to its source file.
                "label": label,
                "source": rel,
                "speaker": os.path.basename(rel).split("_")[0],
            }
        )

    with open(manifest_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    durs.sort()
    log("    staged {} clips -> {}".format(len(records), out_dir))
    if durs:
        log(
            "    duration: min {:.2f}s  p05 {:.2f}s  median {:.2f}s  max {:.2f}s  total {:.2f} h".format(
                durs[0], durs[int(0.05 * len(durs))], durs[len(durs) // 2], durs[-1], sum(durs) / 3600.0
            )
        )
    log(
        "    {} distinct words; rarest {}, commonest {}".format(
            len(labels), min(labels.values()), max(labels.values())
        )
    )
    if skipped:
        log("    SKIPPED {} clip(s), first few: {}".format(len(skipped), skipped[:3]))

    with open(done, "w") as f:
        f.write("version=v0.0{} split={} n={}\n".format(args.version, args.split, len(records)))
    log("    done. Evaluate with:  DATASETS=\"{}:{}\"".format(args.dataset_name, args.split))
    return 0


if __name__ == "__main__":
    sys.exit(main())
