#!/usr/bin/env python3
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
"""Talk to the model and watch it transcribe, correcting itself as it goes.

Words appear at the fast front's latency (chunk 2 = 0.16 s) as PROVISIONAL text,
and every block (chunk 14 = 1.12 s) the slow front rewrites them with the
authoritative version -- the same self-correction the offline decoder
serialises, driven live.

    python scripts/multi_front_live.py --ckpt <ckpt>              # microphone
    python scripts/multi_front_live.py --ckpt <ckpt> --wav a.wav  # replay a file

ENTER ends the current utterance (the tail is flushed through the slow front so
the transcript is fully authoritative) and starts a new one. Ctrl-D or 'q' quits.

Capture goes through `arecord` rather than a Python audio package: ALSA is
already present everywhere this runs, while PortAudio/sounddevice needs a system
library install. --wav exists so the pipeline can be exercised on a machine with
no microphone at all.
"""

import argparse
import os
import queue
import shutil
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SR = 16000
SAMPLES_PER_FRAME = 1280  # one encoder frame = 0.08 s


class Capture:
    """Raw 16 kHz mono PCM arriving in a queue, from a mic or a file."""

    def __init__(self, wav=None, pcm=None, chunk_samples=2560):
        self.q = queue.Queue()
        self.wav = wav
        self.pcm = pcm
        self.chunk = chunk_samples
        self._stop = threading.Event()
        self._proc = None
        self._t = None

    def start(self):
        self._stop.clear()
        target = self._wav_loop if self.wav else (self._pcm_loop if self.pcm else self._mic_loop)
        self._t = threading.Thread(target=target, daemon=True)
        self._t.start()

    def _mic_loop(self):
        import numpy as np

        cmd = ["arecord", "-q", "-f", "S16_LE", "-r", str(SR), "-c", "1", "-t", "raw", "-"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        nbytes = self.chunk * 2
        while not self._stop.is_set():
            buf = self._proc.stdout.read(nbytes)
            if not buf:
                break
            self.q.put(np.frombuffer(buf, dtype="<i2").astype("float32") / 32768.0)
        self._kill()

    def _pcm_loop(self):
        """Raw s16le 16 kHz mono from a file or FIFO.

        This is how a REMOTE microphone reaches the model: the laptop pipes its
        mic into a FIFO over ssh while this process reads it, leaving stdin free
        for the ENTER that ends an utterance. Opening a FIFO blocks until the
        writer attaches, which is the behaviour we want -- it waits for the mic.
        """
        import numpy as np

        nbytes = self.chunk * 2
        with open(self.pcm, "rb") as f:
            while not self._stop.is_set():
                buf = f.read(nbytes)
                if not buf:
                    break
                self.q.put(np.frombuffer(buf, dtype="<i2").astype("float32") / 32768.0)
        self.q.put(None)

    def _wav_loop(self):
        """Replay a file at REAL TIME, so timing behaves as it would live."""
        import numpy as np
        import soundfile as sf

        data, sr = sf.read(self.wav, dtype="float32", always_2d=True)
        data = data[:, 0]
        if sr != SR:
            idx = (np.arange(int(len(data) * SR / sr)) * sr / SR).astype(int)
            data = data[np.clip(idx, 0, len(data) - 1)]
        t0 = time.time()
        for i in range(0, len(data), self.chunk):
            if self._stop.is_set():
                break
            piece = data[i : i + self.chunk]
            due = (i + len(piece)) / SR
            ahead = due - (time.time() - t0)
            if ahead > 0:
                time.sleep(ahead)
            self.q.put(piece)
        self.q.put(None)  # end of file

    def _kill(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=1)
            except Exception:
                self._proc.kill()
        self._proc = None

    def stop(self):
        self._stop.set()
        self._kill()


class Display:
    """One stream: provisional text is written, then rubbed out and replaced."""

    def __init__(self, width=96):
        self.width = width
        self.printed = 0
        self.col = 0
        self.shown = ""

    def reset(self):
        self.printed = self.col = 0
        self.shown = ""

    def update(self, committed, provisional):
        out = sys.stdout
        if self.shown:
            out.write("\b" * len(self.shown) + " " * len(self.shown) + "\b" * len(self.shown))
            self.col -= len(self.shown)
            self.shown = ""
        tail = committed[self.printed :]
        if tail:
            out.write(tail)
            self.printed = len(committed)
            self.col += len(tail)
        if provisional:
            if self.col + len(provisional) + 1 > self.width:
                out.write("\n")
                self.col = 0
            piece = (" " if self.col and not provisional.startswith(" ") else "") + provisional
            out.write(piece)
            self.shown = piece
            self.col += len(piece)
        out.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wav", default="", help="replay this file instead of the microphone")
    ap.add_argument(
        "--pcm",
        default="",
        help="read raw s16le 16kHz mono from this path (file or FIFO) -- use for a mic "
        "piped in over ssh from another machine",
    )
    ap.add_argument("--fast", type=int, default=2)
    ap.add_argument("--slow", type=int, default=14)
    ap.add_argument("--max_seconds", type=float, default=60.0, help="hard cap per utterance")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model_class", default="nemo.collections.speechlm2.models.script_model.ScriptSTTModel")
    args = ap.parse_args()

    if not args.wav and not args.pcm and not shutil.which("arecord"):
        raise SystemExit("arecord not found -- install alsa-utils, or use --wav to replay a file")

    import numpy as np
    import torch

    from scripts.script_leaderboard_eval import load_model

    dev = torch.device(args.device)
    model = load_model(args.ckpt, args.model_class, dev, torch.bfloat16, prompt_control=True)
    model.eval()
    trained = list(getattr(model.core_cfg, "chunk_size", []) or [])
    missing = [c for c in (args.fast, args.slow) if trained and c not in trained]
    if missing:
        raise SystemExit(f"checkpoint trained on {trained}; fronts {missing} are not among them")

    prompt_txt = (
        "You are doing streaming speech recognition. Given the transcript so far and the "
        "representation of the next audio chunk, output the words spoken in that chunk."
    )
    fast_frames = args.fast
    slow_frames = args.slow
    chunk_samples = fast_frames * SAMPLES_PER_FRAME

    def decode(audio_np, chunk, lo, hi, history):
        """Decode encoder frames [lo, hi) at one chunk size, given a text history."""
        a = torch.from_numpy(audio_np).unsqueeze(0).to(dev)
        n = torch.tensor([audio_np.shape[0]], device=dev)
        with torch.inference_mode():
            frames = model.encode_frames(a, n, chunk)
            if int(frames[0].shape[0]) < hi:
                return None
            return model.generate(
                a,
                n,
                system_prompt=prompt_txt,
                chunk_size_override=chunk,
                history_text=[history],
                frame_range=(lo, hi),
                precomputed_frames=frames,
            )[0].strip()

    print(
        f"\nfronts {args.fast}/{args.slow} -> first paint {args.fast*0.08:.2f}s, "
        f"settles every {args.slow*0.08:.2f}s"
    )
    print("ENTER ends the utterance and starts a new one.  'q' + ENTER quits.\n")

    disp = Display()
    utt = 0
    while True:
        utt += 1
        src = (
            f"replaying {os.path.basename(args.wav)}"
            if args.wav
            else (f"reading {args.pcm}" if args.pcm else "listening")
        )
        print(f"--- utterance {utt} ({src}) -- speak, then press ENTER ---")
        cap = Capture(wav=args.wav or None, pcm=args.pcm or None, chunk_samples=chunk_samples)

        stop_flag = threading.Event()
        quit_flag = threading.Event()

        def wait_enter():
            line = sys.stdin.readline()
            if line.strip().lower() == "q":
                quit_flag.set()
            stop_flag.set()

        watcher = threading.Thread(target=wait_enter, daemon=True)
        watcher.start()
        cap.start()

        audio = np.zeros(0, dtype="float32")
        committed, provisional = "", ""
        blocks_done = 0
        chunks_in_block = 0
        disp.reset()
        # Clock starts on the FIRST SAMPLE, not now: the cap is a limit on how
        # long an utterance may be, not on how long you may take to start
        # speaking. Timing from here would expire the utterance while the user
        # is still thinking, or while a remote writer is still attaching.
        t0 = None
        eof = False

        while not stop_flag.is_set() and not eof:
            try:
                piece = cap.q.get(timeout=0.1)
            except queue.Empty:
                continue
            if piece is None:  # wav exhausted
                eof = True
                break
            audio = np.concatenate([audio, piece])
            # Drain whatever else arrived while we were busy. Falling behind must
            # not build a backlog we then replay in slow motion.
            while True:
                try:
                    more = cap.q.get_nowait()
                except queue.Empty:
                    break
                if more is None:
                    eof = True
                    break
                audio = np.concatenate([audio, more])
            if t0 is None:
                t0 = time.time()
            if time.time() - t0 > args.max_seconds:
                break

            a = torch.from_numpy(audio).unsqueeze(0).to(dev)
            n = torch.tensor([audio.shape[0]], device=dev)
            with torch.inference_mode():
                fast_frames_t = model.encode_frames(a, n, args.fast)
            avail = int(fast_frames_t[0].shape[0])
            blo = blocks_done * slow_frames

            # Provisional = the fast front over the block SO FAR, in one call.
            # Decoding each 2-frame chunk separately would re-encode and re-prefill
            # per chunk and fall behind real time; this coalesces whatever audio
            # arrived since the last update, so a slow machine degrades to coarser
            # repaints rather than drifting further and further behind.
            edge = blo + ((avail - blo) // fast_frames) * fast_frames
            if edge > blo:
                with torch.inference_mode():
                    got = model.generate(
                        a,
                        n,
                        system_prompt=prompt_txt,
                        chunk_size_override=args.fast,
                        history_text=[committed],
                        frame_range=(blo, edge),
                        precomputed_frames=fast_frames_t,
                    )[0].strip()
                provisional = got
                disp.update(committed, provisional)

            # The block closed: the slow front overrules whatever is on screen.
            while avail >= blo + slow_frames:
                auth = decode(audio, args.slow, blo, blo + slow_frames, committed)
                if auth is None:
                    break
                committed = (committed + " " + auth).strip() if auth else committed
                provisional = ""
                blocks_done += 1
                blo = blocks_done * slow_frames
                disp.update(committed, "")

        cap.stop()
        # Flush the tail: run the slow front over the leftover audio so the final
        # transcript is authoritative rather than ending on provisional text.
        blo = blocks_done * slow_frames
        a = torch.from_numpy(audio).unsqueeze(0).to(dev) if audio.size else None
        if a is not None:
            with torch.inference_mode():
                n = torch.tensor([audio.shape[0]], device=dev)
                total = int(model.encode_frames(a, n, args.slow)[0].shape[0])
            if total > blo:
                auth = decode(audio, args.slow, blo, total, committed)
                if auth:
                    committed = (committed + " " + auth).strip()
        disp.update(committed, "")
        print(f"\n\nFINAL  {committed}")
        print(f"       ({len(audio)/SR:.1f}s audio, {blocks_done} full blocks)\n")

        if quit_flag.is_set():
            break
        if eof and not audio.size:
            # The source is exhausted (file finished, FIFO writer detached, or
            # arecord died). Re-opening it would return EOF instantly and spin
            # out empty utterances forever -- which is exactly what it did.
            print("audio source ended -- stopping.\n")
            break
        if eof and args.pcm:
            print("PCM writer detached -- stopping.\n")
            break
        if args.wav:
            # A file ends on its own, so ask before replaying. A live mic does
            # not: ENTER already means "that utterance is done, start the next".
            try:
                if input("ENTER to replay, 'q' to quit: ").strip().lower() == "q":
                    break
            except EOFError:
                break


if __name__ == "__main__":
    main()
