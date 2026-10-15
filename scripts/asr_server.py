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
"""Always-on multi-front ASR server: load the model once, transcribe on demand.

The model takes about a minute to load, which is intolerable per utterance. This
holds it resident and accepts sessions over a socket: each connection is one
utterance -- raw PCM in, live transcription out -- so a laptop can talk to it
whenever without paying the load again.

Desktop (leave running, e.g. in tmux):

    python scripts/asr_server.py --ckpt <ckpt>

MacBook (one utterance; Ctrl-C when done speaking):

    ffmpeg -f avfoundation -i ":1" -ar 16000 -ac 1 -f s16le -loglevel error - \\
      | ssh <host> ./asr

Transcription streams BACK over the same connection, so it appears on the
laptop. Provisional words are written and then rubbed out with backspaces when
the slow front overrules them, exactly as in the local viewer.

The server also checks that what arrives actually looks like speech. Clipped
sub-100 Hz rumble is still "audio arriving in real time", and silently
transcribing it to nothing wastes everyone's afternoon -- so it says so.
"""

import argparse
import math
import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SR = 16000
SAMPLES_PER_FRAME = 1280


def speech_report(a):
    """Does this look like speech? Returns (ok, one-line diagnosis)."""
    import numpy as np

    if a.size < SR // 2:
        return True, ""
    n = a.size
    S = np.abs(np.fft.rfft(a * np.hanning(n))) ** 2
    f = np.fft.rfftfreq(n, 1 / SR)
    tot = S.sum() or 1.0
    band = 100.0 * S[(f >= 300) & (f < 3400)].sum() / tot
    lowr = 100.0 * S[f < 100].sum() / tot
    clip = 100.0 * float((np.abs(a) >= 0.999).mean())
    fr = a[: n // 800 * 800].reshape(-1, 800)
    db = 20 * np.log10(np.maximum(np.sqrt((fr**2).mean(1)), 1e-9))
    floor = float(np.percentile(db, 10))
    if band < 15:
        return False, (
            f"!! this does not look like speech: only {band:.0f}% of energy in 300-3400 Hz "
            f"({lowr:.0f}% below 100 Hz). Check the mic device and input gain."
        )
    if clip > 1:
        return False, f"!! {clip:.1f}% of samples are clipped -- lower the input gain (-af volume=0.3)."
    if floor > -30:
        return (
            False,
            f"!! no silence between words (10th-pct frame {floor:.0f} dBFS) -- likely noise or gain too high.",
        )
    return True, f"   speech-band {band:.0f}%, clipping {clip:.1f}%, floor {floor:.0f} dBFS"


class Session:
    """One utterance: PCM in from the socket, display updates back out."""

    def __init__(self, model, sock, args, prompt_txt):
        self.m, self.s, self.a, self.p = model, sock, args, prompt_txt
        self.printed = self.col = 0
        self.shown = ""

    def send(self, text):
        try:
            self.s.sendall(text.encode("utf-8", "replace"))
        except OSError:
            raise ConnectionResetError

    def update(self, committed, provisional):
        buf = ""
        if self.shown:
            buf += "\b" * len(self.shown) + " " * len(self.shown) + "\b" * len(self.shown)
            self.col -= len(self.shown)
            self.shown = ""
        tail = committed[self.printed :]
        if tail:
            buf += tail
            self.printed = len(committed)
            self.col += len(tail)
        if provisional:
            if self.col + len(provisional) + 1 > 96:
                buf += "\n"
                self.col = 0
            piece = (" " if self.col and not provisional.startswith(" ") else "") + provisional
            buf += piece
            self.shown = piece
            self.col += len(piece)
        if buf:
            self.send(buf)

    def run(self):
        import numpy as np
        import torch

        a_, m, args = self.a, self.m, self.a
        fast, slow = a_.fast, a_.slow
        dev = next(m.parameters()).device
        chunk_bytes = fast * SAMPLES_PER_FRAME * 2

        self.send(f"\r\n[server] listening -- fronts {fast}/{slow}, first paint {fast*0.08:.2f}s\r\n")
        audio = np.zeros(0, dtype="float32")
        committed, blocks = "", 0
        t0 = None
        self.s.settimeout(1.0)
        while True:
            try:
                buf = self.s.recv(chunk_bytes)
            except socket.timeout:
                continue
            except OSError:
                break
            if not buf:
                break
            audio = np.concatenate([audio, np.frombuffer(buf, dtype="<i2").astype("float32") / 32768.0])
            if t0 is None:
                t0 = time.time()
            if time.time() - t0 > a_.max_seconds:
                break

            t = torch.from_numpy(audio).unsqueeze(0).to(dev)
            n = torch.tensor([audio.shape[0]], device=dev)
            with torch.inference_mode():
                ff = m.encode_frames(t, n, fast)
            avail = int(ff[0].shape[0])
            blo = blocks * slow
            edge = blo + ((avail - blo) // fast) * fast
            if edge > blo:
                with torch.inference_mode():
                    prov = m.generate(
                        t,
                        n,
                        system_prompt=self.p,
                        chunk_size_override=fast,
                        history_text=[committed],
                        frame_range=(blo, edge),
                        precomputed_frames=ff,
                    )[0].strip()
                self.update(committed, prov)
            while avail >= blo + slow:
                with torch.inference_mode():
                    sf_ = m.encode_frames(t, n, slow)
                    auth = m.generate(
                        t,
                        n,
                        system_prompt=self.p,
                        chunk_size_override=slow,
                        history_text=[committed],
                        frame_range=(blo, blo + slow),
                        precomputed_frames=sf_,
                    )[0].strip()
                committed = (committed + " " + auth).strip() if auth else committed
                blocks += 1
                blo = blocks * slow
                self.update(committed, "")

        # Flush the tail through the slow front so the result is authoritative.
        if audio.size:
            t = torch.from_numpy(audio).unsqueeze(0).to(dev)
            n = torch.tensor([audio.shape[0]], device=dev)
            with torch.inference_mode():
                sf_ = m.encode_frames(t, n, slow)
                total = int(sf_[0].shape[0])
                if total > blocks * slow:
                    auth = m.generate(
                        t,
                        n,
                        system_prompt=self.p,
                        chunk_size_override=slow,
                        history_text=[committed],
                        frame_range=(blocks * slow, total),
                        precomputed_frames=sf_,
                    )[0].strip()
                    committed = (committed + " " + auth).strip() if auth else committed
        self.update(committed, "")
        ok, diag = speech_report(audio)
        self.send(f"\r\n\r\n[final] {committed}\r\n")
        self.send(f"[audio] {audio.size/SR:.1f}s, {blocks} blocks\r\n")
        if diag:
            self.send(diag + "\r\n")
        if not ok:
            self.send(
                "[hint]  the transcript above is probably empty because the AUDIO is wrong, " "not the model.\r\n"
            )
        return committed


def serve(args):
    import torch

    from scripts.script_leaderboard_eval import load_model

    dev = torch.device(args.device)
    print("loading model ...", flush=True)
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

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(4)
    print(f"ready on {args.host}:{args.port} -- fronts {args.fast}/{args.slow}", flush=True)
    print("each connection is one utterance; the client closing its side ends it", flush=True)
    while True:
        conn, addr = srv.accept()
        print(f"[{time.strftime('%H:%M:%S')}] session from {addr}", flush=True)
        try:
            text = Session(model, conn, args, prompt_txt).run()
            print(f"[{time.strftime('%H:%M:%S')}] -> {text[:80]!r}", flush=True)
        except (ConnectionResetError, BrokenPipeError):
            print("  client went away", flush=True)
        except Exception as e:  # one bad session must not take the server down
            print(f"  session error: {type(e).__name__}: {e}", flush=True)
        finally:
            try:
                conn.close()
            except OSError:
                pass


def client(args):
    """Relay stdin -> server and server -> stdout. Used as: ssh host ./asr"""
    s = socket.create_connection((args.host, args.port))

    def pump_back():
        while True:
            b = s.recv(4096)
            if not b:
                break
            sys.stdout.buffer.write(b)
            sys.stdout.buffer.flush()

    t = threading.Thread(target=pump_back, daemon=True)
    t.start()
    try:
        while True:
            b = sys.stdin.buffer.read(4096)
            if not b:
                break
            s.sendall(b)
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    try:
        s.shutdown(socket.SHUT_WR)  # tell the server the utterance is over
    except OSError:
        pass
    t.join(timeout=120)
    s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--client", action="store_true", help="relay mode (run on the desktop via ssh)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9876)
    ap.add_argument("--fast", type=int, default=2)
    ap.add_argument("--slow", type=int, default=14)
    ap.add_argument("--max_seconds", type=float, default=120.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model_class", default="nemo.collections.speechlm2.models.script_model.ScriptSTTModel")
    args = ap.parse_args()
    if args.client:
        client(args)
    else:
        if not args.ckpt:
            raise SystemExit("--ckpt is required when running the server")
        serve(args)


if __name__ == "__main__":
    main()
