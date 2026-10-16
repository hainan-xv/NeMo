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

DIM = "\033[2m"
STRIKE = "\033[9;2m"
OVERLAY = "\u0336"  # COMBINING LONG STROKE OVERLAY


def struck(text: str, style: str) -> str:
    """Render text as crossed out.

    macOS Terminal.app ignores SGR 9 (strikethrough), so the ANSI form shows up
    merely dim there -- the line never appears. The unicode form puts a combining
    overlay after each character, which draws an actual stroke in any terminal
    and any font that has the glyph. It costs bytes, not columns: combining marks
    are zero-width, so the caller must advance the cursor by the BASE length.
    """
    if style == "ansi":
        return f"{STRIKE}{text}{RESET}"
    return DIM + "".join(c + OVERLAY for c in text) + RESET


RESET = "\033[0m"
BOLD = "\033[1m"


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
        self.col = 0
        self.shown = ""  # provisional text currently on screen
        self.vis = 0  # its printable length (excludes ANSI escapes)

    def send(self, text):
        try:
            self.s.sendall(text.encode("utf-8", "replace"))
        except OSError:
            raise ConnectionResetError

    def paint(self, provisional):
        """Show the fast front's current guess. Erasable until the block closes."""
        buf = ""
        if self.shown:
            buf += "\b" * self.vis + " " * self.vis + "\b" * self.vis
            self.col -= self.vis
            self.shown, self.vis = "", 0
        if provisional:
            if self.col + len(provisional) + 1 > 92:
                buf += "\r\n"
                self.col = 0
            piece = (" " if self.col else "") + provisional
            buf += DIM + piece + RESET
            self.shown, self.vis = piece, len(piece)
            self.col += len(piece)
        if buf:
            self.send(buf)

    def commit_block(self, provisional, final):
        """Freeze a block: wrong words stay, crossed out; the fix follows.

        The provisional text is erased and immediately re-drawn rather than
        simply overwritten, because it has to change colour -- struck-through
        for what the slow front rejected, normal for what it kept. The net
        effect on screen is append-only: nothing the model showed you
        disappears, so the self-correction stays readable.
        """
        from nemo.collections.speechlm2.parts.multi_front import Block, _deferred_tail, strike_markup

        # A trailing deletion is a DEFERRAL, not a rejection: the slow front moved
        # the word into the next block and it will be repainted there. Striking it
        # out claims the model got a word wrong that was always right -- which is
        # how "supposed" came to look like an error. Drop it and let the next
        # block paint it, so strikethrough means "actually rejected".
        deferred = _deferred_tail(Block(provisional, final))
        if deferred:
            keep = provisional.split()[: len(provisional.split()) - len(deferred)]
            provisional = " ".join(keep)

        buf = ""
        if self.shown:
            buf += "\b" * self.vis + " " * self.vis + "\b" * self.vis
            self.col -= self.vis
            self.shown, self.vis = "", 0
        for text, is_struck in strike_markup(provisional, final):
            if self.col + len(text) + 1 > 92:
                buf += "\r\n"
                self.col = 0
            piece = (" " if self.col else "") + text
            buf += struck(piece, self.a.strike) if is_struck else piece
            # Advance by the BASE length: combining overlays add bytes, not columns.
            self.col += len(piece)
        if buf:
            self.send(buf)

    def run(self):
        import numpy as np
        import torch

        a_, m, args = self.a, self.m, self.a
        dev = next(m.parameters()).device

        # Optional one-line preamble so the CLIENT can choose the operating point
        # per utterance: "CFG fast=2 slow=28 fast_delay=0 slow_delay=3\n".
        # Anything else is treated as audio, so older clients keep working.
        fast, slow = a_.fast, a_.slow
        fast_delay, slow_delay = a_.fast_delay, a_.slow_delay
        pre = b""
        self.s.settimeout(2.0)
        try:
            while b"\n" not in pre and len(pre) < 256:
                b = self.s.recv(64)
                if not b:
                    break
                pre += b
        except (socket.timeout, OSError):
            pass
        head, _, rest = pre.partition(b"\n")
        if head.startswith(b"CFG "):
            for tok in head.decode("ascii", "replace").split()[1:]:
                k, _, v = tok.partition("=")
                try:
                    iv = int(v)
                except ValueError:
                    continue
                if k == "fast_delay":
                    fast_delay = iv
                elif k == "slow_delay":
                    slow_delay = iv
                elif k == "fast":
                    fast = iv
                elif k == "slow":
                    slow = iv
            leftover = rest
        else:
            leftover = pre  # no preamble: every byte was audio
        fast_delay = max(0, min(a_.max_delay, fast_delay))
        slow_delay = max(0, min(a_.max_delay, slow_delay))
        # Validate the pair HERE, not at startup: it now arrives per session, and
        # a chunk size the checkpoint never trained still produces fluent text --
        # it would look like it worked while decoding an operating point the
        # model has never seen.
        trained = list(getattr(self.m.core_cfg, "chunk_size", []) or [])
        bad = [c for c in (fast, slow) if trained and c not in trained]
        if bad or slow % fast or slow == fast:
            why = (
                f"chunk sizes {bad} are not in {trained}"
                if bad
                else f"slow ({slow}) must be a different, exact multiple of fast ({fast})"
            )
            self.send(f"\r\n[server] rejected: {why}\r\n")
            return ""

        chunk_bytes = fast * SAMPLES_PER_FRAME * 2
        self.send(
            f"\r\n[server] fronts {fast}/{slow} frames, delays {fast_delay}/{slow_delay} "
            f"-> first paint {(fast+fast_delay)*0.08:.2f}s, settles {(slow+slow_delay)*0.08:.2f}s\r\n"
        )
        audio = np.zeros(0, dtype="float32")
        if leftover:
            if len(leftover) % 2:
                leftover = leftover[:-1]
            audio = np.frombuffer(leftover, dtype="<i2").astype("float32") / 32768.0
        committed, blocks = "", 0
        last_prov = ""
        t0 = None
        last = time.time()
        timed_out = False
        self.s.settimeout(1.0)
        while True:
            try:
                buf = self.s.recv(chunk_bytes)
            except socket.timeout:
                # A client that stops sending without closing (ffmpeg killed, ssh
                # half-open, laptop asleep) must not pin the session open forever:
                # that is what wedged the server and made later clients hang.
                if time.time() - last > a_.idle_timeout:
                    # Note it, but do not print here: the live line still has
                    # provisional text on it and a message mid-line garbles both.
                    timed_out = True
                    break
                continue
            except OSError:
                break
            if not buf:
                break
            last = time.time()
            # Drain whatever else is already waiting. Reading one chunk per cycle
            # means that if a cycle costs more than the chunk's duration the lag
            # grows without bound and the display falls progressively behind --
            # which looks exactly like "it is not real time".
            self.s.setblocking(False)
            try:
                while True:
                    more = self.s.recv(65536)
                    if not more:
                        break
                    buf += more
            except (BlockingIOError, OSError):
                pass
            finally:
                self.s.setblocking(True)
                self.s.settimeout(1.0)
            audio = np.concatenate([audio, np.frombuffer(buf, dtype="<i2").astype("float32") / 32768.0])
            if t0 is None:
                t0 = time.time()
            if time.time() - t0 > a_.max_seconds:
                break

            t_upd = time.time()
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
                        num_delay_frames=fast_delay,
                    )[0].strip()
                self.paint(prov)
                last_prov = prov
            if a_.timing:
                recvd = audio.size / SR
                wall = time.time() - t0 if t0 else 0.0
                print(
                    f"    +{wall:6.2f}s wall | {recvd:6.2f}s audio in | "
                    f"{(blocks*slow)*0.08:6.2f}s committed | lag {wall - recvd:+.2f}s | "
                    f"update {1000*(time.time()-t_upd):5.0f}ms",
                    flush=True,
                )
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
                        num_delay_frames=slow_delay,
                    )[0].strip()
                self.commit_block(last_prov, auth)
                last_prov = ""
                committed = (committed + " " + auth).strip() if auth else committed
                blocks += 1
                blo = blocks * slow

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
                        num_delay_frames=slow_delay,
                    )[0].strip()
                    self.commit_block(last_prov, auth)
                    last_prov = ""
                    committed = (committed + " " + auth).strip() if auth else committed
        if last_prov:
            self.commit_block(last_prov, last_prov)
        ok, diag = speech_report(audio)
        self.send("\r\n\r\n" + "-" * 72 + "\r\n")
        if timed_out:
            self.send(f"[server] no audio for {self.a.idle_timeout:.0f}s -- utterance ended\r\n")
        self.send(f"{BOLD}{committed}{RESET}\r\n")
        self.send(f"[audio] {audio.size/SR:.1f}s, {blocks} blocks\r\n")
        if diag:
            self.send(diag + "\r\n")
        if not ok:
            self.send(
                "[hint]  the transcript above is probably empty because the AUDIO is wrong, " "not the model.\r\n"
            )
        # Close cleanly. If the socket is closed while unread audio is still in
        # the receive buffer, Linux answers with RST instead of FIN and the
        # client's reader dies with ECONNRESET after a perfectly good session.
        # Shut down our write side, then drain whatever the client already sent.
        try:
            self.s.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        try:
            self.s.settimeout(2.0)
            while self.s.recv(65536):
                pass
        except (OSError, socket.timeout):
            pass
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
    # One decode at a time (single GPU), but each connection gets its own thread
    # so a wedged session cannot stop the server accepting -- a new client now
    # waits its turn instead of hanging on a dead predecessor.
    gpu = threading.Lock()

    def handle(conn, addr):
        print(f"[{time.strftime('%H:%M:%S')}] session from {addr}", flush=True)
        try:
            with gpu:
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

    while True:
        conn, addr = srv.accept()
        threading.Thread(target=handle, args=(conn, addr), daemon=True).start()


def client(args):
    """Relay stdin -> server and server -> stdout. Used as: ssh host ./asr"""
    s = socket.create_connection((args.host, args.port))
    # Tell the server this utterance's operating point before any audio.
    s.sendall(
        f"CFG fast={args.fast} slow={args.slow} "
        f"fast_delay={args.fast_delay} slow_delay={args.slow_delay}\n".encode()
    )

    def pump_back():
        while True:
            try:
                b = s.recv(4096)
            except (ConnectionResetError, OSError):
                break  # server finished and hung up; the transcript is already out
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
    ap.add_argument("--fast_delay", type=int, default=3, help="emission delay (frames) for the fast front")
    ap.add_argument("--slow_delay", type=int, default=3, help="emission delay (frames) for the slow front")
    ap.add_argument("--max_delay", type=int, default=6, help="largest delay this checkpoint was trained on")
    ap.add_argument("--max_seconds", type=float, default=120.0)
    ap.add_argument(
        "--idle_timeout",
        type=float,
        default=8.0,
        help="end an utterance after this long with no audio, even if the client never closes",
    )
    ap.add_argument(
        "--strike",
        default="unicode",
        choices=["unicode", "ansi"],
        help="how to cross out corrected words: unicode combining overlay (works everywhere, "
        "including macOS Terminal) or ANSI SGR 9 (ignored by some terminals)",
    )
    ap.add_argument("--timing", action="store_true", help="log a per-update latency breakdown")
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
