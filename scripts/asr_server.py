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
CLEAR_LINE = "\033[K"  # erase to end of LINE -- no absolute position, scroll-safe


class _LiveAudio:
    """Hand a decode its audio as it is captured, instead of all at once.

    The decode thread calls ``read`` and parks until the socket thread has
    pushed that many samples, so the state machine advances in step with the
    microphone. ``close`` marks the end: the next read returns whatever is left,
    and a short read is how the decode learns the stream is over -- a live
    stream cannot know its own length in advance.

    ``close`` must be called on every exit path, or the decode thread stays
    parked forever holding the session open.
    """

    def __init__(self, device):
        import numpy as np

        self._np = np
        self._device = device
        self._buf = np.zeros(0, dtype="float32")
        self._closed = False
        self._cv = threading.Condition()

    def push(self, samples):
        with self._cv:
            self._buf = self._np.concatenate([self._buf, samples])
            self._cv.notify_all()

    def close(self):
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def read(self, _stream, start, n):
        import torch

        with self._cv:
            while len(self._buf) < start + n and not self._closed:
                self._cv.wait(timeout=0.5)
            chunk = self._buf[start : start + n]
        return torch.from_numpy(self._np.ascontiguousarray(chunk)).to(self._device)


def _pcm(buf, carry):
    """Decode little-endian 16-bit PCM, carrying a split sample to the next read.

    Returns (samples, new_carry). A socket read can end mid-sample, so the
    length is not reliably even; truncating instead of carrying would misalign
    every following sample by one byte and turn the rest of the utterance into
    noise.
    """
    import numpy as np

    buf = carry + bytes(buf)
    carry = b""
    if len(buf) % 2:
        buf, carry = buf[:-1], buf[-1:]
    return np.frombuffer(buf, dtype="<i2").astype("float32") / 32768.0, carry


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

    def __init__(self, model, sock, args, prompt_txt, alt=None, nemotron=None):
        self.m, self.s, self.a, self.p = model, sock, args, prompt_txt
        # The INTERLEAVED baseline, as (model, its own system prompt), or None
        # when the server was started without --interleaved_ckpt. It is a second
        # arm rather than a second server so both can be compared back to back
        # without reloading, and it carries its OWN instruction: the SCRIPT
        # prompt would put it out of distribution (see scripts/speechlm_asr_shim).
        self.alt = alt
        # (model, [trained chunk sizes]) for the nemotron RNN-T arm, or None.
        self.nemotron = nemotron
        self.cols = 88
        self.prompt_suffix = None  # inference-time text appended after the controls
        self.col = 0  # column of the PERMANENT text
        self.vis = 0  # on-screen length of the provisional, 0 if none

    def send(self, text):
        try:
            self.s.sendall(text.encode("utf-8", "replace"))
        except OSError:
            raise ConnectionResetError

    def paint(self, provisional):
        """Show the fast front's current guess, on ONE line, erasable.

        Two earlier approaches both failed for structural reasons:
          * backspaces alone cannot cross a line wrap, so a provisional longer
            than the line left debris;
          * DECSC/DECRC save an ABSOLUTE position, so as soon as the screen
            scrolls -- which it does constantly in a real session -- restore
            lands on the wrong row and the clear wipes the wrong text.

        So keep the provisional strictly within one line: wrap to a fresh line
        first if it will not fit, and show only its tail if it cannot fit at all.
        Then backspacing to its start is always safe, with no absolute cursor
        state to go stale. The truncation is cosmetic -- the block's full text is
        written permanently when it commits.
        """
        buf = ""
        if self.vis:
            buf += "\b" * self.vis + CLEAR_LINE
            self.vis = 0
        if provisional:
            # NEVER break the line for the provisional. The break would be
            # permanent while the text is transient, so the committed transcript
            # ended up wrapping at 35-55 columns instead of filling the width --
            # line breaks were being decided by the fast front's length. Only
            # commit_block, which writes permanent text, may wrap.
            avail = self.cols - self.col - (1 if self.col else 0)
            if avail >= 8:
                text = provisional if len(provisional) <= avail else "..." + provisional[-(avail - 3) :]
                piece = (" " if self.col else "") + text
                buf += DIM + piece + RESET
                self.vis = len(piece)  # on-screen length, so we can back over it
        if buf:
            self.send(buf)

    def _wrapped(self, text, start_col):
        """Break text so no line exceeds the client's width."""
        out, col = "", start_col
        for word in text.split(" "):
            if not word:
                continue
            need = len(word) + (1 if col else 0)
            if col and col + need > self.cols:
                out += "\r\n"
                col = 0
                need = len(word)
            out += (" " if col else "") + word
            col += need
        return out

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
        if self.vis:
            buf += "\b" * self.vis + CLEAR_LINE
            self.vis = 0
        # Wrap per WORD, not per segment: a segment can be a long span, and
        # breaking only between segments lets one overflow the line, after which
        # the terminal soft-wraps and the column count no longer matches reality.
        for text, is_struck in strike_markup(provisional, final):
            for word in text.split(" "):
                if not word:
                    continue
                need = len(word) + (1 if self.col else 0)
                if self.col and self.col + need > self.cols:
                    buf += "\r\n"
                    self.col = 0
                    need = len(word)
                piece = (" " if self.col else "") + word
                buf += struck(piece, self.a.strike) if is_struck else piece
                # Advance by the BASE length: overlays add bytes, not columns.
                self.col += need
        if buf:
            self.send(buf)

    def emit_final(self, text, audio_s, blocks, diag, ok, timed_out):
        """Render the end-of-utterance summary. Overridden by the web front-end.

        A named hook rather than a pile of send() calls: the browser needs the
        transcript and the diagnostics as SEPARATE things, and classifying them
        by eye ("lines starting with [ are meta") put the speech-band line
        inside the transcript box.
        """
        self.send("\r\n\r\n" + "-" * 72 + "\r\n")
        if timed_out:
            self.send(f"[server] no audio for {self.a.idle_timeout:.0f}s -- utterance ended\r\n")
        words, line, lines = text.split(), "", []
        for w in words:
            if line and len(line) + 1 + len(w) > self.cols:
                lines.append(line)
                line = w
            else:
                line = f"{line} {w}" if line else w
        if line:
            lines.append(line)
        for ln in lines or [""]:
            self.send(f"{BOLD}{ln}{RESET}\r\n")
        self.send(f"[audio] {audio_s:.1f}s, {blocks} blocks\r\n")
        if diag:
            self.send(diag + "\r\n")
        if not ok:
            self.send(
                "[hint]  the transcript above is probably empty because the AUDIO is wrong, " "not the model.\r\n"
            )

    def _nt_emit(self, txt, shown, final):
        """Show RNN-T output, committing only COMPLETE words.

        The decoder emits subword tokens, so a word can straddle two streaming
        steps: one step ends "...architect", the next has "...architecture".
        Committing the raw delta makes that a separate segment, and the display
        puts a space before every segment -- which is how "architecture" came
        out as "architect ure".

        So commit only up to the last space (whole words), and paint the
        trailing partial word as provisional until it completes. Nothing is
        ever revised here; the provisional tail is just not yet a word.
        """
        cut = txt.rfind(" ") + 1 if not final else len(txt)
        if cut > len(shown):
            new = txt[len(shown) : cut].strip()
            if new:
                self.commit_block(new, new)
            shown = txt[:cut]
        if not final:
            tail = txt[cut:].strip()
            self.paint(tail)
        else:
            self.paint("")
        return shown

    def run_nemotron_2f(self, leftover, fast, slow):
        """Self-correcting RNN-T: a fast front that is overruled by a slow one.

        Verified on 8/8 AMI utterances that the committed text equals decoding
        with the SLOW front alone, so the responsiveness costs nothing in final
        accuracy -- the RNN-T analogue of SCRIPT's mf_final == single-slow.

        Resync is a state copy here rather than SCRIPT's text re-prefill: the
        encoder caches and previous_hypotheses go into conformer_stream_step
        and come back out, so the fast front can simply adopt the slow front's
        state at each block boundary.
        """
        import numpy as np
        import torch

        from nemo.collections.speechlm2.parts.nemotron_stream import NemotronTwoFront

        a_ = self.a
        model, chunks = self.nemotron
        bad = [c for c in (fast, slow) if c not in chunks]
        if bad or slow % fast or slow == fast:
            why = (
                f"chunk sizes {bad} are not in {chunks}"
                if bad
                else f"slow ({slow}) must be a different, exact multiple of fast ({fast})"
            )
            self.send(f"\r\n[server] rejected: {why}\r\n")
            return ""
        dev = next(model.parameters()).device
        tf = NemotronTwoFront(model, fast, slow, dev)
        self.send(
            f"\r\n[server] nemotron RNN-T, fronts {fast}/{slow} frames "
            f"-> first paint {fast*0.08:.2f}s, settles {slow*0.08:.2f}s\r\n"
            "[server] the slow front overrules the fast one; struck-out words are real corrections\r\n"
        )

        def drain(final):
            events, tail = tf.poll(final=final)
            for _kind, prov, fin in events:
                self.commit_block(prov, fin)
            self.paint(tail)

        carry = b""
        audio, carry = _pcm(leftover, carry)
        parts = [audio] if audio.size else []
        if audio.size:
            tf.push(torch.from_numpy(audio))
        t0 = None
        last = time.time()
        timed_out = False
        self.s.settimeout(1.0)
        while True:
            try:
                buf = self.s.recv(max(2, fast) * SAMPLES_PER_FRAME * 2)
            except socket.timeout:
                if time.time() - last > a_.idle_timeout:
                    timed_out = True
                    break
                continue
            except OSError:
                break
            if not buf:
                break
            last = time.time()
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
            fresh, carry = _pcm(buf, carry)
            if fresh.size:
                parts.append(fresh)
                tf.push(torch.from_numpy(fresh))
            if t0 is None:
                t0 = time.time()
            if time.time() - t0 > a_.max_seconds:
                break
            drain(final=False)

        drain(final=True)
        wave = np.concatenate(parts) if parts else np.zeros(0, dtype="float32")
        ok, diag = speech_report(wave)
        self.emit_final(tf.committed, wave.size / SR, 0, diag, ok, timed_out)
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
        return tf.committed

    def run_nemotron(self, leftover, chunk):
        """Cache-aware RNN-T arm: streams live, one front, never revises.

        Unlike the interleaved baseline this decodes genuinely incrementally --
        conformer_stream_step takes the encoder caches and previous_hypotheses
        and hands them back, so audio is consumed as it arrives with no
        re-decoding. Its emissions are committed, so nothing is struck out.

        NOT yet self-correcting, though the state handles are all here: the
        model carries chunk sizes 1/2/7/14 in one checkpoint and NemotronStream
        exposes snapshot()/restore(), which is what a second front would need.
        """
        import numpy as np
        import torch

        from nemo.collections.speechlm2.parts.nemotron_stream import NemotronStream

        a_ = self.a
        model, chunks = self.nemotron
        if chunk not in chunks:
            self.send(f"\r\n[server] rejected: chunk {chunk} is not in {chunks}\r\n")
            return ""
        dev = next(model.parameters()).device
        st = NemotronStream(model, chunk, dev)
        self.send(
            f"\r\n[server] nemotron RNN-T, chunk {chunk} frames -> {chunk*0.08:.2f}s steps\r\n"
            "[server] one front, committed as decoded, never revised\r\n"
        )

        carry = b""
        audio, carry = _pcm(leftover, carry)
        chunks_sent = [audio] if audio.size else []
        if audio.size:
            st.push(torch.from_numpy(audio))
        shown = ""
        t0 = None
        last = time.time()
        timed_out = False
        self.s.settimeout(1.0)
        while True:
            try:
                buf = self.s.recv(max(2, chunk) * SAMPLES_PER_FRAME * 2)
            except socket.timeout:
                if time.time() - last > a_.idle_timeout:
                    timed_out = True
                    break
                continue
            except OSError:
                break
            if not buf:
                break
            last = time.time()
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
            fresh, carry = _pcm(buf, carry)
            if fresh.size:
                chunks_sent.append(fresh)
                # push() appends to the stream's own raw buffer and re-derives
                # mel frames over all of it, so only the NEW audio goes in.
                st.push(torch.from_numpy(fresh))
            if t0 is None:
                t0 = time.time()
            if time.time() - t0 > a_.max_seconds:
                break
            while st.ready():
                txt = st.step_once(last=False)
                shown = self._nt_emit(txt, shown, final=False)

        if st.remaining() > 0:
            txt = st.step_once(last=True)
            shown = self._nt_emit(txt, shown, final=True)
        else:
            shown = self._nt_emit(shown, shown, final=True)

        wave = np.concatenate(chunks_sent) if chunks_sent else np.zeros(0, dtype="float32")
        ok, diag = speech_report(wave)
        self.emit_final(shown, wave.size / SR, 0, diag, ok, timed_out)
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
        return shown

    def run_script_fixed(self, leftover, chunk, delay):
        """SCRIPT at ONE fixed chunk size: no fast front, no revision.

        The two-front path needs slow to be a different exact multiple of fast,
        so a model trained at a single chunk size (the encoder-free arms, which
        use CHUNK_SIZES=[14]) cannot use it -- it would be rejected outright.
        There is also nothing to revise with: self-correction needs a SECOND,
        slower operating point, and these checkpoints only ever saw one.

        So this emits each block once, committed, as soon as its audio is
        complete. Latency is (chunk + delay) * 0.08s and the text never changes
        after it appears.
        """
        import numpy as np
        import torch

        a_, m = self.a, self.m
        dev = next(m.parameters()).device
        trained = list(getattr(m.core_cfg, "chunk_size", []) or [])
        if trained and chunk not in trained:
            self.send(f"\r\n[server] rejected: chunk {chunk} is not in {trained}\r\n")
            return ""
        delay = max(0, min(a_.max_delay, delay))
        self.send(
            f"\r\n[server] fixed chunk {chunk} frames, delay {delay} "
            f"-> emits every {(chunk + delay) * 0.08:.2f}s (no self-correction)\r\n"
        )

        chunk_bytes = chunk * SAMPLES_PER_FRAME * 2
        carry = b""
        audio, carry = _pcm(leftover, carry)
        committed, blocks = "", 0
        t0 = None
        last = time.time()
        self.s.settimeout(1.0)
        while True:
            try:
                buf = self.s.recv(chunk_bytes)
            except socket.timeout:
                if time.time() - last > a_.idle_timeout:
                    break
                continue
            except OSError:
                break
            if not buf:
                break
            last = time.time()
            # Same drain as the two-front path: one recv per cycle lets lag grow
            # without bound when a cycle costs more than the chunk's duration.
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
            fresh, carry = _pcm(buf, carry)
            audio = np.concatenate([audio, fresh])
            if t0 is None:
                t0 = time.time()
            if time.time() - t0 > a_.max_seconds:
                break

            t = torch.from_numpy(audio).unsqueeze(0).to(dev)
            n = torch.tensor([audio.shape[0]], device=dev)
            with torch.inference_mode():
                ff = m.encode_frames(t, n, chunk)
            avail = int(ff[0].shape[0])
            while avail >= (blocks + 1) * chunk:
                lo = blocks * chunk
                with torch.inference_mode():
                    out = m.generate(
                        t,
                        n,
                        system_prompt=self.p,
                        chunk_size_override=chunk,
                        history_text=[committed],
                        frame_range=(lo, lo + chunk),
                        precomputed_frames=ff,
                        prompt_suffix=self.prompt_suffix,
                        num_delay_frames=delay,
                    )[0].strip()
                # commit_block with an empty provisional: nothing to strike out,
                # so this is a plain append.
                self.commit_block("", out)
                committed = (committed + " " + out).strip() if out else committed
                blocks += 1

        # Flush the tail so the final partial block is not dropped.
        if audio.size:
            t = torch.from_numpy(audio).unsqueeze(0).to(dev)
            n = torch.tensor([audio.shape[0]], device=dev)
            with torch.inference_mode():
                ff = m.encode_frames(t, n, chunk)
                total = int(ff[0].shape[0])
                if total > blocks * chunk:
                    out = m.generate(
                        t,
                        n,
                        system_prompt=self.p,
                        chunk_size_override=chunk,
                        history_text=[committed],
                        frame_range=(blocks * chunk, total),
                        precomputed_frames=ff,
                        prompt_suffix=self.prompt_suffix,
                        num_delay_frames=delay,
                    )[0].strip()
                    self.commit_block("", out)
                    committed = (committed + " " + out).strip() if out else committed
        self.send("\r\n")
        return committed

    def run_interleaved(self, leftover, chunk, delay):
        """Baseline arm: ONE decode that follows the microphone as it speaks.

        The model's decode always was a streaming state machine -- it steps
        chunk by chunk, keeps an encoder cache and commits text at each
        boundary. What stopped it being usable live was that generate() indexed
        a waveform tensor that had to be complete, and returned only the final
        string. Two hooks fix exactly that: audio_source supplies each chunk on
        demand (blocking here until the socket has delivered it) and on_emit
        reports each chunk as the state machine commits it. Nothing about the
        decode itself changes.

        So the decode runs in a worker thread while this one reads the socket,
        and the two meet at _LiveAudio. Measured: 20s of audio fed at real time
        decodes in 20.3s wall with the first words on screen 0.65s behind the
        audio.

        STILL NO SELF-CORRECTION, and deliberately. These emissions are
        committed and never revised -- that is the layout difference the
        comparison is about, and the earlier version that faked a second front
        by re-decoding the prefix cost ~0.55x real time to produce revisions
        that were an artefact of re-decoding.
        """
        import numpy as np
        import torch
        from transformers import GenerationConfig

        a_ = self.a
        model, base_prompt = self.alt
        dev = next(model.parameters()).device

        trained = list(getattr(model.core_cfg, "chunk_size", []) or [])
        if trained and chunk not in trained:
            self.send(f"\r\n[server] rejected: chunk {chunk} is not in {trained}\r\n")
            return ""
        prompt = base_prompt
        if self.prompt_suffix:
            prompt = f"{base_prompt.rstrip()} {self.prompt_suffix}"

        self.send(
            f"\r\n[server] interleaved baseline, chunk {chunk} frames, emit delay {delay} "
            f"-> first words {(chunk+delay)*0.08:.2f}s in\r\n"
            "[server] one front: chunks appear as the decode commits them and are never revised\r\n"
        )

        live = _LiveAudio(dev)
        carry = b""
        first, carry = _pcm(leftover, carry)
        if first.size:
            live.push(first)

        result = {}

        def worker():
            try:
                with torch.inference_mode():
                    res = model.generate(
                        # audios is read only for device and dtype now; the
                        # samples come from audio_source. audio_lens is an upper
                        # bound -- a live stream does not know its own length.
                        audios=torch.zeros(1, 0, device=dev, dtype=torch.float32),
                        audio_lens=torch.tensor([int(a_.max_seconds * SR) + SR], device=dev),
                        audio_source=live.read,
                        on_emit=lambda _b, text, _so_far: text and self.commit_block(text, text),
                        system_prompt=prompt,
                        max_new_tokens=64,
                        generation_config=GenerationConfig(do_sample=False),
                        chunk_size_override=chunk,
                        # Both measured on this checkpoint: the state machine is
                        # the path the model was designed for (6.03 vs 6.63
                        # macro) and offline embeddings drop words at chunk
                        # starts (17.79).
                        use_offline_embs=False,
                        use_state_machine_inference=True,
                        emit_delay_frames=delay,
                        return_alignments=False,
                        return_debug_logs=False,
                    )
                texts = res.texts if hasattr(res, "texts") else res
                result["text"] = (texts[0] or "").strip()
            except Exception as e:  # surfaced below; the reader must still drain
                result["error"] = f"{type(e).__name__}: {e}"

        decoder = threading.Thread(target=worker, daemon=True)
        decoder.start()

        audio = [first] if first.size else []
        t0 = None
        last = time.time()
        timed_out = False
        self.s.settimeout(1.0)
        try:
            while decoder.is_alive():
                try:
                    buf = self.s.recv(max(2, chunk) * SAMPLES_PER_FRAME * 2)
                except socket.timeout:
                    if time.time() - last > a_.idle_timeout:
                        timed_out = True
                        break
                    continue
                except OSError:
                    break
                if not buf:
                    break
                last = time.time()
                # Drain whatever else is waiting, so the decode is never starved
                # by this loop reading one chunk per cycle.
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
                fresh, carry = _pcm(buf, carry)
                if fresh.size:
                    audio.append(fresh)
                    live.push(fresh)
                if t0 is None:
                    t0 = time.time()
                if time.time() - t0 > a_.max_seconds:
                    break
        finally:
            # Unblock the provider on EVERY exit, including an exception: a
            # decode thread parked on a condition variable would otherwise hold
            # the session open until the process died.
            live.close()
        decoder.join(timeout=120)

        if result.get("error"):
            self.send(f"\r\n[server] decode failed: {result['error']}\r\n")
        final = result.get("text", "")
        wave = np.concatenate(audio) if audio else np.zeros(0, dtype="float32")
        ok, diag = speech_report(wave)
        self.emit_final(final, wave.size / SR, 0, diag, ok, timed_out)
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
        return final

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
        # The CLIENT's terminal width. Wrapping wider than the real terminal lets
        # IT soft-wrap first, and a soft wrap is copied without a space -- which
        # is how "streaming fashion" came out as "streamingfashion".
        cols = a_.cols
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
        arm = "script"
        il_chunk, il_delay = a_.il_chunk, a_.il_delay
        nt_chunk, nt_fast = a_.nt_chunk, a_.nt_fast
        head, _, rest = pre.partition(b"\n")
        if head.startswith(b"CFG "):
            for tok in head.decode("ascii", "replace").split()[1:]:
                k, _, v = tok.partition("=")
                if k == "model":
                    arm = v
                    continue
                try:
                    iv = int(v)
                except ValueError:
                    continue
                if k == "nt_chunk":
                    nt_chunk = iv
                    continue
                if k == "nt_fast":
                    nt_fast = iv
                    continue
                if k == "il_chunk":
                    il_chunk = iv
                    continue
                if k == "il_delay":
                    il_delay = iv
                    continue
                if k == "fast_delay":
                    fast_delay = iv
                elif k == "slow_delay":
                    slow_delay = iv
                elif k == "fast":
                    fast = iv
                elif k == "slow":
                    slow = iv
                elif k == "cols":
                    cols = iv
            leftover = rest
        else:
            leftover = pre  # no preamble: every byte was audio
        self.cols = max(40, min(200, cols)) - 4
        if arm == "nemotron":
            if self.nemotron is None:
                self.send("\r\n[server] rejected: this server was started without --nemotron_ckpt\r\n")
                return ""
            if nt_fast and nt_fast != nt_chunk:
                return self.run_nemotron_2f(leftover, nt_fast, nt_chunk)
            return self.run_nemotron(leftover, nt_chunk)

        if arm == "interleaved":
            if self.alt is None:
                self.send("\r\n[server] rejected: this server was started without --interleaved_ckpt\r\n")
                return ""
            return self.run_interleaved(leftover, il_chunk, il_delay)

        # ONE operating point -> single front, no revision. A model trained at a
        # single chunk size (the encoder-free arms) can only ever do this, and
        # the two-front validation below would reject it outright.
        if fast == slow:
            return self.run_script_fixed(leftover, fast, fast_delay)

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
        carry = b""
        audio, carry = _pcm(leftover, carry)
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
            fresh, carry = _pcm(buf, carry)
            audio = np.concatenate([audio, fresh])
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
                        prompt_suffix=self.prompt_suffix,
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
                        prompt_suffix=self.prompt_suffix,
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
                        prompt_suffix=self.prompt_suffix,
                        num_delay_frames=slow_delay,
                    )[0].strip()
                    self.commit_block(last_prov, auth)
                    last_prov = ""
                    committed = (committed + " " + auth).strip() if auth else committed
        if last_prov:
            self.commit_block(last_prov, last_prov)
        ok, diag = speech_report(audio)
        self.emit_final(committed, audio.size / SR, blocks, diag, ok, timed_out)
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


# The interleaved baseline's OWN training instruction. Using the SCRIPT prompt
# here would put it out of distribution -- see scripts/speechlm_asr_shim.py.
INTERLEAVED_PROMPT = "Transcribe the audio into text."

SCRIPT_PROMPT = (
    "You are doing streaming speech recognition. Given the transcript so far and the "
    "representation of the next audio chunk, output the words spoken in that chunk."
)


def load_interleaved(args):
    """Load the baseline arm, or return None when the server was started without it.

    Returns (model, prompt) so the pair travels together: the baseline's prompt
    is not interchangeable with SCRIPT's.
    """
    import torch

    if not args.interleaved_ckpt:
        return None
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from speechlm_leaderboard_eval import load_model as load_interleaved_model

    dev = torch.device(args.interleaved_device or args.device)
    print(f"loading interleaved baseline on {dev} ...", flush=True)
    m = load_interleaved_model(args.interleaved_ckpt, args.interleaved_class, dev, torch.bfloat16)
    m.eval()
    print(f"  interleaved ready (trained chunks {list(m.core_cfg.chunk_size)})", flush=True)
    return m, INTERLEAVED_PROMPT


NEMOTRON_DEFAULT = (
    "/home/hainanx/.cache/huggingface/hub/models--nvidia--nemotron-speech-streaming-en-0.6b/"
    "snapshots/ef3bf40c90df5cd2de55cc07e06681e03d8e6ee4/nemotron-speech-streaming-en-0.6b.nemo"
)


def load_nemotron(args):
    """Load the nemotron RNN-T arm, or None when not requested.

    Trained with att_context_size [[70,13],[70,6],[70,1],[70,0]] -- right
    context is chunk-1, so chunk sizes 1, 2, 7 and 14 all live in one
    checkpoint, the same multi-lookahead property SCRIPT's two fronts rely on.
    """
    import torch

    if not args.nemotron_ckpt:
        return None
    if args.nemotron_ckpt == "default":
        args.nemotron_ckpt = NEMOTRON_DEFAULT
    from nemo.collections.asr.models import ASRModel

    from nemo.collections.speechlm2.parts.nemotron_stream import available_chunks

    dev = torch.device(args.nemotron_device or args.device)
    print(f"loading nemotron RNN-T on {dev} ...", flush=True)
    m = ASRModel.restore_from(args.nemotron_ckpt, map_location=dev)
    m.eval()
    chunks = available_chunks(m)
    print(f"  nemotron ready (trained chunks {chunks})", flush=True)
    return m, chunks


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
    prompt_txt = SCRIPT_PROMPT
    alt = load_interleaved(args)
    nemotron = load_nemotron(args)

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
                text = Session(model, conn, args, prompt_txt, alt, nemotron).run()
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
        f"CFG model={args.arm} fast={args.fast} slow={args.slow} cols={args.cols} "
        f"fast_delay={args.fast_delay} slow_delay={args.slow_delay} "
        f"il_chunk={args.il_chunk} il_delay={args.il_delay} nt_chunk={args.nt_chunk} nt_fast={args.nt_fast}\n".encode()
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
    ap.add_argument("--cols", type=int, default=92, help="fallback terminal width if the client sends none")
    ap.add_argument("--timing", action="store_true", help="log a per-update latency breakdown")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model_class", default="nemo.collections.speechlm2.models.script_model.ScriptSTTModel")
    ap.add_argument(
        "--interleaved_ckpt",
        default="",
        help="optional SECOND arm: the interleaved streaming SpeechLM baseline, held resident "
        "alongside SCRIPT so the two can be compared back to back without a reload",
    )
    ap.add_argument(
        "--interleaved_device",
        default="",
        help="device for the baseline (default: same as --device; a second GPU keeps the two arms "
        "from competing for memory)",
    )
    ap.add_argument(
        "--interleaved_class",
        default="nemo.collections.speechlm2.models.streaming_stt_model.StreamingSTTModel",
    )
    ap.add_argument(
        "--nemotron_ckpt",
        default="",
        help="optional THIRD arm: the nemotron cache-aware RNN-T (.nemo). Pass 'default' for the HF cache copy",
    )
    ap.add_argument("--nemotron_device", default="", help="device for the nemotron arm (default: --device)")
    ap.add_argument("--nt_chunk", type=int, default=14, help="chunk size (frames) for the nemotron arm")
    ap.add_argument(
        "--nt_fast",
        type=int,
        default=0,
        help="fast front (frames) for the nemotron arm; 0 = single front, no self-correction",
    )
    ap.add_argument("--il_chunk", type=int, default=14, help="chunk size (frames) for the interleaved arm")
    ap.add_argument("--il_delay", type=int, default=0, help="emit delay (frames) for the interleaved arm")
    ap.add_argument(
        "--arm",
        default="script",
        choices=["script", "interleaved", "nemotron"],
        help="which arm the client asks for",
    )
    args = ap.parse_args()
    if args.client:
        client(args)
    else:
        if not args.ckpt:
            raise SystemExit("--ckpt is required when running the server")
        serve(args)


if __name__ == "__main__":
    main()
