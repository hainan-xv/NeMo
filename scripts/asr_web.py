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
"""Browser front-end for the self-correcting ASR server.

A page with a Record button, served together with its WebSocket from ONE port,
so reaching it needs a single ssh tunnel:

    desktop:  python scripts/asr_web.py --ckpt <ckpt>
    laptop :  ssh -L 9880:localhost:9880 <host>
              open http://localhost:9880

Why a browser instead of the terminal. Crossing out text in a terminal means
erasing it, and erasing means backspaces (which cannot cross a line wrap) or
absolute cursor saves (which go stale the moment the screen scrolls). Both
failed in practice. In a DOM, a correction is a CSS class. The browser also
captures the microphone directly -- with gain control and echo cancellation --
which removes ffmpeg, device-index picking, and the clipped rumble that made
an early capture transcribe to nothing.

The decode itself is NOT reimplemented here: WebSession subclasses the terminal
server's Session and replaces only the display, so resync, deferral handling
and the tail flush cannot drift between the two front-ends.
"""

import argparse
import asyncio
import json
import os
import queue
import re
import socket
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.asr_server import SR, Session, speech_report  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asr_web.html")


class QueueSocket:
    """Socket-shaped adapter over two queues, so Session can drive a WebSocket.

    Session speaks recv/sendall/settimeout/setblocking. The web handler speaks
    asyncio queues. This is the seam between them, and it exists so the decode
    loop stays byte-identical to the terminal path.
    """

    def __init__(self, audio_q, out_q):
        self.audio_q, self.out_q = audio_q, out_q
        self.timeout = 1.0
        self.blocking = True
        self._buf = b""
        self._eof = False  # sticky: Session drains with recv(), and a one-shot
        # sentinel would be consumed there and the session would then sit until
        # the idle timeout instead of stopping when the user pressed Stop.

    def settimeout(self, t):
        self.timeout = t

    def setblocking(self, flag):
        self.blocking = flag

    def recv(self, n):
        if self._eof and not self._buf:
            return b""
        if self._buf:
            out, self._buf = self._buf[:n], self._buf[n:]
            return out
        try:
            item = self.audio_q.get(block=self.blocking, timeout=self.timeout if self.blocking else None)
        except queue.Empty:
            if self.blocking:
                raise socket.timeout
            raise BlockingIOError
        if item is None:
            self._eof = True
            return b""  # client stopped: end of utterance
        out, self._buf = item[:n], item[n:]
        return out

    def sendall(self, data):
        self.out_q.put(data)

    def shutdown(self, how):
        pass

    def close(self):
        pass


class WebSession(Session):
    """Same decode, structured events instead of escape codes."""

    def __init__(self, model, sock, args, prompt_txt, emit):
        super().__init__(model, sock, args, prompt_txt)
        self.emit = emit

    def send(self, text):
        # Banner / final / diagnostics arrive here as preformatted lines, styled
        # for a terminal. Strip the escapes: in the page, styling is CSS.
        text = ANSI.sub("", text)
        for line in text.replace("\r", "").split("\n"):
            s = line.strip()
            if s and set(s) != {"-"}:
                self.emit({"t": "log", "text": s})

    def emit_final(self, text, audio_s, blocks, diag, ok, timed_out):
        self.emit(
            {
                "t": "final",
                "text": text,
                "audio_s": round(audio_s, 1),
                "blocks": blocks,
                "diag": diag.strip(),
                "ok": bool(ok),
                "timed_out": bool(timed_out),
            }
        )

    def paint(self, provisional):
        self.emit({"t": "prov", "text": provisional})

    def commit_block(self, provisional, final):
        from nemo.collections.speechlm2.parts.multi_front import Block, _deferred_tail, strike_markup

        # Same rule as the terminal path: a trailing deletion is a deferral, not
        # a rejection, so it must not be shown struck through.
        deferred = _deferred_tail(Block(provisional, final))
        if deferred:
            kept = provisional.split()[: len(provisional.split()) - len(deferred)]
            provisional = " ".join(kept)
        segs = [[t, bool(k)] for t, k in strike_markup(provisional, final)]
        self.emit({"t": "commit", "segs": segs})


async def ws_handler(request):
    from aiohttp import WSMsgType, web

    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    app = request.app
    audio_q, out_q = queue.Queue(), queue.Queue()
    loop = asyncio.get_running_loop()
    done = threading.Event()

    cfg = {}
    first = await ws.receive()
    if first.type == WSMsgType.TEXT:
        cfg = json.loads(first.data)

    class A:  # per-session argument view, so one client cannot affect another
        pass

    a = A()
    a.__dict__.update(vars(app["args"]))
    for k in ("fast", "slow", "fast_delay", "slow_delay"):
        if k in cfg:
            a.__dict__[k] = int(cfg[k])

    # Free-text biasing appended to the base instruction, BEFORE the control
    # clauses that render_control_prompt adds -- so the trained template still
    # ends the prompt the way the model saw it in training.
    prompt = app["prompt"]
    extra = (cfg.get("prompt_extra") or "").strip()

    def emit(ev):
        loop.call_soon_threadsafe(out_q.put, ev)

    def run():
        try:
            sess = WebSession(app["model"], QueueSocket(audio_q, out_q), a, prompt, emit)
            sess.prompt_suffix = extra or None
            sess.run()
        except Exception as e:  # a bad session must not take the server down
            emit({"t": "log", "text": f"session error: {type(e).__name__}: {e}"})
        finally:
            emit({"t": "done"})
            done.set()

    if extra:
        emit({"t": "log", "text": f"[prompt] + {extra}"})
    worker = threading.Thread(target=run, daemon=True)
    worker.start()

    async def pump():
        while True:
            try:
                ev = out_q.get_nowait()
            except queue.Empty:
                if done.is_set():
                    break
                await asyncio.sleep(0.02)
                continue
            if isinstance(ev, (bytes, bytearray)):
                continue  # Session.sendall path is unused in web mode
            await ws.send_json(ev)
            if ev.get("t") == "done":
                break

    pumper = asyncio.create_task(pump())
    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                audio_q.put(bytes(msg.data))
            elif msg.type == WSMsgType.TEXT and json.loads(msg.data).get("t") == "stop":
                audio_q.put(None)
            elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                break
    finally:
        audio_q.put(None)
        await pumper
        await ws.close()
    return ws


async def index(request):
    from aiohttp import web

    return web.FileResponse(PAGE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=9880)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--fast", type=int, default=2)
    ap.add_argument("--slow", type=int, default=28)
    ap.add_argument("--fast_delay", type=int, default=1)
    ap.add_argument("--slow_delay", type=int, default=3)
    ap.add_argument("--max_delay", type=int, default=4)
    ap.add_argument("--max_seconds", type=float, default=600.0)
    ap.add_argument("--idle_timeout", type=float, default=8.0)
    ap.add_argument("--strike", default="unicode")
    ap.add_argument("--cols", type=int, default=100)
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model_class", default="nemo.collections.speechlm2.models.script_model.ScriptSTTModel")
    args = ap.parse_args()

    import torch
    from aiohttp import web

    from scripts.script_leaderboard_eval import load_model

    print("loading model ...", flush=True)
    model = load_model(args.ckpt, args.model_class, torch.device(args.device), torch.bfloat16, prompt_control=True)
    model.eval()
    prompt = (
        "You are doing streaming speech recognition. Given the transcript so far and the "
        "representation of the next audio chunk, output the words spoken in that chunk."
    )
    app = web.Application(client_max_size=1 << 22)
    app["model"], app["args"], app["prompt"] = model, args, prompt
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    print(f"ready on http://{args.host}:{args.port}  (trained chunks {list(model.core_cfg.chunk_size)})", flush=True)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
