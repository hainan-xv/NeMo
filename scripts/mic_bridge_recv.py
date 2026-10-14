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
"""Prove the laptop-microphone-over-ssh link, with no model involved.

ONE job: receive raw PCM and say whether it is real, continuous audio. Keeping
this separate from transcription means a failure has one obvious cause -- if the
meter moves when you speak, the transport works and anything that goes wrong
later belongs to the model side.

Desktop (this machine):

    ssh-in and run:  python scripts/mic_bridge_recv.py --out /tmp/cap.wav
    ...or have the laptop invoke it directly (see below).

MacBook:

    brew install ffmpeg
    ffmpeg -f avfoundation -list_devices true -i ""      # find the mic index
    ffmpeg -f avfoundation -i ":0" -ar 16000 -ac 1 -f s16le -loglevel error - \\
      | ssh <desktop> 'python /path/to/scripts/mic_bridge_recv.py --out /tmp/cap.wav'

Speak. Each line shows level and whether audio is keeping up with real time:

    [  3.2s]  rms -28.4 dBFS  peak -12.1  |#########           |  1.00x realtime

What the numbers mean:
  * rms around -60 dBFS or below and a flat bar = SILENCE. The stream is
    connected but the mic is not: wrong device index, or macOS has not granted
    Terminal microphone permission.
  * realtime much below 1.00x = the link is not delivering audio fast enough;
    expect gaps. Over a slow VPN this is the number that will show it.
  * no lines at all = nothing is arriving; the ssh/ffmpeg pipeline is wrong.
"""

import argparse
import math
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=int, default=16000)
    ap.add_argument("--out", default="", help="also write what arrives to this wav")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0 = until EOF)")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between meter lines")
    args = ap.parse_args()

    import numpy as np

    block = int(args.rate * args.interval)
    nbytes = block * 2
    src = sys.stdin.buffer

    wav = None
    if args.out:
        import soundfile as sf

        wav = sf.SoundFile(args.out, mode="w", samplerate=args.rate, channels=1, subtype="PCM_16")

    print(f"listening for raw s16le {args.rate} Hz mono on stdin -- speak into the laptop mic", flush=True)
    t0 = time.time()
    total = 0
    loud_blocks = 0
    blocks = 0
    try:
        while True:
            buf = src.read(nbytes)
            if not buf:
                break
            a = np.frombuffer(buf, dtype="<i2").astype("float32") / 32768.0
            total += a.size
            blocks += 1
            rms = float(np.sqrt((a**2).mean())) if a.size else 0.0
            peak = float(np.abs(a).max()) if a.size else 0.0
            db = 20 * math.log10(rms) if rms > 1e-9 else -99.0
            dbp = 20 * math.log10(peak) if peak > 1e-9 else -99.0
            if db > -50:
                loud_blocks += 1
            bars = int(max(0, min(20, (db + 60) / 3)))
            el = time.time() - t0
            rt = (total / args.rate) / el if el > 0 else 0.0
            if wav:
                wav.write(a)
            print(
                f"[{total/args.rate:6.1f}s]  rms {db:6.1f} dBFS  peak {dbp:6.1f}  "
                f"|{'#'*bars}{' '*(20-bars)}|  {rt:4.2f}x realtime",
                flush=True,
            )
            if args.seconds and total / args.rate >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if wav:
            wav.close()

    el = time.time() - t0
    secs = total / args.rate
    print(f"\n{'='*64}")
    print(f"received {secs:.1f}s of audio in {el:.1f}s wall  ({secs/el if el else 0:.2f}x realtime)")
    if wav:
        print(f"saved {args.out}  -- play it back to confirm it is your voice")
    if blocks == 0:
        print("VERDICT: nothing arrived. The ffmpeg | ssh pipeline is not delivering bytes.")
    elif loud_blocks == 0:
        print("VERDICT: connected, but SILENT. Wrong --i device index, or macOS has not")
        print("         granted Terminal microphone permission (System Settings ->")
        print("         Privacy & Security -> Microphone).")
    elif el and secs / el < 0.9:
        print(f"VERDICT: audio arrives but only at {secs/el:.2f}x realtime -- the link is")
        print("         too slow and live transcription will have gaps.")
    else:
        print(f"VERDICT: OK. Real audio, in real time ({loud_blocks}/{blocks} blocks above -50 dBFS).")
        print("         The transport works; wire it into multi_front_live.py --pcm next.")


if __name__ == "__main__":
    main()
