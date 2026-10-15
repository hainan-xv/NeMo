#!/bin/bash
# Push-to-talk client for the always-on ASR server. RUNS ON THE LAPTOP.
#
#   ENTER  start speaking
#   ENTER  finish the utterance (transcript is finalised and printed)
#   ...repeats forever...
#   Ctrl-C quit
#
# The loop has to live here rather than on the desktop: ssh's stdin carries the
# audio, so the keyboard cannot reach the remote end. Each utterance is one
# ffmpeg run piped to one ssh session; ControlMaster keeps the TCP/auth
# handshake from being paid every time.
set -u

HOST=${HOST:-hainanx@10.110.41.74}
KEY=${KEY:-$HOME/Documents/draco-rno}
DEV=${DEV:-:1}            # ffmpeg -list_devices: ":1" = MacBook Air Microphone
GAIN=${GAIN:-1.0}         # lower (e.g. 0.3) if the server reports clipping

SSH_OPTS=(-i "$KEY" -o ControlMaster=auto -o ControlPath=/tmp/asrctl-%r@%h:%p -o ControlPersist=300)
FIFO=$(mktemp -u /tmp/asrmic.XXXXXX)
mkfifo "$FIFO"
trap 'rm -f "$FIFO"; echo; echo "bye."; exit 0' EXIT INT

echo "push-to-talk  ->  $HOST   device $DEV"
echo "ENTER starts, ENTER finishes, Ctrl-C quits."
n=0
while true; do
    read -r -p $'\n[ENTER to speak] ' _ || exit 0
    n=$((n + 1))
    printf '\033[2m(recording -- ENTER to finish)\033[0m\n'

    # -nostdin: without it ffmpeg eats the ENTER we are waiting on.
    ffmpeg -nostdin -f avfoundation -i "$DEV" -ar 16000 -ac 1 \
           -af "volume=${GAIN}" -f s16le -loglevel error - > "$FIFO" &
    ff=$!
    ssh "${SSH_OPTS[@]}" "$HOST" ./asr < "$FIFO" &
    ss=$!

    read -r _ || true                 # second ENTER ends the utterance
    kill "$ff" 2>/dev/null            # closing the writer gives ssh EOF
    wait "$ss" 2>/dev/null            # let the server finalise and print
done
