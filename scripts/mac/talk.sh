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
DEV=${DEV:-}              # set to e.g. ":1" to skip the device prompt
GAIN=${GAIN:-1.0}         # lower (e.g. 0.3) if the server reports clipping

# --- pick a microphone -------------------------------------------------------
# avfoundation indexes audio and video SEPARATELY and prints both lists in the
# same "[N] Name" format, so the video block must be skipped or ":0" ends up
# meaning a camera. Only the lines after "audio devices:" are microphones.
pick_device() {
    local listing devs n i name
    echo "scanning microphones ..." >&2
    listing=$(ffmpeg -f avfoundation -list_devices true -i "" 2>&1)
    devs=$(printf '%s\n' "$listing" \
           | sed -n '/AVFoundation audio devices:/,$p' \
           | sed -n 's/.*\] \[\([0-9][0-9]*\)\] \(.*\)$/\1|\2/p')
    if [ -z "$devs" ]; then
        echo "no audio devices found. Is ffmpeg installed, and has Terminal been" >&2
        echo "granted Microphone access (System Settings -> Privacy & Security)?" >&2
        exit 1
    fi
    echo >&2
    printf '%s\n' "$devs" | while IFS='|' read -r i name; do
        printf '  [%s] %s\n' "$i" "$name" >&2
    done
    echo >&2
    # Default to the built-in mic, not index 0: on this laptop [0] is a virtual
    # "Horizon Microphone" that captures sub-100 Hz rumble and transcribes to
    # nothing. Fall back to the first device only if no built-in one is listed.
    n=$(printf '%s\n' "$devs" | grep -iE '\|(macbook|built-?in|internal)' | head -1 | cut -d'|' -f1)
    [ -z "$n" ] && n=$(printf '%s\n' "$devs" | head -1 | cut -d'|' -f1)
    printf 'microphone index [%s]: ' "$n" >&2
    read -r choice </dev/tty
    [ -n "$choice" ] && n=$choice
    if ! printf '%s\n' "$devs" | cut -d'|' -f1 | grep -qx "$n"; then
        echo "no such audio device: $n" >&2
        exit 1
    fi
    name=$(printf '%s\n' "$devs" | awk -F'|' -v k="$n" '$1==k{print $2}')
    echo "using [$n] $name" >&2
    printf ':%s' "$n"
}
[ -z "$DEV" ] && DEV=$(pick_device)

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
