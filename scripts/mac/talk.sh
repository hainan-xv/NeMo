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

# --- pick the two fronts -----------------------------------------------------
# This checkpoint trained chunk sizes 2, 7, 14 and 28. slow must be a different,
# exact multiple of fast, or a block would not contain a whole number of fast
# chunks and the provisional/authoritative spans would not line up.
pick_fronts() {
    local d
    echo >&2
    echo "front chunk sizes, in 80ms frames. Trained: 2, 7, 14, 28, 42." >&2
    printf '  fast front (provisional) [%s]: ' "${FAST_DEFAULT:-2}" >&2
    read -r d </dev/tty; FAST=${d:-${FAST_DEFAULT:-2}}
    printf '  slow front (authoritative) [%s]: ' "${SLOW_DEFAULT:-14}" >&2
    read -r d </dev/tty; SLOW=${d:-${SLOW_DEFAULT:-14}}
    for d in "$FAST" "$SLOW"; do
        case "$d" in
            2|7|14|28|42) ;;
            *) echo "chunk size $d was never trained (use 2, 7, 14, 28 or 42)" >&2; exit 1 ;;
        esac
    done
    [ "$FAST" -eq "$SLOW" ] && { echo "the two fronts must differ" >&2; exit 1; }
    [ $((SLOW % FAST)) -ne 0 ] && { echo "slow ($SLOW) must be an exact multiple of fast ($FAST)" >&2; exit 1; }
    echo "  -> $((SLOW / FAST)) fast chunks per block" >&2
}
[ -z "${FAST:-}" ] && pick_fronts

# --- pick emission delays ----------------------------------------------------
# Delay is the number of frames the model waits past a word before committing to
# it, and it ADDS to the chunk latency: first paint is (fast + fast_delay) * 80ms.
# The fast front is provisional, so a small delay buys responsiveness cheaply.
# NOT zero though: on this checkpoint chunk 2 with delay 0 degenerates and drops
# most of the utterance, while delay 1 decodes it correctly. Trained range 0-4.
pick_delays() {
    local d
    echo >&2
    echo "emission delay, in 80ms frames (0-4). Higher = more accurate, slower." >&2
    printf '  fast front (chunk %s) delay [%s]: ' "${FAST:-2}" "${FAST_DELAY_DEFAULT:-1}" >&2
    read -r d </dev/tty; FAST_DELAY=${d:-${FAST_DELAY_DEFAULT:-1}}
    printf '  slow front (chunk %s) delay [%s]: ' "${SLOW:-14}" "${SLOW_DELAY_DEFAULT:-3}" >&2
    read -r d </dev/tty; SLOW_DELAY=${d:-${SLOW_DELAY_DEFAULT:-3}}
    for d in "$FAST_DELAY" "$SLOW_DELAY"; do
        case "$d" in
            ''|*[!0-9]*) echo "delay must be a number 0-6, got '$d'" >&2; exit 1 ;;
        esac
        [ "$d" -gt 4 ] && { echo "delay $d was never trained (max 4)" >&2; exit 1; }
    done
    awk -v f="${FAST:-2}" -v fd="$FAST_DELAY" -v s="${SLOW:-14}" -v sd="$SLOW_DELAY" \
        'BEGIN{printf "  -> first paint %.2fs, settles every %.2fs\n",(f+fd)*0.08,(s+sd)*0.08}' >&2
}
[ -z "${FAST_DELAY:-}" ] && pick_delays

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
    # We KILL ffmpeg to end the utterance, so its dying "Broken pipe" complaints
    # are expected noise. Filter exactly those, and nothing else, so a genuine
    # capture failure (wrong device, permission denied) still reaches you.
    ffmpeg -nostdin -f avfoundation -i "$DEV" -ar 16000 -ac 1 \
           -af "volume=${GAIN}" -f s16le -loglevel error - > "$FIFO" \
        2> >(grep -vE 'Broken pipe|Error muxing|Error submitting|Terminating thread|Error writing trailer|Error closing file|Last message repeated' >&2) &
    ff=$!
    # Pass the REAL terminal width so the server wraps to it. Wrapping wider
    # lets the terminal soft-wrap first, and soft wraps copy without a space.
    ssh "${SSH_OPTS[@]}" "$HOST" ./asr --fast "$FAST" --slow "$SLOW" \
        --cols "$(tput cols 2>/dev/null || echo 80)" \
        --fast_delay "$FAST_DELAY" --slow_delay "$SLOW_DELAY" < "$FIFO" &
    ss=$!

    read -r _ || true                 # second ENTER ends the utterance
    kill "$ff" 2>/dev/null            # closing the writer gives ssh EOF
    wait "$ss" 2>/dev/null            # let the server finalise and print
done
