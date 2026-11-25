#!/usr/bin/env bash
# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

# Pair the Moonlight flatpak with `g2g-desktop host --protocol gamestream`, list its apps and
# stream the desktop for a few seconds. Moonlight runs on its own Xvfb display, which the host
# also captures, or with CAPTURE=run the host captures its own headless display, so nothing
# opens on the desktop. CAPTURE=portal captures the desktop session through the ScreenCast
# portal, whose share dialog a user approves on the desktop.
#
# Prerequisites: Xvfb, glxgears for CAPTURE=run, and Moonlight from Flathub:
#   flatpak install --user flathub com.moonlight_stream.Moonlight
#
# Usage: tools/moonlight-interop.sh <g2g-desktop binary built with --features gamestream>
# Logs go to $OUT, default ~/tmp/moonlight-interop. CODEC picks H.264 (default) or HEVC.
set -euo pipefail

BIN=$1
OUT=${OUT:-$HOME/tmp/moonlight-interop}
PIN=4721
CODEC=${CODEC:-H.264}
STREAM_SECONDS=${STREAM_SECONDS:-8}
# each run starts unpaired: fresh config directories for Moonlight and the host
rm -rf "$OUT"
mkdir -p "$OUT/moonlight-config" "$OUT/host-config"
# flatpak overrides XDG_CONFIG_HOME from outside, so a shell inside the sandbox sets it
# killing `flatpak run` leaves the sandboxed moonlight running, so it is stopped by its own command line
stop_moonlight() {
    pkill -f "^moonlight $1 127\.0\.0\.1" || true
}
moonlight() {
    flatpak run --nosocket=wayland --socket=x11 --env=QT_QPA_PLATFORM=xcb \
        --filesystem="$OUT" --command=sh com.moonlight_stream.Moonlight \
        -c 'XDG_CONFIG_HOME="$0" exec moonlight "$@"' "$OUT/moonlight-config" "$@"
}

exec 4< <(Xvfb -displayfd 3 -screen 0 1920x1080x24 -nolisten tcp 3>&1 >/dev/null 2>&1 & echo "$!")
read -r XVFB <&4
read -r DISPLAY_NUMBER <&4
export DISPLAY=":$DISPLAY_NUMBER"
unset WAYLAND_DISPLAY

case ${CAPTURE:-x11} in
x11) capture=(--x11-display "$DISPLAY") ;;
run)
    for node in /dev/dri/renderD*; do
        [ "$(basename "$(readlink -f "/sys/class/drm/${node##*/}/device/driver")")" = nvidia ] && break
    done
    capture=(--run glxgears --render-node "$node")
    ;;
portal) capture=() ;;
esac
XDG_CONFIG_HOME="$OUT/host-config" "$BIN" host --protocol gamestream "${capture[@]}" \
    >"$OUT/host.log" 2>&1 &
HOST=$!
trap 'stop_moonlight "(pair|quit|stream)"; kill $HOST $XVFB 2>/dev/null || true' EXIT
for _ in $(seq 240); do
    grep -q 'Moonlight host' "$OUT/host.log" && break
    sleep 0.5
done

moonlight pair 127.0.0.1 --pin "$PIN" >"$OUT/pair.log" 2>&1 &
PAIR=$!
for _ in $(seq 120); do
    grep -q 'asks to pair' "$OUT/host.log" && break
    sleep 0.25
done
grep -q 'asks to pair' "$OUT/host.log"
"$BIN" pin "$PIN"
# moonlight's pair command waits on a "Pairing completed" dialog, so it is closed once the host has paired it
for _ in $(seq 120); do
    grep -q 'phrase=pairchallenge' "$OUT/pair.log" && break
    sleep 0.25
done
sleep 1
stop_moonlight pair
wait "$PAIR" || true
grep -q 'paired a Moonlight client' "$OUT/host.log"

moonlight list 127.0.0.1 >"$OUT/list.log" 2>&1
grep -qx 'Desktop' "$OUT/list.log"
echo "moonlight-interop: paired and listed the Desktop app"

moonlight stream 127.0.0.1 Desktop --720 --fps 30 --video-codec "$CODEC" \
    --video-decoder software --display-mode windowed --no-quit-after >"$OUT/stream.log" 2>&1 &
STREAM=$!
# a fresh moonlight config spends about 40 seconds checking for gamepad mappings before it launches
for _ in $(seq 240); do
    grep -q 'Launch response' "$OUT/stream.log" && break
    sleep 0.5
done
grep -q 'Launch response' "$OUT/stream.log"
sleep "$STREAM_SECONDS"
# moonlight writes its streaming log only once the session ends, so the host ends it
moonlight quit 127.0.0.1 >"$OUT/quit.log" 2>&1 &
QUIT=$!
# moonlight's quit command, like pair, waits on a dialog after the host answers
for _ in $(seq 40); do
    grep -q 'Quit response' "$OUT/quit.log" && break
    sleep 0.25
done
stop_moonlight quit
wait "$QUIT" || true
for _ in $(seq 40); do
    kill -0 "$STREAM" 2>/dev/null || break
    sleep 0.25
done
stop_moonlight stream
wait "$STREAM" || true
grep -q 'Received first video packet' "$OUT/stream.log"
decoded=$(sed -n 's/^Decoding frame rate: \([0-9.]*\) FPS$/\1/p' "$OUT/stream.log")
if [ -z "$decoded" ] || grep -q 'avcodec_send_packet() failed' "$OUT/stream.log"; then
    echo "moonlight-interop: Moonlight did not decode the stream, see $OUT/stream.log" >&2
    exit 1
fi
echo "moonlight-interop: Moonlight decoded $CODEC at $decoded fps"
