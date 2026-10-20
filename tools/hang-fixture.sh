#!/usr/bin/env bash
# Record the hang fixture (g2g-plugins/tests/fixtures/hang/) off a local moq-relay.
# MOQ_BIN is a release build directory of github.com/moq-dev/moq holding moq-relay and moq.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
FIX="$HERE/g2g-plugins/tests/fixtures/hang"
BIN="${MOQ_BIN:?set MOQ_BIN to the directory holding moq-relay and moq}"
PORT="${HANG_FIXTURE_PORT:-28743}"
URL="http://127.0.0.1:$PORT/"
BROADCAST=demo/fixture.hang
WORK="$(mktemp -d)"
trap 'kill $(jobs -p) 2>/dev/null; rm -rf "$WORK"' EXIT

cat > "$WORK/relay.toml" <<TOML
[listen]
bind = "127.0.0.1:$PORT"
tls.generate = ["localhost", "127.0.0.1"]
[web.http]
listen = "127.0.0.1:$PORT"
[auth]
public = "**"
TOML
"$BIN/moq-relay" "$WORK/relay.toml" > "$WORK/relay.log" 2>&1 &
sleep 2

ffmpeg -nostdin -hide_banner -loglevel error -re -f lavfi -i testsrc2=size=160x90:rate=24 \
    -c:v libx264 -preset veryfast -bf 0 -g 12 -pix_fmt yuv420p -an -f mpegts - |
    "$BIN/moq" --connect "$URL" --broadcast "$BROADCAST" import ts > "$WORK/publish.log" 2>&1 &
sleep 6

fetch() {
    "$BIN/moq" --connect "$URL" --broadcast "$BROADCAST" fetch "$@" 2>> "$WORK/fetch.log"
}
fetch catalog.json > "$FIX/catalog.json"
fetch --json 0.avc3 > "$WORK/newest.jsonl"
newest=$(sed -nE '1s/.*"group":([0-9]+).*/\1/p' "$WORK/newest.jsonl")
# the two groups before the newest are complete
for group in $((newest - 2)) $((newest - 1)); do
    fetch --json --group "$group" 0.avc3
done > "$FIX/0.avc3.jsonl"
echo "recorded groups $((newest - 2))-$((newest - 1)) into $FIX"
