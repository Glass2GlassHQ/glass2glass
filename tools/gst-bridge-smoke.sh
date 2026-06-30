#!/usr/bin/env bash
# Build the g2g-bridge GStreamer plugin (libgstglass2glass.so) and validate that
# the `glass2glass` element runs an embedded g2g sub-graph inside a real
# gst-launch pipeline. This needs the host's GStreamer (gst-launch-1.0,
# gst-inspect-1.0, dev libs) and so is validated locally, not in CI.
#
# Prerequisites:
#   - gstreamer-1.0 + gstreamer-base-1.0 dev packages (pkg-config finds them).
#   - gst-launch-1.0 / gst-inspect-1.0 on PATH (gstreamer1-tools / -plugins-base).
#
# Usage: tools/gst-bridge-smoke.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== building libgstglass2glass.so =="
cargo build -p g2g-bridge --features gstreamer

# GStreamer derives the plugin name from the `libgst<name>.so` filename, so the
# cargo cdylib (libg2g_bridge.so) is published under the expected name.
plugdir="target/gstplugins"
mkdir -p "$plugdir"
cp -f target/debug/libg2g_bridge.so "$plugdir/libgstglass2glass.so"
export GST_PLUGIN_PATH="$PWD/$plugdir"

echo "== gst-inspect-1.0 glass2glass =="
gst-inspect-1.0 glass2glass >/dev/null || { echo "FAIL: element not registered"; exit 1; }
echo "  registered OK"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
caps="video/x-raw,format=RGBA,width=64,height=64,framerate=1/1"

run() { # fragment outfile
  gst-launch-1.0 videotestsrc num-buffers=1 ! "$caps" \
    ! glass2glass "fragment=$1" ! filesink location="$work/$2" >/dev/null 2>&1
}

echo "== data-flow checks (embedded sub-graph transforms the frame) =="
run "identity" ident.raw
run "videoconvert" cv.raw
run "videoflip method=horizontal-flip" flip.raw
run "videoflip method=horizontal-flip ! videoflip method=horizontal-flip" flip2.raw

fail=0
expect_size=$((64 * 64 * 4))
for f in ident cv flip flip2; do
  sz=$(stat -c%s "$work/$f.raw" 2>/dev/null || echo 0)
  [ "$sz" = "$expect_size" ] || { echo "FAIL: $f.raw is $sz bytes (want $expect_size)"; fail=1; }
done

cmp -s "$work/ident.raw" "$work/cv.raw"   && echo "  PASS videoconvert == identity (RGBA passthrough)" || { echo "FAIL videoconvert changed bytes"; fail=1; }
cmp -s "$work/ident.raw" "$work/flip.raw" && { echo "FAIL flip had no effect"; fail=1; } || echo "  PASS flip != identity (frame transformed)"
cmp -s "$work/ident.raw" "$work/flip2.raw" && echo "  PASS flip!flip == identity (byte-exact reversible)" || { echo "FAIL double-flip != identity"; fail=1; }

echo "== caps/size-changing checks (output-caps property) =="
# Downscale 64x64 -> 32x16 RGBA (2048 bytes).
out_scale="video/x-raw,format=RGBA,width=32,height=16,framerate=1/1"
gst-launch-1.0 videotestsrc num-buffers=1 ! "$caps" \
  ! glass2glass fragment=videoscale output-caps="$out_scale" \
  ! filesink location="$work/scale.raw" >/dev/null 2>&1
sz=$(stat -c%s "$work/scale.raw" 2>/dev/null || echo 0)
[ "$sz" = "$((32 * 16 * 4))" ] && echo "  PASS videoscale 64x64->32x16 ($sz bytes)" || { echo "FAIL downscale is $sz bytes (want 2048)"; fail=1; }

# Format change RGBA -> I420 (planar, 64*64*3/2 = 6144 bytes).
out_fmt="video/x-raw,format=I420,width=64,height=64,framerate=1/1"
gst-launch-1.0 videotestsrc num-buffers=1 ! "$caps" \
  ! glass2glass fragment=videoconvert output-caps="$out_fmt" \
  ! filesink location="$work/i420.raw" >/dev/null 2>&1
sz=$(stat -c%s "$work/i420.raw" 2>/dev/null || echo 0)
[ "$sz" = "$((64 * 64 * 3 / 2))" ] && echo "  PASS videoconvert RGBA->I420 ($sz bytes)" || { echo "FAIL format change is $sz bytes (want 6144)"; fail=1; }

[ "$fail" = 0 ] && echo "== all bridge smoke checks passed ==" || { echo "== bridge smoke FAILED =="; exit 1; }
