#!/usr/bin/env bash
# Build the g2g-bridge GStreamer plugin (libgstglass2glass.so) and validate that
# the `glass2glass` element runs an embedded g2g sub-graph inside a real
# gst-launch pipeline. This needs the host's GStreamer (gst-launch-1.0,
# gst-inspect-1.0, dev libs) and so is validated locally, not in CI.
#
# Prerequisites:
#   - gstreamer-1.0 + gstreamer-base-1.0 dev packages (pkg-config finds them).
#   - gst-launch-1.0 / gst-inspect-1.0 on PATH (gstreamer1-tools / -plugins-base).
#   - a wgpu adapter (a GPU, or a software Vulkan driver such as lavapipe).
#
# Usage: tools/gst-bridge-smoke.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== building libgstglass2glass.so =="
cargo build -p g2g-bridge --features gstreamer,wgpu

# GStreamer derives the plugin name from the `libgst<name>.so` filename, so the
# cargo cdylib (libg2g_bridge.so) is published under the expected name.
plugdir="target/gstplugins"
mkdir -p "$plugdir"
cp -f "${CARGO_TARGET_DIR:-target}/debug/libg2g_bridge.so" "$plugdir/libgstglass2glass.so"
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

echo "== wgpu checks (the frame round-trips through GPU memory) =="
# the output caps default to the input caps, so the compositor matches the input framerate
run "wgpucompositor width=64 height=64 framerate=1/1 gpu-output=true" gpu.raw
run "wgpucompositor width=64 height=64 framerate=1/1 gpu-output=true ! wgpudownload" gpudl.raw
# videotestsrc pixels are opaque, so compositing a single input leaves them unchanged.
cmp -s "$work/ident.raw" "$work/gpu.raw"   && echo "  PASS wgpucompositor == identity (download auto-plugged)" || { echo "FAIL wgpucompositor changed bytes"; fail=1; }
cmp -s "$work/ident.raw" "$work/gpudl.raw" && echo "  PASS wgpucompositor ! wgpudownload == identity" || { echo "FAIL wgpudownload changed bytes"; fail=1; }

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

# compositor labels its output 30/1 by default, so a 1/1 input needs output-caps at that rate.
out_rate="video/x-raw,format=RGBA,width=64,height=64,framerate=30/1"
gst-launch-1.0 videotestsrc num-buffers=1 ! "$caps" \
  ! glass2glass fragment="compositor width=64 height=64" output-caps="$out_rate" \
  ! filesink location="$work/rate.raw" >/dev/null 2>&1
cmp -s "$work/ident.raw" "$work/rate.raw" && echo "  PASS compositor 1/1->30/1 == identity" || { echo "FAIL framerate change lost the frame"; fail=1; }

# Without output-caps those rates cannot negotiate: the element must error out, not hang.
rate_rc=0
timeout 20 gst-launch-1.0 videotestsrc num-buffers=1 ! "$caps" \
  ! glass2glass fragment="compositor width=64 height=64" ! fakesink >/dev/null 2>&1 || rate_rc=$?
case "$rate_rc" in
  0) echo "FAIL mismatched framerate ran without error"; fail=1 ;;
  124) echo "FAIL mismatched framerate hung"; fail=1 ;;
  *) echo "  PASS mismatched framerate fails fast (exit $rate_rc)" ;;
esac

echo "== row layout checks (glass2glass matches GStreamer's own element) =="
# Both outputs are converted losslessly to 4-byte pixels, whose rows GStreamer never
# pads, because GStreamer leaves its own row padding uninitialized.
same_as_gst() { # label input-caps gst-element fragment output-caps(empty: preserving) unpadded-format
  local label=$1 in_caps=$2 element=$3 fragment=$4 out_caps=$5 unpadded_format=$6
  local reference_caps=${out_caps:-$in_caps}
  local output_caps_property=()
  [ -n "$out_caps" ] && output_caps_property=("output-caps=$out_caps")
  rm -f "$work/reference.raw" "$work/bridged.raw"
  gst-launch-1.0 videotestsrc num-buffers=1 ! "$in_caps" ! $element ! "$reference_caps" \
    ! videoconvert ! "video/x-raw,format=$unpadded_format" \
    ! filesink location="$work/reference.raw" >/dev/null 2>&1 || true
  gst-launch-1.0 videotestsrc num-buffers=1 ! "$in_caps" \
    ! glass2glass fragment="$fragment" "${output_caps_property[@]}" \
    ! videoconvert ! "video/x-raw,format=$unpadded_format" \
    ! filesink location="$work/bridged.raw" >/dev/null 2>&1 || true
  local size
  size=$(stat -c%s "$work/reference.raw" 2>/dev/null || echo 0)
  if [ "$size" != 0 ] && cmp -s "$work/reference.raw" "$work/bridged.raw"; then
    echo "  PASS $label ($size bytes)"
  else
    echo "FAIL $label: $(stat -c%s "$work/bridged.raw" 2>/dev/null || echo 0) bytes differ from GStreamer's $size"
    fail=1
  fi
}
raw() { echo "video/x-raw,format=$1,width=$2,height=$3,framerate=1/1"; }
flip="videoflip method=vertical-flip"
# 37 and 38-pixel rows are not a multiple of 4 bytes, so GStreamer pads them and g2g does not.
for size in "37 3" "64 4"; do
  same_as_gst "identity RGB $size" "$(raw RGB $size)" identity identity "" RGBA
  same_as_gst "videoconvert RGB->RGBA $size" "$(raw RGB $size)" videoconvert videoconvert "$(raw RGBA $size)" RGBA
  same_as_gst "videoconvert RGBA->RGB $size" "$(raw RGBA $size)" videoconvert videoconvert "$(raw RGB $size)" RGBA
done
for format in I420 NV12; do
  same_as_gst "identity $format 37 5" "$(raw $format 37 5)" identity identity "" AYUV
  # g2g's videoflip refuses odd 4:2:0 sizes
  for size in "38 6" "64 4"; do
    same_as_gst "vertical flip $format $size" "$(raw $format $size)" "$flip" "$flip" "" AYUV
  done
done

[ "$fail" = 0 ] && echo "== all bridge smoke checks passed ==" || { echo "== bridge smoke FAILED =="; exit 1; }
