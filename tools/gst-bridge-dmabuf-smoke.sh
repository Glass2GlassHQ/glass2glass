#!/usr/bin/env bash
# Validate the bridge's zero-copy dma-buf round-trip: a dma-buf GstBuffer in ->
# glass2glass (fragment=identity) -> dma-buf GstBuffer out, proving both wiring
# steps (input auto-detect + import via g2g_bridge_push_dmabuf; output wrap-back
# via gst_dmabuf_allocator_alloc). Uses a memfd wrapped as dma-buf memory, so no
# special hardware / dma-buf producer element is needed. Host GStreamer only, so
# validated locally, not in CI.
#
# Prerequisites: gstreamer-1.0, gstreamer-app-1.0, gstreamer-allocators-1.0 dev
# packages, and a C compiler.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== building libgstglass2glass.so =="
cargo build -p g2g-bridge --features gstreamer

plugdir="target/gstplugins"
mkdir -p "$plugdir"
cp -f target/debug/libg2g_bridge.so "$plugdir/libgstglass2glass.so"
export GST_PLUGIN_PATH="$PWD/$plugdir"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

cat > "$work/dmabuf_roundtrip.c" <<'EOF'
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <gst/allocators/gstdmabuf.h>
#include <sys/mman.h>
#include <unistd.h>

int main(void) {
  gst_init(NULL, NULL);
  const gsize size = 256 * 16; /* RGBA 64x16 */

  int fd = memfd_create("g2g-dmabuf-test", 0);
  if (fd < 0 || ftruncate(fd, size) != 0) { g_printerr("memfd setup failed\n"); return 2; }

  GstElement *pipe = gst_parse_launch(
      "appsrc name=src is-live=false format=time "
      "caps=video/x-raw,format=RGBA,width=64,height=16,framerate=1/1 ! "
      "glass2glass fragment=identity ! appsink name=sink", NULL);
  if (!pipe) { g_printerr("pipeline build failed\n"); return 2; }
  GstElement *src = gst_bin_get_by_name(GST_BIN(pipe), "src");
  GstElement *sink = gst_bin_get_by_name(GST_BIN(pipe), "sink");

  /* Wrap the fd as dma-buf memory (allocator takes ownership of the fd). */
  GstAllocator *alloc = gst_dmabuf_allocator_new();
  GstMemory *mem = gst_dmabuf_allocator_alloc(alloc, fd, size);
  GstBuffer *buf = gst_buffer_new();
  gst_buffer_append_memory(buf, mem);
  GST_BUFFER_PTS(buf) = 0;
  if (!gst_is_dmabuf_memory(gst_buffer_peek_memory(buf, 0))) {
    g_printerr("input buffer is not dma-buf\n"); return 2;
  }

  gst_element_set_state(pipe, GST_STATE_PLAYING);
  if (gst_app_src_push_buffer(GST_APP_SRC(src), buf) != GST_FLOW_OK) {
    g_printerr("push failed\n"); return 1;
  }
  gst_app_src_end_of_stream(GST_APP_SRC(src));

  int rc = 1;
  GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
  if (sample) {
    GstMemory *om = gst_buffer_peek_memory(gst_sample_get_buffer(sample), 0);
    if (om && gst_is_dmabuf_memory(om)) {
      g_print("  PASS dma-buf in -> glass2glass(identity) -> dma-buf out (out fd=%d)\n",
              gst_dmabuf_memory_get_fd(om));
      rc = 0;
    } else {
      g_printerr("  FAIL output is not dma-buf memory\n");
    }
    gst_sample_unref(sample);
  } else {
    g_printerr("  FAIL no output sample\n");
  }

  gst_element_set_state(pipe, GST_STATE_NULL);
  gst_object_unref(alloc);
  gst_object_unref(src);
  gst_object_unref(sink);
  gst_object_unref(pipe);
  return rc;
}
EOF

echo "== compiling dma-buf round-trip harness =="
cc "$work/dmabuf_roundtrip.c" -o "$work/dmabuf_roundtrip" -D_GNU_SOURCE \
  $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-allocators-1.0)

echo "== running dma-buf round-trip =="
"$work/dmabuf_roundtrip"
echo "== bridge dma-buf round-trip passed =="
