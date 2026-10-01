#!/usr/bin/env bash
# Validate the bridge's zero-copy dma-buf round-trip: a dma-buf GstBuffer in ->
# glass2glass (fragment=identity) -> dma-buf GstBuffer out, for packed RGBA and
# for NV12 at the default and at a padded stride, proving both wiring
# steps (input auto-detect + import via g2g_bridge_push_dmabuf; output wrap-back
# via gst_dmabuf_allocator_alloc). Uses a memfd wrapped as dma-buf memory, so no
# special hardware / dma-buf producer element is needed. Host GStreamer only, so
# validated locally, not in CI.
#
# Prerequisites: gstreamer-1.0, gstreamer-app-1.0, gstreamer-allocators-1.0,
# gstreamer-video-1.0 dev packages, and a C compiler.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== building libgstglass2glass.so =="
cargo build -p g2g-bridge --features gstreamer

plugdir="target/gstplugins"
mkdir -p "$plugdir"
cp -f "${CARGO_TARGET_DIR:-target}/debug/libg2g_bridge.so" "$plugdir/libgstglass2glass.so"
export GST_PLUGIN_PATH="$PWD/$plugdir"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

cat > "$work/dmabuf_roundtrip.c" <<'EOF'
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <gst/allocators/gstdmabuf.h>
#include <gst/video/video.h>
#include <dirent.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define MEMFD_NAME "g2g-dmabuf-test"

/* Open descriptors onto the test memfd, the input's and every dup of it. */
static int memfd_count(void) {
  DIR *dir = opendir("/proc/self/fd");
  if (!dir) return -1;
  int count = 0;
  struct dirent *entry;
  char path[64], target[256];
  while ((entry = readdir(dir))) {
    g_snprintf(path, sizeof path, "/proc/self/fd/%s", entry->d_name);
    ssize_t n = readlink(path, target, sizeof target - 1);
    if (n <= 0) continue;
    target[n] = '\0';
    if (strstr(target, MEMFD_NAME)) count++;
  }
  closedir(dir);
  return count;
}

/* A padded NV12 frame must come back with the stride and chroma offset it went in with. */
static int roundtrip(const char *format, int width, int height, int stride, gsize size,
                     gboolean padded_nv12) {
  int fd = memfd_create(MEMFD_NAME, 0);
  if (fd < 0 || ftruncate(fd, size) != 0) { g_printerr("memfd setup failed\n"); return 2; }

  gchar *launch = g_strdup_printf(
      "appsrc name=src is-live=false format=time "
      "caps=video/x-raw,format=%s,width=%d,height=%d,framerate=1/1 ! "
      "glass2glass fragment=identity ! appsink name=sink", format, width, height);
  GstElement *pipe = gst_parse_launch(launch, NULL);
  g_free(launch);
  if (!pipe) { g_printerr("pipeline build failed\n"); return 2; }
  GstElement *src = gst_bin_get_by_name(GST_BIN(pipe), "src");
  GstElement *sink = gst_bin_get_by_name(GST_BIN(pipe), "sink");

  /* Wrap the fd as dma-buf memory (allocator takes ownership of the fd). */
  GstAllocator *alloc = gst_dmabuf_allocator_new();
  GstMemory *mem = gst_dmabuf_allocator_alloc(alloc, fd, size);
  GstBuffer *buf = gst_buffer_new();
  gst_buffer_append_memory(buf, mem);
  GST_BUFFER_PTS(buf) = 0;
  if (padded_nv12) {
    gsize offsets[GST_VIDEO_MAX_PLANES] = {0, (gsize)stride * height};
    gint strides[GST_VIDEO_MAX_PLANES] = {stride, stride};
    gst_buffer_add_video_meta_full(buf, GST_VIDEO_FRAME_FLAG_NONE, GST_VIDEO_FORMAT_NV12, width,
                                   height, 2, offsets, strides);
  }
  if (!gst_is_dmabuf_memory(gst_buffer_peek_memory(buf, 0))) {
    g_printerr("input buffer is not dma-buf\n"); return 2;
  }

  gst_element_set_state(pipe, GST_STATE_PLAYING);
  if (gst_app_src_push_buffer(GST_APP_SRC(src), buf) != GST_FLOW_OK) {
    g_printerr("push failed\n"); return 1;
  }
  gst_app_src_end_of_stream(GST_APP_SRC(src));

  int rc = 1;
  GstBuffer *kept = NULL;
  GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
  if (sample) {
    GstBuffer *out = gst_sample_get_buffer(sample);
    GstMemory *om = gst_buffer_peek_memory(out, 0);
    GstVideoMeta *meta = gst_buffer_get_video_meta(out);
    gsize out_size = gst_buffer_get_size(out);
    if (!om || !gst_is_dmabuf_memory(om)) {
      g_printerr("  FAIL %s: output is not dma-buf memory\n", format);
    } else if (out_size != size) {
      g_printerr("  FAIL %s stride %d: output is %" G_GSIZE_FORMAT " bytes (want %" G_GSIZE_FORMAT ")\n",
                 format, stride, out_size, size);
    } else if (padded_nv12 && (!meta || meta->stride[0] != stride || meta->stride[1] != stride ||
                               meta->offset[1] != (gsize)stride * height)) {
      g_printerr("  FAIL %s stride %d: output video meta does not carry the padded layout\n",
                 format, stride);
    } else if (!GST_MEMORY_IS_READONLY(om)) {
      g_printerr("  FAIL %s: output dma-buf memory is writable\n", format);
    } else {
      rc = 0;
      g_print("  PASS dma-buf %s %dx%d stride %d in -> glass2glass(identity) -> dma-buf out, %" G_GSIZE_FORMAT " bytes\n",
              format, width, height, stride, out_size);
    }
    kept = gst_buffer_ref(out);
    gst_sample_unref(sample);
  } else {
    g_printerr("  FAIL %s: no output sample\n", format);
  }

  gst_element_set_state(pipe, GST_STATE_NULL);
  if (kept) {
    /* With the pipeline gone, only the kept output still holds descriptors onto the memfd. */
    int held = memfd_count();
    gst_buffer_unref(kept);
    int left = memfd_count();
    if (held != 2 || left != 0) {
      g_printerr("  FAIL %s: the output held %d descriptors (want 2: GStreamer's and the g2g frame's), %d once freed (want 0)\n",
                 format, held, left);
      rc = 1;
    } else {
      g_print("  PASS %s: the g2g frame stays alive until GStreamer frees the output\n", format);
    }
  }
  gst_object_unref(alloc);
  gst_object_unref(src);
  gst_object_unref(sink);
  gst_object_unref(pipe);
  return rc;
}

int main(void) {
  gst_init(NULL, NULL);
  int rc = 0;
  rc |= roundtrip("RGBA", 64, 16, 256, 256 * 16, FALSE);
  /* NV12 is luma rows then half-height interleaved chroma rows. */
  rc |= roundtrip("NV12", 64, 16, 64, 64 * 16 * 3 / 2, FALSE);
  rc |= roundtrip("NV12", 64, 16, 128, 128 * 16 * 3 / 2, TRUE);
  return rc;
}
EOF

echo "== compiling dma-buf round-trip harness =="
cc "$work/dmabuf_roundtrip.c" -o "$work/dmabuf_roundtrip" -D_GNU_SOURCE \
  $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-allocators-1.0 gstreamer-video-1.0)

echo "== running dma-buf round-trip =="
"$work/dmabuf_roundtrip"
echo "== bridge dma-buf round-trip passed =="
