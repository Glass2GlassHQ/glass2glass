#!/usr/bin/env bash
# Validate the bridge's zero-copy dma-buf round-trip: a dma-buf GstBuffer in ->
# glass2glass (fragment=identity) -> dma-buf GstBuffer out, for packed RGBA and
# for NV12 at the default and at a padded stride, proving both wiring
# steps (input auto-detect + import via g2g_bridge_push_dmabuf; output wrap-back
# via gst_dmabuf_allocator_alloc). Uses a memfd wrapped as dma-buf memory, so no
# special hardware / dma-buf producer element is needed.
#
# A GPU case then chains two glass2glass elements on `memory:DMABuf` caps: a
# linear GBM buffer goes through `dmabuftowgpu ! wgputodmabuf`, crosses
# GStreamer as dma-buf memory, and comes back through `dmabuftowgpu !
# wgpudownload` byte-for-byte. A tiled dma-buf from `gldownload` must fail fast.
# Host GStreamer and GPU only, so validated locally, not in CI.
#
# Prerequisites: gstreamer-1.0, gstreamer-app-1.0, gstreamer-allocators-1.0,
# gstreamer-video-1.0, gbm dev packages, the GStreamer GL plugins, a C compiler,
# and a GPU whose Vulkan driver imports dma-buf. RENDER_NODE picks the GBM
# device (default: the render node that did not boot the display, else the only one).
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== building libgstglass2glass.so =="
cargo build -p g2g-bridge --features gstreamer,wgpu

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
#include <fcntl.h>
#include <gbm.h>
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

#define GPU_WIDTH 64
#define GPU_HEIGHT 16
#define RGBA_BYTES_PER_PIXEL 4
/* GStreamer's drm-format for RGBA, the fourcc of GBM_FORMAT_ABGR8888. */
#define RGBA_DRM_FORMAT "AB24"

static gboolean saw_dmabuf_between;

static GstPadProbeReturn note_dmabuf(GstPad *pad, GstPadProbeInfo *info, gpointer user) {
  (void)pad; (void)user;
  GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER(info);
  saw_dmabuf_between = gst_is_dmabuf_memory(gst_buffer_peek_memory(buf, 0));
  return GST_PAD_PROBE_OK;
}

/* Two glass2glass elements pass a GPU frame to each other as dma-buf memory. */
static int chained_gpu_roundtrip(const char *render_node) {
  const int row_bytes = GPU_WIDTH * RGBA_BYTES_PER_PIXEL;
  int drm = open(render_node, O_RDWR | O_CLOEXEC);
  struct gbm_device *dev = drm >= 0 ? gbm_create_device(drm) : NULL;
  struct gbm_bo *bo = dev ? gbm_bo_create(dev, GPU_WIDTH, GPU_HEIGHT, GBM_FORMAT_ABGR8888,
                                          GBM_BO_USE_LINEAR) : NULL;
  if (!bo) { g_printerr("  FAIL no linear GBM buffer on %s\n", render_node); return 1; }
  guint8 expected[GPU_WIDTH * GPU_HEIGHT * RGBA_BYTES_PER_PIXEL];
  for (gsize i = 0; i < sizeof expected; i++) expected[i] = (guint8)(i * 7 + 3);
  uint32_t stride = 0;
  void *map_data = NULL;
  guint8 *pixels = gbm_bo_map(bo, 0, 0, GPU_WIDTH, GPU_HEIGHT, GBM_BO_TRANSFER_WRITE, &stride,
                              &map_data);
  if (!pixels) { g_printerr("  FAIL GBM buffer does not map\n"); return 1; }
  for (int row = 0; row < GPU_HEIGHT; row++)
    memcpy(pixels + (gsize)row * stride, expected + row * row_bytes, row_bytes);
  gbm_bo_unmap(bo, map_data);

  gchar *launch = g_strdup_printf(
      "appsrc name=src is-live=false format=time "
      "caps=\"video/x-raw(memory:DMABuf),format=DMA_DRM,drm-format=%s,width=%d,height=%d,framerate=1/1\" ! "
      "glass2glass fragment=\"dmabuftowgpu ! wgputodmabuf\" ! "
      "glass2glass name=down fragment=\"dmabuftowgpu ! wgpudownload\" "
      "output-caps=\"video/x-raw,format=RGBA,width=%d,height=%d,framerate=1/1\" ! "
      "appsink name=sink", RGBA_DRM_FORMAT, GPU_WIDTH, GPU_HEIGHT, GPU_WIDTH, GPU_HEIGHT);
  GstElement *pipe = gst_parse_launch(launch, NULL);
  g_free(launch);
  if (!pipe) { g_printerr("  FAIL GPU pipeline build failed\n"); return 1; }
  GstElement *src = gst_bin_get_by_name(GST_BIN(pipe), "src");
  GstElement *down = gst_bin_get_by_name(GST_BIN(pipe), "down");
  GstElement *sink = gst_bin_get_by_name(GST_BIN(pipe), "sink");
  GstPad *between = gst_element_get_static_pad(down, "sink");
  gst_pad_add_probe(between, GST_PAD_PROBE_TYPE_BUFFER, note_dmabuf, NULL, NULL);

  GstAllocator *alloc = gst_dmabuf_allocator_new();
  GstBuffer *buf = gst_buffer_new();
  gst_buffer_append_memory(buf, gst_dmabuf_allocator_alloc(alloc, gbm_bo_get_fd(bo),
                                                           (gsize)stride * GPU_HEIGHT));
  GST_BUFFER_PTS(buf) = 0;
  if ((int)stride != row_bytes) {
    gsize offsets[GST_VIDEO_MAX_PLANES] = {0};
    gint strides[GST_VIDEO_MAX_PLANES] = {(gint)stride};
    gst_buffer_add_video_meta_full(buf, GST_VIDEO_FRAME_FLAG_NONE, GST_VIDEO_FORMAT_RGBA, GPU_WIDTH,
                                   GPU_HEIGHT, 1, offsets, strides);
  }

  gst_element_set_state(pipe, GST_STATE_PLAYING);
  int rc = 1;
  if (gst_app_src_push_buffer(GST_APP_SRC(src), buf) == GST_FLOW_OK) {
    gst_app_src_end_of_stream(GST_APP_SRC(src));
    GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
    GstMapInfo map;
    if (!sample) {
      g_printerr("  FAIL GPU chain: no output sample\n");
    } else if (!saw_dmabuf_between) {
      g_printerr("  FAIL GPU chain: the frame between the two elements is not dma-buf memory\n");
    } else if (!gst_buffer_map(gst_sample_get_buffer(sample), &map, GST_MAP_READ)) {
      g_printerr("  FAIL GPU chain: output does not map\n");
    } else {
      if (map.size == sizeof expected && memcmp(map.data, expected, sizeof expected) == 0) {
        rc = 0;
        g_print("  PASS linear dma-buf RGBA %dx%d stride %u -> glass2glass(dmabuftowgpu ! wgputodmabuf) "
                "-> dma-buf -> glass2glass(dmabuftowgpu ! wgpudownload) -> same %" G_GSIZE_FORMAT " bytes\n",
                GPU_WIDTH, GPU_HEIGHT, stride, map.size);
      } else {
        g_printerr("  FAIL GPU chain: %" G_GSIZE_FORMAT " output bytes differ from the input\n", map.size);
      }
      gst_buffer_unmap(gst_sample_get_buffer(sample), &map);
    }
    if (sample) gst_sample_unref(sample);
  } else {
    g_printerr("  FAIL GPU chain: push failed\n");
  }

  gst_element_set_state(pipe, GST_STATE_NULL);
  gst_object_unref(between);
  gst_object_unref(alloc);
  gst_object_unref(src);
  gst_object_unref(down);
  gst_object_unref(sink);
  gst_object_unref(pipe);
  gbm_bo_destroy(bo);
  gbm_device_destroy(dev);
  close(drm);
  return rc;
}

int main(int argc, char **argv) {
  gst_init(NULL, NULL);
  int rc = 0;
  if (argc > 1) return chained_gpu_roundtrip(argv[1]);
  rc |= roundtrip("RGBA", 64, 16, 256, 256 * 16, FALSE);
  /* NV12 is luma rows then half-height interleaved chroma rows. */
  rc |= roundtrip("NV12", 64, 16, 64, 64 * 16 * 3 / 2, FALSE);
  rc |= roundtrip("NV12", 64, 16, 128, 128 * 16 * 3 / 2, TRUE);
  return rc;
}
EOF

echo "== compiling dma-buf round-trip harness =="
cc "$work/dmabuf_roundtrip.c" -o "$work/dmabuf_roundtrip" -D_GNU_SOURCE \
  $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-allocators-1.0 gstreamer-video-1.0 gbm)

echo "== running dma-buf round-trip =="
"$work/dmabuf_roundtrip"

render_node="${RENDER_NODE:-}"
if [ -z "$render_node" ]; then
  for node in /dev/dri/renderD*; do
    render_node="$node"
    [ "$(cat "/sys/class/drm/$(basename "$node")/device/boot_vga" 2>/dev/null)" = 0 ] && break
  done
fi
echo "== running chained GPU dma-buf round-trip on $render_node =="
"$work/dmabuf_roundtrip" "$render_node"

echo "== tiled dma-buf from gldownload must fail fast =="
tiled_rc=0
timeout 20 gst-launch-1.0 videotestsrc num-buffers=1 ! video/x-raw,format=RGBA,width=64,height=16 \
  ! glupload ! gldownload ! "video/x-raw(memory:DMABuf)" \
  ! glass2glass fragment="dmabuftowgpu ! wgpudownload" ! fakesink >/dev/null 2>&1 || tiled_rc=$?
case "$tiled_rc" in
  0) echo "FAIL a tiled dma-buf ran without error"; exit 1 ;;
  124) echo "FAIL a tiled dma-buf hung"; exit 1 ;;
  *) echo "  PASS tiled dma-buf refused (exit $tiled_rc)" ;;
esac
echo "== bridge dma-buf round-trip passed =="
