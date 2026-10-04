/* `gstwrap` C helper: drives a real GStreamer pipeline
 * `appsrc ! <element> ! appsink` so a g2g graph can host an unported GStreamer
 * element (design/README.md §7, the reverse of g2g-bridge). The Rust element
 * (src/gstwrap.rs) owns all g2g plumbing (caps negotiation, Frame mapping) and
 * calls these C-ABI functions to feed/drain the embedded GStreamer pipeline.
 *
 * System input is copied into a GstBuffer and system output is copied out to a
 * heap block the caller frees. A raw video frame is copied plane by plane
 * between the caller's layout and the caps' default GstVideoInfo layout, so
 * the wrapped element never sees a layout it might not honour. A dma-buf frame is pushed as a GstDmaBufMemory
 * over a dup of its fd, and a dma-buf sample is handed back as a dup of its fd
 * plus the plane layout, so neither direction copies pixels.
 *
 * The pipeline runs on GStreamer's own streaming threads; appsrc push and
 * appsink pull are MT-safe, so the Rust side drives them from its runner task
 * without owning those threads. appsrc caps and the optional appsink caps filter
 * are set programmatically (not in the parse string) to avoid caps-quoting.
 */
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <gst/allocators/gstdmabuf.h>
#include <gst/video/video.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* The Rust side mirrors the plane arrays below with a fixed length of 4. */
G_STATIC_ASSERT(GST_VIDEO_MAX_PLANES == 4);

/* try_pull / try_pull_dmabuf results, mirrored on the Rust side. */
#define G2G_GSTWRAP_PULLED 1
#define G2G_GSTWRAP_NOT_READY 0
#define G2G_GSTWRAP_EOS (-1)
#define G2G_GSTWRAP_NOT_DMABUF (-2)
#define G2G_GSTWRAP_FAILED (-3)

/* where each plane of a raw video frame sits in the caller's memory */
typedef struct G2gGstWrapPlanes {
  unsigned int n_planes;
  size_t offsets[GST_VIDEO_MAX_PLANES];
  int strides[GST_VIDEO_MAX_PLANES];
  size_t size;
} G2gGstWrapPlanes;

typedef struct G2gGstWrap {
  GstElement *pipeline;
  GstAppSrc *src;
  GstAppSink *sink;
  GstAllocator *dmabuf_allocator;
  /* the appsrc caps, for the plane layout of a pushed frame */
  gboolean input_is_video;
  GstVideoInfo input_info;
  /* the caps `output_planes` describes, a sample in other caps fails the pull */
  gboolean has_output_planes;
  GstVideoInfo output_info;
  G2gGstWrapPlanes output_planes;
} G2gGstWrap;

/* plane offsets count from the memory start, `memory_offset` bytes into the fd */
typedef struct G2gGstWrapDmaBufSample {
  GstSample *sample;
  int fd;
  uint64_t pts;
  size_t memory_offset;
  size_t memory_size;
  unsigned int height;
  unsigned int n_planes;
  size_t plane_offsets[GST_VIDEO_MAX_PLANES];
  int plane_strides[GST_VIDEO_MAX_PLANES];
} G2gGstWrapDmaBufSample;

typedef void (*G2gGstWrapRelease)(void *keep_alive);

/* Build and start `appsrc ! <element_desc> ! appsink`. `in_caps` is the g2g
 * input caps serialized as a gst caps string (set on appsrc). `out_caps`, when
 * non-NULL/non-empty, is set on the appsink as a caps filter, giving a
 * caps-driven element (videoscale, videoconvert) a downstream fixate target and
 * declaring the produced format. `output_planes`, when non-NULL, is the layout
 * system output is copied into, for the output caps (`out_caps`, else
 * `in_caps`). Returns NULL on any failure. */
G2gGstWrap *g2g_gstwrap_create(const char *element_desc, const char *in_caps,
                               const char *out_caps,
                               const G2gGstWrapPlanes *output_planes) {
  /* gst_init is idempotent; safe to call once per wrapped element. */
  gst_init(NULL, NULL);

  if (element_desc == NULL || element_desc[0] == '\0') {
    return NULL;
  }

  /* appsrc: time-format, not live (the g2g runner paces it), bounded queue so a
   * slow element back-pressures the feed instead of unbounded buffering. */
  gchar *desc = g_strdup_printf(
      "appsrc name=g2gsrc format=time is-live=false max-bytes=16777216 "
      "! %s "
      "! appsink name=g2gsink sync=false max-buffers=8 drop=false",
      element_desc);
  GError *err = NULL;
  GstElement *pipeline = gst_parse_launch(desc, &err);
  g_free(desc);
  if (pipeline == NULL || err != NULL) {
    if (err != NULL) {
      g_error_free(err);
    }
    if (pipeline != NULL) {
      gst_object_unref(pipeline);
    }
    return NULL;
  }

  GstElement *src = gst_bin_get_by_name(GST_BIN(pipeline), "g2gsrc");
  GstElement *sink = gst_bin_get_by_name(GST_BIN(pipeline), "g2gsink");
  if (src == NULL || sink == NULL) {
    if (src != NULL) {
      gst_object_unref(src);
    }
    if (sink != NULL) {
      gst_object_unref(sink);
    }
    gst_object_unref(pipeline);
    return NULL;
  }

  gboolean input_is_video = FALSE;
  GstVideoInfo input_info;
  gst_video_info_init(&input_info);
  if (in_caps != NULL && in_caps[0] != '\0') {
    GstCaps *caps = gst_caps_from_string(in_caps);
    if (caps != NULL) {
      input_is_video = gst_video_info_from_caps(&input_info, caps);
      gst_app_src_set_caps(GST_APP_SRC(src), caps);
      gst_caps_unref(caps);
    }
  }
  gboolean output_is_video = input_is_video;
  GstVideoInfo output_info = input_info;
  if (out_caps != NULL && out_caps[0] != '\0') {
    GstCaps *caps = gst_caps_from_string(out_caps);
    output_is_video = FALSE;
    if (caps != NULL) {
      output_is_video = gst_video_info_from_caps(&output_info, caps);
      gst_app_sink_set_caps(GST_APP_SINK(sink), caps);
      gst_caps_unref(caps);
    }
  }
  if (output_planes != NULL &&
      (!output_is_video ||
       output_planes->n_planes != GST_VIDEO_INFO_N_PLANES(&output_info))) {
    gst_object_unref(src);
    gst_object_unref(sink);
    gst_object_unref(pipeline);
    return NULL;
  }

  if (gst_element_set_state(pipeline, GST_STATE_PLAYING) ==
      GST_STATE_CHANGE_FAILURE) {
    gst_object_unref(src);
    gst_object_unref(sink);
    gst_object_unref(pipeline);
    return NULL;
  }

  G2gGstWrap *w = calloc(1, sizeof(G2gGstWrap));
  if (w == NULL) {
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(src);
    gst_object_unref(sink);
    gst_object_unref(pipeline);
    return NULL;
  }
  w->pipeline = pipeline;
  w->src = GST_APP_SRC(src);
  w->sink = GST_APP_SINK(sink);
  w->dmabuf_allocator = gst_dmabuf_allocator_new();
  w->input_is_video = input_is_video;
  w->input_info = input_info;
  if (output_planes != NULL) {
    w->has_output_planes = TRUE;
    w->output_info = output_info;
    w->output_planes = *output_planes;
  }
  return w;
}

static GstBuffer *wrap_frame(uint8_t *data, GstMemoryFlags flags,
                             const GstVideoInfo *info,
                             const G2gGstWrapPlanes *planes) {
  GstBuffer *buf = gst_buffer_new_wrapped_full(flags, data, planes->size, 0,
                                               planes->size, NULL, NULL);
  gsize offsets[GST_VIDEO_MAX_PLANES] = {0};
  gint strides[GST_VIDEO_MAX_PLANES] = {0};
  for (unsigned int i = 0; i < planes->n_planes; i++) {
    offsets[i] = planes->offsets[i];
    strides[i] = planes->strides[i];
  }
  gst_buffer_add_video_meta_full(
      buf, GST_VIDEO_FRAME_FLAG_NONE, GST_VIDEO_INFO_FORMAT(info),
      GST_VIDEO_INFO_WIDTH(info), GST_VIDEO_INFO_HEIGHT(info), planes->n_planes,
      offsets, strides);
  return buf;
}

/* a buffer without a GstVideoMeta is read in the default layout of `info` */
static gboolean copy_frame(GstBuffer *dst, GstBuffer *src,
                           const GstVideoInfo *info) {
  GstVideoFrame src_frame;
  GstVideoFrame dst_frame;
  if (!gst_video_frame_map(&src_frame, info, src, GST_MAP_READ)) {
    return FALSE;
  }
  if (!gst_video_frame_map(&dst_frame, info, dst, GST_MAP_WRITE)) {
    gst_video_frame_unmap(&src_frame);
    return FALSE;
  }
  gboolean copied = gst_video_frame_copy(&dst_frame, &src_frame);
  gst_video_frame_unmap(&dst_frame);
  gst_video_frame_unmap(&src_frame);
  return copied;
}

static GstBuffer *copy_to_default_layout(G2gGstWrap *w, const uint8_t *data,
                                         size_t len,
                                         const G2gGstWrapPlanes *planes) {
  if (!w->input_is_video ||
      planes->n_planes != GST_VIDEO_INFO_N_PLANES(&w->input_info) ||
      planes->size > len) {
    return NULL;
  }
  GstBuffer *src = wrap_frame((uint8_t *)data, GST_MEMORY_FLAG_READONLY,
                              &w->input_info, planes);
  GstBuffer *dst =
      gst_buffer_new_allocate(NULL, GST_VIDEO_INFO_SIZE(&w->input_info), NULL);
  gboolean copied = dst != NULL && copy_frame(dst, src, &w->input_info);
  gst_buffer_unref(src);
  if (!copied) {
    if (dst != NULL) {
      gst_buffer_unref(dst);
    }
    return NULL;
  }
  return dst;
}

/* Push one buffer (copied) with presentation timestamp `pts_ns`. `planes`,
 * when non-NULL, says where the raw video planes sit in `data`. Returns 0 on
 * success, -1 if the pipeline rejected the buffer (flushing / EOS / error). */
int g2g_gstwrap_push(G2gGstWrap *w, const uint8_t *data, size_t len,
                     const G2gGstWrapPlanes *planes, uint64_t pts_ns) {
  if (w == NULL) {
    return -1;
  }
  GstBuffer *buf = NULL;
  if (planes != NULL) {
    buf = copy_to_default_layout(w, data, len, planes);
  } else {
    buf = gst_buffer_new_allocate(NULL, len, NULL);
    if (buf != NULL) {
      gst_buffer_fill(buf, 0, data, len);
    }
  }
  if (buf == NULL) {
    return -1;
  }
  GST_BUFFER_PTS(buf) = (GstClockTime)pts_ns;
  GST_BUFFER_DTS(buf) = (GstClockTime)pts_ns;
  /* push_buffer takes ownership of `buf`. */
  GstFlowReturn r = gst_app_src_push_buffer(w->src, buf);
  return r == GST_FLOW_OK ? 0 : -1;
}

static GQuark keep_alive_quark(void) {
  return g_quark_from_static_string("g2g-gstwrap-keep-alive");
}

/* dma-buf reports its size only through lseek(SEEK_END) */
static off_t fd_size(int fd) {
  off_t size = lseek(fd, 0, SEEK_END);
  lseek(fd, 0, SEEK_SET);
  return size;
}

/* `release(keep_alive)` runs when GStreamer frees the memory, or on failure */
int g2g_gstwrap_push_dmabuf(G2gGstWrap *w, int fd, size_t data_offset,
                            size_t required_end, unsigned int n_planes,
                            const size_t *plane_offsets,
                            const int *plane_strides, uint64_t pts_ns,
                            void *keep_alive, G2gGstWrapRelease release) {
  if (w == NULL || n_planes > GST_VIDEO_MAX_PLANES ||
      (n_planes > 0 && (!w->input_is_video ||
                        n_planes != GST_VIDEO_INFO_N_PLANES(&w->input_info)))) {
    release(keep_alive);
    return -1;
  }
  int own = dup(fd);
  if (own < 0) {
    release(keep_alive);
    return -1;
  }
  off_t size = fd_size(own);
  if (size < 0 || required_end > (size_t)size || data_offset > (size_t)size) {
    close(own);
    release(keep_alive);
    return -1;
  }
  GstMemory *mem =
      gst_dmabuf_allocator_alloc(w->dmabuf_allocator, own, (gsize)size);
  if (mem == NULL) {
    close(own);
    release(keep_alive);
    return -1;
  }
  gst_mini_object_set_qdata(GST_MINI_OBJECT(mem), keep_alive_quark(),
                            keep_alive, release);
  /* an in-place element would otherwise write into the upstream frame */
  GST_MINI_OBJECT_FLAG_SET(mem, GST_MEMORY_FLAG_READONLY);
  if (n_planes == 0) {
    gst_memory_resize(mem, (gssize)data_offset, (gsize)size - data_offset);
  }

  GstBuffer *buf = gst_buffer_new();
  gst_buffer_append_memory(buf, mem);
  if (n_planes > 0) {
    gsize offsets[GST_VIDEO_MAX_PLANES] = {0};
    gint strides[GST_VIDEO_MAX_PLANES] = {0};
    for (unsigned int i = 0; i < n_planes; i++) {
      offsets[i] = plane_offsets[i];
      strides[i] = plane_strides[i];
    }
    gst_buffer_add_video_meta_full(
        buf, GST_VIDEO_FRAME_FLAG_NONE, GST_VIDEO_INFO_FORMAT(&w->input_info),
        GST_VIDEO_INFO_WIDTH(&w->input_info),
        GST_VIDEO_INFO_HEIGHT(&w->input_info), n_planes, offsets, strides);
  }
  GST_BUFFER_PTS(buf) = (GstClockTime)pts_ns;
  GST_BUFFER_DTS(buf) = (GstClockTime)pts_ns;
  GstFlowReturn r = gst_app_src_push_buffer(w->src, buf);
  return r == GST_FLOW_OK ? 0 : -1;
}

/* layout from the GstVideoMeta, else the sample caps, else `n_planes` stays 0 */
int g2g_gstwrap_try_pull_dmabuf(G2gGstWrap *w, G2gGstWrapDmaBufSample *out) {
  if (w == NULL) {
    return G2G_GSTWRAP_EOS;
  }
  GstSample *sample = gst_app_sink_try_pull_sample(w->sink, 0);
  if (sample == NULL) {
    return gst_app_sink_is_eos(w->sink) ? G2G_GSTWRAP_EOS
                                        : G2G_GSTWRAP_NOT_READY;
  }
  GstBuffer *buf = gst_sample_get_buffer(sample);
  GstMemory *mem = buf != NULL && gst_buffer_n_memory(buf) == 1
                       ? gst_buffer_peek_memory(buf, 0)
                       : NULL;
  if (mem == NULL || !gst_is_dmabuf_memory(mem)) {
    gst_sample_unref(sample);
    return G2G_GSTWRAP_NOT_DMABUF;
  }
  int fd = dup(gst_dmabuf_memory_get_fd(mem));
  if (fd < 0) {
    gst_sample_unref(sample);
    return G2G_GSTWRAP_FAILED;
  }

  memset(out, 0, sizeof(*out));
  gsize memory_offset = 0;
  out->memory_size = gst_memory_get_sizes(mem, &memory_offset, NULL);
  out->memory_offset = memory_offset;
  GstVideoMeta *meta = gst_buffer_get_video_meta(buf);
  GstVideoInfo info;
  GstCaps *caps = gst_sample_get_caps(sample);
  if (meta != NULL) {
    out->height = meta->height;
    out->n_planes = meta->n_planes;
    for (unsigned int i = 0; i < meta->n_planes; i++) {
      out->plane_offsets[i] = meta->offset[i];
      out->plane_strides[i] = meta->stride[i];
    }
  } else if (caps != NULL && gst_video_info_from_caps(&info, caps)) {
    out->height = GST_VIDEO_INFO_HEIGHT(&info);
    out->n_planes = GST_VIDEO_INFO_N_PLANES(&info);
    for (unsigned int i = 0; i < out->n_planes; i++) {
      out->plane_offsets[i] = GST_VIDEO_INFO_PLANE_OFFSET(&info, i);
      out->plane_strides[i] = GST_VIDEO_INFO_PLANE_STRIDE(&info, i);
    }
  }
  out->sample = sample;
  out->fd = fd;
  out->pts = (uint64_t)GST_BUFFER_PTS(buf);
  return G2G_GSTWRAP_PULLED;
}

void g2g_gstwrap_sample_unref(GstSample *sample) { gst_sample_unref(sample); }

size_t g2g_gstwrap_dmabuf_sample_size(void) {
  return sizeof(G2gGstWrapDmaBufSample);
}

size_t g2g_gstwrap_planes_size(void) { return sizeof(G2gGstWrapPlanes); }

/* a sample in caps other than the ones `output_planes` was built for fails */
static int copy_out_planes(G2gGstWrap *w, GstSample *sample, GstBuffer *buf,
                           uint8_t **out_data, size_t *out_len) {
  GstVideoInfo info;
  GstCaps *caps = gst_sample_get_caps(sample);
  if (caps == NULL || !gst_video_info_from_caps(&info, caps) ||
      GST_VIDEO_INFO_FORMAT(&info) != GST_VIDEO_INFO_FORMAT(&w->output_info) ||
      GST_VIDEO_INFO_WIDTH(&info) != GST_VIDEO_INFO_WIDTH(&w->output_info) ||
      GST_VIDEO_INFO_HEIGHT(&info) != GST_VIDEO_INFO_HEIGHT(&w->output_info)) {
    return G2G_GSTWRAP_FAILED;
  }
  size_t size = w->output_planes.size;
  uint8_t *copy = malloc(size > 0 ? size : 1);
  if (copy == NULL) {
    return G2G_GSTWRAP_FAILED;
  }
  GstBuffer *dst = wrap_frame(copy, 0, &info, &w->output_planes);
  gboolean copied = copy_frame(dst, buf, &info);
  gst_buffer_unref(dst);
  if (!copied) {
    free(copy);
    return G2G_GSTWRAP_FAILED;
  }
  *out_data = copy;
  *out_len = size;
  return G2G_GSTWRAP_PULLED;
}

/* Non-blocking drain of one processed frame. Returns 1 and fills the out params
 * (caller frees `*out_data` with g2g_gstwrap_free_buf) when a sample is ready, 0
 * when none is ready yet (the element has internal latency), -1 at EOS, and -3
 * when the sample could not be mapped or copied. */
int g2g_gstwrap_try_pull(G2gGstWrap *w, uint8_t **out_data, size_t *out_len,
                         uint64_t *out_pts) {
  if (w == NULL) {
    return G2G_GSTWRAP_EOS;
  }
  /* 0 timeout: return immediately if nothing is queued. */
  GstSample *sample = gst_app_sink_try_pull_sample(w->sink, 0);
  if (sample == NULL) {
    return gst_app_sink_is_eos(w->sink) ? G2G_GSTWRAP_EOS
                                        : G2G_GSTWRAP_NOT_READY;
  }
  GstBuffer *buf = gst_sample_get_buffer(sample);
  if (buf != NULL && w->has_output_planes) {
    int result = copy_out_planes(w, sample, buf, out_data, out_len);
    *out_pts = (uint64_t)GST_BUFFER_PTS(buf);
    gst_sample_unref(sample);
    return result;
  }
  GstMapInfo map;
  if (buf == NULL || !gst_buffer_map(buf, &map, GST_MAP_READ)) {
    gst_sample_unref(sample);
    return G2G_GSTWRAP_FAILED;
  }
  uint8_t *copy = malloc(map.size > 0 ? map.size : 1);
  if (copy == NULL) {
    gst_buffer_unmap(buf, &map);
    gst_sample_unref(sample);
    return G2G_GSTWRAP_FAILED;
  }
  memcpy(copy, map.data, map.size);
  *out_data = copy;
  *out_len = map.size;
  /* GST_BUFFER_PTS may be GST_CLOCK_TIME_NONE; it passes through as-is, which
   * the Rust side reads as FrameTiming::PTS_NONE. */
  *out_pts = (uint64_t)GST_BUFFER_PTS(buf);
  gst_buffer_unmap(buf, &map);
  gst_sample_unref(sample);
  return G2G_GSTWRAP_PULLED;
}

void g2g_gstwrap_free_buf(uint8_t *p) { free(p); }

/* Signal end-of-stream on the feed; the element flushes its buffered frames,
 * which the caller then drains with try_pull until it returns -1. */
void g2g_gstwrap_eos(G2gGstWrap *w) {
  if (w != NULL) {
    gst_app_src_end_of_stream(w->src);
  }
}

void g2g_gstwrap_free(G2gGstWrap *w) {
  if (w == NULL) {
    return;
  }
  if (w->pipeline != NULL) {
    gst_element_set_state(w->pipeline, GST_STATE_NULL);
    gst_object_unref(w->pipeline);
  }
  if (w->src != NULL) {
    gst_object_unref(w->src);
  }
  if (w->sink != NULL) {
    gst_object_unref(w->sink);
  }
  if (w->dmabuf_allocator != NULL) {
    gst_object_unref(w->dmabuf_allocator);
  }
  free(w);
}
