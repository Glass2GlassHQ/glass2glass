/* `gstwrap` C helper: drives a real GStreamer pipeline
 * `appsrc ! <element> ! appsink` so a g2g graph can host an unported GStreamer
 * element (DESIGN.md §7, the reverse of g2g-bridge). The Rust element
 * (src/gstwrap.rs) owns all g2g plumbing (caps negotiation, Frame mapping) and
 * calls these C-ABI functions to feed/drain the embedded GStreamer pipeline.
 *
 * v1 is system-memory: input buffers are copied into a GstBuffer, output samples
 * are copied out to a heap block the caller frees. Zero-copy dma-buf handoff is
 * future work (it would import a GstDmaBufMemory on both sides).
 *
 * The pipeline runs on GStreamer's own streaming threads; appsrc push and
 * appsink pull are MT-safe, so the Rust side drives them from its runner task
 * without owning those threads. appsrc caps and the optional appsink caps filter
 * are set programmatically (not in the parse string) to avoid caps-quoting.
 */
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct G2gGstWrap {
  GstElement *pipeline;
  GstAppSrc *src;
  GstAppSink *sink;
} G2gGstWrap;

/* Build and start `appsrc ! <element_desc> ! appsink`. `in_caps` is the g2g
 * input caps serialized as a gst caps string (set on appsrc). `out_caps`, when
 * non-NULL/non-empty, is set on the appsink as a caps filter, giving a
 * caps-driven element (videoscale, videoconvert) a downstream fixate target and
 * declaring the produced format. Returns NULL on any failure. */
G2gGstWrap *g2g_gstwrap_create(const char *element_desc, const char *in_caps,
                               const char *out_caps) {
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

  if (in_caps != NULL && in_caps[0] != '\0') {
    GstCaps *caps = gst_caps_from_string(in_caps);
    if (caps != NULL) {
      gst_app_src_set_caps(GST_APP_SRC(src), caps);
      gst_caps_unref(caps);
    }
  }
  if (out_caps != NULL && out_caps[0] != '\0') {
    GstCaps *caps = gst_caps_from_string(out_caps);
    if (caps != NULL) {
      gst_app_sink_set_caps(GST_APP_SINK(sink), caps);
      gst_caps_unref(caps);
    }
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
  return w;
}

/* Push one buffer (copied) with presentation timestamp `pts_ns`. Returns 0 on
 * success, -1 if the pipeline rejected the buffer (flushing / EOS / error). */
int g2g_gstwrap_push(G2gGstWrap *w, const uint8_t *data, size_t len,
                     uint64_t pts_ns) {
  if (w == NULL) {
    return -1;
  }
  GstBuffer *buf = gst_buffer_new_allocate(NULL, len, NULL);
  if (buf == NULL) {
    return -1;
  }
  gst_buffer_fill(buf, 0, data, len);
  GST_BUFFER_PTS(buf) = (GstClockTime)pts_ns;
  GST_BUFFER_DTS(buf) = (GstClockTime)pts_ns;
  /* push_buffer takes ownership of `buf`. */
  GstFlowReturn r = gst_app_src_push_buffer(w->src, buf);
  return r == GST_FLOW_OK ? 0 : -1;
}

/* Non-blocking drain of one processed frame. Returns 1 and fills the out params
 * (caller frees `*out_data` with g2g_gstwrap_free_buf) when a sample is ready, 0
 * when none is ready yet (the element has internal latency), and -1 at EOS. */
int g2g_gstwrap_try_pull(G2gGstWrap *w, uint8_t **out_data, size_t *out_len,
                         uint64_t *out_pts) {
  if (w == NULL) {
    return -1;
  }
  /* 0 timeout: return immediately if nothing is queued. */
  GstSample *sample = gst_app_sink_try_pull_sample(w->sink, 0);
  if (sample == NULL) {
    return gst_app_sink_is_eos(w->sink) ? -1 : 0;
  }
  GstBuffer *buf = gst_sample_get_buffer(sample);
  GstMapInfo map;
  if (buf == NULL || !gst_buffer_map(buf, &map, GST_MAP_READ)) {
    gst_sample_unref(sample);
    return 0;
  }
  uint8_t *copy = malloc(map.size > 0 ? map.size : 1);
  if (copy == NULL) {
    gst_buffer_unmap(buf, &map);
    gst_sample_unref(sample);
    return 0;
  }
  memcpy(copy, map.data, map.size);
  *out_data = copy;
  *out_len = map.size;
  /* GST_BUFFER_PTS may be GST_CLOCK_TIME_NONE; the Rust side treats that as 0. */
  *out_pts = (uint64_t)GST_BUFFER_PTS(buf);
  gst_buffer_unmap(buf, &map);
  gst_sample_unref(sample);
  return 1;
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
  free(w);
}
