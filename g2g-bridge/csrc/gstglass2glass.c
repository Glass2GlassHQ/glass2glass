/* GStreamer element `glass2glass`: embeds a g2g sub-graph inside a GStreamer
 * pipeline (design/README.md §7). This is the thin GObject shell over the Rust
 * `BridgeGraph` impedance core; it owns all GStreamer/GObject boilerplate and
 * delegates the actual work to the C-ABI functions in `src/ffi.rs`.
 *
 * v1 is an in-place transform: the embedded fragment must preserve caps and
 * buffer size (a wgpu effect, videobalance, an ML preprocessor that keeps the
 * pixel format). Caps/size-changing fragments are future work (they need
 * output-buffer allocation + g2g->GstCaps mapping). Pads are ANY; the input
 * caps handed to g2g are the negotiated sink caps, serialized.
 */
#include <gst/gst.h>
#include <gst/allocators/gstdmabuf.h>
#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>
#include <string.h>
#include <unistd.h>

/* ---- Rust C-ABI core (src/ffi.rs) ---------------------------------------- */
typedef struct G2gBridge G2gBridge;
typedef struct {
  int kind; /* 0 = system bytes, 1 = dma-buf fd */
  const unsigned char *data;
  size_t len;
  int fd;
  unsigned int stride;
  unsigned int offset;
  unsigned long long pts_ns;
  void *owner;
} G2gOut;

/* The Rust side mirrors the plane arrays below with a fixed length of 4. */
G_STATIC_ASSERT(GST_VIDEO_MAX_PLANES == 4);

/* where each plane of a tightly packed g2g frame sits (`GstVideoPlanes` in g2g-plugins) */
typedef struct {
  unsigned int n_planes;
  size_t offsets[GST_VIDEO_MAX_PLANES];
  int strides[GST_VIDEO_MAX_PLANES];
  size_t size;
} G2gBridgePlanes;

extern G2gBridge *g2g_bridge_create(const char *fragment, const char *in_caps,
                                    const char *out_caps);
extern int g2g_bridge_tight_planes(const char *caps, G2gBridgePlanes *out);
extern int g2g_bridge_push_buf(G2gBridge *b, const unsigned char *data, size_t len,
                               unsigned long long pts_ns);
extern int g2g_bridge_push_dmabuf(G2gBridge *b, int fd, unsigned int stride, unsigned int offset,
                                  unsigned long long pts_ns);
extern int g2g_bridge_pull_buf(G2gBridge *b, G2gOut *out);
extern void g2g_bridge_out_release(G2gOut *out);
extern void g2g_bridge_destroy(G2gBridge *b);

/* ---- GObject type -------------------------------------------------------- */
#define GST_TYPE_GLASS2GLASS (gst_glass2glass_get_type())
G_DECLARE_FINAL_TYPE(GstGlass2Glass, gst_glass2glass, GST, GLASS2GLASS, GstBaseTransform)

struct _GstGlass2Glass {
  GstBaseTransform parent;
  gchar *fragment;     /* the g2g sub-pipeline, e.g. "videobalance saturation=0" */
  gchar *input_caps;   /* optional override of the serialized sink caps */
  gchar *output_caps;  /* if set, the sub-graph rescales/reformats to these caps */
  G2gBridge *bridge;   /* live between set_caps and stop */
  guint in_stride;     /* input plane-0 stride, for a dma-buf input with no video meta */
  GstVideoInfo in_info;    /* negotiated input format, for repacking system input rows */
  gboolean have_in_planes; /* whether `in_planes` holds the tight layout of the input caps */
  G2gBridgePlanes in_planes;
  GstVideoInfo out_info;   /* negotiated output format, for dma-buf output sizing */
  gboolean have_out_info;  /* whether the output caps are raw video */
  gboolean have_out_planes; /* whether `out_planes` holds the tight layout of the output caps */
  G2gBridgePlanes out_planes;
  GstAllocator *dmabuf_alloc; /* wraps a produced dma-buf fd into a GstBuffer */
};

G_DEFINE_TYPE(GstGlass2Glass, gst_glass2glass, GST_TYPE_BASE_TRANSFORM)

GST_DEBUG_CATEGORY_STATIC(glass2glass_debug);
#define GST_CAT_DEFAULT glass2glass_debug

enum { PROP_0, PROP_FRAGMENT, PROP_INPUT_CAPS, PROP_OUTPUT_CAPS };

static GstStaticPadTemplate sink_template =
    GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);
static GstStaticPadTemplate src_template =
    GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);

/* ---- properties ---------------------------------------------------------- */
static void gst_glass2glass_set_property(GObject *object, guint prop_id, const GValue *value,
                                         GParamSpec *pspec) {
  GstGlass2Glass *self = GST_GLASS2GLASS(object);
  switch (prop_id) {
    case PROP_FRAGMENT:
      g_free(self->fragment);
      self->fragment = g_value_dup_string(value);
      break;
    case PROP_INPUT_CAPS:
      g_free(self->input_caps);
      self->input_caps = g_value_dup_string(value);
      break;
    case PROP_OUTPUT_CAPS:
      g_free(self->output_caps);
      self->output_caps = g_value_dup_string(value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
  }
}

static void gst_glass2glass_get_property(GObject *object, guint prop_id, GValue *value,
                                         GParamSpec *pspec) {
  GstGlass2Glass *self = GST_GLASS2GLASS(object);
  switch (prop_id) {
    case PROP_FRAGMENT:
      g_value_set_string(value, self->fragment);
      break;
    case PROP_INPUT_CAPS:
      g_value_set_string(value, self->input_caps);
      break;
    case PROP_OUTPUT_CAPS:
      g_value_set_string(value, self->output_caps);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
  }
}

/* ---- caps negotiation ---------------------------------------------------- */
/* Advertise what this element can turn the given caps into. With `output-caps`
 * set the sub-graph rescales/reformats, so the sink->src direction offers those
 * caps; without it the element is caps-preserving (src == sink), which lets the
 * base class run the fast in-place path. The src->sink direction cannot be
 * inverted for an arbitrary fragment, so it offers ANY (upstream fixes the real
 * input caps) when output-caps is set. */
static GstCaps *gst_glass2glass_transform_caps(GstBaseTransform *base, GstPadDirection direction,
                                               GstCaps *caps, GstCaps *filter) {
  GstGlass2Glass *self = GST_GLASS2GLASS(base);
  GstCaps *others;
  if (self->output_caps) {
    others = (direction == GST_PAD_SINK) ? gst_caps_from_string(self->output_caps)
                                         : gst_caps_new_any();
  } else {
    others = gst_caps_ref(caps); /* preserving: same caps both directions */
  }
  if (filter) {
    GstCaps *clipped = gst_caps_intersect_full(filter, others, GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref(others);
    others = clipped;
  }
  return others;
}

/* GStreamer 1.24+ dma-buf caps say format=DMA_DRM and name the pixels in drm-format. */
static gboolean video_info_from_caps(GstVideoInfo *info, GstCaps *caps) {
  if (!gst_video_is_dma_drm_caps(caps))
    return gst_video_info_from_caps(info, caps);
  GstVideoInfoDmaDrm drm_info;
  return gst_video_info_dma_drm_from_caps(&drm_info, caps) &&
         gst_video_info_dma_drm_to_video_info(&drm_info, info);
}

/* Output buffer size for a given (raw video) caps, needed when the element is
 * not operating in place. */
static gboolean gst_glass2glass_get_unit_size(GstBaseTransform *base, GstCaps *caps, gsize *size) {
  (void)base;
  GstVideoInfo info;
  if (!video_info_from_caps(&info, caps))
    return FALSE;
  *size = GST_VIDEO_INFO_SIZE(&info);
  return TRUE;
}

/* ---- row layout ---------------------------------------------------------- */
/* GStreamer pads rows to 4 bytes by default and g2g packs them tight, so a system
 * frame is copied between the two layouts unless they already agree. */

static gboolean tight_planes(const gchar *caps, const GstVideoInfo *info, G2gBridgePlanes *planes) {
  return g2g_bridge_tight_planes(caps, planes) &&
         planes->n_planes == GST_VIDEO_INFO_N_PLANES(info);
}

static gboolean layout_is_tight(const G2gBridgePlanes *planes, const gsize *offsets,
                                const gint *strides) {
  for (guint i = 0; i < planes->n_planes; i++) {
    if (offsets[i] != planes->offsets[i] || strides[i] != planes->strides[i])
      return FALSE;
  }
  return TRUE;
}

static gboolean buffer_is_tight(GstBuffer *buf, const GstVideoInfo *info,
                                const G2gBridgePlanes *planes) {
  GstVideoMeta *meta = gst_buffer_get_video_meta(buf);
  if (meta)
    return layout_is_tight(planes, meta->offset, meta->stride);
  return layout_is_tight(planes, info->offset, info->stride);
}

static GstBuffer *wrap_frame(guint8 *data, GstMemoryFlags flags, const GstVideoInfo *info,
                             const G2gBridgePlanes *planes) {
  GstBuffer *buf = gst_buffer_new_wrapped_full(flags, data, planes->size, 0, planes->size, NULL, NULL);
  gsize offsets[GST_VIDEO_MAX_PLANES] = {0};
  gint strides[GST_VIDEO_MAX_PLANES] = {0};
  for (guint i = 0; i < planes->n_planes; i++) {
    offsets[i] = planes->offsets[i];
    strides[i] = planes->strides[i];
  }
  gst_buffer_add_video_meta_full(buf, GST_VIDEO_FRAME_FLAG_NONE, GST_VIDEO_INFO_FORMAT(info),
                                 GST_VIDEO_INFO_WIDTH(info), GST_VIDEO_INFO_HEIGHT(info),
                                 planes->n_planes, offsets, strides);
  return buf;
}

/* a buffer without a GstVideoMeta is read in the default layout of `info` */
static gboolean copy_frame(GstBuffer *dst, GstBuffer *src, const GstVideoInfo *info) {
  GstVideoFrame src_frame;
  GstVideoFrame dst_frame;
  if (!gst_video_frame_map(&src_frame, info, src, GST_MAP_READ))
    return FALSE;
  if (!gst_video_frame_map(&dst_frame, info, dst, GST_MAP_WRITE)) {
    gst_video_frame_unmap(&src_frame);
    return FALSE;
  }
  gboolean copied = gst_video_frame_copy(&dst_frame, &src_frame);
  gst_video_frame_unmap(&dst_frame);
  gst_video_frame_unmap(&src_frame);
  return copied;
}

/* ---- transform vmethods -------------------------------------------------- */
/* Build the sub-graph once caps are fixed. `incaps` describes the buffers the
 * embedded appsrc receives; `outcaps` (== incaps for a preserving fragment) the
 * frames it produces. Records the input stride and output format for the
 * dma-buf import and output (which carry no byte length of their own). */
static gboolean gst_glass2glass_set_caps(GstBaseTransform *base, GstCaps *incaps,
                                         GstCaps *outcaps) {
  GstGlass2Glass *self = GST_GLASS2GLASS(base);
  if (self->bridge) {
    g2g_bridge_destroy(self->bridge);
    self->bridge = NULL;
  }

  gboolean have_in_info = video_info_from_caps(&self->in_info, incaps);
  self->in_stride = have_in_info ? (guint)GST_VIDEO_INFO_PLANE_STRIDE(&self->in_info, 0) : 0;
  self->have_out_info = video_info_from_caps(&self->out_info, outcaps);

  gchar *negotiated_in = gst_caps_to_string(incaps);
  gchar *outstr = gst_caps_to_string(outcaps);
  self->have_in_planes = have_in_info && tight_planes(negotiated_in, &self->in_info, &self->in_planes);
  self->have_out_planes =
      self->have_out_info && tight_planes(outstr, &self->out_info, &self->out_planes);  const gchar *instr = self->input_caps ? self->input_caps : negotiated_in;
  const char *frag = self->fragment ? self->fragment : "identity";
  self->bridge = g2g_bridge_create(frag, instr, outstr);
  if (!self->bridge)
    GST_ERROR_OBJECT(self, "failed to build g2g sub-graph: fragment=\"%s\" in=\"%s\" out=\"%s\"",
                     frag, instr, outstr);
  g_free(negotiated_in);
  g_free(outstr);
  return self->bridge != NULL;
}

/* Push one input buffer into the sub-graph. A dma-buf-backed buffer is imported
 * zero-copy (the fd, not the mapped bytes); any other memory is mapped and its
 * bytes copied in, repacked into tight rows when GStreamer's rows are padded. */
static gboolean push_input(GstGlass2Glass *self, GstBuffer *in) {
  guint64 pts = GST_BUFFER_PTS_IS_VALID(in) ? GST_BUFFER_PTS(in) : 0;
  GstMemory *mem = gst_buffer_peek_memory(in, 0);
  if (mem && gst_is_dmabuf_memory(mem)) {
    int fd = gst_dmabuf_memory_get_fd(mem);
    guint stride = self->in_stride, offset = 0;
    GstVideoMeta *vmeta = gst_buffer_get_video_meta(in);
    if (vmeta && vmeta->n_planes > 0) {
      stride = (guint)vmeta->stride[0];
      offset = (guint)vmeta->offset[0];
    }
    return g2g_bridge_push_dmabuf(self->bridge, fd, stride, offset, pts);
  }
  if (self->have_in_planes && !buffer_is_tight(in, &self->in_info, &self->in_planes)) {
    guint8 *tight = g_malloc(self->in_planes.size);
    GstBuffer *dst = wrap_frame(tight, 0, &self->in_info, &self->in_planes);
    gboolean ok = copy_frame(dst, in, &self->in_info) &&
                  g2g_bridge_push_buf(self->bridge, tight, self->in_planes.size, pts);
    gst_buffer_unref(dst);
    g_free(tight);
    return ok;
  }
  GstMapInfo map;
  if (!gst_buffer_map(in, &map, GST_MAP_READ))
    return FALSE;
  /* GStreamer's buffer may run past the last tight row */
  gsize len = self->have_in_planes ? MIN(map.size, self->in_planes.size) : map.size;
  gboolean ok = g2g_bridge_push_buf(self->bridge, map.data, len, pts);
  gst_buffer_unmap(in, &map);
  return ok;
}

/* The plane layout g2g gives a dma-buf frame at this luma stride, and its size. */
static gsize dmabuf_layout(const GstVideoInfo *info, guint stride, guint offset,
                           gsize offsets[GST_VIDEO_MAX_PLANES],
                           gint strides[GST_VIDEO_MAX_PLANES]) {
  const GstVideoFormatInfo *finfo = info->finfo;
  gsize next = offset;
  for (guint plane = 0; plane < GST_VIDEO_INFO_N_PLANES(info); plane++) {
    guint comp = 0;
    while (GST_VIDEO_FORMAT_INFO_PLANE(finfo, comp) != plane)
      comp++;
    guint plane_stride = stride * GST_VIDEO_FORMAT_INFO_PSTRIDE(finfo, comp) /
                         GST_VIDEO_FORMAT_INFO_PSTRIDE(finfo, 0);
    plane_stride >>= GST_VIDEO_FORMAT_INFO_W_SUB(finfo, comp);
    gint rows = GST_VIDEO_FORMAT_INFO_SCALE_HEIGHT(finfo, comp, GST_VIDEO_INFO_HEIGHT(info));
    offsets[plane] = next;
    strides[plane] = (gint)plane_stride;
    next += (gsize)plane_stride * rows;
  }
  return next;
}

static void release_held_frame(gpointer held) {
  g2g_bridge_out_release(held);
  g_free(held);
}

/* Build the downstream buffer from a pulled frame: a system frame becomes an
 * owned GstBuffer (bytes copied, into GStreamer's padded rows when they differ
 * from g2g's tight ones); a dma-buf frame is wrapped zero-copy into a
 * dma-buf GstBuffer (its fd dup'ed, so the g2g frame keeps its own), and the
 * frame moves onto that memory, clearing `out->owner`. */
static GstBuffer *wrap_output(GstGlass2Glass *self, G2gOut *out) {
  if (out->kind == 1) {
    if (!self->have_out_info)
      return NULL;
    if (!self->dmabuf_alloc)
      self->dmabuf_alloc = gst_dmabuf_allocator_new();
    const GstVideoInfo *info = &self->out_info;
    gboolean default_layout =
        out->offset == 0 && (gint)out->stride == GST_VIDEO_INFO_PLANE_STRIDE(info, 0);
    gsize offsets[GST_VIDEO_MAX_PLANES];
    gint strides[GST_VIDEO_MAX_PLANES];
    gsize size = default_layout ? GST_VIDEO_INFO_SIZE(info)
                                : dmabuf_layout(info, out->stride, out->offset, offsets, strides);
    /* dup: the g2g frame owns `out->fd`; GStreamer's dma-buf memory owns its own. */
    GstMemory *mem = gst_dmabuf_allocator_alloc(self->dmabuf_alloc, dup(out->fd), size);
    if (!mem)
      return NULL;
    /* Hold the frame until GStreamer frees the memory: its producer recycles the buffer on release. */
    G2gOut *held = g_new(G2gOut, 1);
    *held = *out;
    out->owner = NULL;
    gst_mini_object_set_qdata(GST_MINI_OBJECT(mem), g_quark_from_static_string("g2g-bridge-frame"),
                              held, release_held_frame);
    GST_MINI_OBJECT_FLAG_SET(mem, GST_MEMORY_FLAG_READONLY);
    GstBuffer *buf = gst_buffer_new();
    gst_buffer_append_memory(buf, mem);
    if (!default_layout)
      gst_buffer_add_video_meta_full(buf, GST_VIDEO_FRAME_FLAG_NONE, GST_VIDEO_INFO_FORMAT(info),
                                     GST_VIDEO_INFO_WIDTH(info), GST_VIDEO_INFO_HEIGHT(info),
                                     GST_VIDEO_INFO_N_PLANES(info), offsets, strides);
    return buf;
  }
  const G2gBridgePlanes *planes = &self->out_planes;
  if (self->have_out_planes &&
      !layout_is_tight(planes, self->out_info.offset, self->out_info.stride)) {
    if (out->len < planes->size)
      return NULL;
    GstBuffer *src = wrap_frame((guint8 *)out->data, GST_MEMORY_FLAG_READONLY, &self->out_info,
                                planes);
    gsize size = GST_VIDEO_INFO_SIZE(&self->out_info);
    GstBuffer *dst = gst_buffer_new_allocate(NULL, size, NULL);
    /* the row padding would otherwise carry whatever the allocator left there */
    gboolean copied = dst && gst_buffer_memset(dst, 0, 0, size) == size &&
                      copy_frame(dst, src, &self->out_info);
    gst_buffer_unref(src);
    if (!copied && dst) {
      gst_buffer_unref(dst);
      dst = NULL;
    }
    return dst;
  }
  GstBuffer *buf = gst_buffer_new_allocate(NULL, out->len, NULL);
  if (!buf)
    return NULL;
  GstMapInfo map;
  if (!gst_buffer_map(buf, &map, GST_MAP_WRITE)) {
    gst_buffer_unref(buf);
    return NULL;
  }
  memcpy(map.data, out->data, MIN(map.size, out->len));
  gst_buffer_unmap(buf, &map);
  return buf;
}

/* Produce one output buffer from the queued input. Overriding `generate_output`
 * (rather than transform/transform_ip) lets the output buffer differ from the
 * input in size *and* memory kind (system or dma-buf), which the in-place model
 * cannot express. */
static GstFlowReturn gst_glass2glass_generate_output(GstBaseTransform *base, GstBuffer **outbuf) {
  GstGlass2Glass *self = GST_GLASS2GLASS(base);
  *outbuf = NULL;
  if (!self->bridge)
    return GST_FLOW_NOT_NEGOTIATED;

  /* Take ownership of the buffer the default submit_input_buffer queued. */
  GstBuffer *in = base->queued_buf;
  base->queued_buf = NULL;
  if (!in)
    return GST_BASE_TRANSFORM_FLOW_DROPPED;

  if (!push_input(self, in)) {
    GST_ERROR_OBJECT(self, "sub-graph did not accept the buffer (stalled)");
    gst_buffer_unref(in);
    return GST_FLOW_ERROR;
  }

  G2gOut out;
  int r = g2g_bridge_pull_buf(self->bridge, &out);
  if (r < 0) {
    gst_buffer_unref(in);
    /* -1 EOS, -2 a memory domain g2g has no download converter for (wgpu and
     * CUDA frames are downloaded inside the sub-graph), -3 the sub-graph failed. */
    if (r == -3)
      GST_ELEMENT_ERROR(self, STREAM, FAILED, ("g2g sub-graph failed (reason logged above)"),
                        ("fragment=\"%s\"", self->fragment ? self->fragment : "identity"));
    return (r == -1) ? GST_FLOW_EOS : GST_FLOW_ERROR;
  }

  GstBuffer *result = wrap_output(self, &out);
  g2g_bridge_out_release(&out);
  if (!result) {
    gst_buffer_unref(in);
    return GST_FLOW_ERROR;
  }
  /* Carry timestamps / flags from the input. */
  gst_buffer_copy_into(result, in, GST_BUFFER_COPY_TIMESTAMPS | GST_BUFFER_COPY_FLAGS, 0, -1);
  gst_buffer_unref(in);
  *outbuf = result;
  return GST_FLOW_OK;
}

static gboolean gst_glass2glass_stop(GstBaseTransform *base) {
  GstGlass2Glass *self = GST_GLASS2GLASS(base);
  if (self->bridge) {
    g2g_bridge_destroy(self->bridge);
    self->bridge = NULL;
  }
  return TRUE;
}

/* ---- lifecycle ----------------------------------------------------------- */
static void gst_glass2glass_finalize(GObject *object) {
  GstGlass2Glass *self = GST_GLASS2GLASS(object);
  if (self->bridge)
    g2g_bridge_destroy(self->bridge);
  if (self->dmabuf_alloc)
    gst_object_unref(self->dmabuf_alloc);
  g_free(self->fragment);
  g_free(self->input_caps);
  g_free(self->output_caps);
  G_OBJECT_CLASS(gst_glass2glass_parent_class)->finalize(object);
}

static void gst_glass2glass_class_init(GstGlass2GlassClass *klass) {
  GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
  GstBaseTransformClass *base_class = GST_BASE_TRANSFORM_CLASS(klass);

  gobject_class->set_property = gst_glass2glass_set_property;
  gobject_class->get_property = gst_glass2glass_get_property;
  gobject_class->finalize = gst_glass2glass_finalize;

  g_object_class_install_property(
      gobject_class, PROP_FRAGMENT,
      g_param_spec_string("fragment", "Fragment",
                          "g2g sub-pipeline run as appsrc ! <fragment> ! appsink",
                          "identity", G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property(
      gobject_class, PROP_INPUT_CAPS,
      g_param_spec_string("input-caps", "Input caps",
                          "Override the input caps handed to the sub-graph "
                          "(default: the negotiated sink caps, serialized)",
                          NULL, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
  g_object_class_install_property(
      gobject_class, PROP_OUTPUT_CAPS,
      g_param_spec_string("output-caps", "Output caps",
                          "Caps the sub-graph produces, when it rescales or "
                          "reformats (e.g. a videoscale fragment). Unset means "
                          "the fragment preserves caps and size (in-place).",
                          NULL, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_set_static_metadata(
      element_class, "glass2glass bridge", "Filter/Effect",
      "Runs an embedded glass2glass sub-graph", "glass2glass");
  gst_element_class_add_static_pad_template(element_class, &sink_template);
  gst_element_class_add_static_pad_template(element_class, &src_template);

  base_class->transform_caps = gst_glass2glass_transform_caps;
  base_class->get_unit_size = gst_glass2glass_get_unit_size;
  base_class->set_caps = gst_glass2glass_set_caps;
  /* `generate_output` (not transform/transform_ip) so the output buffer can
   * differ from the input in size and memory kind (system or dma-buf). */
  base_class->generate_output = gst_glass2glass_generate_output;
  base_class->stop = gst_glass2glass_stop;
}

static void gst_glass2glass_init(GstGlass2Glass *self) {
  self->fragment = NULL;
  self->input_caps = NULL;
  self->output_caps = NULL;
  self->bridge = NULL;
  self->in_stride = 0;
  self->have_in_planes = FALSE;
  self->have_out_info = FALSE;
  self->have_out_planes = FALSE;
  self->dmabuf_alloc = NULL;
}

size_t g2g_bridge_planes_size(void);
size_t g2g_bridge_planes_size(void) { return sizeof(G2gBridgePlanes); }

/* ---- plugin init --------------------------------------------------------- */
/* The plugin entry points (`gst_plugin_glass2glass_get_desc` / `_register`) and
 * the `GstPluginDesc` are authored in Rust (src/ffi.rs): rustc exports only its
 * own `#[no_mangle]` symbols from a cdylib, localizing anything pulled from a C
 * archive, so a C `GST_PLUGIN_DEFINE` descriptor would not be visible to
 * GStreamer's loader. This function is the `plugin_init` the Rust descriptor
 * points at; it does the actual element registration. Reached via a function
 * pointer, so it need not be exported. */
gboolean glass2glass_plugin_init(GstPlugin *plugin);
gboolean glass2glass_plugin_init(GstPlugin *plugin) {
  GST_DEBUG_CATEGORY_INIT(glass2glass_debug, "glass2glass", 0, "glass2glass bridge");
  return gst_element_register(plugin, "glass2glass", GST_RANK_NONE, GST_TYPE_GLASS2GLASS);
}
