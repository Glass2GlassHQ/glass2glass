/* GStreamer element `glass2glass`: embeds a g2g sub-graph inside a GStreamer
 * pipeline (DESIGN.md §7). This is the thin GObject shell over the Rust
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

extern G2gBridge *g2g_bridge_create(const char *fragment, const char *in_caps,
                                    const char *out_caps);
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
  guint out_height;    /* negotiated output height, for dma-buf output sizing */
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

/* Output buffer size for a given (raw video) caps, needed when the element is
 * not operating in place. */
static gboolean gst_glass2glass_get_unit_size(GstBaseTransform *base, GstCaps *caps, gsize *size) {
  (void)base;
  GstVideoInfo info;
  if (!gst_video_info_from_caps(&info, caps))
    return FALSE;
  *size = GST_VIDEO_INFO_SIZE(&info);
  return TRUE;
}

/* ---- transform vmethods -------------------------------------------------- */
/* Build the sub-graph once caps are fixed. `incaps` describes the buffers the
 * embedded appsrc receives; `outcaps` (== incaps for a preserving fragment) the
 * frames it produces. Records the input stride and output height for the
 * dma-buf import (which carries no byte length of its own). */
static gboolean gst_glass2glass_set_caps(GstBaseTransform *base, GstCaps *incaps,
                                         GstCaps *outcaps) {
  GstGlass2Glass *self = GST_GLASS2GLASS(base);
  if (self->bridge) {
    g2g_bridge_destroy(self->bridge);
    self->bridge = NULL;
  }

  GstVideoInfo ininfo, outinfo;
  self->in_stride = gst_video_info_from_caps(&ininfo, incaps)
                        ? (guint)GST_VIDEO_INFO_PLANE_STRIDE(&ininfo, 0)
                        : 0;
  self->out_height =
      gst_video_info_from_caps(&outinfo, outcaps) ? GST_VIDEO_INFO_HEIGHT(&outinfo) : 0;

  gchar *instr = self->input_caps ? g_strdup(self->input_caps) : gst_caps_to_string(incaps);
  gchar *outstr = gst_caps_to_string(outcaps);
  const char *frag = self->fragment ? self->fragment : "identity";
  self->bridge = g2g_bridge_create(frag, instr, outstr);
  if (!self->bridge)
    GST_ERROR_OBJECT(self, "failed to build g2g sub-graph: fragment=\"%s\" in=\"%s\" out=\"%s\"",
                     frag, instr, outstr);
  g_free(instr);
  g_free(outstr);
  return self->bridge != NULL;
}

/* Push one input buffer into the sub-graph. A dma-buf-backed buffer is imported
 * zero-copy (the fd, not the mapped bytes); any other memory is mapped and its
 * bytes copied in. */
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
  GstMapInfo map;
  if (!gst_buffer_map(in, &map, GST_MAP_READ))
    return FALSE;
  gboolean ok = g2g_bridge_push_buf(self->bridge, map.data, map.size, pts);
  gst_buffer_unmap(in, &map);
  return ok;
}

/* Build the downstream buffer from a pulled frame: a system frame becomes an
 * owned GstBuffer (bytes copied); a dma-buf frame is wrapped zero-copy into a
 * dma-buf GstBuffer (its fd dup'ed, so the g2g frame keeps its own). */
static GstBuffer *wrap_output(GstGlass2Glass *self, const G2gOut *out) {
  if (out->kind == 1) {
    if (!self->dmabuf_alloc)
      self->dmabuf_alloc = gst_dmabuf_allocator_new();
    gsize size = (gsize)out->offset + (gsize)out->stride * self->out_height;
    /* dup: the g2g frame owns `out->fd`; GStreamer's dma-buf memory owns its own. */
    GstMemory *mem = gst_dmabuf_allocator_alloc(self->dmabuf_alloc, dup(out->fd), size);
    if (!mem)
      return NULL;
    GstBuffer *buf = gst_buffer_new();
    gst_buffer_append_memory(buf, mem);
    return buf;
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
    /* -1 EOS, -2 a memory domain the shell cannot hand back (download it in the
     * sub-graph, e.g. end the fragment with a wgpu/cuda download). */
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
  self->out_height = 0;
  self->dmabuf_alloc = NULL;
}

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
