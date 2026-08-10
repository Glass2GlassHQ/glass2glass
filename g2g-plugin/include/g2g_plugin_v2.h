/* glass2glass plugin ABI, generation 2.
 *
 * A v2 plugin is a shared library that exports one data symbol,
 * `g2g_plugin_v2_descriptor`, of type G2gPluginDescriptor. The host reads and
 * validates it before calling any of the plugin's code, so what the descriptor
 * declares is a promise made in advance, and the host holds the plugin to it.
 *
 * Everything here is plain C: structs, integers, pointer+length pairs, and
 * function pointers. That is the point. It is the same surface the Rust SDK
 * emits, so a plugin written in C loads into the same host as one built by a
 * different rustc.
 *
 * Hand-written, and kept in step with `g2g-plugin/src/abi/` by the layout test
 * in `g2g-plugins/tests/plugin_c_abi.rs`: the C fixture reports sizeof for every
 * struct below and the test compares each one against the Rust type.
 *
 * License: MPL-2.0, like the rest of glass2glass.
 */

#ifndef G2G_PLUGIN_V2_H
#define G2G_PLUGIN_V2_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- version handshake -------------------------------------------------- */

/* The ASCII bytes "G2GABIv2" read as a little-endian uint64. First field of the
 * descriptor, so a wrong or garbage symbol fails the very first check. */
#define G2G_V2_MAGIC 0x3276494241473247ULL

/* The ABI generation this header describes. A semantic change to any existing
 * field bumps this; the host refuses anything else. */
#define G2G_V2_ABI_VERSION 2u

/* Growing a struct inside one ABI generation works two ways, and a plugin
 * should understand both:
 *
 *  - Every versioned struct starts with its own `struct_size`. The host copies
 *    min(plugin_size, host_size) bytes into a zeroed local, so a plugin built
 *    against an older, smaller header leaves the host's newer fields null and
 *    the host substitutes its own default. Always set `struct_size` with
 *    sizeof() on the struct as this header declares it. Never lie about it: the
 *    host trusts it as the count of bytes you actually wrote.
 *
 *  - The trailing `reserved` function-pointer slots let a future entry point
 *    appear without the size changing, so an older host reading a newer plugin
 *    simply ignores it. Set them all to NULL.
 */

/* --- status codes ------------------------------------------------------- */

typedef int32_t G2gStatus;

#define G2G_STATUS_OK 0
#define G2G_STATUS_ERROR (-1)            /* unclassified failure */
#define G2G_STATUS_CAPS_MISMATCH (-2)    /* the caps offered are not acceptable */
#define G2G_STATUS_NOT_CONFIGURED (-3)   /* data before a successful configure */
#define G2G_STATUS_UNSUPPORTED_DOMAIN (-4)
#define G2G_STATUS_SHUTDOWN (-5)
#define G2G_STATUS_PROPERTY_UNKNOWN (-6) /* no property by that name */
#define G2G_STATUS_PROPERTY_VALUE (-7)   /* wrong kind, or out of range */

/* --- strings ------------------------------------------------------------ */

/* A UTF-8 string as pointer + length. NOT NUL-terminated: `len` is
 * authoritative. The host bounds `len`, requires a non-null `ptr` whenever
 * `len > 0`, checks the bytes are valid UTF-8, and copies anything it keeps. */
typedef struct {
    const uint8_t *ptr;
    size_t len;
} G2gStr;

/* --- caps --------------------------------------------------------------- */

#define G2G_DIM_ANY 0u
#define G2G_DIM_FIXED 1u /* the value is in `min` */
#define G2G_DIM_RANGE 2u /* inclusive [min, max] */

typedef struct {
    uint32_t kind; /* G2G_DIM_* */
    uint32_t min;
    uint32_t max;
} G2gDim;

/* Framerate in Q16 fixed-point frames per second. */
typedef struct {
    uint32_t kind; /* G2G_DIM_* */
    uint32_t min_q16;
    uint32_t max_q16;
} G2gRate;

#define G2G_INTERLACE_ANY 0u
#define G2G_INTERLACE_PROGRESSIVE 1u
#define G2G_INTERLACE_INTERLEAVED 2u

/* Format / codec codes are frozen: assigned once and never reused. The full
 * tables live in `g2g-plugin/src/abi/caps.rs`; the common ones are here. A code
 * the host does not know is refused, never reinterpreted. */
#define G2G_RAW_VIDEO_NV12 1u
#define G2G_RAW_VIDEO_I420 2u
#define G2G_RAW_VIDEO_RGBA8 3u
#define G2G_RAW_VIDEO_BGRA8 4u
#define G2G_RAW_VIDEO_YUYV 5u

#define G2G_VIDEO_CODEC_H264 1u
#define G2G_VIDEO_CODEC_H265 2u
#define G2G_VIDEO_CODEC_AV1 3u

#define G2G_AUDIO_AAC 1u
#define G2G_AUDIO_OPUS 2u
#define G2G_AUDIO_PCM_S16LE 9u
#define G2G_AUDIO_PCM_F32LE 10u

#define G2G_TEXT_UTF8 1u

typedef struct {
    uint32_t format; /* G2G_RAW_VIDEO_* */
    G2gDim width;
    G2gDim height;
    G2gRate framerate;
    uint32_t interlace; /* G2G_INTERLACE_* */
} G2gRawVideoCaps;

typedef struct {
    uint32_t codec; /* G2G_VIDEO_CODEC_* */
    G2gDim width;
    G2gDim height;
    G2gRate framerate;
} G2gCompressedVideoCaps;

typedef struct {
    uint32_t format;   /* G2G_AUDIO_* */
    uint32_t channels; /* 0 means "any / unknown"; at most 255 */
    uint32_t sample_rate;
} G2gAudioCaps;

typedef struct {
    uint32_t encoding;
} G2gByteStreamCaps;

typedef struct {
    uint32_t format; /* G2G_TEXT_* */
} G2gTextCaps;

typedef union {
    G2gRawVideoCaps raw_video;
    G2gCompressedVideoCaps compressed_video;
    G2gAudioCaps audio;
    G2gByteStreamCaps byte_stream;
    G2gTextCaps text;
} G2gCapsBody;

#define G2G_CAPS_NONE 0u
#define G2G_CAPS_RAW_VIDEO 1u
#define G2G_CAPS_COMPRESSED_VIDEO 2u
#define G2G_CAPS_AUDIO 3u
#define G2G_CAPS_BYTE_STREAM 4u
#define G2G_CAPS_TEXT 5u

/* `tag` is the sole authority on which union member is live. Set `reserved` to
 * zero. */
typedef struct {
    uint32_t tag; /* G2G_CAPS_* */
    uint32_t reserved;
    G2gCapsBody body;
} G2gCaps;

/* An ordered set of caps alternatives, highest preference first: the data form
 * of a pad template. `count == 0` means "any" on a sink pad and "whatever the
 * input was" on a source pad. */
typedef struct {
    const G2gCaps *alternatives;
    size_t count;
} G2gCapsSet;

/* --- frames and packets ------------------------------------------------- */

/* One System-memory frame. There is no GPU memory in v2: a v2 element only ever
 * sees plain bytes, and the host inserts a converter in front of a GPU-resident
 * producer.
 *
 * OWNERSHIP TRAVELS WITH THE STRUCT. Whoever receives a G2gFrame owns the
 * payload and must call `free(free_user)` exactly once, unless `free` is NULL
 * (static bytes, nothing to release). Forwarding a frame downstream passes that
 * obligation on; dropping one means calling `free` yourself. */
typedef struct {
    const uint8_t *data;
    size_t len;
    void (*free)(void *user);
    void *free_user;
    uint64_t pts_ns;
    uint64_t dts_ns;
    uint64_t duration_ns;
    uint64_t capture_ns;
    uint64_t arrival_ns;
    uint64_t sequence;
    uint32_t keyframe; /* non-zero starts an independently decodable unit */
    uint32_t reserved;
} G2gFrame;

#define G2G_PACKET_NONE 0u         /* empty slot, or the callee took what was here */
#define G2G_PACKET_CAPS_CHANGED 1u /* `caps` is live */
#define G2G_PACKET_DATA_FRAME 2u   /* `frame` is live */
#define G2G_PACKET_EOS 3u          /* flush buffered output; do NOT push an EOS */
#define G2G_PACKET_FLUSH 4u        /* discard buffered data and reset */

/* The host handles segments itself and never delivers a tick, so those two
 * pipeline packets have no tag here. */
typedef struct {
    uint32_t tag; /* G2G_PACKET_* */
    uint32_t reserved;
    G2gCaps caps;
    G2gFrame frame;
} G2gPacket;

/* --- the poll boundary -------------------------------------------------- */

/* An async task context, borrowed for the duration of a call. Opaque: pass it
 * through to `poll_push`, never dereference it. */
typedef struct G2gFfiContext G2gFfiContext;

#define G2G_POLL_READY 0u
#define G2G_POLL_PENDING 1u
#define G2G_POLL_PANICKED 2u /* the callee's poll panicked; treat as a failure */

/* `value` is meaningful only when `tag` is G2G_POLL_READY. */
typedef struct {
    uint8_t tag; /* G2G_POLL_* */
    uint8_t padding_[3];
    G2gStatus value;
} G2gPoll;

/* An asynchronous result, in the layout the `async-ffi` crate defines.
 *
 * The host polls it by calling `poll(state, cx)` until it answers non-pending,
 * then calls `drop(state)` exactly once. It may also call `drop` without ever
 * reaching ready, which is how a cancelled operation looks: release whatever
 * the state owns, including any frame payload still in it. */
typedef struct {
    void *state;
    G2gPoll (*poll)(void *state, G2gFfiContext *cx);
    void (*drop)(void *state);
} G2gFuture;

/* --- pushing downstream ------------------------------------------------- */

typedef struct {
    uint32_t struct_size;
    uint32_t version;

    /* Drive one packet toward the next element.
     *
     * `packet` is an in/out slot. The host takes ownership of what is in it on
     * the FIRST call and rewrites the tag to G2G_PACKET_NONE; if the answer is
     * pending, call again with the same (now empty) slot until it is not. So
     * the tag is not how you tell whether there is work left: the host tracks
     * that. Ready + G2G_STATUS_OK means downstream accepted the packet. */
    G2gPoll (*poll_push)(void *ctx, G2gFfiContext *cx, G2gPacket *packet);

    void (*reserved[4])(void);
} G2gOutputSinkVtable;

typedef struct {
    void *ctx;
    const G2gOutputSinkVtable *vtable;
} G2gOutputSink;

/* --- properties --------------------------------------------------------- */

#define G2G_PROP_NONE 0u
#define G2G_PROP_BOOL 1u
#define G2G_PROP_INT 2u     /* int64 */
#define G2G_PROP_UINT 3u    /* uint64 */
#define G2G_PROP_DOUBLE 4u
#define G2G_PROP_FRACTION 5u
#define G2G_PROP_STR 6u
/* There is deliberately no flag-set kind: its value is a list of strings, and
 * the ownership rules would double this surface for one rare property shape.
 * An element that declares one is refused at registration. */

typedef struct {
    int32_t num;
    int32_t den; /* never zero */
} G2gFraction;

/* A string property payload.
 *
 * `free == NULL` means BORROWED, valid only for the duration of the call that
 * carried it: copy anything you keep. `free != NULL` means you OWN it and must
 * call `free(free_user)` once. The host lends you a borrowed string in
 * set_property; you hand back an owned one from get_property. */
typedef struct {
    const uint8_t *ptr;
    size_t len;
    void (*free)(void *user);
    void *free_user;
} G2gPropStr;

typedef union {
    uint32_t boolean; /* zero false, non-zero true */
    int64_t integer;
    uint64_t uinteger;
    double real;
    G2gFraction fraction;
    G2gPropStr string;
} G2gPropValueBody;

typedef struct {
    uint32_t kind; /* G2G_PROP_* */
    uint32_t reserved;
    G2gPropValueBody body;
} G2gPropValue;

/* Static description of one property, as `gst-inspect` would print it. */
typedef struct {
    G2gStr name; /* lowercase ASCII letters, digits, '-', '_'; starts with a letter */
    uint32_t kind;
    uint32_t readable;
    uint32_t writable;
    uint32_t reserved;
    G2gStr blurb;
    G2gStr default_value; /* parseable for `kind`, or empty for none */
} G2gPropertySpec;

/* --- elements ----------------------------------------------------------- */

typedef struct {
    G2gStr long_name;
    G2gStr klass;
    G2gStr description;
    G2gStr author;
} G2gElementMetadata;

/* Per-instance entry points.
 *
 * `process` and `destroy` are REQUIRED. The rest may be NULL, and the host then
 * uses its own default: accept the caps, ignore the output caps, report no such
 * property. A shorter `struct_size` from an older header defaults the same way.
 *
 * Thread contract: the host owns an instance exclusively and never calls into
 * it from two threads at once, but it MAY move it between threads between
 * calls. An element whose state is bound to one thread (a GL context, a COM
 * apartment) is outside this ABI's contract. */
typedef struct {
    uint32_t struct_size;
    uint32_t version;

    /* Accept the negotiated input caps. Return G2G_STATUS_CAPS_MISMATCH to
     * reject. Writing a caps value with a tag other than G2G_CAPS_NONE through
     * `refixate` asks the solver to retry with that proposal. */
    G2gStatus (*configure_pipeline)(void *elem, const G2gCaps *caps, G2gCaps *refixate);

    /* Receive this element's own negotiated OUTPUT caps. */
    G2gStatus (*configure_output)(void *elem, const G2gCaps *caps);

    /* Handle one packet, pushing any output through `out`. REQUIRED.
     *
     * Ownership of the packet's payload transfers to you. The returned future
     * borrows both `elem` and `out`; the host keeps them valid until it drops
     * the future, and does not touch `elem` while the future is alive. */
    G2gFuture (*process)(void *elem, G2gPacket packet, G2gOutputSink out);

    /* `value` is borrowed for the call. */
    G2gStatus (*set_property)(void *elem, G2gStr name, const G2gPropValue *value);

    /* Write the value through `out` and return G2G_STATUS_OK, or
     * G2G_STATUS_PROPERTY_UNKNOWN. */
    G2gStatus (*get_property)(void *elem, G2gStr name, G2gPropValue *out);

    /* Destroy an instance `create` built. REQUIRED. Called once per instance. */
    void (*destroy)(void *elem);

    void (*reserved[6])(void);
} G2gElementVtable;

#define G2G_ELEMENT_TRANSFORM 1u /* one in, one out */
#define G2G_ELEMENT_SINK 2u      /* terminal */

/* One element handed to the host's registrar.
 *
 * Every pointer in here must stay valid for the life of the process: the host
 * builds instances from these fields long after `register` returned, and it
 * never unloads a plugin it accepted. Statics are the natural answer. */
typedef struct {
    uint32_t struct_size;
    uint32_t kind; /* G2G_ELEMENT_*; must match what the descriptor declared */
    G2gStr name;   /* must match a declared capability, character set as above */
    G2gElementMetadata metadata;

    /* Caps accepted on the input pad; empty means any. */
    G2gCapsSet sink_caps;
    /* Caps produced. Empty means "the input unchanged", which is what a
     * pass-through declares. A non-empty set must be fully concrete: an "any"
     * dimension or framerate here cannot be fixated and is refused. */
    G2gCapsSet source_caps;

    const G2gPropertySpec *properties;
    size_t property_count;

    const G2gElementVtable *vtable; /* never NULL */
    void *(*create)(void);          /* REQUIRED; NULL return means failure */

    void (*reserved[4])(void);
} G2gElementRegistration;

/* The host-owned object you register through. */
typedef struct {
    uint32_t struct_size;
    uint32_t version;
    void *ctx;

    /* Returns G2G_STATUS_OK, or an error when the host refuses the element (an
     * undeclared name, a malformed registration, too many elements). A refusal
     * fails the whole load, so there is nothing to unwind: return the status. */
    G2gStatus (*register_element)(void *ctx, const G2gElementRegistration *element);

    void (*reserved[4])(void);
} G2gRegistrar;

/* --- the descriptor ----------------------------------------------------- */

/* One thing the plugin declares it will register. The host reads this list
 * before running any plugin code, and a caller-supplied policy decides whether
 * to allow it. A kind the host does not know is reported to that policy rather
 * than refused outright. */
typedef struct {
    uint32_t kind; /* G2G_ELEMENT_* */
    uint32_t reserved;
    G2gStr name;
} G2gCapability;

/* The static exported as `g2g_plugin_v2_descriptor`.
 *
 * Declare exactly what you will register. The host checks each registration
 * against this list, and one element that is not on it fails the entire load,
 * including the ones that were. */
typedef struct {
    uint64_t magic;       /* G2G_V2_MAGIC */
    uint32_t abi_version; /* G2G_V2_ABI_VERSION */
    uint32_t struct_size;
    G2gStr name;
    G2gStr version;
    const G2gCapability *capabilities;
    size_t capability_count;

    /* Register the declared elements. REQUIRED. Called once, only after the
     * descriptor validates and the host's policy allows it. Must not let an
     * exception or a panic escape. */
    G2gStatus (*register_elements)(const G2gRegistrar *registrar);

    void (*reserved[4])(void);
} G2gPluginDescriptor;

/* The symbol the host looks up. Define it as:
 *
 *   G2G_PLUGIN_EXPORT const G2gPluginDescriptor g2g_plugin_v2_descriptor = { ... };
 */
#define G2G_PLUGIN_DESCRIPTOR_SYMBOL "g2g_plugin_v2_descriptor"

#if defined(_WIN32)
#define G2G_PLUGIN_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define G2G_PLUGIN_EXPORT __attribute__((visibility("default")))
#else
#define G2G_PLUGIN_EXPORT
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* G2G_PLUGIN_V2_H */
