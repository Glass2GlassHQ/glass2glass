/* A glass2glass plugin written in plain C against g2g_plugin_v2.h.
 *
 * The element is `cpasser`: it counts data frames and forwards them unchanged,
 * and exposes `count` (read-only) and `enabled` (drops frames when false).
 *
 * The point is not the element. It is that this file links against nothing from
 * glass2glass, knows no Rust type, and still negotiates, receives frames with
 * their payload ownership, awaits downstream backpressure through the poll
 * boundary, and answers property calls.
 *
 * Compiled and loaded by g2g-plugins/tests/plugin_c_abi.rs.
 */

#include <stdlib.h>
#include <string.h>

#include "g2g_plugin_v2.h"

/* --- element instance --------------------------------------------------- */

typedef struct {
    uint64_t seen;
    int enabled;
} Counter;

static void *counter_create(void) {
    Counter *c = calloc(1, sizeof *c);
    if (c) {
        c->enabled = 1;
    }
    return c;
}

static void counter_destroy(void *elem) { free(elem); }

/* --- one in-flight push ------------------------------------------------- */

/* The state behind the future `process` returns. It owns the packet until the
 * host takes it, which is why `push_drop` releases whatever is left: a cancelled
 * push must not leak the frame. */
typedef struct {
    G2gPacket packet;
    G2gOutputSink sink;
} PushState;

static G2gPoll poll_ready(G2gStatus status) {
    G2gPoll poll;
    memset(&poll, 0, sizeof poll);
    poll.tag = G2G_POLL_READY;
    poll.value = status;
    return poll;
}

static G2gPoll push_poll(void *state, G2gFfiContext *cx) {
    PushState *s = (PushState *)state;
    if (!s) {
        return poll_ready(G2G_STATUS_ERROR);
    }
    if (s->packet.tag == G2G_PACKET_NONE) {
        /* Nothing to send (a dropped frame, or the host already took it). */
        return poll_ready(G2G_STATUS_OK);
    }
    return s->sink.vtable->poll_push(s->sink.ctx, cx, &s->packet);
}

static void push_drop(void *state) {
    PushState *s = (PushState *)state;
    if (!s) {
        return;
    }
    /* Anything still here was never handed over, so releasing it is this side's
     * job. The host clears the tag when it takes the packet. */
    if (s->packet.tag == G2G_PACKET_DATA_FRAME && s->packet.frame.free) {
        s->packet.frame.free(s->packet.frame.free_user);
    }
    free(s);
}

static void release_frame(G2gPacket *packet) {
    if (packet->tag == G2G_PACKET_DATA_FRAME && packet->frame.free) {
        packet->frame.free(packet->frame.free_user);
    }
    packet->tag = G2G_PACKET_NONE;
}

static G2gFuture counter_process(void *elem, G2gPacket packet, G2gOutputSink out) {
    Counter *c = (Counter *)elem;
    G2gFuture future;
    PushState *state;

    if (packet.tag == G2G_PACKET_DATA_FRAME) {
        c->seen += 1;
        if (!c->enabled) {
            release_frame(&packet);
        }
    } else if (packet.tag == G2G_PACKET_EOS) {
        /* The runner emits the pipeline's single EOS; a transform must not push
         * one of its own. */
        packet.tag = G2G_PACKET_NONE;
    }

    state = (PushState *)calloc(1, sizeof *state);
    if (state) {
        state->packet = packet;
        state->sink = out;
    } else {
        release_frame(&packet);
    }

    future.state = state;
    future.poll = push_poll;
    future.drop = push_drop;
    return future;
}

/* --- properties --------------------------------------------------------- */

static int name_is(G2gStr name, const char *want) {
    size_t len = strlen(want);
    return name.len == len && name.ptr && memcmp(name.ptr, want, len) == 0;
}

static G2gStatus counter_set_property(void *elem, G2gStr name, const G2gPropValue *value) {
    Counter *c = (Counter *)elem;
    if (!value) {
        return G2G_STATUS_PROPERTY_VALUE;
    }
    if (name_is(name, "enabled")) {
        if (value->kind != G2G_PROP_BOOL) {
            return G2G_STATUS_PROPERTY_VALUE;
        }
        c->enabled = value->body.boolean != 0;
        return G2G_STATUS_OK;
    }
    if (name_is(name, "count")) {
        return G2G_STATUS_PROPERTY_VALUE; /* read-only */
    }
    return G2G_STATUS_PROPERTY_UNKNOWN;
}

static G2gStatus counter_get_property(void *elem, G2gStr name, G2gPropValue *out) {
    Counter *c = (Counter *)elem;
    if (!out) {
        return G2G_STATUS_ERROR;
    }
    memset(out, 0, sizeof *out);
    if (name_is(name, "count")) {
        out->kind = G2G_PROP_UINT;
        out->body.uinteger = c->seen;
        return G2G_STATUS_OK;
    }
    if (name_is(name, "enabled")) {
        out->kind = G2G_PROP_BOOL;
        out->body.boolean = c->enabled ? 1u : 0u;
        return G2G_STATUS_OK;
    }
    return G2G_STATUS_PROPERTY_UNKNOWN;
}

/* --- tables ------------------------------------------------------------- */

#define G2G_STR(literal) \
    { (const uint8_t *)(literal), sizeof(literal) - 1 }

static const G2gPropertySpec PROPERTIES[] = {
    {G2G_STR("count"), G2G_PROP_UINT, 1u, 0u, 0u, G2G_STR("data frames seen so far"),
     {NULL, 0}},
    {G2G_STR("enabled"), G2G_PROP_BOOL, 1u, 1u, 0u,
     G2G_STR("forward frames; drop them when false"), G2G_STR("true")},
};

/* `struct_size` deliberately stops before the reserved slots, so this element
 * exercises the host's short-vtable path: it reads the prefix this plugin
 * actually wrote and zero-fills the rest. */
static const G2gElementVtable VTABLE = {
    offsetof(G2gElementVtable, reserved),
    1u,
    counter_process,
    counter_destroy,
    NULL, /* configure_pipeline: accept whatever was negotiated */
    NULL, /* configure_output */
    counter_set_property,
    counter_get_property,
    {NULL, NULL, NULL, NULL, NULL, NULL},
};

static G2gStatus register_elements(const G2gRegistrar *registrar) {
    G2gElementRegistration element;
    if (!registrar) {
        return G2G_STATUS_ERROR;
    }
    memset(&element, 0, sizeof element);
    element.struct_size = sizeof element;
    element.kind = G2G_ELEMENT_TRANSFORM;
    element.name = (G2gStr)G2G_STR("cpasser");
    element.metadata.long_name = (G2gStr)G2G_STR("C counting filter");
    element.metadata.klass = (G2gStr)G2G_STR("Filter/Effect/Video");
    element.metadata.description =
        (G2gStr)G2G_STR("Counts data frames and forwards them unchanged (written in C).");
    element.metadata.author = (G2gStr)G2G_STR("third-party");
    /* Empty caps sets: accepts anything, produces what it was given. */
    element.properties = PROPERTIES;
    element.property_count = sizeof PROPERTIES / sizeof PROPERTIES[0];
    element.vtable = &VTABLE;
    element.create = counter_create;
    return registrar->register_element(registrar->ctx, &element);
}

static const G2gCapability CAPABILITIES[] = {
    {G2G_ELEMENT_TRANSFORM, 0u, G2G_STR("cpasser")},
};

G2G_PLUGIN_EXPORT const G2gPluginDescriptor g2g_plugin_v2_descriptor = {
    G2G_V2_MAGIC,
    G2G_V2_ABI_VERSION,
    sizeof(G2gPluginDescriptor),
    G2G_STR("g2g-c-example-plugin"),
    G2G_STR("0.1.0"),
    CAPABILITIES,
    sizeof CAPABILITIES / sizeof CAPABILITIES[0],
    register_elements,
    {NULL, NULL, NULL, NULL},
};

/* --- layout probe ------------------------------------------------------- */

/* Reports sizeof for every ABI struct, in the order the Rust test expects, so
 * a drift between this header and `g2g-plugin/src/abi/` fails a test instead of
 * corrupting memory at runtime. */
G2G_PLUGIN_EXPORT void g2g_c_plugin_layout(size_t *out, size_t count) {
    const size_t sizes[] = {
        sizeof(G2gStr),
        sizeof(G2gDim),
        sizeof(G2gRate),
        sizeof(G2gCaps),
        sizeof(G2gCapsSet),
        sizeof(G2gFrame),
        sizeof(G2gPacket),
        sizeof(G2gPoll),
        sizeof(G2gFuture),
        sizeof(G2gOutputSinkVtable),
        sizeof(G2gOutputSink),
        sizeof(G2gPropStr),
        sizeof(G2gPropValue),
        sizeof(G2gPropertySpec),
        sizeof(G2gElementMetadata),
        sizeof(G2gElementVtable),
        sizeof(G2gElementRegistration),
        sizeof(G2gRegistrar),
        sizeof(G2gCapability),
        sizeof(G2gPluginDescriptor),
    };
    size_t i;
    size_t n = sizeof sizes / sizeof sizes[0];
    if (!out) {
        return;
    }
    for (i = 0; i < count; i++) {
        out[i] = i < n ? sizes[i] : 0;
    }
}
