/* glass2glass C ABI (first slice): parse a gst-launch-style pipeline string,
 * run it, and poll the pipeline bus. See g2g-capi/src/lib.rs for the contract.
 *
 * Link against libg2g_capi (cdylib) or libg2g_capi.a (staticlib).
 *
 * Threading: each pipeline runs on its own background thread. The g2g_pipeline_*
 * calls for one handle are not internally synchronized; drive a given handle
 * from one thread at a time.
 */
#ifndef G2G_H
#define G2G_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque pipeline handle. */
typedef struct G2gPipeline G2gPipeline;

/* G2gBusMessage.kind values. */
enum {
  G2G_BUS_STREAM_START = 0,
  G2G_BUS_EOS = 1,
  G2G_BUS_ERROR = 2,
  G2G_BUS_WARNING = 3,
  G2G_BUS_INFO = 4,
  G2G_BUS_STATE_CHANGED = 5,
  G2G_BUS_BUFFERING = 6,
  G2G_BUS_DURATION_CHANGED = 7,
  G2G_BUS_OTHER = 99
};

/* A flattened pipeline bus message.
 *
 * `text` is borrowed from the pipeline and valid only until the next
 * g2g_pipeline_bus_poll() or g2g_pipeline_free() on the same handle; copy it if
 * you need to keep it. `a` / `b` are kind-specific:
 *   G2G_BUS_BUFFERING        -> a = fill percent (0..100)
 *   G2G_BUS_STATE_CHANGED    -> a = new state, b = old state
 *                               (0 Null, 1 Ready, 2 Paused, 3 Playing)
 *   G2G_BUS_DURATION_CHANGED -> a = duration in nanoseconds
 */
typedef struct {
  int kind;
  const char *text;
  uint64_t a;
  uint64_t b;
} G2gBusMessage;

/* Frame counters from g2g_pipeline_wait(). */
typedef struct {
  uint64_t frames_emitted;
  uint64_t frames_consumed;
  uint64_t frames_dropped;
} G2gStats;

/* Parse `description` and start running it on a background thread.
 * Returns NULL on a null/invalid string or parse error; if `err_out` is
 * non-NULL it receives an owned message (free with g2g_string_free). */
G2gPipeline *g2g_pipeline_launch(const char *description, char **err_out);

/* Drain one bus message into *out. Returns 1 if written, 0 if none pending. */
int g2g_pipeline_bus_poll(G2gPipeline *p, G2gBusMessage *out);

/* 1 once the run thread has finished (EOS or error), else 0. */
int g2g_pipeline_is_done(const G2gPipeline *p);

/* Block until the run ends, writing final stats to *out if non-NULL.
 * Returns 0 on a clean run, -1 on pipeline error. Idempotent. */
int g2g_pipeline_wait(G2gPipeline *p, G2gStats *out);

/* Join the run thread (waits for natural EOS, no early cancel yet) and free. */
void g2g_pipeline_free(G2gPipeline *p);

/* Free a string returned by this library (e.g. g2g_pipeline_launch err_out). */
void g2g_string_free(char *s);

/* ---- appsrc / appsink (M233) -------------------------------------------------
 *
 * The application feeds buffers into an `appsrc channel=<name>` and/or receives
 * them from an `appsink channel=<name>`. Register the feed / callback BEFORE
 * launching the pipeline that names the matching channel (the channel name
 * defaults to "default"). Frame bytes are copied across the boundary in v1.
 */

/* Opaque appsrc push handle. */
typedef struct AppSrc G2gAppSrc;

/* appsink per-frame callback. Invoked on the pipeline's run thread with a
 * borrowed view of the frame bytes (copy them if you need to keep them).
 * End-of-stream is delivered as data == NULL, len == 0. Must be thread-safe. */
typedef void (*G2gAppSinkCallback)(const uint8_t *data, size_t len,
                                   uint64_t pts_ns, void *user);

/* Register an appsrc feed under `channel` (NULL -> "default"). */
G2gAppSrc *g2g_appsrc_new(const char *channel);

/* Push `len` bytes (copied) with timestamp `pts_ns`. 1 if accepted, 0 if the
 * feed is full (retry) or the pipeline is gone. */
int g2g_appsrc_push(const G2gAppSrc *src, const uint8_t *data, size_t len,
                    uint64_t pts_ns);

/* Buffer-free callback for a zero-copy lend; invoked once with `user` when the
 * pipeline has fully consumed the lent buffer. */
typedef void (*G2gFreeFunc)(void *user);

/* Push `len` bytes ZERO-COPY: the pipeline reads them in place and calls
 * `free(user)` once the frame is dropped. A NULL `free` lends for the whole run
 * with no reclamation. 1 if accepted; 0 if full/closed (free fires immediately)
 * or `data` is NULL with len>0 (not taken, free not called). A mutating element
 * downstream copies the bytes out first, so the lend stays read-only. */
int g2g_appsrc_push_lend(const G2gAppSrc *src, const uint8_t *data, size_t len,
                         uint64_t pts_ns, G2gFreeFunc free, void *user);

/* Signal end-of-stream; the appsrc emits a final EOS. */
int g2g_appsrc_end_of_stream(const G2gAppSrc *src);

/* Free an appsrc handle (also closes the feed). */
void g2g_appsrc_free(G2gAppSrc *src);

/* Register the per-frame callback for `appsink channel=<name>` (NULL ->
 * "default"). Call before launch. A NULL callback is ignored. */
void g2g_appsink_set_callback(const char *channel, G2gAppSinkCallback cb,
                              void *user);

#ifdef __cplusplus
}
#endif

#endif /* G2G_H */
