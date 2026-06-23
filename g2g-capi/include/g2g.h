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

#ifdef __cplusplus
}
#endif

#endif /* G2G_H */
