/* glass2glass C ABI example: drive a pipeline from C end to end.
 *
 * Builds a two-element pipeline (`appsrc ! appsink`), pushes a few synthetic
 * RGBA frames in, pulls them back out zero-copy, polls the bus, and waits for
 * the final frame counters. This is the smallest realistic shape of an
 * application embedding g2g through the language-neutral C waist; see
 * ../../g2g-capi/include/g2g.h for the full contract.
 *
 * Build & run: see README.md (or `make run` in this directory).
 */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "g2g.h"

#define WIDTH 2
#define HEIGHT 2
#define FRAME_BYTES (WIDTH * HEIGHT * 4) /* RGBA */
#define NUM_FRAMES 3

int main(void) {
    /* Register the application endpoints BEFORE launching: the pipeline binds
     * its `appsrc`/`appsink` elements to these channels by name at parse time. */
    G2gAppSrc *src = g2g_appsrc_new("in");
    G2gAppSink *sink = g2g_appsink_new("out");

    char *err = NULL;
    G2gPipeline *p = g2g_pipeline_launch(
        "appsrc channel=in caps=video/x-raw,format=RGBA,"
        "width=2,height=2,framerate=30/1 ! appsink channel=out",
        &err);
    if (p == NULL) {
        fprintf(stderr, "launch failed: %s\n", err ? err : "(unknown)");
        g2g_string_free(err);
        return 1;
    }

    /* Push NUM_FRAMES solid-color frames, each byte = the frame index, with a
     * 33 ms (30 fps) presentation timestamp step. */
    uint8_t frame[FRAME_BYTES];
    for (int i = 0; i < NUM_FRAMES; i++) {
        memset(frame, i + 1, sizeof frame);
        uint64_t pts_ns = (uint64_t)i * 33333333u;
        while (!g2g_appsrc_push(src, frame, sizeof frame, pts_ns)) {
            /* feed full: a real app would yield; here the queue is ample */
        }
    }
    g2g_appsrc_end_of_stream(src);

    /* Pull every frame back. A pulled sample owns its bytes (zero-copy, the same
     * buffer that flowed through the pipeline) until g2g_sample_free. */
    int pulled = 0;
    G2gSample *s = NULL;
    while (g2g_appsink_pull(sink, &s)) {
        const uint8_t *data = g2g_sample_data(s);
        size_t len = g2g_sample_len(s);
        uint64_t pts = g2g_sample_pts(s);
        printf("frame %d: %zu bytes, first=0x%02X, pts=%" PRIu64 " ns\n",
               pulled, len, data[0], pts);
        g2g_sample_free(s);
        pulled++;
    }

    /* Drain any bus messages the run posted (stream-start, eos, errors). */
    G2gBusMessage msg;
    while (g2g_pipeline_bus_poll(p, &msg)) {
        printf("bus: kind=%d%s%s\n", msg.kind, msg.text ? " " : "",
               msg.text ? msg.text : "");
    }

    /* Wait for the run to finish and read the frame counters. */
    G2gStats stats;
    int rc = g2g_pipeline_wait(p, &stats);
    printf("done rc=%d: emitted=%" PRIu64 " consumed=%" PRIu64 " dropped=%" PRIu64
           " (pulled %d)\n",
           rc, stats.frames_emitted, stats.frames_consumed, stats.frames_dropped,
           pulled);

    g2g_appsrc_free(src);
    g2g_appsink_free(sink);
    g2g_pipeline_free(p);

    return (rc == 0 && pulled == NUM_FRAMES) ? 0 : 1;
}
