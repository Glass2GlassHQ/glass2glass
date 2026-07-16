/* Host-side proof for the C seam ABI (M650): a real C caller that IS the
 * peripheral. It registers a C capture callback (the board's mic/DMA, here the
 * library's shared reference ramp so the input is byte-identical to the Rust
 * reference) and a C send callback (the board's network stack, here a checksum),
 * then drives the pipeline one frame at a time from a plain C loop, exactly as
 * an MCU superloop would. The C-seam wire checksum must equal the pipeline's own
 * Rust reference over the same input, proving the C capture/send seams add and
 * drop nothing. This is the zero-Rust driver-integration path a C/RTOS shop uses.
 */
#include "include/g2g_cffi.h"

#include <stdint.h>
#include <stdio.h>

#define FRAMES 25

/* ── C capture seam: fill each frame from the shared reference ramp ──────────── */
static uint32_t cap_offset = 0;

static intptr_t harness_capture(void *ctx, uint8_t *buf, size_t len) {
    (void)ctx;
    g2g_audio_egress_fill_ramp(buf, len, cap_offset);
    cap_offset += (uint32_t)(len / 2); /* one S16 sample per 2 bytes */
    return (intptr_t)len;
}

/* ── C send seam: the board's network stack, here a wire checksum ───────────── */
static uint64_t send_sum = 0;
static uint32_t send_packets = 0;

static int32_t harness_send(void *ctx, const uint8_t *header, size_t header_len,
                            const uint8_t *payload, size_t payload_len) {
    (void)ctx;
    for (size_t i = 0; i < header_len; i++) send_sum += header[i];
    for (size_t i = 0; i < payload_len; i++) send_sum += payload[i];
    send_packets++;
    return 0;
}

int main(void) {
    /* Register the C seams and seed the RTP identity to match the reference. */
    if (g2g_audio_egress_init(harness_capture, NULL, harness_send, NULL,
                              G2G_AUDIO_EGRESS_REF_SSRC, 0) != 0) {
        fprintf(stderr, "FAIL: g2g_audio_egress_init\n");
        return 1;
    }

    /* The MCU superloop: one frame per iteration, control back to C each time. */
    int emitted = 0;
    for (int i = 0; i < FRAMES; i++) {
        int32_t rc = g2g_audio_egress_step();
        if (rc == 1) {
            emitted++;
        } else {
            fprintf(stderr, "FAIL: g2g_audio_egress_step frame %d returned %d\n", i, rc);
            return 1;
        }
    }

    uint64_t got = send_sum + ((uint64_t)send_packets << 32);
    uint64_t want = g2g_audio_egress_reference(FRAMES);
    if (got != want) {
        fprintf(stderr, "FAIL: C-seam wire checksum %llu != Rust reference %llu\n",
                (unsigned long long)got, (unsigned long long)want);
        return 1;
    }

    printf("host run OK: %d frames captured + G.711-encoded + RTP-sent through C "
           "seams, wire checksum %llu (matches the Rust reference)\n",
           emitted, (unsigned long long)got);
    g2g_audio_egress_reset();
    return 0;
}
