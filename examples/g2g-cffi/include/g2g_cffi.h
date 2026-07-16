/* C seam ABI (M650): drive a g2g audio egress pipeline
 * (capture -> G.711 mu-law encode -> RTP) from your own C code, with your
 * existing C drivers as the peripheral. You provide two callbacks, register
 * them once, then step the pipeline one frame at a time from your superloop;
 * control returns to you between frames. No Rust is written on the integration
 * side. This is the inverse of examples/g2g-freertos, where the C app calls a
 * whole-pipeline entry point; here g2g calls your capture and send back.
 */
#ifndef G2G_CFFI_H
#define G2G_CFFI_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Capture callback: fill up to `len` bytes at `buf` with ONE frame of mono
 * S16LE audio (your DMA/mic driver), and return the byte count written (>= 0,
 * normally `len`), or a negative value to report a capture fault. `ctx` is the
 * opaque handle you passed to init (your driver state). */
typedef intptr_t (*g2g_capture_fn)(void *ctx, uint8_t *buf, size_t len);

/* Send callback: transmit one datagram, `header` (the RTP fixed header, 12
 * bytes) immediately followed by `payload` (your network stack). Return 0 on
 * success or a negative value on a transport fault. Header and payload are
 * separate pointers so a scatter-gather stack stays zero-copy. */
typedef int32_t (*g2g_send_fn)(void *ctx, const uint8_t *header, size_t header_len,
                               const uint8_t *payload, size_t payload_len);

/* Wire the pipeline from your callbacks. `ssrc` / `sequence` seed the RTP
 * stream identity (an MCU picks these at boot). Returns 0. Replaces any
 * pipeline a prior init left in place. Call from your single pipeline thread. */
int32_t g2g_audio_egress_init(g2g_capture_fn capture, void *capture_ctx,
                              g2g_send_fn send, void *send_ctx,
                              uint32_t ssrc, uint16_t sequence);

/* Run exactly ONE frame (capture -> encode -> send) and return control:
 *    1  a packet was emitted this frame,
 *    0  end of stream,
 *   -1  a stage errored,
 *   -2  a stage suspended (this pipeline needs a real executor),
 *   -3  init has not run.
 * Call once per frame from your superloop, yielding in between. */
int32_t g2g_audio_egress_step(void);

/* Drop the pipeline, releasing your callbacks. A fresh init may follow. */
void g2g_audio_egress_reset(void);

/* The input frame size (S16LE bytes) your capture callback is handed per frame,
 * from the library (single source of truth), so you can size a scratch buffer. */
size_t g2g_audio_egress_frame_bytes(void);

/* ── Proof helpers (used by harness.c to check the C seam is byte-transparent) ──
 * Fill `buf` (`len` S16LE bytes) with the reference capture ramp starting at
 * absolute sample index `global_sample`, so a C capture and the Rust reference
 * feed byte-identical input. Advance `global_sample` by len/2 per frame. */
void g2g_audio_egress_fill_ramp(uint8_t *buf, size_t len, uint32_t global_sample);

/* Run the same pipeline with native Rust seams over the reference ramp for
 * `frames` frames and return its wire checksum (sum of every emitted byte plus
 * the packet count in the high 32 bits). A C-seam run over the same ramp and
 * this SSRC / sequence 0 must reproduce it exactly. */
uint64_t g2g_audio_egress_reference(uint32_t frames);

/* The RTP identity the reference uses; init with this (and sequence 0) to match
 * g2g_audio_egress_reference. */
#define G2G_AUDIO_EGRESS_REF_SSRC 0x0A0B0C0Du

#ifdef __cplusplus
}
#endif

#endif /* G2G_CFFI_H */
