/* Public C interface of the g2g static pipeline library, provided by the g2g
 * Zephyr module (M647). An application that lists this module gets this header
 * on its include path automatically; it need only `#include <g2g.h>` and call
 * the entry points below.
 *
 * Each `*_run` executes a heap-free, panic-free g2g pipeline to completion and
 * returns a wire checksum; the paired `*_expected` returns the value a correct
 * run produces, so a caller compares against the library's own constant rather
 * than hard-coding it. */
#ifndef G2G_H
#define G2G_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The camera -> transform -> SPI-display proof pipeline (64 frames). */
uint64_t g2g_noalloc_run(void);
uint64_t g2g_noalloc_expected(void);

/* The flagship audio graph: capture -> convert -> resample -> mix -> encode
 * -> RTP, checksummed over the emitted RTP wire bytes. */
uint64_t g2g_audio_run(void);
uint64_t g2g_audio_expected(void);

#ifdef __cplusplus
}
#endif

#endif /* G2G_H */
