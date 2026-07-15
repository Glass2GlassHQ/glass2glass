/* Host-side behavioral check for the g2g-noalloc proof crate (M626): link the
 * staticlib built for the *host* and actually run the pipeline, so the symbol
 * checks in tools/noalloc-check.sh are backed by a real execution (guards
 * against the pipeline linking but hanging or computing nothing).
 *
 * The pipeline blits 64 frames through the SPI display element onto a stub
 * bus; the checksum over the wire bytes must match the pipeline's own
 * expected constant (see noalloc-pipeline's EXPECTED_CHECKSUM).
 */
#include <stdint.h>
#include <stdio.h>

extern uint64_t g2g_noalloc_run(void);
extern uint64_t g2g_noalloc_expected(void);
extern uint64_t g2g_audio_run(void);
extern uint64_t g2g_audio_expected(void);

int main(void) {
    uint64_t sum = g2g_noalloc_run();
    uint64_t want = g2g_noalloc_expected();
    if (sum != want) {
        fprintf(stderr, "FAIL: g2g_noalloc_run checksum %llu != %llu\n",
                (unsigned long long)sum, (unsigned long long)want);
        return 1;
    }
    printf("host run OK: 64 frames, checksum %llu\n", (unsigned long long)sum);

    /* The flagship audio graph (M644): capture -> convert -> resample ->
     * mix -> encode -> RTP, checksummed over the RTP wire bytes. */
    uint64_t asum = g2g_audio_run();
    uint64_t awant = g2g_audio_expected();
    if (asum != awant) {
        fprintf(stderr, "FAIL: g2g_audio_run checksum %llu != %llu\n",
                (unsigned long long)asum, (unsigned long long)awant);
        return 1;
    }
    printf("host run OK: flagship audio graph, checksum %llu\n",
           (unsigned long long)asum);
    return 0;
}
