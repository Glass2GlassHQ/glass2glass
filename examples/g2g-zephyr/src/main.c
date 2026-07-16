/* Zephyr executor proof (M637), now via the g2g Zephyr module (M647):
 * Zephyr's main thread runs the exact g2g heap-free pipelines (the g2g-noalloc
 * staticlib the no-heap / panic-free symbol proofs cover) on QEMU's
 * lm3s6965evb Cortex-M3 (the qemu_cortex_m3 board) and verifies the wire
 * checksums on-target. The integration is now the packaged path: the app
 * declares nothing about the library, it just `#include <g2g.h>` and calls
 * the entry points; the g2g Zephyr module (on ZEPHYR_EXTRA_MODULES) supplies
 * the header and links the staticlib. The verdict is the semihosting exit
 * code plus a banner on the Zephyr console. */
#include <stdint.h>

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

/* Provided by the g2g Zephyr module: the entry points and their expected
 * checksums, no local extern declarations. */
#include <g2g.h>

/* ARM semihosting: r0 = operation, r1 = parameter, BKPT 0xAB. */
static void sh_call(int op, const void *arg)
{
	register int r0 __asm__("r0") = op;
	register const void *r1 __asm__("r1") = arg;
	__asm__ volatile("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory");
}

#define SH_SYS_EXIT 0x18
#define SH_ADP_STOPPED_APPLICATION_EXIT ((const void *)0x20026)

int main(void)
{
	uint64_t sum = g2g_noalloc_run();
	/* The flagship audio graph (M644): capture -> convert -> resample ->
	 * mix -> encode -> RTP, verified against its host-pinned checksum. */
	uint64_t asum = g2g_audio_run();

	if (sum == g2g_noalloc_expected() && asum == g2g_audio_expected()) {
		printk("g2g-zephyr: video + flagship audio ran under Zephyr on Cortex-M3, checksums OK\n");
		sh_call(SH_SYS_EXIT, SH_ADP_STOPPED_APPLICATION_EXIT); /* qemu exit 0 */
	} else {
		printk("g2g-zephyr: FAIL, wrong checksum\n");
		sh_call(SH_SYS_EXIT, 0); /* anything but ApplicationExit: qemu exit 1 */
	}
	for (;;) {
	}
	return 0;
}
