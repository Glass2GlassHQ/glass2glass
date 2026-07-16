/* FreeRTOS executor proof (M633): a statically-allocated FreeRTOS task runs
 * the exact g2g heap-free pipeline (the g2g-noalloc staticlib the no-heap /
 * panic-free symbol proofs cover: camera seam -> transform -> SPI display
 * element) on QEMU's MPS2-AN386 Cortex-M4, and verifies the wire checksum
 * on-target. This is the C-shop integration path: link libg2g_noalloc.a,
 * call one function from a task. The verdict is the semihosting exit code
 * plus a static banner. */
#include <stdint.h>

#include "FreeRTOS.h"
#include "task.h"

extern uint64_t g2g_noalloc_run(void);
extern uint64_t g2g_noalloc_expected(void);
extern uint64_t g2g_audio_run(void);
extern uint64_t g2g_audio_expected(void);

/* ARM semihosting: r0 = operation, r1 = parameter, BKPT 0xAB. */
static void sh_call(int op, const void *arg)
{
    register int r0 __asm__("r0") = op;
    register const void *r1 __asm__("r1") = arg;
    __asm__ volatile("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory");
}

#define SH_SYS_WRITE0 0x04
#define SH_SYS_EXIT 0x18
#define SH_ADP_STOPPED_APPLICATION_EXIT ((const void *)0x20026)

/* Worst-case stacks (tools/footprint-report.sh): video pipeline ~1.4 KB, the
 * flagship audio graph ~6.5 KB; 3072 words = 12 KB leaves generous margin
 * for the task context + C shims. */
#define PIPELINE_STACK_WORDS 3072

static StaticTask_t pipeline_tcb;
static StackType_t pipeline_stack[PIPELINE_STACK_WORDS];

static StaticTask_t idle_tcb;
static StackType_t idle_stack[configMINIMAL_STACK_SIZE];

void vApplicationGetIdleTaskMemory(StaticTask_t **tcb, StackType_t **stack,
                                   configSTACK_DEPTH_TYPE *words)
{
    *tcb = &idle_tcb;
    *stack = idle_stack;
    *words = configMINIMAL_STACK_SIZE;
}

static void pipeline_task(void *arg)
{
    (void)arg;
    uint64_t sum = g2g_noalloc_run();
    /* The flagship audio graph (M644): capture -> convert -> resample ->
     * mix -> encode -> RTP, verified against its host-pinned checksum. */
    uint64_t asum = g2g_audio_run();
    if (sum == g2g_noalloc_expected() && asum == g2g_audio_expected()) {
        sh_call(SH_SYS_WRITE0,
                "g2g-freertos: video + flagship audio ran under FreeRTOS on Cortex-M4, checksums OK\n");
        sh_call(SH_SYS_EXIT, SH_ADP_STOPPED_APPLICATION_EXIT); /* qemu exit 0 */
    } else {
        sh_call(SH_SYS_WRITE0, "g2g-freertos: FAIL, wrong checksum\n");
        sh_call(SH_SYS_EXIT, 0); /* anything but ApplicationExit: qemu exit 1 */
    }
    for (;;) {
    }
}

int main(void)
{
    (void)xTaskCreateStatic(pipeline_task, "g2g", PIPELINE_STACK_WORDS, NULL, 1, pipeline_stack,
                            &pipeline_tcb);
    vTaskStartScheduler();
    /* Unreachable: static allocation cannot fail to start the scheduler. */
    for (;;) {
    }
}
