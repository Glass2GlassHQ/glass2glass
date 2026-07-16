/* Minimal Cortex-M4 startup for QEMU MPS2-AN386 (M633): vector table, FPU
 * enable, .data/.bss init. The SVC/PendSV/SysTick slots point at the FreeRTOS
 * CM4F port handlers (renamed via FreeRTOSConfig.h). */
#include <stdint.h>

extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _estack;

extern void SVC_Handler(void);
extern void PendSV_Handler(void);
extern void SysTick_Handler(void);
extern int main(void);

void Reset_Handler(void);

static void Default_Handler(void)
{
    for (;;) {
    }
}

__attribute__((section(".isr_vector"), used)) static void (*const vector_table[])(void) = {
    (void (*)(void))(&_estack), /* initial stack pointer */
    Reset_Handler,              /* reset */
    Default_Handler,            /* NMI */
    Default_Handler,            /* HardFault */
    Default_Handler,            /* MemManage */
    Default_Handler,            /* BusFault */
    Default_Handler,            /* UsageFault */
    0, 0, 0, 0,                 /* reserved */
    SVC_Handler,                /* SVCall -> vPortSVCHandler */
    Default_Handler,            /* DebugMonitor */
    0,                          /* reserved */
    PendSV_Handler,             /* PendSV -> xPortPendSVHandler */
    SysTick_Handler,            /* SysTick -> xPortSysTickHandler */
};

void Reset_Handler(void)
{
    /* Enable CP10/CP11 (FPU) before any hard-float-ABI code runs. */
    volatile uint32_t *cpacr = (volatile uint32_t *)0xE000ED88;
    *cpacr |= (0xFu << 20);
    __asm volatile("dsb\n isb");

    /* Copy .data from flash, zero .bss. */
    uint32_t *src = &_sidata;
    for (uint32_t *dst = &_sdata; dst < &_edata;) {
        *dst++ = *src++;
    }
    for (uint32_t *dst = &_sbss; dst < &_ebss;) {
        *dst++ = 0;
    }

    (void)main();
    for (;;) {
    }
}
