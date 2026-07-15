/* FreeRTOS configuration for the M633 proof: Cortex-M4F on QEMU MPS2-AN386,
 * static allocation ONLY (configSUPPORT_DYNAMIC_ALLOCATION = 0, so no
 * FreeRTOS heap implementation is even linked), which keeps the whole image
 * consistent with the g2g no-heap guarantee: the g2g pipeline links no
 * allocator, and the RTOS side allocates nothing either. */
#ifndef FREERTOS_CONFIG_H
#define FREERTOS_CONFIG_H

#define configUSE_PREEMPTION                    1
#define configUSE_IDLE_HOOK                     0
#define configUSE_TICK_HOOK                     0
#define configCPU_CLOCK_HZ                      (25000000UL) /* MPS2 AN386 */
#define configTICK_RATE_HZ                      ((TickType_t)1000)
#define configMAX_PRIORITIES                    (5)
#define configMINIMAL_STACK_SIZE                ((unsigned short)128)
#define configMAX_TASK_NAME_LEN                 (10)
#define configUSE_TRACE_FACILITY                0
#define configUSE_16_BIT_TICKS                  0
#define configIDLE_SHOULD_YIELD                 1
#define configUSE_MUTEXES                       0
#define configUSE_COUNTING_SEMAPHORES           0
#define configUSE_RECURSIVE_MUTEXES             0
#define configQUEUE_REGISTRY_SIZE               0
#define configUSE_QUEUE_SETS                    0
#define configUSE_TIMERS                        0
#define configUSE_TASK_NOTIFICATIONS            1
#define configCHECK_FOR_STACK_OVERFLOW          0
#define configUSE_MALLOC_FAILED_HOOK            0

/* Static allocation only: no heap_x.c is compiled or linked. */
#define configSUPPORT_STATIC_ALLOCATION         1
#define configSUPPORT_DYNAMIC_ALLOCATION        0

#define INCLUDE_vTaskPrioritySet                0
#define INCLUDE_uxTaskPriorityGet               0
#define INCLUDE_vTaskDelete                     0
#define INCLUDE_vTaskSuspend                    0
#define INCLUDE_vTaskDelayUntil                 0
#define INCLUDE_vTaskDelay                      0

/* Cortex-M interrupt priorities (3 priority bits on the QEMU MPS2 NVIC,
 * matching the official FreeRTOS CORTEX_MPS2_QEMU demo). */
#define configPRIO_BITS                         3
#define configKERNEL_INTERRUPT_PRIORITY         (255)
#define configMAX_SYSCALL_INTERRUPT_PRIORITY    (5 << (8 - configPRIO_BITS))

#define configASSERT(x)                                                                            \
    if ((x) == 0) {                                                                                \
        taskDISABLE_INTERRUPTS();                                                                  \
        for (;;)                                                                                   \
            ;                                                                                      \
    }

/* Map the CM4F port handlers onto the CMSIS vector names startup.c uses. */
#define vPortSVCHandler                         SVC_Handler
#define xPortPendSVHandler                      PendSV_Handler
#define xPortSysTickHandler                     SysTick_Handler

#endif /* FREERTOS_CONFIG_H */
