# g2g-freertos: the FreeRTOS executor proof

The third leg of the executor matrix: the same heap-free pipeline the symbol
proofs cover (camera seam -> transform -> SPI display element, from
[`g2g-noalloc`](../g2g-noalloc)) running under a **FreeRTOS** task on QEMU's
MPS2-AN386 Cortex-M4 — the bare single-poll executor ([`g2g-qemu`](../g2g-qemu))
and Embassy ([`g2g-embassy`](../g2g-embassy)) being the other two.

This is the C-shop integration path, and it is deliberately thin: link
`libg2g_noalloc.a` (the Rust staticlib with a C ABI), call `g2g_noalloc_run()`
from a task, compare against `g2g_noalloc_expected()`. The application uses
FreeRTOS **static allocation only** (`configSUPPORT_DYNAMIC_ALLOCATION = 0`,
so no FreeRTOS heap implementation is even compiled), keeping the whole image
consistent with the g2g no-heap guarantee: neither the pipeline nor the RTOS
allocates.

Pieces: `main.c` (the task + semihosting verdict), `startup.c` (vector table,
FPU enable, data/bss init), `FreeRTOSConfig.h` (CM4F, static-only),
`mps2_an386.ld`. The FreeRTOS kernel itself is fetched pinned
(`V11.2.0`) into the gitignored `target/` on first run, not vendored.

## Run the check

```sh
tools/freertos-check.sh
```

builds the staticlib, cross-compiles the application with `arm-none-eabi-gcc`
(override with `$FREERTOS_CC`), boots it on `qemu-system-arm -machine
mps2-an386` (`$QEMU_SYSTEM_ARM` overrides), and asserts the on-target checksum
banner + semihosting exit code.
