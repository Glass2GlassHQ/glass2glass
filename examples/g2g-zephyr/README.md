# g2g-zephyr: the Zephyr executor proof

The fourth leg of the executor matrix: the same heap-free pipeline the symbol
proofs cover (camera seam -> transform -> SPI display element, from
[`g2g-noalloc`](../g2g-noalloc)) running under **Zephyr** on QEMU's
lm3s6965evb Cortex-M3 (the `qemu_cortex_m3` board) — the bare single-poll
executor ([`g2g-qemu`](../g2g-qemu)), Embassy ([`g2g-embassy`](../g2g-embassy))
and FreeRTOS ([`g2g-freertos`](../g2g-freertos)) being the other three.

Since M647 this consumes the [**g2g Zephyr module**](../g2g-zephyr-module)
rather than hand-linking the archive: the app declares nothing about the
library (no include path, no `target_link_libraries`), it just
`#include <g2g.h>` and calls `g2g_noalloc_run()` / `g2g_audio_run()`. The g2g
module, handed to Zephyr via `ZEPHYR_EXTRA_MODULES` (what a west manifest entry
would arrange), provides the header and links the prebuilt `libg2g_noalloc.a`
(the Rust staticlib with a C ABI, built for `thumbv7m-none-eabi` to match the
board's soft-float Cortex-M3). What this proves beyond FreeRTOS is the packaged
*build-system* integration: a Zephyr shop consumes g2g as a drop-in module. No
west workspace is required: the check script clones the pinned Zephyr tree
(plus the CMSIS module at the revision Zephyr's own manifest pins) into the
gitignored `target/` and points CMake at them directly.

Pieces: `src/main.c` (the thread + printk banner + semihosting verdict,
including `<g2g.h>` from the module), `CMakeLists.txt` (just the Zephyr app
glue: the module does the g2g wiring), `prj.conf` (a bigger main stack: the
audio graph keeps ~6.5 KB of state in locals). The module itself is
[`examples/g2g-zephyr-module`](../g2g-zephyr-module).

## Run the check

```sh
tools/zephyr-check.sh
```

builds the staticlib, configures the Zephyr app with the GNU Arm Embedded
toolchain (`arm-none-eabi-gcc`, override the prefix dir with
`$GNUARMEMB_TOOLCHAIN_PATH`), boots it on `qemu-system-arm -machine
lm3s6965evb` (`$QEMU_SYSTEM_ARM` overrides), and asserts the on-target
checksum banner + semihosting exit code. Python needs Zephyr's base build
deps (`pyelftools`, `PyYAML`, `packaging`, `pykwalify`); `cmake`, `ninja` and
`dtc` must be on PATH.
