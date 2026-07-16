# g2g-qemu: the emulated Cortex-M execution proof

[`g2g-noalloc`](../g2g-noalloc) proves at link time that a whole g2g pipeline
needs no heap and no panic machinery. This crate closes the remaining gap in
that story: the same code had never *executed* on the Cortex-M ISA, only on a
host. It links the exact `g2g-noalloc` pipeline (as an rlib: same code, same
panic handler the symbol proofs cover) into a bare-metal `cortex-m-rt` binary
and runs it on QEMU's MPS2-AN386 board (Cortex-M4): real Thumb-2 instructions,
real 32-bit pointers, no host stand-in.

The binary verifies the pipeline's checksum on-target (64 frames blitted
through the real SPI display element onto a stub bus; the checksum covers the
element's init parameters, window addressing, and RGB565 pixel stream) and
reports the verdict as the semihosting exit code plus a static banner; no
formatting machinery is linked on top of the pipeline.

## Run the check

```sh
tools/qemu-check.sh
```

builds this crate and boots it on `qemu-system-arm -machine mps2-an386`. On a
host without qemu-system-arm, point `$QEMU_SYSTEM_ARM` at an equivalent (e.g.
a podman wrapper running the Fedora package):

```sh
QEMU_SYSTEM_ARM="podman run --rm -v .../release:/w:ro,z fedora:43 ..." tools/qemu-check.sh
```

This is emulation, deliberately not a `Hardware` conformance row: it proves
the ISA/pointer-width story, while on-device validation (STM32H747) stays a
separate, hardware-gated milestone.

Kept out of the normal workspace build (see the root `Cargo.toml` `exclude`);
built only by the check script / CI.
