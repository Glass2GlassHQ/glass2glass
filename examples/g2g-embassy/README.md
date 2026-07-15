# g2g-embassy: the Embassy executor proof

[`g2g-qemu`](../g2g-qemu) proves the heap-free pipeline executes on the
Cortex-M ISA under a bare single-poll executor. This crate proves the shape a
production MCU deployment actually uses: a real **Embassy** task awaits the
same pipeline future (`noalloc-pipeline::run_async`, the camera seam ->
transform -> SPI display element chain the symbol proofs cover) on
`embassy-executor`'s thread-mode Cortex-M executor, on QEMU's MPS2-AN386.

Because the static element model is executor-agnostic `async` (no `dyn`, no
`Box`), the Embassy integration is exactly this thin: `#[embassy_executor::main]`
plus one `.await`. No adaptation layer exists, which is the point.

The binary verifies the pipeline's wire checksum on-target and reports the
verdict as the semihosting exit code plus a static banner.

## Run the check

```sh
tools/qemu-check.sh
```

builds and boots both this crate and `g2g-qemu` on `qemu-system-arm -machine
mps2-an386` (`$QEMU_SYSTEM_ARM` overrides the emulator binary, e.g. a podman
wrapper).

Kept out of the normal workspace build (see the root `Cargo.toml` `exclude`);
built only by the check script / CI.
