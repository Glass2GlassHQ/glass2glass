# g2g-stm32mp2-m0plus

Runs the heap-free display pipeline (`noalloc-pipeline::run`) in a loop on the
STM32MP257's Cortex-M0+. Linux on the A35 cores loads it through remoteproc, and
the firmware writes its run count and checksum to the shared region at
`0x200C6000` (`ipc-shmem-2` in the board's device tree).

The M0+ has 16 KiB for code at `0x200C0000` and 8 KiB for data at `0x200C4000`.
The release build uses about 6 KB of code and no static data.

## Run on the board

```sh
cargo build --release
scp target/thumbv6m-none-eabi/release/g2g-stm32mp2-m0plus <board>:/lib/firmware/g2g-m0plus.elf
scp read_report.py <board>:
```

On the board, as root:

```sh
cd /sys/class/remoteproc/remoteproc0
echo g2g-m0plus.elf > firmware
echo start > state
python3 ~/read_report.py
echo stop > state
echo rproc-m0-fw > firmware
```

`runs` rises between reads, `wrong_checksums` stays 0 and `last_checksum` is
`noalloc_pipeline::EXPECTED_CHECKSUM` (2261). On an STM32MP257F-EV1 it runs
about 720 times a second, 46,000 4x4 frames a second.

Kept out of the normal workspace build (see the root `Cargo.toml` `exclude`).
