# MCU boards

g2g's heap-free core and `g2g-mcu` have run on these microcontrollers. Each
entry is real silicon with the harness that runs it, followed by what the board
could do next. The design behind the harnesses is in
[design/embedded.md](design/embedded.md).

| Board | Core | Rust target | Harness | First run |
| :--- | :--- | :--- | :--- | :--- |
| Seeed XIAO ESP32S3 Sense | Xtensa LX7, two cores, 240 MHz | `xtensa-esp32s3-none-elf` | `examples/g2g-esp32s3` | M2333, 2026-11-15 |
| STM32MP257F-EV1, Cortex-M0+ | Armv6-M | `thumbv6m-none-eabi` | `examples/g2g-stm32mp2-m0plus` | M2466, 2026-11-19 |

Targets that have not run on silicon yet:

- QEMU MPS2-AN386 (emulated Cortex-M4): `examples/g2g-qemu` and
  `examples/g2g-embassy`, run by `tools/qemu-check.sh`.
- ESP32-P4-EYE (RISC-V): `examples/g2g-esp32p4` is written but has not run on a
  board.
- NUCLEO-H743ZI2 (Cortex-M7): `examples/g2g-stm32h743` compiles only.

## XIAO ESP32S3 Sense

Runs the display, audio and keyword-feature graphs with checksums that match the
host bit for bit. Live, it scores the PDM microphone for "yes" and "no", scores
the OV3660 camera for a person with TFLite Micro and ESP-NN, and streams the
camera over Wi-Fi as RFC 4175 RTP with the person label in each frame. Person
detection takes 64 ms per frame with its arena in PSRAM. Setup and flashing are
in the harness README.

What it could do next:

- Average keyword scores over several windows. Single windows of background
  noise score "no" at 0.4 to 0.75 today.
- Send the microphone over Wi-Fi as RTP beside the video, using `g2g-mcu`'s
  G.711 or IMA ADPCM encoder and RTP payloader, the elements of the flagship
  audio graph.
- Send only the label when no one is in view, to cut Wi-Fi traffic.

## STM32MP257F-EV1 Cortex-M0+

The M0+ is the smallest of the STM32MP257's three processor types, beside two
Cortex-A35 cores running Linux and a Cortex-M33. It sits in the SmartRun power
domain, so it can keep running while the rest of the chip is powered down.
Linux loads and starts its firmware through remoteproc (`remoteproc0`), and no
resource table is needed.

Memory, from the board's device tree and ST's CM0PLUS_DEMO linker script:

| Region | Address | Size |
| :--- | :--- | :--- |
| Code (`cm0-cube-fw`) | `0x200C0000` | 16 KiB |
| Data and stack (`cm0-cube-data`) | `0x200C4000` | 8 KiB |
| Shared with Linux (`ipc-shmem-2`) | `0x200C6000` | 8 KiB |

The harness runs the heap-free display pipeline in a loop and writes its run
count and checksum to the shared region, which Linux reads through `/dev/mem`.
It runs 723 times a second (46,000 4x4 frames a second) with every checksum
correct, in 6 KB of code and no static data.

Peripherals in the SmartRun domain, which the Linux device tree leaves disabled
and so free for the M0+: SPI8 (`0x46020000`), LPUART1 (`0x46030000`), I2C8
(`0x46040000`), I3C4 (`0x46080000`) and LPTIM4 and LPTIM5 (`0x46060000`,
`0x46070000`). LPTIM3 (`0x46050000`) is in use by Linux. ST's demo also uses
IPCC2 to signal the A35 cores, the GPIOZ port, and LPUART1 on GPIOZ pins 7 and 8
for its console. ST's documentation also lists the ADF1 audio digital filter,
LPDMA, RTC, EXTI2 and HSEM as reachable from the M0+.

Two limits apply. The M0+ crashes when the chip enters or leaves the LPLV-Stop1,
LPLV-Stop2 and Standby1 low-power modes unless the firmware waits in WFI around
the clock change, the workaround in ST's CM0PLUS_DEMO (STM32MP25 errata). And
the firmware needs register drivers for each peripheral it uses. The
`stm32mp2-pac` and `consortium-hal-stm32mp2` crates exist but have not been
checked for the M0+.

What it could do next:

- Print over LPUART1 with `g2g-mcu`'s `UartSink`, on GPIOZ 7 and 8 like ST's
  demo. This is the smallest step that needs a peripheral driver.
- Log a sensor while Linux sleeps: an SHT3x on I2C8 read by `Sht3xSrc`, readings
  written to the shared region, and the A35 woken through IPCC2 when a reading
  crosses a threshold. Needs the low-power workaround and a sensor wired to the
  I2C8 pins.
- Watch for sound on a PDM microphone through ADF1 and wake Linux to run speech
  recognition. Keyword spotting does not fit in 8 KiB of RAM, but a level
  detector does. Whether the EV1 board routes a microphone to ADF1 is not known.
- Watch the Linux pipeline: Linux writes a heartbeat to the shared region, and
  the M0+ resets or wakes it when the heartbeat stops, using `g2g-mcu`'s watchdog
  supervision.

The Cortex-M33 on the same chip runs ST's USB-PD firmware. g2g-mcu could run
there only by linking into that firmware, but it has 8 MB of DDR for code and 8
MB for data, enough for the keyword and person models.
