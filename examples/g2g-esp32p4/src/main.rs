//! ESP32-P4 (RISC-V) on-device Tier-1 harness: drive the heap-free g2g display
//! pipeline (`noalloc_pipeline::run_display_with`, the same camera -> transform
//! -> `SpiDisplaySink` graph every no_std proof covers) onto the ESP32-P4-EYE's
//! ST7789 panel over esp-hal's SPI + GPIO, on real silicon. This is the payoff
//! of the M656 port: the pipeline already links for `riscv32imafc`; here a real
//! HAL supplies the peripherals the proof stub bus stood in for.
//!
//! STATUS (drafted, not CI-built). esp-hal's `esp32p4` support is on the esp-hal
//! git `main` branch only (no published crate has an `esp32p4` feature as of
//! esp-hal 1.1.1), so this crate is excluded from the workspace and not built
//! in CI. It is a faithful draft against esp-hal 1.x: the GPIO map and a few
//! driver calls MUST be checked against the ESP32-P4-EYE schematic and the
//! exact esp-hal revision before flashing. See README.md.
//!
//! The pipeline streams the full 240x240 panel in 16-row bands
//! (`run_display_banded_with`, M659): the ring holds one 15 KB band, never a
//! 230 KB framebuffer, so a full refresh runs from MCU-small memory.

#![no_std]
#![no_main]

use embedded_hal::delay::DelayNs;
use embedded_hal::digital::OutputPin;
use embedded_hal_bus::spi::ExclusiveDevice;
use esp_hal::delay::Delay;
use esp_hal::gpio::{Level, Output, OutputConfig};
use esp_hal::spi::master::{Config as SpiConfig, Spi};
use esp_hal::spi::Mode;
use esp_hal::time::Rate;

// esp-hal ships no panic handler for a bare RISC-V chip (esp-backtrace would,
// but it also lacks an esp32p4 release); a minimal handler keeps the draft
// dependency-light. Swap in esp-backtrace once it ships esp32p4.
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[esp_hal::main]
fn main() -> ! {
    let p = esp_hal::init(esp_hal::Config::default());

    // --- Panel wiring -------------------------------------------------------
    // ST7789 over SPI2, mode 0, 40 MHz. VERIFY every GPIO against the
    // ESP32-P4-EYE schematic; the numbers below are placeholders for the
    // SPI clock/data + the CS / D-C / RESET control lines.
    let spi = Spi::new(
        p.SPI2,
        SpiConfig::default().with_frequency(Rate::from_mhz(40)).with_mode(Mode::_0),
    )
    .expect("SPI2 configure")
    .with_sck(p.GPIO9) //  panel SCL  (VERIFY)
    .with_mosi(p.GPIO10); // panel SDA  (VERIFY)

    let cs = Output::new(p.GPIO11, Level::High, OutputConfig::default()); // (VERIFY)
    let mut dc = Output::new(p.GPIO12, Level::High, OutputConfig::default()); // (VERIFY)
    let mut rst = Output::new(p.GPIO13, Level::High, OutputConfig::default()); // (VERIFY)
    let mut delay = Delay::new();

    // ST7789 hardware reset pulse (datasheet: >=10 us low, settle >=120 ms).
    let _ = rst.set_low();
    delay.delay_ms(20);
    let _ = rst.set_high();
    delay.delay_ms(150);

    // embedded-hal 1.0 `SpiDevice` = bus + CS + inter-transfer delay. This is
    // the seam `SpiDisplaySink` consumes; D/C is a separate `OutputPin`.
    let mut dev = ExclusiveDevice::new(spi, cs, Delay::new()).expect("exclusive SPI device");

    // Drive the board-agnostic g2g pipeline. `run_display_banded_with` is
    // generic over the embedded-hal seams, so the real device/pin/timer slot
    // straight in; it streams one full 240x240 refresh in 16-row bands from a
    // 15 KB ring. Loop to keep refreshing. Real MIPI-CSI capture is Tier 2:
    // bridge the ESP-IDF camera driver through the M650 `CFrameGrabber` C-seam,
    // and the hardware H.264 encoder through `CH264Encoder` (M660).
    let mut init_delay = Delay::new();
    loop {
        let _ = noalloc_pipeline::drive_ready(noalloc_pipeline::run_display_banded_with(
            &mut dev,
            &mut dc,
            &mut init_delay,
        ));
    }
}
