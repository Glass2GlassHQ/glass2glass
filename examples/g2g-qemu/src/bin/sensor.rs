//! Peripheral-breadth proof (M654): an I2C sensor -> UART telemetry pipeline on
//! the emulated Cortex-M4. A mock SHT3x on the I2C bus returns a datasheet
//! response (raw words + CRC-8); the `Sht3xSrc` driver validates the CRCs and
//! converts to milli-units per the datasheet transfer functions; a `UartSink`
//! streams each reading out a mock UART. The proof asserts the bytes that
//! reached the UART equal the datasheet conversion of the sensor's raw values,
//! so the I2C read + CRC + conversion + UART egress all run correctly on real
//! Thumb-2 code.
//!
//! `tools/qemu-check.sh` boots this and asserts the banner + exit.

#![no_std]
#![no_main]

use core::cell::RefCell;

use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hio};
use embedded_hal::i2c::{ErrorKind, I2c, Operation};

use g2g_core::error::G2gError;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{drive_ready, run_source_sink};
use g2g_mcu::sht3x::{crc8, raw_to_milli_rh, raw_to_millicelsius};
use g2g_mcu::uart::SerialTx;
use g2g_mcu::{Sht3xSrc, UartSink, SHT3X_ADDR_DEFAULT, SHT3X_READING_BYTES};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

/// The sensor's raw temperature / humidity words for this run.
const RAW_T: u16 = 0x5000;
const RAW_RH: u16 = 0x9000;
/// Readings to stream.
const READINGS: u32 = 4;

/// A mock SHT3x returning a fixed datasheet response (raw words + CRC-8 each).
struct MockSht3x;
impl embedded_hal::i2c::ErrorType for MockSht3x {
    type Error = ErrorKind;
}
impl I2c for MockSht3x {
    fn transaction(&mut self, _addr: u8, ops: &mut [Operation<'_>]) -> Result<(), Self::Error> {
        let t = RAW_T.to_be_bytes();
        let rh = RAW_RH.to_be_bytes();
        let resp = [t[0], t[1], crc8(&t), rh[0], rh[1], crc8(&rh)];
        for op in ops {
            if let Operation::Read(buf) = op {
                let n = buf.len().min(resp.len());
                if let (Some(d), Some(s)) = (buf.get_mut(..n), resp.get(..n)) {
                    d.copy_from_slice(s);
                }
            }
        }
        Ok(())
    }
}

/// A mock UART that accumulates a rolling checksum + byte count of everything
/// written (a fixed-size capture, no allocation).
struct CaptureTx<'a> {
    acc: &'a RefCell<(u64, u32)>,
}
impl SerialTx for CaptureTx<'_> {
    async fn write_all(&mut self, bytes: &[u8]) -> Result<(), G2gError> {
        let mut a = self.acc.borrow_mut();
        for &b in bytes {
            a.0 = a.0.wrapping_mul(1_000_003).wrapping_add(b as u64);
            a.1 = a.1.wrapping_add(1);
        }
        Ok(())
    }
}

/// The expected UART checksum: `READINGS` copies of [t_mC i32 LE, rh i32 LE],
/// converted per the datasheet, folded with the same rolling hash.
fn reference() -> (u64, u32) {
    let mut acc = 0u64;
    let mut count = 0u32;
    for _ in 0..READINGS {
        let mut push = |bytes: [u8; 4]| {
            for b in bytes {
                acc = acc.wrapping_mul(1_000_003).wrapping_add(b as u64);
                count = count.wrapping_add(1);
            }
        };
        push(raw_to_millicelsius(RAW_T).to_le_bytes());
        push(raw_to_milli_rh(RAW_RH).to_le_bytes());
    }
    (acc, count)
}

#[entry]
fn main() -> ! {
    let (want_acc, want_count) = reference();

    let capture = RefCell::new((0u64, 0u32));
    let ring: StaticLendRing<2, SHT3X_READING_BYTES> = StaticLendRing::new();
    // SAFETY: the ring outlives the pipeline run below.
    let src = unsafe { Sht3xSrc::with_ring(MockSht3x, SHT3X_ADDR_DEFAULT, &ring, 1_000_000) }
        .with_frame_limit(READINGS);
    let mut sink = UartSink::new(CaptureTx { acc: &capture });

    let _ = drive_ready(run_source_sink(src, &mut sink));

    let (acc, count) = *capture.borrow();
    let ok = acc == want_acc && count == want_count && count == READINGS * SHT3X_READING_BYTES as u32;

    if let Ok(mut out) = hio::hstdout() {
        let mut line = [0u8; 64];
        let mut pos = 0;
        put_str(&mut line, &mut pos, "g2g-sensor: uart-bytes=");
        put_u32(&mut line, &mut pos, count);
        put_str(&mut line, &mut pos, if ok { " OK\n" } else { " FAIL\n" });
        let _ = out.write_all(line.get(..pos).unwrap_or(&[]));
    }

    debug::exit(if ok { debug::EXIT_SUCCESS } else { debug::EXIT_FAILURE });
    loop {}
}

/// Append `v` in decimal to `buf` at `pos` (no `core::fmt`).
fn put_u32(buf: &mut [u8], pos: &mut usize, v: u32) {
    let mut digits = [0u8; 10];
    let mut n = 0;
    let mut v = v;
    loop {
        if let Some(d) = digits.get_mut(n) {
            *d = b'0' + (v % 10) as u8;
        }
        n += 1;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    while n > 0 {
        n -= 1;
        if let (Some(dst), Some(&src)) = (buf.get_mut(*pos), digits.get(n)) {
            *dst = src;
            *pos += 1;
        }
    }
}

fn put_str(buf: &mut [u8], pos: &mut usize, s: &str) {
    for &b in s.as_bytes() {
        if let Some(dst) = buf.get_mut(*pos) {
            *dst = b;
            *pos += 1;
        }
    }
}
