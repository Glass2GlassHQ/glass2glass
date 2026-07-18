//! M629: `SpiDisplaySink` drives a mock `embedded-hal` SPI bus + D/C pin and
//! the recorded wire traffic is asserted against the ST7789 datasheet: the
//! init sequence (opcodes, parameters, D/C phases, reset/wake delays), the
//! CASET/RASET window addressing with panel offsets, and the streamed
//! RGBA8888 -> big-endian RGB565 pixel conversion. The mocks stand in for the
//! external bus only; the element under test is the real one.

use std::cell::RefCell;
use std::future::Future;
use std::rc::Rc;

use embedded_hal::delay::DelayNs;
use embedded_hal::spi::{ErrorKind, Operation, SpiDevice};
use g2g_core::error::{G2gError, HardwareError};
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::MemoryDomain;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::StaticSink;
use g2g_mcu::SpiDisplaySink;

/// Shared bus recorder: every SPI write with the D/C level it was sent under.
#[derive(Default)]
struct BusLog {
    writes: Vec<(bool, Vec<u8>)>,
    dc_high: bool,
    fail_spi: bool,
}

struct MockSpi(Rc<RefCell<BusLog>>);

impl embedded_hal::spi::ErrorType for MockSpi {
    type Error = ErrorKind;
}

impl SpiDevice for MockSpi {
    fn transaction(&mut self, ops: &mut [Operation<'_, u8>]) -> Result<(), Self::Error> {
        let mut log = self.0.borrow_mut();
        if log.fail_spi {
            return Err(ErrorKind::Other);
        }
        for op in ops {
            if let Operation::Write(bytes) = op {
                let dc = log.dc_high;
                log.writes.push((dc, bytes.to_vec()));
            }
        }
        Ok(())
    }
}

struct MockDc(Rc<RefCell<BusLog>>);

impl embedded_hal::digital::ErrorType for MockDc {
    type Error = core::convert::Infallible;
}

impl embedded_hal::digital::OutputPin for MockDc {
    fn set_low(&mut self) -> Result<(), Self::Error> {
        self.0.borrow_mut().dc_high = false;
        Ok(())
    }
    fn set_high(&mut self) -> Result<(), Self::Error> {
        self.0.borrow_mut().dc_high = true;
        Ok(())
    }
}

/// Records requested delays (in ms) instead of sleeping.
#[derive(Default)]
struct MockDelay(Vec<u32>);

impl DelayNs for MockDelay {
    fn delay_ns(&mut self, ns: u32) {
        self.0.push(ns / 1_000_000);
    }
}

/// Reassemble the raw write log into DCS (opcode, payload-bytes) pairs: a
/// D/C-low single byte opens a command; every following D/C-high write is its
/// payload (parameters or RAMWR pixel data).
fn commands(log: &BusLog) -> Vec<(u8, Vec<u8>)> {
    let mut out: Vec<(u8, Vec<u8>)> = Vec::new();
    for (dc_high, bytes) in &log.writes {
        if *dc_high {
            out.last_mut()
                .expect("data write before any command")
                .1
                .extend(bytes);
        } else {
            assert_eq!(bytes.len(), 1, "command writes are single opcodes");
            out.push((bytes[0], Vec::new()));
        }
    }
    out
}

fn block_on<F: Future>(fut: F) -> F::Output {
    g2g_core::drive_ready(fut).expect("the static chain never suspends")
}

fn rig(width: u16, height: u16) -> (Rc<RefCell<BusLog>>, SpiDisplaySink<MockSpi, MockDc>) {
    let log = Rc::new(RefCell::new(BusLog::default()));
    let sink = SpiDisplaySink::st7789(MockSpi(log.clone()), MockDc(log.clone()), width, height);
    (log, sink)
}

/// A 2x2 RGBA frame (red, green, blue, white) lent from a ring, as an MCU
/// capture path would produce it.
fn rgba_2x2_frame(ring: &StaticLendRing<1, 16>) -> Frame {
    let mut slot = ring.acquire().expect("free slot");
    slot.buf_mut().copy_from_slice(&[
        255, 0, 0, 255, // red
        0, 255, 0, 255, // green
        0, 0, 255, 255, // blue
        255, 255, 255, 255, // white
    ]);
    // SAFETY: the ring outlives the frame in every test that calls this.
    let slice = unsafe { slot.publish(16) };
    Frame::new(MemoryDomain::System(slice), FrameTiming::default(), 0)
}

#[test]
fn init_follows_the_st7789_datasheet() {
    let (log, mut sink) = rig(240, 240);
    let mut delay = MockDelay::default();
    sink.init(&mut delay).expect("init");

    let cmds = commands(&log.borrow());
    let expected: Vec<(u8, Vec<u8>)> = vec![
        (0x01, vec![]),     // SWRESET
        (0x11, vec![]),     // SLPOUT
        (0x3A, vec![0x55]), // COLMOD: RGB565
        (0x36, vec![0x00]), // MADCTL: portrait
        (0x21, vec![]),     // INVON (ST7789 IPS glass)
        (0x13, vec![]),     // NORON
        (0x29, vec![]),     // DISPON
    ];
    assert_eq!(cmds, expected, "DCS init sequence");
    assert_eq!(delay.0, vec![150, 120, 10, 100], "reset/wake delays (ms)");
}

#[test]
fn ili9341_skips_inversion() {
    let log = Rc::new(RefCell::new(BusLog::default()));
    let mut sink = SpiDisplaySink::ili9341(MockSpi(log.clone()), MockDc(log.clone()), 240, 320);
    sink.init(&mut MockDelay::default()).expect("init");
    let opcodes: Vec<u8> = commands(&log.borrow()).iter().map(|(c, _)| *c).collect();
    assert!(opcodes.contains(&0x20), "INVOFF sent");
    assert!(!opcodes.contains(&0x21), "no INVON");
}

#[test]
fn blit_windows_and_converts_to_rgb565() {
    // A module mounted at (1, 80) of the controller RAM (the 240x240 ST7789
    // glass sits at y offset 80 of the 240x320 RAM; scaled down here).
    let (log, sink) = rig(2, 2);
    let mut sink = sink.with_offset(1, 80);
    sink.init(&mut MockDelay::default()).expect("init");
    log.borrow_mut().writes.clear();

    let ring: StaticLendRing<1, 16> = StaticLendRing::new();
    block_on(sink.consume(rgba_2x2_frame(&ring))).expect("blit");

    let cmds = commands(&log.borrow());
    let expected: Vec<(u8, Vec<u8>)> = vec![
        (0x2A, vec![0, 1, 0, 2]),   // CASET: columns 1..=2
        (0x2B, vec![0, 80, 0, 81]), // RASET: rows 80..=81
        (
            0x2C, // RAMWR: red, green, blue, white as big-endian RGB565
            vec![0xF8, 0x00, 0x07, 0xE0, 0x00, 0x1F, 0xFF, 0xFF],
        ),
    ];
    assert_eq!(cmds, expected, "window + pixel stream");
}

/// A `width x rows` band lent from a ring: pixel 0 red, the rest black, so
/// each band's RGB565 stream is identifiable and sized (`width * rows * 2`).
fn band<const B: usize>(ring: &StaticLendRing<1, B>, bytes: usize) -> Frame {
    let mut slot = ring.acquire().expect("free slot");
    let buf = slot.buf_mut();
    for b in buf.iter_mut() {
        *b = 0;
    }
    if let [r, _g, _b, a, ..] = buf {
        *r = 255;
        *a = 255;
    }
    // SAFETY: the ring outlives the frame within each test.
    let slice = unsafe { slot.publish(bytes) };
    Frame::new(MemoryDomain::System(slice), FrameTiming::default(), 0)
}

/// The (row0, row1) inclusive window of each RASET (opcode 0x2B) in order.
fn raset_rows(cmds: &[(u8, Vec<u8>)]) -> Vec<(u16, u16)> {
    cmds.iter()
        .filter(|(op, _)| *op == 0x2B)
        .map(|(_, p)| {
            (
                u16::from_be_bytes([p[0], p[1]]),
                u16::from_be_bytes([p[2], p[3]]),
            )
        })
        .collect()
}

#[test]
fn banded_streaming_addresses_successive_vertical_windows() {
    // A 2x4 panel streamed in 2-row bands: two bands per refresh, then wrap.
    let (log, sink) = rig(2, 4);
    let mut sink = sink.with_stripe(2);
    sink.init(&mut MockDelay::default()).expect("init");
    log.borrow_mut().writes.clear();

    let ring: StaticLendRing<1, 16> = StaticLendRing::new(); // 2*2*4 = one band
    block_on(sink.consume(band(&ring, 16))).expect("band 0");
    block_on(sink.consume(band(&ring, 16))).expect("band 1");
    block_on(sink.consume(band(&ring, 16))).expect("band 2 (wrapped)");

    let cmds = commands(&log.borrow());
    // Band pixel stream: pixel 0 red (0xF800), the other 3 black.
    let px = vec![0xF8, 0x00, 0, 0, 0, 0, 0, 0];
    let expected: Vec<(u8, Vec<u8>)> = vec![
        (0x2A, vec![0, 0, 0, 1]), // band 0: cols 0..=1
        (0x2B, vec![0, 0, 0, 1]), //         rows 0..=1
        (0x2C, px.clone()),
        (0x2A, vec![0, 0, 0, 1]), // band 1: cols 0..=1
        (0x2B, vec![0, 2, 0, 3]), //         rows 2..=3
        (0x2C, px.clone()),
        (0x2A, vec![0, 0, 0, 1]), // band 2: cursor wrapped to the top
        (0x2B, vec![0, 0, 0, 1]), //         rows 0..=1 again
        (0x2C, px),
    ];
    assert_eq!(
        cmds, expected,
        "each band addresses the next vertical window, wrapping"
    );
}

#[test]
fn banded_streaming_covers_a_full_240x240_panel_from_a_tiny_ring() {
    // The real ESP32-P4-EYE case: a 240x240 panel a full RGBA frame (230 KB)
    // could never ring-buffer, streamed as 16-row bands (15 KB each).
    let (log, sink) = rig(240, 240);
    let mut sink = sink.with_stripe(16);
    sink.init(&mut MockDelay::default()).expect("init");
    log.borrow_mut().writes.clear();

    const BAND_BYTES: usize = 240 * 16 * 4;
    let ring: StaticLendRing<1, BAND_BYTES> = StaticLendRing::new();
    for _ in 0..(240 / 16) {
        block_on(sink.consume(band(&ring, BAND_BYTES))).expect("band");
    }

    let cmds = commands(&log.borrow());
    let rows = raset_rows(&cmds);
    let expected_rows: Vec<(u16, u16)> = (0..240).step_by(16).map(|y| (y, y + 15)).collect();
    assert_eq!(
        rows, expected_rows,
        "15 bands tile rows 0..240 in 16-row steps"
    );

    // Every panel pixel was written exactly once: 240*240 pixels * 2 RGB565
    // bytes, streamed through the fixed chunk buffer with no full framebuffer.
    let pixel_bytes: usize = cmds
        .iter()
        .filter(|(op, _)| *op == 0x2C)
        .map(|(_, p)| p.len())
        .sum();
    assert_eq!(
        pixel_bytes,
        240 * 240 * 2,
        "one full refresh, exactly one write per pixel"
    );
}

#[test]
fn banded_streaming_rejects_a_mistiled_stripe_and_wrong_sized_band() {
    // A stripe that does not divide the panel: the second band would run past
    // the last row, so it is rejected rather than addressing off-panel RAM.
    let (_log, sink) = rig(2, 3);
    let mut sink = sink.with_stripe(2);
    sink.init(&mut MockDelay::default()).expect("init");
    let ring: StaticLendRing<1, 16> = StaticLendRing::new();
    block_on(sink.consume(band(&ring, 16))).expect("band 0 (rows 0..=1)");
    // cursor is now 2; a 2-row band would need rows 2..=3 but height is 3.
    assert_eq!(
        block_on(sink.consume(band(&ring, 16))),
        Err(G2gError::CapsMismatch),
        "a stripe that does not tile the panel is rejected"
    );

    // A band whose payload is not width*stripe*4 is rejected too.
    let (_log2, sink2) = rig(2, 4);
    let mut sink2 = sink2.with_stripe(2);
    sink2.init(&mut MockDelay::default()).expect("init");
    let short: StaticLendRing<1, 16> = StaticLendRing::new();
    assert_eq!(
        block_on(sink2.consume(band(&short, 12))),
        Err(G2gError::CapsMismatch),
        "a wrong-sized band is rejected"
    );
}

#[test]
fn wrong_payload_size_is_caps_mismatch_before_any_spi_traffic() {
    let (log, mut sink) = rig(4, 4); // expects 64 bytes, frame carries 16
    sink.init(&mut MockDelay::default()).expect("init");
    log.borrow_mut().writes.clear();

    let ring: StaticLendRing<1, 16> = StaticLendRing::new();
    let err = block_on(sink.consume(rgba_2x2_frame(&ring)));
    assert_eq!(err, Err(G2gError::CapsMismatch));
    assert!(log.borrow().writes.is_empty(), "nothing reached the bus");
}

#[test]
fn uninitialized_sink_refuses_frames() {
    let (_, mut sink) = rig(2, 2);
    let ring: StaticLendRing<1, 16> = StaticLendRing::new();
    let err = block_on(sink.consume(rgba_2x2_frame(&ring)));
    assert_eq!(err, Err(G2gError::NotConfigured));
}

#[test]
fn spi_failure_surfaces_as_peripheral_error() {
    let (log, mut sink) = rig(2, 2);
    sink.init(&mut MockDelay::default()).expect("init");
    log.borrow_mut().fail_spi = true;

    let ring: StaticLendRing<1, 16> = StaticLendRing::new();
    let err = block_on(sink.consume(rgba_2x2_frame(&ring)));
    assert_eq!(err, Err(G2gError::Hardware(HardwareError::Peripheral)));
}
