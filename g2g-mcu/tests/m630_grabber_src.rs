//! M630: `GrabberSrc` (the `FrameGrabber` camera seam) captures into a
//! `StaticLendRing` and lends the frames downstream zero-copy. The mock
//! grabber stands in for the DMA peripheral only; the element under test is
//! real, and the integration test runs a whole camera -> SPI-display pipeline
//! on mock peripherals through the static runner.

use std::cell::RefCell;
use std::future::Future;
use std::rc::Rc;

use embedded_hal::spi::{ErrorKind, Operation, SpiDevice};
use g2g_core::error::{G2gError, HardwareError};
use g2g_core::memory::MemoryDomain;
use g2g_core::run_source_sink;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::StaticSource;
use g2g_mcu::{FrameGrabber, GrabberSrc, SpiDisplaySink};

fn block_on<F: Future>(fut: F) -> F::Output {
    g2g_core::drive_ready(fut).expect("the static chain never suspends")
}

fn leaked_ring<const N: usize, const B: usize>() -> &'static StaticLendRing<N, B> {
    Box::leak(Box::new(StaticLendRing::new()))
}

/// A test-pattern "camera": stamps the capture index into the first byte.
#[derive(Default)]
struct PatternGrabber {
    captures: u32,
}

impl FrameGrabber for PatternGrabber {
    async fn capture(&mut self, buf: &mut [u8]) -> Result<usize, G2gError> {
        if let Some(b) = buf.first_mut() {
            *b = (self.captures & 0xff) as u8;
        }
        self.captures += 1;
        Ok(buf.len())
    }
}

#[test]
fn captures_flow_with_sequence_and_pts() {
    let ring: &'static StaticLendRing<2, 8> = leaked_ring();
    let mut src = GrabberSrc::new(PatternGrabber::default(), ring, 33_000_000).with_frame_limit(3);

    for expect in 0u64..3 {
        let frame = block_on(src.next()).expect("capture").expect("frame");
        assert_eq!(frame.sequence, expect);
        assert_eq!(
            frame.timing.pts_ns,
            expect * 33_000_000,
            "interval-derived PTS"
        );
        let MemoryDomain::System(s) = &frame.domain else {
            panic!("system frame")
        };
        assert_eq!(
            s.as_slice().first().copied(),
            Some(expect as u8),
            "pattern payload"
        );
        assert_eq!(s.as_slice().len(), 8, "full-slot capture");
    }
    assert!(
        block_on(src.next()).expect("eos poll").is_none(),
        "frame limit ends the stream"
    );
}

#[test]
fn lends_ring_memory_zero_copy() {
    let ring: &'static StaticLendRing<2, 8> = leaked_ring();
    let mut src = GrabberSrc::new(PatternGrabber::default(), ring, 0).with_frame_limit(1);
    let frame = block_on(src.next()).expect("capture").expect("frame");
    let MemoryDomain::System(s) = &frame.domain else {
        panic!("system frame")
    };
    let ptr = s.as_slice().as_ptr();
    assert!(
        ring.contains(ptr),
        "payload aliases the ring slot: no copy was made"
    );
}

#[test]
fn exhausted_ring_is_a_sizing_error() {
    let ring: &'static StaticLendRing<2, 8> = leaked_ring();
    let mut src = GrabberSrc::new(PatternGrabber::default(), ring, 0);
    // Hold both lent frames so no slot can be reclaimed.
    let _a = block_on(src.next()).expect("capture").expect("frame");
    let _b = block_on(src.next()).expect("capture").expect("frame");
    assert_eq!(
        block_on(src.next()).expect_err("ring is full"),
        G2gError::PoolExhausted
    );
}

/// A grabber that violates its contract by claiming more bytes than the slot.
struct LyingGrabber;

impl FrameGrabber for LyingGrabber {
    async fn capture(&mut self, buf: &mut [u8]) -> Result<usize, G2gError> {
        Ok(buf.len() + 1)
    }
}

#[test]
fn oversized_capture_claim_is_rejected() {
    let ring: &'static StaticLendRing<2, 8> = leaked_ring();
    let mut src = GrabberSrc::new(LyingGrabber, ring, 0);
    assert_eq!(
        block_on(src.next()).expect_err("oversized claim"),
        G2gError::CapsMismatch
    );
}

/// A grabber whose peripheral fails.
struct FailingGrabber;

impl FrameGrabber for FailingGrabber {
    async fn capture(&mut self, _buf: &mut [u8]) -> Result<usize, G2gError> {
        Err(G2gError::Hardware(HardwareError::Peripheral))
    }
}

#[test]
fn capture_failure_propagates() {
    let ring: &'static StaticLendRing<2, 8> = leaked_ring();
    let mut src = GrabberSrc::new(FailingGrabber, ring, 0);
    assert_eq!(
        block_on(src.next()).expect_err("peripheral failure"),
        G2gError::Hardware(HardwareError::Peripheral)
    );
}

// --- camera -> display integration on mock peripherals ---

#[derive(Default)]
struct BusLog {
    writes: Vec<(bool, Vec<u8>)>,
    dc_high: bool,
}

struct MockSpi(Rc<RefCell<BusLog>>);

impl embedded_hal::spi::ErrorType for MockSpi {
    type Error = ErrorKind;
}

impl SpiDevice for MockSpi {
    fn transaction(&mut self, ops: &mut [Operation<'_, u8>]) -> Result<(), Self::Error> {
        let mut log = self.0.borrow_mut();
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

struct NoDelay;

impl embedded_hal::delay::DelayNs for NoDelay {
    fn delay_ns(&mut self, _ns: u32) {}
}

#[test]
fn camera_to_display_pipeline_runs_end_to_end() {
    // 2x2 RGBA frames: the grabber stamps the capture index into pixel 0's
    // red channel; the display converts to RGB565, so frame k's RAMWR opens
    // with the big-endian pixel (k & 0xF8) << 8.
    let ring: &'static StaticLendRing<2, 16> = leaked_ring();
    let src = GrabberSrc::new(PatternGrabber::default(), ring, 0).with_frame_limit(9);

    let log = Rc::new(RefCell::new(BusLog::default()));
    let mut sink = SpiDisplaySink::st7789(MockSpi(log.clone()), MockDc(log.clone()), 2, 2);
    sink.init(&mut NoDelay).expect("init");
    log.borrow_mut().writes.clear();

    block_on(run_source_sink(src, &mut sink)).expect("pipeline");

    // 9 frames x (CASET + RASET + RAMWR + pixel data): pick out each RAMWR's
    // first pixel and check the captured pattern reached the panel.
    let log = log.borrow();
    let mut ramwr_first_px = Vec::new();
    let mut after_ramwr = false;
    for (dc_high, bytes) in &log.writes {
        if !*dc_high {
            after_ramwr = bytes == &[0x2C];
        } else if after_ramwr {
            ramwr_first_px.push((bytes[0], bytes[1]));
            after_ramwr = false;
        }
    }
    let expected: Vec<(u8, u8)> = (0u8..9).map(|k| (k & 0xF8, 0)).collect();
    assert_eq!(
        ramwr_first_px, expected,
        "each captured frame reached the panel as RGB565"
    );
}
