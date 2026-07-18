//! M634: the `forbid(unsafe)` posture proof. This test crate is compiled
//! under `#![forbid(unsafe_code)]` (a hard error even for `#[allow]`d unsafe,
//! so it cannot be smuggled back in) and still builds and runs a complete
//! camera -> display pipeline through the safe g2g surface:
//!
//! - the DMA ring is a `static` (`StaticLendRing::new` is `const`), so
//!   `GrabberSrc::new` (the safe constructor) applies: `'static` makes the
//!   zero-copy lend sound by construction;
//! - the elements, runner, and executor (`drive_ready`) are all safe APIs.
//!
//! The point for a safety-critical shop: application code on the g2g MCU
//! surface needs no `unsafe` at all; every `unsafe` block stays inside the
//! framework, each with a documented contract.

#![forbid(unsafe_code)]

use std::cell::RefCell;
use std::rc::Rc;

use embedded_hal::delay::DelayNs;
use embedded_hal::spi::{ErrorKind, Operation, SpiDevice};
use g2g_core::error::G2gError;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{drive_ready, run_source_sink};
use g2g_mcu::{FrameGrabber, GrabberSrc, SpiDisplaySink};

/// The application's DMA ring, in a `static` as on a real MCU. `const new`
/// plus the ring's `Sync` bound make this possible without any unsafe.
static RING: StaticLendRing<2, 16> = StaticLendRing::new();

/// A test-pattern camera adapter (a real one arms DMA and awaits completion).
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

#[derive(Default)]
struct BusLog {
    data_bytes: Vec<u8>,
    dc_high: bool,
}

struct MockSpi(Rc<RefCell<BusLog>>);

impl embedded_hal::spi::ErrorType for MockSpi {
    type Error = ErrorKind;
}

impl SpiDevice for MockSpi {
    fn transaction(&mut self, ops: &mut [Operation<'_, u8>]) -> Result<(), Self::Error> {
        let mut log = self.0.borrow_mut();
        if log.dc_high {
            for op in ops.iter() {
                if let Operation::Write(bytes) = op {
                    log.data_bytes.extend_from_slice(bytes);
                }
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

impl DelayNs for NoDelay {
    fn delay_ns(&mut self, _ns: u32) {}
}

#[test]
fn whole_pipeline_builds_and_runs_without_unsafe() {
    // Camera over the static ring: the SAFE GrabberSrc constructor.
    let src = GrabberSrc::new(PatternGrabber::default(), &RING, 33_333_333).with_frame_limit(4);

    let log = Rc::new(RefCell::new(BusLog::default()));
    let mut sink = SpiDisplaySink::st7789(MockSpi(log.clone()), MockDc(log.clone()), 2, 2);
    sink.init(&mut NoDelay).expect("init");
    log.borrow_mut().data_bytes.clear();

    drive_ready(run_source_sink(src, &mut sink))
        .expect("static chain never suspends")
        .expect("pipeline");

    // 4 frames x (CASET 4 + RASET 4 + RAMWR pixels 8) bytes with D/C high.
    assert_eq!(
        log.borrow().data_bytes.len(),
        4 * (4 + 4 + 8),
        "all frames reached the panel"
    );
}
