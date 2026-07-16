//! The heap-free proof pipeline (shared by `g2g-noalloc` and `g2g-qemu`): a
//! real [`GrabberSrc`] capture source (M630, the camera seam from `g2g-mcu`,
//! fed by a stub pattern grabber over a `StaticLendRing`) -> pass-through
//! transform -> a real [`SpiDisplaySink`] (M629, the ST7789-family element)
//! driving a stub SPI bus, wired with the static element model. The transform
//! link carries negotiated `Caps::Tensor` (M636: `TensorShape` is fixed-rank
//! inline, so tensor caps are heap-free): the caps intersect at build and
//! every frame is validated against the negotiated tensor's byte size, so the
//! ML caps kind is covered by the same no-alloc / panic-free / footprint /
//! QEMU proofs as the media pipeline. Wired with the static element model
//! (`g2g_core::staticelem`) and driven to completion by a single noop-waker
//! poll. Every stage is a concrete type, so the whole chain monomorphizes to
//! unboxed `async` state machines: no `dyn`, no `Box`, no allocation, and
//! (with `g2g-core` default-features=false) no `alloc` crate anywhere in the
//! graph. Because the sink is the real display element, every proof built on
//! this crate (no-alloc + panic-free symbols, footprint budgets, the QEMU
//! Cortex-M run) covers an actual peripheral element, not a toy stage.
//!
//! Every reachable path is also panic-free: no unwraps, no slice-index bounds
//! panics, no overflow / division panics, and the single-poll executor lets
//! the optimizer discharge the compiler's resumed-after-completion guard.
//! `tools/noalloc-check.sh` asserts both properties on the `g2g-noalloc`
//! archive that wraps this crate.
//!
//! The stub bus checksums every byte the element sends with D/C high (command
//! parameters and RAMWR pixel data), so a run is verifiable end to end:
//! [`run`] returns [`EXPECTED_CHECKSUM`] only if the init sequence, the
//! window addressing, and the RGBA -> RGB565 conversion all put the right
//! bytes on the wire for all 64 frames.

#![no_std]

pub mod audio;

// Re-exported so the leaf proof binaries (g2g-qemu and friends) need no
// direct g2g-core dependency for the executor.
pub use g2g_core::drive_ready;

use core::cell::Cell;
use core::convert::Infallible;

use embedded_hal::delay::DelayNs;
use embedded_hal::digital::OutputPin;
use embedded_hal::spi::{ErrorKind, Operation, SpiDevice};
use g2g_core::error::G2gError;
use g2g_core::frame::Frame;
use g2g_core::memory::MemoryDomain;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{
    run_source_sink, run_source_transform_sink, Caps, StaticTransform, TensorDType, TensorLayout,
    TensorShape,
};
use g2g_mcu::{FrameGrabber, GrabberSrc, SpiDisplaySink};

const SLOTS: usize = 4;
/// 4x4 RGBA frames: one ring slot is exactly one frame payload.
const WIDTH: u16 = 4;
const HEIGHT: u16 = 4;
const BYTES: usize = (WIDTH as usize) * (HEIGHT as usize) * 4;
const FRAMES: u32 = 64;
/// Nominal 30 fps, feeding `GrabberSrc`'s interval-derived PTS.
const FRAME_INTERVAL_NS: u64 = 33_333_333;

/// The checksum [`run`] produces: the sum of every byte the display element
/// sends with D/C high. Derived from the ST7789 wire protocol:
/// - init parameters: COLMOD `0x55` + MADCTL `0x00`;
/// - per frame: CASET `[0,0,0,3]` + RASET `[0,0,0,3]` (the 4x4 window) = 6,
///   plus the RAMWR pixel stream, where pixel 0 is `(seq, 0, 0)` RGBA (the
///   source stamps the sequence into the red channel) so its big-endian
///   RGB565 encoding contributes `seq & 0xF8`, and the other 15 pixels are
///   black (all zero bytes).
pub const EXPECTED_CHECKSUM: u64 = {
    let mut sum: u64 = 0x55 + 0x00;
    let mut seq: u64 = 0;
    while seq < FRAMES as u64 {
        sum += 6 + (seq & 0xF8);
        seq += 1;
    }
    sum
};

/// The stub "camera": stamps the capture index into pixel 0's red channel
/// (standing in for a DMA fill). `first_mut`, not `[0]`: indexing would link a
/// bounds-check panic path, and the wrapping archive proves it is panic-free.
/// `pub` (like audio's [`audio::tone_a`]) so the generated-graph proof
/// (`mcugen-graphs`) can drive its emitted display pipeline with the same mock
/// capture peripheral this reference uses.
pub struct PatternGrabber {
    captures: u64,
}

/// The stub camera peripheral, starting at capture index 0.
pub fn pattern_grabber() -> PatternGrabber {
    PatternGrabber { captures: 0 }
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

/// The tensor description of one RGBA frame on the transform link: an NHWC u8
/// tensor `[1, HEIGHT, WIDTH, 4]`. `TensorShape` is fixed-rank inline (M636),
/// so tensor caps are part of the no-alloc subset and the heap-free archive
/// can carry them.
fn frame_tensor_caps() -> Caps {
    Caps::Tensor {
        dtype: TensorDType::U8,
        shape: TensorShape::new([1, HEIGHT as u32, WIDTH as u32, 4]),
        layout: TensorLayout::Nhwc,
    }
}

/// Negotiate the transform link's tensor caps (produced side vs accepted
/// side) and size one frame under them (element count x dtype size). Both
/// sides pass through `black_box`, so the full `Caps::intersect` (every caps
/// kind's arm, `intersect` is not inlined) plus the tensor sizing are
/// genuinely in the archive, covered by the no-alloc + panic-free symbol
/// proofs rather than constant-folded away. `None` when the sides do not
/// intersect.
fn negotiate_frame_bytes() -> Option<usize> {
    let produced = core::hint::black_box(frame_tensor_caps());
    let accepted = core::hint::black_box(frame_tensor_caps());
    match produced.intersect(&accepted).ok()? {
        Caps::Tensor { dtype, shape, .. } => Some(shape.elements().saturating_mul(dtype.size())),
        _ => None,
    }
}

/// A pass-through transform that validates each frame against the link's
/// negotiated tensor caps (so the stage is real work, not elided) and
/// forwards the frame.
struct Touch {
    /// Byte size a frame must have under the negotiated `Caps::Tensor`.
    frame_bytes: usize,
}

impl StaticTransform for Touch {
    async fn process(&mut self, input: Frame) -> Result<Option<Frame>, G2gError> {
        // The payload must be exactly the negotiated tensor: a mismatch fails
        // the pipeline (and with it the checksum comparison) honestly.
        if let MemoryDomain::System(s) = &input.domain {
            if s.as_slice().len() != self.frame_bytes {
                return Err(G2gError::CapsMismatch);
            }
            let _ = core::hint::black_box(s.as_slice().first().copied());
        }
        Ok(Some(input))
    }
}

/// The stub 4-wire panel bus: a D/C line plus an SPI device that checksums
/// every byte written while D/C is high (an emulated panel's view of the
/// element's parameters + pixel stream). Shared by reference, no allocation.
/// `pub` (with `spi`/`dc`/`checksum` accessors) so the `mcugen-graphs` proof
/// can hand the generated display pipeline the same stub panel this reference
/// drives, then read back the wire checksum.
pub struct StubBus {
    sum: Cell<u64>,
    dc_high: Cell<bool>,
}

impl StubBus {
    /// A fresh stub panel: zero checksum, D/C low.
    pub fn new() -> Self {
        StubBus { sum: Cell::new(0), dc_high: Cell::new(false) }
    }

    /// The SPI-device seam for this bus (an immutable borrow: the checksum
    /// lives in `Cell`s, so `spi`, `dc`, and [`Self::checksum`] coexist).
    pub fn spi(&self) -> StubSpi<'_> {
        StubSpi(self)
    }

    /// The D/C-pin seam for this bus.
    pub fn dc(&self) -> StubDc<'_> {
        StubDc(self)
    }

    /// The wire checksum accumulated so far (matches [`EXPECTED_CHECKSUM`]
    /// after a correct run of the 64-frame display pipeline).
    pub fn checksum(&self) -> u64 {
        self.sum.get()
    }
}

impl Default for StubBus {
    fn default() -> Self {
        Self::new()
    }
}

/// The SPI-device seam over a [`StubBus`] (obtain via [`StubBus::spi`]).
pub struct StubSpi<'b>(&'b StubBus);

impl embedded_hal::spi::ErrorType for StubSpi<'_> {
    type Error = ErrorKind;
}

impl SpiDevice for StubSpi<'_> {
    fn transaction(&mut self, ops: &mut [Operation<'_, u8>]) -> Result<(), Self::Error> {
        if self.0.dc_high.get() {
            for op in ops.iter() {
                if let Operation::Write(bytes) = op {
                    for b in bytes.iter() {
                        self.0.sum.set(self.0.sum.get().wrapping_add(*b as u64));
                    }
                }
            }
        }
        Ok(())
    }
}

/// The D/C-pin seam over a [`StubBus`] (obtain via [`StubBus::dc`]).
pub struct StubDc<'b>(&'b StubBus);

impl embedded_hal::digital::ErrorType for StubDc<'_> {
    type Error = Infallible;
}

impl OutputPin for StubDc<'_> {
    fn set_low(&mut self) -> Result<(), Self::Error> {
        self.0.dc_high.set(false);
        Ok(())
    }
    fn set_high(&mut self) -> Result<(), Self::Error> {
        self.0.dc_high.set(true);
        Ok(())
    }
}

/// The stub panel needs no reset/wake time; real boards pass their HAL timer.
/// `pub` so the `mcugen-graphs` proof can supply it as the generated display
/// pipeline's delay seam.
pub struct NoDelay;

impl DelayNs for NoDelay {
    fn delay_ns(&mut self, _ns: u32) {}
}

/// Build and run the camera -> transform -> SPI-display pipeline over a
/// stack-local `StaticLendRing`, driving whatever 4-wire panel the caller
/// hands it: an `embedded-hal` 1.0 [`SpiDevice`], a D/C [`OutputPin`], and a
/// [`DelayNs`] for the ST7789 reset/wake timings. The stub bus below is one
/// backend (used by every heap-free proof); a real board passes its HAL's SPI
/// device + GPIO + timer instead, unchanged, which is exactly how the
/// `examples/g2g-esp32p4` esp-hal harness reuses this pipeline on RISC-V
/// silicon. Generic, so it monomorphizes per backend with no `dyn`/`Box` and
/// stays in the no-alloc subset. Returns `Ok` when all [`FRAMES`] frames were
/// captured, converted, and pushed to the panel.
pub async fn run_display_with<SPI, DC, D>(
    spi: SPI,
    dc: DC,
    delay: &mut D,
) -> Result<(), G2gError>
where
    SPI: SpiDevice,
    DC: OutputPin,
    D: DelayNs,
{
    // Negotiate the transform link's tensor caps up front; no intersection
    // means no pipeline.
    let frame_bytes = negotiate_frame_bytes().ok_or(G2gError::CapsMismatch)?;
    let ring: StaticLendRing<SLOTS, BYTES> = StaticLendRing::new();
    // SAFETY: `ring` outlives every lent frame: the runner drains the pipeline
    // (each frame is dropped by the sink) before this future completes and
    // drops the ring, per `with_ring`'s contract.
    let source = unsafe {
        GrabberSrc::with_ring(pattern_grabber(), &ring, FRAME_INTERVAL_NS)
    }
    .with_frame_limit(FRAMES);
    let mut sink = SpiDisplaySink::st7789(spi, dc, WIDTH, HEIGHT);
    sink.init(delay)?;
    // `&mut sink` (a StaticSink via the blanket impl).
    run_source_transform_sink(source, Touch { frame_bytes }, &mut sink).await
}

/// The proof pipeline over the stub panel, returning that bus's wire checksum
/// ([`EXPECTED_CHECKSUM`] when everything worked). A plain future, so a real
/// executor can drive it (an Embassy task awaits this in `g2g-embassy`);
/// [`run`] wraps it for executor-less callers. Delegates to
/// [`run_display_with`]: the stub bus is just the backend the proofs pin, so
/// the board-agnostic runner is what every no-alloc / panic-free / footprint /
/// QEMU proof exercises.
pub async fn run_async() -> u64 {
    let bus = StubBus::new();
    // The stub panel needs no reset/wake time.
    let mut delay = NoDelay;
    // A run error still returns the checksum accumulated so far, which fails
    // the comparison honestly; the bus keeps it readable after the borrow.
    let _ = run_display_with(bus.spi(), bus.dc(), &mut delay).await;
    bus.sum.get()
}

/// The ESP32-P4-EYE's 1.54" ST7789 panel and the banding used to stream it:
/// 240x240 refreshed [`STRIPE`] rows at a time, so the pipeline ring holds one
/// band (240x16x4 = 15 KB), never a 230 KB framebuffer.
pub const PANEL_W: u16 = 240;
/// Panel height (see [`PANEL_W`]).
pub const PANEL_H: u16 = 240;
/// Rows per band; must divide [`PANEL_H`].
pub const STRIPE: u16 = 16;
const BAND_BYTES: usize = (PANEL_W as usize) * (STRIPE as usize) * 4;
/// Bands in one full-screen refresh.
pub const BANDS_PER_REFRESH: u32 = (PANEL_H / STRIPE) as u32;

/// Stream one full 240x240 refresh to a banded [`SpiDisplaySink`] over the
/// caller's panel (the Tier-1.5 full-panel path): [`BANDS_PER_REFRESH`] frames
/// of `PANEL_W x STRIPE` RGBA, each written to the next vertical window, so a
/// large panel is refreshed from an MCU-small ring. Generic over the same
/// `embedded-hal` seams as [`run_display_with`], so the esp-hal harness drives
/// the real panel by passing its SPI device / GPIO / timer.
pub async fn run_display_banded_with<SPI, DC, D>(
    spi: SPI,
    dc: DC,
    delay: &mut D,
) -> Result<(), G2gError>
where
    SPI: SpiDevice,
    DC: OutputPin,
    D: DelayNs,
{
    let ring: StaticLendRing<2, BAND_BYTES> = StaticLendRing::new();
    // SAFETY: the ring outlives every lent band (the pipeline drains before the
    // future completes and drops the ring, per with_ring's contract).
    let source = unsafe { GrabberSrc::with_ring(pattern_grabber(), &ring, FRAME_INTERVAL_NS) }
        .with_frame_limit(BANDS_PER_REFRESH);
    let mut sink = SpiDisplaySink::st7789(spi, dc, PANEL_W, PANEL_H).with_stripe(STRIPE);
    sink.init(delay)?;
    run_source_sink(source, &mut sink).await
}

/// [`run_async`] driven by `g2g_core`'s safe single-poll executor
/// ([`drive_ready`]; the static chain never suspends, so one poll completes
/// it, and single-polling is what keeps the archive panic-free). A `Pending`
/// pipeline yields 0, which fails the checksum comparison honestly.
pub fn run() -> u64 {
    drive_ready(run_async()).unwrap_or(0)
}
