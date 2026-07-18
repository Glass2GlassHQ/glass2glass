//! M660: the hardware-H.264 encoder seam. A scripted mock replays the encoder
//! peripheral contract (accept one raw I420 frame, emit one Annex-B access unit,
//! report its byte count + keyframe flag), so `HwH264Enc`'s real logic is
//! asserted on the host: I420 geometry validation before any peripheral traffic,
//! verbatim access-unit delivery, the output-size / keyframe bookkeeping, fault
//! surfacing, and ring back-pressure. The `CH264Encoder` C-seam is driven
//! through a real `extern "C"` callback and must produce byte-identical output,
//! proving the zero-Rust hardware path is transparent. A `camera -> encode`
//! integration drives the whole chain over `GrabberSrc`. What a mock cannot
//! prove, that real ESP32-P4 silicon agrees with its datasheet, is the deferred
//! on-device `Hardware` conformance row.

mod util;

use core::ffi::c_void;

use g2g_core::error::G2gError;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::StaticTransform;
use g2g_mcu::hwh264::i420_len;
use g2g_mcu::{CH264Encoder, H264EncodeInfo, H264Encoder, HwH264Enc};
use util::{block_on, frame_of, payload};

/// 16x16 I420 = 256 luma + 2*64 chroma = 384 bytes.
const W: u16 = 16;
const H: u16 = 16;
const RAW: usize = 384;
/// Output slot: an access unit is far smaller than the raw frame here.
const OUT_BYTES: usize = 256;

/// The Annex-B access unit our scripted encoder emits for the `n`th frame: a
/// 4-byte start code, a NAL header (IDR type 5 on keyframes, non-IDR type 1
/// otherwise), then two payload bytes derived from the input so the mock and
/// the C seam must agree byte-for-byte.
fn expected_au(first_raw_byte: u8, n: u8, keyframe: bool) -> Vec<u8> {
    let nal_header = if keyframe { 0x65 } else { 0x41 };
    vec![0x00, 0x00, 0x00, 0x01, nal_header, first_raw_byte, n]
}

/// A scripted encoder: keyframe every 4th frame, output per [`expected_au`],
/// counting frames it is fed.
struct MockEnc {
    n: u8,
    fail: bool,
}

impl H264Encoder for &mut MockEnc {
    async fn encode(&mut self, raw: &[u8], out: &mut [u8]) -> Result<H264EncodeInfo, G2gError> {
        if self.fail {
            return Err(G2gError::Hardware(
                g2g_core::error::HardwareError::Peripheral,
            ));
        }
        let keyframe = self.n % 4 == 0;
        let au = expected_au(raw.first().copied().unwrap_or(0), self.n, keyframe);
        // The adapter contract: an undersized output buffer must fail.
        if out.len() < au.len() {
            return Err(G2gError::CapsMismatch);
        }
        out[..au.len()].copy_from_slice(&au);
        self.n = self.n.wrapping_add(1);
        Ok(H264EncodeInfo {
            len: au.len(),
            keyframe,
        })
    }
}

/// A raw I420 frame whose first byte is `tag` (the rest zeroed), so the encoded
/// access unit is identifiable.
fn i420_frame(ring: &StaticLendRing<1, RAW>, tag: u8, seq: u64) -> g2g_core::frame::Frame {
    let mut buf = [0u8; RAW];
    buf[0] = tag;
    frame_of(ring, &buf, seq * 1000, seq)
}

#[test]
fn i420_sizing_is_checked() {
    assert_eq!(i420_len(16, 16), Some(384));
    assert_eq!(i420_len(1920, 1080), Some(1920 * 1080 * 3 / 2));
    assert_eq!(i420_len(0, 16), None, "zero dimension");
    assert_eq!(i420_len(15, 16), None, "odd width is not 4:2:0");
    assert_eq!(i420_len(16, 15), None, "odd height is not 4:2:0");
}

#[test]
fn encodes_each_frame_to_its_access_unit_with_keyframe_cadence() {
    let mut enc = MockEnc { n: 0, fail: false };
    let ring: StaticLendRing<1, OUT_BYTES> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame within this test.
    let mut hw =
        unsafe { HwH264Enc::with_ring(&mut enc, W, H, &ring) }.expect("valid 4:2:0 geometry");
    let src: StaticLendRing<1, RAW> = StaticLendRing::new();

    for n in 0..6u8 {
        let out = block_on(hw.process(i420_frame(&src, 0xA0 + n, n as u64)))
            .expect("encode ok")
            .expect("one access unit per frame");
        let keyframe = n % 4 == 0;
        assert_eq!(
            payload(&out),
            expected_au(0xA0 + n, n, keyframe),
            "frame {n}: the access unit is delivered verbatim"
        );
        assert_eq!(
            out.sequence, n as u64,
            "sequence carried from the raw frame"
        );
        assert_eq!(
            hw.info(),
            Some(H264EncodeInfo { len: 7, keyframe }),
            "frame {n}: reported info recorded"
        );
    }
}

#[test]
fn wrong_sized_input_is_rejected_before_the_peripheral() {
    let mut enc = MockEnc { n: 0, fail: false };
    let ring: StaticLendRing<1, OUT_BYTES> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame within this test.
    let mut hw = unsafe { HwH264Enc::with_ring(&mut enc, W, H, &ring) }.expect("geometry");
    let src: StaticLendRing<1, RAW> = StaticLendRing::new();
    // A 383-byte "frame" is not a whole 16x16 I420 (384 bytes).
    let short = block_on(hw.process(frame_of(&src, &[0u8; 383], 0, 0)));
    assert!(
        matches!(short, Err(G2gError::CapsMismatch)),
        "a partial capture never reaches the encoder"
    );
    assert_eq!(enc.n, 0, "the encoder was not called");
}

#[test]
fn invalid_geometry_has_no_element() {
    let mut enc = MockEnc { n: 0, fail: false };
    let ring: StaticLendRing<1, OUT_BYTES> = StaticLendRing::new();
    // SAFETY: the ring outlives the (never-published) element.
    let bad = unsafe { HwH264Enc::with_ring(&mut enc, 15, 16, &ring) };
    assert!(bad.is_none(), "odd width rejected at construction");
}

// --- The C-seam (CH264Encoder): a real extern "C" encoder driving the same
// element must produce byte-identical output. ------------------------------

/// C encoder state: a frame counter, mirroring `MockEnc`'s cadence.
#[repr(C)]
struct CEncCtx {
    n: u8,
}

/// The C hardware-encoder driver stand-in: same contract as `MockEnc`, through
/// the raw-pointer FFI ABI `CH264Encoder` expects.
unsafe extern "C" fn c_encode(
    ctx: *mut c_void,
    raw: *const u8,
    raw_len: usize,
    out: *mut u8,
    out_cap: usize,
    keyframe: *mut i32,
) -> isize {
    // SAFETY: `ctx` is the `CEncCtx` the test passed to `CH264Encoder::new`.
    let ctx = unsafe { &mut *(ctx as *mut CEncCtx) };
    // SAFETY: the EncodeFn contract: `raw` is valid for `raw_len` bytes.
    let first = if raw_len > 0 { unsafe { *raw } } else { 0 };
    let kf = ctx.n % 4 == 0;
    let au = expected_au(first, ctx.n, kf);
    if out_cap < au.len() {
        return -1;
    }
    // SAFETY: `out` is valid for `out_cap` (>= au.len()) bytes and disjoint
    // from the local `au`.
    unsafe { core::ptr::copy_nonoverlapping(au.as_ptr(), out, au.len()) };
    // SAFETY: `keyframe` is a valid out-pointer per the EncodeFn contract.
    unsafe { *keyframe = i32::from(kf) };
    ctx.n = ctx.n.wrapping_add(1);
    au.len() as isize
}

#[test]
fn c_seam_encoder_is_byte_transparent() {
    let mut ctx = CEncCtx { n: 0 };
    // SAFETY: `c_encode` is a valid function; `ctx` outlives the encoder.
    let cenc = unsafe { CH264Encoder::new(c_encode, &mut ctx as *mut _ as *mut c_void) };
    let ring: StaticLendRing<1, OUT_BYTES> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame within this test.
    let mut hw = unsafe { HwH264Enc::with_ring(cenc, W, H, &ring) }.expect("geometry");
    let src: StaticLendRing<1, RAW> = StaticLendRing::new();

    for n in 0..6u8 {
        let out = block_on(hw.process(i420_frame(&src, 0xA0 + n, n as u64)))
            .expect("encode ok")
            .expect("one access unit");
        assert_eq!(
            payload(&out),
            expected_au(0xA0 + n, n, n % 4 == 0),
            "frame {n}: the C-seam encoder matches the Rust mock byte-for-byte"
        );
    }
}

// --- camera -> encode integration over the real GrabberSrc runner. ---------

use g2g_core::run_source_transform_sink;
use g2g_core::StaticSink;
use g2g_mcu::{FrameGrabber, GrabberSrc};

/// A mock I420 camera: fills each frame with a per-capture tag byte.
struct I420Camera {
    n: u8,
}

impl FrameGrabber for I420Camera {
    async fn capture(&mut self, buf: &mut [u8]) -> Result<usize, G2gError> {
        for b in buf.iter_mut() {
            *b = 0;
        }
        if let Some(first) = buf.first_mut() {
            *first = 0xA0 + self.n;
        }
        self.n = self.n.wrapping_add(1);
        Ok(buf.len())
    }
}

/// Collects each encoded access unit.
#[derive(Default)]
struct CollectAus {
    aus: Vec<Vec<u8>>,
}

impl StaticSink for &mut CollectAus {
    async fn consume(&mut self, frame: g2g_core::frame::Frame) -> Result<(), G2gError> {
        self.aus.push(payload(&frame).to_vec());
        Ok(())
    }
}

#[test]
fn camera_to_encode_pipeline_runs_end_to_end() {
    let src_ring: StaticLendRing<2, RAW> = StaticLendRing::new();
    // SAFETY: rings outlive the pipeline (drained before this scope ends).
    let source = unsafe { GrabberSrc::with_ring(I420Camera { n: 0 }, &src_ring, 33_333_333) }
        .with_frame_limit(4);
    let mut enc = MockEnc { n: 0, fail: false };
    let out_ring: StaticLendRing<2, OUT_BYTES> = StaticLendRing::new();
    // SAFETY: rings outlive the pipeline (drained before this scope ends).
    let hw = unsafe { HwH264Enc::with_ring(&mut enc, W, H, &out_ring) }.expect("geometry");
    let mut sink = CollectAus::default();

    block_on(run_source_transform_sink(source, hw, &mut sink)).expect("pipeline runs");

    let expected: Vec<Vec<u8>> = (0..4u8)
        .map(|n| expected_au(0xA0 + n, n, n % 4 == 0))
        .collect();
    assert_eq!(
        sink.aus, expected,
        "camera -> HW H.264 encode delivers the AU stream"
    );
    // The first frame is an IDR keyframe (random-access point).
    assert!(sink.aus[0].contains(&0x65), "frame 0 is an IDR");
}
