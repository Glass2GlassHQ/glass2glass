//! M640: the hardware-JPEG codec seam. A scripted mock replays the STM32H7
//! peripheral contract (accept one JFIF stream, report the header-derived
//! image parameters, emit MCU-block-ordered bytes), so `HwJpegDec`'s real
//! logic is asserted on the host: framing validation before any peripheral
//! traffic, verbatim bitstream delivery, the MCU-tiling output-size
//! cross-check (checked math over header-derived dimensions), fault
//! surfacing, and ring back-pressure. What a mock cannot prove, that real
//! silicon agrees with its datasheet, is exactly the deferred on-device
//! `Hardware` conformance row.

mod util;

use g2g_core::error::{G2gError, HardwareError};
use g2g_core::staticpool::StaticLendRing;
use g2g_core::StaticTransform;
use g2g_mcu::hwjpeg::decoded_len;
use g2g_mcu::{HwJpegDec, JpegDecoder, JpegImageInfo, JpegSubsampling};
use util::{block_on, frame_of, payload};

/// A scripted peripheral: reports `info`, writes `emit` bytes of a counting
/// pattern, and records every bitstream it is fed.
struct MockJpeg {
    info: JpegImageInfo,
    emit: usize,
    fed: Vec<Vec<u8>>,
    fail: bool,
}

impl MockJpeg {
    fn new(info: JpegImageInfo) -> Self {
        Self {
            info,
            emit: decoded_len(info).unwrap(),
            fed: Vec::new(),
            fail: false,
        }
    }
}

impl JpegDecoder for &mut MockJpeg {
    async fn decode(
        &mut self,
        jpeg: &[u8],
        out: &mut [u8],
    ) -> Result<(JpegImageInfo, usize), G2gError> {
        if self.fail {
            return Err(G2gError::Hardware(HardwareError::Peripheral));
        }
        self.fed.push(jpeg.to_vec());
        // The adapter contract: an undersized output buffer must fail.
        if out.len() < self.emit {
            return Err(G2gError::CapsMismatch);
        }
        for (i, b) in out.iter_mut().take(self.emit).enumerate() {
            *b = (i % 251) as u8;
        }
        Ok((self.info, self.emit))
    }
}

/// A minimal well-framed stand-in bitstream (SOI ... EOI).
fn jpeg_bytes(body: &[u8]) -> Vec<u8> {
    let mut v = vec![0xFF, 0xD8];
    v.extend_from_slice(body);
    v.extend_from_slice(&[0xFF, 0xD9]);
    v
}

const QCIF_420: JpegImageInfo = JpegImageInfo {
    width: 176,
    height: 144,
    subsampling: JpegSubsampling::Ycbcr420,
};

#[test]
fn mcu_tiling_output_sizes() {
    let len = |w, h, s| {
        decoded_len(JpegImageInfo {
            width: w,
            height: h,
            subsampling: s,
        })
    };
    // Exact multiples.
    assert_eq!(len(16, 16, JpegSubsampling::Ycbcr420), Some(384));
    assert_eq!(len(8, 8, JpegSubsampling::Gray), Some(64));
    assert_eq!(len(8, 8, JpegSubsampling::Ycbcr444), Some(192));
    assert_eq!(len(16, 8, JpegSubsampling::Ycbcr422), Some(256));
    // Partial MCUs round up to whole tiles.
    assert_eq!(len(17, 9, JpegSubsampling::Ycbcr422), Some(2 * 2 * 256));
    assert_eq!(len(1, 1, JpegSubsampling::Ycbcr420), Some(384));
    // QCIF 4:2:0: 11 x 9 MCUs.
    assert_eq!(len(176, 144, JpegSubsampling::Ycbcr420), Some(11 * 9 * 384));
    // Header-derived dimensions are untrusted: zero is malformed, the
    // largest legal header sizes without overflow.
    assert_eq!(len(0, 144, JpegSubsampling::Ycbcr420), None);
    assert_eq!(len(176, 0, JpegSubsampling::Gray), None);
    assert!(len(u16::MAX, u16::MAX, JpegSubsampling::Ycbcr444).is_some());
}

#[test]
fn decodes_and_exposes_info() {
    let mut mock = MockJpeg::new(QCIF_420);
    let expect_len = mock.emit;
    let input_ring: StaticLendRing<1, 64> = StaticLendRing::new();
    let out_ring: StaticLendRing<2, { 11 * 9 * 384 }> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame in this test.
    let mut dec = unsafe { HwJpegDec::with_ring(&mut mock, &out_ring) };
    assert_eq!(dec.info(), None, "no decode yet");

    let jpeg = jpeg_bytes(&[1, 2, 3, 4]);
    let input = frame_of(&input_ring, &jpeg, 321, 4);
    let out = block_on(dec.process(input))
        .expect("decode")
        .expect("frame");
    assert_eq!(
        payload(&out).len(),
        expect_len,
        "one QCIF 4:2:0 block stream"
    );
    assert_eq!(
        payload(&out)[..5],
        [0, 1, 2, 3, 4],
        "the peripheral's bytes, zero-copy"
    );
    assert_eq!(out.timing.pts_ns, 321, "timing inherited");
    assert_eq!(out.sequence, 4, "sequence inherited");
    assert_eq!(
        dec.info(),
        Some(QCIF_420),
        "header-derived parameters exposed"
    );
    drop(out);
    let fed = dec.free();
    assert_eq!(fed.fed, vec![jpeg], "bitstream delivered verbatim");
}

#[test]
fn framing_is_validated_before_the_peripheral() {
    let mut mock = MockJpeg::new(QCIF_420);
    let input_ring: StaticLendRing<1, 64> = StaticLendRing::new();
    let out_ring: StaticLendRing<1, { 11 * 9 * 384 }> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame in this test.
    let mut dec = unsafe { HwJpegDec::with_ring(&mut mock, &out_ring) };

    for bad in [
        &[][..],
        &[0xFF][..],
        &[0xFF, 0xD8, 0, 0][..],
        &[0, 0, 0xFF, 0xD9][..],
    ] {
        let input = frame_of(&input_ring, bad, 0, 0);
        assert_eq!(
            block_on(dec.process(input)).unwrap_err(),
            G2gError::CapsMismatch
        );
    }
    assert!(
        dec.free().fed.is_empty(),
        "no peripheral traffic for bad framing"
    );
}

#[test]
fn peripheral_disagreeing_with_its_header_is_a_fault() {
    let mut mock = MockJpeg::new(QCIF_420);
    mock.emit -= 1; // writes one byte fewer than its own report implies
    let input_ring: StaticLendRing<1, 64> = StaticLendRing::new();
    let out_ring: StaticLendRing<1, { 11 * 9 * 384 }> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame in this test.
    let mut dec = unsafe { HwJpegDec::with_ring(&mut mock, &out_ring) };

    let input = frame_of(&input_ring, &jpeg_bytes(&[0]), 0, 0);
    assert_eq!(
        block_on(dec.process(input)).unwrap_err(),
        G2gError::Hardware(HardwareError::Peripheral)
    );
}

#[test]
fn undersized_slot_and_failures_surface() {
    // A ring slot smaller than the image: the adapter contract fails the
    // decode (it sees out.len() up front), and the element propagates it.
    let mut mock = MockJpeg::new(QCIF_420);
    let input_ring: StaticLendRing<1, 64> = StaticLendRing::new();
    let small_ring: StaticLendRing<1, 384> = StaticLendRing::new();
    // SAFETY: the rings outlive every frame in this test.
    let mut dec = unsafe { HwJpegDec::with_ring(&mut mock, &small_ring) };
    let input = frame_of(&input_ring, &jpeg_bytes(&[0]), 0, 0);
    assert_eq!(
        block_on(dec.process(input)).unwrap_err(),
        G2gError::CapsMismatch
    );

    // A hard peripheral failure propagates.
    let mut mock = MockJpeg::new(QCIF_420);
    mock.fail = true;
    let out_ring: StaticLendRing<1, { 11 * 9 * 384 }> = StaticLendRing::new();
    // SAFETY: as above.
    let mut dec = unsafe { HwJpegDec::with_ring(&mut mock, &out_ring) };
    let input = frame_of(&input_ring, &jpeg_bytes(&[0]), 0, 0);
    assert_eq!(
        block_on(dec.process(input)).unwrap_err(),
        G2gError::Hardware(HardwareError::Peripheral)
    );

    // Ring exhaustion: the only slot is still lent out.
    let mut mock = MockJpeg::new(QCIF_420);
    let out_ring: StaticLendRing<1, { 11 * 9 * 384 }> = StaticLendRing::new();
    // SAFETY: as above.
    let mut dec = unsafe { HwJpegDec::with_ring(&mut mock, &out_ring) };
    let held = block_on(dec.process(frame_of(&input_ring, &jpeg_bytes(&[0]), 0, 0)))
        .expect("decode")
        .expect("frame");
    let err = block_on(dec.process(frame_of(&input_ring, &jpeg_bytes(&[0]), 0, 1))).unwrap_err();
    assert_eq!(err, G2gError::PoolExhausted);
    drop(held);
}
