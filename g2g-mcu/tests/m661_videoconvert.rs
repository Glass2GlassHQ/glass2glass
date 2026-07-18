//! M661: the heap-free packed-YUYV 4:2:2 -> planar I420 4:2:0 convert, the MCU
//! twin of the host `VideoConvert`. Asserts the Y/U/V plane layout on a hand
//! computed vector, rejects a mis-sized frame, and runs a
//! `camera (YUYV) -> convert` pipeline over the real `GrabberSrc` runner. The
//! output is exactly `i420_len(w, h)` bytes, the size `HwH264Enc` (M660) accepts,
//! so this closes the camera -> H.264-encode gap.

mod util;

use g2g_core::error::G2gError;
use g2g_core::run_source_transform_sink;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{StaticSink, StaticTransform};
use g2g_mcu::hwh264::i420_len;
use g2g_mcu::videoconvert::yuyv_len;
use g2g_mcu::{FrameGrabber, GrabberSrc, YuyvToI420};
use util::{block_on, frame_of, payload};

#[test]
fn yuyv_sizing_is_checked() {
    assert_eq!(yuyv_len(4, 2), Some(16), "4x2 packed YUYV = 2 bytes/pixel");
    assert_eq!(yuyv_len(640, 480), Some(640 * 480 * 2));
    assert_eq!(yuyv_len(0, 2), None, "zero dimension");
    assert_eq!(yuyv_len(3, 2), None, "odd width has no whole YUYV pair");
}

/// A 4x2 packed-YUYV frame with distinct, hand-chosen samples so the plane
/// split and the vertical chroma average are both checkable.
fn yuyv_4x2() -> [u8; 16] {
    [
        // row 0: [Y0,U0,Y1,V0, Y2,U1,Y3,V1]
        10, 100, 20, 200, 30, 104, 40, 204, //
        // row 1: [Y4,U2,Y5,V2, Y6,U3,Y7,V3]
        50, 108, 60, 208, 70, 112, 80, 212,
    ]
}

#[test]
fn converts_yuyv_to_planar_i420() {
    let src: StaticLendRing<1, 16> = StaticLendRing::new();
    let ring: StaticLendRing<1, 12> = StaticLendRing::new(); // i420_len(4,2) = 12
                                                             // SAFETY: the rings outlive the frame within this test.
    let mut cvt = unsafe { YuyvToI420::with_ring(4, 2, &ring) }.expect("valid geometry");
    let out = block_on(cvt.process(frame_of(&src, &yuyv_4x2(), 0, 7)))
        .expect("convert ok")
        .expect("one frame out");

    let expected = [
        10, 20, 30, 40, 50, 60, 70, 80, // Y plane: every luma sample, in order
        104, 108, // U: avg(U0,U2)=avg(100,108), avg(U1,U3)=avg(104,112)
        204, 208, // V: avg(V0,V2)=avg(200,208), avg(V1,V3)=avg(204,212)
    ];
    assert_eq!(
        payload(&out),
        expected,
        "planar I420 with vertically averaged chroma"
    );
    assert_eq!(out.sequence, 7, "sequence carried through");
    assert_eq!(
        payload(&out).len(),
        i420_len(4, 2).unwrap(),
        "output is exactly one I420 frame"
    );
}

#[test]
fn wrong_sized_frame_is_rejected() {
    let src: StaticLendRing<1, 16> = StaticLendRing::new();
    let ring: StaticLendRing<1, 12> = StaticLendRing::new();
    // SAFETY: the rings outlive the frame within this test.
    let mut cvt = unsafe { YuyvToI420::with_ring(4, 2, &ring) }.expect("geometry");
    // 14 bytes is not a whole 4x2 YUYV frame (16 bytes).
    let short = block_on(cvt.process(frame_of(&src, &[0u8; 14], 0, 0)));
    assert!(
        matches!(short, Err(G2gError::CapsMismatch)),
        "a partial frame is rejected"
    );
}

/// A mock YUYV camera filling each frame with a fixed test pattern.
struct YuyvCamera;

impl FrameGrabber for YuyvCamera {
    async fn capture(&mut self, buf: &mut [u8]) -> Result<usize, G2gError> {
        let pat = yuyv_4x2();
        for (b, p) in buf.iter_mut().zip(pat.iter().cycle()) {
            *b = *p;
        }
        Ok(buf.len())
    }
}

#[derive(Default)]
struct CollectI420 {
    frames: Vec<Vec<u8>>,
}

impl StaticSink for &mut CollectI420 {
    async fn consume(&mut self, frame: g2g_core::frame::Frame) -> Result<(), G2gError> {
        self.frames.push(payload(&frame).to_vec());
        Ok(())
    }
}

#[test]
fn camera_yuyv_to_i420_pipeline_runs() {
    let src_ring: StaticLendRing<2, 16> = StaticLendRing::new();
    // SAFETY: the rings outlive the pipeline (drained before this scope ends).
    let source =
        unsafe { GrabberSrc::with_ring(YuyvCamera, &src_ring, 33_333_333) }.with_frame_limit(3);
    let out_ring: StaticLendRing<2, 12> = StaticLendRing::new();
    // SAFETY: as above.
    let cvt = unsafe { YuyvToI420::with_ring(4, 2, &out_ring) }.expect("geometry");
    let mut sink = CollectI420::default();

    block_on(run_source_transform_sink(source, cvt, &mut sink)).expect("pipeline runs");

    assert_eq!(
        sink.frames.len(),
        3,
        "one I420 frame per captured YUYV frame"
    );
    let expected = [10, 20, 30, 40, 50, 60, 70, 80, 104, 108, 204, 208];
    for f in &sink.frames {
        assert_eq!(
            f.as_slice(),
            expected,
            "each converted frame is the I420 of the pattern"
        );
    }
}
