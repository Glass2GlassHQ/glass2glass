//! M661 integration: the whole new video egress chain as one heap-free static
//! pipeline, `camera (YUYV) -> YuyvToI420 -> HwH264Enc -> RtpSink`, driven by
//! the real `run_source_transform_sink` runner. A tag byte stamped at the
//! camera is followed all the way to the RTP payload, proving the convert
//! (M661), the encoder seam (M660), and the RTP egress (M643) compose. The
//! encoder is a scripted mock (real HW H.264 on silicon is the deferred
//! `Hardware` row); everything else is the production element.

mod util;

use g2g_core::error::G2gError;
use g2g_core::mediaclock::MediaClock;
use g2g_core::rtp::RTP_HEADER_LEN;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{run_source_transform_sink, Chain};
use g2g_mcu::rtp::PacketSender;
use g2g_mcu::{FrameGrabber, GrabberSrc, H264EncodeInfo, H264Encoder, HwH264Enc, RtpSink, YuyvToI420};
use util::block_on;

const W: u16 = 16;
const H: u16 = 16;
const YUYV_BYTES: usize = 16 * 16 * 2; // 512
const I420_BYTES: usize = 16 * 16 * 3 / 2; // 384
const AU_MAX: usize = 64;
const FRAMES: u32 = 4;

/// The scripted encoder's access unit for frame `n`: start code + NAL header
/// (IDR every 4th) + the input's first luma byte + the frame index, so the AU
/// is traceable to the pixel the camera stamped.
fn au(first_luma: u8, n: u8) -> Vec<u8> {
    let kf = n % 4 == 0;
    vec![0x00, 0x00, 0x00, 0x01, if kf { 0x65 } else { 0x41 }, first_luma, n]
}

struct MockEnc {
    n: u8,
}

impl H264Encoder for &mut MockEnc {
    async fn encode(&mut self, raw: &[u8], out: &mut [u8]) -> Result<H264EncodeInfo, G2gError> {
        let a = au(raw.first().copied().unwrap_or(0), self.n);
        if out.len() < a.len() {
            return Err(G2gError::CapsMismatch);
        }
        out[..a.len()].copy_from_slice(&a);
        let kf = self.n % 4 == 0;
        self.n = self.n.wrapping_add(1);
        Ok(H264EncodeInfo { len: a.len(), keyframe: kf })
    }
}

/// A mock YUYV camera stamping the frame counter into pixel 0's luma (`Y0` is
/// the first YUYV byte), the rest black.
struct YuyvCamera {
    n: u8,
}

impl FrameGrabber for YuyvCamera {
    async fn capture(&mut self, buf: &mut [u8]) -> Result<usize, G2gError> {
        for b in buf.iter_mut() {
            *b = 0;
        }
        if let Some(y0) = buf.first_mut() {
            *y0 = 0xA0 + self.n;
        }
        self.n = self.n.wrapping_add(1);
        Ok(buf.len())
    }
}

/// Captures each RTP packet's header and payload.
#[derive(Default)]
struct CollectSender {
    packets: Vec<([u8; RTP_HEADER_LEN], Vec<u8>)>,
}

impl PacketSender for &mut CollectSender {
    async fn send(&mut self, header: &[u8; RTP_HEADER_LEN], payload: &[u8]) -> Result<(), G2gError> {
        self.packets.push((*header, payload.to_vec()));
        Ok(())
    }
}

#[test]
fn camera_yuyv_convert_h264_rtp_end_to_end() {
    // Source: a YUYV camera into a 512-byte ring slot.
    let cam_ring: StaticLendRing<2, YUYV_BYTES> = StaticLendRing::new();
    // SAFETY: every ring outlives the pipeline (drained before this scope ends).
    let source =
        unsafe { GrabberSrc::with_ring(YuyvCamera { n: 0 }, &cam_ring, 33_333_333) }
            .with_frame_limit(FRAMES);

    // Transforms: YUYV->I420 then I420->H.264 AU, fused with `Chain`.
    let cvt_ring: StaticLendRing<2, I420_BYTES> = StaticLendRing::new();
    // SAFETY: as above.
    let convert = unsafe { YuyvToI420::with_ring(W, H, &cvt_ring) }.expect("geometry");
    let mut enc = MockEnc { n: 0 };
    let enc_ring: StaticLendRing<2, AU_MAX> = StaticLendRing::new();
    // SAFETY: as above.
    let encode = unsafe { HwH264Enc::with_ring(&mut enc, W, H, &enc_ring) }.expect("geometry");

    // Sink: RTP egress (PT 96 dynamic H.264, 90 kHz) into a collecting sender.
    let mut sender = CollectSender::default();
    let rtp = RtpSink::new(&mut sender, MediaClock::video(), 96, 0xDEAD_BEEF, 0);

    block_on(run_source_transform_sink(source, Chain(convert, encode), rtp))
        .expect("the camera -> convert -> encode -> RTP pipeline runs");

    assert_eq!(sender.packets.len(), FRAMES as usize, "one RTP packet per captured frame");
    for (i, (header, payload)) in sender.packets.iter().enumerate() {
        let n = i as u8;
        // The payload is exactly the encoder's access unit for the pixel the
        // camera stamped (0xA0 + n), carried through convert + encode + RTP.
        assert_eq!(payload.as_slice(), au(0xA0 + n, n), "packet {i}: AU payload end to end");
        // RTP header: V=2 / no pad/ext/CC, dynamic PT 96, sequential, our SSRC.
        assert_eq!(header[0], 0x80, "packet {i}: version 2, no padding/extension/CSRC");
        assert_eq!(header[1] & 0x7F, 96, "packet {i}: payload type 96");
        assert_eq!(u16::from_be_bytes([header[2], header[3]]), n as u16, "packet {i}: sequence");
        assert_eq!(
            u32::from_be_bytes([header[8], header[9], header[10], header[11]]),
            0xDEAD_BEEF,
            "packet {i}: SSRC"
        );
    }
    // The first frame's access unit is an IDR keyframe (random-access point).
    assert!(sender.packets[0].1.contains(&0x65), "frame 0 is an IDR");
}
