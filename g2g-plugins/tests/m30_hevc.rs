//! M30: HEVC through the codec-aware MF elements. `MfEncode(H265)` ->
//! `MfDecode(H265)` round trip against the real Media Foundation MFTs.
//!
//! Run with:
//!
//! ```powershell
//! cargo test -p g2g-plugins --features "mf-encode mf-decode" --test m30_hevc
//! ```
//!
//! HEVC support is environment-dependent: the MS HEVC decoder ships as a Store
//! extension, and a usable synchronous HEVC encoder MFT may be absent (hardware
//! encoders are commonly asynchronous, which this element does not drive). When
//! either MFT is unavailable, `configure_pipeline` returns a `Hardware` error
//! and the round-trip test skips rather than failing.

#![cfg(all(target_os = "windows", feature = "mf-encode", feature = "mf-decode"))]

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{Caps, Dim, G2gError, Rate, RawVideoFormat, VideoCodec};
use g2g_plugins::mfdecode::MfDecode;
use g2g_plugins::mfencode::MfEncode;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
const FPS_Q16: u32 = 30 << 16;
const FRAMES: usize = 10;
const FRAME_DURATION_NS: u64 = 33_333_333;

#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            self.packets.push(packet);
            Ok(PushOutcome::Accepted)
        })
    }
}

impl Collect {
    fn caps_changes(&self) -> Vec<Caps> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::CapsChanged(c) => Some(c.clone()),
                _ => None,
            })
            .collect()
    }

    fn data_frames(&self) -> Vec<&Frame> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f),
                _ => None,
            })
            .collect()
    }
}

fn nv12_frame(index: usize) -> Frame {
    let w = WIDTH as usize;
    let h = HEIGHT as usize;
    let mut data = vec![128u8; w * h * 3 / 2];
    for row in 0..h {
        for col in 0..w {
            data[row * w + col] = ((row + col + index * 8) % 256) as u8;
        }
    }
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns: index as u64 * FRAME_DURATION_NS,
            dts_ns: index as u64 * FRAME_DURATION_NS,
            duration_ns: FRAME_DURATION_NS,
            capture_ns: index as u64 * FRAME_DURATION_NS,
            ..FrameTiming::default()
        },
        sequence: index as u64,
        meta: Default::default(),
    }
}

fn starts_with_annexb_start_code(data: &[u8]) -> bool {
    data.starts_with(&[0, 0, 0, 1]) || data.starts_with(&[0, 0, 1])
}

fn nv12_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(FPS_Q16),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn hevc_encode_decode_round_trip_or_skip() {
    let mut enc = MfEncode::new()
        .with_codec(VideoCodec::H265)
        .with_bitrate(1_000_000);
    let narrowed = enc.intercept_caps(&nv12_caps()).expect("intercept NV12");
    match enc.configure_pipeline(&narrowed) {
        Ok(_) => {}
        Err(G2gError::Hardware(_)) => {
            std::eprintln!("skipping: no usable synchronous HEVC encoder MFT on this host");
            return;
        }
        Err(e) => panic!("unexpected encoder configure error: {e:?}"),
    }

    let mut encoded = Collect::default();
    for i in 0..FRAMES {
        enc.process(PipelinePacket::DataFrame(nv12_frame(i)), &mut encoded)
            .await
            .expect("encode DataFrame");
    }
    enc.process(PipelinePacket::Eos, &mut encoded)
        .await
        .expect("encode Eos");

    let hevc_caps = encoded.caps_changes().pop().expect("encoder emitted caps");
    assert!(
        matches!(
            hevc_caps,
            Caps::CompressedVideo {
                codec: VideoCodec::H265,
                ..
            }
        ),
        "encoder must emit H.265 caps, got {hevc_caps:?}"
    );
    for f in encoded.data_frames() {
        let MemoryDomain::System(slice) = &f.domain else {
            panic!("encoder must emit System-domain frames");
        };
        assert!(
            starts_with_annexb_start_code(slice.as_slice()),
            "HEVC output must be Annex-B"
        );
    }

    let mut dec = MfDecode::new().with_codec(VideoCodec::H265);
    let narrowed = dec.intercept_caps(&hevc_caps).expect("intercept H.265");
    match dec.configure_pipeline(&narrowed) {
        Ok(_) => {}
        Err(G2gError::Hardware(_)) => {
            std::eprintln!("skipping decode: HEVC decoder MFT unavailable on this host");
            return;
        }
        Err(e) => panic!("unexpected decoder configure error: {e:?}"),
    }

    let mut decoded = Collect::default();
    for f in encoded.data_frames() {
        let MemoryDomain::System(slice) = &f.domain else {
            unreachable!()
        };
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(
                slice.as_slice().to_vec().into_boxed_slice(),
            )),
            timing: f.timing,
            sequence: f.sequence,
            meta: Default::default(),
        };
        dec.process(PipelinePacket::DataFrame(frame), &mut decoded)
            .await
            .expect("decode DataFrame");
    }
    dec.process(PipelinePacket::Eos, &mut decoded)
        .await
        .expect("decode Eos");

    let caps_changes = decoded.caps_changes();
    assert!(
        matches!(
            caps_changes.first(),
            Some(Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(WIDTH),
                height: Dim::Fixed(HEIGHT),
                ..
            })
        ),
        "decoded caps must match the encoded geometry, got {caps_changes:?}"
    );

    let frames = decoded.data_frames();
    assert_eq!(
        frames.len(),
        FRAMES,
        "every encoded picture must decode back"
    );
    let expected_len = (WIDTH * HEIGHT * 3 / 2) as usize;
    for f in frames {
        let MemoryDomain::System(slice) = &f.domain else {
            panic!("decoder must emit System-domain frames");
        };
        assert_eq!(slice.as_slice().len(), expected_len, "packed NV12 length");
    }
}
