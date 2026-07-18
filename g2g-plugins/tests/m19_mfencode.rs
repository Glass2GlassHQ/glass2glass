//! M19: `MfEncode` end-to-end against the real Media Foundation H.264
//! encoder MFT (the MS software encoder, so no GPU is needed), plus the
//! encode -> decode round trip through `MfDecode` when both features are on.
//!
//! Run with:
//!
//! ```powershell
//! cargo test -p g2g-plugins --features "mf-encode mf-decode" --test m19_mfencode
//! ```

#![cfg(all(target_os = "windows", feature = "mf-encode"))]

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{Caps, ConfigureOutcome, Dim, G2gError, Rate, RawVideoFormat, VideoCodec};
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

/// A synthetic packed NV12 picture: a per-frame luma gradient over a neutral
/// chroma plane, so consecutive frames differ and the encoder has real work.
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

fn input_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(FPS_Q16),
    }
}

/// Drive `MfEncode` over `FRAMES` synthetic pictures and return the collected
/// downstream packets.
async fn encode_frames() -> Collect {
    let mut enc = MfEncode::new().with_bitrate(1_000_000);
    let narrowed = enc.intercept_caps(&input_caps()).expect("intercept NV12");
    let outcome = enc
        .configure_pipeline(&narrowed)
        .expect("encoder MFT must initialise");
    assert!(matches!(outcome, ConfigureOutcome::Accepted));

    let mut sink = Collect::default();
    for i in 0..FRAMES {
        enc.process(PipelinePacket::DataFrame(nv12_frame(i)), &mut sink)
            .await
            .expect("process DataFrame");
    }
    enc.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("process Eos");
    assert_eq!(enc.encoded_count(), FRAMES as u64);
    sink
}

#[tokio::test(flavor = "current_thread")]
async fn encode_emits_h264_caps_and_annexb_access_units() {
    let sink = encode_frames().await;

    let caps_changes = sink.caps_changes();
    assert_eq!(
        caps_changes,
        vec![Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(WIDTH),
            height: Dim::Fixed(HEIGHT),
            framerate: Rate::Fixed(FPS_Q16),
        }],
        "one CapsChanged with the encoded geometry"
    );

    let frames = sink.data_frames();
    assert_eq!(
        frames.len(),
        FRAMES,
        "low-latency mode: one output per input"
    );
    for f in &frames {
        let MemoryDomain::System(slice) = &f.domain else {
            panic!("encoder must emit System-domain frames");
        };
        let data = slice.as_slice();
        assert!(!data.is_empty(), "encoded access unit must not be empty");
        assert!(
            starts_with_annexb_start_code(data),
            "encoded output must be Annex-B (got leading bytes {:?})",
            &data[..data.len().min(8)]
        );
    }
    // The first access unit carries an IDR, so it must be the largest-ish;
    // at minimum it must hold more than a bare start code (SPS/PPS + IDR).
    let MemoryDomain::System(first) = &frames[0].domain else {
        unreachable!()
    };
    assert!(
        first.as_slice().len() > 16,
        "IDR access unit implausibly small"
    );
}

#[cfg(feature = "mf-decode")]
#[tokio::test(flavor = "current_thread")]
async fn encode_decode_round_trip_recovers_all_frames() {
    use g2g_plugins::mfdecode::MfDecode;

    let encoded = encode_frames().await;
    let h264_caps = encoded.caps_changes().pop().expect("encoder emitted caps");

    let mut dec = MfDecode::new();
    let narrowed = dec.intercept_caps(&h264_caps).expect("intercept H.264");
    let outcome = dec
        .configure_pipeline(&narrowed)
        .expect("decoder MFT must initialise");
    assert!(matches!(outcome, ConfigureOutcome::Accepted));

    let mut sink = Collect::default();
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
        dec.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .expect("decode DataFrame");
    }
    dec.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("decode Eos");

    let caps_changes = sink.caps_changes();
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

    let frames = sink.data_frames();
    assert_eq!(
        frames.len(),
        FRAMES,
        "every encoded picture must decode back out"
    );
    let expected_len = (WIDTH * HEIGHT * 3 / 2) as usize;
    for f in frames {
        let MemoryDomain::System(slice) = &f.domain else {
            panic!("decoder must emit System-domain frames");
        };
        assert_eq!(slice.as_slice().len(), expected_len, "packed NV12 length");
    }
}
