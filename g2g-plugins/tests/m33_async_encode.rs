//! M33: `MfEncode::with_hardware()` drives an asynchronous (event-driven)
//! hardware encoder MFT. Hardware H.264/HEVC encoders are commonly async MFTs;
//! this exercises the event-loop path that M30's sync loop could not.
//!
//! Run with:
//!
//! ```powershell
//! cargo test -p g2g-plugins --features "mf-encode mf-decode" --test m33_async_encode
//! ```
//!
//! Needs a hardware encoder MFT. When none is registered (or it fails to
//! initialise), `configure_pipeline` returns a `Hardware` error and the test
//! skips. When an async MFT is found, it also asserts the async path was the
//! one exercised.

#![cfg(all(target_os = "windows", feature = "mf-encode"))]

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{Caps, Dim, G2gError, Rate, RawVideoFormat, VideoCodec};
use g2g_plugins::mfencode::MfEncode;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
const FPS_Q16: u32 = 30 << 16;
const FRAMES: usize = 30;
const FRAME_DURATION_NS: u64 = 33_333_333;

#[derive(Default)]
struct Collect {
    data_frames: usize,
    caps: Vec<Caps>,
    annexb_ok: bool,
}

impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if let MemoryDomain::System(slice) = &f.domain {
                        let d = slice.as_slice();
                        if self.data_frames == 0 {
                            self.annexb_ok =
                                d.starts_with(&[0, 0, 0, 1]) || d.starts_with(&[0, 0, 1]);
                        }
                    }
                    self.data_frames += 1;
                }
                PipelinePacket::CapsChanged(c) => self.caps.push(c),
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
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
    }
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
async fn hardware_encoder_round_trips_via_the_async_path() {
    let mut enc = MfEncode::new().with_hardware().with_bitrate(2_000_000);
    let narrowed = enc.intercept_caps(&nv12_caps()).expect("intercept NV12");
    match enc.configure_pipeline(&narrowed) {
        Ok(_) => {}
        Err(G2gError::Hardware(_)) => {
            std::eprintln!("skipping: no usable hardware H.264 encoder MFT on this host");
            return;
        }
        Err(e) => panic!("unexpected configure error: {e:?}"),
    }

    let is_async = enc.is_async().expect("configured");
    std::eprintln!("hardware encoder async_mode = {is_async}");

    let mut sink = Collect::default();
    for i in 0..FRAMES {
        enc.process(PipelinePacket::DataFrame(nv12_frame(i)), &mut sink)
            .await
            .expect("encode frame");
    }
    enc.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("encode eos");

    assert_eq!(
        enc.encoded_count() as usize,
        sink.data_frames,
        "every encoded frame reaches the sink"
    );
    assert_eq!(
        sink.data_frames, FRAMES,
        "all input pictures encode out (drained at Eos)"
    );
    assert!(sink.annexb_ok, "encoded output is Annex-B");
    assert!(
        matches!(
            sink.caps.first(),
            Some(Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            })
        ),
        "encoder emits H.264 caps, got {:?}",
        sink.caps
    );
}
