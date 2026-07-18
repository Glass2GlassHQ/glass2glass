//! M13: hardware-gated smoke test for `VaapiH264Dec`.
//!
//! Ignored by default — requires:
//! - Linux with a libva-capable render node (default `/dev/dri/renderD128`).
//! - An H.264 Annex-B fixture file path in `G2G_H264_FIXTURE`.
//!
//! Run with:
//!
//! ```sh
//! G2G_H264_FIXTURE=/path/to/clip.h264 cargo test -p g2g-plugins \
//!     --features vaapi --test vaapi_smoke -- --ignored --nocapture
//! ```

#![cfg(all(target_os = "linux", feature = "vaapi"))]

use std::sync::Arc;

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{Caps, ConfigureOutcome, Dim, G2gError, Rate, RawVideoFormat, VideoCodec};
use g2g_plugins::vaapidec::VaapiH264Dec;

/// `OutputSink` that records every packet it receives. The decoder feeds it
/// `CapsChanged` (once per geometry change) followed by `DataFrame`s.
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

#[tokio::test]
#[ignore = "requires libva-capable hardware and a G2G_H264_FIXTURE path"]
async fn vaapi_h264_decodes_fixture() {
    let Some(path) = std::env::var_os("G2G_H264_FIXTURE") else {
        eprintln!("skipping: set G2G_H264_FIXTURE=/path/to/clip.h264 to run");
        return;
    };
    let bitstream = std::fs::read(&path).expect("read H.264 fixture");
    assert!(!bitstream.is_empty(), "fixture is empty");

    let mut dec = VaapiH264Dec::new();

    // Phase 1/2 negotiation surrogates: we know the upstream is H.264 with
    // unknown geometry until SPS lands.
    let upstream = Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    };
    let narrowed = dec.intercept_caps(&upstream).expect("intercept H.264");
    assert!(matches!(
        narrowed,
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            ..
        }
    ));
    let outcome = match dec.configure_pipeline(&narrowed) {
        Ok(o) => o,
        Err(e) => {
            eprintln!(
                "skipping: cros-codecs decoder failed to initialise on this host: {:?} \
                 (vainfo working is necessary but not sufficient — cros-codecs 0.0.6 \
                 also requires GBM `NV12` allocation, which AMD radeonsi does not expose)",
                e
            );
            return;
        }
    };
    assert!(matches!(outcome, ConfigureOutcome::Accepted));

    let mut sink = Collect::default();

    let frame = Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bitstream.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    };

    dec.process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
        .expect("process DataFrame");
    dec.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("process Eos drains DPB");

    let caps_changes: Vec<_> = sink
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::CapsChanged(c) => Some(c.clone()),
            _ => None,
        })
        .collect();
    let data_frames: Vec<_> = sink
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .collect();

    eprintln!(
        "decoded {} frame(s); {} CapsChanged emitted",
        data_frames.len(),
        caps_changes.len()
    );
    assert!(
        !caps_changes.is_empty(),
        "expected at least one NV12 CapsChanged"
    );
    assert!(
        !data_frames.is_empty(),
        "expected at least one decoded frame"
    );

    let first = caps_changes.first().unwrap();
    match first {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            ..
        } => {
            eprintln!("first NV12 caps: {}x{}", w, h);
            assert!(*w > 0 && *h > 0);
            // Sanity-check the first decoded frame's pixel buffer matches the
            // advertised geometry (Y + interleaved UV = w*h*3/2).
            let f = data_frames.first().unwrap();
            let expected = (*w as usize) * (*h as usize) * 3 / 2;
            match &f.domain {
                MemoryDomain::System(slice) => {
                    assert_eq!(
                        slice.as_slice().len(),
                        expected,
                        "NV12 byte length mismatch"
                    );
                }
                _ => panic!("decoder must emit System-domain NV12 frames"),
            }
        }
        other => panic!("expected NV12 fixed caps, got {:?}", other),
    }

    // Suppress unused warning when the test ever moves to multiple inputs.
    let _ = Arc::<()>::new(());
}
