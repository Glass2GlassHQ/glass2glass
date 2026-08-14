//! M13: hardware-gated smoke test for `VaapiH264Dec`, and its M1036 H.265 twin.
//!
//! Ignored by default — requires:
//! - Linux with a libva-capable render node (default `/dev/dri/renderD128`).
//! - An Annex-B fixture file path in `G2G_H264_FIXTURE` / `G2G_H265_FIXTURE`.
//!
//! Run with:
//!
//! ```sh
//! G2G_H264_FIXTURE=/path/to/clip.h264 cargo test -p g2g-plugins \
//!     --features vaapi --test vaapi_smoke -- --ignored --nocapture
//! ```

#![cfg(all(target_os = "linux", feature = "vaapi"))]

use g2g_core::element::{AsyncElement, OutputSink, PushOutcome, Reconfigure};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{Caps, ConfigureOutcome, Dim, G2gError, Rate, RawVideoFormat};
use g2g_plugins::vaapidec::{H264Codec, H265Codec, VaapiCodec, VaapiDec};

/// `OutputSink` that records every packet it receives. The decoder feeds it
/// `CapsChanged` (once per geometry change) followed by `DataFrame`s.
#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            self.packets.push(packet);
            Ok(PushOutcome::Accepted)
        })
    }
}

#[tokio::test]
#[ignore = "requires libva-capable hardware and a G2G_H264_FIXTURE path"]
async fn vaapi_h264_decodes_fixture() {
    decode_fixture::<H264Codec>("G2G_H264_FIXTURE").await;
}

#[tokio::test]
#[ignore = "requires libva-capable hardware and a G2G_H265_FIXTURE path"]
async fn vaapi_h265_decodes_fixture() {
    decode_fixture::<H265Codec>("G2G_H265_FIXTURE").await;
}

/// Feed one whole Annex-B file through the decoder and check what came out.
/// Returns early (skips) when the fixture is unset or the host has no usable
/// VAAPI decoder.
async fn decode_fixture<C: VaapiCodec>(fixture_var: &str) {
    let Some(path) = std::env::var_os(fixture_var) else {
        eprintln!("skipping: set {fixture_var}=/path/to/clip to run");
        return;
    };
    let bitstream = std::fs::read(&path).expect("read fixture");
    assert!(!bitstream.is_empty(), "fixture is empty");

    let mut dec = VaapiDec::<C>::new();

    // Phase 1/2 negotiation surrogates: we know the upstream codec, with
    // unknown geometry until the SPS lands.
    let upstream = Caps::CompressedVideo {
        codec: C::CODEC,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    };
    let narrowed = dec.intercept_caps(&upstream).expect("intercept the codec");
    assert_eq!(narrowed, upstream);
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

    // M1036: a fixture whose resolution changes mid-stream must also leave an
    // upstream proposal behind. A single-resolution fixture (the usual case)
    // must leave none.
    let reconfigure = dec.take_reconfigure();
    if caps_changes.len() > 1 {
        assert!(
            matches!(reconfigure, Some(Reconfigure::Propose(_))),
            "a mid-stream resolution change must propose new input caps upstream, got {reconfigure:?}"
        );
    } else {
        assert_eq!(
            reconfigure, None,
            "a fixed-resolution stream must not ask upstream to renegotiate"
        );
    }
}
