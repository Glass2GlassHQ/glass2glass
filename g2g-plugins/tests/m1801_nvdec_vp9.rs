#![cfg(all(target_os = "linux", feature = "nvdec"))]

#[path = "common/collect_sink.rs"]
mod collect_sink;
use collect_sink::CollectSink;

#[path = "common/ivf.rs"]
mod ivf;
use ivf::ivf_frames;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{AsyncElement, Caps, Dim, G2gError, RawVideoFormat};
use g2g_plugins::ivfdemux::IvfDemux;
use g2g_plugins::nvdec::NvDec;

const FIXTURE: &[u8] = include_bytes!("fixtures/vp9_320x240.ivf");
// libvpx's show_existing_frame vector: one-byte packets repeat frames shown before
const REPEATS_SHOWN_FRAMES: &[u8] = include_bytes!("fixtures/vp90-2-10-show-existing-frame2.ivf");
// a 320x180 keyframe, then 640x360 and 1280x720 frames that predict from it
const SPATIAL_LAYERS: &[u8] = include_bytes!("fixtures/vp90-2-22-svc_1280x720_3_1packet.ivf");
const FRAME_PERIOD_NS: u64 = 33_333_333;

fn coded_frame(bytes: &[u8], index: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into())),
        timing: FrameTiming {
            pts_ns: index * FRAME_PERIOD_NS,
            ..FrameTiming::default()
        },
        sequence: index,
        meta: Default::default(),
    }
}

#[tokio::test]
async fn nvdec_decodes_vp9() {
    let caps = IvfDemux::stream_caps(FIXTURE).expect("an IVF header");
    let Caps::CompressedVideo { width, height, .. } = caps.clone() else {
        panic!("the fixture is compressed video: {caps:?}");
    };
    let mut decoder = NvDec::new();
    if decoder.configure_pipeline(&caps).is_err() {
        eprintln!("skipping: NVDEC unavailable on this host");
        return;
    }
    let frames = ivf_frames(FIXTURE);
    let mut sink = CollectSink::default();
    for (index, bytes) in frames.iter().enumerate() {
        decoder
            .process(
                PipelinePacket::DataFrame(coded_frame(bytes, index as u64)),
                &mut sink,
            )
            .await
            .expect("NVDEC decodes the VP9 frame");
    }
    decoder
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("flush");

    let decoded = sink
        .packets
        .iter()
        .filter(|packet| matches!(packet, PipelinePacket::DataFrame(_)))
        .count();
    assert_eq!(decoded, frames.len());
    let announced: Vec<&Caps> = sink
        .packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::CapsChanged(caps) => Some(caps),
            _ => None,
        })
        .collect();
    assert!(
        matches!(
            announced.as_slice(),
            [Caps::RawVideo { format: RawVideoFormat::Nv12, width: w, height: h, .. }]
                if *w == width && *h == height && matches!(w, Dim::Fixed(_))
        ),
        "one NV12 caps at the stream's size: {announced:?}"
    );
}

#[tokio::test]
async fn nvdec_refuses_a_vp9_frame_that_outgrows_its_references() {
    let caps = IvfDemux::stream_caps(SPATIAL_LAYERS).expect("an IVF header");
    let mut decoder = NvDec::new();
    if decoder.configure_pipeline(&caps).is_err() {
        eprintln!("skipping: NVDEC unavailable on this host");
        return;
    }
    let frames = ivf_frames(SPATIAL_LAYERS);
    let mut sink = CollectSink::default();
    let result = decoder
        .process(
            PipelinePacket::DataFrame(coded_frame(frames[0], 0)),
            &mut sink,
        )
        .await;
    assert_eq!(result, Err(G2gError::CapsMismatch));
}

#[tokio::test]
async fn nvdec_repeats_a_vp9_frame_still_held_downstream() {
    let caps = IvfDemux::stream_caps(REPEATS_SHOWN_FRAMES).expect("an IVF header");
    let mut decoder = NvDec::new();
    if decoder.configure_pipeline(&caps).is_err() {
        eprintln!("skipping: NVDEC unavailable on this host");
        return;
    }
    let frames = ivf_frames(REPEATS_SHOWN_FRAMES);
    // the sink keeps every decoded frame, so each repeat finds its first output still mapped
    let mut sink = CollectSink::default();
    for (index, bytes) in frames.iter().enumerate() {
        decoder
            .process(
                PipelinePacket::DataFrame(coded_frame(bytes, index as u64)),
                &mut sink,
            )
            .await
            .expect("NVDEC decodes or repeats the VP9 frame");
    }
    decoder
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("flush");
    let decoded = sink
        .packets
        .iter()
        .filter(|packet| matches!(packet, PipelinePacket::DataFrame(_)))
        .count();
    assert_eq!(decoded, frames.len());
}
