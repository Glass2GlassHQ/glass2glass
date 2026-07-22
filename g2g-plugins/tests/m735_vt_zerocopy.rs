//! M735: the `CVPixelBuffer` zero-copy memory domain on the macOS CI runner.
//! `VtDecode` in `cv-output` mode emits retained IOSurface-backed pixel
//! buffers (`MemoryDomain::CvPixelBuffer`), and `VtEncode` consumes them
//! directly, so a decode -> encode transcode never stages pixels in system
//! memory. Content is verified by decoding the transcode back with the packed
//! (System) path and re-running the m731 luma checks.
#![cfg(all(target_os = "macos", feature = "vtdecode", feature = "vtencode"))]

use core::future::Future;
use core::pin::Pin;

use g2g_core::conformance::{ConformanceDimension, Evidence};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, G2gError, OutputSink, PipelineClock, PushOutcome,
    Rate, RawVideoFormat, VideoCodec,
};
use g2g_plugins::registry::default_registry;
use g2g_plugins::vtdecode::VtDecode;
use g2g_plugins::vtencode::VtEncode;

const H264_CLIP: &[u8] = include_bytes!("fixtures/h264_640x480.h264");
const FRAME_DURATION_NS: u64 = 33_333_333;
const FOURCC_420V: u32 = 0x3432_3076; // '420v'

#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            self.packets.push(packet);
            Ok(PushOutcome::Accepted)
        })
    }
}

impl Collect {
    fn into_data_frames(self) -> Vec<Frame> {
        self.packets
            .into_iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f),
                _ => None,
            })
            .collect()
    }
}

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Split the fixture into access units with the real re-framing parser.
async fn access_units(es: &[u8]) -> Vec<Frame> {
    let caps = Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    };
    let mut parse = g2g_plugins::h264parse::H264Parse::reframing();
    parse.configure_pipeline(&caps).expect("configure parser");
    let mut sink = Collect::default();
    for (i, chunk) in es.chunks(4096).enumerate() {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(chunk.to_vec().into_boxed_slice())),
            FrameTiming::default(),
            i as u64,
        );
        parse
            .process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .expect("parse chunk");
    }
    parse
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("Eos");
    sink.into_data_frames()
}

/// Drive `el` over `frames` + Eos, collecting its output frames.
async fn run_element<E: AsyncElement>(el: &mut E, frames: Vec<Frame>) -> Vec<Frame> {
    let mut sink = Collect::default();
    for f in frames {
        el.process(PipelinePacket::DataFrame(f), &mut sink)
            .await
            .expect("process DataFrame");
    }
    el.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("drain at Eos");
    sink.into_data_frames()
}

#[tokio::test(flavor = "current_thread")]
async fn cv_output_transcode_never_touches_system_memory() {
    let mut aus = access_units(H264_CLIP).await;
    let fed = aus.len();
    for (i, au) in aus.iter_mut().enumerate() {
        au.timing.pts_ns = i as u64 * FRAME_DURATION_NS;
    }

    // Decode with zero-copy output.
    let mut dec = VtDecode::h264().with_cv_output();
    let caps = Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
    };
    let narrowed = dec.intercept_caps(&caps).expect("intercept H.264");
    assert!(matches!(
        dec.configure_pipeline(&narrowed).expect("decoder session"),
        ConfigureOutcome::Accepted
    ));
    let decoded = run_element(&mut dec, aus).await;
    assert_eq!(decoded.len(), fed, "every access unit decoded");

    // Every frame is a retained IOSurface-backed '420v' CVPixelBuffer.
    for f in &decoded {
        let MemoryDomain::CvPixelBuffer(buf) = &f.domain else {
            panic!(
                "cv-output must emit CvPixelBuffer frames, got {:?}",
                f.domain
            );
        };
        assert_eq!((buf.width, buf.height), (640, 480));
        assert_eq!(buf.pixel_format, FOURCC_420V, "pinned by the dest attrs");
        assert!(buf.io_surface_backed, "requested IOSurface backing");
        assert_ne!(buf.pixel_buffer, 0, "live CVPixelBufferRef");
    }

    // Encode the pixel buffers directly (no System staging).
    let mut enc = VtEncode::h264().with_bitrate(1_000_000);
    let nv12 = Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
    };
    let narrowed = enc.intercept_caps(&nv12).expect("intercept NV12");
    enc.configure_pipeline(&narrowed).expect("encoder session");
    let encoded = run_element(&mut enc, decoded).await;
    assert_eq!(encoded.len(), fed, "every picture re-encoded");

    // Decode the transcode back on the packed path and re-run the m731 luma
    // checks: the test card's near-black and near-white must survive the
    // zero-copy round trip.
    let mut check = VtDecode::h264();
    let narrowed = check.intercept_caps(&caps).expect("intercept H.264");
    check
        .configure_pipeline(&narrowed)
        .expect("decoder session");
    let frames = run_element(&mut check, encoded).await;
    assert_eq!(frames.len(), fed, "every transcoded picture decodes");
    let Some(slice) = frames[0].domain.as_system_slice() else {
        panic!("packed path emits System frames");
    };
    let luma = &slice[..640 * 480];
    let min = *luma.iter().min().unwrap();
    let max = *luma.iter().max().unwrap();
    assert!(min <= 30, "no near-black content (min {min})");
    assert!(max >= 190, "no bright content (max {max})");
    eprintln!("zero-copy transcode: {fed} frames, checked luma {min}..={max}");

    g2g_plugins::conformance::persist::record_evidence(
        "vtdecode",
        &Evidence::new(ConformanceDimension::Hardware)
            .platform(format!("macOS {} VideoToolbox", std::env::consts::ARCH))
            .codec("h264")
            .detail("zero-copy CVPixelBuffer decode-to-encode transcode"),
    )
    .expect("record hardware evidence");
}

/// The same zero-copy transcode through a text pipeline (`cv-output` as a
/// launch property, the encoder consuming the CvPixelBuffer domain).
#[tokio::test(flavor = "current_thread")]
async fn cv_output_transcodes_in_a_text_pipeline() {
    let path = std::env::temp_dir().join(format!("g2g-m735-{}.h264", std::process::id()));
    std::fs::write(&path, H264_CLIP).expect("write temp fixture");
    let line = format!(
        "filesrc location={} ! h264parse ! vtdec cv-output=true ! vtenc_h264 ! fakesink",
        path.display()
    );
    let reg = default_registry();
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("`{line}` runs: {e:?}"));
    std::fs::remove_file(&path).ok();
    assert_eq!(stats.frames_consumed, 10, "every fixture frame transcoded");
}
