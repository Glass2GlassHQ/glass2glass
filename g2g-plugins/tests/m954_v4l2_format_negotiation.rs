//! M954 format-flexible `v4l2src`: the device's pixel formats become caps
//! alternatives, and negotiation picks which one the camera runs in. The
//! headline case is MJPEG-mode UVC decoded downstream by `mjpegdec`.
//!
//! The fourcc mapping is pure and runs anywhere. The capture tests need a real
//! `/dev/videoN` UVC device the running user can open, and one that offers
//! MJPEG, so they are ignored by default like the other v4l2 smoke tests.
//! Override the device with `G2G_V4L2_DEVICE`. Run with `--test-threads=1`:
//! the tests share the camera, and a parallel open fails with EBUSY.
//!
//! ```sh
//! cargo test -p g2g-plugins --features v4l2 \
//!     --test m954_v4l2_format_negotiation -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(all(target_os = "linux", feature = "v4l2"))]

use core::future::Future;
use core::pin::Pin;

use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, MemoryDomain, OutputSink,
    PipelineClock, PipelinePacket, Rate, RawVideoFormat, VideoCodec,
};
use g2g_plugins::capturepixelformat::CapturePixelFormat;
use g2g_plugins::v4l2src::{format_for_fourcc, V4l2Src};

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn device() -> String {
    std::env::var("G2G_V4L2_DEVICE").unwrap_or_else(|_| "/dev/video0".to_string())
}

/// Every fourcc the element carries maps to the caps that format produces, and
/// an unmapped one is refused rather than guessed at. No camera needed.
#[test]
fn fourccs_map_to_the_caps_they_produce() {
    for (fourcc, expected) in [
        (b"YUYV", CapturePixelFormat::Yuyv),
        (b"NV12", CapturePixelFormat::Nv12),
        (b"YU12", CapturePixelFormat::I420),
        (b"MJPG", CapturePixelFormat::Mjpeg),
    ] {
        assert_eq!(format_for_fourcc(fourcc), Some(expected), "{expected:?}");
    }
    // MJPEG is the compressed one: it must not land on a raw format, or a
    // JPEG buffer would be handed to a chain expecting pixels.
    assert_eq!(
        format_for_fourcc(b"MJPG")
            .expect("mjpg")
            .caps(1280, 720, 30),
        Caps::CompressedVideo {
            codec: VideoCodec::Mjpeg,
            width: Dim::Fixed(1280),
            height: Dim::Fixed(720),
            framerate: Rate::Fixed(30 << 16),
        }
    );
    assert!(matches!(
        format_for_fourcc(b"YUYV").expect("yuyv").caps(640, 480, 30),
        Caps::RawVideo {
            format: RawVideoFormat::Yuyv,
            ..
        }
    ));
    // greyscale and bayer have no g2g caps, so they are skipped, not mapped.
    assert_eq!(format_for_fourcc(b"GREY"), None);
    assert_eq!(format_for_fourcc(b"BA81"), None);
}

/// Sink that keeps the first frame's bytes and the caps it was told about, so a
/// test can check what actually came down the link.
#[derive(Default)]
struct FirstFrameSink {
    caps: Option<Caps>,
    first: Option<Vec<u8>>,
    frames: u64,
}

impl AsyncElement for FirstFrameSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.caps = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::CapsChanged(caps) => self.caps = Some(caps),
                PipelinePacket::DataFrame(frame) => {
                    self.frames += 1;
                    if let (None, MemoryDomain::System(slice)) = (&self.first, &frame.domain) {
                        self.first = Some(slice.as_slice().to_vec());
                    }
                }
                _ => {}
            }
            Ok(())
        })
    }
}

/// The device's formats become caps alternatives, with raw YUYV first: a chain
/// that pins nothing must keep getting the raw frames it always did, never the
/// camera's compressed mode.
#[tokio::test]
#[ignore = "needs a real /dev/videoN device (set G2G_V4L2_DEVICE)"]
async fn advertised_set_leads_with_raw_yuyv() {
    let mut src = V4l2Src::new(device()).with_size(640, 480);
    let preferred = SourceLoop::intercept_caps(&mut src)
        .await
        .expect("probe the camera");
    assert!(
        matches!(
            preferred,
            Caps::RawVideo {
                format: RawVideoFormat::Yuyv,
                ..
            }
        ),
        "preferred format must stay YUYV, got {preferred:?}"
    );

    let CapsConstraint::Produces(set) = SourceLoop::caps_constraint(&mut src)
        .await
        .expect("probe the camera")
    else {
        panic!("a source must produce");
    };
    let advertised = set.alternatives();
    assert_eq!(
        advertised.first(),
        Some(&preferred),
        "the preferred format must lead the advertised set"
    );
    // every alternative must fixate, or pinning one fails negotiation.
    for caps in advertised {
        assert_eq!(&caps.fixate().expect("fixates"), caps);
    }
    eprintln!(
        "{} advertises {} formats: {:?}",
        device(),
        advertised.len(),
        advertised
            .iter()
            .map(|c| c.to_gst_string())
            .collect::<Vec<_>>()
    );
}

/// The MJPEG path end to end without a decoder: pinning the link to
/// `image/jpeg` selects the camera's MJPEG mode and the frames really are
/// JPEGs (SOI marker, and a length no raw frame of that geometry could have).
#[tokio::test]
#[ignore = "needs a real /dev/videoN device with an MJPEG mode (set G2G_V4L2_DEVICE)"]
async fn mjpeg_pinned_chain_captures_jpeg_frames() {
    let dev = device();
    let mut src = V4l2Src::new(dev.clone())
        .with_size(1280, 720)
        .with_fps(30)
        .with_frame_limit(5);
    let mut filter = g2g_plugins::capsfilter::CapsFilter::new(Caps::CompressedVideo {
        codec: VideoCodec::Mjpeg,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    });
    let mut sink = FirstFrameSink::default();

    let stats = tokio::time::timeout(
        std::time::Duration::from_secs(20),
        g2g_core::runtime::run_source_transform_sink(
            &mut src,
            &mut filter,
            &mut sink,
            &ZeroClock,
            2,
        ),
    )
    .await
    .expect("capture finishes within 20s")
    .expect("mjpeg capture pipeline runs");

    assert_eq!(
        stats.frames_emitted, 5,
        "source emitted the requested count"
    );
    assert!(sink.frames > 0, "no frames reached the sink");
    assert!(
        matches!(
            sink.caps,
            Some(Caps::CompressedVideo {
                codec: VideoCodec::Mjpeg,
                ..
            })
        ),
        "the link must carry MJPEG, got {:?}",
        sink.caps
    );
    let first = sink.first.expect("first frame kept");
    assert_eq!(
        &first[..2],
        &[0xFF, 0xD8],
        "an MJPEG access unit starts with the JPEG SOI marker"
    );
    // A raw 1280x720 YUYV frame is a fixed 1_843_200 bytes; a JPEG of the same
    // picture is far smaller, so the length alone rules out a raw buffer.
    assert!(
        first.len() < 1_843_200,
        "frame of {} bytes is not compressed",
        first.len()
    );
    eprintln!(
        "captured {} MJPEG frames from {dev}, first is {} bytes",
        sink.frames,
        first.len()
    );
}

/// The headline case: `v4l2src ! mjpegdec ! sink`. Nothing pins the link, the
/// decoder's own constraint does: it accepts only MJPEG, so the solver drops the
/// source's raw alternatives and the camera runs in MJPEG mode. What arrives is
/// decoded raw video of the negotiated geometry.
#[cfg(feature = "mjpeg")]
#[tokio::test]
#[ignore = "needs a real /dev/videoN device with an MJPEG mode (set G2G_V4L2_DEVICE)"]
async fn mjpegdec_downstream_selects_the_cameras_mjpeg_mode() {
    let mut src = V4l2Src::new(device())
        .with_size(1280, 720)
        .with_fps(30)
        .with_frame_limit(5);
    let mut decoder = g2g_plugins::mjpegdec::MjpegDec::new();
    let mut sink = FirstFrameSink::default();

    let stats = tokio::time::timeout(
        std::time::Duration::from_secs(20),
        g2g_core::runtime::run_source_transform_sink(
            &mut src,
            &mut decoder,
            &mut sink,
            &ZeroClock,
            2,
        ),
    )
    .await
    .expect("capture finishes within 20s")
    .expect("mjpeg decode pipeline runs");

    assert_eq!(
        stats.frames_emitted, 5,
        "source emitted the requested count"
    );
    assert!(sink.frames > 0, "no decoded frames reached the sink");
    let Some(Caps::RawVideo {
        format,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        ..
    }) = sink.caps
    else {
        panic!("the decoder must deliver raw video, got {:?}", sink.caps);
    };
    assert_eq!(format, RawVideoFormat::Rgba8);
    let decoded = sink.first.expect("first decoded frame kept");
    assert_eq!(
        decoded.len(),
        (width as usize) * (height as usize) * 4,
        "a decoded RGBA frame is one word per pixel"
    );
    // a decoded camera picture is not a constant plane.
    assert!(
        decoded.iter().any(|&b| b != decoded[0]),
        "decoded frame is uniform, so nothing was really decoded"
    );
    eprintln!(
        "decoded {} frames of {width}x{height} RGBA from {}'s MJPEG mode",
        sink.frames,
        device()
    );

    // real-camera evidence for the MJPEG path, so `g2g-inspect --maturity`
    // derives it the way the YUYV capture is already derived (M949).
    use g2g_core::conformance::{ConformanceDimension, Evidence};
    use g2g_plugins::conformance::persist;
    persist::record_evidence(
        "v4l2src",
        &Evidence::new(ConformanceDimension::Hardware)
            .platform(persist::v4l2_platform_tag(&device()))
            .codec("mjpeg")
            .detail("negotiated the camera's MJPEG mode and mjpegdec decoded it to RGBA"),
    )
    .expect("record hardware evidence");
}

/// The same selection from a launch line through the DAG runner, which reads the
/// source's whole produce set (M954): a text pipeline gets the MJPEG mode too.
#[cfg(feature = "mjpeg")]
#[tokio::test]
#[ignore = "needs a real /dev/videoN device with an MJPEG mode (set G2G_V4L2_DEVICE)"]
async fn mjpeg_launch_pipeline_decodes_to_raw() {
    let registry = g2g_plugins::registry::default_registry();
    let text = format!(
        "v4l2src device={} width=1280 height=720 num-buffers=5 ! mjpegdec ! fakesink",
        device()
    );
    let graph = g2g_core::runtime::parse_launch(&registry, &text).expect("pipeline parses");
    let stats = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        g2g_core::runtime::run_graph(graph, &ZeroClock, 2),
    )
    .await
    .expect("pipeline finishes within 30s")
    .expect("mjpeg decode pipeline runs");
    assert_eq!(
        stats.frames_consumed, 5,
        "every decoded frame should reach the sink"
    );
    eprintln!("decoded {} frames via {text}", stats.frames_consumed);
}
