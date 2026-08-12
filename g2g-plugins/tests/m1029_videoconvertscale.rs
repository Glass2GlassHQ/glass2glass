//! M1029: `videoconvertscale` converts and scales in one element.
//!
//! The pixels must match what `videoconvert ! videoscale` produces, because the
//! element reuses both, and the format has to reach the output caps so a
//! downstream capsfilter can pin it.

use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, Dim, Frame, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelinePacket, PropValue, PushOutcome, Rate, RawVideoFormat,
};
use g2g_plugins::videoconvertscale::VideoConvertScale;

#[derive(Default)]
struct CollectSink {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for CollectSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        self.packets
            .push(packet_slot.take().expect("poll_push without a packet"));
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn raw(format: RawVideoFormat, w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30),
        interlace: g2g_core::Interlace::Any,
    }
}

fn frame(bytes: Vec<u8>) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns: 0,
            dts_ns: 0,
            duration_ns: 0,
            capture_ns: 0,
            arrival_ns: 0,
            keyframe: false,
        },
        sequence: 0,
        meta: Default::default(),
    }
}

/// One 8x8 RGBA frame with a per-pixel gradient, so a resample that reads the
/// wrong rows shows up in the output bytes.
fn rgba_gradient(w: usize, h: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            bytes.extend_from_slice(&[(x * 8) as u8, (y * 8) as u8, 64, 255]);
        }
    }
    bytes
}

fn run(element: &mut VideoConvertScale, input: Caps, bytes: Vec<u8>) -> Vec<PipelinePacket> {
    run_negotiated(element, input, None, bytes)
}

/// Negotiate the output too, which is what the runner does and what lets the
/// element take its fused single-pass path.
fn run_negotiated(
    element: &mut VideoConvertScale,
    input: Caps,
    output: Option<Caps>,
    bytes: Vec<u8>,
) -> Vec<PipelinePacket> {
    element.configure_pipeline(&input).unwrap();
    if let Some(output) = output {
        element.configure_output(&output).unwrap();
    }
    let mut sink = CollectSink::default();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    rt.block_on(element.process(PipelinePacket::DataFrame(frame(bytes)), &mut sink))
        .unwrap();
    sink.packets
}

#[test]
fn converts_and_scales_in_one_element() {
    let mut element = VideoConvertScale::new(RawVideoFormat::Rgb8, 4, 4);
    let packets = run(
        &mut element,
        raw(RawVideoFormat::Rgba8, 8, 8),
        rgba_gradient(8, 8),
    );

    let caps = packets
        .iter()
        .find_map(|p| match p {
            PipelinePacket::CapsChanged(c) => Some(c.clone()),
            _ => None,
        })
        .expect("the new format and geometry are announced downstream");
    assert_eq!(caps, raw(RawVideoFormat::Rgb8, 4, 4));

    let PipelinePacket::DataFrame(out) = packets.last().unwrap() else {
        panic!("expected a DataFrame downstream");
    };
    let bytes = out.domain.require_system_slice("test").unwrap();
    // 4x4 RGB: three bytes per pixel, no alpha.
    assert_eq!(bytes.len(), 4 * 4 * 3);
}

/// The whole point of the element: same pixels as the two it replaces.
#[test]
fn matches_videoconvert_then_videoscale() {
    let source = rgba_gradient(8, 8);
    let converted = g2g_plugins::videoconvert::convert(
        &source,
        RawVideoFormat::Rgba8,
        RawVideoFormat::Rgb8,
        8,
        8,
    );

    let mut element = VideoConvertScale::new(RawVideoFormat::Rgb8, 8, 8);
    let packets = run(&mut element, raw(RawVideoFormat::Rgba8, 8, 8), source);
    let PipelinePacket::DataFrame(out) = packets.last().unwrap() else {
        panic!("expected a DataFrame downstream");
    };
    let bytes = out.domain.require_system_slice("test").unwrap();
    assert_eq!(bytes, converted.as_ref());
}

#[test]
fn the_format_property_names_the_output_format() {
    let mut element = VideoConvertScale::auto();
    element
        .set_property("format", PropValue::Str("NV12".into()))
        .unwrap();
    element.set_property("width", PropValue::Uint(4)).unwrap();
    element.set_property("height", PropValue::Uint(4)).unwrap();

    let packets = run(
        &mut element,
        raw(RawVideoFormat::Rgba8, 8, 8),
        rgba_gradient(8, 8),
    );
    let caps = packets
        .iter()
        .find_map(|p| match p {
            PipelinePacket::CapsChanged(c) => Some(c.clone()),
            _ => None,
        })
        .expect("caps announced");
    assert_eq!(caps, raw(RawVideoFormat::Nv12, 4, 4));
}

/// The fused path is only reachable once the output caps are negotiated, and it
/// has to agree with the two elements it replaces. Reordering channels commutes
/// with bilinear sampling, so for a packed pair the two are byte-identical.
#[test]
fn the_fused_pass_matches_scaling_then_converting() {
    let source = rgba_gradient(8, 8);
    let two_step = g2g_plugins::videoconvert::convert(
        &g2g_plugins::videoscale::scale(&source, RawVideoFormat::Rgba8, 8, 8, 4, 4),
        RawVideoFormat::Rgba8,
        RawVideoFormat::Rgb8,
        4,
        4,
    );

    let mut element = VideoConvertScale::auto();
    let packets = run_negotiated(
        &mut element,
        raw(RawVideoFormat::Rgba8, 8, 8),
        Some(raw(RawVideoFormat::Rgb8, 4, 4)),
        source,
    );
    let PipelinePacket::DataFrame(out) = packets.last().unwrap() else {
        panic!("expected a DataFrame downstream");
    };
    let bytes = out.domain.require_system_slice("test").unwrap();
    assert_eq!(bytes, two_step.as_ref());
}

/// A 4:2:0 input takes the fused path too: sampled in YUV, converted once per
/// output pixel. A flat frame pins the color math and the plane addressing.
#[test]
fn a_flat_nv12_frame_fuses_to_the_expected_rgb() {
    // Y=126 with neutral chroma is mid grey in BT.601 limited range.
    let mut nv12 = vec![126u8; 8 * 8];
    nv12.extend(core::iter::repeat_n(128u8, 8 * 8 / 2));

    let mut element = VideoConvertScale::auto();
    let packets = run_negotiated(
        &mut element,
        raw(RawVideoFormat::Nv12, 8, 8),
        Some(raw(RawVideoFormat::Rgb8, 4, 4)),
        nv12,
    );
    let PipelinePacket::DataFrame(out) = packets.last().unwrap() else {
        panic!("expected a DataFrame downstream");
    };
    let bytes = out.domain.require_system_slice("test").unwrap();
    assert_eq!(bytes.len(), 4 * 4 * 3);
    assert!(
        bytes.iter().all(|&b| b == 128),
        "a flat grey frame stays flat grey through the fused pass"
    );
}

/// The element has to work through `parse_launch` and the runner, not just when
/// a test configures it by hand: a caps filter downstream is what pins its
/// output, and the runner is what calls `configure_pipeline` / `configure_output`.
#[cfg(feature = "std")]
mod launch {
    use g2g_core::runtime::{parse_launch, run_graph};
    use g2g_core::PipelineClock;
    use g2g_plugins::registry::default_registry;

    struct ZeroClock;
    impl PipelineClock for ZeroClock {
        fn now_ns(&self) -> u64 {
            0
        }
    }

    async fn runs(pipeline: &str) -> u64 {
        let reg = default_registry();
        let graph = parse_launch(&reg, pipeline).expect("pipeline parses");
        run_graph(graph, &ZeroClock, 4)
            .await
            .expect("pipeline runs")
            .frames_consumed
    }

    #[tokio::test]
    async fn scales_in_a_launch_line() {
        assert_eq!(
            runs("videotestsrc num-buffers=5 ! videoconvertscale ! video/x-raw,width=320,height=240 ! fakesink").await,
            5
        );
    }

    #[tokio::test]
    async fn converts_in_a_launch_line() {
        assert_eq!(
            runs("videotestsrc num-buffers=5 ! videoconvertscale ! video/x-raw,format=NV12 ! fakesink").await,
            5
        );
    }

    #[tokio::test]
    async fn converts_and_scales_in_a_launch_line() {
        assert_eq!(
            runs("videotestsrc num-buffers=5 ! videoconvertscale ! video/x-raw,format=NV12,width=320,height=240 ! fakesink").await,
            5
        );
    }
}
