//! M1916: `y4menc` sends one packet per frame, the stream header inside the
//! first, as GStreamer's y4menc does.
#![cfg(feature = "std")]

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AsyncElement, Caps, Dim, G2gError, Interlace, OutputSink, PipelineClock, PushOutcome, Rate,
    RawVideoFormat,
};
use g2g_plugins::registry::default_registry;
use g2g_plugins::y4m::Y4mEnc;

const FRAMES: u64 = 3;

const FRAME_MARKER: &[u8] = b"FRAME\n";

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[derive(Default)]
struct CaptureSink {
    frames: Vec<Vec<u8>>,
}

impl OutputSink for CaptureSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        if let Some(PipelinePacket::DataFrame(frame)) = packet_slot.take() {
            let bytes = frame.domain.as_system_slice().expect("system bytes");
            self.frames.push(bytes.to_vec());
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

#[tokio::test]
async fn a_launch_line_consumes_one_packet_per_frame() {
    let reg = default_registry();
    let line = format!("videotestsrc num-buffers={FRAMES} ! videoconvert ! y4menc ! fakesink");
    let graph = parse_launch(&reg, &line).expect("the line parses and negotiates");
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("the pipeline runs");
    assert_eq!(stats.frames_emitted, FRAMES);
    assert_eq!(stats.frames_consumed, FRAMES);
}

#[tokio::test]
async fn the_stream_header_leads_the_first_frame_packet() {
    let (width, height) = (64, 48);
    let format = RawVideoFormat::I420;
    let caps = Caps::RawVideo {
        format,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        framerate: Rate::Fixed(25 << 16),
        interlace: Interlace::Progressive,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    };
    let frame_bytes = format
        .unpadded_frame_bytes(width, height)
        .expect("a planar frame size") as usize;
    let planes: Vec<Vec<u8>> = (0..FRAMES as u8)
        .map(|fill| vec![fill; frame_bytes])
        .collect();

    let mut element = Y4mEnc::new();
    element
        .configure_pipeline(&caps)
        .expect("planar YUV is accepted");
    let mut sink = CaptureSink::default();
    for bytes in &planes {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.clone().into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        element
            .process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .expect("the frame encodes");
    }

    assert_eq!(sink.frames.len() as u64, FRAMES, "one packet per frame");
    let header_end = sink.frames[0]
        .iter()
        .position(|&b| b == b'\n')
        .expect("a stream header line")
        + 1;
    assert!(sink.frames[0].starts_with(b"YUV4MPEG2 "));
    let mut bodies = vec![&sink.frames[0][header_end..]];
    bodies.extend(sink.frames[1..].iter().map(Vec::as_slice));
    for (body, plane) in bodies.iter().zip(&planes) {
        assert_eq!(&body[..FRAME_MARKER.len()], FRAME_MARKER);
        assert_eq!(&body[FRAME_MARKER.len()..], plane.as_slice());
    }
}
