#![cfg(feature = "std")]

use g2g_core::fanout::MultiOutputSink;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, G2gError, MultiOutputElement, OutputSink, PushOutcome,
    Segment,
};
use g2g_plugins::flvdemux::FlvDemux;
use g2g_plugins::mkvdemux::{MkvDemux, MkvDemuxN, MkvStream};
use g2g_plugins::mp4demuxn::{forwardable_streams, Mp4DemuxN, Mp4Port};
use g2g_plugins::psdemux::{PsDemuxN, PsStream};
use g2g_plugins::tsdemux::{TsDemuxN, TsStream};

// ffmpeg -f lavfi -i testsrc=size=64x48:rate=25:duration=2 -c:v libx264
//   -x264-params open_gop=1:keyint=9:min-keyint=9:scenecut=0:b-adapt=0:bframes=2:b-pyramid=none o.mp4
// ffmpeg -ss 0.36 -i o.mp4 -c copy -copyts -frames:v 9 <out>: the open-GOP I frame at 0.36 comes first,
// the two B frames after it present at 0.28 and 0.32
const LEADING_BFRAMES_MKV: &str = "h264_64x48_25fps_leading_bframes.mkv";
const LEADING_BFRAMES_FLV: &str = "h264_64x48_25fps_leading_bframes.flv";
// the same cut with -movflags +empty_moov+frag_every_frame, so the I frame's fragment carries no B frame
const LEADING_BFRAMES_FRAGMENTED_MP4: &str = "h264_64x48_25fps_leading_bframes_frag.mp4";
// its video comes first in the file, its audio starts 10 ms before the video
const AUDIO_FIRST_TS: &str = "h264_mp2_audio_first.ts";
// ffmpeg -f lavfi -i testsrc=size=64x48:rate=25:duration=0.4 -itsoffset -0.04
//   -f lavfi -i sine=frequency=440:sample_rate=48000:duration=0.44
//   -c:v mpeg2video -g 5 -c:a mp2 -ac 1 -b:a 32k -f mpeg <out>
// its video comes first in the file with dts 0.51, its audio starts at 0.5
const AUDIO_FIRST_PS: &str = "mpeg2_mp2_audio_first.mpg";

fn fixture(name: &str) -> Vec<u8> {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

fn data_frame(bytes: Vec<u8>) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

fn byte_stream(encoding: ByteStreamEncoding) -> Caps {
    Caps::ByteStream { encoding }
}

// one port's output: the pts of each frame with the segment it went out under
#[derive(Debug, Default)]
struct PortOutput {
    segment: Option<Segment>,
    frames: Vec<(u64, Segment)>,
}

impl PortOutput {
    fn record(&mut self, packet: PipelinePacket) {
        match packet {
            PipelinePacket::Segment(segment) => self.segment = Some(segment),
            PipelinePacket::DataFrame(frame) => {
                let segment = self.segment.expect("a segment before the first frame");
                self.frames.push((frame.timing.pts_ns, segment));
            }
            _ => {}
        }
    }

    fn earliest_pts_ns(&self) -> u64 {
        self.frames
            .iter()
            .map(|&(pts_ns, _)| pts_ns)
            .min()
            .expect("the port emits frames")
    }

    fn assert_nothing_clipped(&self, label: &str) {
        for &(pts_ns, segment) in &self.frames {
            assert!(
                pts_ns >= segment.start,
                "{label}: the frame at {pts_ns} ns went out under a segment starting at {} ns",
                segment.start
            );
        }
    }
}

#[derive(Debug, Default)]
struct SinglePort(PortOutput);

impl OutputSink for SinglePort {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        self.0
            .record(packet_slot.take().expect("poll_push without a packet"));
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

#[derive(Debug)]
struct Ports(Vec<PortOutput>);

impl MultiOutputSink for Ports {
    fn port_count(&self) -> usize {
        self.0.len()
    }

    fn poll_push_to(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        port: usize,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        self.0[port].record(packet_slot.take().expect("poll_push_to without a packet"));
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

async fn demux_single(mut demux: impl AsyncElement, input: Caps, file: &str) -> PortOutput {
    demux.configure_pipeline(&input).expect("configure");
    let mut sink = SinglePort::default();
    for packet in [data_frame(fixture(file)), PipelinePacket::Eos] {
        demux.process(packet, &mut sink).await.expect("demux");
    }
    sink.0
}

async fn demux_ports(
    mut demux: impl MultiOutputElement,
    port_count: usize,
    input: Caps,
    file: &str,
) -> Vec<PortOutput> {
    demux.configure_pipeline(&input).expect("configure");
    let mut sink = Ports((0..port_count).map(|_| PortOutput::default()).collect());
    for packet in [data_frame(fixture(file)), PipelinePacket::Eos] {
        demux.process(packet, &mut sink).await.expect("demux");
    }
    sink.0
}

// every port ends on one segment starting at the earliest unit of any port
fn assert_lowered_to_the_earliest_unit(label: &str, ports: &[PortOutput]) {
    for (port, output) in ports.iter().enumerate() {
        output.assert_nothing_clipped(&format!("{label} port {port}"));
    }
    let earliest_pts_ns = ports.iter().map(PortOutput::earliest_pts_ns).min();
    let segments: Vec<_> = ports.iter().map(|output| output.segment).collect();
    assert!(
        segments.iter().all(|segment| *segment == segments[0]),
        "{label}: the ports share one segment, got {segments:?}"
    );
    assert_eq!(
        segments[0].map(|segment| segment.start),
        earliest_pts_ns,
        "{label}: the segment starts at the earliest unit"
    );
}

#[tokio::test]
async fn mkv_leading_b_frames_lower_the_segment() {
    let input = byte_stream(ByteStreamEncoding::Matroska);
    let single = demux_single(
        MkvDemux::new().with_stream(MkvStream::H264),
        input.clone(),
        LEADING_BFRAMES_MKV,
    )
    .await;
    single.assert_nothing_clipped("MkvDemux");
    let segment = single.segment.expect("a segment");
    assert_eq!(segment.start, single.earliest_pts_ns());
    assert_eq!(
        segment.to_stream_time(segment.start),
        Some(0),
        "positions count from the lowered stream start"
    );

    let ports = demux_ports(
        MkvDemuxN::new(vec![MkvStream::H264]),
        1,
        input,
        LEADING_BFRAMES_MKV,
    )
    .await;
    assert_lowered_to_the_earliest_unit("MkvDemuxN", &ports);
}

#[tokio::test]
async fn flv_leading_b_frames_lower_the_segment() {
    let output = demux_single(
        FlvDemux::new(),
        byte_stream(ByteStreamEncoding::Flv),
        LEADING_BFRAMES_FLV,
    )
    .await;
    output.assert_nothing_clipped("FlvDemux");
    let segment = output.segment.expect("a segment");
    assert_eq!(segment.start, output.earliest_pts_ns());
    assert_eq!(segment.to_stream_time(segment.start), Some(0));
}

#[tokio::test]
async fn fragmented_mp4_leading_b_frames_lower_the_segment() {
    let ports: Vec<Mp4Port> = forwardable_streams(&fixture(LEADING_BFRAMES_FRAGMENTED_MP4))
        .into_iter()
        .map(|stream| Mp4Port {
            track_id: stream.track_id,
            caps: stream.caps,
        })
        .collect();
    let port_count = ports.len();
    let outputs = demux_ports(
        Mp4DemuxN::new(ports),
        port_count,
        byte_stream(ByteStreamEncoding::Mp4),
        LEADING_BFRAMES_FRAGMENTED_MP4,
    )
    .await;
    assert_lowered_to_the_earliest_unit("Mp4DemuxN", &outputs);
}

#[tokio::test]
async fn ts_audio_starting_first_lowers_every_port() {
    let ports = demux_ports(
        TsDemuxN::new(vec![TsStream::H264, TsStream::Mp2]),
        2,
        byte_stream(ByteStreamEncoding::MpegTs),
        AUDIO_FIRST_TS,
    )
    .await;
    assert_lowered_to_the_earliest_unit("TsDemuxN", &ports);
}

#[tokio::test]
async fn ps_audio_starting_first_lowers_every_port() {
    let ports = demux_ports(
        PsDemuxN::new(vec![PsStream::Mpeg2, PsStream::Mp2]),
        2,
        byte_stream(ByteStreamEncoding::MpegPs),
        AUDIO_FIRST_PS,
    )
    .await;
    assert_lowered_to_the_earliest_unit("PsDemuxN", &ports);
}
