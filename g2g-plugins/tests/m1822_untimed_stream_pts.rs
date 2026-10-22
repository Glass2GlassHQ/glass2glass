#![cfg(feature = "std")]

use std::path::PathBuf;

use g2g_core::frame::Frame;
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{parse_launch, run_graph_with_bus};
use g2g_core::{
    AsyncElement, AudioFormat, Bus, BusMessage, Caps, FrameTiming, PipelineClock, PipelinePacket,
};
use g2g_plugins::registry::default_registry;
use g2g_plugins::wavparse::WavParse;

#[path = "common/collect_sink.rs"]
mod collect_sink;
use collect_sink::CollectSink;

const NANOS_PER_SECOND: u64 = 1_000_000_000;
// the rates the fixtures' sequence headers declare
const H264_BFRAMES_FRAME_RATE: u64 = 30;
const H265_BFRAMES_FRAME_RATE: u64 = 30;
const MPEG2_BFRAMES_FRAME_RATE: u64 = 25;
const BUS_CAPACITY: usize = 64;
const LINK_CAPACITY: usize = 4;
// a chunk size that splits the fixture's samples over several buffers
const WAV_CHUNK_BYTES: usize = 1000;
const WAV_BYTE_RATE_OFFSET: usize = 28;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn fixture(name: &str) -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
        .display()
        .to_string()
}

async fn checksum_pts(pipeline: &str) -> Vec<u64> {
    let graph = parse_launch(&default_registry(), pipeline).expect("pipeline parses");
    let (bus, handle) = Bus::new(BUS_CAPACITY);
    let run = async move {
        let result = run_graph_with_bus(graph, &ZeroClock, LINK_CAPACITY, &handle).await;
        drop(handle);
        result
    };
    let collect = async {
        let mut pts = Vec::new();
        while let Some(message) = bus.recv().await {
            if let BusMessage::Checksum { pts_ns, .. } = message {
                pts.push(pts_ns);
            }
        }
        pts
    };
    let (result, pts) = tokio::join!(run, collect);
    result.expect("pipeline runs");
    pts
}

fn frame_grid(frames: usize, frame_rate: u64) -> Vec<u64> {
    (0..frames as u64)
        .map(|frame| (frame * NANOS_PER_SECOND + frame_rate / 2) / frame_rate)
        .collect()
}

fn sorted(mut pts: Vec<u64>) -> Vec<u64> {
    pts.sort_unstable();
    pts
}

#[tokio::test]
async fn h264_elementary_stream_takes_display_order_times_from_zero() {
    let pts = checksum_pts(&format!(
        "filesrc location={} ! h264parse ! checksumsink",
        fixture("h264_64x48_bframes.h264")
    ))
    .await;
    assert!(
        pts.windows(2).any(|pair| pair[1] < pair[0]),
        "the fixture codes B frames ahead of their display slot: {pts:?}"
    );
    assert_eq!(
        sorted(pts.clone()),
        frame_grid(pts.len(), H264_BFRAMES_FRAME_RATE)
    );
}

#[tokio::test]
async fn second_idr_continues_the_timeline_after_the_first_sequence() {
    let sequence = std::fs::read(fixture("h264_64x48_bframes.h264")).expect("h264 fixture");
    let path = std::env::temp_dir().join(format!("g2g-m1822-two-idr-{}.h264", std::process::id()));
    std::fs::write(&path, [sequence.as_slice(), sequence.as_slice()].concat())
        .expect("write the two-sequence stream");
    let pts = checksum_pts(&format!(
        "filesrc location={} ! h264parse ! checksumsink",
        path.display()
    ))
    .await;
    std::fs::remove_file(&path).expect("remove the two-sequence stream");
    assert_eq!(
        sorted(pts.clone()),
        frame_grid(pts.len(), H264_BFRAMES_FRAME_RATE)
    );
}

#[tokio::test]
async fn h265_elementary_stream_takes_display_order_times_from_zero() {
    let pts = checksum_pts(&format!(
        "filesrc location={} ! h265parse ! checksumsink",
        fixture("h265_640x480_bframes.h265")
    ))
    .await;
    assert!(!pts.is_empty());
    assert_eq!(
        sorted(pts.clone()),
        frame_grid(pts.len(), H265_BFRAMES_FRAME_RATE)
    );
}

#[tokio::test]
async fn mpeg2_elementary_stream_takes_display_order_times_from_zero() {
    let pts = checksum_pts(&format!(
        "filesrc location={} ! mpegvideoparse ! checksumsink",
        fixture("mpeg2_64x48_25fps_bframes.m2v")
    ))
    .await;
    assert!(
        pts.windows(2).any(|pair| pair[1] < pair[0]),
        "the fixture codes B frames ahead of their display slot: {pts:?}"
    );
    assert_eq!(
        sorted(pts.clone()),
        frame_grid(pts.len(), MPEG2_BFRAMES_FRAME_RATE)
    );
}

#[tokio::test]
async fn h264_without_a_frame_rate_leaves_later_frames_unstamped() {
    let pts = checksum_pts(&format!(
        "filesrc location={} ! h264parse ! checksumsink",
        fixture("h264_64x48_no_vui_timing.h264")
    ))
    .await;
    assert!(pts.len() > 1, "{pts:?}");
    assert_eq!(pts[0], 0, "the first frame opens the stream at 0");
    assert!(
        pts[1..].iter().all(|pts| *pts == FrameTiming::PTS_NONE),
        "{pts:?}"
    );
}

#[cfg(feature = "ffmpeg")]
#[tokio::test]
async fn decoded_h264_elementary_stream_rises_by_the_frame_period() {
    let pts = checksum_pts(&format!(
        "filesrc location={} ! decodebin ! videoconvert ! checksumsink",
        fixture("h264_64x48_bframes.h264")
    ))
    .await;
    assert!(!pts.is_empty());
    assert_eq!(pts, frame_grid(pts.len(), H264_BFRAMES_FRAME_RATE));
}

#[tokio::test]
async fn wavparse_times_each_buffer_by_its_data_offset() {
    let file = std::fs::read(fixture("sine_8k_mono_s16le.wav")).expect("wav fixture");
    let byte_rate = u64::from(u32::from_le_bytes(
        file[WAV_BYTE_RATE_OFFSET..WAV_BYTE_RATE_OFFSET + 4]
            .try_into()
            .unwrap(),
    ));
    let mut parse = WavParse::new();
    parse
        .configure_pipeline(&Caps::ByteStream {
            encoding: g2g_core::ByteStreamEncoding::Wav,
        })
        .expect("wavparse configures");
    let mut out = CollectSink::default();
    for (sequence, chunk) in file.chunks(WAV_CHUNK_BYTES).enumerate() {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(chunk.to_vec().into_boxed_slice())),
            FrameTiming::default(),
            sequence as u64,
        );
        parse
            .process(PipelinePacket::DataFrame(frame), &mut out)
            .await
            .expect("wavparse takes the chunk");
    }
    parse
        .process(PipelinePacket::Eos, &mut out)
        .await
        .expect("wavparse takes eos");
    let caps = out.packets.iter().find_map(|packet| match packet {
        PipelinePacket::CapsChanged(Caps::Audio { format, .. }) => Some(*format),
        _ => None,
    });
    assert!(matches!(caps, Some(AudioFormat::PcmS16Le)), "{caps:?}");
    let buffers: Vec<(u64, usize)> = out
        .packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some((
                frame.timing.pts_ns,
                frame.domain.as_system_slice().expect("system memory").len(),
            )),
            _ => None,
        })
        .collect();
    assert!(buffers.len() > 1, "the chunks split the data: {buffers:?}");
    let mut offset = 0u64;
    for (pts_ns, len) in buffers {
        // the buffer starts `offset` bytes into the data, less than a nanosecond's worth before its pts
        let elapsed = u128::from(pts_ns) * u128::from(byte_rate);
        let exact = u128::from(offset) * u128::from(NANOS_PER_SECOND);
        assert!(
            elapsed >= exact && elapsed - exact < u128::from(byte_rate),
            "pts {pts_ns} at data offset {offset}"
        );
        offset += len as u64;
    }
}
