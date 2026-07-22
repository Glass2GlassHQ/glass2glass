//! M20: file I/O elements. `FileSink` records a pipeline's byte stream;
//! `FileSrc` replays a file as `DataFrame` chunks. Together they close the
//! record / playback loop (e.g. testsrc -> encode -> filesink, then
//! filesrc -> parse -> decode).

use core::future::Future;
use core::pin::Pin;

use std::path::PathBuf;

use g2g_core::element::{BoxFuture, PushOutcome};
use g2g_core::runtime::{run_simple_pipeline, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, HardwareError, OutputSink,
    PipelineClock, PipelinePacket, Rate, VideoCodec,
};
use g2g_plugins::filesink::FileSink;
use g2g_plugins::filesrc::FileSrc;
use g2g_plugins::videotestsrc::VideoTestSrc;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Unique per-test temp path so parallel tests never collide.
fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g_m20_{}_{}", std::process::id(), name))
}

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Collects every packet a directly-driven source pushes.
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

/// Wildcard sink element that concatenates received frame bytes, for
/// runner-driven replay assertions.
#[derive(Default)]
struct ByteCollectSink {
    bytes: Vec<u8>,
    eos_seen: bool,
}

impl AsyncElement for ByteCollectSink {
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

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    let Some(slice) = f.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.bytes.extend_from_slice(slice);
                }
                PipelinePacket::Eos => self.eos_seen = true,
                _ => {}
            }
            Ok(())
        })
    }
}

/// The deterministic RGBA pattern `VideoTestSrc` emits: byte `i` of frame
/// `seq` is `(i + seq) & 0xFF`.
fn testsrc_expected_bytes(width: usize, height: usize, frames: u64) -> Vec<u8> {
    let bytes_per_frame = width * height * 4;
    let mut expected = Vec::with_capacity(bytes_per_frame * frames as usize);
    for seq in 0..frames {
        for i in 0..bytes_per_frame {
            expected.push(((i as u64).wrapping_add(seq) & 0xFF) as u8);
        }
    }
    expected
}

#[tokio::test]
async fn filesink_records_pipeline_byte_stream() {
    let path = temp_path("record");
    let mut src = VideoTestSrc::new(32, 16, 30, 5);
    let mut sink = FileSink::new(&path);

    run_simple_pipeline(&mut src, &mut sink, &NullClock, 4)
        .await
        .expect("pipeline run");

    assert!(sink.eos_seen(), "EOS must reach the sink");
    assert_eq!(sink.frames_written(), 5);
    let expected = testsrc_expected_bytes(32, 16, 5);
    assert_eq!(sink.bytes_written(), expected.len() as u64);
    let recorded = std::fs::read(&path).expect("recording exists");
    assert_eq!(recorded, expected, "file must hold the frames in order");
    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn filesrc_chunks_file_and_emits_eos() {
    let path = temp_path("replay");
    let payload: Vec<u8> = (0..20u8).collect();
    std::fs::write(&path, &payload).expect("write fixture");

    let mut src = FileSrc::new(&path, h264_caps()).with_chunk_size(7);
    let caps = src.intercept_caps().await.expect("declared caps");
    assert_eq!(caps, h264_caps());
    src.configure_pipeline(&caps).expect("configure");

    let mut out = Collect::default();
    let produced = src.run(&mut out).await.expect("run to EOS");
    assert_eq!(produced, 3, "20 bytes in 7-byte chunks is 3 frames");

    let mut reassembled = Vec::new();
    let mut chunk_lens = Vec::new();
    let mut eos = false;
    for (i, p) in out.packets.iter().enumerate() {
        match p {
            PipelinePacket::DataFrame(f) => {
                let Some(slice) = f.domain.as_system_slice() else {
                    panic!("FileSrc must emit System frames");
                };
                assert_eq!(f.sequence, i as u64, "sequences count up from zero");
                chunk_lens.push(slice.len());
                reassembled.extend_from_slice(slice);
            }
            PipelinePacket::Eos => eos = true,
            other => panic!("unexpected packet {other:?}"),
        }
    }
    assert_eq!(chunk_lens, vec![7, 7, 6]);
    assert_eq!(reassembled, payload);
    assert!(eos, "FileSrc must emit a final Eos");
    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn record_then_replay_round_trip_through_runner() {
    let path = temp_path("roundtrip");

    let mut src = VideoTestSrc::new(16, 8, 30, 4);
    let mut record = FileSink::new(&path);
    run_simple_pipeline(&mut src, &mut record, &NullClock, 4)
        .await
        .expect("record run");

    // Replay with a chunk size that doesn't align with frame boundaries, so
    // the test proves byte-stream fidelity rather than frame echoing.
    let mut replay = FileSrc::new(&path, h264_caps()).with_chunk_size(100);
    let mut collect = ByteCollectSink::default();
    run_simple_pipeline(&mut replay, &mut collect, &NullClock, 4)
        .await
        .expect("replay run");

    assert!(collect.eos_seen);
    assert_eq!(collect.bytes, testsrc_expected_bytes(16, 8, 4));
    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn filesink_fails_loud_before_configure_and_on_bad_path() {
    let mut sink = FileSink::new(temp_path("never_created"));
    let mut out = Collect::default();
    let r = sink
        .process(PipelinePacket::Eos, &mut out)
        .await
        .expect_err("unconfigured sink must reject packets");
    assert_eq!(r, G2gError::NotConfigured);

    // A path inside a directory that doesn't exist cannot be created.
    let mut bad = FileSink::new(temp_path("no_such_dir").join("rec.h264"));
    let err = bad
        .configure_pipeline(&h264_caps())
        .expect_err("create must fail");
    assert!(
        matches!(err, G2gError::Hardware(HardwareError::Io(_))),
        "expected structured Io error, got {err:?}"
    );
}

#[tokio::test]
async fn filesrc_missing_file_fails_loud() {
    let mut src = FileSrc::new(temp_path("does_not_exist"), h264_caps());
    let caps = src.intercept_caps().await.expect("declared caps");
    src.configure_pipeline(&caps).expect("configure");
    let mut out = Collect::default();
    let err = src.run(&mut out).await.expect_err("open must fail");
    assert!(
        matches!(err, G2gError::Hardware(HardwareError::Io(_))),
        "expected structured Io error, got {err:?}"
    );
    assert!(out.packets.is_empty(), "nothing may be emitted on failure");
}
