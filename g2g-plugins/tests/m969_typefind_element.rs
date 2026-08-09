//! M969 - `typefind` as a mid-graph element. Content sniffing already types a
//! file at negotiation (`filesrc`, M478); this is the runtime half, for a byte
//! stream whose source can only guess: the element holds back the leading
//! frames, sniffs them, and re-declares its output caps with a `CapsChanged`
//! before the data flows on unchanged.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{parse_launch, run_graph, run_source_transform_sink, SourceLoop};
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, ConfigureOutcome, FrameTiming, G2gError, MemoryDomain,
    OutputSink, PipelineClock, PipelinePacket, VideoCodec,
};
use g2g_plugins::registry::default_registry;
use g2g_plugins::typefind::TypeFind;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn bytestream(encoding: ByteStreamEncoding) -> Caps {
    Caps::ByteStream { encoding }
}

/// Byte source that declares a type it cannot actually know (the socket /
/// application-push case) and pushes `payload` in fixed-size chunks, so the
/// element under test has to accumulate across frames.
struct ChunkedByteSource {
    declared: Caps,
    payload: Vec<u8>,
    chunk: usize,
}

impl SourceLoop for ChunkedByteSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.declared.clone()))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let mut pushed = 0u64;
            for (i, part) in self.payload.chunks(self.chunk).enumerate() {
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(
                        part.to_vec().into_boxed_slice(),
                    )),
                    timing: FrameTiming::default(),
                    sequence: i as u64,
                    meta: Default::default(),
                };
                out.push(PipelinePacket::DataFrame(frame)).await?;
                pushed += 1;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(pushed)
        })
    }
}

#[derive(Default)]
struct SinkLog {
    caps_changes: Vec<Caps>,
    bytes: Vec<u8>,
    frames: u32,
}

/// Sink that accepts any caps and records what reached it: every `CapsChanged`
/// and every payload byte, so the test can check both the re-declared type and
/// that the data passed through untouched.
struct RecordingSink {
    log: Arc<Mutex<SinkLog>>,
}

impl AsyncElement for RecordingSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let log = Arc::clone(&self.log);
        Box::pin(async move {
            let mut log = log.lock().unwrap();
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    log.frames += 1;
                    if let Some(bytes) = frame.domain.as_system_slice() {
                        log.bytes.extend_from_slice(bytes);
                    }
                }
                PipelinePacket::CapsChanged(caps) => log.caps_changes.push(caps),
                _ => {}
            }
            Ok(())
        })
    }
}

/// Run `source -> typefind -> sink` over `payload` in `chunk`-sized frames, with
/// the source declaring `declared`.
async fn run_typefind(
    declared: Caps,
    payload: Vec<u8>,
    chunk: usize,
) -> (Result<(), G2gError>, SinkLog) {
    let mut source = ChunkedByteSource {
        declared,
        payload,
        chunk,
    };
    let mut typefind = TypeFind::new();
    let log = Arc::new(Mutex::new(SinkLog::default()));
    let mut sink = RecordingSink {
        log: Arc::clone(&log),
    };
    let outcome = run_source_transform_sink(&mut source, &mut typefind, &mut sink, &ZeroClock, 8)
        .await
        .map(|_| ());
    let recorded = std::mem::take(&mut *log.lock().unwrap());
    (outcome, recorded)
}

/// Container bytes arriving under a wrong declared type are re-typed mid-stream:
/// the sink sees a `CapsChanged` naming the sniffed container, and every byte
/// still arrives unchanged. The 3-byte chunking forces the element to accumulate
/// across frames before the 4-byte magic is complete.
#[tokio::test]
async fn container_bytes_retype_the_stream_and_pass_through() {
    for (payload, expected) in [
        (matroska_header(), ByteStreamEncoding::Matroska),
        (ogg_header(), ByteStreamEncoding::Ogg),
    ] {
        let (outcome, log) =
            run_typefind(bytestream(ByteStreamEncoding::MpegTs), payload.clone(), 3).await;
        outcome.unwrap_or_else(|e| panic!("{expected:?} stream runs: {e:?}"));
        assert!(
            log.caps_changes.contains(&bytestream(expected)),
            "sniffed {expected:?} reached the sink, got {:?}",
            log.caps_changes
        );
        assert_eq!(log.bytes, payload, "{expected:?} bytes passed through");
    }
}

/// A raw Annex-B elementary stream sniffs to `Caps::CompressedVideo` (not a
/// container), so a bare H.264 byte stream types itself for a parser downstream.
#[tokio::test]
async fn annexb_bytes_retype_to_compressed_video() {
    let payload = h264_annexb();
    let (outcome, log) =
        run_typefind(bytestream(ByteStreamEncoding::MpegTs), payload.clone(), 3).await;
    outcome.expect("annex-b stream runs");
    assert!(
        log.caps_changes.iter().any(|caps| matches!(
            caps,
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            }
        )),
        "sniffed H.264 caps reached the sink, got {:?}",
        log.caps_changes
    );
    assert_eq!(log.bytes, payload, "annex-b bytes passed through");
}

/// Bytes that match nothing fail the run once the header budget is spent, rather
/// than flowing on under a type nobody verified.
#[tokio::test]
async fn unsniffable_bytes_fail_the_run() {
    // 0xEE never starts a signature and is not valid UTF-8, so no sniff can hit.
    let payload = vec![0xEEu8; 16 * 1024];
    let (outcome, log) = run_typefind(bytestream(ByteStreamEncoding::MpegTs), payload, 512).await;
    assert_eq!(
        outcome,
        Err(G2gError::CapsMismatch),
        "an untypable stream fails loud"
    );
    assert_eq!(log.frames, 0, "no untyped frame reached the sink");
}

/// `typefind` is reachable from a launch line: the graph parses, negotiates on
/// the source's (wrong) declared type, and the sniff corrects it at runtime.
#[tokio::test]
async fn parse_launch_line_with_typefind_runs() {
    // A `.dat` extension gives filesrc no hint, so it declares its MPEG-TS
    // default over what is really a Matroska file.
    let path = std::env::temp_dir().join(format!("g2g-m969-{}.dat", std::process::id()));
    std::fs::write(&path, matroska_header()).expect("write temp");
    let line = format!("filesrc location={} ! typefind ! fakesink", path.display());

    let registry = default_registry();
    let graph = parse_launch(&registry, &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let stats = run_graph(graph, &ZeroClock, 4).await;
    std::fs::remove_file(&path).ok();
    let consumed = stats
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"))
        .frames_consumed;
    assert!(consumed >= 1, "the file's bytes reached the sink");
}

/// EBML header plus enough filler that the source pushes several frames.
fn matroska_header() -> Vec<u8> {
    let mut data = vec![0x1A, 0x45, 0xDF, 0xA3];
    data.extend((0u8..60).map(|i| i.wrapping_mul(7)));
    data
}

/// An Ogg page capture pattern plus filler.
fn ogg_header() -> Vec<u8> {
    let mut data = b"OggS\0\x02\0\0\0\0\0\0\0\0".to_vec();
    data.extend((0u8..50).map(|i| i.wrapping_mul(3)));
    data
}

/// Annex-B: an access-unit delimiter then an SPS, which is what decides H.264.
fn h264_annexb() -> Vec<u8> {
    let mut data = vec![0x00, 0x00, 0x00, 0x01, 0x09, 0x10];
    data.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1E, 0x8C, 0x8D]);
    data.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x00]);
    data
}
