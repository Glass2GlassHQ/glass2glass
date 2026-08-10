//! M888 CMAF chunked consumption: with `low-latency` set, `DashSrc` reads a
//! segment response as a stream and pushes each complete chunk
//! (`styp` / `moof`+`mdat`) downstream as it arrives, so a segment a low-latency
//! packager is still writing plays from its first chunk.
//!
//! Two claims, both against a real ffmpeg-authored LL-DASH stream (manifest, init
//! and chunked segment fixtures from `ffmpeg -f lavfi -i
//! testsrc=size=128x96:rate=15 -t 1 -c:v libx264 -f dash -ldash 1 -streaming 1
//! -seg_duration 1 -frag_duration 0.2 -movflags cmaf`, one segment of 15 chunks),
//! served by a local HTTP server:
//!
//! 1. Equivalence: a segment consumed chunk by chunk delivers the same byte stream
//!    as one consumed whole, and `Fmp4Demux` recovers byte-identical access units
//!    from both.
//! 2. Early emission: when the server writes the first chunk and then holds the
//!    rest, frames come out before the body completes. The server releases the
//!    tail only once the client has emitted (an ordering hook, not a sleep), and
//!    the whole-response mode is the control: it emits nothing until the tail
//!    lands.

#![cfg(feature = "dash")]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    ByteStreamEncoding, Caps, G2gError, MemoryDomain, OutputSink, PipelinePacket, PushOutcome,
};
use g2g_plugins::dashsrc::DashSrc;
use g2g_plugins::fmp4demux::Fmp4Demux;

const MPD: &[u8] = include_bytes!("fixtures/cmaf_ldash.mpd");
const INIT: &[u8] = include_bytes!("fixtures/cmaf_ldash_init.m4s");
const SEGMENT: &[u8] = include_bytes!("fixtures/cmaf_ldash_chunked.m4s");

/// How long the dribbling server waits for the client to emit before writing the
/// tail anyway, so a client that cannot emit early fails the assertion instead of
/// hanging the test.
const RELEASE_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Default)]
struct CaptureSink {
    /// Each `DataFrame` payload, in order.
    frames: Vec<Vec<u8>>,
    /// Set when a media frame arrives, to release a withheld response tail.
    emitted: Option<Arc<AtomicBool>>,
    /// Per frame, whether the server had already written the withheld tail when
    /// that frame arrived (the early-emission observation).
    tail_written_at: Vec<bool>,
    /// The flag the server sets once it writes the tail.
    tail_written: Option<Arc<AtomicBool>>,
}

impl CaptureSink {
    fn body(&self) -> Vec<u8> {
        self.frames.concat()
    }
}

impl OutputSink for CaptureSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    let tail = self
                        .tail_written
                        .as_ref()
                        .is_some_and(|w| w.load(Ordering::SeqCst));
                    self.tail_written_at.push(tail);
                    self.frames.push(s.to_vec());
                    // The init segment is emitted before the media request goes
                    // out, so only a media frame releases the withheld tail.
                    if self.frames.len() > 1 {
                        if let Some(e) = &self.emitted {
                            e.store(true, Ordering::SeqCst);
                        }
                    }
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn byte_frame(bytes: Vec<u8>) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    }
}

/// The CMAF chunks of `data`: each run of boxes up to and including an `mdat`
/// (so the leading `styp` rides the first chunk), which is what the source is
/// expected to emit one frame per.
fn chunk_spans(data: &[u8]) -> Vec<&[u8]> {
    let mut chunks = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i + 8 <= data.len() {
        let size = u32::from_be_bytes(data[i..i + 4].try_into().unwrap()) as usize;
        assert!(size >= 8 && i + size <= data.len(), "fixture box framing");
        let mdat = &data[i + 4..i + 8] == b"mdat";
        i += size;
        if mdat {
            chunks.push(&data[start..i]);
            start = i;
        }
    }
    assert_eq!(start, data.len(), "fixture ends on a chunk boundary");
    chunks
}

/// Serve the fixture stream. `withhold` splits every media-segment response after
/// that many bytes: the head goes out immediately, the tail only once the client
/// has emitted a frame (or [`RELEASE_TIMEOUT`] passes). Returns the manifest URL.
fn serve(withhold: Option<(usize, Arc<AtomicBool>, Arc<AtomicBool>)>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut stream) = conn else { break };
            let mut req = Vec::new();
            let mut byte = [0u8; 1];
            while stream.read(&mut byte).unwrap_or(0) == 1 {
                req.push(byte[0]);
                if req.ends_with(b"\r\n\r\n") {
                    break;
                }
            }
            let line = String::from_utf8_lossy(&req);
            let path = line.split_whitespace().nth(1).unwrap_or("");
            let (body, media) = match path {
                "/manifest.mpd" => (MPD, false),
                "/init-stream0.m4s" => (INIT, false),
                "/chunk-stream0-00001.m4s" => (SEGMENT, true),
                _ => {
                    let _ = stream.write_all(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    );
                    continue;
                }
            };
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(header.as_bytes());
            match withhold.as_ref().filter(|_| media) {
                Some((head, emitted, tail_written)) => {
                    let _ = stream.write_all(&body[..*head]);
                    let _ = stream.flush();
                    let deadline = Instant::now() + RELEASE_TIMEOUT;
                    while !emitted.load(Ordering::SeqCst) && Instant::now() < deadline {
                        thread::sleep(Duration::from_millis(5));
                    }
                    tail_written.store(true, Ordering::SeqCst);
                    let _ = stream.write_all(&body[*head..]);
                }
                None => {
                    let _ = stream.write_all(body);
                }
            }
            let _ = stream.flush();
        }
    });
    format!("http://127.0.0.1:{port}/manifest.mpd")
}

/// Run `DashSrc` over the served stream to completion.
async fn run_dash(url: String, low_latency: bool, sink: &mut CaptureSink) {
    let mut src = DashSrc::new(url);
    if low_latency {
        src = src.with_low_latency();
    }
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .unwrap();
    src.run(sink).await.expect("DASH run");
}

/// The access units `Fmp4Demux` recovers from a byte stream, fed as one frame per
/// element of `parts` (so the framing the source chose is what the demuxer sees).
async fn demux(parts: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let mut dmx = Fmp4Demux::new();
    dmx.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .unwrap();
    let mut sink = CaptureSink::default();
    for part in parts {
        dmx.process(
            PipelinePacket::DataFrame(byte_frame(part.clone())),
            &mut sink,
        )
        .await
        .expect("demux");
    }
    dmx.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("demux eos");
    sink.frames
}

/// Chunked consumption is byte-for-byte the whole-segment consumption, only
/// framed per chunk: same total byte stream, one frame per CMAF chunk, and
/// `Fmp4Demux` recovers the same access units from either framing.
#[tokio::test]
async fn chunked_consumption_matches_whole_segment() {
    let mut whole = CaptureSink::default();
    run_dash(serve(None), false, &mut whole).await;
    let mut chunked = CaptureSink::default();
    run_dash(serve(None), true, &mut chunked).await;

    // The whole-response path is one frame for the init and one for the segment.
    assert_eq!(whole.frames.len(), 2, "init + whole segment");
    assert_eq!(whole.frames[1], SEGMENT, "the segment arrived intact");

    let spans = chunk_spans(SEGMENT);
    assert!(spans.len() > 1, "the fixture segment is chunked");
    assert_eq!(
        chunked.frames.len(),
        1 + spans.len(),
        "init + one per chunk"
    );
    assert_eq!(chunked.frames[0], INIT);
    for (got, want) in chunked.frames[1..].iter().zip(&spans) {
        assert_eq!(got.as_slice(), *want, "each frame is one complete chunk");
    }
    assert_eq!(
        chunked.body(),
        whole.body(),
        "the byte stream downstream is identical"
    );

    let aus_whole = demux(&whole.frames).await;
    let aus_chunked = demux(&chunked.frames).await;
    assert!(!aus_whole.is_empty(), "the fixture demuxes to access units");
    assert_eq!(
        aus_chunked, aus_whole,
        "the same access units come out of either framing"
    );
}

/// The low-latency claim: with the server holding the segment's tail, the client
/// still emits, and its first frame is out before the tail is written. The
/// whole-response mode is the control, emitting only after the tail lands.
#[tokio::test]
async fn chunks_emit_before_the_response_completes() {
    let head = chunk_spans(SEGMENT)[0].len();

    let emitted = Arc::new(AtomicBool::new(false));
    let tail_written = Arc::new(AtomicBool::new(false));
    let url = serve(Some((
        head,
        Arc::clone(&emitted),
        Arc::clone(&tail_written),
    )));
    let mut sink = CaptureSink {
        emitted: Some(Arc::clone(&emitted)),
        tail_written: Some(Arc::clone(&tail_written)),
        ..CaptureSink::default()
    };
    run_dash(url, true, &mut sink).await;
    assert!(
        sink.frames.len() > 2,
        "the withheld segment still emitted per chunk: {} frames",
        sink.frames.len()
    );
    assert!(
        !sink.tail_written_at[1],
        "the first media chunk was emitted while the rest of the segment was withheld"
    );
    assert_eq!(
        sink.body(),
        [INIT, SEGMENT].concat(),
        "the whole segment still arrived, in order"
    );

    // Control: consuming whole responses, nothing can be emitted until the
    // withheld tail is written (the server gives up after RELEASE_TIMEOUT).
    let emitted = Arc::new(AtomicBool::new(false));
    let tail_written = Arc::new(AtomicBool::new(false));
    let url = serve(Some((
        head,
        Arc::clone(&emitted),
        Arc::clone(&tail_written),
    )));
    let mut sink = CaptureSink {
        emitted: Some(Arc::clone(&emitted)),
        tail_written: Some(Arc::clone(&tail_written)),
        ..CaptureSink::default()
    };
    run_dash(url, false, &mut sink).await;
    assert!(
        sink.tail_written_at[1],
        "the whole-segment path emits no media until the response completes"
    );
    assert_eq!(sink.body(), [INIT, SEGMENT].concat());
}
