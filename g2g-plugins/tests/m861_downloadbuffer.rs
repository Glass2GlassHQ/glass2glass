//! M861 `downloadbuffer`: a pushed, non-seekable byte stream absorbed into a
//! temp file and served back as a seekable byte source.
//!
//! The headline is the moov-at-end MP4 over HTTP. `httpsrc` produces the
//! streaming `ByteStream{IsoBmff}`, which the whole-file `qtdemux` (`Mp4Demux`)
//! structurally rejects, so `httpsrc ! qtdemux` does not even build. Spilling to
//! disk makes the stream a whole file with random access, so `downloadbuffer`
//! hands `qtdemux` the `ByteStream{Mp4}` it wants, and the same movie comes out
//! as reading the file locally.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{parse_launch, SeekController};
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, G2gError, OutputSink, PropValue, PushOutcome, Seek,
};
use g2g_plugins::downloadbuffer::DownloadBuffer;
use g2g_plugins::registry::default_registry;

// --- helpers --------------------------------------------------------------

/// Records the packets a transform pushes downstream.
#[derive(Default)]
struct CaptureSink {
    bytes: Vec<u8>,
    /// Packet kinds in order, as short tags, so `Flush` ordering is checkable.
    tags: Vec<&'static str>,
}

impl OutputSink for CaptureSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.bytes.extend_from_slice(s);
                    }
                    self.tags.push("data");
                }
                PipelinePacket::Flush => self.tags.push("flush"),
                PipelinePacket::Eos => self.tags.push("eos"),
                _ => self.tags.push("other"),
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn bytes_frame(data: &[u8]) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.to_vec().into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

fn isobmff() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    }
}

/// A configured buffer plus a controller wired to its downstream-facing seek.
fn configured(template: &str) -> (DownloadBuffer, SeekController) {
    let ctl = SeekController::new();
    let mut db = DownloadBuffer::new()
        .with_temp_template(template)
        .with_blocksize(16)
        .with_seek(ctl.clone());
    db.configure_pipeline(&isobmff()).expect("configures");
    (db, ctl)
}

// --- the element ----------------------------------------------------------

/// The pushed bytes come out unchanged, and land in a spill file that is gone
/// once the element drops.
#[tokio::test]
async fn spills_to_a_temp_file_and_serves_the_same_bytes() {
    let payload: Vec<u8> = (0..500u32).map(|i| (i % 251) as u8).collect();
    let (mut db, _ctl) = configured("g2g-m861-spillXXXXXX");
    let spill = db.temp_location().expect("created at configure").to_owned();
    assert!(spill.exists(), "the spill file exists before any data");

    let mut sink = CaptureSink::default();
    for chunk in payload.chunks(97) {
        db.process(bytes_frame(chunk), &mut sink)
            .await
            .expect("spills");
    }
    db.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("eos drains the tail");

    assert_eq!(sink.bytes, payload, "every byte served, in order");
    assert_eq!(db.high_water(), payload.len() as u64);
    assert_eq!(
        std::fs::read(&spill).expect("read the spill"),
        payload,
        "and the whole stream is on disk"
    );

    drop(db);
    assert!(!spill.exists(), "temp-remove deletes it on drop");
}

/// `temp-remove=false` keeps the download, so it can be reused.
#[tokio::test]
async fn temp_remove_false_keeps_the_spill_file() {
    let mut db = DownloadBuffer::new()
        .with_temp_template("g2g-m861-keepXXXXXX")
        .with_temp_remove(false);
    db.configure_pipeline(&isobmff()).expect("configures");
    let spill = db.temp_location().expect("created").to_owned();
    let mut sink = CaptureSink::default();
    db.process(bytes_frame(b"kept"), &mut sink)
        .await
        .expect("spills");
    drop(db);
    assert_eq!(std::fs::read(&spill).expect("still there"), b"kept");
    let _ = std::fs::remove_file(&spill);
}

/// Two instances never share a spill file: the `XXXXXX` token is per-instance.
#[test]
fn concurrent_instances_get_distinct_spill_files() {
    let mut a = DownloadBuffer::new().with_temp_template("g2g-m861-uniqXXXXXX");
    let mut b = DownloadBuffer::new().with_temp_template("g2g-m861-uniqXXXXXX");
    a.configure_pipeline(&isobmff()).expect("a configures");
    b.configure_pipeline(&isobmff()).expect("b configures");
    assert_ne!(a.temp_location(), b.temp_location());
}

/// A flushing byte seek rewinds the served stream: `Flush` first, then the
/// bytes from the requested offset, read back out of the spill file.
#[tokio::test]
async fn a_byte_seek_re_serves_from_the_spill_file() {
    let payload: Vec<u8> = (0..64u8).collect();
    let (mut db, ctl) = configured("g2g-m861-seekXXXXXX");
    let mut sink = CaptureSink::default();
    db.process(bytes_frame(&payload), &mut sink)
        .await
        .expect("spills");
    assert_eq!(sink.bytes, payload, "served straight through first");

    // Rewind to a byte offset the download already passed. Nothing new arrives:
    // the re-served bytes can only have come from the spill file.
    sink.bytes.clear();
    ctl.seek(Seek::flush_to(40));
    db.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("serves the seek");
    assert_eq!(sink.bytes, payload[40..], "re-served from offset 40");
    assert_eq!(
        sink.tags.iter().position(|t| *t == "flush"),
        Some(4),
        "Flush precedes the re-served chunks: {:?}",
        sink.tags
    );
}

/// A seek past the high-water mark is not an error: the read waits, and resumes
/// exactly at the requested offset once those bytes arrive.
#[tokio::test]
async fn a_seek_past_the_high_water_mark_waits_for_the_bytes() {
    let (mut db, ctl) = configured("g2g-m861-hwmXXXXXX");
    let mut sink = CaptureSink::default();
    db.process(bytes_frame(&[0u8; 32]), &mut sink)
        .await
        .expect("spills");
    sink.bytes.clear();

    ctl.seek(Seek::flush_to(48));
    db.process(bytes_frame(&[1u8; 8]), &mut sink)
        .await
        .expect("still short of the target");
    assert!(
        sink.bytes.is_empty(),
        "high-water is 40, the read waits at 48"
    );

    db.process(bytes_frame(&[2u8; 16]), &mut sink)
        .await
        .expect("the target arrives");
    // High-water is now 56, so bytes 48..56 are the tail of the last chunk.
    assert_eq!(sink.bytes, [2u8; 8], "resumed exactly at offset 48");
}

/// The properties are declared (so `parse_launch` can look up their kinds) and
/// round-trip, with `temp-location` read-only.
#[test]
fn properties_round_trip() {
    let mut db = DownloadBuffer::new();
    let names: Vec<&str> = db.properties().iter().map(|s| s.name).collect();
    for want in [
        "temp-template",
        "temp-location",
        "temp-remove",
        "max-size-bytes",
        "blocksize",
    ] {
        assert!(names.contains(&want), "{want} declared, got {names:?}");
    }
    assert_eq!(db.get_property("temp-remove"), Some(PropValue::Bool(true)));
    assert_eq!(
        db.get_property("max-size-bytes"),
        Some(PropValue::Uint(2 * 1024 * 1024))
    );
    db.set_property("temp-remove", PropValue::Bool(false))
        .expect("settable");
    db.set_property("blocksize", PropValue::Uint(4096))
        .expect("settable");
    assert_eq!(db.get_property("temp-remove"), Some(PropValue::Bool(false)));
    assert_eq!(db.get_property("blocksize"), Some(PropValue::Uint(4096)));
    assert!(
        db.set_property("temp-location", PropValue::Str("/x".into()))
            .is_err(),
        "temp-location reports the file in use, it does not choose it"
    );
}

// --- launch wiring --------------------------------------------------------

/// `downloadbuffer` is in the registry with its properties, and its output is
/// the whole-file `ByteStream{Mp4}` the progressive demuxer needs. Without it a
/// pushed stream has no solution, which is the whole point of the element.
#[test]
fn downloadbuffer_is_what_makes_a_pushed_mp4_negotiate() {
    use g2g_plugins::mp4demux::Mp4Demux;

    let reg = default_registry();
    parse_launch(
        &reg,
        "filesrc location=/x/movie.mp4 \
         ! downloadbuffer temp-template=g2g-m861-launchXXXXXX max-size-bytes=4096 \
         ! qtdemux ! h264parse ! fakesink",
    )
    .expect("parses")
    .finish()
    .expect("registered with its properties, and its output is a whole-file MP4");

    // The streaming form a pushed transport produces is structurally rejected by
    // the whole-file demuxer. The spill rewrites it to the whole-file form,
    // which is exactly what the demuxer's own caps check then accepts.
    let demux = Mp4Demux::new();
    assert!(
        demux.intercept_caps(&isobmff()).is_err(),
        "a pushed IsoBmff stream cannot feed the whole-file demuxer on its own"
    );
    let spilled = DownloadBuffer::new().propose_output_caps(&isobmff());
    assert_eq!(
        spilled,
        Caps::ByteStream {
            encoding: ByteStreamEncoding::Mp4
        }
    );
    assert!(demux.intercept_caps(&spilled).is_ok());
}

// --- the acceptance scenario ----------------------------------------------

/// The moov-at-end MP4 over a pushed, non-seekable HTTP transport, played
/// through the whole-file `qtdemux` and compared against reading the same file
/// locally. Needs `httpsrc`, so the fixture lives here with it.
#[cfg(feature = "http-src")]
mod pushed_over_http {
    use super::*;

    use std::path::PathBuf;

    use g2g_core::runtime::run_graph;
    use g2g_core::{Dim, MultiInputElement, PipelineClock, Rate, VideoCodec};
    use g2g_plugins::mp4muxn::Mp4MuxN;

    struct ZeroClock;
    impl PipelineClock for ZeroClock {
        fn now_ns(&self) -> u64 {
            0
        }
    }

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("g2g-m861-{}-{name}", std::process::id()))
    }

    fn h264_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(320),
            height: Dim::Fixed(240),
            framerate: Rate::Fixed(30 << 16),
        }
    }

    fn annexb(nals: &[&[u8]]) -> Vec<u8> {
        let mut v = Vec::new();
        for n in nals {
            v.extend_from_slice(&[0, 0, 0, 1]);
            v.extend_from_slice(n);
        }
        v
    }

    /// The top-level 4ccs of a file, in order.
    fn top_level_boxes(file: &[u8]) -> Vec<[u8; 4]> {
        let mut out = Vec::new();
        let mut at = 0usize;
        while at + 8 <= file.len() {
            let size = u32::from_be_bytes(file[at..at + 4].try_into().unwrap()) as usize;
            if size < 8 || at + size > file.len() {
                break;
            }
            out.push(file[at + 4..at + 8].try_into().unwrap());
            at += size;
        }
        out
    }

    /// A progressive MP4 with its `moov` after the `mdat`: the layout a pushed,
    /// non-seekable transport cannot serve without random access. Authored with the
    /// in-repo muxer, `fragmented=false faststart=false`.
    async fn author_moov_at_end() -> Vec<u8> {
        let aus = [
            (
                annexb(&[
                    &[0x67, 0x42, 0x00, 0x1E, 0x88],
                    &[0x68, 0xCE, 0x3C, 0x80],
                    &[0x65, 0x11],
                ]),
                0u64,
            ),
            (annexb(&[&[0x41, 0x22]]), 40_000_000),
            (annexb(&[&[0x41, 0x33]]), 80_000_000),
            (annexb(&[&[0x41, 0x44]]), 120_000_000),
        ];
        let mut m = Mp4MuxN::new(1).with_fragmented(false).with_faststart(false);
        let mut sink = CaptureSink::default();
        m.configure_pipeline(0, &h264_caps()).expect("configures");
        for (i, (au, pts)) in aus.iter().enumerate() {
            let f = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(au.clone().into_boxed_slice())),
                FrameTiming {
                    pts_ns: *pts,
                    ..FrameTiming::default()
                },
                i as u64,
            );
            m.process(0, PipelinePacket::DataFrame(f), &mut sink)
                .await
                .expect("mux");
        }
        m.process(0, PipelinePacket::Eos, &mut sink)
            .await
            .expect("mux eos");

        let file = sink.bytes;
        assert_eq!(
            top_level_boxes(&file),
            vec![*b"ftyp", *b"mdat", *b"moov"],
            "the fixture's sample index sits after the media"
        );
        file
    }

    /// Serve `payload` once over HTTP/1.1 on an ephemeral port, returning the URL.
    fn serve_once(payload: Vec<u8>) -> String {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut req = Vec::new();
            let mut byte = [0u8; 1];
            while stream.read(&mut byte).unwrap_or(0) == 1 {
                req.push(byte[0]);
                if req.ends_with(b"\r\n\r\n") {
                    break;
                }
            }
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                payload.len()
            );
            stream.write_all(header.as_bytes()).unwrap();
            stream.write_all(&payload).unwrap();
            stream.flush().unwrap();
        });
        format!("http://127.0.0.1:{port}/movie.mp4")
    }

    /// The moov-at-end MP4 over a pushed, non-seekable HTTP transport plays through
    /// the whole-file `qtdemux`, and produces exactly the frames that reading the
    /// same file locally does.
    #[tokio::test]
    async fn a_pushed_moov_at_end_mp4_plays_and_matches_the_local_file() {
        let file = author_moov_at_end().await;
        let local = temp_path("movie.mp4");
        std::fs::write(&local, &file).expect("stage the file");
        let url = serve_once(file.clone());

        let reg = default_registry();

        // Reference: the seekable local file straight into the demuxer.
        let want = temp_path("local.h264");
        let _ = std::fs::remove_file(&want);
        let graph = parse_launch(
            &reg,
            &format!(
                "filesrc location={} ! qtdemux ! filesink location={}",
                local.display(),
                want.display()
            ),
        )
        .expect("local chain parses");
        let local_stats = run_graph(graph, &ZeroClock, 4).await.expect("local runs");

        // The pushed transport, made seekable by the spill.
        let got = temp_path("pushed.h264");
        let _ = std::fs::remove_file(&got);
        let graph = parse_launch(
            &reg,
            &format!(
                "httpsrc location={url} bytestream-format=mp4 \
             ! downloadbuffer temp-template=g2g-m861-acceptXXXXXX \
             ! qtdemux ! filesink location={}",
                got.display()
            ),
        )
        .expect("pushed chain parses");
        let pushed_stats = run_graph(graph, &ZeroClock, 4).await.expect("pushed runs");

        let want_bytes = std::fs::read(&want).expect("local wrote access units");
        let got_bytes = std::fs::read(&got).expect("pushed wrote access units");
        assert!(!want_bytes.is_empty(), "the demuxer emitted something");
        assert_eq!(
            local_stats.frames_consumed, 4,
            "every access unit in the fixture came back"
        );
        assert_eq!(
            got_bytes, want_bytes,
            "the pushed stream demuxes to the same access units as the local file"
        );
        assert_eq!(
            pushed_stats.frames_consumed, local_stats.frames_consumed,
            "and the same number of them"
        );

        let _ = std::fs::remove_file(&local);
        let _ = std::fs::remove_file(&want);
        let _ = std::fs::remove_file(&got);
    }
}
