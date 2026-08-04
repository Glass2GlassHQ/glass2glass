//! M366 - Ogg/Opus demuxer seek (`OggDemux` over a seekable `FileSrc`). Drives an
//! upstream byte-seek and re-syncs from the packet at or after the target. The
//! Ogg demuxer carries no per-packet PTS, so the element now accumulates one from
//! each Opus packet's decoded duration (TOC byte, 48 kHz); every audio packet is
//! a resync point.
//!
//! The clip is five 20 ms Opus packets (PTS 0, 20, 40, 60, 80 ms). A seek to
//! 50 ms resumes from the 60 ms packet.
//!
//! M862 adds the proportional first guess: with enough of the file played to
//! interpolate through, the seek lands near the target instead of at offset `0`,
//! and delivers exactly the packets the re-scan would. A chained file falls back
//! to the re-scan, since one global proportion cannot describe two physical
//! streams.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::element::{AsyncElement, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, PipelinePacket};
use g2g_core::memory::MemoryDomain;
use g2g_core::runtime::{SeekController, SourceLoop};
use g2g_core::{ByteStreamEncoding, Caps, G2gError, Seek};
use g2g_plugins::filesrc::FileSrc;
use g2g_plugins::oggdemux::OggDemux;

use std::path::PathBuf;

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g_m366_{}_{}.opus", std::process::id(), name))
}

/// One Ogg page carrying `packets` (each laced into 255-byte segments), at
/// granule position `granule`.
fn page_g(header_type: u8, serial: u32, seq: u32, granule: u64, packets: &[&[u8]]) -> Vec<u8> {
    let mut table = Vec::new();
    let mut body = Vec::new();
    for p in packets {
        let mut n = p.len();
        loop {
            let seg = n.min(255);
            table.push(seg as u8);
            n -= seg;
            if seg < 255 {
                break;
            }
        }
        body.extend_from_slice(p);
    }
    let mut out = b"OggS".to_vec();
    out.push(0); // version
    out.push(header_type);
    out.extend_from_slice(&granule.to_le_bytes());
    out.extend_from_slice(&serial.to_le_bytes());
    out.extend_from_slice(&seq.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // CRC (ignored on read)
    out.push(table.len() as u8);
    out.extend_from_slice(&table);
    out.extend_from_slice(&body);
    out
}

fn page(header_type: u8, serial: u32, seq: u32, packets: &[&[u8]]) -> Vec<u8> {
    page_g(header_type, serial, seq, 0, packets)
}

fn opus_head(channels: u8) -> Vec<u8> {
    let mut h = b"OpusHead".to_vec();
    h.push(1);
    h.push(channels);
    h.extend_from_slice(&[0, 0]);
    h.extend_from_slice(&48_000u32.to_le_bytes());
    h.extend_from_slice(&[0, 0, 0]);
    h
}

/// OpusHead (BOS) + OpusTags + five 20 ms audio packets (TOC 0x08 = SILK NB
/// 20 ms, one frame), each tagged with a distinct second byte.
fn synthetic_ogg() -> Vec<u8> {
    let serial = 0x0BAD_F00D;
    let pkts: Vec<Vec<u8>> = (0..5u8).map(|i| vec![0x08, 0xA0 + i]).collect();
    let refs: Vec<&[u8]> = pkts.iter().map(|p| p.as_slice()).collect();
    let mut s = Vec::new();
    s.extend_from_slice(&page(0x02, serial, 0, &[&opus_head(2)]));
    s.extend_from_slice(&page(0x00, serial, 1, &[b"OpusTags\0\0\0\0"]));
    s.extend_from_slice(&page(0x00, serial, 2, &refs));
    s
}

/// Samples one 20 ms Opus packet decodes to at 48 kHz.
const PACKET_SAMPLES: u64 = 960;
const PACKET_NS: u64 = 20_000_000;

/// A 20 ms Opus packet (TOC 0x08) of `payload` bytes carrying `index`.
fn audio_packet(index: u32, payload: usize) -> Vec<u8> {
    let mut p = vec![0x08u8, index as u8, (index >> 8) as u8];
    p.resize(payload.max(3), 0x5A);
    p
}

/// One physical stream: OpusHead + OpusTags, then `count` 20 ms packets laced
/// `per_page` to a page whose granule position is the sample count decoded
/// through it (the anchor a mid-file landing re-times from). The final page is
/// flagged end-of-stream.
fn opus_chain(serial: u32, first_index: u32, count: u32, per_page: u32, payload: usize) -> Vec<u8> {
    let mut out = page_g(0x02, serial, 0, 0, &[&opus_head(2)]);
    out.extend_from_slice(&page_g(0x00, serial, 1, 0, &[b"OpusTags\0\0\0\0"]));
    let mut seq = 2u32;
    let mut done = 0u32;
    while done < count {
        let n = per_page.min(count - done);
        let pkts: Vec<Vec<u8>> = (0..n)
            .map(|i| audio_packet(first_index + done + i, payload))
            .collect();
        let refs: Vec<&[u8]> = pkts.iter().map(|p| p.as_slice()).collect();
        done += n;
        let last = done == count;
        let granule = u64::from(done) * PACKET_SAMPLES;
        out.extend_from_slice(&page_g(
            if last { 0x04 } else { 0x00 },
            serial,
            seq,
            granule,
            &refs,
        ));
        seq += 1;
    }
    out
}

/// One frame the demuxer emitted: the codec config rides the same pad, at time
/// zero with no duration.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Emitted {
    payload: Vec<u8>,
    pts_ns: u64,
    duration_ns: u64,
}

impl Emitted {
    /// Codec config rather than audio: `OpusHead`, or one of the three Vorbis
    /// headers (packet type byte odd, then the `vorbis` magic).
    fn is_config(&self) -> bool {
        self.payload.starts_with(b"OpusHead")
            || (self.payload.len() > 7
                && self.payload[0] & 1 == 1
                && &self.payload[1..7] == b"vorbis")
    }
}

#[derive(Default)]
struct Capture {
    frames: Vec<Emitted>,
    flushes: usize,
    segments: usize,
}
impl OutputSink for Capture {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(Frame {
                    domain: MemoryDomain::System(s),
                    timing,
                    ..
                }) => self.frames.push(Emitted {
                    payload: s.as_slice().to_vec(),
                    pts_ns: timing.pts_ns,
                    duration_ns: timing.duration_ns,
                }),
                // A flush discards what came before it, as a downstream element
                // does: only the post-seek stream is compared.
                PipelinePacket::Flush => {
                    self.flushes += 1;
                    self.frames.clear();
                }
                PipelinePacket::Segment(_) => self.segments += 1,
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Sits between the source and the demuxer: counts the bytes the source served
/// (the evidence that a seek did not re-read the file) and arms the app seek
/// once enough of the file has played for the demuxer to have anchors.
struct Chain<'a> {
    demux: &'a mut OggDemux,
    capture: &'a mut Capture,
    bytes: u64,
    arm: Option<(u64, u64, SeekController)>,
}
impl OutputSink for Chain<'_> {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(Frame {
                domain: MemoryDomain::System(s),
                ..
            }) = &packet
            {
                self.bytes += s.as_slice().len() as u64;
            }
            self.demux.process(packet, self.capture).await?;
            if let Some((at, target_ns, ctl)) = &self.arm {
                if self.bytes >= *at {
                    ctl.seek(Seek::flush_to(*target_ns));
                    self.arm = None;
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Play `path` through `filesrc ! oggdemux`, optionally seeking to `target_ns`
/// once `after_bytes` have been served. Returns what the demuxer emitted after
/// the last flush and how many bytes the source read in total.
async fn play(path: &PathBuf, seek: Option<(u64, u64)>, chunk: usize) -> (Capture, u64) {
    play_stream(path, seek, chunk, "opus").await
}

async fn play_stream(
    path: &PathBuf,
    seek: Option<(u64, u64)>,
    chunk: usize,
    stream: &str,
) -> (Capture, u64) {
    let byte = SeekController::new();
    let time = SeekController::new();

    let mut src = FileSrc::new(
        path,
        Caps::ByteStream {
            encoding: ByteStreamEncoding::Ogg,
        },
    )
    .with_chunk_size(chunk)
    .with_seek(byte.clone());
    let mut demux = OggDemux::new().with_seek(time.clone(), byte.clone());
    demux
        .set_property("stream", g2g_core::PropValue::Str(stream.into()))
        .expect("stream property");

    let caps = {
        let c: Pin<Box<dyn Future<Output = _>>> = Box::pin(src.intercept_caps());
        c.await.expect("probe")
    };
    src.configure_pipeline(&caps).expect("configure src");
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Ogg,
        })
        .expect("configure demux");

    let mut capture = Capture::default();
    let bytes = {
        let mut chain = Chain {
            demux: &mut demux,
            capture: &mut capture,
            bytes: 0,
            arm: seek.map(|(target_ns, at)| (at, target_ns, time.clone())),
        };
        src.run(&mut chain).await.expect("filesrc runs");
        chain.bytes
    };
    (capture, bytes)
}

/// What a seek to `target_ns` must deliver: the codec config, then every packet
/// at or after the target, exactly as a full scan produced them.
fn expected_after(reference: &[Emitted], target_ns: u64) -> Vec<Emitted> {
    reference
        .iter()
        .filter(|e| e.is_config() || e.pts_ns >= target_ns)
        .cloned()
        .collect()
}

/// Compare two frame lists, reporting the first difference as
/// `(pts, duration, payload length)`: a whole payload list is megabytes.
fn assert_same_frames(got: &[Emitted], want: &[Emitted], what: &str) {
    let brief = |e: &Emitted| (e.pts_ns, e.duration_ns, e.payload.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(brief(g), brief(w), "{what}: frame {i} (pts, dur, len)");
        assert_eq!(g.payload, w.payload, "{what}: frame {i} payload");
    }
    assert_eq!(got.len(), want.len(), "{what}: frame count");
}

#[tokio::test]
async fn oggdemux_seeks_to_the_target_packet_over_filesrc() {
    let path = temp_path("seek");
    std::fs::write(&path, synthetic_ogg()).unwrap();

    let byte = SeekController::new();
    let time = SeekController::new();
    // Seek to 50 ms: resume from the first packet at/after it, the 60 ms one.
    time.seek(Seek::flush_to(50_000_000));

    let mut src = FileSrc::new(
        &path,
        Caps::ByteStream {
            encoding: ByteStreamEncoding::Ogg,
        },
    )
    .with_chunk_size(16)
    .with_seek(byte.clone());
    let mut demux = OggDemux::new().with_seek(time.clone(), byte.clone());

    let caps = {
        let c: Pin<Box<dyn Future<Output = _>>> = Box::pin(src.intercept_caps());
        c.await.expect("probe")
    };
    src.configure_pipeline(&caps).expect("configure src");
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Ogg,
        })
        .expect("configure demux");

    let mut capture = Capture::default();
    {
        let mut chain = Chain {
            demux: &mut demux,
            capture: &mut capture,
            bytes: 0,
            arm: None,
        };
        src.run(&mut chain).await.expect("filesrc runs");
    }

    assert!(
        capture.flushes >= 1,
        "the upstream byte-seek flushed downstream"
    );
    assert!(capture.segments >= 1, "a resume segment was emitted");
    // The demuxer forwards OpusHead in-band (the decoder's pre-skip source),
    // then the packets at/after the target: 0, 20, 40 ms dropped, resume from
    // 60 ms (0xA3) and 80 ms (0xA4).
    let payloads: Vec<Vec<u8>> = capture.frames.iter().map(|e| e.payload.clone()).collect();
    assert_eq!(
        payloads,
        vec![opus_head(2), vec![0x08u8, 0xA3], vec![0x08u8, 0xA4]],
        "resumed from the 60 ms packet, pre-target packets discarded"
    );
    let _ = std::fs::remove_file(&path);
}

/// M862: a seek near the end of a long file lands on a proportional byte-offset
/// guess and delivers exactly what a full scan does, without re-reading the file
/// from zero.
#[tokio::test]
async fn oggdemux_guessed_seek_matches_a_full_scan_without_rereading() {
    let path = temp_path("guess");
    // 3000 packets of 20 ms = 60 s, four to a page: ~620 kB.
    let count = 3000u32;
    std::fs::write(&path, opus_chain(0x0BAD_F00D, 0, count, 4, 200)).unwrap();
    let file_len = std::fs::metadata(&path).unwrap().len();

    // Reference: the whole file, no seek.
    let (reference, scanned) = play(&path, None, 8 * 1024).await;
    assert_eq!(
        reference.frames.len() as u32,
        count + 1,
        "every packet plus the in-band OpusHead"
    );
    assert_eq!(scanned, file_len, "the reference read the file once");

    // Seek to 50 s, armed once ~96 kB has played (enough for two anchors).
    let target_ns = 50 * 1_000_000_000;
    let (seeked, bytes) = play(&path, Some((target_ns, 96 * 1024)), 8 * 1024).await;

    assert_eq!(seeked.flushes, 1, "one byte-seek, no fallback re-scan");
    assert!(seeked.segments >= 1, "a resume segment was emitted");
    assert_same_frames(
        &seeked.frames,
        &expected_after(&reference.frames, target_ns),
        "guessed landing",
    );
    assert_eq!(
        seeked.frames[1].pts_ns, target_ns,
        "resumed on the packet at the target"
    );
    assert!(
        bytes < file_len,
        "the seek did not re-read the file: {bytes} of {file_len} bytes served"
    );
}

/// M862: the guess lands past the target when the interpolation runs long. The
/// element re-seeks and still delivers the full-scan packets.
#[tokio::test]
async fn oggdemux_recovers_when_the_guess_lands_past_the_target() {
    let path = temp_path("overshoot");
    // The first 10 s of the file are denser than the rest, so a proportion
    // measured there runs long and the landing sits past the target.
    let serial = 0x0BAD_F00Du32;
    let count = 1500u32;
    let mut data = page_g(0x02, serial, 0, 0, &[&opus_head(2)]);
    data.extend_from_slice(&page_g(0x00, serial, 1, 0, &[b"OpusTags\0\0\0\0"]));
    for i in 0..count {
        let payload = if i < 500 { 200 } else { 120 };
        let granule = u64::from(i + 1) * PACKET_SAMPLES;
        let last = i + 1 == count;
        data.extend_from_slice(&page_g(
            if last { 0x04 } else { 0x00 },
            serial,
            i + 2,
            granule,
            &[&audio_packet(i, payload)],
        ));
    }
    std::fs::write(&path, &data).unwrap();

    let (reference, _) = play(&path, None, 8 * 1024).await;
    // Seek to 25 s of 30 s, armed after ~80 kB (inside the dense stretch).
    let target_ns = 25 * 1_000_000_000;
    let (seeked, _) = play(&path, Some((target_ns, 80 * 1024)), 8 * 1024).await;

    assert!(
        seeked.flushes >= 2,
        "the first landing overshot, so a second seek followed"
    );
    assert_same_frames(
        &seeked.frames,
        &expected_after(&reference.frames, target_ns),
        "recovered seek",
    );
}

/// M862: a chained file (two physical streams back to back) cannot be described
/// by one global proportion, so a landing in the wrong chain falls back to the
/// re-scan and stays correct.
#[tokio::test]
async fn oggdemux_chained_seek_falls_back_to_the_rescan() {
    let path = temp_path("chained");
    let per_chain = 1500u32; // 30 s each
    let mut data = opus_chain(0x1111_1111, 0, per_chain, 4, 200);
    data.extend_from_slice(&opus_chain(0x2222_2222, per_chain, per_chain, 4, 200));
    std::fs::write(&path, &data).unwrap();
    let file_len = std::fs::metadata(&path).unwrap().len();

    let (reference, _) = play(&path, None, 8 * 1024).await;
    assert_eq!(
        reference.frames.len() as u32,
        2 * per_chain + 2,
        "both chains' packets plus one OpusHead each"
    );

    // Seek into the second chain, armed while the first one is still playing.
    let target_ns = 45 * 1_000_000_000;
    let (seeked, bytes) = play(&path, Some((target_ns, 96 * 1024)), 8 * 1024).await;

    assert_same_frames(
        &seeked.frames,
        &expected_after(&reference.frames, target_ns),
        "chained seek",
    );
    assert!(
        bytes > file_len,
        "the fallback re-scanned the file: {bytes} served of {file_len}"
    );
    let last = seeked.frames.last().expect("packets were emitted");
    assert_eq!(
        last.pts_ns,
        u64::from(2 * per_chain - 1) * PACKET_NS,
        "played to the end of the second chain"
    );
}

/// M862 on a real encoder's file: Vorbis packets are timed by the lapped
/// `(prev + cur) / 4` model against an initial priming anchor, so a mid-file
/// landing has to pick both up from the page granule and the packet the landing
/// page ended on. Pink noise mixes short and long blocks, where getting either
/// wrong shifts the timeline. Skipped without ffmpeg.
#[tokio::test]
async fn oggdemux_guessed_seek_is_exact_on_an_encoded_vorbis_file() {
    let path = temp_path("vorbis");
    let path = path.with_extension("ogg");
    if std::process::Command::new("ffmpeg")
        .arg("-version")
        .output()
        .is_err()
    {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let ok = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "lavfi"])
        .args(["-i", "anoisesrc=d=60:c=pink:r=44100"])
        .args(["-ac", "2", "-c:a", "libvorbis", "-f", "ogg"])
        .arg(&path)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        eprintln!("skipping: ffmpeg cannot encode vorbis");
        return;
    }
    let file_len = std::fs::metadata(&path).unwrap().len();

    let (reference, _) = play_stream(&path, None, 8 * 1024, "vorbis").await;
    let target_ns = 50 * 1_000_000_000;
    let (seeked, bytes) =
        play_stream(&path, Some((target_ns, 96 * 1024)), 8 * 1024, "vorbis").await;

    assert_eq!(seeked.flushes, 1, "one byte-seek, no fallback re-scan");
    assert_same_frames(
        &seeked.frames,
        &expected_after(&reference.frames, target_ns),
        "guessed landing",
    );
    assert!(
        bytes < file_len,
        "the seek did not re-read the file: {bytes} of {file_len} bytes served"
    );
    let _ = std::fs::remove_file(&path);
}
