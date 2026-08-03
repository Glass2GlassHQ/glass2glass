//! M827: chained Ogg. A second physical stream (a fresh beginning-of-stream
//! page after the first one's end-of-stream page, what a recorded radio stream
//! looks like) plays through on the same output pad: its headers go downstream
//! in-band, its parameters re-announce via `CapsChanged`, and a `Segment` opens
//! its timeline at the summed playable duration of the chains before it, so the
//! chains concatenate sample-exactly. A chain that changes codec fails loud.
//!
//! Reference peer: ffmpeg builds the vectors (two independently encoded files
//! concatenated, the canonical chained form), ffprobe's `duration_ts` /
//! `initial_padding` fix where the second chain must start, and ffmpeg's own
//! decode of each part is the PCM oracle. ffmpeg's packet timestamps restart at
//! the boundary for Opus (its own decode of a chained file still runs the two
//! parts back to back) while GStreamer reports the concatenated duration
//! (1 s + 1 s = 2 s) for both mappings; g2g presents the GStreamer timeline.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::path::{Path, PathBuf};
use std::process::Command;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
#[cfg(any(feature = "opus", feature = "vorbis"))]
use g2g_core::runtime::parse_launch;
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, G2gError, MultiOutputElement,
    MultiOutputSink, OutputSink, PropValue, PushOutcome, Segment,
};
use g2g_plugins::ogg::OggCodec;
use g2g_plugins::oggdemux::{OggDemux, OggDemuxN, OggPort};
#[cfg(any(feature = "opus", feature = "vorbis"))]
use g2g_plugins::registry::default_registry;

#[cfg(any(feature = "opus", feature = "vorbis"))]
struct ZeroClock;
#[cfg(any(feature = "opus", feature = "vorbis"))]
impl g2g_core::PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// What the demuxer put out: the announced caps, the chain segments, and every
/// frame's payload with its timing.
#[derive(Default)]
struct CaptureSink {
    caps: Vec<Caps>,
    segments: Vec<Segment>,
    frames: Vec<(Vec<u8>, FrameTiming)>,
}

impl CaptureSink {
    /// The audio packets: everything that is not an in-band codec header.
    fn audio(&self) -> Vec<(Vec<u8>, FrameTiming)> {
        self.frames
            .iter()
            .filter(|(p, _)| !is_header(p))
            .cloned()
            .collect()
    }
}

impl OutputSink for CaptureSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::CapsChanged(c) => self.caps.push(c),
                PipelinePacket::Segment(s) => self.segments.push(s),
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.frames.push((s.to_vec(), f.timing));
                    }
                }
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

impl MultiOutputSink for CaptureSink {
    fn port_count(&self) -> usize {
        1
    }

    fn push_to<'a>(
        &'a mut self,
        _port: usize,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        OutputSink::push(self, packet)
    }
}

/// A codec setup header, which must never reach the output as audio.
fn is_header(packet: &[u8]) -> bool {
    packet.starts_with(b"OpusHead")
        || packet.starts_with(b"OpusTags")
        || packet.starts_with(b"fLaC")
        || (packet.first().is_some_and(|b| b & 1 == 1) && packet[1..].starts_with(b"vorbis"))
}

fn chunk(data: &[u8]) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.to_vec().into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

fn temp_path(tag: &str, ext: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m827-{tag}-{}.{ext}", std::process::id()))
}

/// Encode one tone to `path` with ffmpeg. `codec` is `libopus` or `libvorbis`.
fn encode(path: &Path, codec: &str, freq: u32, channels: u8, rate: u32) -> Option<()> {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        return None;
    }
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "lavfi", "-i"])
        .arg(format!(
            "sine=frequency={freq}:duration=1:sample_rate={rate}"
        ))
        .args(["-ac", &channels.to_string(), "-c:a", codec])
        .arg(path)
        .status()
        .ok()?;
    status.success().then_some(())
}

/// Two independently encoded tones plus the bytes of the two concatenated (the
/// chained file), or `None` when the host has no ffmpeg.
fn chained_vector(tag: &str, codec: &str, channels: u8, rate: u32) -> Option<ChainedVector> {
    let first = temp_path(&format!("{tag}-1"), "ogg");
    let second = temp_path(&format!("{tag}-2"), "ogg");
    encode(&first, codec, 440, channels, rate)?;
    encode(&second, codec, 880, channels, rate)?;
    let first_bytes = std::fs::read(&first).ok()?;
    let second_bytes = std::fs::read(&second).ok()?;
    let chained = temp_path(tag, "ogg");
    let mut bytes = first_bytes.clone();
    bytes.extend_from_slice(&second_bytes);
    std::fs::write(&chained, &bytes).ok()?;
    Some(ChainedVector {
        first,
        second,
        chained,
        first_bytes,
        second_bytes,
        bytes,
    })
}

struct ChainedVector {
    first: PathBuf,
    second: PathBuf,
    chained: PathBuf,
    first_bytes: Vec<u8>,
    second_bytes: Vec<u8>,
    bytes: Vec<u8>,
}

impl Drop for ChainedVector {
    fn drop(&mut self) {
        for p in [&self.first, &self.second, &self.chained] {
            let _ = std::fs::remove_file(p);
        }
    }
}

/// A file's timing as ffprobe measures it, in nanoseconds: `playable` is its
/// end granule less the encoder delay the decoder discards (where the next
/// chain must start), `padding` that delay, which the demuxer's packet
/// timestamps still count (they run on the coded timeline).
struct Probed {
    playable: u64,
    padding: u64,
}

fn probe_ns(path: &Path) -> Probed {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=duration_ts,initial_padding,time_base",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(path)
        .output()
        .expect("ffprobe runs");
    let text = String::from_utf8_lossy(&out.stdout);
    let field = |key: &str| -> String {
        text.lines()
            .find_map(|l| l.strip_prefix(&format!("{key}=")))
            .unwrap_or_else(|| panic!("ffprobe reports {key} for {}", path.display()))
            .trim()
            .to_string()
    };
    let duration_ts: u64 = field("duration_ts").parse().expect("duration_ts");
    let padding: u64 = field("initial_padding").parse().unwrap_or(0);
    let rate: u64 = field("time_base")
        .rsplit('/')
        .next()
        .and_then(|d| d.parse().ok())
        .expect("time base denominator");
    let ns = |samples: u64| samples * 1_000_000_000 / rate;
    Probed {
        playable: ns(duration_ts - padding),
        padding: ns(padding),
    }
}

/// Demux `ogg` for `stream`, fed in small chunks so a chain boundary lands
/// mid-buffer.
async fn demux(ogg: &[u8], stream: &str) -> CaptureSink {
    let mut d = OggDemux::new();
    d.set_property("stream", PropValue::Str(stream.into()))
        .expect("stream property");
    d.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::Ogg,
    })
    .expect("configure");
    let mut sink = CaptureSink::default();
    for piece in ogg.chunks(997) {
        AsyncElement::process(&mut d, chunk(piece), &mut sink)
            .await
            .expect("demux");
    }
    AsyncElement::process(&mut d, PipelinePacket::Eos, &mut sink)
        .await
        .expect("eos");
    sink
}

#[cfg(any(feature = "opus", feature = "vorbis"))]
/// ffmpeg's own decode of `path` to interleaved S16LE, the PCM oracle.
fn ffmpeg_pcm(path: &Path) -> Vec<u8> {
    let out = Command::new("ffmpeg")
        .args(["-loglevel", "error", "-i"])
        .arg(path)
        .args(["-f", "s16le", "-c:a", "pcm_s16le", "pipe:1"])
        .output()
        .expect("ffmpeg decodes");
    assert!(out.status.success(), "ffmpeg decodes {}", path.display());
    out.stdout
}

#[cfg(any(feature = "opus", feature = "vorbis"))]
/// Run a launch line to completion, returning the frames the sink consumed.
async fn run_line(line: &str) -> u64 {
    let reg = default_registry();
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    g2g_core::runtime::run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"))
        .frames_consumed
}

#[cfg(any(feature = "opus", feature = "vorbis"))]
/// Every sample within 1 LSB of the reference, same length.
fn assert_pcm_matches(pcm: &[u8], reference: &[u8]) {
    assert_eq!(pcm.len(), reference.len(), "decoded length");
    let mut max_diff = 0i32;
    for (a, b) in pcm.chunks_exact(2).zip(reference.chunks_exact(2)) {
        let x = i16::from_le_bytes([a[0], a[1]]) as i32;
        let y = i16::from_le_bytes([b[0], b[1]]) as i32;
        max_diff = max_diff.max((x - y).abs());
    }
    assert!(max_diff <= 1, "samples within 1 LSB, got {max_diff}");
}

/// A chained file demuxes as both physical streams back to back: chain 2's
/// packets are exactly what it yields alone, shifted onto the timeline by
/// chain 1's playable duration, announced by one `Segment`.
async fn assert_chain_continues(vector: &ChainedVector, stream: &str, headers_per_chain: usize) {
    let alone1 = demux(&vector.first_bytes, stream).await;
    let alone2 = demux(&vector.second_bytes, stream).await;
    let both = demux(&vector.bytes, stream).await;
    let offset = probe_ns(&vector.first).playable;

    assert_eq!(
        both.segments.len(),
        1,
        "one segment opens the chained physical stream"
    );
    let seg = both.segments[0];
    assert_eq!(
        seg.start, offset,
        "chain 2 starts at chain 1's playable end"
    );
    assert_eq!(seg.base, offset, "running time carries on");
    assert_eq!(seg.time, 0, "stream time restarts for the new chain");

    let headers: Vec<&Vec<u8>> = both
        .frames
        .iter()
        .map(|(p, _)| p)
        .filter(|p| is_header(p))
        .collect();
    assert_eq!(
        headers.len(),
        2 * headers_per_chain,
        "each chain's codec config is forwarded in-band, once"
    );

    let audio = both.audio();
    let expected: Vec<(Vec<u8>, FrameTiming)> = alone1
        .audio()
        .into_iter()
        .chain(alone2.audio().into_iter().map(|(p, mut t)| {
            t.pts_ns += offset;
            t.dts_ns += offset;
            (p, t)
        }))
        .collect();
    assert_eq!(
        audio.len(),
        expected.len(),
        "every packet of both chains came out"
    );
    for (i, ((got, gt), (want, wt))) in audio.iter().zip(expected.iter()).enumerate() {
        assert_eq!(got, want, "packet {i} payload");
        assert_eq!(
            (gt.pts_ns, gt.duration_ns),
            (wt.pts_ns, wt.duration_ns),
            "packet {i} timing"
        );
    }
    // The chained timeline reaches the second chain's full granule: its playable
    // span on top of the first chain's, plus the encoder delay the coded
    // timeline counts and the decoder then discards (zero for Vorbis).
    let (_, last) = audio.last().expect("audio packets");
    let second = probe_ns(&vector.second);
    assert_eq!(
        last.pts_ns + last.duration_ns,
        offset + second.playable + second.padding,
        "the chained timeline ends where both chains do"
    );
}

#[tokio::test]
async fn chained_opus_continues_the_timeline() {
    let Some(vector) = chained_vector("opus-demux", "libopus", 1, 48_000) else {
        eprintln!("skipping: no ffmpeg");
        return;
    };
    assert_chain_continues(&vector, "opus", 1).await;
}

#[tokio::test]
async fn chained_vorbis_continues_the_timeline() {
    let Some(vector) = chained_vector("vorbis-demux", "libvorbis", 2, 44_100) else {
        eprintln!("skipping: no ffmpeg");
        return;
    };
    assert_chain_continues(&vector, "vorbis", 3).await;
}

/// The Ogg-FLAC mapping chains on the same terms: the mapping is per chain, so
/// the second one's `\x7fFLAC` header rebuilds the decoder's STREAMINFO.
#[tokio::test]
async fn chained_ogg_flac_continues_the_timeline() {
    let Some(vector) = chained_vector("flac-demux", "flac", 2, 44_100) else {
        eprintln!("skipping: no ffmpeg");
        return;
    };
    assert_chain_continues(&vector, "flac", 1).await;
}

/// The second chain's own parameters reach the decoder: a stereo chain after a
/// mono one re-announces its caps.
#[tokio::test]
async fn a_chained_parameter_change_re_announces_caps() {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let mono = temp_path("params-1", "opus");
    let stereo = temp_path("params-2", "opus");
    encode(&mono, "libopus", 440, 1, 48_000).expect("mono encode");
    encode(&stereo, "libopus", 880, 2, 48_000).expect("stereo encode");
    let mut bytes = std::fs::read(&mono).unwrap();
    bytes.extend_from_slice(&std::fs::read(&stereo).unwrap());
    let _ = std::fs::remove_file(&mono);
    let _ = std::fs::remove_file(&stereo);

    let out = demux(&bytes, "opus").await;
    assert_eq!(
        out.caps,
        vec![
            Caps::Audio {
                format: AudioFormat::Opus,
                channels: 1,
                sample_rate: 48_000
            },
            Caps::Audio {
                format: AudioFormat::Opus,
                channels: 2,
                sample_rate: 48_000
            }
        ],
        "each chain announces its own channel count"
    );
}

/// One output pad names one codec, so a chain that switches codec has nowhere
/// to go: the demuxer fails loud instead of going quiet for the rest of the
/// file.
#[tokio::test]
async fn a_chain_that_changes_codec_fails_loud() {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let opus = temp_path("cross-1", "opus");
    let vorbis = temp_path("cross-2", "ogg");
    encode(&opus, "libopus", 440, 1, 48_000).expect("opus encode");
    encode(&vorbis, "libvorbis", 880, 1, 44_100).expect("vorbis encode");
    let mut bytes = std::fs::read(&opus).unwrap();
    bytes.extend_from_slice(&std::fs::read(&vorbis).unwrap());
    let _ = std::fs::remove_file(&opus);
    let _ = std::fs::remove_file(&vorbis);

    let mut d = OggDemux::new();
    d.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::Ogg,
    })
    .expect("configure");
    let mut sink = CaptureSink::default();
    let mut err = None;
    for piece in bytes.chunks(997) {
        if let Err(e) = AsyncElement::process(&mut d, chunk(piece), &mut sink).await {
            err = Some(e);
            break;
        }
    }
    assert_eq!(
        err,
        Some(G2gError::CapsMismatch),
        "a Vorbis chain after an Opus one is refused"
    );
    assert!(
        !sink.audio().is_empty(),
        "the first chain played before the refusal"
    );
}

/// The multi-output demuxer routes by slot, so a chained file continues on the
/// same port with a segment of its own.
#[tokio::test]
async fn the_fanout_demuxer_continues_a_chain_on_its_port() {
    let Some(vector) = chained_vector("fanout", "libopus", 1, 48_000) else {
        eprintln!("skipping: no ffmpeg");
        return;
    };
    let mut d = OggDemuxN::new(vec![OggPort::new(0, OggCodec::Opus)]);
    MultiOutputElement::configure_pipeline(
        &mut d,
        &Caps::ByteStream {
            encoding: ByteStreamEncoding::Ogg,
        },
    )
    .expect("configure");
    let mut sink = CaptureSink::default();
    for piece in vector.bytes.chunks(997) {
        MultiOutputElement::process(&mut d, chunk(piece), &mut sink)
            .await
            .expect("demux");
    }

    let offset = probe_ns(&vector.first).playable;
    assert_eq!(sink.segments.len(), 1, "the chain opens a segment");
    assert_eq!(sink.segments[0].start, offset);
    let single = demux(&vector.bytes, "opus").await;
    assert_eq!(
        sink.audio().len(),
        single.audio().len(),
        "the port carries both chains"
    );
}

/// Decoding a chained file runs both parts back to back, sample for sample as
/// ffmpeg's own decode of the same file does, and lasts exactly as long as the
/// two parts decoded on their own. (Neither decoder is torn down at the
/// boundary, so the first frames of the second chain carry over a little
/// decoder state; that is what the ffmpeg oracle does too.)
#[cfg(feature = "opus")]
#[tokio::test]
async fn chained_opus_decodes_to_the_concatenated_pcm() {
    let Some(vector) = chained_vector("opus-decode", "libopus", 1, 48_000) else {
        eprintln!("skipping: no ffmpeg");
        return;
    };
    let reference = ffmpeg_pcm(&vector.chained);
    assert_eq!(
        reference.len(),
        ffmpeg_pcm(&vector.first).len() + ffmpeg_pcm(&vector.second).len(),
        "the chained decode lasts exactly as long as the two parts"
    );

    let out = temp_path("opus-decode-out", "raw");
    let _ = std::fs::remove_file(&out);
    let line = format!(
        "filesrc location={} ! oggdemux ! opusdec ! audioconvert ! \
         audio/x-raw,format=S16LE,rate=48000,channels=1 ! filesink location={}",
        vector.chained.display(),
        out.display()
    );
    assert!(run_line(&line).await > 0, "{line}");
    let pcm = std::fs::read(&out).expect("pcm written");
    let _ = std::fs::remove_file(&out);
    assert_pcm_matches(&pcm, &reference);
}

/// The Vorbis chain likewise, and exactly: the second chain's ident + setup
/// headers rebuild the decoder, so each chain decodes as it does alone.
/// (ffmpeg's own decode of a chained Vorbis file is longer, it re-primes
/// neither the window nor the granule anchor at the boundary.)
#[cfg(feature = "vorbis")]
#[tokio::test]
async fn chained_vorbis_decodes_to_the_concatenated_pcm() {
    let Some(vector) = chained_vector("vorbis-decode", "libvorbis", 2, 44_100) else {
        eprintln!("skipping: no ffmpeg");
        return;
    };
    let mut reference = ffmpeg_pcm(&vector.first);
    reference.extend_from_slice(&ffmpeg_pcm(&vector.second));

    let out = temp_path("vorbis-decode-out", "raw");
    let _ = std::fs::remove_file(&out);
    let line = format!(
        "filesrc location={} ! oggdemux stream=vorbis ! vorbisdec ! audioconvert ! \
         audio/x-raw,format=S16LE,rate=44100,channels=2 ! filesink location={}",
        vector.chained.display(),
        out.display()
    );
    assert!(run_line(&line).await > 0, "{line}");
    let pcm = std::fs::read(&out).expect("pcm written");
    let _ = std::fs::remove_file(&out);
    assert_pcm_matches(&pcm, &reference);
}
