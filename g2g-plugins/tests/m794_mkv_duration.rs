//! M794: the Matroska `Info` `Duration`, so a g2g-written file reports its
//! length instead of `N/A`.
//!
//! Only the two-pass (`seekable`) mode can carry one: the total is not known
//! until EOS, and a streaming caller has already emitted its header by then. So
//! the mode reserves an 8-byte float placeholder in `Info` and patches it at
//! finalize, next to the `SeekHead` Cues position it already patches (M770).
//!
//! The value is the presentation duration in `TimestampScale` units: the highest
//! block end across tracks, each block's timestamp plus its own duration, both
//! rounded to the container's millisecond tick. That is ffmpeg's arithmetic, and
//! on a remux of an ffmpeg-authored file g2g lands on the same number.
//!
//! Measured baselines (ffmpeg `n8.1.2`, a 1.0 s 48 kHz stereo libopus stream and
//! a 1.0 s 25 fps H.264 + Opus pair):
//!
//! * ffmpeg's `Duration` is `1008.0` for both, which ffprobe reports as the
//!   container's `format=duration=1.008000`. Per-stream `duration` stays `N/A`
//!   in Matroska, so the container field is the one to read.
//! * 1008 = the last Opus block's timestamp (1001 ms) plus its `BlockDuration`
//!   (7 ms). Both are rounded: the real tail is 6.5 ms, and ffmpeg's own block
//!   sits 1 ms late because it rounds the pre-roll twice. The millisecond grid
//!   is the container's `TimestampScale`, not a g2g approximation, so matching
//!   ffmpeg means rounding the same way rather than writing 1006.5.
#![cfg(feature = "std")]

use std::path::PathBuf;
use std::process::Command;

use g2g_core::conformance::{ConformanceDimension, Evidence};
use g2g_core::fanout::MultiOutputSink;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AudioFormat, ByteStreamEncoding, Caps, Dim, G2gError, MultiInputElement, MultiOutputElement,
    OutputSink, PushOutcome, Rate, VideoCodec,
};
use g2g_plugins::conformance::persist;
use g2g_plugins::mkvdemux::{MkvDemuxN, MkvStream};
use g2g_plugins::mkvmuxn::MkvMuxN;

#[derive(Default)]
struct CaptureSink {
    frames: Vec<(Vec<u8>, FrameTiming)>,
}

impl CaptureSink {
    fn bytes(&self) -> Vec<u8> {
        self.frames.iter().flat_map(|(b, _)| b.clone()).collect()
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
                    self.frames.push((s.to_vec(), f.timing));
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Records each port's frames, so an A/V demux feeds an A/V mux.
#[derive(Default)]
struct PortCapture {
    ports: Vec<PortFrames>,
}

impl MultiOutputSink for PortCapture {
    fn poll_push_to(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        port: usize,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.ports[port].push((s.to_vec(), f.timing));
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }

    fn port_count(&self) -> usize {
        self.ports.len()
    }
}

/// One elementary stream's frames, as a demuxer port hands them over.
type PortFrames = Vec<(Vec<u8>, FrameTiming)>;

fn frame(data: Vec<u8>, timing: FrameTiming) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        timing,
        0,
    ))
}

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
        && Command::new("ffprobe").arg("-version").output().is_ok()
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m794-{}-{name}", std::process::id()))
}

fn opus_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::Opus,
        channels: 2,
        sample_rate: 48_000,
    }
}

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(25 << 16),
    }
}

/// ffmpeg-authored fixtures: a 1 s Opus-only Matroska (stream-copied out of Ogg,
/// so the Opus timing is the encoder's own) and a 1 s H.264 + Opus one.
fn author_opus_mkv(path: &PathBuf) -> Vec<u8> {
    // Derive the intermediate from the destination: tests run concurrently in
    // one process, and a shared intermediate name races (one test deletes it
    // while another's ffmpeg reads it).
    let ogg = path.with_extension("author.opus");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "lavfi", "-i"])
        .arg("sine=frequency=440:duration=1.0:sample_rate=48000")
        .args(["-ac", "2", "-c:a", "libopus"])
        .arg(&ogg)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg authored the Ogg source");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&ogg)
        .args(["-c", "copy"])
        .arg(path)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg copied the Opus into Matroska");
    let _ = std::fs::remove_file(&ogg);
    std::fs::read(path).expect("read fixture")
}

fn author_av_mkv(path: &PathBuf) -> Vec<u8> {
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "lavfi", "-i"])
        .arg("testsrc=size=320x240:rate=25:duration=1")
        .args(["-f", "lavfi", "-i"])
        .arg("sine=frequency=440:duration=1:sample_rate=48000")
        .args([
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "libopus",
            "-ac",
            "2",
        ])
        .arg(path)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg authored the A/V fixture");
    std::fs::read(path).expect("read fixture")
}

/// The container duration ffprobe reports, or `None` for `N/A`. Matroska carries
/// it on the Segment, not the streams, so this is the `format` field.
fn probed_duration(path: &PathBuf) -> Option<f64> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
        ])
        .arg(path)
        .output()
        .expect("run ffprobe");
    assert!(
        out.status.success(),
        "ffprobe read {}: {}",
        path.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    println!("ffprobe {} format=duration -> {text}", path.display());
    text.parse().ok()
}

/// The `Info` `Duration` element's own value (TimestampScale units), read out of
/// the file's header region so the test judges the bytes, not just ffprobe.
fn duration_element(file: &[u8]) -> Option<f64> {
    let header_end = file
        .windows(4)
        .position(|w| w == [0x1F, 0x43, 0xB6, 0x75])
        .unwrap_or(file.len());
    let at = file[..header_end]
        .windows(2)
        .position(|w| w == [0x44, 0x89])?;
    let size = usize::from(file[at + 2] & 0x7F);
    let body = &file[at + 3..at + 3 + size];
    Some(match size {
        4 => f64::from(f32::from_be_bytes(body.try_into().ok()?)),
        8 => f64::from_be_bytes(body.try_into().ok()?),
        _ => return None,
    })
}

/// Demux a Matroska file onto `ports`, returning each port's frames.
async fn demux_ports(file: &[u8], ports: Vec<MkvStream>) -> Vec<PortFrames> {
    let count = ports.len();
    let mut demux = MkvDemuxN::new(ports);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        })
        .expect("configure mkvdemux");
    let mut tap = PortCapture {
        ports: vec![Vec::new(); count],
    };
    demux
        .process(frame(file.to_vec(), FrameTiming::default()), &mut tap)
        .await
        .expect("demux");
    demux
        .process(PipelinePacket::Eos, &mut tap)
        .await
        .expect("demux eos");
    tap.ports
}

/// Mux per-input frame lists into one Matroska file, in the two-pass mode so the
/// `Duration` and the `SeekHead` are filled in at EOS.
async fn mux_seekable(inputs: &[(Caps, PortFrames)]) -> Vec<u8> {
    let mut mux = MkvMuxN::new(inputs.len()).with_seekable(true);
    for (i, (caps, _)) in inputs.iter().enumerate() {
        mux.configure_pipeline(i, caps).expect("configure mkvmux");
    }
    let mut sink = CaptureSink::default();
    for (i, (_, frames)) in inputs.iter().enumerate() {
        for (data, timing) in frames {
            mux.process(i, frame(data.clone(), *timing), &mut sink)
                .await
                .expect("mux");
        }
    }
    for i in 0..inputs.len() {
        mux.process(i, PipelinePacket::Eos, &mut sink)
            .await
            .expect("mux eos");
    }
    sink.bytes()
}

/// Opus: g2g's remux of an ffmpeg file reports the duration ffmpeg's own does.
#[tokio::test]
async fn opus_remux_reports_the_same_duration_as_ffmpeg() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let src = temp_path("src-opus.mkv");
    let file = author_opus_mkv(&src);
    let theirs = probed_duration(&src).expect("ffmpeg's file reports a duration");

    let ports = demux_ports(&file, vec![MkvStream::Opus]).await;
    let muxed = mux_seekable(&[(opus_caps(), ports[0].clone())]).await;
    let out = temp_path("out-opus.mkv");
    std::fs::write(&out, &muxed).expect("write muxed");
    let ours = probed_duration(&out).expect("the g2g file reports a duration, not N/A");

    // Same blocks in, same block ends out, so the same number: 1.008 s.
    assert_eq!(
        ours, theirs,
        "g2g's remux reports exactly the duration ffmpeg's file does"
    );
    assert_eq!(
        duration_element(&muxed),
        Some(theirs * 1000.0),
        "the Info Duration element itself holds it, in TimestampScale ticks"
    );

    persist::record_evidence(
        "matroskamux",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("opus")
            .detail("ffprobe reads the same duration from a g2g remux as from ffmpeg's own"),
    )
    .expect("record oracle evidence");

    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&out);
}

/// A/V: the duration is the highest block end across both tracks, so a video
/// track that outlasts the audio (or the reverse) still reports the whole file.
#[tokio::test]
async fn av_remux_reports_the_same_duration_as_ffmpeg() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let src = temp_path("src-av.mkv");
    let file = author_av_mkv(&src);
    let theirs = probed_duration(&src).expect("ffmpeg's file reports a duration");

    let ports = demux_ports(&file, vec![MkvStream::H264, MkvStream::Opus]).await;
    assert!(
        !ports[0].is_empty() && !ports[1].is_empty(),
        "both tracks demuxed: {} video, {} audio frames",
        ports[0].len(),
        ports[1].len()
    );
    let muxed = mux_seekable(&[
        (h264_caps(), ports[0].clone()),
        (opus_caps(), ports[1].clone()),
    ])
    .await;
    let out = temp_path("out-av.mkv");
    std::fs::write(&out, &muxed).expect("write muxed");
    let ours = probed_duration(&out).expect("the g2g file reports a duration, not N/A");

    assert_eq!(
        ours, theirs,
        "the A/V remux reports exactly the duration ffmpeg's file does"
    );

    persist::record_evidence(
        "matroskamux",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .detail("ffprobe reads the same duration from a g2g A/V remux as from ffmpeg's own"),
    )
    .expect("record oracle evidence");

    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&out);
}

/// A source whose timestamps are g2g's own, not inherited from an ffmpeg file:
/// Ogg in, Matroska out. The duration is then this file's own last block end,
/// one tick under ffmpeg's remux of the same stream, because ffmpeg places every
/// block 1 ms late (it rounds the negative pre-roll start to a whole millisecond
/// before offsetting). Asserted rather than hidden: the tick is ffmpeg's
/// rounding, and reproducing it would mean putting our blocks off the 20 ms grid.
#[tokio::test]
async fn an_ogg_source_reports_its_own_exact_last_block_end() {
    use g2g_core::{AsyncElement, PropValue};
    use g2g_plugins::oggdemux::OggDemux;

    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let ogg = temp_path("src.opus");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "lavfi", "-i"])
        .arg("sine=frequency=440:duration=1.0:sample_rate=48000")
        .args(["-ac", "2", "-c:a", "libopus"])
        .arg(&ogg)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg authored the Ogg source");
    let ogg_bytes = std::fs::read(&ogg).expect("read fixture");

    let mut demux = OggDemux::new();
    demux
        .set_property("stream", PropValue::Str("opus".into()))
        .expect("stream property");
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Ogg,
        })
        .expect("configure oggdemux");
    let mut sink = CaptureSink::default();
    for piece in ogg_bytes.chunks(1021) {
        demux
            .process(frame(piece.to_vec(), FrameTiming::default()), &mut sink)
            .await
            .expect("demux");
    }
    demux
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("demux eos");

    let muxed = mux_seekable(&[(opus_caps(), sink.frames.clone())]).await;
    let out = temp_path("out-from-ogg.mkv");
    std::fs::write(&out, &muxed).expect("write muxed");
    let ours = probed_duration(&out).expect("the g2g file reports a duration");

    // The last packet sits at 1000 ms and lasts 6.5, which the millisecond grid
    // rounds to 7: 1.007 s.
    assert_eq!(duration_element(&muxed), Some(1007.0));
    assert!(
        (ours - 1.007).abs() < 1e-9,
        "the duration is this file's own last block end, got {ours}"
    );

    let reference = temp_path("ref-from-ogg.mkv");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&ogg)
        .args(["-c", "copy"])
        .arg(&reference)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg remuxed the same stream");
    let theirs = probed_duration(&reference).expect("ffmpeg's remux reports a duration");
    assert!(
        (ours - theirs).abs() <= 0.001 + 1e-9,
        "within one TimestampScale tick of ffmpeg: {ours} vs {theirs}"
    );

    let _ = std::fs::remove_file(&ogg);
    let _ = std::fs::remove_file(&out);
    let _ = std::fs::remove_file(&reference);
}

/// The streaming mode is unchanged: an unknown-size live stream has no length to
/// declare, so no `Duration` is written and ffprobe reports `N/A`. The two-pass
/// mode is what buys the number.
#[tokio::test]
async fn the_streaming_mode_still_writes_no_duration() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let src = temp_path("src-stream.mkv");
    let file = author_opus_mkv(&src);
    let ports = demux_ports(&file, vec![MkvStream::Opus]).await;

    let mut mux = MkvMuxN::new(1);
    mux.configure_pipeline(0, &opus_caps())
        .expect("configure mkvmux");
    let mut sink = CaptureSink::default();
    for (data, timing) in &ports[0] {
        mux.process(0, frame(data.clone(), *timing), &mut sink)
            .await
            .expect("mux");
    }
    mux.process(0, PipelinePacket::Eos, &mut sink)
        .await
        .expect("mux eos");
    let streamed = sink.bytes();

    assert_eq!(
        duration_element(&streamed),
        None,
        "the streaming header reserves no Duration"
    );
    let out = temp_path("out-stream.mkv");
    std::fs::write(&out, &streamed).expect("write muxed");
    assert_eq!(
        probed_duration(&out),
        None,
        "so ffprobe reports N/A, as it did before this milestone"
    );

    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&out);
}
