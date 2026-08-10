//! M858: writing chained Ogg. `oggmux` treats a `Segment` arriving after audio
//! as a chain boundary: the logical bitstream in progress closes on an
//! end-of-stream page and the next link opens on a fresh serial. That is exactly
//! the packet `oggdemux` emits at a chain boundary (M827), so a chained file
//! survives a demux -> mux round trip.
//!
//! Reference peer: ffmpeg authors the vector (two independently encoded files
//! concatenated, the canonical chained form), and on the g2g-written remux
//! ffprobe must find both links' packets and ffmpeg must decode it to the
//! samples it decodes the original chained file to.
#![cfg(feature = "std")]

use std::path::{Path, PathBuf};
use std::process::Command;

use g2g_core::conformance::{ConformanceDimension, Evidence};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, G2gError, OutputSink, PropValue,
    PushOutcome, Segment,
};
use g2g_plugins::conformance::persist;
use g2g_plugins::oggdemux::OggDemux;
use g2g_plugins::oggmux::OggMux;

/// One thing an element pushed, in order: the segments matter as much as the
/// frames here, since a segment is what opens a chain link.
#[derive(Clone, Debug, PartialEq)]
enum Event {
    Frame(Vec<u8>, FrameTiming),
    Segment(Segment),
}

#[derive(Default)]
struct CaptureSink {
    events: Vec<Event>,
}

impl CaptureSink {
    fn bytes(&self) -> Vec<u8> {
        self.events
            .iter()
            .filter_map(|e| match e {
                Event::Frame(b, _) => Some(b.clone()),
                Event::Segment(_) => None,
            })
            .collect::<Vec<_>>()
            .concat()
    }

    fn payloads(&self) -> Vec<Vec<u8>> {
        self.events
            .iter()
            .filter_map(|e| match e {
                Event::Frame(b, _) => Some(b.clone()),
                Event::Segment(_) => None,
            })
            .collect()
    }

    fn segments(&self) -> Vec<Segment> {
        self.events
            .iter()
            .filter_map(|e| match e {
                Event::Segment(s) => Some(*s),
                Event::Frame(..) => None,
            })
            .collect()
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
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.events.push(Event::Frame(s.to_vec(), f.timing));
                    }
                }
                PipelinePacket::Segment(s) => self.events.push(Event::Segment(s)),
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn frame(data: Vec<u8>, timing: FrameTiming) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        timing,
        0,
    ))
}

fn ogg_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::Ogg,
    }
}

fn audio_caps(format: AudioFormat, channels: u8, sample_rate: u32) -> Caps {
    Caps::Audio {
        format,
        channels,
        sample_rate,
    }
}

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
        && Command::new("ffprobe").arg("-version").output().is_ok()
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m858-{}-{name}", std::process::id()))
}

/// Encode one tone to `path` with ffmpeg.
fn encode(path: &Path, codec: &str, freq: u32, channels: u8, rate: u32) {
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "lavfi", "-i"])
        .arg(format!(
            "sine=frequency={freq}:duration=1:sample_rate={rate}"
        ))
        .args(["-ac", &channels.to_string(), "-ar", &rate.to_string()])
        .args(["-c:a", codec])
        .arg(path)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg authored the {codec} fixture");
}

/// ffprobe's packet count for the file's audio stream. It walks every chain, so
/// a chained file counts both links.
fn packet_count(path: &Path) -> u32 {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "a:0", "-count_packets"])
        .args(["-show_entries", "stream=nb_read_packets", "-of", "csv=p=0"])
        .arg(path)
        .output()
        .expect("run ffprobe");
    assert!(
        out.status.success(),
        "ffprobe read {}: {}",
        path.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse()
        .expect("ffprobe packet count")
}

/// ffmpeg's decode of `path` to interleaved 16-bit PCM. A chained file restarts
/// its timestamps at every link, which the raw output muxer warns about on
/// ffmpeg's own chained files too, so that one line is tolerated and nothing
/// else: a container the peer cannot read still fails the test.
fn decode_pcm(path: &Path) -> Vec<u8> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-f", "s16le", "-c:a", "pcm_s16le", "-"])
        .output()
        .expect("run ffmpeg");
    let err = String::from_utf8_lossy(&out.stderr);
    let complaints: Vec<&str> = err
        .lines()
        .filter(|l| !l.contains("non monotonically increasing dts"))
        .collect();
    assert!(
        out.status.success() && complaints.is_empty(),
        "ffmpeg decoded {} cleanly: {err}",
        path.display()
    );
    assert!(!out.stdout.is_empty(), "ffmpeg decoded some audio");
    out.stdout
}

/// Demux an Ogg byte stream, keeping the frames and the chain segments in the
/// order `oggdemux` pushed them.
async fn oggdemux_events(ogg: &[u8], stream: &str) -> CaptureSink {
    let mut d = OggDemux::new();
    d.set_property("stream", PropValue::Str(stream.into()))
        .expect("stream property");
    d.configure_pipeline(&ogg_caps()).expect("configure");
    let mut sink = CaptureSink::default();
    for piece in ogg.chunks(1021) {
        d.process(frame(piece.to_vec(), FrameTiming::default()), &mut sink)
            .await
            .expect("demux");
    }
    d.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("demux eos");
    sink
}

/// Replay what the demuxer emitted, in order, into `oggmux`: the chain segments
/// travel with the packets, so the muxer sees the boundary where the source had
/// one.
async fn oggmux_bytes(events: &CaptureSink, caps: &Caps) -> Vec<u8> {
    let mut m = OggMux::new();
    m.configure_pipeline(caps).expect("configure");
    let mut sink = CaptureSink::default();
    for event in &events.events {
        let packet = match event {
            Event::Frame(data, timing) => frame(data.clone(), *timing),
            Event::Segment(seg) => PipelinePacket::Segment(*seg),
        };
        m.process(packet, &mut sink).await.expect("mux");
    }
    m.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("mux eos");
    sink.bytes()
}

/// Two 1 s tones concatenated (the canonical chained file), then a g2g demux ->
/// mux round trip over it, checked against ffmpeg.
async fn assert_chained_remux(
    name: &str,
    encoder: &str,
    stream: &str,
    format: AudioFormat,
    rate: u32,
) {
    let ext = if stream == "opus" { "opus" } else { "ogg" };
    let first = temp_path(&format!("{name}-1.{ext}"));
    let second = temp_path(&format!("{name}-2.{ext}"));
    let chained = temp_path(&format!("{name}-chained.{ext}"));
    let remuxed = temp_path(&format!("{name}-remuxed.{ext}"));
    encode(&first, encoder, 440, 2, rate);
    encode(&second, encoder, 880, 2, rate);
    let mut source = std::fs::read(&first).expect("read link 1");
    source.extend_from_slice(&std::fs::read(&second).expect("read link 2"));
    std::fs::write(&chained, &source).expect("write chained");

    let demuxed = oggdemux_events(&source, stream).await;
    assert_eq!(
        demuxed.segments().len(),
        1,
        "{name}: the source's chain boundary reached the muxer as a segment"
    );

    // Opus always decodes at 48 kHz whatever the file's nominal input rate.
    let granule_rate = if stream == "opus" { 48_000 } else { rate };
    let muxed = oggmux_bytes(&demuxed, &audio_caps(format, 2, granule_rate)).await;
    std::fs::write(&remuxed, &muxed).expect("write remux");

    // g2g reads its own chained output back: both links, packet for packet, with
    // the boundary announced again.
    let again = oggdemux_events(&muxed, stream).await;
    assert_eq!(
        again.segments().len(),
        1,
        "{name}: the remux is chained, not one long link"
    );
    assert_eq!(
        again.payloads(),
        demuxed.payloads(),
        "{name}: both links survive the remux packet for packet"
    );

    // Reference peer. ffprobe walks the chains, so its packet count covers both
    // links of the g2g-written file.
    let (in_first, in_second, in_remux) = (
        packet_count(&first),
        packet_count(&second),
        packet_count(&remuxed),
    );
    println!("ffprobe {name} packets: {in_first} + {in_second} in, {in_remux} out");
    assert_eq!(
        in_remux,
        in_first + in_second,
        "{name}: ffprobe finds both links' packets in the g2g-written file"
    );
    // A remux changes framing, never samples.
    assert_eq!(
        decode_pcm(&remuxed),
        decode_pcm(&chained),
        "{name}: ffmpeg decodes the chained remux to the source's samples"
    );

    persist::record_evidence(
        "oggmux",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec(stream)
            .detail("ffmpeg decodes a g2g-written chained Ogg to the source's samples"),
    )
    .expect("record oracle evidence");
    persist::record_evidence(
        "oggmux",
        &Evidence::new(ConformanceDimension::RoundTrip)
            .codec(stream)
            .detail(
                "a chained Ogg survives oggdemux -> oggmux -> oggdemux, chain boundary included",
            ),
    )
    .expect("record round-trip evidence");

    for p in [&first, &second, &chained, &remuxed] {
        let _ = std::fs::remove_file(p);
    }
}

#[tokio::test]
async fn chained_opus_is_written_and_ffmpeg_reads_both_links() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    assert_chained_remux("opus", "libopus", "opus", AudioFormat::Opus, 48_000).await;
}

#[tokio::test]
async fn chained_vorbis_is_written_and_ffmpeg_reads_both_links() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    assert_chained_remux("vorbis", "libvorbis", "vorbis", AudioFormat::Vorbis, 44_100).await;
}
