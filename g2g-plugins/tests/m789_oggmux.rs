//! M789: `oggmux`, the mux direction of the Ogg container. Each of the three
//! mappings `oggdemux` speaks (Opus, Vorbis, Ogg-FLAC) survives a g2g demux ->
//! mux -> demux round trip packet for packet, and ffmpeg is the reference peer
//! on the muxed file: ffprobe reports the right codec / channels / rate /
//! duration, and ffmpeg's decode of a remux is bit-identical to its decode of
//! the source (a remux changes framing, never samples).
#![cfg(feature = "std")]

use std::path::PathBuf;
use std::process::Command;

use g2g_core::conformance::{ConformanceDimension, Evidence};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, G2gError, OutputSink, PropValue,
    PushOutcome,
};
use g2g_plugins::conformance::persist;
use g2g_plugins::flacparse::FlacParse;
use g2g_plugins::oggdemux::OggDemux;
use g2g_plugins::oggmux::OggMux;

/// Captures the frames an element pushes, keeping their timing (the muxer reads
/// `duration_ns` to reproduce a source's end-of-stream trim).
#[derive(Default)]
struct CaptureSink {
    frames: Vec<(Vec<u8>, FrameTiming)>,
}

impl CaptureSink {
    fn bytes(&self) -> Vec<u8> {
        self.frames.iter().flat_map(|(b, _)| b.clone()).collect()
    }

    fn payloads(&self) -> Vec<Vec<u8>> {
        self.frames.iter().map(|(b, _)| b.clone()).collect()
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
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
        && Command::new("ffprobe").arg("-version").output().is_ok()
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m789-{}-{name}", std::process::id()))
}

/// Encode an ffmpeg lavfi source to `path` with `codec`, returning its bytes.
fn author(path: &PathBuf, codec: &str, source: &str, channels: u8, rate: u32) -> Vec<u8> {
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "lavfi", "-i", source])
        .args(["-ac", &channels.to_string(), "-ar", &rate.to_string()])
        .args(["-c:a", codec])
        .arg(path)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg authored the {codec} fixture");
    std::fs::read(path).expect("read fixture")
}

/// ffprobe's `codec_name=... / channels=...` lines for the first audio stream.
/// ffprobe emits the entries in its own field order, so they come back keyed.
fn probe(path: &PathBuf) -> Vec<(String, String)> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "a:0"])
        .args([
            "-show_entries",
            "stream=codec_name,channels,sample_rate,duration",
            "-of",
            "default=nw=1",
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
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// One probed field, or a failure naming what ffprobe did report.
fn field<'a>(probed: &'a [(String, String)], key: &str) -> &'a str {
    probed
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
        .unwrap_or_else(|| panic!("ffprobe reported {key}, got {probed:?}"))
}

/// ffmpeg's decode of `path` as raw interleaved 16-bit PCM. Fails the test if
/// ffmpeg reports any decode error, so a container the peer cannot read is
/// caught here rather than silently comparing two empty buffers.
fn decode_pcm(path: &PathBuf) -> Vec<u8> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-f", "s16le", "-c:a", "pcm_s16le", "-"])
        .output()
        .expect("run ffmpeg");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success() && err.is_empty(),
        "ffmpeg decoded {} cleanly: {err}",
        path.display()
    );
    assert!(!out.stdout.is_empty(), "ffmpeg decoded some audio");
    out.stdout
}

/// Demux an Ogg byte stream, returning the frames `oggdemux` emits (the in-band
/// codec headers first, then the audio packets).
async fn oggdemux_frames(ogg: &[u8], stream: &str) -> CaptureSink {
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

/// Mux `frames` (headers then audio, as an upstream demuxer / parser emits them)
/// into an Ogg byte stream.
async fn oggmux_bytes(frames: &CaptureSink, caps: &Caps) -> Vec<u8> {
    let mut m = OggMux::new();
    m.configure_pipeline(caps).expect("configure");
    let mut sink = CaptureSink::default();
    for (data, timing) in &frames.frames {
        m.process(frame(data.clone(), *timing), &mut sink)
            .await
            .expect("mux");
    }
    m.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("mux eos");
    sink.bytes()
}

/// Demux -> mux -> demux one Ogg mapping and check ffmpeg agrees about the
/// result. `codec` is ffprobe's name, `stream` the `oggdemux` selection.
async fn assert_remux_matches(name: &str, encoder: &str, codec: &str, stream: &str, rate: u32) {
    let format = match stream {
        "opus" => AudioFormat::Opus,
        "vorbis" => AudioFormat::Vorbis,
        _ => AudioFormat::Flac,
    };
    let container = if stream == "opus" { "opus" } else { "ogg" };
    let src = temp_path(&format!("src-{name}.{container}"));
    let bytes = author(
        &src,
        encoder,
        &format!("sine=frequency=440:duration=1.0:sample_rate={rate}"),
        2,
        rate,
    );

    let demuxed = oggdemux_frames(&bytes, stream).await;
    assert!(
        demuxed.frames.len() > 2,
        "the source demuxed to headers + audio"
    );
    // Opus always decodes at 48 kHz whatever the file's nominal input rate.
    let granule_rate = if stream == "opus" { 48_000 } else { rate };
    let muxed = oggmux_bytes(&demuxed, &audio_caps(format, 2, granule_rate)).await;
    assert_eq!(&muxed[..4], b"OggS", "an Ogg byte stream");

    // g2g reads its own output back to the same packets.
    let again = oggdemux_frames(&muxed, stream).await;
    assert_eq!(
        again.payloads(),
        demuxed.payloads(),
        "{name}: packets survive the remux byte for byte"
    );

    // Reference peer: ffprobe on the g2g-muxed file.
    let out = temp_path(&format!("out-{name}.{container}"));
    std::fs::write(&out, &muxed).expect("write muxed");
    let probed = probe(&out);
    println!("ffprobe {name}: {probed:?}");
    assert_eq!(field(&probed, "codec_name"), codec, "{name}: ffprobe codec");
    assert_eq!(field(&probed, "channels"), "2", "{name}: ffprobe channels");
    assert_eq!(
        field(&probed, "sample_rate"),
        granule_rate.to_string(),
        "{name}: ffprobe rate"
    );
    // Within one packet of the source's own reported duration.
    let duration: f64 = field(&probed, "duration")
        .parse()
        .expect("ffprobe duration");
    let source_probed = probe(&src);
    let source_duration: f64 = field(&source_probed, "duration")
        .parse()
        .expect("ffprobe source duration");
    assert!(
        (duration - source_duration).abs() < 0.03,
        "{name}: ffprobe duration {duration} matches the source's {source_duration}"
    );

    // A remux changes framing, never samples: ffmpeg's two decodes must match.
    assert_eq!(
        decode_pcm(&out),
        decode_pcm(&src),
        "{name}: ffmpeg decodes the remux to the source's samples"
    );

    persist::record_evidence(
        "oggmux",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec(codec)
            .detail("ffmpeg decodes the g2g-muxed Ogg to the source's samples"),
    )
    .expect("record oracle evidence");
    persist::record_evidence(
        "oggmux",
        &Evidence::new(ConformanceDimension::RoundTrip)
            .codec(codec)
            .detail("oggdemux -> oggmux -> oggdemux is packet-exact"),
    )
    .expect("record round-trip evidence");

    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&out);
}

#[tokio::test]
async fn opus_remux_is_packet_exact_and_ffmpeg_reads_it() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    assert_remux_matches("opus", "libopus", "opus", "opus", 48_000).await;
}

#[tokio::test]
async fn vorbis_remux_is_packet_exact_and_ffmpeg_reads_it() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    assert_remux_matches("vorbis", "libvorbis", "vorbis", "vorbis", 44_100).await;
}

/// Ogg-FLAC comes from the native `.flac` side: `flacparse` frames the byte
/// stream, `oggmux` wraps it in the `\x7fFLAC` mapping.
#[tokio::test]
async fn flac_from_a_native_stream_muxes_into_the_ogg_mapping() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let rate = 44_100u32;
    let src = temp_path("src-flac.flac");
    let bytes = author(
        &src,
        "flac",
        &format!("sine=frequency=440:duration=1.0:sample_rate={rate}"),
        2,
        rate,
    );

    let mut parse = FlacParse::new();
    let caps = audio_caps(AudioFormat::Flac, 2, rate);
    parse
        .configure_pipeline(&caps)
        .expect("configure flacparse");
    let mut parsed = CaptureSink::default();
    for piece in bytes.chunks(1021) {
        parse
            .process(frame(piece.to_vec(), FrameTiming::default()), &mut parsed)
            .await
            .expect("parse");
    }
    parse
        .process(PipelinePacket::Eos, &mut parsed)
        .await
        .expect("parse eos");
    assert!(parsed.frames.len() > 2, "header + FLAC frames");

    let muxed = oggmux_bytes(&parsed, &caps).await;
    let out = temp_path("out-flac.oga");
    std::fs::write(&out, &muxed).expect("write muxed");

    // g2g reads its own Ogg-FLAC back to the same frames (past the in-band
    // header, which the mapping rewrites into its own first packet).
    let again = oggdemux_frames(&muxed, "flac").await;
    assert_eq!(
        again.payloads()[1..],
        parsed.payloads()[1..],
        "FLAC frames survive the mux byte for byte"
    );

    let probed = probe(&out);
    println!("ffprobe flac: {probed:?}");
    assert_eq!(field(&probed, "codec_name"), "flac");
    assert_eq!(field(&probed, "channels"), "2");
    assert_eq!(field(&probed, "sample_rate"), rate.to_string());
    let duration: f64 = field(&probed, "duration")
        .parse()
        .expect("ffprobe duration");
    assert!(
        (duration - 1.0).abs() < 0.05,
        "ffprobe duration {duration} is the source's 1.0 s"
    );
    // FLAC is lossless, so the two decodes are bit-identical.
    assert_eq!(
        decode_pcm(&out),
        decode_pcm(&src),
        "ffmpeg decodes the Ogg-FLAC to the native stream's samples"
    );

    persist::record_evidence(
        "oggmux",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("flac")
            .detail("ffmpeg decodes the g2g-muxed Ogg-FLAC to the native stream's samples"),
    )
    .expect("record oracle evidence");

    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&out);
}

/// The encode direction: g2g's own Opus encoder straight into `oggmux`, with no
/// source container to copy a header from (the `OpusHead` is synthesized).
#[cfg(feature = "opus")]
#[tokio::test]
async fn opusenc_output_muxes_into_a_file_ffmpeg_plays() {
    use g2g_plugins::opusenc::OpusEnc;

    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    // One second of 48 kHz stereo S16LE sine.
    let rate = 48_000u32;
    let pcm: Vec<u8> = (0..rate)
        .flat_map(|n| {
            let v =
                ((n as f32 * 440.0 * core::f32::consts::TAU / rate as f32).sin() * 8000.0) as i16;
            [v.to_le_bytes(), v.to_le_bytes()]
        })
        .flatten()
        .collect();

    let mut enc = OpusEnc::new();
    enc.configure_pipeline(&audio_caps(AudioFormat::PcmS16Le, 2, rate))
        .expect("configure opusenc");
    let mut encoded = CaptureSink::default();
    for piece in pcm.chunks(4096) {
        enc.process(frame(piece.to_vec(), FrameTiming::default()), &mut encoded)
            .await
            .expect("encode");
    }
    enc.process(PipelinePacket::Eos, &mut encoded)
        .await
        .expect("encode eos");
    assert!(!encoded.frames.is_empty(), "opusenc produced packets");

    let muxed = oggmux_bytes(&encoded, &audio_caps(AudioFormat::Opus, 2, rate)).await;
    let out = temp_path("out-opusenc.opus");
    std::fs::write(&out, &muxed).expect("write muxed");

    let probed = probe(&out);
    println!("ffprobe opusenc: {probed:?}");
    assert_eq!(field(&probed, "codec_name"), "opus");
    assert_eq!(field(&probed, "channels"), "2");
    assert_eq!(field(&probed, "sample_rate"), "48000");
    let duration: f64 = field(&probed, "duration")
        .parse()
        .expect("ffprobe duration");
    assert!(
        (duration - 1.0).abs() < 0.03,
        "ffprobe duration {duration} is within a frame of the encoded second"
    );
    // ffmpeg decodes it without complaint, the point of a muxer nobody else wrote.
    let decoded = decode_pcm(&out);
    assert!(
        decoded.len() > (rate as usize * 4) / 2,
        "ffmpeg decoded a full second, got {} bytes",
        decoded.len()
    );

    persist::record_evidence(
        "oggmux",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("opus")
            .detail("ffmpeg decodes g2g-encoded Opus muxed by oggmux"),
    )
    .expect("record oracle evidence");

    let _ = std::fs::remove_file(&out);
}
