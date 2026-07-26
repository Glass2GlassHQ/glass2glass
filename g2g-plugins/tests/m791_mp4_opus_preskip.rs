//! M791: Opus pre-skip survives MP4 in both directions, and the presentation
//! timeline it defines is exact.
//!
//! Read side: the `dOps` OpusSpecificBox is parsed whole and rebuilt into an
//! RFC 7845 `OpusHead` forwarded in-band ahead of the audio, the convention
//! `OggDemux` already uses, so `OpusDec` trims the encoder delay and a remux
//! keeps the source's value. Write side: an in-band `OpusHead` is consumed as
//! codec config (never written as a sample) and its real pre-skip / output gain
//! / channel mapping go into the `dOps`; a freshly encoded stream carries none,
//! so the `dOps` falls back to libopus' 312-sample lookahead. The Opus `trak`
//! also gets the `edts`/`elst` the Opus-in-ISOBMFF binding requires, whose
//! `media_time` is the pre-skip.
//!
//! ffmpeg is the reference peer throughout: it authors the fixtures, ffprobe
//! judges the timeline, and its own libopus decode is the alignment oracle for
//! g2g's.
//!
//! Known limit, checked here rather than papered over: `Mp4MuxN` writes a
//! **fragmented** MP4, and ffmpeg's mov demuxer derives a fragmented file's
//! duration by summing sample durations, applying an edit list only as a
//! timestamp shift. So ffprobe reports the media duration (pre-roll included)
//! with `start_time` pushed negative by the pre-skip, and the presentation
//! *end* (`start_time + duration`) is what lands exactly on the source length.
//! ffmpeg's own fragmented output behaves the same way. An exactly-reported
//! `duration=1.000000` needs a real sample table, i.e. a non-fragmented muxer
//! mode, which this milestone does not add.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::path::PathBuf;
use std::process::Command;

use g2g_core::conformance::{ConformanceDimension, Evidence};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, G2gError, MultiInputElement,
    MultiOutputElement, MultiOutputSink, OutputSink, PropValue, PushOutcome,
};
use g2g_plugins::conformance::persist;
use g2g_plugins::mp4demuxn::{forwardable_streams, Mp4DemuxN, Mp4Port};
use g2g_plugins::mp4muxn::Mp4MuxN;
use g2g_plugins::oggdemux::OggDemux;

/// The 48 kHz lookahead ffmpeg and g2g write into a synthesized Opus header.
const ENCODER_PRE_SKIP: u16 = 312;

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
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.frames.push((s.to_vec(), f.timing));
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

#[derive(Default)]
struct PortCapture {
    frames: Vec<(Vec<u8>, FrameTiming)>,
}

impl MultiOutputSink for PortCapture {
    fn push_to<'a>(
        &'a mut self,
        _port: usize,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.frames.push((s.to_vec(), f.timing));
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }

    fn port_count(&self) -> usize {
        1
    }
}

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
    std::env::temp_dir().join(format!("g2g-m791-{}-{name}", std::process::id()))
}

/// Encode a 1 s 48 kHz stereo sine to `path` with libopus, returning its bytes.
/// The container is whatever the extension selects (`.opus` = Ogg, `.mp4`).
fn author(path: &PathBuf) -> Vec<u8> {
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "lavfi", "-i"])
        .arg("sine=frequency=440:duration=1.0:sample_rate=48000")
        .args(["-ac", "2", "-c:a", "libopus"])
        .arg(path)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg authored {}", path.display());
    std::fs::read(path).expect("read fixture")
}

/// ffprobe's `key=value` lines for the first audio stream plus the container.
fn probe(path: &PathBuf) -> Vec<(String, String)> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "a:0"])
        .args([
            "-show_entries",
            "stream=codec_name,channels,sample_rate,duration,start_time",
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

fn field<'a>(probed: &'a [(String, String)], key: &str) -> &'a str {
    probed
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
        .unwrap_or_else(|| panic!("ffprobe reported {key}, got {probed:?}"))
}

fn probed_f64(probed: &[(String, String)], key: &str) -> f64 {
    field(probed, key)
        .parse()
        .unwrap_or_else(|_| panic!("ffprobe {key} is a number, got {:?}", field(probed, key)))
}

/// ffmpeg's decode of `path` as raw interleaved 16-bit PCM. Fails the test if
/// ffmpeg reports any decode error, so a container the peer cannot read is
/// caught here rather than silently comparing two empty buffers.
fn decode_pcm(path: &PathBuf) -> Vec<u8> {
    decode_raw(path, &[], "s16le", "pcm_s16le")
}

/// The same decode as raw interleaved 32-bit float, through **libopus** rather
/// than ffmpeg's own Opus decoder, so it is the closest comparison to g2g's
/// (also libopus). Still not bit-identical: two Opus decoders need only agree
/// within the spec's tolerance, and the S16 path adds ffmpeg's dither on top.
#[cfg(feature = "opus")]
fn decode_f32_libopus(path: &PathBuf) -> Vec<u8> {
    decode_raw(path, &["-c:a", "libopus"], "f32le", "pcm_f32le")
}

fn decode_raw(path: &PathBuf, decoder: &[&str], format: &str, codec: &str) -> Vec<u8> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error"])
        .args(decoder)
        .arg("-i")
        .arg(path)
        .args(["-f", format, "-c:a", codec, "-"])
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

/// The payload of the first `dOps` box in an ISO-BMFF file (the box walk is a
/// plain scan: the 4cc is unambiguous enough for a test oracle).
fn dops_payload(file: &[u8]) -> Vec<u8> {
    let at = file
        .windows(4)
        .position(|w| w == b"dOps")
        .expect("the file carries a dOps");
    let size = u32::from_be_bytes(file[at - 4..at].try_into().unwrap()) as usize;
    assert!(size >= 8 + 11, "a dOps holds at least its fixed fields");
    file[at + 4..at - 4 + size].to_vec()
}

/// The `PreSkip` field of a `dOps` payload (big-endian, offset 2).
fn dops_pre_skip(payload: &[u8]) -> u16 {
    u16::from_be_bytes([payload[2], payload[3]])
}

/// The single `elst` entry of an ISO-BMFF file: (segment_duration, media_time).
fn elst_entry(file: &[u8]) -> (u32, i32) {
    let at = file
        .windows(4)
        .position(|w| w == b"elst")
        .expect("the file carries an elst");
    let p = &file[at + 8..]; // past the 4cc + version/flags
    assert_eq!(
        u32::from_be_bytes(p[0..4].try_into().unwrap()),
        1,
        "one edit list entry"
    );
    (
        u32::from_be_bytes(p[4..8].try_into().unwrap()),
        i32::from_be_bytes(p[8..12].try_into().unwrap()),
    )
}

/// The pre-skip an `OpusHead` declares (little-endian, offset 10).
fn head_pre_skip(head: &[u8]) -> u16 {
    assert!(head.starts_with(b"OpusHead"), "an OpusHead");
    u16::from_le_bytes([head[10], head[11]])
}

fn opus_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::Opus,
        channels: 2,
        sample_rate: 48_000,
    }
}

#[cfg(feature = "opus")]
fn f32_samples(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Decode demuxed frames (in-band config first, then audio with the container's
/// per-packet durations) through `OpusDec` to interleaved F32LE.
#[cfg(feature = "opus")]
async fn decode_frames(frames: &[(Vec<u8>, FrameTiming)]) -> Vec<u8> {
    use g2g_plugins::opusdec::OpusDec;

    let mut dec = OpusDec::new();
    dec.configure_pipeline(&opus_caps()).expect("configure");
    dec.configure_output(&Caps::Audio {
        format: AudioFormat::PcmF32Le,
        channels: 2,
        sample_rate: 48_000,
    })
    .expect("float output accepted");
    let mut pcm = CaptureSink::default();
    for (data, timing) in frames {
        dec.process(frame(data.clone(), *timing), &mut pcm)
            .await
            .expect("decode");
    }
    dec.process(PipelinePacket::Eos, &mut pcm)
        .await
        .expect("decode eos");
    pcm.bytes()
}

/// Demux an Ogg Opus byte stream into the frames `OggDemux` emits (the in-band
/// `OpusHead` first, then the audio packets with their trimmed durations).
async fn oggdemux_frames(ogg: &[u8]) -> CaptureSink {
    let mut d = OggDemux::new();
    d.set_property("stream", PropValue::Str("opus".into()))
        .expect("stream property");
    d.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::Ogg,
    })
    .expect("configure oggdemux");
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

/// Demux an Opus-in-MP4 file through `Mp4DemuxN`, returning its frames (the
/// in-band `OpusHead` first, then the audio packets).
async fn mp4demux_frames(file: &[u8]) -> Vec<(Vec<u8>, FrameTiming)> {
    let streams = forwardable_streams(file);
    assert_eq!(streams.len(), 1, "one Opus track discovered");
    let ports = vec![Mp4Port {
        track_id: streams[0].track_id,
        caps: streams[0].caps.clone(),
    }];
    let mut demux = Mp4DemuxN::new(ports);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff,
        })
        .expect("configure mp4demux");
    let mut tap = PortCapture::default();
    demux
        .process(frame(file.to_vec(), FrameTiming::default()), &mut tap)
        .await
        .expect("demux");
    demux
        .process(PipelinePacket::Eos, &mut tap)
        .await
        .expect("demux eos");
    tap.frames
}

/// Mux `frames` (in-band config then audio, as a demuxer emits them) into a
/// fragmented MP4.
async fn mp4mux_bytes(frames: &[(Vec<u8>, FrameTiming)]) -> Vec<u8> {
    let mut mux = Mp4MuxN::new(1);
    mux.configure_pipeline(0, &opus_caps())
        .expect("configure mp4mux");
    let mut sink = CaptureSink::default();
    for (data, timing) in frames {
        mux.process(0, frame(data.clone(), *timing), &mut sink)
            .await
            .expect("mux");
    }
    mux.process(0, PipelinePacket::Eos, &mut sink)
        .await
        .expect("mux eos");
    sink.bytes()
}

/// Ogg -> g2g demux -> `mp4muxn`: the source's real pre-skip reaches the `dOps`
/// and the edit list, and ffmpeg reads the result back to the source's samples.
#[tokio::test]
async fn ogg_remuxed_into_mp4_keeps_the_sources_pre_skip_and_length() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let src = temp_path("src.opus");
    let bytes = author(&src);

    let demuxed = oggdemux_frames(&bytes).await;
    let source_pre_skip = head_pre_skip(&demuxed.frames[0].0);
    assert_eq!(
        source_pre_skip, ENCODER_PRE_SKIP,
        "ffmpeg's libopus encoder declares its 312-sample lookahead"
    );

    let muxed = mp4mux_bytes(&demuxed.frames).await;
    let out = temp_path("out-from-ogg.mp4");
    std::fs::write(&out, &muxed).expect("write muxed");

    // The source's OpusHead became the dOps, not a sample.
    assert_eq!(
        dops_pre_skip(&dops_payload(&muxed)),
        source_pre_skip,
        "the dOps carries the source's pre-skip"
    );
    assert!(
        !muxed.windows(8).any(|w| w == b"OpusHead"),
        "the OpusHead is codec config, never written into the mdat"
    );
    // The edit list trims the pre-roll off the presentation timeline. The
    // segment duration is 0 ("to the end of the media") because a fragmented
    // moov is written before the total length is known.
    assert_eq!(
        elst_entry(&muxed),
        (0, i32::from(source_pre_skip)),
        "the elst skips exactly the pre-skip"
    );

    let probed = probe(&out);
    println!("ffprobe out-from-ogg.mp4: {probed:?}");
    assert_eq!(field(&probed, "codec_name"), "opus");
    assert_eq!(field(&probed, "channels"), "2");
    assert_eq!(field(&probed, "sample_rate"), "48000");

    // The presentation runs from -pre_skip to exactly the source's 1.0 s. On a
    // fragmented file ffmpeg reports the media span and pushes start_time back
    // by the edit, so the *end* is the exact number (see the module note).
    let start = probed_f64(&probed, "start_time");
    let duration = probed_f64(&probed, "duration");
    let expected_start = -f64::from(source_pre_skip) / 48_000.0;
    assert!(
        (start - expected_start).abs() < 1e-6,
        "start_time {start} is the pre-skip {expected_start}"
    );
    assert!(
        (start + duration - 1.0).abs() < 1e-6,
        "presentation ends at exactly 1.0 s, got {}",
        start + duration
    );

    // A remux changes framing, never samples: ffmpeg's decode of the MP4 must
    // reproduce its decode of the source. MP4 has no end-of-stream granule, so
    // the tail encoder padding stays (ffmpeg's own opus -> mp4 remux keeps it
    // too); the source's samples must be a prefix, sample for sample.
    let from_mp4 = decode_pcm(&out);
    let from_src = decode_pcm(&src);
    assert!(
        from_mp4.len() >= from_src.len(),
        "the MP4 decodes at least the source's {} bytes, got {}",
        from_src.len(),
        from_mp4.len()
    );
    assert_eq!(
        &from_mp4[..from_src.len()],
        &from_src[..],
        "the remux decodes to the source's samples"
    );

    persist::record_evidence(
        "mp4mux",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("opus")
            .detail("ffmpeg decodes a g2g ogg -> mp4 remux to the source's samples, pre-skip kept"),
    )
    .expect("record oracle evidence");

    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&out);
}

/// An Opus-in-MP4 decoded through g2g. The same Opus packets in Ogg and in MP4
/// are the two sides: the MP4 fixture is a stream copy of the Ogg one, so the
/// two decodes must agree sample for sample, which they can only do if MP4
/// delivers the same pre-skip and end trim Ogg does. Nothing forwarded the
/// pre-skip out of MP4 before this milestone, so this could not hold.
#[cfg(feature = "opus")]
#[tokio::test]
async fn mp4_carries_the_same_trims_as_ogg_and_aligns_with_ffmpeg() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let ogg = temp_path("src-decode.opus");
    let ogg_bytes = author(&ogg);
    let mp4 = temp_path("src-decode.mp4");
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&ogg)
        .args(["-c", "copy"])
        .arg(&mp4)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg stream-copied the Opus into MP4");
    let mp4_bytes = std::fs::read(&mp4).expect("read mp4 fixture");

    let from_mp4 = mp4demux_frames(&mp4_bytes).await;
    assert!(
        from_mp4[0].0.starts_with(b"OpusHead"),
        "the dOps arrives in band as an OpusHead"
    );
    assert_eq!(
        head_pre_skip(&from_mp4[0].0),
        ENCODER_PRE_SKIP,
        "the file's dOps pre-skip survived the rebuild"
    );
    let from_ogg = oggdemux_frames(&ogg_bytes).await.frames;

    // Same packets, same container-declared trims: the PCM must be identical.
    let mp4_pcm = decode_frames(&from_mp4).await;
    let ogg_pcm = decode_frames(&from_ogg).await;
    assert_eq!(
        mp4_pcm, ogg_pcm,
        "MP4 and Ogg deliver the same pre-skip and end trim for the same packets"
    );
    assert_eq!(
        mp4_pcm.len(),
        48_000 * 2 * 4,
        "one second of 48 kHz stereo F32 survives the two trims"
    );

    // Reference peer: ffmpeg's own libopus decode of the MP4, which trims the
    // pre-skip from the same `dOps` but keeps the tail padding (it ignores the
    // short final sample duration), so g2g's PCM is its prefix. Two libopus
    // builds are not bit-identical, so the check is a tolerance: a misalignment
    // of even one sample on this tone is ~0.03, thirty times the bound.
    let reference = decode_f32_libopus(&mp4);
    assert!(
        mp4_pcm.len() < reference.len(),
        "g2g trims the tail ffmpeg keeps: {} vs {} bytes",
        mp4_pcm.len(),
        reference.len()
    );
    let worst = f32_samples(&mp4_pcm)
        .into_iter()
        .zip(f32_samples(&reference))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("worst sample difference against ffmpeg's libopus decode: {worst}");
    assert!(
        worst < 1e-3,
        "g2g's decode is sample-aligned with ffmpeg's, worst difference {worst}"
    );

    persist::record_evidence(
        "mp4demux",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("opus")
            .detail("g2g demux + decode of an Opus-in-MP4 is sample-aligned with ffmpeg's"),
    )
    .expect("record oracle evidence");

    let _ = std::fs::remove_file(&ogg);
    let _ = std::fs::remove_file(&mp4);
}

/// The encode direction: `OpusEnc` straight into `mp4muxn`, with no source
/// container to copy a header from. The `dOps` falls back to libopus'
/// lookahead, which the encoder itself reports, and the encoded second lands on
/// the media timeline exactly.
#[cfg(feature = "opus")]
#[tokio::test]
async fn opusenc_into_mp4_declares_the_encoders_own_lookahead() {
    use g2g_plugins::opusenc::OpusEnc;

    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
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
    enc.configure_pipeline(&Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: 2,
        sample_rate: rate,
    })
    .expect("configure opusenc");
    let lookahead = enc.lookahead().expect("the encoder reports its lookahead");
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

    let muxed = mp4mux_bytes(&encoded.frames).await;
    let out = temp_path("out-from-enc.mp4");
    std::fs::write(&out, &muxed).expect("write muxed");

    // The muxer has no header to copy, so it declares libopus' lookahead; the
    // encoder's own report is the expectation, not a hard-coded number.
    assert_eq!(
        u32::from(dops_pre_skip(&dops_payload(&muxed))),
        lookahead,
        "the synthesized dOps declares the encoder's real lookahead"
    );
    assert_eq!(elst_entry(&muxed).1 as u32, lookahead, "the elst matches");

    let probed = probe(&out);
    println!("ffprobe out-from-enc.mp4 (lookahead {lookahead}): {probed:?}");
    assert_eq!(field(&probed, "codec_name"), "opus");
    // 1.0 s of input is exactly 50 whole 20 ms frames, so the media timeline is
    // exactly one second and the presentation starts one lookahead before zero.
    let start = probed_f64(&probed, "start_time");
    let duration = probed_f64(&probed, "duration");
    assert!(
        (duration - 1.0).abs() < 1e-6,
        "the encoded second is exactly 1.0 s of media, got {duration}"
    );
    assert!(
        (start + f64::from(lookahead) / 48_000.0).abs() < 1e-6,
        "start_time {start} is one lookahead before zero"
    );
    // ffmpeg reads it back without complaint, the point of a muxer nobody else wrote.
    assert!(
        decode_pcm(&out).len() > (rate as usize * 4) / 2,
        "ffmpeg decoded a full second"
    );

    let _ = std::fs::remove_file(&out);
}

/// MP4 -> g2g demux -> g2g mux: the `dOps` comes back byte-identical, so the
/// pre-skip, output gain and channel mapping all survive a round trip.
#[tokio::test]
async fn mp4_round_trip_reproduces_the_dops_byte_for_byte() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let src = temp_path("src-roundtrip.mp4");
    let file = author(&src);
    let source_dops = dops_payload(&file);

    let frames = mp4demux_frames(&file).await;
    let remuxed = mp4mux_bytes(&frames).await;

    assert_eq!(
        dops_payload(&remuxed),
        source_dops,
        "the dOps survives demux -> mux unchanged"
    );

    persist::record_evidence(
        "mp4mux",
        &Evidence::new(ConformanceDimension::RoundTrip)
            .codec("opus")
            .detail("mp4demux -> mp4mux reproduces the source dOps byte for byte"),
    )
    .expect("record round-trip evidence");

    let _ = std::fs::remove_file(&src);
}
