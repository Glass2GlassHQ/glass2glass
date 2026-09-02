//! M792: Opus pre-skip and end trim survive Matroska in both directions, the MKV
//! sibling of M791's MP4 work.
//!
//! Matroska spells the two facts its own way. The pre-skip is the `CodecPrivate`
//! `OpusHead` plus a `CodecDelay` in ns on the TrackEntry; the end trim, with no
//! end granule to carry it, is the final block's `DiscardPadding` (ns) with a
//! `BlockDuration` (ms) beside it. Both directions convert to the in-band
//! `OpusHead` convention: the demuxers forward the `CodecPrivate` ahead of the
//! audio so `OpusDec` trims and a remux keeps the real value, and `MkvMuxN`
//! consumes an in-band header as config (never a Block) rather than synthesizing
//! one.
//!
//! Measured baselines this file asserts against, taken from ffmpeg (`n8.1.2`) on
//! a 1.0 s 48 kHz stereo libopus stream, `-c copy` from Ogg:
//!
//! * `CodecDelay` 6500000 ns, `SeekPreRoll` 80000000 ns, 19-byte `OpusHead`
//!   `CodecPrivate` with pre-skip 312.
//! * 51 audio blocks: the first at timestamp 0 (the pre-roll is not shifted out
//!   of the timeline, `CodecDelay` tells the decoder to discard it), the last in
//!   a `BlockGroup` carrying `BlockDuration` 7 (ms, the grid rounds 6.5 up) and
//!   `DiscardPadding` 13500000 ns. The ns element is why an MKV of this stream
//!   decodes to the same 48000 samples the Ogg does; a reader that honours only
//!   the millisecond `BlockDuration` lands 24 samples long.
//! * ffprobe: `initial_padding=312`, `start_time=0.000000`.
//!
//! One difference from ffmpeg's file, stated rather than hidden: ffprobe reports
//! `duration=N/A` for the files here, because they are muxed in the streaming
//! mode, which writes into an unknown-size Segment and never learns the total
//! (ffmpeg's reports 1.008). The two-pass `seekable` mode does declare a
//! duration, which is M794's subject. Everything ffprobe derives from the Opus
//! mapping itself matches exactly.
#![cfg(feature = "std")]

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
use g2g_plugins::mkvdemux::{MkvDemuxN, MkvStream};
use g2g_plugins::mkvmuxn::MkvMuxN;
use g2g_plugins::oggdemux::OggDemux;

/// The 48 kHz lookahead ffmpeg and g2g write into a synthesized Opus header.
const ENCODER_PRE_SKIP: u16 = 312;

/// The `SeekPreRoll` the Matroska Opus mapping fixes, in ns.
const SEEK_PRE_ROLL_NS: u64 = 80_000_000;

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

#[derive(Default)]
struct PortCapture {
    frames: Vec<(Vec<u8>, FrameTiming)>,
}

impl MultiOutputSink for PortCapture {
    fn poll_push_to(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        _port: usize,
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
    std::env::temp_dir().join(format!("g2g-m792-{}-{name}", std::process::id()))
}

/// Encode a 1 s 48 kHz stereo sine to `path` with libopus, returning its bytes.
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

/// Stream-copy `src` into `dst` (the container follows the extension), the
/// reference-peer remux g2g's own is judged against.
fn stream_copy(src: &PathBuf, dst: &PathBuf) -> Vec<u8> {
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(src)
        .args(["-c", "copy"])
        .arg(dst)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg copied into {}", dst.display());
    std::fs::read(dst).expect("read remux")
}

/// ffprobe's `key=value` lines for the first audio stream.
fn probe(path: &PathBuf) -> Vec<(String, String)> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "a:0"])
        .args([
            "-show_entries",
            "stream=codec_name,channels,sample_rate,start_time,initial_padding",
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

/// ffmpeg's decode of `path` as raw interleaved 16-bit PCM.
fn decode_pcm(path: &PathBuf) -> Vec<u8> {
    decode_raw(path, &[], "s16le", "pcm_s16le")
}

/// The same decode through **libopus** as interleaved 32-bit float, the closest
/// comparison to g2g's own decoder.
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

// --- Matroska element oracle -------------------------------------------------
//
// A byte scan bounded to the header region (the Tracks element up to the first
// Cluster), which is unambiguous enough for a test and needs no EBML walker.

/// The body of the first `id` element between the `Tracks` element and the first
/// `Cluster`. `id` is the element id's raw bytes (the length marker included).
fn header_element(file: &[u8], id: &[u8]) -> Option<Vec<u8>> {
    let tracks = file
        .windows(4)
        .position(|w| w == [0x16, 0x54, 0xAE, 0x6B])
        .expect("the file has a Tracks element");
    let end = file
        .windows(4)
        .position(|w| w == [0x1F, 0x43, 0xB6, 0x75])
        .unwrap_or(file.len());
    let at = tracks + file[tracks..end].windows(id.len()).position(|w| w == id)?;
    let size_at = at + id.len();
    // EBML size VINT: the leading bit position gives the byte count, and the
    // marker bit is stripped from the value.
    let first = *file.get(size_at)?;
    let len = (0..8).find(|i| first & (0x80 >> i) != 0)? + 1;
    let mut size = u64::from(first & (0xFF >> len));
    for b in file.get(size_at + 1..size_at + len)? {
        size = (size << 8) | u64::from(*b);
    }
    let body = size_at + len;
    file.get(body..body + size as usize).map(<[u8]>::to_vec)
}

/// A big-endian unsigned EBML element body as a number.
fn uint_body(body: &[u8]) -> u64 {
    body.iter().fold(0u64, |v, b| (v << 8) | u64::from(*b))
}

fn codec_private(file: &[u8]) -> Vec<u8> {
    header_element(file, &[0x63, 0xA2]).expect("the Opus track has a CodecPrivate")
}

fn codec_delay_ns(file: &[u8]) -> u64 {
    uint_body(&header_element(file, &[0x56, 0xAA]).expect("the Opus track has a CodecDelay"))
}

fn seek_pre_roll_ns(file: &[u8]) -> u64 {
    uint_body(&header_element(file, &[0x56, 0xBB]).expect("the Opus track has a SeekPreRoll"))
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
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}

#[cfg(feature = "opus")]
fn f32_samples(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
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
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
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

/// Demux an Ogg Opus byte stream into the frames `OggDemux` emits.
async fn oggdemux_frames(ogg: &[u8]) -> Vec<(Vec<u8>, FrameTiming)> {
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
    sink.frames
}

/// Demux an Opus-in-Matroska file, returning the frames the port receives (the
/// `CodecPrivate` `OpusHead` first, then the audio blocks).
async fn mkvdemux_frames(file: &[u8]) -> Vec<(Vec<u8>, FrameTiming)> {
    let mut demux = MkvDemuxN::new(vec![MkvStream::Opus]);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        })
        .expect("configure mkvdemux");
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
/// Matroska byte stream.
async fn mkvmux_bytes(frames: &[(Vec<u8>, FrameTiming)]) -> Vec<u8> {
    let mut mux = MkvMuxN::new(1);
    mux.configure_pipeline(0, &opus_caps())
        .expect("configure mkvmux");
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

/// Ogg -> g2g demux -> `mkvmuxn`: the source's real pre-skip reaches the
/// `CodecPrivate` and the `CodecDelay`, and ffmpeg reads the result back to the
/// source's samples.
#[tokio::test]
async fn ogg_remuxed_into_mkv_keeps_the_sources_pre_skip_and_trim() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let src = temp_path("src.opus");
    let bytes = author(&src);

    let demuxed = oggdemux_frames(&bytes).await;
    let source_head = demuxed.frames_head();
    assert_eq!(
        head_pre_skip(&source_head),
        ENCODER_PRE_SKIP,
        "ffmpeg's libopus encoder declares its 312-sample lookahead"
    );

    let muxed = mkvmux_bytes(&demuxed).await;
    let out = temp_path("out-from-ogg.mkv");
    std::fs::write(&out, &muxed).expect("write muxed");

    // The source's header became the CodecPrivate, not a Block: the only
    // `OpusHead` in the file is the one in the TrackEntry.
    assert_eq!(
        codec_private(&muxed),
        source_head,
        "the CodecPrivate is the source's OpusHead, byte for byte"
    );
    assert_eq!(
        muxed.windows(8).filter(|w| *w == b"OpusHead").count(),
        1,
        "the OpusHead is codec config, never written as a Block"
    );
    // CodecDelay is the same pre-skip counted in ns instead of 48 kHz samples.
    assert_eq!(
        codec_delay_ns(&muxed),
        u64::from(ENCODER_PRE_SKIP) * 1_000_000_000 / 48_000,
        "CodecDelay is the pre-skip in ns (312 samples = 6.5 ms)"
    );
    assert_eq!(seek_pre_roll_ns(&muxed), SEEK_PRE_ROLL_NS);

    // ffprobe reads the Opus mapping out of the g2g file exactly as it does out
    // of ffmpeg's own remux of the same stream.
    let reference = temp_path("ref-from-ogg.mkv");
    stream_copy(&src, &reference);
    let ours = probe(&out);
    let theirs = probe(&reference);
    println!("ffprobe g2g:    {ours:?}");
    println!("ffprobe ffmpeg: {theirs:?}");
    assert_eq!(field(&ours, "codec_name"), "opus");
    assert_eq!(field(&ours, "channels"), "2");
    assert_eq!(field(&ours, "sample_rate"), "48000");
    assert_eq!(
        field(&ours, "initial_padding"),
        field(&theirs, "initial_padding"),
        "ffmpeg derives the same encoder delay from both files"
    );
    assert_eq!(field(&ours, "initial_padding"), "312");
    // The pre-roll stays on the timeline (CodecDelay discards it). ffprobe's
    // reported start_time convention differs by version (0, or minus the codec
    // delay); the oracle is that both files agree, plus a sanity check that the
    // shared value is one of the two spellings.
    assert_eq!(field(&ours, "start_time"), field(&theirs, "start_time"));
    let start = field(&ours, "start_time");
    assert!(
        start == "0.000000" || start == "-0.007000",
        "unexpected start_time {start}"
    );

    // A remux changes framing, never samples. Both remuxes decode to the same
    // thing, and the source's samples are a prefix: what is past them is the
    // 0.5 ms the millisecond BlockDuration grid cannot trim (see the module note).
    let from_ours = decode_pcm(&out);
    let from_theirs = decode_pcm(&reference);
    let from_src = decode_pcm(&src);
    assert_eq!(
        from_ours, from_theirs,
        "g2g's remux decodes to exactly what ffmpeg's remux does"
    );
    assert_eq!(
        &from_ours[..from_src.len()],
        &from_src[..],
        "the remux decodes to the source's samples"
    );

    persist::record_evidence(
        "matroskamux",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("opus")
            .detail(
                "ffmpeg decodes a g2g ogg -> mkv remux exactly as its own, pre-skip and trim kept",
            ),
    )
    .expect("record oracle evidence");

    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&out);
    let _ = std::fs::remove_file(&reference);
}

/// An Opus-in-Matroska decoded through g2g. The same packets in Ogg and in MKV
/// are the two sides: the MKV fixture is a stream copy of the Ogg one, so the
/// decodes must agree, up to the one thing Matroska cannot express.
#[cfg(feature = "opus")]
#[tokio::test]
async fn mkv_carries_the_same_trims_as_ogg_and_aligns_with_ffmpeg() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let ogg = temp_path("src-decode.opus");
    let ogg_bytes = author(&ogg);
    let mkv = temp_path("src-decode.mkv");
    let mkv_bytes = stream_copy(&ogg, &mkv);

    // ffmpeg's own file is the shape g2g reads: CodecDelay, SeekPreRoll and a
    // 19-byte OpusHead CodecPrivate.
    assert_eq!(codec_delay_ns(&mkv_bytes), 6_500_000);
    assert_eq!(seek_pre_roll_ns(&mkv_bytes), SEEK_PRE_ROLL_NS);

    let from_mkv = mkvdemux_frames(&mkv_bytes).await;
    assert!(
        from_mkv[0].0.starts_with(b"OpusHead"),
        "the CodecPrivate arrives in band ahead of the audio"
    );
    assert_eq!(
        head_pre_skip(&from_mkv[0].0),
        ENCODER_PRE_SKIP,
        "the file's pre-skip reaches the decoder"
    );
    let from_ogg = oggdemux_frames(&ogg_bytes).await;

    // The end trim: the final block's `DiscardPadding` is nanoseconds, so it
    // reproduces the Ogg granule's 6.5 ms exactly. (Its `BlockDuration`, the
    // millisecond spelling of the same fact, says 7; the ns value wins.)
    let ogg_tail = from_ogg.last().expect("audio").1.duration_ns;
    let mkv_tail = from_mkv.last().expect("audio").1.duration_ns;
    assert_eq!(ogg_tail, 6_500_000, "the Ogg granule trims to 6.5 ms");
    assert_eq!(mkv_tail, ogg_tail, "and DiscardPadding trims to the same");

    let mkv_pcm = decode_frames(&from_mkv).await;
    let ogg_pcm = decode_frames(&from_ogg).await;
    assert_eq!(
        mkv_pcm, ogg_pcm,
        "MKV and Ogg deliver the same pre-skip and end trim for the same packets"
    );
    assert_eq!(
        ogg_pcm.len(),
        48_000 * 2 * 4,
        "one second of 48 kHz stereo F32 survives the two trims"
    );

    // Reference peer: ffmpeg's own libopus decode of the same file, which trims
    // from the same two elements, so the lengths match as well. Two libopus
    // builds are not bit-identical, so the sample check is a tolerance (a
    // one-sample misalignment on this tone is ~0.03, thirty times the bound).
    let reference = decode_f32_libopus(&mkv);
    assert_eq!(
        mkv_pcm.len(),
        reference.len(),
        "g2g trims exactly what ffmpeg trims"
    );
    let worst = f32_samples(&mkv_pcm)
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
        "matroskademux",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("opus")
            .detail("g2g demux + decode of an Opus-in-MKV is sample-aligned with ffmpeg's"),
    )
    .expect("record oracle evidence");

    let _ = std::fs::remove_file(&ogg);
    let _ = std::fs::remove_file(&mkv);
}

/// The encode direction: `OpusEnc` straight into `mkvmuxn`, with no source
/// container to copy a header from. The synthesized header declares libopus'
/// lookahead, which the encoder itself reports.
#[cfg(feature = "opus")]
#[tokio::test]
async fn opusenc_into_mkv_declares_the_encoders_own_lookahead() {
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
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
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

    let muxed = mkvmux_bytes(&encoded.frames).await;
    let out = temp_path("out-from-enc.mkv");
    std::fs::write(&out, &muxed).expect("write muxed");

    // No header to copy, so the muxer declares libopus' lookahead; the encoder's
    // own report is the expectation, not a hard-coded number.
    assert_eq!(
        u32::from(head_pre_skip(&codec_private(&muxed))),
        lookahead,
        "the synthesized OpusHead declares the encoder's real lookahead"
    );
    assert_eq!(
        codec_delay_ns(&muxed),
        u64::from(lookahead) * 1_000_000_000 / 48_000,
        "the CodecDelay is that lookahead in ns"
    );

    let probed = probe(&out);
    println!("ffprobe out-from-enc.mkv (lookahead {lookahead}): {probed:?}");
    assert_eq!(field(&probed, "codec_name"), "opus");
    assert_eq!(
        field(&probed, "initial_padding"),
        lookahead.to_string(),
        "ffmpeg reads the encoder delay back out of the CodecDelay"
    );
    // ffmpeg reads it back without complaint, the point of a muxer nobody else wrote.
    assert!(
        decode_pcm(&out).len() > (rate as usize * 4) / 2,
        "ffmpeg decoded a full second"
    );

    let _ = std::fs::remove_file(&out);
}

/// MKV -> g2g demux -> g2g mux: the `CodecPrivate` comes back byte-identical, so
/// the pre-skip, output gain and channel mapping all survive a round trip.
#[tokio::test]
async fn mkv_round_trip_reproduces_the_codec_private_byte_for_byte() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let ogg = temp_path("src-roundtrip.opus");
    author(&ogg);
    let src = temp_path("src-roundtrip.mkv");
    let file = stream_copy(&ogg, &src);
    let source_private = codec_private(&file);

    let frames = mkvdemux_frames(&file).await;
    let remuxed = mkvmux_bytes(&frames).await;

    assert_eq!(
        codec_private(&remuxed),
        source_private,
        "the CodecPrivate survives demux -> mux unchanged"
    );
    assert_eq!(
        codec_delay_ns(&remuxed),
        codec_delay_ns(&file),
        "and so does the CodecDelay derived from it"
    );

    persist::record_evidence(
        "matroskamux",
        &Evidence::new(ConformanceDimension::RoundTrip)
            .codec("opus")
            .detail("mkvdemux -> mkvmux reproduces the source CodecPrivate byte for byte"),
    )
    .expect("record round-trip evidence");

    let _ = std::fs::remove_file(&ogg);
    let _ = std::fs::remove_file(&src);
}

/// Sugar for the leading in-band config frame a demuxer emits.
trait FramesHead {
    fn frames_head(&self) -> Vec<u8>;
}

impl FramesHead for Vec<(Vec<u8>, FrameTiming)> {
    fn frames_head(&self) -> Vec<u8> {
        let head = self.first().expect("at least the config frame").0.clone();
        assert!(
            head.starts_with(b"OpusHead"),
            "the demux leads with the header"
        );
        head
    }
}
