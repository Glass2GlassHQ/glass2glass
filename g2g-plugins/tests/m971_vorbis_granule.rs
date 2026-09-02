//! M971: the Vorbis end-granule trim. A logical bitstream's final granule
//! position is its count of playable samples: the first audio packet primes the
//! overlap window and decodes to nothing, and the encoder's tail padding sits
//! past the granule. `oggdemux` clips both ends, so a decode yields exactly that
//! many samples whether the audio spans several pages or one, and a remux hands
//! `oggmux` a timeline that reproduces the source's granule axis.
//!
//! Oracles: the file's own final granule position, parsed with the real
//! [`g2g_plugins::ogg::OggDemuxer`], and ffmpeg for the sample values. ffmpeg is
//! not a length oracle for a stream whose audio fits on a single page: there it
//! drops the priming packet's `blocksize / 2` off the tail (checked against
//! gstreamer's decode and against the encoder's own input length, both of which
//! give the granule), so the exact-length assertions use the granule and ffmpeg
//! is compared over the samples the two decodes share.
#![cfg(all(feature = "std", feature = "vorbis"))]

use std::path::{Path, PathBuf};
use std::process::Command;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, G2gError, OutputSink, PipelineClock,
    PropValue, PushOutcome,
};
use g2g_plugins::ogg::OggDemuxer;
use g2g_plugins::oggdemux::OggDemux;
use g2g_plugins::oggmux::OggMux;
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Captures pushed frames with their timing; the muxer reads `duration_ns` to
/// place its granule positions.
#[derive(Default)]
struct CaptureSink {
    frames: Vec<(Vec<u8>, FrameTiming)>,
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

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m971-{}-{name}", std::process::id()))
}

/// One ffmpeg-authored Ogg Vorbis fixture.
struct Fixture {
    name: &'static str,
    /// lavfi source description.
    source: &'static str,
    channels: u8,
    sample_rate: u32,
}

/// A short tone: its audio fits on one page, the case ffmpeg reads short.
const SHORT_TONE: Fixture = Fixture {
    name: "short-tone",
    source: "sine=frequency=440:duration=0.5:sample_rate=44100",
    channels: 2,
    sample_rate: 44_100,
};

/// Repeated transients, so the encoder switches between long and short blocks
/// and the lapped packet durations vary.
const TRANSIENTS: Fixture = Fixture {
    name: "transients",
    source: "aevalsrc=0.6*sin(3000*t)*lt(mod(t\\,0.1)\\,0.01):d=3.1:s=44100",
    channels: 2,
    sample_rate: 44_100,
};

/// Short mono at a low rate: the tail padding spans more than the final
/// packet's own output.
const MONO_NARROWBAND: Fixture = Fixture {
    name: "mono-narrowband",
    source: "sine=frequency=1000:duration=0.037:sample_rate=8000",
    channels: 1,
    sample_rate: 8_000,
};

/// Encode `fixture` with libvorbis into a temporary `.ogg`, returning its path.
fn author(fixture: &Fixture, tag: &str) -> PathBuf {
    let path = temp_path(&format!("{tag}-{}.ogg", fixture.name));
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "lavfi", "-i"])
        .arg(fixture.source)
        .args(["-ac", &fixture.channels.to_string()])
        .args(["-ar", &fixture.sample_rate.to_string()])
        .args(["-c:a", "libvorbis"])
        .arg(&path)
        .status()
        .expect("run ffmpeg");
    assert!(
        status.success(),
        "ffmpeg authored the {} fixture",
        fixture.name
    );
    path
}

/// The final granule position of an Ogg byte stream: the Vorbis stream's count
/// of playable samples per channel.
fn final_granule(ogg: &[u8]) -> u64 {
    let mut demuxer = OggDemuxer::new();
    demuxer.push_data(ogg);
    demuxer
        .end_granule()
        .expect("the file carries an EOS granule")
}

/// ffmpeg's decode of `path` to interleaved S16LE.
fn ffmpeg_pcm(path: &PathBuf) -> Vec<u8> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-f", "s16le", "-c:a", "pcm_s16le", "-"])
        .output()
        .expect("run ffmpeg");
    assert!(out.status.success(), "ffmpeg decoded {}", path.display());
    out.stdout
}

/// g2g's decode of `path` to interleaved S16LE, through the launch line a user
/// would write.
async fn g2g_pcm(path: &Path, fixture: &Fixture) -> Vec<u8> {
    let out = path.with_extension("raw");
    let _ = std::fs::remove_file(&out);
    let line = format!(
        "filesrc location={src} ! oggdemux stream=vorbis ! vorbisdec ! audioconvert ! \
         audio/x-raw,format=S16LE,rate={rate},channels={ch} ! filesink location={out}",
        src = path.display(),
        rate = fixture.sample_rate,
        ch = fixture.channels,
        out = out.display(),
    );
    let graph = parse_launch(&default_registry(), &line).expect("pipeline parses");
    run_graph(graph, &ZeroClock, 4)
        .await
        .expect("pipeline runs");
    let pcm = std::fs::read(&out).expect("pcm written");
    let _ = std::fs::remove_file(&out);
    pcm
}

/// Demux an Ogg byte stream to the frames `oggdemux` emits: the three in-band
/// Vorbis headers, then the timed audio packets.
async fn oggdemux_frames(ogg: &[u8]) -> CaptureSink {
    let mut demux = OggDemux::new();
    demux
        .set_property("stream", PropValue::Str("vorbis".into()))
        .expect("stream property");
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Ogg,
        })
        .expect("configure");
    let mut sink = CaptureSink::default();
    for piece in ogg.chunks(1021) {
        demux
            .process(frame(piece.to_vec(), FrameTiming::default()), &mut sink)
            .await
            .expect("demux");
    }
    demux
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("demux eos");
    sink
}

/// Mux demuxed frames back into an Ogg byte stream.
async fn oggmux_bytes(frames: &CaptureSink, fixture: &Fixture) -> Vec<u8> {
    let mut mux = OggMux::new();
    mux.configure_pipeline(&Caps::Audio {
        format: AudioFormat::Vorbis,
        channels: fixture.channels,
        sample_rate: fixture.sample_rate,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    })
    .expect("configure");
    let mut sink = CaptureSink::default();
    for (data, timing) in &frames.frames {
        mux.process(frame(data.clone(), *timing), &mut sink)
            .await
            .expect("mux");
    }
    mux.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("mux eos");
    sink.frames.iter().flat_map(|(b, _)| b.clone()).collect()
}

/// Every sample of `pcm` is within one int16 LSB of `reference` over the samples
/// the two decodes share (both decode the same lossy bitstream, rounding to
/// int16 independently).
fn assert_samples_agree(pcm: &[u8], reference: &[u8], name: &str) {
    let shared = pcm.len().min(reference.len());
    assert!(shared > 0, "{name}: both decodes produced audio");
    let mut max_diff = 0i32;
    for (a, b) in pcm[..shared]
        .as_chunks::<2>()
        .0
        .iter()
        .zip(reference[..shared].as_chunks::<2>().0.iter())
    {
        let x = i16::from_le_bytes([a[0], a[1]]) as i32;
        let y = i16::from_le_bytes([b[0], b[1]]) as i32;
        max_diff = max_diff.max((x - y).abs());
    }
    assert!(
        max_diff <= 1,
        "{name}: samples within 1 LSB of ffmpeg, got {max_diff}"
    );
}

/// A decode yields exactly the final granule position's worth of samples: the
/// priming packet's silent lead is clipped and the encoder's tail padding is
/// trimmed, for a single-page stream as much as a multi-page one.
#[tokio::test]
async fn decode_length_is_the_final_granule() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    for fixture in [SHORT_TONE, TRANSIENTS, MONO_NARROWBAND] {
        let src = author(&fixture, "decode");
        let granule = final_granule(&std::fs::read(&src).expect("read fixture"));
        let pcm = g2g_pcm(&src, &fixture).await;
        assert_eq!(
            pcm.len(),
            granule as usize * 2 * fixture.channels as usize,
            "{}: decoded {granule} samples per channel",
            fixture.name
        );
        assert_samples_agree(&pcm, &ffmpeg_pcm(&src), fixture.name);
        let _ = std::fs::remove_file(&src);
    }
}

/// A demux -> mux round trip reproduces the source's granule axis: the remux
/// carries the same packets, ends on the same granule position, and decodes to
/// the same samples.
#[tokio::test]
async fn remux_reproduces_the_source_granule() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    for fixture in [SHORT_TONE, TRANSIENTS, MONO_NARROWBAND] {
        let src = author(&fixture, "remux");
        let bytes = std::fs::read(&src).expect("read fixture");
        let demuxed = oggdemux_frames(&bytes).await;
        let muxed = oggmux_bytes(&demuxed, &fixture).await;
        assert_eq!(
            final_granule(&muxed),
            final_granule(&bytes),
            "{}: the remux ends on the source's granule",
            fixture.name
        );

        let out = temp_path(&format!("remuxed-{}.ogg", fixture.name));
        std::fs::write(&out, &muxed).expect("write remux");
        assert_eq!(
            g2g_pcm(&out, &fixture).await,
            g2g_pcm(&src, &fixture).await,
            "{}: the remux decodes to the source's samples",
            fixture.name
        );
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&out);
    }
}

/// ffmpeg decodes a g2g remux to the source's samples, on a fixture whose audio
/// spans several pages either way (the case ffmpeg's own reader gets right, as
/// the precondition asserts).
#[tokio::test]
async fn ffmpeg_decodes_a_remux_to_the_source_samples() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let fixture = TRANSIENTS;
    let src = author(&fixture, "peer");
    let bytes = std::fs::read(&src).expect("read fixture");
    let reference = ffmpeg_pcm(&src);
    assert_eq!(
        reference.len(),
        final_granule(&bytes) as usize * 2 * fixture.channels as usize,
        "ffmpeg reads the source to its granule, so it can serve as the oracle"
    );

    let demuxed = oggdemux_frames(&bytes).await;
    let muxed = oggmux_bytes(&demuxed, &fixture).await;
    let out = temp_path("transients-remux-ffmpeg.ogg");
    std::fs::write(&out, &muxed).expect("write remux");
    assert_eq!(
        ffmpeg_pcm(&out),
        reference,
        "ffmpeg decodes the remux to the source's samples"
    );
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&out);
}
