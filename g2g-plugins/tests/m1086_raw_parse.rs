//! M1086 headerless raw framers: `rawvideoparse` cuts a `.yuv` dump into frames
//! and `rawaudioparse` cuts a `.pcm` dump into sample-aligned buffers, both from
//! the shape their properties declare.
//!
//! There is no header to check against, so the reference is an existing fixture
//! read through the element that does know its shape: the payload of the
//! `y4m_64x48_i420.y4m` frames must come back out of `rawvideoparse` unchanged
//! at the geometry ffprobe recorded, and the `data` chunk of
//! `sine_8k_mono_s16le.wav` must come back out of `rawaudioparse` unchanged at
//! the rate and channel count its `fmt ` chunk declares.
//!
//! `default_registry` is `std`-gated, so this file is too: run with
//! `cargo test -p g2g-plugins --features std`.
#![cfg(feature = "std")]

use std::path::PathBuf;

use g2g_core::frame::Frame;
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, Dim, FrameTiming, G2gError, Interlace,
    OutputSink, PipelineClock, PipelinePacket, PropValue, PushOutcome, Rate, RawVideoFormat,
};
use g2g_plugins::rawaudioparse::RawAudioParse;
use g2g_plugins::rawvideoparse::RawVideoParse;
use g2g_plugins::registry::default_registry;

/// The y4m fixture whose frames stand in for a raw video dump, and the wav
/// fixture whose `data` chunk stands in for a raw audio dump.
const Y4M_FIXTURE: &str = "y4m_64x48_i420.y4m";
const WAV_FIXTURE: &str = "sine_8k_mono_s16le.wav";
const MULAW_FIXTURE: &str = "sine_8k_mono_mulaw.wav";

/// The line before each frame's planes in a y4m file.
const FRAME_MARKER: &[u8] = b"FRAME\n";

/// `RIFF` + size + `WAVE`, then 4-byte id + 4-byte size per chunk.
const RIFF_HEADER_LEN: usize = 12;
const CHUNK_HEADER_LEN: usize = 8;

/// Bytes per input buffer when an element is driven directly: an odd size, so
/// frames and sample frames straddle buffer boundaries.
const CHUNK_LEN: usize = 997;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        self.packets
            .push(packet_slot.take().expect("poll_push without a packet"));
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

impl Collect {
    fn payloads(&self) -> Vec<Vec<u8>> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => {
                    Some(f.domain.as_system_slice().expect("system frame").to_vec())
                }
                _ => None,
            })
            .collect()
    }

    fn timing(&self) -> Vec<(u64, u64)> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some((f.timing.pts_ns, f.timing.duration_ns)),
                _ => None,
            })
            .collect()
    }

    fn caps(&self) -> Vec<Caps> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::CapsChanged(c) => Some(c.clone()),
                _ => None,
            })
            .collect()
    }
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn read_fixture(name: &str) -> Vec<u8> {
    std::fs::read(fixture(name)).unwrap_or_else(|e| panic!("fixture {name}: {e}"))
}

fn temp_path(tag: &str, extension: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "g2g-m1086-{tag}-{}.{extension}",
        std::process::id()
    ))
}

/// What the checked-in `ffprobe` output says about the y4m fixture. The file is
/// single-stream, so the first value of each key is the one wanted.
struct Y4mProbe {
    width: u32,
    height: u32,
    framerate: (u32, u32),
    frames: usize,
}

impl Y4mProbe {
    fn read() -> Y4mProbe {
        let path = fixture(Y4M_FIXTURE).with_extension("json");
        let json = std::fs::read_to_string(&path).expect("the probe output is checked in");
        let value =
            |key: &str| json_value(&json, key).unwrap_or_else(|| panic!("{key} in {path:?}"));
        assert_eq!(value("pix_fmt"), "yuv420p", "the fixture is I420");
        let rate = value("r_frame_rate");
        let (numerator, denominator) = rate.split_once('/').expect("a num/den framerate");
        Y4mProbe {
            width: value("width").parse().expect("a width"),
            height: value("height").parse().expect("a height"),
            framerate: (
                numerator.parse().expect("a numerator"),
                denominator.parse().expect("a denominator"),
            ),
            frames: value("nb_read_frames").parse().expect("a frame count"),
        }
    }

    fn framerate_q16(&self) -> u32 {
        ((u64::from(self.framerate.0) << 16) / u64::from(self.framerate.1)) as u32
    }

    fn caps(&self) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: Rate::Fixed(self.framerate_q16()),
            interlace: Interlace::Progressive,
        }
    }

    fn frame_period_ns(&self) -> u64 {
        1_000_000_000u64 * 65536 / u64::from(self.framerate_q16())
    }
}

/// The first value of `"key":` in an ffprobe JSON dump, string or number.
fn json_value(json: &str, key: &str) -> Option<String> {
    let at = json.find(&format!("\"{key}\":"))?;
    let rest = json[at..].split_once(':')?.1.trim_start();
    let text = if let Some(quoted) = rest.strip_prefix('"') {
        quoted.split('"').next()?
    } else {
        rest.split([',', '\n', '}']).next()?.trim()
    };
    Some(text.to_string())
}

/// The planes of every frame in a y4m file, headers stripped: the same bytes a
/// `.yuv` dump of the clip would hold.
fn y4m_frame_payloads(file: &[u8]) -> Vec<Vec<u8>> {
    let mut payloads = Vec::new();
    let mut at = 0;
    // The stream header ends at the first FRAME line; every frame's planes run
    // from its own FRAME line to the next one (or the end of file).
    while let Some(found) = find(&file[at..], FRAME_MARKER) {
        let body = at + found + FRAME_MARKER.len();
        let end = find(&file[body..], FRAME_MARKER).map_or(file.len(), |next| body + next);
        payloads.push(file[body..end].to_vec());
        at = end;
    }
    payloads
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// The `data` chunk of a RIFF/WAVE file.
fn wav_data_chunk(file: &[u8]) -> &[u8] {
    let mut at = RIFF_HEADER_LEN;
    while at + CHUNK_HEADER_LEN <= file.len() {
        let id = &file[at..at + 4];
        let size =
            u32::from_le_bytes([file[at + 4], file[at + 5], file[at + 6], file[at + 7]]) as usize;
        let body = at + CHUNK_HEADER_LEN;
        if id == b"data" {
            return &file[body..(body + size).min(file.len())];
        }
        at = body + size + size % 2;
    }
    panic!("no data chunk");
}

/// `(channels, sample_rate)` from a RIFF/WAVE `fmt ` chunk, so the properties
/// the test sets are the fixture's own numbers.
fn wav_format(file: &[u8]) -> (u8, u32) {
    let mut at = RIFF_HEADER_LEN;
    while at + CHUNK_HEADER_LEN <= file.len() {
        let id = &file[at..at + 4];
        let size =
            u32::from_le_bytes([file[at + 4], file[at + 5], file[at + 6], file[at + 7]]) as usize;
        let body = at + CHUNK_HEADER_LEN;
        if id == b"fmt " {
            let channels = u16::from_le_bytes([file[body + 2], file[body + 3]]) as u8;
            let rate = u32::from_le_bytes([
                file[body + 4],
                file[body + 5],
                file[body + 6],
                file[body + 7],
            ]);
            return (channels, rate);
        }
        at = body + size + size % 2;
    }
    panic!("no fmt chunk");
}

/// Push `bytes` through `element` in `CHUNK_LEN` pieces, then end the stream.
async fn drive<E: AsyncElement>(element: &mut E, bytes: &[u8]) -> Collect {
    let mut out = Collect::default();
    for piece in bytes.chunks(CHUNK_LEN) {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(piece.to_vec().into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        element
            .process(PipelinePacket::DataFrame(frame), &mut out)
            .await
            .expect("the chunk parses");
    }
    element
        .process(PipelinePacket::Eos, &mut out)
        .await
        .expect("the tail flushes");
    out
}

/// Run `line` and return what its `filesink` wrote.
async fn run_to_file(line: &str, output: &PathBuf) -> Vec<u8> {
    let reg = default_registry();
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .expect("the pipeline runs");
    let bytes = std::fs::read(output).expect("the sink wrote a file");
    std::fs::remove_file(output).ok();
    bytes
}

// ---------------------------------------------------------------------------
// rawvideoparse
// ---------------------------------------------------------------------------

#[tokio::test]
async fn video_frames_match_the_y4m_fixture_payloads() {
    let probe = Y4mProbe::read();
    let payloads = y4m_frame_payloads(&read_fixture(Y4M_FIXTURE));
    assert_eq!(
        payloads.len(),
        probe.frames,
        "the fixture split into frames"
    );

    let mut parser = RawVideoParse::new()
        .with_geometry(probe.width, probe.height)
        .with_framerate(probe.framerate.0, probe.framerate.1);
    parser
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Raw,
        })
        .expect("a raw byte stream");
    let dump: Vec<u8> = payloads.concat();
    let out = drive(&mut parser, &dump).await;

    assert_eq!(out.payloads(), payloads, "every frame comes back whole");
    assert_eq!(out.caps(), vec![probe.caps()], "the declared shape, once");
    let period = probe.frame_period_ns();
    let expected: Vec<(u64, u64)> = (0..probe.frames as u64)
        .map(|index| (index * period, period))
        .collect();
    assert_eq!(out.timing(), expected);
}

#[tokio::test]
async fn a_yuv_file_plays_through_a_launch_line() {
    let probe = Y4mProbe::read();
    let dump: Vec<u8> = y4m_frame_payloads(&read_fixture(Y4M_FIXTURE)).concat();
    let input = temp_path("frames", "yuv");
    std::fs::write(&input, &dump).expect("the dump is written");
    let output = temp_path("relayed", "yuv");
    // No `bytestream-format`: the `.yuv` extension types the file as a raw dump.
    let line = format!(
        "filesrc location={} ! rawvideoparse width={} height={} format=I420 framerate={}/{} ! filesink location={}",
        input.display(),
        probe.width,
        probe.height,
        probe.framerate.0,
        probe.framerate.1,
        output.display()
    );
    let written = run_to_file(&line, &output).await;
    std::fs::remove_file(&input).ok();
    assert_eq!(written, dump, "the pixels reach the sink unchanged");
}

/// gst's older and bin names for the framers resolve to them. Each is
/// identified by a property only that framer declares.
#[tokio::test]
async fn the_gst_aliases_reach_the_framers() {
    let reg = default_registry();
    for (alias, property) in [
        ("videoparse", "frame-size"),
        ("unalignedvideoparse", "frame-size"),
        ("audioparse", "num-channels"),
        ("unalignedaudioparse", "num-channels"),
    ] {
        let element = reg
            .make_element(alias)
            .unwrap_or_else(|| panic!("`{alias}` is registered"));
        assert!(
            element
                .properties()
                .iter()
                .any(|spec| spec.name == property),
            "`{alias}` resolves to the framer declaring `{property}`"
        );
    }
}

// ---------------------------------------------------------------------------
// rawaudioparse
// ---------------------------------------------------------------------------

#[tokio::test]
async fn audio_buffers_match_the_wav_data_chunk() {
    let file = read_fixture(WAV_FIXTURE);
    let (channels, sample_rate) = wav_format(&file);
    let samples = wav_data_chunk(&file);

    let mut parser = RawAudioParse::new()
        .with_rate(sample_rate)
        .with_channels(channels);
    parser
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Raw,
        })
        .expect("a raw byte stream");
    let out = drive(&mut parser, samples).await;

    assert_eq!(
        out.payloads().concat(),
        samples,
        "every sample reaches the sink, in order"
    );
    assert_eq!(
        out.caps(),
        vec![Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels,
            sample_rate,
        }]
    );
    // S16LE: two bytes a sample, one channel per sample frame.
    let frame_bytes = 2 * usize::from(channels);
    assert!(
        out.payloads()
            .iter()
            .all(|buffer| buffer.len() % frame_bytes == 0),
        "no buffer boundary splits a sample frame"
    );
    let total_samples = (samples.len() / frame_bytes) as u64;
    let (last_pts, last_duration) = *out.timing().last().expect("a buffer");
    assert_eq!(
        last_pts + last_duration,
        total_samples * 1_000_000_000 / u64::from(sample_rate),
        "the stream lasts as long as its samples"
    );
}

#[tokio::test]
async fn mulaw_buffers_match_the_wav_data_chunk() {
    let file = read_fixture(MULAW_FIXTURE);
    let (channels, sample_rate) = wav_format(&file);
    let samples = wav_data_chunk(&file);

    let mut parser = RawAudioParse::new()
        .with_rate(sample_rate)
        .with_channels(channels);
    parser
        .set_property("format", PropValue::Str("mulaw".into()))
        .expect("mu-law is a format value");
    parser
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Raw,
        })
        .expect("a raw byte stream");
    let out = drive(&mut parser, samples).await;

    assert_eq!(out.payloads().concat(), samples);
    assert_eq!(
        out.caps(),
        vec![Caps::Audio {
            format: AudioFormat::Mulaw,
            channels,
            sample_rate,
        }],
        "the caps `mulawdec` accepts"
    );
}

#[tokio::test]
async fn a_pcm_file_decodes_through_a_launch_line() {
    let file = read_fixture(MULAW_FIXTURE);
    let (channels, sample_rate) = wav_format(&file);
    let input = temp_path("mulaw", "pcm");
    std::fs::write(&input, wav_data_chunk(&file)).expect("the dump is written");
    let output = temp_path("decoded", "s16le");
    let line = format!(
        "filesrc location={} ! rawaudioparse format=mulaw sample-rate={sample_rate} num-channels={channels} ! mulawdec ! filesink location={}",
        input.display(),
        output.display()
    );
    let written = run_to_file(&line, &output).await;
    std::fs::remove_file(&input).ok();
    // The same decode the wav path produces, which ffmpeg's own decode matched
    // in `m1073_g711_adpcm.rs`.
    assert_eq!(written, read_fixture("sine_8k_mono_mulaw_decoded.s16le"));
}
