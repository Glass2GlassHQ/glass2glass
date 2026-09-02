//! M1073: the G.711 (`mulawenc` / `mulawdec` / `alawenc` / `alawdec`) and IMA
//! ADPCM (`adpcmenc` / `adpcmdec`) elements, checked against ffmpeg in both
//! directions. The codec math itself is proven bit-exact in `g2g-mcu`
//! (`m638_g711.rs`, `m639_adpcm.rs`), so a mismatch here is a wiring bug.
//!
//! The fixtures are one second of a 440 Hz sine at 8 kHz mono, made with:
//!
//! ```text
//! ffmpeg -f lavfi -i "sine=frequency=440:sample_rate=8000:duration=1" -c:a pcm_s16le sine_8k_mono_s16le.wav
//! ffmpeg -i sine_8k_mono_s16le.wav -c:a pcm_mulaw      sine_8k_mono_mulaw.wav
//! ffmpeg -i sine_8k_mono_s16le.wav -c:a pcm_alaw       sine_8k_mono_alaw.wav
//! ffmpeg -i sine_8k_mono_s16le.wav -c:a adpcm_ima_wav  sine_8k_mono_imaadpcm.wav
//! ffmpeg -i sine_8k_mono_mulaw.wav     -f s16le sine_8k_mono_mulaw_decoded.s16le
//! ffmpeg -i sine_8k_mono_alaw.wav      -f s16le sine_8k_mono_alaw_decoded.s16le
//! ffmpeg -i sine_8k_mono_imaadpcm.wav  -f s16le sine_8k_mono_imaadpcm_decoded.s16le
//! ```
//!
//! `default_registry` is `std`-gated, so this file is too.
#![cfg(feature = "std")]

use std::path::PathBuf;

use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{is_raw_audio, parse_launch, run_graph};
use g2g_core::{
    AsyncElement, AudioFormat, Caps, Frame, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelineClock, PipelinePacket, PushOutcome, ANY_CHANNELS, ANY_SAMPLE_RATE,
};
use g2g_plugins::g711::{MulawDec, G711_CLOCK_RATE_HZ, G711_DEFAULT_CHANNELS};
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// `RIFF` + size + `WAVE`, then 4-byte id + 4-byte size per chunk.
const RIFF_HEADER_LEN: usize = 12;
const CHUNK_HEADER_LEN: usize = 8;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn read_fixture(name: &str) -> Vec<u8> {
    std::fs::read(fixture(name)).unwrap_or_else(|e| panic!("fixture {name}: {e}"))
}

/// The `data` chunk of a RIFF/WAVE file: the coded bytes ffmpeg wrote, which is
/// what an encoder element has to reproduce.
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

/// Run `filesrc location=<input> ! wavparse ! <element> ! filesink` and return
/// what the sink wrote.
async fn transcode(input: &str, element: &str) -> Vec<u8> {
    let tag: String = element
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .collect();
    let output = std::env::temp_dir().join(format!("m1073_{tag}.raw"));
    let line = format!(
        "filesrc location={} ! wavparse ! {element} ! filesink location={}",
        fixture(input).display(),
        output.display()
    );
    let reg = default_registry();
    let graph = parse_launch(&reg, &line).expect("pipeline parses");
    run_graph(graph, &ZeroClock, 4)
        .await
        .expect("pipeline runs");
    let bytes = std::fs::read(&output).expect("the sink wrote a file");
    std::fs::remove_file(&output).ok();
    bytes
}

#[tokio::test]
async fn mulaw_fixture_decodes_bit_exact() {
    let decoded = transcode("sine_8k_mono_mulaw.wav", "mulawdec").await;
    assert_eq!(decoded, read_fixture("sine_8k_mono_mulaw_decoded.s16le"));
}

#[tokio::test]
async fn alaw_fixture_decodes_bit_exact() {
    let decoded = transcode("sine_8k_mono_alaw.wav", "alawdec").await;
    assert_eq!(decoded, read_fixture("sine_8k_mono_alaw_decoded.s16le"));
}

#[tokio::test]
async fn adpcm_fixture_decodes_bit_exact() {
    let decoded = transcode("sine_8k_mono_imaadpcm.wav", "adpcmdec").await;
    assert_eq!(decoded, read_fixture("sine_8k_mono_imaadpcm_decoded.s16le"));
}

#[tokio::test]
async fn mulaw_encode_matches_ffmpeg() {
    let coded = transcode("sine_8k_mono_s16le.wav", "mulawenc").await;
    let reference = read_fixture("sine_8k_mono_mulaw.wav");
    assert_eq!(coded, wav_data_chunk(&reference));
}

#[tokio::test]
async fn alaw_encode_matches_ffmpeg() {
    let coded = transcode("sine_8k_mono_s16le.wav", "alawenc").await;
    let reference = read_fixture("sine_8k_mono_alaw.wav");
    assert_eq!(coded, wav_data_chunk(&reference));
}

/// The tail block is padded with silence, exactly as ffmpeg's encoder flushes
/// its own, so the whole coded stream matches byte for byte. The
/// `audioconvert` pins mono: a WAV's channel count is not known until the
/// `fmt ` chunk is read, so negotiation would otherwise fixate to stereo.
#[tokio::test]
async fn adpcm_encode_matches_ffmpeg() {
    let coded = transcode(
        "sine_8k_mono_s16le.wav",
        "audioconvert channels=1 ! adpcmenc",
    )
    .await;
    let reference = read_fixture("sine_8k_mono_imaadpcm.wav");
    assert_eq!(coded, wav_data_chunk(&reference));
}

async fn round_trip(line: &str) -> u64 {
    let reg = default_registry();
    let graph = parse_launch(&reg, line).expect("pipeline parses");
    run_graph(graph, &ZeroClock, 4)
        .await
        .expect("pipeline runs")
        .frames_consumed
}

#[tokio::test]
async fn mulaw_round_trips_through_a_launch_line() {
    let frames =
        round_trip("audiotestsrc num-buffers=10 ! audioconvert ! mulawenc ! mulawdec ! fakesink")
            .await;
    assert_eq!(frames, 10);
}

#[tokio::test]
async fn alaw_round_trips_through_a_launch_line() {
    let frames =
        round_trip("audiotestsrc num-buffers=10 ! audioconvert ! alawenc ! alawdec ! fakesink")
            .await;
    assert_eq!(frames, 10);
}

/// ADPCM packs samples into fixed-size blocks, so the frame count downstream is
/// set by the block size, not by how many buffers went in.
#[tokio::test]
async fn adpcm_round_trips_through_a_launch_line() {
    let frames = round_trip(
        "audiotestsrc num-buffers=10 ! audioconvert channels=1 ! adpcmenc ! adpcmdec ! fakesink",
    )
    .await;
    assert!(frames > 0, "decoded blocks reached the sink");
}

#[derive(Default)]
struct CollectSink {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for CollectSink {
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

/// An RTSP source that has not read the layout yet negotiates on the sentinels,
/// so the decoder has to announce the concrete telephony layout before the PCM
/// a sink would try to play.
#[tokio::test]
async fn a_decoder_on_sentinel_caps_announces_the_telephony_layout() {
    let mut decoder = MulawDec::new();
    decoder
        .configure_pipeline(&Caps::Audio {
            format: AudioFormat::Mulaw,
            channels: ANY_CHANNELS,
            sample_rate: ANY_SAMPLE_RATE,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        })
        .expect("the sentinel layout is accepted");
    let mut sink = CollectSink::default();
    let frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(vec![0xFFu8; 4].into_boxed_slice())),
        FrameTiming::default(),
        0,
    );
    decoder
        .process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
        .expect("decodes");
    assert!(matches!(
        sink.packets.first(),
        Some(PipelinePacket::CapsChanged(Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: G711_DEFAULT_CHANNELS,
            sample_rate: G711_CLOCK_RATE_HZ,
            ..
        }))
    ));
    let PipelinePacket::DataFrame(out) = &sink.packets[1] else {
        panic!("the caps come before the first frame");
    };
    assert_eq!(
        out.domain.require_system_slice("test").unwrap().len(),
        8,
        "one byte in, one 16-bit sample out"
    );
}

/// `rtspsrc ! decodebin` on a PCMU camera: the SDP names a concrete 8 kHz mono
/// stream, and the search has to reach PCM from it.
#[test]
fn decodebin_reaches_pcm_from_a_concrete_g711_stream() {
    let reg = default_registry();
    for (format, decoder) in [
        (AudioFormat::Mulaw, "mulawdec"),
        (AudioFormat::Alaw, "alawdec"),
    ] {
        let coded = Caps::Audio {
            format,
            channels: G711_DEFAULT_CHANNELS,
            sample_rate: G711_CLOCK_RATE_HZ,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        let chain = reg
            .autoplug_names(&coded, &is_raw_audio, 4)
            .unwrap_or_else(|| panic!("{decoder} decodes {format:?}"));
        assert_eq!(chain, vec![decoder]);
    }
}
