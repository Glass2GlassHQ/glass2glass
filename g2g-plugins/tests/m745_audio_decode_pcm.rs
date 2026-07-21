//! M745: audio decode-to-PCM from the launch CLI.
//!
//! Three things had to line up for a decode-to-raw-PCM line to negotiate and
//! run from `parse_launch`:
//!   - an audio decoder fixates a concrete PCM output (rate / format) even when
//!     the demuxer only knows the channel count once it parses the stream, so
//!     `audioconvert` has something to negotiate against;
//!   - `audioconvert` is caps-driven, taking its output format / channels from a
//!     downstream capsfilter (a mono `channels=1` sink pin), else passthrough;
//!   - a raw-PCM file sink resolves from launch (`wavsink`, and `filesink`
//!     accepts `audio/x-raw`).
//!
//! `default_registry` is `std`-gated and the decoder is the `opus` element, so
//! this file is gated on `opus`.
#![cfg(feature = "opus")]

use g2g_core::format_element::CapsConstraint;
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{AsyncElement, AudioFormat, Caps, PipelineClock, ANY_CHANNELS, ANY_SAMPLE_RATE};
use g2g_plugins::opusdec::OpusDec;
use g2g_plugins::opusparse::OPUS_RATE_HZ;
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

async fn run_line(line: &str) -> u64 {
    let reg = default_registry();
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("{line:?} should parse: {e:?}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("{line:?} should run: {e:?}"))
        .frames_consumed
}

fn tmp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(alloc_name(name))
}

fn alloc_name(name: &str) -> String {
    format!("g2g_m745_{}_{}", std::process::id(), name)
}

/// The exact caps `OggDemux` advertises for Opus before it parses `OpusHead`:
/// the channel count and rate are placeholders, refined later via `CapsChanged`.
fn opus_demux_placeholder() -> Caps {
    Caps::Audio {
        format: AudioFormat::Opus,
        channels: ANY_CHANNELS,
        sample_rate: ANY_SAMPLE_RATE,
    }
}

/// The regression at the heart of the bug: `OpusDec` used to derive an empty
/// output set for the demuxer's `ANY_CHANNELS` placeholder, leaving the edge
/// unconstrained. It must now derive a `PcmS16Le` output that fixates to a
/// concrete rate / channel count the downstream can pin against.
#[test]
fn opusdec_derives_fixable_pcm_from_demux_placeholder() {
    let dec = OpusDec::new();
    let CapsConstraint::DerivedOutput(derive) = dec.caps_constraint_as_transform() else {
        panic!("OpusDec should expose a DerivedOutput transform constraint");
    };
    let out = derive(&opus_demux_placeholder());
    assert!(
        !out.is_empty(),
        "placeholder Opus input must still derive a PCM output"
    );
    let fixed = out
        .fixate()
        .expect("derived output must fixate to a concrete caps");
    match fixed {
        Caps::Audio {
            format,
            channels,
            sample_rate,
        } => {
            assert_eq!(format, AudioFormat::PcmS16Le);
            assert_eq!(sample_rate, OPUS_RATE_HZ, "Opus always decodes at 48 kHz");
            assert!(channels > 0, "channels fixate to a concrete count");
        }
        other => panic!("expected fixated PCM audio, got {other:?}"),
    }
}

/// `wavsink` resolves from launch and writes a RIFF/WAVE file (its `location`
/// property is applied). Previously the element compiled but was not registered.
#[tokio::test]
async fn wavsink_resolves_from_launch_and_writes_pcm() {
    let path = tmp("wav.wav");
    let _ = std::fs::remove_file(&path);
    let line = format!(
        "audiotestsrc num-buffers=5 ! audioconvert ! wavsink location={}",
        path.display()
    );
    assert_eq!(run_line(&line).await, 5, "{line}");
    let bytes = std::fs::read(&path).expect("wav written");
    assert!(bytes.len() > 44, "header + PCM data present");
    assert_eq!(&bytes[..4], b"RIFF");
    let _ = std::fs::remove_file(&path);
}

/// The downstream half of the decode-to-PCM line: a caps-driven `audioconvert`
/// takes a mono `channels=1` capsfilter and writes raw PCM to `filesink`
/// (`filesink` accepts `audio/x-raw`). A stereo source proves the downmix.
#[tokio::test]
async fn audioconvert_caps_driven_mono_to_filesink() {
    let path = tmp("mono.raw");
    let _ = std::fs::remove_file(&path);
    let line = format!(
        "audiotestsrc num-buffers=5 channels=2 ! audioconvert \
         ! audio/x-raw,format=S16LE,rate=48000,channels=1 ! filesink location={}",
        path.display()
    );
    assert_eq!(run_line(&line).await, 5, "{line}");
    let bytes = std::fs::read(&path).expect("raw pcm written");
    assert!(!bytes.is_empty(), "raw pcm is non-empty");
    // mono S16 is 2 bytes per frame.
    assert_eq!(bytes.len() % 2, 0, "whole S16 mono samples");
    let _ = std::fs::remove_file(&path);
}

/// End-to-end Opus decode-to-S16LE through the launch parser: the encoder ->
/// parser -> decoder -> caps-driven convert -> mono capsfilter -> WAV sink chain
/// negotiates (OpusDec fixates its output) and flows. A stereo source exercises
/// the downmix to the mono capsfilter.
#[tokio::test]
async fn opus_decode_to_s16le_negotiates_and_flows() {
    let path = tmp("opus.wav");
    let _ = std::fs::remove_file(&path);
    let line = format!(
        "audiotestsrc num-buffers=10 channels=2 ! opusenc ! opusparse ! opusdec \
         ! audioconvert ! audio/x-raw,format=S16LE,rate=48000,channels=1 ! wavsink location={}",
        path.display()
    );
    // The Opus codec buffers, so decoded buffers do not map 1:1 to the 10 fed;
    // what matters is the chain negotiates and decoded PCM reaches the sink.
    assert!(run_line(&line).await > 0, "{line}");
    let bytes = std::fs::read(&path).expect("wav written");
    assert!(bytes.len() > 44, "decoded PCM reached the sink");
    // mono 48 kHz S16 header.
    assert_eq!(u16::from_le_bytes([bytes[22], bytes[23]]), 1, "mono");
    assert_eq!(
        u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]),
        OPUS_RATE_HZ,
        "48 kHz"
    );
    let _ = std::fs::remove_file(&path);
}
