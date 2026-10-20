//! `WhisperTranscribe` against the real whisper.cpp runtime on its own jfk.wav
//! sample with the quantized tiny.en model, both fetched by
//! `tools/whisper-fixture.sh` into the gitignored `fixtures/whisper`; skips
//! when absent. Run:
//!
//! ```powershell
//! cargo test -p g2g-ml --features whisper --test whisper_transcribe
//! ```
//!
//! With `whisper-cuda` or `whisper-vulkan` the same tests run on the GPU.

#![cfg(feature = "whisper")]

use std::path::PathBuf;

use g2g_core::element::{AsyncElement, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{AudioFormat, Caps, ChannelLayout, G2gError, PropValue};
use g2g_ml::whispertranscribe::{WhisperTranscribe, SAMPLE_RATE};

const MODEL: &str = "ggml-tiny.en-q5_1.bin";
const SAMPLE: &str = "jfk.wav";
/// The sample is JFK's "ask not" line; the tiny model gets this part verbatim.
const EXPECTED_PHRASE: &str = "ask not what your country can do for you";
const NS_PER_SECOND: u64 = 1_000_000_000;
/// The sample runs 11 s, so a 4 s window yields two full windows plus a tail.
const SHORT_WINDOW_SECONDS: f64 = 4.0;
const WINDOW_LONGER_THAN_SAMPLE_SECONDS: f64 = 30.0;
/// Input frames of one second each.
const FRAME_SAMPLES: usize = SAMPLE_RATE as usize;
const S16_FULL_SCALE: f32 = 32768.0;
const RIFF_HEADER_BYTES: usize = 12;
const CHUNK_HEADER_BYTES: usize = 8;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/whisper")
}

/// The fixture paths, or `None` (skip) when `tools/whisper-fixture.sh` has not run.
fn fixtures() -> Option<(PathBuf, Vec<f32>)> {
    let model = fixture_dir().join(MODEL);
    let sample = fixture_dir().join(SAMPLE);
    if !model.exists() || !sample.exists() {
        eprintln!(
            "whisper fixtures absent ({}); run tools/whisper-fixture.sh. skipping.",
            fixture_dir().display()
        );
        return None;
    }
    let wav = std::fs::read(sample).expect("read sample");
    Some((model, wav_s16_mono_to_f32(&wav)))
}

/// The `data` chunk of a 16-bit PCM RIFF file as f32 samples.
fn wav_s16_mono_to_f32(wav: &[u8]) -> Vec<f32> {
    assert_eq!(&wav[..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    let mut offset = RIFF_HEADER_BYTES;
    while offset + CHUNK_HEADER_BYTES <= wav.len() {
        let id = &wav[offset..offset + 4];
        let size = u32::from_le_bytes(wav[offset + 4..offset + 8].try_into().unwrap()) as usize;
        let body = &wav[offset + CHUNK_HEADER_BYTES..offset + CHUNK_HEADER_BYTES + size];
        if id == b"fmt " {
            let channels = u16::from_le_bytes(body[2..4].try_into().unwrap());
            let rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
            let bits = u16::from_le_bytes(body[14..16].try_into().unwrap());
            assert_eq!(
                (channels, rate, bits),
                (1, SAMPLE_RATE, 16),
                "sample format"
            );
        }
        if id == b"data" {
            return body
                .as_chunks::<2>()
                .0
                .iter()
                .map(|sample| f32::from(i16::from_le_bytes(*sample)) / S16_FULL_SCALE)
                .collect();
        }
        offset += CHUNK_HEADER_BYTES + size + (size % 2);
    }
    panic!("no data chunk");
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
        let packet = packet_slot.take().expect("poll_push without a packet");
        self.packets.push(packet);
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn input_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::PcmF32Le,
        channels: 1,
        sample_rate: SAMPLE_RATE,
        channel_layout: ChannelLayout::UNSPECIFIED,
    }
}

fn audio_frame(samples: &[f32], pts_ns: u64, sequence: u64) -> Frame {
    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            duration_ns: samples.len() as u64 * NS_PER_SECOND / u64::from(SAMPLE_RATE),
            ..FrameTiming::default()
        },
        sequence,
    )
}

/// Feed the sample in one-second frames starting at `start_ns`, then EOS, and
/// return every (text, pts, duration) the element emitted after its caps.
async fn run(
    element: &mut WhisperTranscribe,
    samples: &[f32],
    start_ns: u64,
) -> Vec<(String, u64, u64)> {
    element
        .configure_pipeline(&input_caps())
        .expect("configure");
    let mut sink = Collect::default();
    for (index, chunk) in samples.chunks(FRAME_SAMPLES).enumerate() {
        let pts_ns = start_ns + index as u64 * NS_PER_SECOND;
        element
            .process(
                PipelinePacket::DataFrame(audio_frame(chunk, pts_ns, index as u64)),
                &mut sink,
            )
            .await
            .expect("frame");
    }
    element
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("eos");

    let mut packets = sink.packets.into_iter();
    match packets.next() {
        Some(PipelinePacket::CapsChanged(caps)) => assert_eq!(caps, WhisperTranscribe::text_caps()),
        other => panic!("text caps go out before the first segment, got {other:?}"),
    }
    packets
        .map(|packet| {
            let PipelinePacket::DataFrame(frame) = packet else {
                panic!("expected only text frames after the caps, got {packet:?}");
            };
            let text = String::from_utf8(frame.domain.as_system_slice().unwrap().to_vec()).unwrap();
            (text, frame.timing.pts_ns, frame.timing.duration_ns)
        })
        .collect()
}

#[test]
fn only_16k_mono_f32_is_accepted() {
    let element = WhisperTranscribe::new();
    assert_eq!(
        element.intercept_caps(&input_caps()),
        Ok(WhisperTranscribe::text_caps())
    );
    let stereo = Caps::Audio {
        format: AudioFormat::PcmF32Le,
        channels: 2,
        sample_rate: SAMPLE_RATE,
        channel_layout: ChannelLayout::UNSPECIFIED,
    };
    assert_eq!(element.intercept_caps(&stereo), Err(G2gError::CapsMismatch));
    let s16 = Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: 1,
        sample_rate: SAMPLE_RATE,
        channel_layout: ChannelLayout::UNSPECIFIED,
    };
    assert_eq!(element.intercept_caps(&s16), Err(G2gError::CapsMismatch));
}

fn transcript(segments: &[(String, u64, u64)]) -> String {
    segments
        .iter()
        .map(|(text, _, _)| text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

#[tokio::test]
async fn the_whole_sample_is_transcribed_at_eos() {
    let Some((model, samples)) = fixtures() else {
        return;
    };
    let mut element = WhisperTranscribe::from_file(model.to_str().unwrap())
        .expect("model loads")
        .with_chunk_seconds(WINDOW_LONGER_THAN_SAMPLE_SECONDS);
    let segments = run(&mut element, &samples, 0).await;
    assert!(!segments.is_empty(), "no segments");
    let text = transcript(&segments);
    assert!(text.contains(EXPECTED_PHRASE), "transcript was: {text}");
    let sample_ns = samples.len() as u64 * NS_PER_SECOND / u64::from(SAMPLE_RATE);
    for (text, pts_ns, duration_ns) in &segments {
        assert!(
            *pts_ns < sample_ns,
            "{text:?} starts past the end of the audio"
        );
        assert!(*duration_ns > 0, "{text:?} has no duration");
    }
}

#[tokio::test]
async fn windows_are_transcribed_as_they_fill_and_stamped_on_the_input_timeline() {
    let Some((model, samples)) = fixtures() else {
        return;
    };
    let start_ns = 100 * NS_PER_SECOND;
    let window_ns = (SHORT_WINDOW_SECONDS * NS_PER_SECOND as f64) as u64;
    let mut element = WhisperTranscribe::from_file(model.to_str().unwrap())
        .expect("model loads")
        .with_chunk_seconds(SHORT_WINDOW_SECONDS);
    let segments = run(&mut element, &samples, start_ns).await;
    assert!(
        segments.len() >= 2,
        "one segment per window at least: {segments:?}"
    );
    let first_pts = segments[0].1;
    assert!(
        (start_ns..start_ns + window_ns).contains(&first_pts),
        "first segment {first_pts} is not inside the first window from {start_ns}"
    );
    assert!(
        segments
            .iter()
            .any(|(_, pts_ns, _)| *pts_ns >= start_ns + window_ns),
        "no segment carries the second window's offset: {segments:?}"
    );
    assert!(
        segments.windows(2).all(|pair| pair[0].1 <= pair[1].1),
        "segment timestamps go backwards: {segments:?}"
    );
    let text = transcript(&segments);
    assert!(text.contains("country"), "transcript was: {text}");
}

#[test]
fn properties_round_trip_and_a_bad_model_is_a_bad_value() {
    let mut element = WhisperTranscribe::new();
    assert_eq!(
        element.set_property("model", PropValue::Str("/nonexistent/nope.bin".into())),
        Err(g2g_core::PropError::Value)
    );
    element
        .set_property("language", PropValue::Str("auto".into()))
        .unwrap();
    element
        .set_property("chunk-seconds", PropValue::Double(2.5))
        .unwrap();
    element
        .set_property("translate", PropValue::Bool(true))
        .unwrap();
    element.set_property("threads", PropValue::Uint(2)).unwrap();
    assert_eq!(
        element.set_property("chunk-seconds", PropValue::Double(0.0)),
        Err(g2g_core::PropError::Value)
    );
    assert_eq!(
        element.get_property("language"),
        Some(PropValue::Str("auto".into()))
    );
    assert_eq!(
        element.get_property("chunk-seconds"),
        Some(PropValue::Double(2.5))
    );
    assert_eq!(
        element.get_property("translate"),
        Some(PropValue::Bool(true))
    );
    assert_eq!(element.get_property("threads"), Some(PropValue::Uint(2)));
    assert!(
        matches!(
            element.configure_pipeline(&input_caps()),
            Err(G2gError::NotConfigured)
        ),
        "no model, no run"
    );
}

#[cfg(any(feature = "whisper-cuda", feature = "whisper-vulkan"))]
#[test]
fn gpu_defaults_follow_whisper_rs() {
    let defaults = whisper_rs::WhisperContextParameters::default();
    let element = WhisperTranscribe::new();
    let declared = |name: &str| {
        element
            .properties()
            .iter()
            .find(|spec| spec.name == name)
            .and_then(|spec| spec.default)
            .unwrap_or_else(|| panic!("{name} declares no default"))
    };
    assert_eq!(
        element.get_property("use-gpu"),
        Some(PropValue::Bool(defaults.use_gpu))
    );
    assert_eq!(declared("use-gpu"), defaults.use_gpu.to_string());
    assert_eq!(
        element.get_property("gpu-device"),
        Some(PropValue::Uint(u64::try_from(defaults.gpu_device).unwrap()))
    );
    assert_eq!(declared("gpu-device"), defaults.gpu_device.to_string());
}

#[cfg(any(feature = "whisper-cuda", feature = "whisper-vulkan"))]
#[tokio::test]
async fn turning_the_gpu_off_after_load_reloads_on_the_cpu() {
    let Some((model, samples)) = fixtures() else {
        return;
    };
    let mut element = WhisperTranscribe::from_file(model.to_str().unwrap())
        .expect("model loads")
        .with_chunk_seconds(WINDOW_LONGER_THAN_SAMPLE_SECONDS);
    element
        .set_property("use-gpu", PropValue::Bool(false))
        .expect("reloads");
    assert_eq!(
        element.get_property("use-gpu"),
        Some(PropValue::Bool(false))
    );
    let text = transcript(&run(&mut element, &samples, 0).await);
    assert!(text.contains(EXPECTED_PHRASE), "transcript was: {text}");
}
