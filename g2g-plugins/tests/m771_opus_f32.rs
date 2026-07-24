//! M771: Opus float (F32) PCM in / out. `OpusEnc` accepts `PcmF32Le` input
//! (libopus' `opus_encode_float`, no S16 quantize) and `OpusDec` emits
//! `PcmF32Le` when negotiated (`opus_decode_float`). Round-trips a float tone
//! end to end and checks the float decode agrees with the S16 decode of the
//! same packets to within one quantization step.

#![cfg(feature = "opus")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelinePacket, PushOutcome,
};
use g2g_plugins::opusdec::OpusDec;
use g2g_plugins::opusenc::OpusEnc;
use g2g_plugins::opusparse::OPUS_RATE_HZ;

#[derive(Default)]
struct CaptureSink {
    caps: Vec<Caps>,
    frames: Vec<Vec<u8>>,
}
impl OutputSink for CaptureSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::CapsChanged(c) => self.caps.push(c),
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.frames.push(s.to_vec());
                    }
                }
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn frame(data: Vec<u8>, pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            ..FrameTiming::default()
        },
        0,
    ))
}

fn audio_caps(format: AudioFormat) -> Caps {
    Caps::Audio {
        format,
        channels: 2,
        sample_rate: OPUS_RATE_HZ,
    }
}

/// `n` samples/channel of a ~480 Hz stereo tone as interleaved F32LE bytes.
fn sine_f32(n: usize, amp: f32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(n * 8);
    for i in 0..n {
        let s = ((i as f32) * core::f32::consts::TAU / 100.0).sin() * amp;
        for _ in 0..2 {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
    }
    bytes
}

fn f32_samples(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Encode a float tone with `OpusEnc` on the F32 input path, returning packets.
async fn encode_f32_tone(n: usize) -> Vec<Vec<u8>> {
    let mut enc = OpusEnc::new();
    enc.configure_pipeline(&audio_caps(AudioFormat::PcmF32Le))
        .expect("F32 input accepted");
    let mut sink = CaptureSink::default();
    enc.process(frame(sine_f32(n, 0.6), 0), &mut sink)
        .await
        .expect("encode");
    enc.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("flush");
    assert!(!sink.frames.is_empty(), "packets emitted");
    sink.frames
}

/// Decode `packets` with a fresh `OpusDec` negotiated to `format`.
async fn decode_all(packets: &[Vec<u8>], format: AudioFormat) -> (Vec<Caps>, Vec<u8>) {
    let mut dec = OpusDec::new();
    dec.configure_pipeline(&audio_caps(AudioFormat::Opus))
        .expect("configure");
    dec.configure_output(&audio_caps(format))
        .expect("output format accepted");
    let mut sink = CaptureSink::default();
    for (i, p) in packets.iter().enumerate() {
        dec.process(frame(p.clone(), i as u64 * 20_000_000), &mut sink)
            .await
            .expect("decode");
    }
    (sink.caps, sink.frames.concat())
}

#[tokio::test]
async fn f32_tone_round_trips_through_float_apis() {
    // 4 whole 20 ms frames.
    let packets = encode_f32_tone(960 * 4).await;
    assert_eq!(packets.len(), 4, "one packet per 20 ms frame");

    let (caps, pcm) = decode_all(&packets, AudioFormat::PcmF32Le).await;
    assert!(
        caps.iter().any(|c| matches!(
            c,
            Caps::Audio {
                format: AudioFormat::PcmF32Le,
                channels: 2,
                sample_rate: OPUS_RATE_HZ,
            }
        )),
        "F32 output caps announced, got {caps:?}"
    );
    let samples = f32_samples(&pcm);
    assert_eq!(samples.len(), 960 * 4 * 2, "all samples decoded");
    // Loud tone in -> loud tone out (Opus is lossy; check energy, not samples).
    let rms = (samples
        .iter()
        .map(|s| (*s as f64) * (*s as f64))
        .sum::<f64>()
        / samples.len() as f64)
        .sqrt();
    assert!(
        (0.2..0.7).contains(&rms),
        "decoded energy near the 0.6-amp input (rms {rms:.3})"
    );
}

#[tokio::test]
async fn f32_decode_matches_s16_decode_within_quantization() {
    let packets = encode_f32_tone(960 * 2).await;

    let (_, f32_bytes) = decode_all(&packets, AudioFormat::PcmF32Le).await;
    let (_, s16_bytes) = decode_all(&packets, AudioFormat::PcmS16Le).await;

    let f = f32_samples(&f32_bytes);
    let s: Vec<i16> = s16_bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    assert_eq!(f.len(), s.len(), "same sample count on both paths");
    for (i, (a, b)) in f.iter().zip(&s).enumerate() {
        let b = *b as f32 / 32768.0;
        assert!(
            (a - b).abs() <= 1.5 / 32768.0,
            "sample {i}: float {a} vs s16 {b} beyond one quantization step"
        );
    }
}
