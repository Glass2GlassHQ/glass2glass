//! M177: Opus encode + decode round-trip. `OpusEnc` packs interleaved S16LE PCM
//! into Opus packets (one per 20 ms frame); `OpusParse` recovers the channel
//! count from those packets (proving they are a valid Opus stream); `OpusDec`
//! decodes them back to PCM. Opus is lossy, so the check is structural (packet
//! count, caps, decoded length) plus signal energy preserved (loud in -> loud
//! out, silence -> silence), not sample-exact.

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
use g2g_plugins::opusparse::{OpusParse, OPUS_RATE_HZ};

const FRAME_SAMPLES: usize = (OPUS_RATE_HZ as usize * 20) / 1000; // 960 / channel @ 20 ms

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
                    if let MemoryDomain::System(s) = &f.domain {
                        self.frames.push(s.as_slice().to_vec());
                    }
                }
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// `n` samples/channel of a sine tone at `amp`, interleaved S16LE across
/// `channels`. `amp == 0` yields silence.
fn sine_pcm(channels: u8, n: usize, amp: i16) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(n * channels as usize * 2);
    for i in 0..n {
        // ~480 Hz at 48 kHz: one cycle per 100 samples.
        let phase = (i as f32) * core::f32::consts::TAU / 100.0;
        let s = (phase.sin() * amp as f32) as i16;
        for _ in 0..channels {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
    }
    bytes
}

/// RMS of an interleaved S16LE buffer (all channels pooled).
fn rms(bytes: &[u8]) -> f64 {
    let samples: Vec<i16> = bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum / samples.len() as f64).sqrt()
}

fn pcm_caps(channels: u8) -> Caps {
    Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels,
        sample_rate: OPUS_RATE_HZ,
    }
}

async fn encode(channels: u8, pcm: &[u8]) -> CaptureSink {
    let mut enc = OpusEnc::new().with_bitrate(96_000);
    enc.configure_pipeline(&pcm_caps(channels)).unwrap();
    let mut sink = CaptureSink::default();
    let frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(pcm.to_vec().into_boxed_slice())),
        FrameTiming {
            pts_ns: 0,
            ..FrameTiming::default()
        },
        0,
    );
    enc.process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
        .unwrap();
    enc.process(PipelinePacket::Eos, &mut sink).await.unwrap();
    sink
}

async fn decode(channels: u8, packets: &[Vec<u8>]) -> CaptureSink {
    let mut dec = OpusDec::new();
    dec.configure_pipeline(&Caps::Audio {
        format: AudioFormat::Opus,
        channels,
        sample_rate: OPUS_RATE_HZ,
    })
    .unwrap();
    let mut sink = CaptureSink::default();
    for data in packets {
        let f = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(data.clone().into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        dec.process(PipelinePacket::DataFrame(f), &mut sink)
            .await
            .unwrap();
    }
    sink
}

#[tokio::test]
async fn stereo_roundtrip_preserves_signal_energy() {
    let channels = 2;
    // 4 full 20 ms frames of a loud tone.
    let n = FRAME_SAMPLES * 4;
    let pcm = sine_pcm(channels, n, 10_000);

    let enc = encode(channels, &pcm).await;
    assert_eq!(enc.frames.len(), 4, "one Opus packet per 20 ms frame");
    assert!(enc.frames.iter().all(|p| !p.is_empty()), "no empty packets");
    assert_eq!(
        enc.caps,
        vec![Caps::Audio {
            format: AudioFormat::Opus,
            channels,
            sample_rate: OPUS_RATE_HZ
        }],
        "encoder announces Opus caps once",
    );

    // OpusParse reads the channel count from the encoded packets: proves the
    // bytes are a structurally valid Opus elementary stream.
    let mut parse = OpusParse::new();
    parse
        .configure_pipeline(&Caps::Audio {
            format: AudioFormat::Opus,
            channels: 0,
            sample_rate: 0,
        })
        .unwrap();
    let mut psink = CaptureSink::default();
    for data in &enc.frames {
        let f = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(data.clone().into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        parse
            .process(PipelinePacket::DataFrame(f), &mut psink)
            .await
            .unwrap();
    }
    let parsed_channels = psink.caps.iter().find_map(|c| match c {
        Caps::Audio {
            format: AudioFormat::Opus,
            channels,
            ..
        } => Some(*channels),
        _ => None,
    });
    assert_eq!(
        parsed_channels,
        Some(2),
        "opusparse recovers the stereo channel count"
    );

    // Decode back to PCM and check the tone survived (lossy: energy, not samples).
    let dec = decode(channels, &enc.frames).await;
    assert_eq!(
        dec.caps,
        vec![Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels,
            sample_rate: OPUS_RATE_HZ
        }],
        "decoder announces PCM caps once",
    );
    let total: usize = dec.frames.iter().map(|f| f.len()).sum();
    assert_eq!(
        total,
        n * channels as usize * 2,
        "decoded PCM length matches the input"
    );
    let decoded: Vec<u8> = dec.frames.concat();
    let in_rms = rms(&pcm);
    let out_rms = rms(&decoded);
    assert!(
        out_rms > in_rms * 0.5,
        "decoded tone preserves most of the input energy (in={in_rms:.0}, out={out_rms:.0})",
    );
}

#[tokio::test]
async fn mono_silence_decodes_to_silence() {
    let channels = 1;
    let n = FRAME_SAMPLES * 3;
    let silence = sine_pcm(channels, n, 0); // amp 0 -> all-zero PCM

    let enc = encode(channels, &silence).await;
    assert_eq!(enc.frames.len(), 3, "one packet per frame, mono");

    let dec = decode(channels, &enc.frames).await;
    let decoded: Vec<u8> = dec.frames.concat();
    assert!(
        rms(&decoded) < 5.0,
        "silence in, silence out (rms={:.3})",
        rms(&decoded)
    );
}

/// Sink that reports a downstream bitrate target on every push (the WebRTC BWE
/// shape), so the encoder retargets live (M721).
struct BitrateSink {
    bps: u32,
    frames: usize,
}
impl OutputSink for BitrateSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if matches!(packet, PipelinePacket::DataFrame(_)) {
                self.frames += 1;
            }
            Ok(PushOutcome::Bitrate(self.bps))
        })
    }
}

/// A downstream BWE estimate retargets the live encoder: the packets emitted
/// after the signal are measurably smaller at 8 kb/s than at 96 kb/s.
#[tokio::test]
async fn bitrate_feedback_retargets_the_live_encoder() {
    async fn avg_packet_len_after_feedback(bps: u32) -> f64 {
        let mut enc = OpusEnc::new().with_bitrate(96_000);
        enc.configure_pipeline(&pcm_caps(2)).unwrap();
        let mut sink = BitrateSink { bps, frames: 0 };
        let mut cap = CaptureSink::default();
        // First batch carries the feedback back to the encoder...
        let pcm = sine_pcm(2, FRAME_SAMPLES * 5, 12_000);
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(pcm.into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        enc.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();
        assert!(sink.frames > 0, "first batch emitted packets");
        // ...the second batch encodes at the new target.
        let pcm = sine_pcm(2, FRAME_SAMPLES * 20, 12_000);
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(pcm.into_boxed_slice())),
            FrameTiming {
                pts_ns: 100_000_000,
                ..FrameTiming::default()
            },
            1,
        );
        enc.process(PipelinePacket::DataFrame(frame), &mut cap)
            .await
            .unwrap();
        let total: usize = cap.frames.iter().map(|f| f.len()).sum();
        total as f64 / cap.frames.len().max(1) as f64
    }

    let low = avg_packet_len_after_feedback(8_000).await;
    let high = avg_packet_len_after_feedback(96_000).await;
    assert!(
        low * 2.0 < high,
        "8 kb/s packets ({low:.0} B avg) must be far smaller than 96 kb/s ones ({high:.0} B avg)"
    );
}
