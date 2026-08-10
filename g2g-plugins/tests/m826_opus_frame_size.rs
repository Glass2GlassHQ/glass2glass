//! M826: `OpusEnc` frame duration + complexity. Opus codes frames of 2.5, 5, 10,
//! 20, 40 or 60 ms; the encoder accumulates PCM to whichever the `frame-size`
//! property selects, steps the output PTS by that duration, and hands libopus a
//! matching sample count so the packet's own TOC agrees. `complexity` is
//! libopus' quality/CPU knob (0..=10), applied live via `OPUS_SET_COMPLEXITY`.
//!
//! The duration checks are sample-exact: N whole frames in must be N packets
//! out, spaced one frame apart, decoding back to exactly the input sample count.

#![cfg(feature = "opus")]

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelinePacket, PropError, PropValue, PushOutcome,
};
use g2g_plugins::opusdec::OpusDec;
use g2g_plugins::opusenc::{OpusEnc, OpusFrameSize};
use g2g_plugins::opusparse::OPUS_RATE_HZ;

/// Captures encoded packets with their presentation timestamps.
#[derive(Default)]
struct CaptureSink {
    caps: Vec<Caps>,
    packets: Vec<(Vec<u8>, u64)>,
}
impl OutputSink for CaptureSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            match packet {
                PipelinePacket::CapsChanged(c) => self.caps.push(c),
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.packets.push((s.to_vec(), f.timing.pts_ns));
                    }
                }
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// `n` samples/channel of a tone with a little added detail (so complexity has
/// something to work on), interleaved S16LE across `channels`.
fn tone_pcm(channels: u8, n: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(n * channels as usize * 2);
    for i in 0..n {
        let t = i as f32;
        // 480 Hz fundamental plus a 3.7 kHz partial and a deterministic dither,
        // which keeps the signal from being trivially cheap to code.
        let s = (t * core::f32::consts::TAU / 100.0).sin() * 9_000.0
            + (t * core::f32::consts::TAU / 13.0).sin() * 2_500.0
            + ((i.wrapping_mul(2_654_435_761) >> 20) as i32 % 400 - 200) as f32;
        let s = s as i16;
        for _ in 0..channels {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
    }
    bytes
}

fn pcm_caps(channels: u8) -> Caps {
    Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels,
        sample_rate: OPUS_RATE_HZ,
    }
}

fn pcm_frame(pcm: Vec<u8>) -> Frame {
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(pcm.into_boxed_slice())),
        FrameTiming {
            pts_ns: 0,
            ..FrameTiming::default()
        },
        0,
    )
}

/// Encode `pcm` in one push at `frame_size`, without an EOS flush (so only whole
/// frames come out).
async fn encode_at(frame_size: OpusFrameSize, channels: u8, pcm: Vec<u8>) -> CaptureSink {
    let mut enc = OpusEnc::new()
        .with_bitrate(96_000)
        .with_frame_size(frame_size);
    enc.configure_pipeline(&pcm_caps(channels)).unwrap();
    let mut sink = CaptureSink::default();
    enc.process(PipelinePacket::DataFrame(pcm_frame(pcm)), &mut sink)
        .await
        .unwrap();
    sink
}

/// Decode Opus packets back to PCM bytes, returning the per-packet byte lengths.
async fn decode_lengths(channels: u8, packets: &[(Vec<u8>, u64)]) -> Vec<usize> {
    let mut dec = OpusDec::new();
    dec.configure_pipeline(&Caps::Audio {
        format: AudioFormat::Opus,
        channels,
        sample_rate: OPUS_RATE_HZ,
    })
    .unwrap();

    #[derive(Default)]
    struct PcmSink(Vec<usize>);
    impl OutputSink for PcmSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            core::task::Poll::Ready({
                if let PipelinePacket::DataFrame(f) = packet {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.0.push(s.len());
                    }
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    let mut sink = PcmSink::default();
    for (data, _) in packets {
        let f = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(data.clone().into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        dec.process(PipelinePacket::DataFrame(f), &mut sink)
            .await
            .unwrap();
    }
    sink.0
}

/// Every supported frame size codes the requested duration: the packet count,
/// the PTS spacing and the decoded sample count all follow `frame-size`, exactly.
#[tokio::test]
async fn every_frame_size_codes_its_own_duration_sample_exactly() {
    let channels = 2;
    for frame_size in OpusFrameSize::ALL {
        let samples = frame_size.samples();
        // Six whole frames, so nothing is left buffered.
        let frames = 6;
        let pcm = tone_pcm(channels, samples * frames);
        let enc = encode_at(frame_size, channels, pcm).await;

        assert_eq!(
            enc.packets.len(),
            frames,
            "{frame_size:?}: one packet per {samples}-sample frame"
        );
        assert_eq!(
            enc.caps,
            vec![Caps::Audio {
                format: AudioFormat::Opus,
                channels,
                sample_rate: OPUS_RATE_HZ
            }],
            "{frame_size:?}: Opus caps announced once",
        );
        let want_pts: Vec<u64> = (0..frames as u64).map(|i| i * frame_size.nanos()).collect();
        let got_pts: Vec<u64> = enc.packets.iter().map(|(_, pts)| *pts).collect();
        assert_eq!(
            got_pts,
            want_pts,
            "{frame_size:?}: PTS steps by one frame duration ({} ns)",
            frame_size.nanos()
        );

        // The decoder reads the duration out of each packet's TOC: every packet
        // must decode to exactly the frame's sample count, no rounding slack.
        let want_bytes = samples * channels as usize * 2;
        let lengths = decode_lengths(channels, &enc.packets).await;
        assert_eq!(
            lengths,
            vec![want_bytes; frames],
            "{frame_size:?}: each packet decodes to {samples} samples/channel"
        );
    }
}

/// The frame durations are exact divisors of the 48 kHz clock, 2.5 ms included:
/// each covers a whole number of samples and of nanoseconds, so a packet count
/// times a duration is never off by a rounding step.
#[test]
fn frame_durations_are_exact_at_48_khz() {
    for frame_size in OpusFrameSize::ALL {
        assert_eq!(
            frame_size.samples() as u64 * 1_000_000_000 / OPUS_RATE_HZ as u64,
            frame_size.nanos(),
            "{frame_size:?}: sample count and nanosecond duration agree"
        );
    }
    assert_eq!(OpusFrameSize::Ms2_5.samples(), 120);
    assert_eq!(OpusFrameSize::Ms60.samples(), 2_880);

    assert_eq!(OpusFrameSize::default(), OpusFrameSize::Ms20);
}

/// The encoder lookahead the containers declare as pre-skip is a property of the
/// encoder, not of the frame size, so switching `frame-size` does not silently
/// change what mp4 `dOps` / mkv CodecDelay must carry.
#[test]
fn lookahead_is_the_same_at_every_frame_size() {
    let mut lookaheads = Vec::new();
    for frame_size in OpusFrameSize::ALL {
        let mut enc = OpusEnc::new().with_frame_size(frame_size);
        enc.configure_pipeline(&pcm_caps(2)).unwrap();
        lookaheads.push(
            enc.lookahead()
                .expect("configured encoder reports lookahead"),
        );
    }
    assert_eq!(
        lookaheads,
        vec![lookaheads[0]; OpusFrameSize::ALL.len()],
        "pre-skip is frame-size independent"
    );
}

/// A short frame size still drains a large PCM push completely, and the EOS
/// flush pads only the partial tail.
#[tokio::test]
async fn eos_flush_pads_the_tail_of_a_short_frame_size() {
    let channels = 1;
    let frame_size = OpusFrameSize::Ms5;
    let samples = frame_size.samples();
    // Ten whole 5 ms frames plus half of an eleventh.
    let pcm = tone_pcm(channels, samples * 10 + samples / 2);

    let mut enc = OpusEnc::new()
        .with_bitrate(64_000)
        .with_frame_size(frame_size);
    enc.configure_pipeline(&pcm_caps(channels)).unwrap();
    let mut sink = CaptureSink::default();
    enc.process(PipelinePacket::DataFrame(pcm_frame(pcm)), &mut sink)
        .await
        .unwrap();
    assert_eq!(sink.packets.len(), 10, "whole frames emitted as they fill");
    enc.process(PipelinePacket::Eos, &mut sink).await.unwrap();
    assert_eq!(sink.packets.len(), 11, "the padded tail is the 11th packet");
    assert_eq!(
        sink.packets[10].1,
        10 * frame_size.nanos(),
        "the flushed packet continues the same timeline"
    );

    let lengths = decode_lengths(channels, &sink.packets).await;
    assert_eq!(
        lengths,
        vec![samples * 2; 11],
        "every packet, tail included, is one whole 5 ms frame"
    );
}

/// `frame-size` round-trips through the property pair and rejects any value Opus
/// has no frame of, rather than coercing it to a neighbour.
#[test]
fn frame_size_property_round_trips_and_rejects_invalid_values() {
    let mut e = OpusEnc::new();
    assert_eq!(e.get_property("frame-size"), Some(PropValue::Uint(20)));

    for (value, want) in [
        (2u64, OpusFrameSize::Ms2_5),
        (5, OpusFrameSize::Ms5),
        (10, OpusFrameSize::Ms10),
        (20, OpusFrameSize::Ms20),
        (40, OpusFrameSize::Ms40),
        (60, OpusFrameSize::Ms60),
    ] {
        e.set_property("frame-size", PropValue::Uint(value))
            .unwrap();
        assert_eq!(e.frame_size(), want, "frame-size={value} selects {want:?}");
        assert_eq!(e.get_property("frame-size"), Some(PropValue::Uint(value)));
    }

    // 2 is 2.5 ms, so 3 is not "round it down", and 30 / 25 / 0 are not frames.
    for bad in [0u64, 1, 3, 15, 25, 30, 50, 120] {
        assert_eq!(
            e.set_property("frame-size", PropValue::Uint(bad)),
            Err(PropError::Value),
            "frame-size={bad} is rejected, not coerced"
        );
    }
    assert_eq!(
        e.set_property("frame-size", PropValue::Str("20".into())),
        Err(PropError::Type)
    );
    // The rejected sets left the last good value in place.
    assert_eq!(e.get_property("frame-size"), Some(PropValue::Uint(60)));
}

/// `complexity` round-trips, rejects out-of-range values, and reaches the live
/// libopus encoder: the read-back comes from `OPUS_GET_COMPLEXITY`, not our field.
#[test]
fn complexity_property_reaches_libopus_and_rejects_invalid_values() {
    let mut e = OpusEnc::new();
    assert_eq!(
        e.get_property("complexity"),
        Some(PropValue::Uint(9)),
        "libopus' own default"
    );

    e.configure_pipeline(&pcm_caps(2)).unwrap();
    assert_eq!(e.complexity(), 9, "the built encoder runs at the default");

    for value in 0..=10u64 {
        e.set_property("complexity", PropValue::Uint(value))
            .unwrap();
        assert_eq!(
            e.complexity() as u64,
            value,
            "libopus reports back complexity {value}"
        );
        assert_eq!(e.get_property("complexity"), Some(PropValue::Uint(value)));
    }

    for bad in [11u64, 100, u64::from(u32::MAX)] {
        assert_eq!(
            e.set_property("complexity", PropValue::Uint(bad)),
            Err(PropError::Value),
            "complexity={bad} is rejected"
        );
    }
    assert_eq!(
        e.set_property("complexity", PropValue::Bool(true)),
        Err(PropError::Type)
    );
    assert_eq!(
        e.complexity(),
        10,
        "a rejected set does not disturb libopus"
    );

    // The builder is the same knob, clamped rather than rejecting.
    assert_eq!(OpusEnc::new().with_complexity(3).complexity(), 3);
    assert_eq!(OpusEnc::new().with_complexity(200).complexity(), 10);
}

/// Complexity is a real encode setting, not a stored number: the same PCM coded
/// at 0 and at 10 produces different bytes.
#[tokio::test]
async fn complexity_changes_the_encoded_bytestream() {
    async fn encode_at_complexity(complexity: u8) -> Vec<Vec<u8>> {
        let mut enc = OpusEnc::new()
            .with_bitrate(64_000)
            .with_complexity(complexity);
        enc.configure_pipeline(&pcm_caps(2)).unwrap();
        let mut sink = CaptureSink::default();
        let pcm = tone_pcm(2, OpusFrameSize::Ms20.samples() * 8);
        enc.process(PipelinePacket::DataFrame(pcm_frame(pcm)), &mut sink)
            .await
            .unwrap();
        sink.packets.into_iter().map(|(p, _)| p).collect()
    }

    let cheap = encode_at_complexity(0).await;
    let best = encode_at_complexity(10).await;
    assert_eq!(cheap.len(), best.len(), "same framing either way");
    assert!(!cheap.is_empty(), "packets were produced");
    assert_ne!(
        cheap, best,
        "complexity 0 and 10 must not encode to identical bytes"
    );
}
