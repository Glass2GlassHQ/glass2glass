//! M769: `FfmpegAacEnc` end-to-end encode validation. Encodes a real 440 Hz
//! stereo tone to ADTS AAC, asserts the ADTS framing (syncword, 48 kHz stereo
//! header fields, one AU per output frame), then has the ffmpeg CLI decode the
//! stream back to PCM and checks the tone survived (duration and dominant
//! frequency via zero crossings), so the encode core is validated against the
//! reference peer, not just self-consistent.
#![cfg(all(target_os = "linux", feature = "ffmpeg"))]

use core::future::Future;
use core::pin::Pin;
use std::process::Command;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{AsyncElement, AudioFormat, Caps, G2gError, OutputSink, PushOutcome};
use g2g_plugins::ffmpegaacenc::FfmpegAacEnc;

#[derive(Default)]
struct CaptureSink {
    frames: Vec<Vec<u8>>,
}
impl OutputSink for CaptureSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.frames.push(s.to_vec());
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Interleaved stereo S16LE 440 Hz sine at 48 kHz, `n` samples per channel.
fn tone(n: usize) -> Vec<u8> {
    let mut pcm = Vec::with_capacity(n * 4);
    for i in 0..n {
        let s =
            (20_000.0 * (2.0 * core::f64::consts::PI * 440.0 * i as f64 / 48_000.0).sin()) as i16;
        pcm.extend_from_slice(&s.to_le_bytes());
        pcm.extend_from_slice(&s.to_le_bytes());
    }
    pcm
}

#[tokio::test]
async fn encodes_a_tone_ffmpeg_can_decode() {
    let mut enc = FfmpegAacEnc::new();
    enc.configure_pipeline(&Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: 2,
        sample_rate: 48_000,
    })
    .expect("open the aac encoder");

    // 0.5 s in ~10 ms chunks, so the frame-sized (1024-sample) rebuffering is
    // exercised across input boundaries.
    const TOTAL: usize = 24_000;
    let pcm = tone(TOTAL);
    let mut sink = CaptureSink::default();
    for (i, chunk) in pcm.chunks(480 * 4).enumerate() {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(chunk.to_vec().into_boxed_slice())),
            FrameTiming {
                pts_ns: (i as u64) * 10_000_000,
                ..FrameTiming::default()
            },
            i as u64,
        );
        block_send(&mut enc, PipelinePacket::DataFrame(frame), &mut sink).await;
    }
    block_send(&mut enc, PipelinePacket::Eos, &mut sink).await;

    // One ADTS AU per output frame, each with a valid 48 kHz stereo header.
    let n_units = sink.frames.len();
    assert!(
        (20..=30).contains(&n_units),
        "24000 samples / 1024 per AU = ~24 AUs (+ encoder delay), got {n_units}"
    );
    for (i, au) in sink.frames.iter().enumerate() {
        assert!(au.len() > 7, "AU {i} carries a payload");
        assert_eq!(au[0], 0xFF, "AU {i} syncword");
        assert_eq!(au[1] & 0xF0, 0xF0, "AU {i} syncword");
        let sr_index = (au[2] >> 2) & 0x0F;
        assert_eq!(sr_index, 3, "AU {i} 48 kHz (index 3)");
        let channels = ((au[2] & 1) << 2) | (au[3] >> 6);
        assert_eq!(channels, 2, "AU {i} stereo");
        let frame_len =
            (((au[3] & 3) as usize) << 11) | ((au[4] as usize) << 3) | ((au[5] >> 5) as usize);
        assert_eq!(frame_len, au.len(), "AU {i} header length matches");
    }

    // Oracle: ffmpeg decodes the ADTS stream back to PCM, and the tone
    // survives (sample count near the input, dominant frequency ~440 Hz).
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("ADTS framing validated; skipping decode oracle: no ffmpeg CLI");
        return;
    }
    let dir = std::env::temp_dir();
    let stamp = std::process::id();
    let aac = dir.join(format!("g2g-m769-{stamp}.aac"));
    let out = dir.join(format!("g2g-m769-{stamp}.s16"));
    std::fs::write(&aac, sink.frames.concat()).unwrap();
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&aac)
        .args(["-f", "s16le", "-ac", "1"])
        .arg(&out)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg decodes our ADTS stream");
    let decoded = std::fs::read(&out).unwrap();
    let _ = std::fs::remove_file(&aac);
    let _ = std::fs::remove_file(&out);

    let samples: Vec<i16> = decoded
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    assert!(
        (TOTAL as i64 - samples.len() as i64).unsigned_abs() < 4096,
        "decoded length near the input ({} vs {TOTAL})",
        samples.len()
    );
    // Dominant frequency by zero crossings over the steady middle (the codec's
    // leading priming fades in): 440 Hz -> ~880 crossings/s.
    let mid = &samples[samples.len() / 4..samples.len() * 3 / 4];
    let crossings = mid.windows(2).filter(|w| (w[0] < 0) != (w[1] < 0)).count();
    let seconds = mid.len() as f64 / 48_000.0;
    let freq = crossings as f64 / seconds / 2.0;
    assert!(
        (400.0..480.0).contains(&freq),
        "decoded tone is ~440 Hz, measured {freq:.1}"
    );
}

/// Await the element's boxed process future.
async fn block_send(enc: &mut FfmpegAacEnc, packet: PipelinePacket, sink: &mut CaptureSink) {
    enc.process(packet, sink).await.expect("process");
}
