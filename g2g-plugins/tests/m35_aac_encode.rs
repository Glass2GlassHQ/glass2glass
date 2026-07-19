//! M35: `MfAacEncode` compresses S16 PCM to AAC access units through the real
//! MS AAC encoder MFT, and exposes the AudioSpecificConfig the decoder /
//! container need.
//!
//! Run with:
//!
//! ```powershell
//! cargo test -p g2g-plugins --features mf-aac --test m35_aac_encode
//! ```

#![cfg(all(target_os = "windows", feature = "mf-aac"))]

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{AudioFormat, Caps, G2gError};
use g2g_plugins::mfaacencode::MfAacEncode;

const RATE: u32 = 48_000;
const CHANNELS: u8 = 2;
const SAMPLES_PER_BUFFER: usize = 1024;
const BUFFERS: usize = 10;

#[derive(Default)]
struct Collect {
    aus: Vec<Vec<u8>>,
    caps: Vec<Caps>,
}

impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if let MemoryDomain::System(s) = &f.domain {
                        self.aus.push(s.as_slice().to_vec());
                    }
                }
                PipelinePacket::CapsChanged(c) => self.caps.push(c),
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// One buffer of interleaved s16 PCM: a quiet ramp so frames differ.
fn pcm_buffer(index: usize) -> Frame {
    let mut data = Vec::with_capacity(SAMPLES_PER_BUFFER * CHANNELS as usize * 2);
    for s in 0..SAMPLES_PER_BUFFER {
        let v = (((s + index * 64) % 256) as i16 - 128) * 64;
        for _ in 0..CHANNELS {
            data.extend_from_slice(&v.to_le_bytes());
        }
    }
    let pts = (index * SAMPLES_PER_BUFFER) as u64 * 1_000_000_000 / RATE as u64;
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns: pts,
            dts_ns: pts,
            ..FrameTiming::default()
        },
        sequence: index as u64,
        meta: Default::default(),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn encodes_pcm_to_aac_access_units() {
    let mut enc = MfAacEncode::new();
    let caps = Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: CHANNELS,
        sample_rate: RATE,
    };
    let narrowed = enc.intercept_caps(&caps).expect("intercept s16");
    enc.configure_pipeline(&narrowed)
        .expect("AAC encoder MFT initialises");

    // AudioSpecificConfig is available right after negotiation.
    let asc = enc.audio_specific_config().expect("ASC present").to_vec();
    assert!(!asc.is_empty(), "AudioSpecificConfig is non-empty");

    let mut sink = Collect::default();
    for i in 0..BUFFERS {
        enc.process(PipelinePacket::DataFrame(pcm_buffer(i)), &mut sink)
            .await
            .expect("encode buffer");
    }
    enc.process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("encode eos");

    assert!(
        enc.encoded_count() >= (BUFFERS as u64) - 2,
        "most 1024-sample buffers produce an AU, got {}",
        enc.encoded_count()
    );
    assert_eq!(enc.encoded_count() as usize, sink.aus.len());
    assert!(sink.aus.iter().all(|au| !au.is_empty()), "AUs non-empty");
    assert!(
        matches!(
            sink.caps.first(),
            Some(Caps::Audio {
                format: AudioFormat::Aac,
                channels: CHANNELS,
                sample_rate: RATE,
            })
        ),
        "emits AAC caps, got {:?}",
        sink.caps
    );
}
