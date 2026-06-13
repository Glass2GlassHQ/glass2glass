//! M36: `MfAacEncode -> MfAacDecode` round trip through both real MS AAC MFTs.
//! PCM in, AAC, PCM back; the decoded sample count matches the input within the
//! codec's priming delay.
//!
//! Run with:
//!
//! ```powershell
//! cargo test -p g2g-plugins --features mf-aac --test m36_aac_roundtrip
//! ```

#![cfg(all(target_os = "windows", feature = "mf-aac"))]

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{AudioFormat, Caps, G2gError};
use g2g_plugins::mfaacdecode::MfAacDecode;
use g2g_plugins::mfaacencode::MfAacEncode;

const RATE: u32 = 48_000;
const CHANNELS: u8 = 2;
const SAMPLES_PER_BUFFER: usize = 1024;
const BUFFERS: usize = 20;

#[derive(Default)]
struct Collect {
    frames: Vec<Frame>,
    caps: Vec<Caps>,
}

impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => self.frames.push(f),
                PipelinePacket::CapsChanged(c) => self.caps.push(c),
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn pcm_buffer(index: usize) -> Frame {
    let mut data = Vec::with_capacity(SAMPLES_PER_BUFFER * CHANNELS as usize * 2);
    for s in 0..SAMPLES_PER_BUFFER {
        // a low-frequency triangle so the signal is real but compresses cleanly
        let v = ((((s + index * 32) % 128) as i16) - 64) * 128;
        for _ in 0..CHANNELS {
            data.extend_from_slice(&v.to_le_bytes());
        }
    }
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: index as u64,
    }
}

#[tokio::test(flavor = "current_thread")]
async fn pcm_aac_pcm_round_trip_recovers_the_stream() {
    let in_caps = Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: CHANNELS,
        sample_rate: RATE,
    };

    // --- encode ---
    let mut enc = MfAacEncode::new();
    let narrowed = enc.intercept_caps(&in_caps).expect("intercept s16");
    enc.configure_pipeline(&narrowed).expect("encoder init");
    let asc = enc.audio_specific_config().expect("ASC").to_vec();

    let mut encoded = Collect::default();
    for i in 0..BUFFERS {
        enc.process(PipelinePacket::DataFrame(pcm_buffer(i)), &mut encoded)
            .await
            .expect("encode");
    }
    enc.process(PipelinePacket::Eos, &mut encoded)
        .await
        .expect("encode eos");
    let aac_caps = encoded.caps.first().cloned().expect("aac caps");
    assert!(!encoded.frames.is_empty(), "produced AAC access units");

    // --- decode ---
    let mut dec = MfAacDecode::new().with_audio_specific_config(asc);
    let narrowed = dec.intercept_caps(&aac_caps).expect("intercept aac");
    dec.configure_pipeline(&narrowed).expect("decoder init");

    let mut decoded = Collect::default();
    for f in &encoded.frames {
        let MemoryDomain::System(s) = &f.domain else {
            unreachable!()
        };
        let au = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(
                s.as_slice().to_vec().into_boxed_slice(),
            )),
            timing: f.timing,
            sequence: f.sequence,
        };
        dec.process(PipelinePacket::DataFrame(au), &mut decoded)
            .await
            .expect("decode");
    }
    dec.process(PipelinePacket::Eos, &mut decoded)
        .await
        .expect("decode eos");

    // decoded caps are S16 PCM at the original geometry
    assert!(
        matches!(
            decoded.caps.first(),
            Some(Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: CHANNELS,
                sample_rate: RATE,
            })
        ),
        "decoded PCM caps, got {:?}",
        decoded.caps
    );

    // total decoded sample frames are within an AAC frame of the input
    let frame_bytes = (CHANNELS as usize) * 2;
    let decoded_frames: usize = decoded
        .frames
        .iter()
        .map(|f| match &f.domain {
            MemoryDomain::System(s) => s.as_slice().len() / frame_bytes,
            _ => 0,
        })
        .sum();
    let input_frames = BUFFERS * SAMPLES_PER_BUFFER;
    let diff = input_frames.abs_diff(decoded_frames);
    assert!(
        diff <= 2 * SAMPLES_PER_BUFFER,
        "decoded {decoded_frames} frames vs {input_frames} input (diff {diff})"
    );
}
