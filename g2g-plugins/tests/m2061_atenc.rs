// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(
    target_os = "macos",
    feature = "audiotoolbox-aac",
    feature = "aac-decode"
))]

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, AudioFormat, Caps, ChannelLayout, G2gError, OutputSink, PushOutcome,
    ANY_CHANNELS, ANY_SAMPLE_RATE,
};
use g2g_plugins::atenc::AtEnc;
use g2g_plugins::symphoniadec::AacDec;

const SAMPLE_RATE: u32 = 48_000;
const CHANNELS: u8 = 2;
const TONE_HZ: f64 = 440.0;
const AMPLITUDE: f64 = 0.5;
const SECONDS: u32 = 1;
const BUFFER_FRAMES: usize = 480;
const MAX_LEVEL_ERROR: f64 = 0.1;

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

fn payloads(packets: &[PipelinePacket]) -> Vec<Vec<u8>> {
    packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => frame.domain.as_system_slice().map(<[u8]>::to_vec),
            _ => None,
        })
        .collect()
}

fn frame(bytes: Vec<u8>, sequence: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming::default(),
        sequence,
    ))
}

fn tone() -> Vec<i16> {
    let frames = (SAMPLE_RATE * SECONDS) as usize;
    (0..frames)
        .flat_map(|index| {
            let phase =
                2.0 * std::f64::consts::PI * TONE_HZ * index as f64 / f64::from(SAMPLE_RATE);
            let sample = (phase.sin() * AMPLITUDE * f64::from(i16::MAX)) as i16;
            [sample; CHANNELS as usize]
        })
        .collect()
}

fn root_mean_square(samples: &[i16]) -> f64 {
    let sum: f64 = samples
        .iter()
        .map(|&sample| (f64::from(sample) / f64::from(i16::MAX)).powi(2))
        .sum();
    (sum / samples.len() as f64).sqrt()
}

async fn run<E: AsyncElement>(
    element: &mut E,
    caps: &Caps,
    packets: Vec<PipelinePacket>,
) -> Collect {
    element.configure_pipeline(caps).unwrap();
    let mut sink = Collect::default();
    for packet in packets.into_iter().chain([PipelinePacket::Eos]) {
        element.process(packet, &mut sink).await.unwrap();
    }
    sink
}

#[tokio::test]
async fn aacdec_decodes_what_atenc_encodes() {
    let source = tone();
    let pcm_caps = Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: CHANNELS,
        sample_rate: SAMPLE_RATE,
        channel_layout: ChannelLayout::UNSPECIFIED,
    };
    let buffers = source
        .chunks(BUFFER_FRAMES * usize::from(CHANNELS))
        .enumerate()
        .map(|(index, chunk)| {
            let bytes = chunk
                .iter()
                .flat_map(|sample| sample.to_le_bytes())
                .collect();
            frame(bytes, index as u64)
        })
        .collect();
    let encoded = run(&mut AtEnc::new(), &pcm_caps, buffers).await;
    assert!(encoded.packets.iter().any(|packet| matches!(
        packet,
        PipelinePacket::CapsChanged(Caps::Audio {
            format: AudioFormat::Aac,
            channels: CHANNELS,
            sample_rate: SAMPLE_RATE,
            ..
        })
    )));
    let access_units = payloads(&encoded.packets);
    let input_access_units = source.len() / usize::from(CHANNELS) / 1024;
    assert!(
        access_units.len() > input_access_units,
        "{} access units",
        access_units.len()
    );
    assert!(access_units
        .iter()
        .all(|access_unit| access_unit.starts_with(&[0xFF]) && access_unit[1] & 0xF0 == 0xF0));

    let aac_caps = Caps::Audio {
        format: AudioFormat::Aac,
        channels: ANY_CHANNELS,
        sample_rate: ANY_SAMPLE_RATE,
        channel_layout: ChannelLayout::UNSPECIFIED,
    };
    let decoded_packets = access_units
        .into_iter()
        .enumerate()
        .map(|(index, access_unit)| frame(access_unit, index as u64))
        .collect();
    let decoded = run(&mut AacDec::new(), &aac_caps, decoded_packets).await;
    let decoded: Vec<i16> = payloads(&decoded.packets)
        .concat()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&sample| i16::from_le_bytes(sample))
        .collect();
    assert!(
        decoded.len() >= source.len(),
        "{} decoded samples",
        decoded.len()
    );
    let middle = |samples: &[i16]| {
        let quarter = samples.len() / 4;
        root_mean_square(&samples[quarter..3 * quarter])
    };
    let level_error = (middle(&decoded) - middle(&source)).abs() / middle(&source);
    assert!(level_error <= MAX_LEVEL_ERROR, "level error {level_error}");
}
