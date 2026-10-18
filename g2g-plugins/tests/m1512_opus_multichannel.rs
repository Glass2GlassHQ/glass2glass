#![cfg(feature = "opus")]

#[path = "common/collect_sink.rs"]
mod collect_sink;
use collect_sink::CollectSink;

use audiopus::ffi;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, ChannelLayout, Frame, FrameTiming, MemoryDomain,
    PipelinePacket,
};
use g2g_plugins::opusdec::OpusDec;
use g2g_plugins::opusparse::OPUS_RATE_HZ;

const CHANNELS: usize = 6;
const STREAMS: u8 = 4;
const COUPLED_STREAMS: u8 = 2;
// RFC 7845 family 1 table for 5.1, input in Vorbis order: FL C FR RL RR LFE
const VORBIS_MAPPING: [u8; CHANNELS] = [0, 4, 1, 2, 3, 5];
const VORBIS_CENTER: usize = 1;
// FL FR FC LFE BL BR
const SURROUND_5_1_CENTER: usize = 2;
const FRAME_SAMPLES: usize = 960;
const FRAME_NS: u64 = 20_000_000;
const FRAMES: usize = 10;
const MAX_PACKET_LEN: usize = 4000;
const TONE_PERIOD_SAMPLES: f32 = 100.0;
const TONE_AMPLITUDE: f32 = 12_000.0;

fn encode_center_tone() -> Vec<Vec<u8>> {
    let mut error = 0;
    // SAFETY: the mapping has one entry per input channel and `error` is valid.
    let encoder = unsafe {
        ffi::opus_multistream_encoder_create(
            OPUS_RATE_HZ as i32,
            CHANNELS as i32,
            i32::from(STREAMS),
            i32::from(COUPLED_STREAMS),
            VORBIS_MAPPING.as_ptr(),
            ffi::OPUS_APPLICATION_AUDIO,
            &mut error,
        )
    };
    assert!(!encoder.is_null() && error == ffi::OPUS_OK);
    let mut packets = Vec::new();
    for frame in 0..FRAMES {
        let mut pcm = vec![0i16; FRAME_SAMPLES * CHANNELS];
        for i in 0..FRAME_SAMPLES {
            let t = (frame * FRAME_SAMPLES + i) as f32;
            pcm[i * CHANNELS + VORBIS_CENTER] =
                ((t * core::f32::consts::TAU / TONE_PERIOD_SAMPLES).sin() * TONE_AMPLITUDE) as i16;
        }
        let mut packet = vec![0u8; MAX_PACKET_LEN];
        // SAFETY: `pcm` holds one frame per channel, `packet` holds `MAX_PACKET_LEN` bytes.
        let len = unsafe {
            ffi::opus_multistream_encode(
                encoder,
                pcm.as_ptr(),
                FRAME_SAMPLES as i32,
                packet.as_mut_ptr(),
                MAX_PACKET_LEN as i32,
            )
        };
        packet.truncate(usize::try_from(len).expect("the packet encodes"));
        packets.push(packet);
    }
    // SAFETY: created above, destroyed once.
    unsafe { ffi::opus_multistream_encoder_destroy(encoder) };
    packets
}

fn opus_head() -> Vec<u8> {
    let mut head = b"OpusHead".to_vec();
    head.push(1);
    head.push(CHANNELS as u8);
    head.extend_from_slice(&0u16.to_le_bytes());
    head.extend_from_slice(&OPUS_RATE_HZ.to_le_bytes());
    head.extend_from_slice(&0i16.to_le_bytes());
    head.push(1);
    head.extend_from_slice(&[STREAMS, COUPLED_STREAMS]);
    head.extend_from_slice(&VORBIS_MAPPING);
    head
}

fn frame(data: Vec<u8>, pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            ..FrameTiming::default()
        },
        0,
    ))
}

fn decode(head: Option<Vec<u8>>) -> (Vec<Caps>, Vec<i16>) {
    let mut decoder = OpusDec::new();
    decoder
        .configure_pipeline(&Caps::Audio {
            format: AudioFormat::Opus,
            channels: CHANNELS as u8,
            sample_rate: OPUS_RATE_HZ,
            channel_layout: ChannelLayout::UNSPECIFIED,
        })
        .unwrap();
    let mut sink = CollectSink::default();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let packets = head
        .into_iter()
        .chain(encode_center_tone())
        .enumerate()
        .map(|(index, data)| frame(data, index as u64 * FRAME_NS));
    for packet in packets {
        rt.block_on(decoder.process(packet, &mut sink)).unwrap();
    }
    let mut caps = Vec::new();
    let mut samples = Vec::new();
    for packet in sink.packets {
        match packet {
            PipelinePacket::CapsChanged(c) => caps.push(c),
            PipelinePacket::DataFrame(f) => samples.extend(
                f.domain
                    .require_system_slice("test")
                    .unwrap()
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&b| i16::from_le_bytes(b)),
            ),
            _ => {}
        }
    }
    (caps, samples)
}

fn channel_energy(samples: &[i16], channel: usize) -> f64 {
    samples
        .iter()
        .skip(channel)
        .step_by(CHANNELS)
        .map(|&s| f64::from(s).powi(2))
        .sum()
}

fn assert_center_lands_on_front_center(head: Option<Vec<u8>>) {
    let (caps, samples) = decode(head);
    assert_eq!(
        caps,
        vec![Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: CHANNELS as u8,
            sample_rate: OPUS_RATE_HZ,
            channel_layout: ChannelLayout::SURROUND_5_1,
        }]
    );
    assert_eq!(samples.len(), FRAMES * FRAME_SAMPLES * CHANNELS);
    let center = channel_energy(&samples, SURROUND_5_1_CENTER);
    for channel in (0..CHANNELS).filter(|&c| c != SURROUND_5_1_CENTER) {
        assert!(
            center > 100.0 * channel_energy(&samples, channel),
            "channel {channel} carries the center tone"
        );
    }
}

#[test]
fn an_opus_head_mapping_decodes_5_1_in_layout_order() {
    assert_center_lands_on_front_center(Some(opus_head()));
}

#[test]
fn a_headerless_stream_decodes_with_the_default_mapping() {
    assert_center_lands_on_front_center(None);
}
