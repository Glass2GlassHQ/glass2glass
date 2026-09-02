//! M1030: `wavenc` wraps raw PCM in a RIFF/WAVE byte stream.
//!
//! The header has to describe the negotiated stream (a reader takes the rate,
//! the channel count and the sample width from it) and the samples have to
//! reach the output untouched, so a written file plays back what came in.

use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, Frame, FrameTiming, G2gError,
    MemoryDomain, OutputSink, PipelinePacket, PushOutcome,
};
use g2g_plugins::wavenc::WavEnc;

#[derive(Default)]
struct CollectSink {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for CollectSink {
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

fn frame(bytes: Vec<u8>) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    }
}

fn run(format: AudioFormat, channels: u8, rate: u32, samples: Vec<u8>) -> Vec<Vec<u8>> {
    let mut element = WavEnc::new();
    element
        .configure_pipeline(&Caps::Audio {
            format,
            channels,
            sample_rate: rate,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        })
        .unwrap();
    let mut sink = CollectSink::default();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    rt.block_on(element.process(PipelinePacket::DataFrame(frame(samples)), &mut sink))
        .unwrap();
    sink.packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => {
                Some(f.domain.require_system_slice("test").unwrap().to_vec())
            }
            _ => None,
        })
        .collect()
}

#[test]
fn the_header_describes_the_negotiated_stream() {
    let out = run(AudioFormat::PcmS16Le, 2, 48_000, vec![7u8; 16]);
    let header = &out[0];

    assert_eq!(header.len(), 44, "the canonical RIFF/WAVE header");
    assert_eq!(&header[0..4], b"RIFF");
    assert_eq!(&header[8..12], b"WAVE");
    assert_eq!(&header[12..16], b"fmt ");
    assert_eq!(
        u16::from_le_bytes([header[20], header[21]]),
        1,
        "integer PCM"
    );
    assert_eq!(u16::from_le_bytes([header[22], header[23]]), 2, "channels");
    assert_eq!(
        u32::from_le_bytes([header[24], header[25], header[26], header[27]]),
        48_000,
        "sample rate"
    );
    assert_eq!(
        u32::from_le_bytes([header[28], header[29], header[30], header[31]]),
        48_000 * 4,
        "byte rate is rate * channels * sample width"
    );
    assert_eq!(
        u16::from_le_bytes([header[32], header[33]]),
        4,
        "block align"
    );
    assert_eq!(
        u16::from_le_bytes([header[34], header[35]]),
        16,
        "bit depth"
    );
    assert_eq!(&header[36..40], b"data");
}

#[test]
fn float_pcm_is_tagged_as_ieee_float() {
    let out = run(AudioFormat::PcmF32Le, 1, 44_100, vec![0u8; 8]);
    assert_eq!(
        u16::from_le_bytes([out[0][20], out[0][21]]),
        3,
        "IEEE float, not integer PCM"
    );
    assert_eq!(
        u16::from_le_bytes([out[0][34], out[0][35]]),
        32,
        "bit depth"
    );
}

#[test]
fn the_samples_pass_through_untouched() {
    let samples: Vec<u8> = (0..32u8).collect();
    let out = run(AudioFormat::PcmS16Le, 1, 8_000, samples.clone());
    assert_eq!(out.len(), 2, "the header, then the samples");
    assert_eq!(out[1], samples);
}

#[test]
fn the_header_goes_out_once() {
    let mut element = WavEnc::new();
    element
        .configure_pipeline(&Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 1,
            sample_rate: 8_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        })
        .unwrap();
    let mut sink = CollectSink::default();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    for _ in 0..3 {
        rt.block_on(element.process(PipelinePacket::DataFrame(frame(vec![1u8; 4])), &mut sink))
            .unwrap();
    }
    let frames = sink
        .packets
        .iter()
        .filter(|p| matches!(p, PipelinePacket::DataFrame(_)))
        .count();
    assert_eq!(frames, 4, "one header plus three sample frames");
}

#[test]
fn a_pcm_input_derives_a_wav_byte_stream_output() {
    let element = WavEnc::new();
    let g2g_core::CapsConstraint::DerivedOutput(derive) = element.caps_constraint_as_transform()
    else {
        panic!("a muxer derives its output from its input");
    };
    assert_eq!(
        derive(&Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 1,
            sample_rate: 8_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        })
        .alternatives(),
        &[Caps::ByteStream {
            encoding: ByteStreamEncoding::Wav
        }]
    );
    // A compressed stream is not PCM, so nothing links.
    assert!(derive(&Caps::Audio {
        format: AudioFormat::Opus,
        channels: 2,
        sample_rate: 48_000,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    })
    .alternatives()
    .is_empty());
}

#[test]
fn a_compressed_input_is_refused() {
    let mut element = WavEnc::new();
    assert!(element
        .configure_pipeline(&Caps::Audio {
            format: AudioFormat::Opus,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        })
        .is_err());
}

/// The read side: a file `wavenc` wrote types as WAV and parses back into the
/// PCM it carried, so a written file plays through `decodebin`.
#[test]
fn a_written_file_types_and_parses_back() {
    use g2g_core::ByteStreamEncoding;
    use g2g_plugins::typefind::sniff;
    use g2g_plugins::wavparse::WavParse;

    let samples: Vec<u8> = (0..64u8).collect();
    let written = run(AudioFormat::PcmS16Le, 2, 44_100, samples.clone());
    let file: Vec<u8> = written.concat();

    assert_eq!(
        sniff(&file),
        Some(ByteStreamEncoding::Wav),
        "the header types as a WAV byte stream"
    );

    let mut parse = WavParse::new();
    parse
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Wav,
        })
        .unwrap();
    let mut sink = CollectSink::default();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    rt.block_on(parse.process(PipelinePacket::DataFrame(frame(file)), &mut sink))
        .unwrap();

    let caps = sink
        .packets
        .iter()
        .find_map(|p| match p {
            PipelinePacket::CapsChanged(c) => Some(c.clone()),
            _ => None,
        })
        .expect("the parsed fmt chunk is announced");
    assert_eq!(
        caps,
        Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 44_100,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    );
    let out: Vec<u8> = sink
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => {
                Some(f.domain.require_system_slice("test").unwrap().to_vec())
            }
            _ => None,
        })
        .flatten()
        .collect();
    assert_eq!(out, samples, "the samples survive the round trip");
}

/// Build a WAVE_FORMAT_EXTENSIBLE file: one `fmt ` chunk with a 40-byte body,
/// then a `data` chunk holding `samples`.
///
/// Field offsets within the `fmt ` body are the public WAVEFORMATEXTENSIBLE
/// layout: the 18-byte WAVEFORMATEX (wFormatTag 0, nChannels 2,
/// nSamplesPerSec 4, nAvgBytesPerSec 8, nBlockAlign 12, wBitsPerSample 14,
/// cbSize 16), then wValidBitsPerSample 18, dwChannelMask 20, and the 16-byte
/// SubFormat GUID at 24 (its first two bytes are the wFormatTag it extends).
fn extensible_wav(channels: u16, rate: u32, channel_mask: u32, samples: &[u8]) -> Vec<u8> {
    const FORMAT_EXTENSIBLE: u16 = 0xFFFE;
    const SUBFORMAT_PCM: u16 = 1;
    const BITS: u16 = 16;
    let block_align = channels * BITS / 8;

    let mut fmt = Vec::new();
    fmt.extend_from_slice(&FORMAT_EXTENSIBLE.to_le_bytes()); // 0
    fmt.extend_from_slice(&channels.to_le_bytes()); // 2
    fmt.extend_from_slice(&rate.to_le_bytes()); // 4
    fmt.extend_from_slice(&(rate * u32::from(block_align)).to_le_bytes()); // 8
    fmt.extend_from_slice(&block_align.to_le_bytes()); // 12
    fmt.extend_from_slice(&BITS.to_le_bytes()); // 14
    fmt.extend_from_slice(&22u16.to_le_bytes()); // 16: cbSize
    fmt.extend_from_slice(&BITS.to_le_bytes()); // 18: wValidBitsPerSample
    fmt.extend_from_slice(&channel_mask.to_le_bytes()); // 20: dwChannelMask
    fmt.extend_from_slice(&SUBFORMAT_PCM.to_le_bytes()); // 24: SubFormat GUID
    fmt.extend_from_slice(&[0u8; 14]);
    assert_eq!(fmt.len(), 40);

    let mut body = Vec::new();
    body.extend_from_slice(b"WAVE");
    body.extend_from_slice(b"fmt ");
    body.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
    body.extend_from_slice(&fmt);
    body.extend_from_slice(b"data");
    body.extend_from_slice(&(samples.len() as u32).to_le_bytes());
    body.extend_from_slice(samples);

    let mut file = Vec::new();
    file.extend_from_slice(b"RIFF");
    file.extend_from_slice(&(body.len() as u32).to_le_bytes());
    file.extend_from_slice(&body);
    file
}

fn parse_caps(file: Vec<u8>) -> Caps {
    use g2g_plugins::wavparse::WavParse;

    let mut parse = WavParse::new();
    parse
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Wav,
        })
        .unwrap();
    let mut sink = CollectSink::default();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    rt.block_on(parse.process(PipelinePacket::DataFrame(frame(file)), &mut sink))
        .unwrap();
    sink.packets
        .iter()
        .find_map(|p| match p {
            PipelinePacket::CapsChanged(c) => Some(c.clone()),
            _ => None,
        })
        .expect("the parsed fmt chunk is announced")
}

#[test]
fn an_extensible_fmt_chunk_carries_its_channel_mask_into_the_caps() {
    use g2g_core::ChannelLayout;

    let samples: Vec<u8> = (0..48u8).collect();
    // 0x3f = FL FR FC LFE BL BR, the 5.1 mask.
    assert_eq!(
        parse_caps(extensible_wav(6, 48_000, 0x3f, &samples)),
        Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 6,
            sample_rate: 48_000,
            channel_layout: ChannelLayout::SURROUND_5_1,
        }
    );
    // A mask whose speaker count disagrees with nChannels describes a different
    // stream, so the parse falls back to the count convention.
    assert_eq!(
        parse_caps(extensible_wav(6, 48_000, 0x3, &samples)),
        Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 6,
            sample_rate: 48_000,
            channel_layout: ChannelLayout::UNSPECIFIED,
        }
    );
    // A mask of 0 (the "no positions" WAVEFORMATEXTENSIBLE spelling) stays the
    // wildcard rather than becoming a layout.
    assert_eq!(
        parse_caps(extensible_wav(2, 48_000, 0, &(0..32u8).collect::<Vec<_>>())),
        Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: ChannelLayout::UNSPECIFIED,
        }
    );
}

/// Chunk sizes come from the file, so a chunk claiming more than the stream
/// holds must wait for more input rather than read past the buffer.
#[test]
fn an_overlong_chunk_does_not_read_past_the_buffer() {
    use g2g_core::ByteStreamEncoding;
    use g2g_plugins::wavparse::WavParse;

    let mut file = Vec::new();
    file.extend_from_slice(b"RIFF");
    file.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
    file.extend_from_slice(b"WAVE");
    // A LIST chunk that declares far more than follows it.
    file.extend_from_slice(b"LIST");
    file.extend_from_slice(&0xFFFF_FF00u32.to_le_bytes());
    file.extend_from_slice(&[0u8; 8]);

    let mut parse = WavParse::new();
    parse
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Wav,
        })
        .unwrap();
    let mut sink = CollectSink::default();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    // Waits for the rest of the chunk instead of panicking or emitting samples.
    rt.block_on(parse.process(PipelinePacket::DataFrame(frame(file)), &mut sink))
        .unwrap();
    assert!(sink.packets.is_empty());
}
