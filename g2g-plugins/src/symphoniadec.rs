use g2g_core::ChannelLayout;

use symphonia_core::codecs::audio::AudioDecoder;
use symphonia_core::errors::Error as SymphoniaError;
use symphonia_core::packet::PacketRef;
use symphonia_core::units::{Duration, Timestamp};

use crate::pcmoutput::{f32_to_s16, DecodedPcm};

pub(crate) fn decode_to_s16(
    decoder: &mut dyn AudioDecoder,
    unit: &[u8],
) -> Result<DecodedPcm, SymphoniaError> {
    let packet = PacketRef::new(0, Timestamp::ZERO, Duration::ZERO, unit);
    let buffer = decoder.decode_ref(&packet)?;
    let spec = buffer.spec();
    let channels = u8::try_from(spec.channels().count())
        .ok()
        .filter(|count| *count > 0)
        .ok_or(SymphoniaError::Unsupported("channel count"))?;
    let sample_rate = spec.rate();
    let mut samples = alloc::vec![0f32; buffer.samples_interleaved()];
    buffer.copy_to_slice_interleaved::<f32, _>(&mut samples[..]);
    let pcm = samples
        .into_iter()
        .flat_map(|sample| f32_to_s16(sample).to_le_bytes())
        .collect();
    Ok(DecodedPcm {
        pcm,
        channels,
        sample_rate,
        channel_layout: ChannelLayout::UNSPECIFIED,
    })
}

#[cfg(any(feature = "aac-decode", feature = "mpeg-audio-decode"))]
pub use element::{SymphoniaCodec, SymphoniaDec};

#[cfg(feature = "aac-decode")]
pub use element::{Aac, AacDec};

#[cfg(feature = "mpeg-audio-decode")]
pub use element::{MpegAudio, MpegAudioDec};

#[cfg(any(feature = "aac-decode", feature = "mpeg-audio-decode"))]
mod element {
    use core::future::Future;
    use core::marker::PhantomData;
    use core::pin::Pin;

    use alloc::boxed::Box;
    use alloc::vec::Vec;

    use g2g_core::log::{short_type_name, Target};
    use g2g_core::{
        g2g_error, g2g_warn, pcm_caps_any_rate_and_channels, AsyncElement, AudioFormat, Caps,
        CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, G2gError, OutputSink,
        PadTemplate, PadTemplates, PipelinePacket,
    };

    use symphonia_core::codecs::audio::AudioDecoder;
    use symphonia_core::errors::Error as SymphoniaError;

    use super::decode_to_s16;
    use crate::audioframe::split_frames;
    use crate::pcmoutput::{pcm_s16_decoder_output, DecodedPcm, PcmOutput};

    mod sealed {
        pub trait Sealed {}
    }

    pub trait SymphoniaCodec: sealed::Sealed + Send + 'static {
        type Config: Copy + PartialEq + core::fmt::Debug + Send;
        const FORMATS: &'static [AudioFormat];
        const LONG_NAME: &'static str;
        const DESCRIPTION: &'static str;
        fn frame_len(buf: &[u8]) -> Option<usize>;
        fn frame_config(frame: &[u8]) -> Option<(Self::Config, &[u8])>;
        fn open(config: Self::Config) -> Result<Box<dyn AudioDecoder>, SymphoniaError>;
    }

    pub struct SymphoniaDec<C: SymphoniaCodec> {
        decoder: Option<(C::Config, Box<dyn AudioDecoder>)>,
        configured: bool,
        output: PcmOutput,
        codec: PhantomData<C>,
    }

    impl<C: SymphoniaCodec> core::fmt::Debug for SymphoniaDec<C> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct(C::LONG_NAME)
                .field("config", &self.decoder.as_ref().map(|(config, _)| config))
                .field("configured", &self.configured)
                .field("emitted", &self.output.frames_pushed())
                .finish()
        }
    }

    impl<C: SymphoniaCodec> Default for SymphoniaDec<C> {
        fn default() -> Self {
            Self::new()
        }
    }

    impl<C: SymphoniaCodec> SymphoniaDec<C> {
        pub fn new() -> Self {
            Self {
                decoder: None,
                configured: false,
                output: PcmOutput::default(),
                codec: PhantomData,
            }
        }

        fn decodes(format: AudioFormat) -> bool {
            C::FORMATS.contains(&format)
        }

        fn decode_frame(&mut self, frame: &[u8]) -> Result<Option<DecodedPcm>, G2gError> {
            let target = Target::category(short_type_name::<Self>());
            let Some((config, unit)) = C::frame_config(frame) else {
                g2g_warn!(
                    target,
                    "skipped a {}-byte frame with no usable header",
                    frame.len()
                );
                return Ok(None);
            };
            let decoder = match self.decoder.take() {
                Some((open_config, decoder)) if open_config == config => decoder,
                _ => C::open(config).map_err(|error| {
                    g2g_error!(target, "symphonia cannot decode {config:?}: {error}");
                    G2gError::CapsMismatch
                })?,
            };
            let (_, decoder) = self.decoder.insert((config, decoder));
            match decode_to_s16(decoder.as_mut(), unit) {
                Ok(pcm) => Ok(Some(pcm)),
                // the ffmpeg CLI logs a frame its decoder rejects and decodes on
                Err(SymphoniaError::DecodeError(reason)) => {
                    g2g_warn!(
                        target,
                        "skipped a {}-byte frame symphonia rejects: {reason}",
                        frame.len()
                    );
                    Ok(None)
                }
                Err(error) => {
                    g2g_error!(target, "symphonia cannot decode {config:?}: {error}");
                    Err(G2gError::CapsMismatch)
                }
            }
        }
    }

    impl<C: SymphoniaCodec> AsyncElement for SymphoniaDec<C> {
        type ProcessFuture<'a>
            = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
        where
            Self: 'a;

        fn input_domains(&self) -> g2g_core::memory::DomainSet {
            g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
        }

        fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
            match upstream_caps {
                Caps::Audio { format, .. } if Self::decodes(*format) => Ok(upstream_caps.clone()),
                _ => Err(G2gError::CapsMismatch),
            }
        }

        fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
            CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
                Caps::Audio { format, .. } if Self::decodes(*format) => pcm_s16_decoder_output(),
                _ => CapsSet::from_alternatives(Vec::new()),
            }))
        }

        fn configure_pipeline(
            &mut self,
            absolute_caps: &Caps,
        ) -> Result<ConfigureOutcome, G2gError> {
            match absolute_caps {
                Caps::Audio { format, .. } if Self::decodes(*format) => {
                    self.configured = true;
                    Ok(ConfigureOutcome::Accepted)
                }
                _ => Err(G2gError::CapsMismatch),
            }
        }

        fn metadata(&self) -> ElementMetadata {
            ElementMetadata::new(C::LONG_NAME, "Codec/Decoder/Audio", C::DESCRIPTION, "g2g")
        }

        fn process<'a>(
            &'a mut self,
            packet: PipelinePacket,
            out: &'a mut dyn OutputSink,
        ) -> Self::ProcessFuture<'a> {
            Box::pin(async move {
                if !self.configured {
                    return Err(G2gError::NotConfigured);
                }
                match packet {
                    PipelinePacket::DataFrame(frame) => {
                        let buf = frame
                            .domain
                            .require_system_slice(short_type_name::<Self>())?;
                        let frames = split_frames(buf, C::frame_len);
                        if frames.is_empty() {
                            g2g_warn!(
                                Target::category(short_type_name::<Self>()),
                                "skipped a {}-byte buffer that opens on no frame",
                                buf.len()
                            );
                        }
                        let mut decoded = Vec::with_capacity(frames.len());
                        for unit in frames {
                            decoded.extend(self.decode_frame(unit)?);
                        }
                        self.output
                            .push_consecutive(frame.timing, decoded, out)
                            .await?;
                    }
                    PipelinePacket::CapsChanged(caps) => match &caps {
                        Caps::Audio { format, .. } if Self::decodes(*format) => {}
                        Caps::Audio {
                            format: AudioFormat::PcmS16Le,
                            ..
                        } => self.output.forward_caps(caps, out).await?,
                        _ => return Err(G2gError::CapsMismatch),
                    },
                    PipelinePacket::Flush => {
                        if let Some((_, decoder)) = self.decoder.as_mut() {
                            decoder.reset();
                        }
                        self.output.forget_caps();
                        out.push(PipelinePacket::Flush).await?;
                    }
                    PipelinePacket::Segment(segment) => {
                        self.output.forward_segment(segment, out).await?;
                    }
                    // the runner forwards Eos after process(Eos) returns
                    PipelinePacket::Eos => {}
                    other => {
                        out.push(other).await?;
                    }
                }
                Ok(())
            })
        }
    }

    impl<C: SymphoniaCodec> PadTemplates for SymphoniaDec<C> {
        fn pad_templates() -> Vec<PadTemplate> {
            let compressed = C::FORMATS
                .iter()
                .map(|format| crate::typefind::elementary_audio_caps(*format))
                .collect();
            Vec::from([
                PadTemplate::sink(CapsSet::from_alternatives(compressed)),
                PadTemplate::source(pcm_caps_any_rate_and_channels(|format| {
                    format == AudioFormat::PcmS16Le
                })),
            ])
        }
    }

    #[cfg(feature = "aac-decode")]
    #[derive(Debug)]
    pub struct Aac;

    #[cfg(feature = "aac-decode")]
    pub type AacDec = SymphoniaDec<Aac>;

    #[cfg(feature = "aac-decode")]
    impl sealed::Sealed for Aac {}

    #[cfg(feature = "aac-decode")]
    impl SymphoniaCodec for Aac {
        // the AudioSpecificConfig an ADTS header carries
        type Config = [u8; 2];
        const FORMATS: &'static [AudioFormat] = &[AudioFormat::Aac];
        const LONG_NAME: &'static str = "AAC decoder";
        const DESCRIPTION: &'static str =
            "Decodes ADTS AAC-LC, mono or stereo, to interleaved PcmS16Le (pure Rust, symphonia)";

        fn frame_len(buf: &[u8]) -> Option<usize> {
            crate::aacparse::adts_frame_len(buf)
        }

        fn frame_config(frame: &[u8]) -> Option<(Self::Config, &[u8])> {
            Some((
                crate::aacparse::asc_from_adts(frame)?,
                crate::aacparse::strip_adts(frame),
            ))
        }

        fn open(config: Self::Config) -> Result<Box<dyn AudioDecoder>, SymphoniaError> {
            use symphonia_core::codecs::audio::well_known::CODEC_ID_AAC;
            use symphonia_core::codecs::audio::{AudioCodecParameters, AudioDecoderOptions};
            let mut params = AudioCodecParameters::new();
            params
                .for_codec(CODEC_ID_AAC)
                .with_extra_data(Box::from(config));
            let decoder =
                symphonia_codec_aac::AacDecoder::try_new(&params, &AudioDecoderOptions::default())?;
            Ok(Box::new(decoder))
        }
    }

    #[cfg(feature = "mpeg-audio-decode")]
    #[derive(Debug)]
    pub struct MpegAudio;

    #[cfg(feature = "mpeg-audio-decode")]
    pub type MpegAudioDec = SymphoniaDec<MpegAudio>;

    #[cfg(feature = "mpeg-audio-decode")]
    impl sealed::Sealed for MpegAudio {}

    #[cfg(feature = "mpeg-audio-decode")]
    impl SymphoniaCodec for MpegAudio {
        type Config = symphonia_core::codecs::audio::AudioCodecId;
        const FORMATS: &'static [AudioFormat] = &[AudioFormat::Mp2, AudioFormat::Mp3];
        const LONG_NAME: &'static str = "MPEG audio decoder";
        const DESCRIPTION: &'static str =
            "Decodes MPEG audio layer II (mp2) and III (mp3) to interleaved PcmS16Le (pure Rust, symphonia)";

        fn frame_len(buf: &[u8]) -> Option<usize> {
            crate::audioframe::mpa_frame_len(buf)
        }

        fn frame_config(frame: &[u8]) -> Option<(Self::Config, &[u8])> {
            use crate::audioframe::MpaLayer;
            use symphonia_core::codecs::audio::well_known::{CODEC_ID_MP2, CODEC_ID_MP3};
            let codec = match crate::audioframe::mpa_header(frame)?.layer {
                MpaLayer::Two => CODEC_ID_MP2,
                MpaLayer::Three => CODEC_ID_MP3,
                MpaLayer::One => return None,
            };
            Some((codec, frame))
        }

        fn open(config: Self::Config) -> Result<Box<dyn AudioDecoder>, SymphoniaError> {
            use symphonia_core::codecs::audio::{AudioCodecParameters, AudioDecoderOptions};
            let mut params = AudioCodecParameters::new();
            params.for_codec(config);
            let decoder = symphonia_bundle_mp3::MpaDecoder::try_new(
                &params,
                &AudioDecoderOptions::default(),
            )?;
            Ok(Box::new(decoder))
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use g2g_core::{AudioFormat, Caps, ChannelLayout, ANY_CHANNELS, ANY_SAMPLE_RATE};

    use crate::testutil::{data_bytes, first_caps, CollectSink};

    // symphonia's float maths rounds some samples of these fixtures one step away from libavcodec's
    const MAX_DIFFERENCE_FROM_FFMPEG: i32 = 1;

    fn elementary(format: AudioFormat) -> Caps {
        Caps::Audio {
            format,
            channels: ANY_CHANNELS,
            sample_rate: ANY_SAMPLE_RATE,
            channel_layout: ChannelLayout::UNSPECIFIED,
        }
    }

    fn stereo_44100() -> Caps {
        Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 44_100,
            channel_layout: ChannelLayout::UNSPECIFIED,
        }
    }

    fn assert_matches_ffmpeg(decoded: &CollectSink, reference: &[u8]) {
        assert_eq!(first_caps(&decoded.packets), Some(stereo_44100()));
        let pcm = data_bytes(&decoded.packets);
        assert_eq!(
            pcm.len(),
            reference.len(),
            "the same sample count as ffmpeg"
        );
        let samples = |bytes: &[u8]| -> Vec<i32> {
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&sample| i32::from(i16::from_le_bytes(sample)))
                .collect()
        };
        let largest = samples(&pcm)
            .into_iter()
            .zip(samples(reference))
            .map(|(ours, ffmpeg)| (ours - ffmpeg).abs())
            .max();
        assert!(
            largest <= Some(MAX_DIFFERENCE_FROM_FFMPEG),
            "largest sample difference from ffmpeg {largest:?}"
        );
    }

    #[cfg(feature = "aac-decode")]
    #[test]
    fn aac_matches_ffmpeg() {
        let decoded = crate::testutil::run(
            &mut super::AacDec::new(),
            &elementary(AudioFormat::Aac),
            &[include_bytes!(
                "../tests/fixtures/aac_stereo_44100_adts.aac"
            )],
        );
        assert_matches_ffmpeg(
            &decoded,
            include_bytes!("../tests/fixtures/aac_stereo_44100_adts.aac.s16le"),
        );
    }

    #[cfg(feature = "mpeg-audio-decode")]
    #[test]
    fn mp3_matches_ffmpeg() {
        let parsed = crate::testutil::run(
            &mut crate::mpegaudioparse::MpegAudioParse::new(),
            &elementary(AudioFormat::Mp3),
            &[include_bytes!(
                "../tests/fixtures/mp3_stereo_44100_id3v2.mp3"
            )],
        );
        let frames = parsed
            .packets
            .into_iter()
            .filter(|packet| !matches!(packet, g2g_core::PipelinePacket::Eos));
        let decoded = crate::testutil::run_packets(
            &mut super::MpegAudioDec::new(),
            &elementary(AudioFormat::Mp3),
            frames,
        );
        assert_matches_ffmpeg(
            &decoded,
            include_bytes!("../tests/fixtures/mp3_stereo_44100_id3v2.mp3.s16le"),
        );
    }
}
