// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Windows AAC audio decode element wrapping the Media Foundation AAC
//! Decoder MFT (`CLSID_MSAACDecMFT`). Consumes AAC-LC access units
//! (`AudioFormat::Aac`, `MemoryDomain::System`) and produces interleaved 16-bit
//! PCM (`PcmS16Le`), the decode-side mirror of `MfAacEncode`.
//!
//! The decoder needs the stream's AudioSpecificConfig to configure its input
//! type. An ADTS-framed stream carries it in the first frame's header, and a raw
//! one needs it from [`MfAacDecode::with_audio_specific_config`] (the MP4 `esds`
//! carries it). The MS AAC decoder is
//! synchronous, so the same `ProcessInput`/`ProcessOutput` drain loop as the
//! H.264 decoder applies.
//!
//! Threading: COM is MTA, every call on the owning thread; `Send` is asserted
//! under the same documented contract as `MfDecode`.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use windows::Win32::Media::MediaFoundation::{
    CLSID_MSAACDecMFT, IMFTransform, MFAudioFormat_AAC, MFAudioFormat_PCM, MFCreateMediaType,
    MFMediaType_Audio, MFShutdown, MFStartup, MFSTARTUP_FULL, MFT_MESSAGE_COMMAND_FLUSH,
    MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, MFT_MESSAGE_NOTIFY_START_OF_STREAM,
    MF_MT_AAC_AUDIO_PROFILE_LEVEL_INDICATION, MF_MT_AAC_PAYLOAD_TYPE,
    MF_MT_AUDIO_AVG_BYTES_PER_SECOND, MF_MT_AUDIO_BITS_PER_SAMPLE, MF_MT_AUDIO_BLOCK_ALIGNMENT,
    MF_MT_AUDIO_NUM_CHANNELS, MF_MT_AUDIO_SAMPLES_PER_SECOND, MF_MT_MAJOR_TYPE, MF_MT_SUBTYPE,
    MF_MT_USER_DATA, MF_VERSION,
};
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoUninitialize, CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED,
};

use crate::mfcommon::{
    feed_sample, make_input_sample, mf_err, output_provides_samples, process_output,
    read_sample_bytes, send_drain_command, OutputPull,
};
use crate::pcmoutput::{pcm_s16_decoder_output, DecodedPcm, PcmOutput};
use g2g_core::time::ticks_to_ns;
use g2g_core::{
    pcm_caps_any_rate_and_channels, AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet,
    ChannelLayout, ConfigureOutcome, FrameTiming, G2gError, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, ANY_CHANNELS, ANY_SAMPLE_RATE,
};

const PCM_BITS: u32 = 16;
const AAC_PROFILE_LC: u32 = 0x29;

#[derive(Debug)]
struct DecoderState {
    transform: IMFTransform,
    out_size: u32,
    provides_samples: bool,
}

/// # Example
///
/// ```ignore
/// use g2g_plugins::mfaacdecode::MfAacDecode;
///
/// let decode = MfAacDecode::new().with_audio_specific_config(vec![0x12, 0x10]);
/// ```
#[derive(Debug)]
pub struct MfAacDecode {
    state: Option<DecoderState>,
    com_started: bool,
    configured: bool,
    channels: u8,
    sample_rate: u32,
    /// AudioSpecificConfig supplied before configuration.
    asc: Vec<u8>,
    /// Running output-sample-frame count, for monotonic PCM timing.
    out_frames: u64,
    output: PcmOutput,
}

// SAFETY: same contract as `MfDecode`: COM is MTA, the MFT is moved between
// threads but never aliased, and the runner drives it through `&mut self`.
unsafe impl Send for MfAacDecode {}

impl Default for MfAacDecode {
    fn default() -> Self {
        Self::new()
    }
}

impl MfAacDecode {
    pub fn new() -> Self {
        Self {
            state: None,
            com_started: false,
            configured: false,
            channels: 0,
            sample_rate: 0,
            asc: Vec::new(),
            out_frames: 0,
            output: PcmOutput::default(),
        }
    }

    /// Supply the stream's AudioSpecificConfig (from the MP4 `esds`). Required
    /// for raw access units, which carry no ADTS header to read it from.
    pub fn with_audio_specific_config(mut self, asc: impl Into<Vec<u8>>) -> Self {
        self.asc = asc.into();
        self
    }

    /// Count of decoded PCM `DataFrame`s pushed downstream. Useful in tests.
    pub fn decoded_count(&self) -> u64 {
        self.output.frames_pushed()
    }

    fn open(&mut self, asc: &[u8]) -> Result<(), G2gError> {
        let info =
            crate::aacparse::audio_specific_config_info(asc).ok_or(G2gError::CapsMismatch)?;
        self.state = Some(init_decoder(info.channels, info.sample_rate, asc)?);
        self.channels = info.channels;
        self.sample_rate = info.sample_rate;
        Ok(())
    }

    fn feed(
        &mut self,
        data: &[u8],
        pts_ns: u64,
        decoded: &mut Vec<Vec<u8>>,
    ) -> Result<(), G2gError> {
        let sample = make_input_sample(data, pts_ns)?;
        feed_sample(self, &sample, Self::transform, |element| {
            element.drain(decoded)
        })
    }

    fn transform(&self) -> Result<&IMFTransform, G2gError> {
        Ok(&self
            .state
            .as_ref()
            .ok_or(G2gError::NotConfigured)?
            .transform)
    }

    fn drain(&mut self, decoded: &mut Vec<Vec<u8>>) -> Result<(), G2gError> {
        loop {
            match self.process_output()? {
                Some(pcm) => decoded.push(pcm),
                None => return Ok(()),
            }
        }
    }

    fn drain_eos(&mut self, decoded: &mut Vec<Vec<u8>>) -> Result<(), G2gError> {
        send_drain_command(self.transform()?)?;
        self.drain(decoded)
    }

    fn process_output(&self) -> Result<Option<Vec<u8>>, G2gError> {
        let st = self.state.as_ref().ok_or(G2gError::NotConfigured)?;
        // The first output is a stream change re-asserting the PCM type.
        let OutputPull::Sample(sample) =
            process_output(&st.transform, st.provides_samples, st.out_size)?
        else {
            return Ok(None);
        };
        let pcm = read_sample_bytes(&sample, <[u8]>::to_vec)?;
        Ok((!pcm.is_empty()).then_some(pcm))
    }
}

impl AsyncElement for MfAacDecode {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::Audio {
                format: AudioFormat::Aac,
                ..
            } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// Native `DerivedOutput`: AAC in maps to S16 PCM at the same channels/rate,
    /// or to the decoder placeholders when the input leaves either open.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::Audio {
                format: AudioFormat::Aac,
                channels,
                sample_rate,
                ..
            } if *channels == ANY_CHANNELS || *sample_rate == ANY_SAMPLE_RATE => {
                pcm_s16_decoder_output()
            }
            Caps::Audio {
                format: AudioFormat::Aac,
                channels,
                sample_rate,
                ..
            } => CapsSet::one(Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: *channels,
                sample_rate: *sample_rate,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn metadata(&self) -> g2g_core::ElementMetadata {
        g2g_core::ElementMetadata::new(
            "Media Foundation AAC decoder",
            "Codec/Decoder/Audio",
            "Decodes AAC-LC to interleaved PcmS16Le with the Media Foundation AAC decoder MFT",
            crate::AARON_BOXER,
        )
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(
            absolute_caps,
            Caps::Audio {
                format: AudioFormat::Aac,
                ..
            }
        ) {
            return Err(G2gError::CapsMismatch);
        }

        // SAFETY: COM/MF startup on the calling (owning) thread.
        unsafe {
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
            MFStartup(MF_VERSION, MFSTARTUP_FULL).map_err(mf_err)?;
        }
        self.com_started = true;

        if !self.asc.is_empty() {
            let asc = core::mem::take(&mut self.asc);
            self.open(&asc)?;
            self.asc = asc;
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
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
            let mut decoded = Vec::new();
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let slice = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    let adts_config = crate::aacparse::asc_from_adts(slice);
                    if let (None, Some(asc)) = (&self.state, adts_config) {
                        self.open(&asc)?;
                    }
                    let access_unit = crate::aacparse::strip_adts(slice);
                    self.feed(access_unit, frame.timing.pts_ns, &mut decoded)?;
                }
                PipelinePacket::CapsChanged(c) => match &c {
                    Caps::Audio {
                        format: AudioFormat::Aac,
                        ..
                    } => {}
                    Caps::Audio {
                        format: AudioFormat::PcmS16Le,
                        ..
                    } => self.output.forward_caps(c, out).await?,
                    _ => return Err(G2gError::CapsMismatch),
                },
                PipelinePacket::Flush => {
                    if let Some(st) = self.state.as_ref() {
                        // SAFETY: flush message on the owning thread.
                        unsafe {
                            st.transform
                                .ProcessMessage(MFT_MESSAGE_COMMAND_FLUSH, 0)
                                .map_err(mf_err)?;
                        }
                    }
                    self.output.forget_caps();
                    out.push(PipelinePacket::Flush).await?;
                    return Ok(());
                }
                PipelinePacket::Eos => {
                    self.drain_eos(&mut decoded)?;
                }
                PipelinePacket::Segment(seg) => {
                    self.output.forward_segment(seg, out).await?;
                }
                other => {
                    out.push(other).await?;
                }
            }

            let frame_bytes = 2 * u64::from(self.channels.max(1));
            for pcm in decoded {
                let pts_ns = ticks_to_ns(self.out_frames, self.sample_rate.max(1));
                self.out_frames += pcm.len() as u64 / frame_bytes;
                let timing = FrameTiming {
                    pts_ns,
                    dts_ns: pts_ns,
                    capture_ns: pts_ns,
                    ..FrameTiming::default()
                };
                let decoded = DecodedPcm {
                    pcm,
                    channels: self.channels,
                    sample_rate: self.sample_rate,
                    channel_layout: ChannelLayout::UNSPECIFIED,
                };
                self.output.push(timing, decoded, out).await?;
            }
            Ok(())
        })
    }
}

impl PadTemplates for MfAacDecode {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(crate::typefind::elementary_audio_caps(
                AudioFormat::Aac,
            ))),
            PadTemplate::source(pcm_caps_any_rate_and_channels(|format| {
                format == AudioFormat::PcmS16Le
            })),
        ])
    }
}

impl Drop for MfAacDecode {
    fn drop(&mut self) {
        self.state = None;
        if self.com_started {
            // SAFETY: paired with the startup in configure_pipeline, same thread.
            unsafe {
                let _ = MFShutdown();
                CoUninitialize();
            }
        }
    }
}

/// Create + configure the AAC decoder MFT for the given stream.
fn init_decoder(channels: u8, sample_rate: u32, asc: &[u8]) -> Result<DecoderState, G2gError> {
    // SAFETY: COM object creation on the owning thread.
    let transform: IMFTransform =
        unsafe { CoCreateInstance(&CLSID_MSAACDecMFT, None, CLSCTX_INPROC_SERVER) }
            .map_err(mf_err)?;

    let user_data = heaac_user_data(asc);
    let block_align = 2 * channels as u32;

    // SAFETY: media-type configuration on the owning thread.
    unsafe {
        let input = MFCreateMediaType().map_err(mf_err)?;
        input
            .SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Audio)
            .map_err(mf_err)?;
        input
            .SetGUID(&MF_MT_SUBTYPE, &MFAudioFormat_AAC)
            .map_err(mf_err)?;
        input
            .SetUINT32(&MF_MT_AUDIO_BITS_PER_SAMPLE, PCM_BITS)
            .map_err(mf_err)?;
        input
            .SetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND, sample_rate)
            .map_err(mf_err)?;
        input
            .SetUINT32(&MF_MT_AUDIO_NUM_CHANNELS, channels as u32)
            .map_err(mf_err)?;
        input
            .SetUINT32(&MF_MT_AAC_PAYLOAD_TYPE, 0)
            .map_err(mf_err)?;
        input
            .SetUINT32(&MF_MT_AAC_AUDIO_PROFILE_LEVEL_INDICATION, AAC_PROFILE_LC)
            .map_err(mf_err)?;
        input
            .SetBlob(&MF_MT_USER_DATA, &user_data)
            .map_err(mf_err)?;
        transform.SetInputType(0, &input, 0).map_err(mf_err)?;

        let output = MFCreateMediaType().map_err(mf_err)?;
        output
            .SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Audio)
            .map_err(mf_err)?;
        output
            .SetGUID(&MF_MT_SUBTYPE, &MFAudioFormat_PCM)
            .map_err(mf_err)?;
        output
            .SetUINT32(&MF_MT_AUDIO_BITS_PER_SAMPLE, PCM_BITS)
            .map_err(mf_err)?;
        output
            .SetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND, sample_rate)
            .map_err(mf_err)?;
        output
            .SetUINT32(&MF_MT_AUDIO_NUM_CHANNELS, channels as u32)
            .map_err(mf_err)?;
        output
            .SetUINT32(&MF_MT_AUDIO_BLOCK_ALIGNMENT, block_align)
            .map_err(mf_err)?;
        output
            .SetUINT32(&MF_MT_AUDIO_AVG_BYTES_PER_SECOND, sample_rate * block_align)
            .map_err(mf_err)?;
        transform.SetOutputType(0, &output, 0).map_err(mf_err)?;
    }

    let out_size = output_buffer_size(&transform, channels, sample_rate)?;
    let provides_samples = output_provides_samples(&transform)?;

    // SAFETY: streaming-mode messages on the owning thread.
    unsafe {
        transform
            .ProcessMessage(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0)
            .map_err(mf_err)?;
        transform
            .ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0)
            .map_err(mf_err)?;
    }

    Ok(DecoderState {
        transform,
        out_size,
        provides_samples,
    })
}

/// Build the `MF_MT_USER_DATA` blob the decoder wants: the 12-byte
/// HEAACWAVEINFO tail (payload type 0, LC profile) followed by the ASC.
fn heaac_user_data(asc: &[u8]) -> Vec<u8> {
    let mut d = Vec::with_capacity(12 + asc.len());
    d.extend_from_slice(&0u16.to_le_bytes()); // wPayloadType: raw AAC
    d.extend_from_slice(&(AAC_PROFILE_LC as u16).to_le_bytes()); // wAudioProfileLevelIndication
    d.extend_from_slice(&0u16.to_le_bytes()); // wStructType
    d.extend_from_slice(&0u16.to_le_bytes()); // wReserved1
    d.extend_from_slice(&0u32.to_le_bytes()); // dwReserved2
    d.extend_from_slice(asc);
    d
}

fn output_buffer_size(transform: &IMFTransform, channels: u8, rate: u32) -> Result<u32, G2gError> {
    // SAFETY: stream-info query on the owning thread.
    let info = unsafe { transform.GetOutputStreamInfo(0) }.map_err(mf_err)?;
    // Floor at ~1 s of PCM so an early zero estimate never under-allocates.
    let floor = rate
        .saturating_mul(channels as u32)
        .saturating_mul(2)
        .max(8192);
    Ok(info.cbSize.max(floor.min(1 << 20)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intercept_accepts_aac_rejects_other() {
        let dec = MfAacDecode::new();
        let aac = Caps::Audio {
            format: AudioFormat::Aac,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        assert_eq!(dec.intercept_caps(&aac), Ok(aac));
        let pcm = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        assert_eq!(dec.intercept_caps(&pcm), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn derived_output_maps_aac_to_pcm() {
        let dec = MfAacDecode::new();
        let CapsConstraint::DerivedOutput(f) = dec.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        let out = f(&Caps::Audio {
            format: AudioFormat::Aac,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        });
        assert_eq!(
            out.alternatives(),
            &[Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: 2,
                sample_rate: 48_000,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            }]
        );
    }

    #[test]
    fn user_data_prefixes_asc_with_heaac_info() {
        let asc = [0x12u8, 0x10];
        let d = heaac_user_data(&asc);
        assert_eq!(d.len(), 12 + 2);
        assert_eq!(&d[12..], &asc, "ASC follows the 12-byte info header");
        assert_eq!(u16::from_le_bytes([d[0], d[1]]), 0, "payload type raw AAC");
    }
}
