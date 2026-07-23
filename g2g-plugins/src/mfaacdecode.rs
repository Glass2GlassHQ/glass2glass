//! Windows AAC audio decode element (M36) wrapping the Media Foundation AAC
//! Decoder MFT (`CLSID_MSAACDecMFT`). Consumes raw AAC-LC access units
//! (`AudioFormat::Aac`, `MemoryDomain::System`) and produces interleaved 16-bit
//! PCM (`PcmS16Le`), the decode-side mirror of `MfAacEncode`.
//!
//! The decoder needs the stream's AudioSpecificConfig to configure its input
//! type; supply it with [`MfAacDecode::with_audio_specific_config`] (the
//! encoder exposes it, and the MP4 `esds` carries it). The MS AAC decoder is
//! synchronous, so the same `ProcessInput`/`ProcessOutput` drain loop as the
//! H.264 decoder applies.
//!
//! Threading: COM is MTA, every call on the owning thread; `Send` is asserted
//! under the same documented contract as `MfDecode`.

use core::future::Future;
use core::mem::ManuallyDrop;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use windows::Win32::Media::MediaFoundation::{
    CLSID_MSAACDecMFT, IMFSample, IMFTransform, MFAudioFormat_AAC, MFAudioFormat_PCM,
    MFCreateMediaType, MFCreateMemoryBuffer, MFCreateSample, MFMediaType_Audio, MFShutdown,
    MFStartup, MFSTARTUP_FULL, MFT_MESSAGE_COMMAND_DRAIN, MFT_MESSAGE_COMMAND_FLUSH,
    MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, MFT_MESSAGE_NOTIFY_START_OF_STREAM, MFT_OUTPUT_DATA_BUFFER,
    MFT_OUTPUT_STREAM_PROVIDES_SAMPLES, MF_E_NOTACCEPTING, MF_E_TRANSFORM_NEED_MORE_INPUT,
    MF_E_TRANSFORM_STREAM_CHANGE, MF_MT_AAC_AUDIO_PROFILE_LEVEL_INDICATION, MF_MT_AAC_PAYLOAD_TYPE,
    MF_MT_AUDIO_AVG_BYTES_PER_SECOND, MF_MT_AUDIO_BITS_PER_SAMPLE, MF_MT_AUDIO_BLOCK_ALIGNMENT,
    MF_MT_AUDIO_NUM_CHANNELS, MF_MT_AUDIO_SAMPLES_PER_SECOND, MF_MT_MAJOR_TYPE, MF_MT_SUBTYPE,
    MF_MT_USER_DATA, MF_VERSION,
};
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoUninitialize, CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED,
};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, FrameTiming,
    G2gError, HardwareError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
};

const PCM_BITS: u32 = 16;
const AAC_PROFILE_LC: u32 = 0x29;

#[derive(Debug)]
struct DecoderState {
    transform: IMFTransform,
    out_size: u32,
    provides_samples: bool,
}

#[derive(Debug)]
pub struct MfAacDecode {
    state: Option<DecoderState>,
    com_started: bool,
    configured: bool,
    channels: u8,
    sample_rate: u32,
    /// AudioSpecificConfig supplied before configuration.
    asc: Vec<u8>,
    last_caps: Option<Caps>,
    /// Running output-sample-frame count, for monotonic PCM timing.
    out_frames: u64,
    emitted: u64,
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
            last_caps: None,
            out_frames: 0,
            emitted: 0,
        }
    }

    /// Supply the stream's AudioSpecificConfig (from the encoder or the MP4
    /// `esds`). Required before `configure_pipeline`.
    pub fn with_audio_specific_config(mut self, asc: impl Into<Vec<u8>>) -> Self {
        self.asc = asc.into();
        self
    }

    /// Count of decoded PCM `DataFrame`s pushed downstream. Useful in tests.
    pub fn decoded_count(&self) -> u64 {
        self.emitted
    }

    fn feed(
        &mut self,
        data: &[u8],
        pts_ns: u64,
        decoded: &mut Vec<Vec<u8>>,
    ) -> Result<(), G2gError> {
        let sample = make_input_sample(data, pts_ns)?;
        let mut guard = 0u32;
        while !self.process_input(&sample)? {
            self.drain(decoded)?;
            guard += 1;
            if guard > 64 {
                return Err(G2gError::Hardware(HardwareError::Other));
            }
        }
        self.drain(decoded)
    }

    fn process_input(&self, sample: &IMFSample) -> Result<bool, G2gError> {
        let st = self.state.as_ref().ok_or(G2gError::NotConfigured)?;
        // SAFETY: COM call on the owning thread; `sample` is valid.
        match unsafe { st.transform.ProcessInput(0, sample, 0) } {
            Ok(()) => Ok(true),
            Err(e) if e.code() == MF_E_NOTACCEPTING => Ok(false),
            Err(e) => Err(mf_err(e)),
        }
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
        {
            let st = self.state.as_ref().ok_or(G2gError::NotConfigured)?;
            // SAFETY: drain message on the owning thread.
            unsafe {
                st.transform
                    .ProcessMessage(MFT_MESSAGE_COMMAND_DRAIN, 0)
                    .map_err(mf_err)?;
            }
        }
        self.drain(decoded)
    }

    fn process_output(&self) -> Result<Option<Vec<u8>>, G2gError> {
        let st = self.state.as_ref().ok_or(G2gError::NotConfigured)?;
        // SAFETY: every call on the owning thread. Output sample preallocated
        // unless the MFT provides its own; FFI refs reclaimed after the call.
        unsafe {
            let preallocated = if st.provides_samples {
                None
            } else {
                let buffer = MFCreateMemoryBuffer(st.out_size).map_err(mf_err)?;
                let sample = MFCreateSample().map_err(mf_err)?;
                sample.AddBuffer(&buffer).map_err(mf_err)?;
                Some(sample)
            };
            let mut out = [MFT_OUTPUT_DATA_BUFFER {
                dwStreamID: 0,
                pSample: ManuallyDrop::new(preallocated),
                dwStatus: 0,
                pEvents: ManuallyDrop::new(None),
            }];
            let mut status = 0u32;
            let r = st.transform.ProcessOutput(0, &mut out, &mut status);
            let out_sample = ManuallyDrop::into_inner(core::mem::replace(
                &mut out[0].pSample,
                ManuallyDrop::new(None),
            ));
            drop(ManuallyDrop::into_inner(core::mem::replace(
                &mut out[0].pEvents,
                ManuallyDrop::new(None),
            )));
            match r {
                Ok(()) => {
                    let sample = out_sample.ok_or(G2gError::Hardware(HardwareError::Other))?;
                    let pcm = copy_sample(&sample)?;
                    if pcm.is_empty() {
                        Ok(None)
                    } else {
                        Ok(Some(pcm))
                    }
                }
                Err(e) if e.code() == MF_E_TRANSFORM_NEED_MORE_INPUT => Ok(None),
                // The AAC decoder re-asserts its PCM output type on the first
                // output; re-pick it and retry.
                Err(e) if e.code() == MF_E_TRANSFORM_STREAM_CHANGE => Ok(None),
                Err(e) => Err(mf_err(e)),
            }
        }
    }
}

impl AsyncElement for MfAacDecode {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::Audio {
                format: AudioFormat::Aac,
                ..
            } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// Native `DerivedOutput`: AAC in maps to S16 PCM at the same channels/rate.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::Audio {
                format: AudioFormat::Aac,
                channels,
                sample_rate,
            } => CapsSet::one(Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: *channels,
                sample_rate: *sample_rate,
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (channels, sample_rate) = match absolute_caps {
            Caps::Audio {
                format: AudioFormat::Aac,
                channels,
                sample_rate,
            } => (*channels, *sample_rate),
            _ => return Err(G2gError::CapsMismatch),
        };
        if channels == 0 || sample_rate == 0 || self.asc.is_empty() {
            return Err(G2gError::CapsMismatch);
        }

        // SAFETY: COM/MF startup on the calling (owning) thread.
        unsafe {
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
            MFStartup(MF_VERSION, MFSTARTUP_FULL).map_err(mf_err)?;
        }
        self.com_started = true;

        self.state = Some(init_decoder(channels, sample_rate, &self.asc)?);
        self.channels = channels;
        self.sample_rate = sample_rate;
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
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.feed(slice, frame.timing.pts_ns, &mut decoded)?;
                }
                PipelinePacket::CapsChanged(c) => match &c {
                    Caps::Audio {
                        format: AudioFormat::Aac,
                        channels,
                        sample_rate,
                    } if *channels == self.channels && *sample_rate == self.sample_rate => {}
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
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                    return Ok(());
                }
                PipelinePacket::Eos => {
                    self.drain_eos(&mut decoded)?;
                }
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                other => {
                    out.push(other).await?;
                }
            }

            let pcm_caps = Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: self.channels,
                sample_rate: self.sample_rate,
            };
            let frame_bytes = 2 * self.channels.max(1) as u64; // 16-bit
            for pcm in decoded {
                if self.last_caps.as_ref() != Some(&pcm_caps) {
                    out.push(PipelinePacket::CapsChanged(pcm_caps.clone()))
                        .await?;
                    self.last_caps = Some(pcm_caps.clone());
                }
                let pts_ns = self.out_frames * 1_000_000_000 / self.sample_rate.max(1) as u64;
                let n_frames = pcm.len() as u64 / frame_bytes;
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(pcm.into_boxed_slice())),
                    timing: FrameTiming {
                        pts_ns,
                        dts_ns: pts_ns,
                        capture_ns: pts_ns,
                        ..FrameTiming::default()
                    },
                    sequence: self.emitted,
                    meta: Default::default(),
                };
                self.out_frames += n_frames;
                self.emitted += 1;
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            Ok(())
        })
    }
}

impl PadTemplates for MfAacDecode {
    /// AAC in, S16 PCM out.
    fn pad_templates() -> Vec<PadTemplate> {
        let aac = Caps::Audio {
            format: AudioFormat::Aac,
            channels: 2,
            sample_rate: 48_000,
        };
        let pcm = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::one(aac)),
            PadTemplate::source(CapsSet::one(pcm)),
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

fn output_provides_samples(transform: &IMFTransform) -> Result<bool, G2gError> {
    // SAFETY: stream-info query on the owning thread.
    let info = unsafe { transform.GetOutputStreamInfo(0) }.map_err(mf_err)?;
    Ok(info.dwFlags & MFT_OUTPUT_STREAM_PROVIDES_SAMPLES.0 as u32 != 0)
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

fn make_input_sample(data: &[u8], pts_ns: u64) -> Result<IMFSample, G2gError> {
    let len = data.len() as u32;
    // SAFETY: buffer allocation + locked copy + sample assembly on the owning
    // thread; `ptr` valid for `len` bytes between Lock and Unlock.
    unsafe {
        let buffer = MFCreateMemoryBuffer(len).map_err(mf_err)?;
        let mut ptr: *mut u8 = core::ptr::null_mut();
        buffer.Lock(&mut ptr, None, None).map_err(mf_err)?;
        core::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        buffer.Unlock().map_err(mf_err)?;
        buffer.SetCurrentLength(len).map_err(mf_err)?;
        let sample = MFCreateSample().map_err(mf_err)?;
        sample.AddBuffer(&buffer).map_err(mf_err)?;
        sample
            .SetSampleTime((pts_ns / 100) as i64)
            .map_err(mf_err)?;
        Ok(sample)
    }
}

fn copy_sample(sample: &IMFSample) -> Result<Vec<u8>, G2gError> {
    // SAFETY: contiguous-buffer access on the owning thread.
    unsafe {
        let buffer = sample.ConvertToContiguousBuffer().map_err(mf_err)?;
        let mut ptr: *mut u8 = core::ptr::null_mut();
        let mut len: u32 = 0;
        buffer
            .Lock(&mut ptr, None, Some(&mut len))
            .map_err(mf_err)?;
        let bytes = core::slice::from_raw_parts(ptr, len as usize).to_vec();
        buffer.Unlock().map_err(mf_err)?;
        Ok(bytes)
    }
}

fn mf_err(e: windows::core::Error) -> G2gError {
    G2gError::Hardware(HardwareError::MediaFoundation(e.code().0))
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
        };
        assert_eq!(dec.intercept_caps(&aac), Ok(aac));
        let pcm = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
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
        });
        assert_eq!(
            out.alternatives(),
            &[Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: 2,
                sample_rate: 48_000,
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

    #[test]
    fn configure_without_asc_fails_loud() {
        let mut dec = MfAacDecode::new();
        let aac = Caps::Audio {
            format: AudioFormat::Aac,
            channels: 2,
            sample_rate: 48_000,
        };
        assert!(matches!(
            dec.configure_pipeline(&aac),
            Err(G2gError::CapsMismatch)
        ));
    }
}
