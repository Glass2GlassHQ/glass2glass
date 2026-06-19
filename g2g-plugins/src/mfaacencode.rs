//! Windows AAC audio encode element (M35) wrapping the Media Foundation AAC
//! Encoder MFT. Consumes interleaved 16-bit PCM (`PcmS16Le`) `DataFrame`s and
//! produces raw AAC-LC access units (`AudioFormat::Aac`, `MemoryDomain::System`,
//! one access unit per 1024-sample frame, no ADTS header), the compressed-audio
//! analog of `MfEncode`.
//!
//! The AAC encoder has no fixed CLSID, so it is enumerated by output subtype
//! via `MFTEnumEx` (the MS AAC encoder is synchronous, so the same
//! `ProcessInput`/`ProcessOutput` drain loop as the H.264 encoder applies). The
//! AudioSpecificConfig the decoder/container needs is read from the negotiated
//! output type's `MF_MT_USER_DATA` and exposed via [`MfAacEncode::audio_specific_config`].
//!
//! Threading: COM is MTA, every call on the owning thread; `Send` is asserted
//! under the same documented contract as `MfEncode`.

use core::future::Future;
use core::mem::ManuallyDrop;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use windows::Win32::Media::MediaFoundation::{
    IMFActivate, IMFSample, IMFTransform, MFCreateMediaType, MFCreateMemoryBuffer, MFCreateSample,
    MFShutdown, MFStartup, MFTEnumEx, MFAudioFormat_AAC, MFAudioFormat_PCM, MFMediaType_Audio,
    MFSTARTUP_FULL, MFT_CATEGORY_AUDIO_ENCODER, MFT_ENUM_FLAG_SORTANDFILTER, MFT_ENUM_FLAG_SYNCMFT,
    MFT_MESSAGE_COMMAND_DRAIN, MFT_MESSAGE_COMMAND_FLUSH, MFT_MESSAGE_NOTIFY_BEGIN_STREAMING,
    MFT_MESSAGE_NOTIFY_START_OF_STREAM, MFT_OUTPUT_DATA_BUFFER, MFT_OUTPUT_STREAM_PROVIDES_SAMPLES,
    MFT_REGISTER_TYPE_INFO, MF_E_NOTACCEPTING, MF_E_TRANSFORM_NEED_MORE_INPUT,
    MF_E_TRANSFORM_STREAM_CHANGE, MF_MT_AAC_AUDIO_PROFILE_LEVEL_INDICATION, MF_MT_AAC_PAYLOAD_TYPE,
    MF_MT_AUDIO_AVG_BYTES_PER_SECOND, MF_MT_AUDIO_BITS_PER_SAMPLE, MF_MT_AUDIO_BLOCK_ALIGNMENT,
    MF_MT_AUDIO_NUM_CHANNELS, MF_MT_AUDIO_SAMPLES_PER_SECOND, MF_MT_MAJOR_TYPE, MF_MT_SUBTYPE,
    MF_MT_USER_DATA, MF_VERSION,
};
use windows::Win32::System::Com::{
    CoInitializeEx, CoTaskMemFree, CoUninitialize, COINIT_MULTITHREADED,
};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, FrameTiming,
    G2gError, HardwareError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
};

/// Samples per AAC-LC access unit.
const AAC_FRAME_SAMPLES: u64 = 1024;
/// 16-bit PCM input.
const PCM_BITS: u32 = 16;
/// AAC-LC profile-level indication.
const AAC_PROFILE_LC: u32 = 0x29;
/// HEAACWAVEINFO carries 12 bytes before the AudioSpecificConfig in the
/// output type's `MF_MT_USER_DATA` blob.
const HEAAC_INFO_BYTES: usize = 12;
/// Default output bitrate as bytes/sec (128 kbps). Must be one of the values
/// the MS AAC encoder accepts.
const DEFAULT_BYTES_PER_SEC: u32 = 16_000;
/// Bytes/sec values the MS AAC encoder advertises (96/128/160/192 kbps).
const VALID_BYTES_PER_SEC: [u32; 4] = [12_000, 16_000, 20_000, 24_000];

#[derive(Debug)]
struct EncoderState {
    transform: IMFTransform,
    out_size: u32,
    provides_samples: bool,
}

#[derive(Debug)]
pub struct MfAacEncode {
    state: Option<EncoderState>,
    com_started: bool,
    configured: bool,
    channels: u8,
    sample_rate: u32,
    bytes_per_sec: u32,
    /// AudioSpecificConfig of the negotiated stream, read from the output type.
    asc: Option<Vec<u8>>,
    last_caps: Option<Caps>,
    emitted: u64,
}

// SAFETY: same contract as `MfEncode`: COM is MTA, the MFT is moved between
// threads but never aliased, and the runner drives it through `&mut self`.
unsafe impl Send for MfAacEncode {}

impl Default for MfAacEncode {
    fn default() -> Self {
        Self::new()
    }
}

impl MfAacEncode {
    pub fn new() -> Self {
        Self {
            state: None,
            com_started: false,
            configured: false,
            channels: 0,
            sample_rate: 0,
            bytes_per_sec: DEFAULT_BYTES_PER_SEC,
            asc: None,
            last_caps: None,
            emitted: 0,
        }
    }

    /// Override the AAC output bitrate, given as bytes/sec. Must be one of
    /// 12000/16000/20000/24000 (96/128/160/192 kbps); other values are rejected
    /// at `configure_pipeline`. Call before `configure_pipeline`.
    pub fn with_bytes_per_second(mut self, bytes_per_sec: u32) -> Self {
        self.bytes_per_sec = bytes_per_sec;
        self
    }

    /// The AudioSpecificConfig of the negotiated stream (available after
    /// `configure_pipeline`), needed to configure a decoder or an MP4 `esds`.
    pub fn audio_specific_config(&self) -> Option<&[u8]> {
        self.asc.as_deref()
    }

    /// Count of AAC access units pushed downstream. Useful in tests.
    pub fn encoded_count(&self) -> u64 {
        self.emitted
    }

    fn feed(&mut self, data: &[u8], pts_ns: u64, encoded: &mut Vec<Vec<u8>>) -> Result<(), G2gError> {
        let sample = make_input_sample(data, pts_ns)?;
        let mut guard = 0u32;
        while !self.process_input(&sample)? {
            self.drain(encoded)?;
            guard += 1;
            if guard > 64 {
                return Err(G2gError::Hardware(HardwareError::Other));
            }
        }
        self.drain(encoded)
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

    fn drain(&mut self, encoded: &mut Vec<Vec<u8>>) -> Result<(), G2gError> {
        loop {
            match self.process_output()? {
                Some(au) => encoded.push(au),
                None => return Ok(()),
            }
        }
    }

    fn drain_eos(&mut self, encoded: &mut Vec<Vec<u8>>) -> Result<(), G2gError> {
        {
            let st = self.state.as_ref().ok_or(G2gError::NotConfigured)?;
            // SAFETY: drain message on the owning thread.
            unsafe {
                st.transform
                    .ProcessMessage(MFT_MESSAGE_COMMAND_DRAIN, 0)
                    .map_err(mf_err)?;
            }
        }
        self.drain(encoded)
    }

    /// One `ProcessOutput`; `Some(au)` on an emitted access unit, `None` when
    /// the MFT needs more input.
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
                    Ok(Some(copy_sample(&sample)?))
                }
                Err(e) if e.code() == MF_E_TRANSFORM_NEED_MORE_INPUT => Ok(None),
                // The audio encoder does not change output type mid-stream;
                // treat a stream change defensively as needing input.
                Err(e) if e.code() == MF_E_TRANSFORM_STREAM_CHANGE => Ok(None),
                Err(e) => Err(mf_err(e)),
            }
        }
    }
}

impl AsyncElement for MfAacEncode {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::Audio {
                format: AudioFormat::PcmS16Le,
                ..
            } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// Native `DerivedOutput`: S16 PCM in maps to AAC at the same channels/rate.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels,
                sample_rate,
            } => CapsSet::one(Caps::Audio {
                format: AudioFormat::Aac,
                channels: *channels,
                sample_rate: *sample_rate,
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (channels, sample_rate) = match absolute_caps {
            Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels,
                sample_rate,
            } => (*channels, *sample_rate),
            _ => return Err(G2gError::CapsMismatch),
        };
        if channels == 0 || sample_rate == 0 || !VALID_BYTES_PER_SEC.contains(&self.bytes_per_sec) {
            return Err(G2gError::CapsMismatch);
        }

        // SAFETY: COM/MF startup on the calling (owning) thread.
        unsafe {
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
            MFStartup(MF_VERSION, MFSTARTUP_FULL).map_err(mf_err)?;
        }
        self.com_started = true;

        let (state, asc) = init_encoder(channels, sample_rate, self.bytes_per_sec)?;
        self.state = Some(state);
        self.asc = Some(asc);
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
            let mut encoded = Vec::new();
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.feed(slice.as_slice(), frame.timing.pts_ns, &mut encoded)?;
                }
                PipelinePacket::CapsChanged(c) => {
                    // Only the configured PCM format passes; a change is rejected.
                    match &c {
                        Caps::Audio {
                            format: AudioFormat::PcmS16Le,
                            channels,
                            sample_rate,
                        } if *channels == self.channels && *sample_rate == self.sample_rate => {}
                        _ => return Err(G2gError::CapsMismatch),
                    }
                }
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
                    self.drain_eos(&mut encoded)?;
                }
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
            }

            let aac_caps = Caps::Audio {
                format: AudioFormat::Aac,
                channels: self.channels,
                sample_rate: self.sample_rate,
            };
            // Each access unit covers AAC_FRAME_SAMPLES; derive pts from the
            // emitted-frame count so the stream is monotonic and gap-free.
            let ns_per_frame = AAC_FRAME_SAMPLES * 1_000_000_000 / self.sample_rate.max(1) as u64;
            for au in encoded {
                if self.last_caps.as_ref() != Some(&aac_caps) {
                    out.push(PipelinePacket::CapsChanged(aac_caps.clone())).await?;
                    self.last_caps = Some(aac_caps.clone());
                }
                let pts_ns = self.emitted * ns_per_frame;
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(au.into_boxed_slice())),
                    timing: FrameTiming {
                        pts_ns,
                        dts_ns: pts_ns,
                        duration_ns: ns_per_frame,
                        capture_ns: pts_ns,
                        ..FrameTiming::default()
                    },
                    sequence: self.emitted,
                };
                self.emitted += 1;
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            Ok(())
        })
    }
}

impl PadTemplates for MfAacEncode {
    /// S16 PCM in, AAC out; `Caps::Audio` has no open dims, so the templates
    /// pin the common stereo/48 kHz shape.
    fn pad_templates() -> Vec<PadTemplate> {
        let pcm = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
        };
        let aac = Caps::Audio {
            format: AudioFormat::Aac,
            channels: 2,
            sample_rate: 48_000,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::one(pcm)),
            PadTemplate::source(CapsSet::one(aac)),
        ])
    }
}

impl Drop for MfAacEncode {
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

/// Create + configure the AAC encoder MFT and return its state plus the
/// negotiated AudioSpecificConfig.
fn init_encoder(
    channels: u8,
    sample_rate: u32,
    bytes_per_sec: u32,
) -> Result<(EncoderState, Vec<u8>), G2gError> {
    let transform = enumerate_aac_encoder()?;
    let block_align = 2 * channels as u32; // 16-bit PCM

    // SAFETY: media-type configuration on the owning thread.
    unsafe {
        let output = MFCreateMediaType().map_err(mf_err)?;
        output.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Audio).map_err(mf_err)?;
        output.SetGUID(&MF_MT_SUBTYPE, &MFAudioFormat_AAC).map_err(mf_err)?;
        output.SetUINT32(&MF_MT_AUDIO_BITS_PER_SAMPLE, PCM_BITS).map_err(mf_err)?;
        output.SetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND, sample_rate).map_err(mf_err)?;
        output.SetUINT32(&MF_MT_AUDIO_NUM_CHANNELS, channels as u32).map_err(mf_err)?;
        output.SetUINT32(&MF_MT_AUDIO_AVG_BYTES_PER_SECOND, bytes_per_sec).map_err(mf_err)?;
        output.SetUINT32(&MF_MT_AAC_PAYLOAD_TYPE, 0).map_err(mf_err)?; // raw AAC
        output
            .SetUINT32(&MF_MT_AAC_AUDIO_PROFILE_LEVEL_INDICATION, AAC_PROFILE_LC)
            .map_err(mf_err)?;
        transform.SetOutputType(0, &output, 0).map_err(mf_err)?;

        let input = MFCreateMediaType().map_err(mf_err)?;
        input.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Audio).map_err(mf_err)?;
        input.SetGUID(&MF_MT_SUBTYPE, &MFAudioFormat_PCM).map_err(mf_err)?;
        input.SetUINT32(&MF_MT_AUDIO_BITS_PER_SAMPLE, PCM_BITS).map_err(mf_err)?;
        input.SetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND, sample_rate).map_err(mf_err)?;
        input.SetUINT32(&MF_MT_AUDIO_NUM_CHANNELS, channels as u32).map_err(mf_err)?;
        input.SetUINT32(&MF_MT_AUDIO_BLOCK_ALIGNMENT, block_align).map_err(mf_err)?;
        input
            .SetUINT32(&MF_MT_AUDIO_AVG_BYTES_PER_SECOND, sample_rate * block_align)
            .map_err(mf_err)?;
        transform.SetInputType(0, &input, 0).map_err(mf_err)?;
    }

    let asc = read_audio_specific_config(&transform)?;
    let out_size = output_buffer_size(&transform)?;
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

    Ok((
        EncoderState {
            transform,
            out_size,
            provides_samples,
        },
        asc,
    ))
}

/// Enumerate and activate the (synchronous) MS AAC encoder MFT.
fn enumerate_aac_encoder() -> Result<IMFTransform, G2gError> {
    let out_info = MFT_REGISTER_TYPE_INFO {
        guidMajorType: MFMediaType_Audio,
        guidSubtype: MFAudioFormat_AAC,
    };
    let flags = MFT_ENUM_FLAG_SYNCMFT | MFT_ENUM_FLAG_SORTANDFILTER;
    let mut activates: *mut Option<IMFActivate> = core::ptr::null_mut();
    let mut count = 0u32;
    // SAFETY: MFTEnumEx allocates a CoTaskMem array freed below.
    unsafe {
        MFTEnumEx(
            MFT_CATEGORY_AUDIO_ENCODER,
            flags,
            None,
            Some(&out_info),
            &mut activates,
            &mut count,
        )
        .map_err(mf_err)?;
    }
    let mut chosen: Option<Result<IMFTransform, G2gError>> = None;
    // SAFETY: take ownership of each entry (releasing on drop), activate the first.
    unsafe {
        for i in 0..count as usize {
            let entry = core::ptr::read(activates.add(i));
            if let Some(activate) = entry {
                if chosen.is_none() {
                    chosen = Some(activate.ActivateObject::<IMFTransform>().map_err(mf_err));
                }
            }
        }
        if !activates.is_null() {
            CoTaskMemFree(Some(activates.cast()));
        }
    }
    chosen.unwrap_or(Err(G2gError::Hardware(HardwareError::Other)))
}

/// Read the AudioSpecificConfig from the negotiated output type's
/// `MF_MT_USER_DATA` (HEAACWAVEINFO: 12 info bytes then the ASC).
fn read_audio_specific_config(transform: &IMFTransform) -> Result<Vec<u8>, G2gError> {
    // SAFETY: querying the current output type + its user-data blob.
    unsafe {
        let out_type = transform.GetOutputCurrentType(0).map_err(mf_err)?;
        let size = out_type.GetBlobSize(&MF_MT_USER_DATA).map_err(mf_err)? as usize;
        let mut buf = alloc::vec![0u8; size];
        out_type.GetBlob(&MF_MT_USER_DATA, &mut buf, None).map_err(mf_err)?;
        if buf.len() > HEAAC_INFO_BYTES {
            Ok(buf[HEAAC_INFO_BYTES..].to_vec())
        } else {
            Ok(Vec::new())
        }
    }
}

fn output_provides_samples(transform: &IMFTransform) -> Result<bool, G2gError> {
    // SAFETY: stream-info query on the owning thread.
    let info = unsafe { transform.GetOutputStreamInfo(0) }.map_err(mf_err)?;
    Ok(info.dwFlags & MFT_OUTPUT_STREAM_PROVIDES_SAMPLES.0 as u32 != 0)
}

fn output_buffer_size(transform: &IMFTransform) -> Result<u32, G2gError> {
    // SAFETY: stream-info query on the owning thread.
    let info = unsafe { transform.GetOutputStreamInfo(0) }.map_err(mf_err)?;
    // Floor at a generous AAC access-unit size so an early zero estimate never
    // under-allocates (a 1024-sample AU is well under 1 KiB at these bitrates).
    Ok(info.cbSize.max(8192))
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
        sample.SetSampleTime((pts_ns / 100) as i64).map_err(mf_err)?;
        Ok(sample)
    }
}

fn copy_sample(sample: &IMFSample) -> Result<Vec<u8>, G2gError> {
    // SAFETY: contiguous-buffer access on the owning thread.
    unsafe {
        let buffer = sample.ConvertToContiguousBuffer().map_err(mf_err)?;
        let mut ptr: *mut u8 = core::ptr::null_mut();
        let mut len: u32 = 0;
        buffer.Lock(&mut ptr, None, Some(&mut len)).map_err(mf_err)?;
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
    fn intercept_accepts_s16_rejects_other() {
        let enc = MfAacEncode::new();
        let s16 = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
        };
        assert_eq!(enc.intercept_caps(&s16), Ok(s16));
        let f32 = Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: 2,
            sample_rate: 48_000,
        };
        assert_eq!(enc.intercept_caps(&f32), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn derived_output_maps_pcm_to_aac() {
        let enc = MfAacEncode::new();
        let CapsConstraint::DerivedOutput(f) = enc.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        let out = f(&Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 1,
            sample_rate: 44_100,
        });
        assert_eq!(
            out.alternatives(),
            &[Caps::Audio {
                format: AudioFormat::Aac,
                channels: 1,
                sample_rate: 44_100,
            }]
        );
    }

    #[test]
    fn bitrate_defaults_and_validates() {
        assert_eq!(MfAacEncode::new().bytes_per_sec, DEFAULT_BYTES_PER_SEC);
        assert!(VALID_BYTES_PER_SEC.contains(&DEFAULT_BYTES_PER_SEC));
        // an invalid bitrate is rejected at configure (no MFT created).
        let mut enc = MfAacEncode::new().with_bytes_per_second(9_999);
        let caps = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
        };
        assert!(matches!(
            enc.configure_pipeline(&caps),
            Err(G2gError::CapsMismatch)
        ));
    }
}
