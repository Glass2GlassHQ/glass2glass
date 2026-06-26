//! M307: Android audio via the NDK AAudio API (`AAudioStream`).
//!
//! Two elements, the Android analog of the WASAPI / ALSA / PulseAudio audio
//! elements:
//! - [`AAudioSink`]: an `AsyncElement` sink that renders interleaved PCM
//!   (`PcmS16Le` / `PcmF32Le`) to the default output device.
//! - [`AAudioSrc`]: a `SourceLoop` source that captures interleaved PCM from the
//!   default input device (the microphone).
//!
//! Both wrap the safe `ndk` crate's `audio` module (`AAudioStreamBuilder` ->
//! `AAudioStream`), drive the stream's blocking read / write synchronously, and
//! run on the element's owning task. `AAudioStream` holds a raw pointer and is not
//! `Send`; like the MediaCodec elements, both assert `Send` under the documented
//! single-thread-executor contract so the multi-thread runner accepts them.
//!
//! `aaudio` feature (implies `std`, needs `ndk/audio` = api-level-26).

use core::ffi::c_void;
use core::future::Future;
use core::pin::Pin;

use ndk::audio::{
    AudioDirection, AudioFormat as NdkAudioFormat, AudioPerformanceMode, AudioSharingMode,
    AudioStream, AudioStreamBuilder,
};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, FrameTiming,
    G2gError, HardwareError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
};

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

/// Read / write timeout per AAudio call: 1 second in nanoseconds. Long enough to
/// ride out a scheduling hiccup, short enough that a wedged device surfaces.
const IO_TIMEOUT_NS: i64 = 1_000_000_000;

/// Frames per capture read (10 ms at the stream rate), the audio-buffer cadence
/// the other sources use.
const CAPTURE_MS: u64 = 10;

/// Map any AAudio failure to a structured hardware error.
fn audio_err<E>(_e: E) -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

/// Bytes per sample for a g2g PCM format.
fn bytes_per_sample(format: AudioFormat) -> Option<usize> {
    match format {
        AudioFormat::PcmS16Le => Some(2),
        AudioFormat::PcmF32Le => Some(4),
        // Compressed audio is not a PCM device format.
        AudioFormat::Aac | AudioFormat::Opus => None,
    }
}

/// The AAudio sample format for a g2g PCM format (None for compressed audio).
fn ndk_format(format: AudioFormat) -> Option<NdkAudioFormat> {
    match format {
        AudioFormat::PcmS16Le => Some(NdkAudioFormat::PCM_I16),
        AudioFormat::PcmF32Le => Some(NdkAudioFormat::PCM_Float),
        AudioFormat::Aac | AudioFormat::Opus => None,
    }
}

/// The g2g PCM format for an AAudio sample format (so a capture stream reports the
/// format the device actually opened with).
fn g2g_format(format: NdkAudioFormat) -> Option<AudioFormat> {
    match format {
        NdkAudioFormat::PCM_I16 => Some(AudioFormat::PcmS16Le),
        NdkAudioFormat::PCM_Float => Some(AudioFormat::PcmF32Le),
        _ => None,
    }
}

/// Validate that `caps` is interleaved PCM and extract `(format, channels, rate)`.
fn pcm_params(caps: &Caps) -> Result<(AudioFormat, u8, u32), G2gError> {
    match caps {
        Caps::Audio { format, channels, sample_rate }
            if bytes_per_sample(*format).is_some() && *channels > 0 && *sample_rate > 0 =>
        {
            Ok((*format, *channels, *sample_rate))
        }
        _ => Err(G2gError::CapsMismatch),
    }
}

// ---------------------------------------------------------------------------
// Sink (render / playback)
// ---------------------------------------------------------------------------

/// Renders interleaved PCM to the default AAudio output device.
pub struct AAudioSink {
    stream: Option<AudioStream>,
    format: AudioFormat,
    channels: u8,
    sample_rate: u32,
    configured: bool,
    rendered: u64,
}

// SAFETY: `AudioStream` wraps a raw `AAudioStream` pointer; the element is built
// for a single-thread executor (every stream call lands on the owning task), so
// the pointer is never touched from two threads. Asserted under that contract.
unsafe impl Send for AAudioSink {}

impl core::fmt::Debug for AAudioSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AAudioSink")
            .field("format", &self.format)
            .field("channels", &self.channels)
            .field("sample_rate", &self.sample_rate)
            .field("configured", &self.configured)
            .field("rendered", &self.rendered)
            .finish_non_exhaustive()
    }
}

impl Default for AAudioSink {
    fn default() -> Self {
        Self::new()
    }
}

impl AAudioSink {
    pub fn new() -> Self {
        Self {
            stream: None,
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
            configured: false,
            rendered: 0,
        }
    }

    /// Count of PCM frames written to the device. Useful in tests.
    pub fn rendered(&self) -> u64 {
        self.rendered
    }

    fn open(&mut self) -> Result<(), G2gError> {
        let fmt = ndk_format(self.format).ok_or(G2gError::CapsMismatch)?;
        let stream = AudioStreamBuilder::new()
            .map_err(audio_err)?
            .direction(AudioDirection::Output)
            .format(fmt)
            .channel_count(self.channels as i32)
            .sample_rate(self.sample_rate as i32)
            .performance_mode(AudioPerformanceMode::LowLatency)
            .sharing_mode(AudioSharingMode::Shared)
            .open_stream()
            .map_err(audio_err)?;
        stream.request_start().map_err(audio_err)?;
        self.stream = Some(stream);
        Ok(())
    }

    /// Write a whole PCM buffer, looping over partial writes until every frame is
    /// accepted (or the device errors / times out).
    fn write_all(&mut self, pcm: &[u8]) -> Result<(), G2gError> {
        let bps = bytes_per_sample(self.format).ok_or(G2gError::CapsMismatch)?;
        let frame_bytes = bps * self.channels as usize;
        if frame_bytes == 0 || pcm.len() < frame_bytes {
            return Ok(());
        }
        let stream = self.stream.as_ref().ok_or(G2gError::NotConfigured)?;
        let total_frames = pcm.len() / frame_bytes;
        let mut done = 0usize;
        while done < total_frames {
            let off = done * frame_bytes;
            let remaining = (total_frames - done) as i32;
            // SAFETY: `off` is frame-aligned and `remaining` frames fit within
            // `pcm[off..]`; the stream is an open output stream owned here.
            let wrote = unsafe {
                stream.write(pcm[off..].as_ptr() as *const c_void, remaining, IO_TIMEOUT_NS)
            }
            .map_err(audio_err)?;
            if wrote == 0 {
                // Timed out with no progress: surface rather than spin.
                return Err(G2gError::Hardware(HardwareError::Other));
            }
            done += wrote as usize;
            self.rendered += wrote as u64;
        }
        Ok(())
    }
}

impl AsyncElement for AAudioSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        pcm_params(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::LegacySink(Box::new(|c: &Caps| {
            pcm_params(c)?;
            Ok(c.clone())
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (format, channels, sample_rate) = pcm_params(absolute_caps)?;
        self.format = format;
        self.channels = channels;
        self.sample_rate = sample_rate;
        self.open()?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.write_all(slice.as_slice())?;
                }
                PipelinePacket::Eos => {
                    if let Some(st) = self.stream.as_ref() {
                        let _ = st.request_stop();
                    }
                }
                // PCM caps are fixed at configure; a mid-stream change would need a
                // stream rebuild (not in v1). Control packets are consumed.
                PipelinePacket::CapsChanged(_)
                | PipelinePacket::Flush
                | PipelinePacket::Segment(_) => {}
            }
            Ok(())
        })
    }
}

impl PadTemplates for AAudioSink {
    fn pad_templates() -> Vec<PadTemplate> {
        let pcm = |format| Caps::Audio { format, channels: 2, sample_rate: 48_000 };
        Vec::from([PadTemplate::sink(CapsSet::from_alternatives(Vec::from([
            pcm(AudioFormat::PcmS16Le),
            pcm(AudioFormat::PcmF32Le),
        ])))])
    }
}

// ---------------------------------------------------------------------------
// Source (capture)
// ---------------------------------------------------------------------------

/// Captures interleaved PCM from the default AAudio input device (the mic).
pub struct AAudioSrc {
    /// Requested rate / channels; the opened stream's actuals (which the device
    /// may pick differently) are reported as the caps.
    req_sample_rate: u32,
    req_channels: u8,
    target_buffers: u64,
    stream: Option<AudioStream>,
    caps: Option<Caps>,
    configured: bool,
}

// SAFETY: same single-thread-executor contract as `AAudioSink`.
unsafe impl Send for AAudioSrc {}

impl core::fmt::Debug for AAudioSrc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AAudioSrc")
            .field("req_sample_rate", &self.req_sample_rate)
            .field("req_channels", &self.req_channels)
            .field("target_buffers", &self.target_buffers)
            .field("caps", &self.caps)
            .field("configured", &self.configured)
            .finish_non_exhaustive()
    }
}

impl AAudioSrc {
    /// A capture source requesting `sample_rate` / `channels`, emitting
    /// `target_buffers` buffers then EOS (`u64::MAX` = capture until stopped).
    pub fn new(sample_rate: u32, channels: u8, target_buffers: u64) -> Self {
        Self {
            req_sample_rate: sample_rate.max(1),
            req_channels: channels.max(1),
            target_buffers,
            stream: None,
            caps: None,
            configured: false,
        }
    }

    /// Open the capture stream (if not already) and record the device's actual
    /// format / rate / channels as the produced caps.
    fn ensure_open(&mut self) -> Result<Caps, G2gError> {
        if let Some(caps) = &self.caps {
            return Ok(caps.clone());
        }
        let stream = AudioStreamBuilder::new()
            .map_err(audio_err)?
            .direction(AudioDirection::Input)
            .format(NdkAudioFormat::PCM_I16)
            .channel_count(self.req_channels as i32)
            .sample_rate(self.req_sample_rate as i32)
            .performance_mode(AudioPerformanceMode::LowLatency)
            .sharing_mode(AudioSharingMode::Shared)
            .open_stream()
            .map_err(audio_err)?;
        // The device may have opened with different actuals; report those.
        let format = g2g_format(stream.format()).ok_or(G2gError::CapsMismatch)?;
        let channels = stream.channel_count().max(1) as u8;
        let sample_rate = stream.sample_rate().max(1) as u32;
        let caps = Caps::Audio { format, channels, sample_rate };
        self.stream = Some(stream);
        self.caps = Some(caps.clone());
        Ok(caps)
    }
}

impl SourceLoop for AAudioSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(self.ensure_open())
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        let caps = self.ensure_open();
        core::future::ready(caps.map(|c| CapsConstraint::Produces(CapsSet::one(c))))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let stream = self.stream.as_ref().ok_or(G2gError::NotConfigured)?;
        stream.request_start().map_err(audio_err)?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let Caps::Audio { format, channels, sample_rate } =
                self.caps.clone().ok_or(G2gError::NotConfigured)?
            else {
                return Err(G2gError::CapsMismatch);
            };
            let bps = bytes_per_sample(format).ok_or(G2gError::CapsMismatch)?;
            let frame_bytes = bps * channels as usize;
            let frames_per_buf = (sample_rate as u64 * CAPTURE_MS / 1000).max(1);
            let ns_per_frame = 1_000_000_000u64 / sample_rate as u64;

            let mut total_frames = 0u64;
            let mut seq = 0u64;
            while seq < self.target_buffers {
                let mut buf = vec![0u8; frames_per_buf as usize * frame_bytes];
                let got = {
                    let stream = self.stream.as_ref().ok_or(G2gError::NotConfigured)?;
                    // SAFETY: `buf` holds `frames_per_buf` frames; the stream is an
                    // open input stream owned here.
                    unsafe {
                        stream.read(
                            buf.as_mut_ptr() as *mut c_void,
                            frames_per_buf as i32,
                            IO_TIMEOUT_NS,
                        )
                    }
                    .map_err(audio_err)?
                };
                if got == 0 {
                    continue;
                }
                buf.truncate(got as usize * frame_bytes);

                let pts_ns = total_frames * ns_per_frame;
                let duration_ns = got as u64 * ns_per_frame;
                #[cfg(feature = "std")]
                let arrival_ns = g2g_core::metrics::monotonic_ns();
                #[cfg(not(feature = "std"))]
                let arrival_ns: u64 = 0;
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(buf.into_boxed_slice())),
                    timing: FrameTiming {
                        pts_ns,
                        dts_ns: pts_ns,
                        duration_ns,
                        capture_ns: pts_ns,
                        arrival_ns,
                        keyframe: false,
                    },
                    sequence: seq,
                    meta: Default::default(),
                };
                total_frames += got as u64;
                seq += 1;
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }

            if let Some(st) = self.stream.as_ref() {
                let _ = st.request_stop();
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(seq)
        })
    }
}

impl PadTemplates for AAudioSrc {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::source(CapsSet::one(Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
        }))])
    }
}
