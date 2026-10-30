//! Android audio via the NDK AAudio API (`AAudioStream`).
//!
//! Two elements, the Android analog of the WASAPI / ALSA / PulseAudio audio
//! elements:
//! - [`AAudioSink`]: an `AsyncElement` sink that renders interleaved PCM
//!   (`PcmS16Le` / `PcmF32Le`) to the default output device.
//! - [`AAudioSrc`]: a `SourceLoop` source that captures interleaved PCM from the
//!   default input device (the microphone).
//!
//! Both wrap the safe `ndk` crate's `audio` module (`AAudioStreamBuilder` ->
//! `AAudioStream`) and drive the stream's blocking read / write on a dedicated
//! thread, so the executor never blocks on the device. `AAudioStream` holds a
//! raw pointer and is not `Send`; the source still holds it between negotiation
//! and `run`, so it asserts `Send` under the single-thread-executor contract,
//! like the MediaCodec elements.
//!
//! `aaudio` feature (implies `std`, needs `ndk/audio` = api-level-26).

use core::ffi::c_void;
use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use std::sync::mpsc::Receiver;
use std::sync::Arc;

use ndk::audio::{
    AudioDirection, AudioError, AudioFormat as NdkAudioFormat, AudioPerformanceMode,
    AudioSharingMode, AudioStream, AudioStreamBuilder, AudioStreamState,
};

use g2g_core::frame::Frame;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ClockSync, ConfigureOutcome,
    G2gError, HardwareError, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec,
};

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::audiocapture::{capture_queue, run_capture, CaptureLoopSettings, CaptureSender};
use crate::audioworker::WorkerLink;
use g2g_core::pcm_caps_any_rate_and_channels;

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

/// Interpret an AAudio `read` / `write` result into a frame count. ndk 0.9's
/// `read` / `write` wrappers pipe the (positive) frame count through
/// `AudioError::from_result`, which treats only `AAUDIO_OK` (0) as success, so a
/// completed transfer surfaces as `Err(AudioError::__Unknown(frames))`. Recover
/// that count; a real (negative) AAudio error stays an error.
fn frames_transferred(r: Result<u32, AudioError>) -> Result<u32, G2gError> {
    match r {
        Ok(n) => Ok(n),
        Err(AudioError::__Unknown(n)) if n >= 0 => Ok(n as u32),
        Err(e) => Err(audio_err(e)),
    }
}

/// The AAudio sample format for a g2g PCM format (None for compressed audio).
fn ndk_format(format: AudioFormat) -> Option<NdkAudioFormat> {
    match format {
        AudioFormat::PcmS16Le => Some(NdkAudioFormat::PCM_I16),
        AudioFormat::PcmF32Le => Some(NdkAudioFormat::PCM_Float),
        AudioFormat::Aac | AudioFormat::Opus => None,
        _ => None,
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

// an open stream handed to the one thread that reads or writes it
struct DeviceThreadStream(AudioStream);

// SAFETY: the element moves the stream out to that thread, which is the only
// caller into it from then on and closes it there.
unsafe impl Send for DeviceThreadStream {}

/// Validate that `caps` is interleaved PCM and extract `(format, channels, rate)`.
fn pcm_params(caps: &Caps) -> Result<(AudioFormat, u8, u32), G2gError> {
    match caps {
        Caps::Audio {
            format,
            channels,
            sample_rate,
            ..
        } if ndk_format(*format).is_some() && *channels > 0 && *sample_rate > 0 => {
            Ok((*format, *channels, *sample_rate))
        }
        _ => Err(G2gError::CapsMismatch),
    }
}

// ---------------------------------------------------------------------------
// Sink (render / playback)
// ---------------------------------------------------------------------------

/// Renders interleaved PCM to the default AAudio output device.
///
/// # Example
///
/// ```ignore
/// use g2g_plugins::aaudio::AAudioSink;
///
/// let sink = AAudioSink::new();
/// ```
pub struct AAudioSink {
    /// Bounded, non-blocking link to the thread that owns the stream and makes
    /// the blocking writes (see [`crate::audioworker`]).
    link: Option<WorkerLink<SinkCommand>>,
    format: AudioFormat,
    channels: u8,
    sample_rate: u32,
    rendered: Arc<AtomicU64>,
}

impl core::fmt::Debug for AAudioSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AAudioSink")
            .field("format", &self.format)
            .field("channels", &self.channels)
            .field("sample_rate", &self.sample_rate)
            .field("rendered", &self.rendered())
            .finish_non_exhaustive()
    }
}

impl Default for AAudioSink {
    fn default() -> Self {
        Self::new()
    }
}

enum SinkCommand {
    Samples(Frame),
    Shutdown,
}

impl AAudioSink {
    pub fn new() -> Self {
        Self {
            link: None,
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
            rendered: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Count of PCM frames written to the device. Useful in tests.
    pub fn rendered(&self) -> u64 {
        self.rendered.load(Ordering::Relaxed)
    }

    fn shutdown(&mut self) {
        if let Some(link) = self.link.as_mut() {
            link.abort();
        }
        self.link = None;
    }

    fn open(&self) -> Result<AudioStream, G2gError> {
        let fmt = ndk_format(self.format).ok_or(G2gError::CapsMismatch)?;
        let stream = AudioStreamBuilder::new()
            .map_err(audio_err)?
            .direction(AudioDirection::Output)
            .format(fmt)
            .channel_count(self.channels as i32)
            .sample_rate(self.sample_rate as i32)
            // PerformanceMode::None (not LowLatency): low-latency opens an MMAP
            // stream that is callback-driven, on which a blocking `write` with a
            // timeout fails. This element drives the stream synchronously with
            // blocking writes, so it needs the legacy buffered path. (A
            // callback-based low-latency mode is a follow-up.)
            .performance_mode(AudioPerformanceMode::None)
            .sharing_mode(AudioSharingMode::Shared)
            .open_stream()
            .map_err(audio_err)?;
        stream.request_start().map_err(audio_err)?;
        Ok(stream)
    }
}

impl Drop for AAudioSink {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Write a whole PCM buffer, looping over partial writes until every frame is
/// accepted (or the device errors / times out).
fn write_all(
    stream: &AudioStream,
    pcm: &[u8],
    frame_bytes: usize,
    rendered: &AtomicU64,
) -> Result<(), G2gError> {
    let total_frames = pcm.len() / frame_bytes;
    let mut done = 0usize;
    while done < total_frames {
        let off = done * frame_bytes;
        let remaining = (total_frames - done) as i32;
        // SAFETY: `off` is frame-aligned and `remaining` frames fit within
        // `pcm[off..]`; the stream is an open output stream only this thread uses.
        let wrote = frames_transferred(unsafe {
            stream.write(
                pcm[off..].as_ptr() as *const c_void,
                remaining,
                IO_TIMEOUT_NS,
            )
        })?;
        if wrote == 0 {
            // Timed out with no progress: surface rather than spin.
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        done += wrote as usize;
        rendered.fetch_add(u64::from(wrote), Ordering::Relaxed);
    }
    Ok(())
}

fn render_worker(
    stream: DeviceThreadStream,
    commands: Receiver<SinkCommand>,
    frame_bytes: usize,
    rendered: Arc<AtomicU64>,
) {
    let stream = stream.0;
    loop {
        let frame = match commands.recv() {
            Ok(SinkCommand::Samples(frame)) => frame,
            Ok(SinkCommand::Shutdown) => {
                // stops once what is buffered has played
                let _ = stream.request_stop();
                let _ = stream.wait_for_state_change(AudioStreamState::Stopping, IO_TIMEOUT_NS);
                return;
            }
            // the element aborted: close without playing out
            Err(_) => return,
        };
        let pcm = frame.domain.as_system_slice().unwrap_or_default();
        if write_all(&stream, pcm, frame_bytes, &rendered).is_err() {
            return;
        }
    }
}

impl AsyncElement for AAudioSink {
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
        pcm_params(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(accepted_pcm_caps())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (format, channels, sample_rate) = pcm_params(absolute_caps)?;
        let unchanged =
            (format, channels, sample_rate) == (self.format, self.channels, self.sample_rate);
        if unchanged && self.link.as_ref().is_some_and(|l| l.is_running()) {
            return Ok(ConfigureOutcome::Accepted);
        }
        self.shutdown();
        self.format = format;
        self.channels = channels;
        self.sample_rate = sample_rate;
        let stream = DeviceThreadStream(self.open()?);
        let frame_bytes =
            format.bytes_per_sample().ok_or(G2gError::CapsMismatch)? * usize::from(channels);
        let rendered = Arc::clone(&self.rendered);
        self.link = Some(WorkerLink::spawn("g2g-aaudiosink", move |commands| {
            render_worker(stream, commands, frame_bytes, rendered)
        })?);
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    let link = self.link.as_ref().ok_or(G2gError::NotConfigured)?;
                    // a full queue is the device's back-pressure and yields
                    link.send(SinkCommand::Samples(frame), audio_err(()))
                        .await?;
                }
                PipelinePacket::Eos => {
                    // waits for playout while yielding, so other arms keep running
                    let link = self.link.as_mut().ok_or(G2gError::NotConfigured)?;
                    link.finish(SinkCommand::Shutdown, audio_err(())).await?;
                    self.link = None;
                }
                // PCM caps are fixed at configure; a mid-stream change would need a
                // stream rebuild (not in v1). Control packets are consumed.
                PipelinePacket::CapsChanged(_)
                | PipelinePacket::Flush
                | PipelinePacket::Segment(_) => {}
                // future PipelinePacket variants: no-op (terminal sink).
                _ => {}
            }
            Ok(())
        })
    }
}

fn accepted_pcm_caps() -> CapsSet {
    pcm_caps_any_rate_and_channels(|format| ndk_format(format).is_some())
}

impl PadTemplates for AAudioSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(accepted_pcm_caps())])
    }
}

// ---------------------------------------------------------------------------
// Source (capture)
// ---------------------------------------------------------------------------

/// Captures interleaved PCM from the default AAudio input device (the mic).
///
/// # Example
///
/// ```ignore
/// use g2g_plugins::aaudio::AAudioSrc;
///
/// // 48 kHz stereo, capture until stopped.
/// let src = AAudioSrc::new(48_000, 2, u64::MAX);
/// ```
pub struct AAudioSrc {
    /// Requested rate / channels; the opened stream's actuals (which the device
    /// may pick differently) are reported as the caps.
    req_sample_rate: u32,
    req_channels: u8,
    target_buffers: u64,
    stream: Option<AudioStream>,
    caps: Option<Caps>,
    configured: bool,
    clock_sync: Option<ClockSync>,
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
    /// All three are also settable as the `samplerate` / `channels` /
    /// `num-buffers` properties.
    pub fn new(sample_rate: u32, channels: u8, target_buffers: u64) -> Self {
        Self {
            req_sample_rate: sample_rate.max(1),
            req_channels: channels.max(1),
            target_buffers,
            stream: None,
            caps: None,
            configured: false,
            clock_sync: None,
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
            // PerformanceMode::None for the same reason as the sink: blocking
            // `read` needs the legacy buffered path, not the MMAP low-latency one.
            .performance_mode(AudioPerformanceMode::None)
            .sharing_mode(AudioSharingMode::Shared)
            .open_stream()
            .map_err(audio_err)?;
        // The device may have opened with different actuals; report those.
        let format = g2g_format(stream.format()).ok_or(G2gError::CapsMismatch)?;
        let channels = stream.channel_count().max(1) as u8;
        let sample_rate = stream.sample_rate().max(1) as u32;
        let caps = Caps::Audio {
            format,
            channels,
            sample_rate,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        self.stream = Some(stream);
        self.caps = Some(caps.clone());
        Ok(caps)
    }
}

impl SourceLoop for AAudioSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
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

    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.clock_sync = Some(sync);
    }

    fn properties(&self) -> &'static [PropertySpec] {
        AAUDIOSRC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            // Zero is not a stream AAudio can open, so a degenerate PCM shape is
            // refused instead of carried into the builder.
            "samplerate" => {
                let rate = value.as_uint().ok_or(PropError::Type)?;
                if rate == 0 || rate > u32::MAX as u64 {
                    return Err(PropError::Value);
                }
                self.req_sample_rate = rate as u32;
            }
            "channels" => {
                let channels = value.as_uint().ok_or(PropError::Type)?;
                if channels == 0 || channels > u8::MAX as u64 {
                    return Err(PropError::Value);
                }
                self.req_channels = channels as u8;
            }
            "num-buffers" => crate::numbuffers::set_num_buffers(&mut self.target_buffers, &value)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "samplerate" => Some(PropValue::Uint(u64::from(self.req_sample_rate))),
            "channels" => Some(PropValue::Uint(u64::from(self.req_channels))),
            "num-buffers" => Some(crate::numbuffers::get_num_buffers(self.target_buffers)),
            _ => None,
        }
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let Caps::Audio {
                format,
                channels,
                sample_rate,
                ..
            } = self.caps.clone().ok_or(G2gError::NotConfigured)?
            else {
                return Err(G2gError::CapsMismatch);
            };
            let bps = format.bytes_per_sample().ok_or(G2gError::CapsMismatch)?;
            let frame_bytes = bps * channels as usize;
            let frames_per_read = (sample_rate as u64 * CAPTURE_MS / 1000).max(1) as usize;
            let stream = DeviceThreadStream(self.stream.take().ok_or(G2gError::NotConfigured)?);
            let (sender, receiver) = capture_queue(frame_bytes)?;
            let stop = Arc::new(AtomicBool::new(false));
            let worker_stop = Arc::clone(&stop);
            let worker = std::thread::Builder::new()
                .name(String::from("g2g-aaudiosrc"))
                .spawn(move || {
                    read_stream(stream, frames_per_read, frame_bytes, sender, worker_stop)
                })
                .map_err(audio_err)?;
            let stop_device = move || {
                stop.store(true, Ordering::Relaxed);
                let _ = worker.join();
            };
            let settings = CaptureLoopSettings {
                log_category: "aaudiosrc",
                sample_rate,
                clock_sync: self.clock_sync.clone(),
                buffer_limit: self.target_buffers,
            };
            run_capture(receiver, settings, out, stop_device).await
        })
    }
}

fn read_stream(
    stream: DeviceThreadStream,
    frames_per_read: usize,
    frame_bytes: usize,
    sender: CaptureSender,
    stop: Arc<AtomicBool>,
) {
    let stream = stream.0;
    while !stop.load(Ordering::Relaxed) {
        let mut buf = vec![0u8; frames_per_read * frame_bytes];
        // SAFETY: `buf` holds `frames_per_read` frames; the stream is an open
        // input stream only this thread uses.
        let read = frames_transferred(unsafe {
            stream.read(
                buf.as_mut_ptr() as *mut c_void,
                frames_per_read as i32,
                IO_TIMEOUT_NS,
            )
        });
        let frames = match read {
            Ok(frames) => frames as usize,
            Err(error) => {
                sender.fail(error);
                break;
            }
        };
        if frames == 0 {
            continue;
        }
        buf.truncate(frames * frame_bytes);
        if !sender.push(buf) {
            break;
        }
    }
    let _ = stream.request_stop();
}

impl PadTemplates for AAudioSrc {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::source(pcm_caps_any_rate_and_channels(
            |format| format == AudioFormat::PcmS16Le,
        ))])
    }
}

/// `AAudioSrc`'s settable properties. AAudio capture has no gst element to match
/// names with, so rate / channels take the g2g `alsasrc` / `audiotestsrc` names
/// and `num-buffers` matches gst `basesrc`. Rate and channels are requests: the
/// opened stream's actuals are what the produced caps report.
static AAUDIOSRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new("samplerate", PropKind::Uint, "requested samples per second")
        .with_default("48000"),
    PropertySpec::new("channels", PropKind::Uint, "requested channel count").with_default("2"),
    PropertySpec::new(
        "num-buffers",
        PropKind::Int,
        "buffers to capture then EOS (-1 = forever)",
    )
    .with_default("-1"),
];
