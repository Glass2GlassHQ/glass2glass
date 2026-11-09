//! Linux PulseAudio render sink. Plays interleaved PCM (`PcmU8` / `PcmS16Le` /
//! `PcmS24Le` / `PcmS32Le` / `PcmF32Le`) on the default PulseAudio (or
//! PipeWire-pulse) server, the higher-level sibling of
//! [`AlsaSink`](crate::alsasink::AlsaSink) and the Linux analog of the
//! Windows-only [`WasapiSink`](crate::wasapisink::WasapiSink).
//!
//! ## Channel order
//!
//! The stream carries an explicit channel map built from our interleave order
//! (see `pulsepcm`), so the server routes a 5.1 / 7.1 stream by speaker instead
//! of falling back to its own positional guess.
//!
//! ## Threading and pacing
//!
//! The libpulse "simple" API (`pa_simple_write`) is blocking, so all of it runs
//! on a dedicated worker spun up at `configure_pipeline`, exactly as in
//! `AlsaSink` / `WasapiSink`. The blocking write *is* the pacing: it returns
//! once the server-side buffer has room, which is why the stream asks for a
//! bounded buffer instead of pulse's multi-second default (see
//! [`buffer_attr`]). On `Eos` the worker drains the buffer (`pa_simple_drain`)
//! so the tail is not cut off.
//!
//! ## Clock
//!
//! Like `AlsaSink`, this sink offers a playout-disciplined [`DriftClock`] at
//! [`ClockPriority::AudioProvider`] (the `provide-clock` property, default on),
//! so audio is the A/V sync master. The worker feeds it one observation per
//! write, taking the playout position from `pa_simple_get_latency` (pulse's
//! analog of `snd_pcm_delay`, and already inclusive of the device latency).

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicU64, Ordering};

use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;

use std::time::Duration;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use libpulse_binding::def::BufferAttr;
use libpulse_binding::sample::Spec;
use libpulse_binding::stream::Direction;
use libpulse_binding::time::MicroSeconds;
use libpulse_simple_binding::Simple;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ClockCandidate, ClockPriority, ClockSync,
    ConfigureOutcome, DeviceClockOffset, DriftClock, ElementMetadata, G2gError, HardwareError,
    MonotonicClock, OutputSink, PadTemplate, PadTemplates, PipelineClock, PipelinePacket,
    PropError, PropKind, PropValue, PropertySpec,
};

use crate::audiooutput::{DeviceCmd, OutputDevice, PlayEnd};
use crate::pulsepcm::{pulse_map, pulse_spec, FORMATS};
use g2g_core::pcm_caps_any_rate_and_channels;

/// Target server-side buffer, in playout time. Bounds it the way `AlsaSink`
/// bounds its ALSA ring: pulse's default target is a couple of seconds, so a
/// source that runs faster than real time queues the whole stream without
/// `pa_simple_write` ever blocking, which both defeats the pacing this sink
/// relies on and pins the reported latency at the full backlog, leaving the
/// playout clock with no frames played to observe.
const TARGET_BUFFER: MicroSeconds = MicroSeconds(200_000);
/// Smallest chunk the server asks for, so it refills in 20 ms steps rather than
/// one 200 ms gulp.
const MIN_REQUEST: MicroSeconds = MicroSeconds(20_000);

/// Buffering attributes for the playback stream. `u32::MAX` is pulse's "server
/// default" sentinel; `fragsize` is record-only, so it stays default.
fn buffer_attr(spec: &Spec) -> BufferAttr {
    let bytes = |t| u32::try_from(spec.usec_to_bytes(t)).unwrap_or(u32::MAX);
    BufferAttr {
        maxlength: u32::MAX,
        tlength: bytes(TARGET_BUFFER),
        prebuf: u32::MAX,
        minreq: bytes(MIN_REQUEST),
        fragsize: u32::MAX,
    }
}

const ELEMENT_NAME: &str = "PulseSink";

/// # Example
///
/// ```no_run
/// use g2g_plugins::pulsesink::PulseSink;
///
/// let sink = PulseSink::with_app_name("my-player");
/// ```
pub struct PulseSink {
    app_name: String,
    output: OutputDevice,
    /// Bounded, non-blocking link to the device thread. The executor is
    /// cooperative, so neither the hand-off nor the end-of-stream drain may
    /// block it (see [`crate::audioworker`]).
    link: Option<crate::audioworker::WorkerLink<DeviceCmd>>,
    caps: Option<Caps>,
    bytes_written: Arc<AtomicU64>,
    /// Playout-disciplined master clock (the `AlsaSink` mechanism
    /// over pulse's latency query). The worker feeds it
    /// `(monotonic_now, played_ns)` observations so its `now_ns()` tracks the
    /// real playout rate; a video sink slaves to it when it is elected.
    clock: Arc<DriftClock>,
    /// Whether to offer [`clock`](Self::clock) to the pipeline's clock election
    /// (the `provide-clock` property, default on, GStreamer's `basesink`).
    provide_clock: bool,
    skew: crate::audioskew::SkewSlaver,
}

impl core::fmt::Debug for PulseSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PulseSink")
            .field("app_name", &self.app_name)
            .field("output", &self.output)
            .field("caps", &self.caps)
            .field("bytes_written", &self.bytes_written.load(Ordering::Relaxed))
            .finish()
    }
}

impl Default for PulseSink {
    fn default() -> Self {
        Self::new()
    }
}

impl PulseSink {
    /// Render under the default application name `glass2glass`.
    pub fn new() -> Self {
        Self::with_app_name("glass2glass")
    }

    /// Render under a custom application name (shown in the PulseAudio mixer).
    pub fn with_app_name(name: impl Into<String>) -> Self {
        Self {
            app_name: name.into(),
            output: OutputDevice::new("device", ""),
            link: None,
            caps: None,
            bytes_written: Arc::new(AtomicU64::new(0)),
            clock: Arc::new(DriftClock::new(Arc::new(MonotonicClock))),
            provide_clock: true,
            skew: crate::audioskew::SkewSlaver::default(),
        }
    }

    /// Total PCM bytes written to the server. Useful in tests.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written.load(Ordering::Relaxed)
    }

    /// The playout-disciplined clock this sink offers to election. Exposed for
    /// tests / introspection; its `now_ns()` tracks real playout once the worker
    /// has observed the server.
    pub fn clock(&self) -> Arc<DriftClock> {
        Arc::clone(&self.clock)
    }

    /// Tear down without waiting for playout (reconfigure / drop). The
    /// end-of-stream drain that *does* wait is `WorkerLink::finish`, awaited
    /// from `process(Eos)` so it never blocks the executor.
    fn shutdown(&mut self) {
        if let Some(link) = self.link.as_mut() {
            link.abort();
        }
        self.link = None;
    }
}

impl Drop for PulseSink {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl AsyncElement for PulseSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Feeds the device a host pointer, so it reads a system buffer or maps a
    /// dma-buf in place. A producer in any other domain downloads through the
    /// allocation cascade.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        crate::dmabufmap::audio_input_domains()
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        pulse_spec(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    // channel and rate bounds are left to `pulse_spec` in configure
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(accepted_pcm_caps())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let spec = pulse_spec(absolute_caps)?;

        if self.link.as_ref().is_some_and(|l| l.is_running()) {
            if self.caps.as_ref() == Some(absolute_caps) {
                return Ok(ConfigureOutcome::Accepted);
            }
            self.shutdown();
        }

        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<(), G2gError>>(1);
        let written = Arc::clone(&self.bytes_written);
        let app_name = self.app_name.clone();
        let device = String::from(self.output.name());
        // disciplined even when not offered, skew slaving reads it
        let clock = Arc::clone(&self.clock);
        self.skew.configure(absolute_caps)?;
        let fallback = self.output.fallback(
            spec.bytes_per_second() as u64,
            Arc::clone(&self.bytes_written),
            1,
            |frame| {
                crate::dmabufmap::frame_bytes(frame, ELEMENT_NAME)
                    .ok()
                    .map(|bytes| bytes.as_slice().len())
            },
        );

        let link = crate::audioworker::WorkerLink::spawn("g2g-pulsesink", move |rx| {
            let mut clock_offset = DeviceClockOffset::default();
            crate::audiooutput::run_device_worker(
                device,
                rx,
                fallback,
                ready_tx,
                |device| open_stream(&app_name, device, &spec),
                |simple, rx| play(simple, &spec, rx, &written, &clock, &mut clock_offset),
            );
        })?;

        // The worker reports whether the server connection opened; a host with
        // no PulseAudio server fails loud here rather than dropping audio.
        match ready_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(())) => {}
            // `link`'s Drop reaps the worker, so a failed open needs no join.
            Ok(Err(error)) => return Err(error),
            Err(_) => return Err(G2gError::Hardware(HardwareError::PulseAudio(-1))),
        }

        self.output.opened();
        self.link = Some(link);
        self.caps = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    /// Offer the playout-disciplined [`clock`](Self::clock) as an
    /// [`AudioProvider`](ClockPriority::AudioProvider) so audio becomes the
    /// pipeline master (video slaves to it), unless `provide-clock` is off.
    fn provide_clock(&self) -> Option<ClockCandidate> {
        if !self.provide_clock {
            return None;
        }
        let clock: Arc<dyn PipelineClock + Send + Sync> = self.clock.clone();
        Some(ClockCandidate::new(ClockPriority::AudioProvider, clock))
    }

    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.skew.set_master(sync.clock);
    }

    fn set_bus(&mut self, bus: g2g_core::bus::BusHandle) {
        self.output.set_bus(bus);
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "PulseAudio audio sink",
            "Sink/Audio",
            "Plays interleaved PCM via PulseAudio",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new(
                "client-name",
                PropKind::Str,
                "Application name shown in the PulseAudio mixer",
            )
            .with_default("glass2glass"),
            PropertySpec::new(
                "device",
                PropKind::Str,
                "PulseAudio sink to play to (empty = the server default)",
            )
            .with_default(""),
            PropertySpec::new(
                "provide-clock",
                PropKind::Bool,
                "Provide a playout-disciplined clock so audio is the A/V sync master",
            )
            .with_default("true"),
            crate::audioskew::SLAVE_METHOD_PROPERTY,
            crate::audioskew::DRIFT_TOLERANCE_PROPERTY,
            crate::audiooutput::CONTINUE_ON_ERROR_PROPERTY,
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if let Some(result) = self.skew.set_property(name, &value) {
            return result;
        }
        if let Some(result) = self.output.set_property(name, &value, self.link.is_some()) {
            return result;
        }
        match name {
            "client-name" => {
                self.app_name = value.as_str().ok_or(PropError::Type)?.into();
                Ok(())
            }
            "provide-clock" => {
                self.provide_clock = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        if let Some(value) = self.skew.get_property(name) {
            return Some(value);
        }
        if let Some(value) = self.output.get_property(name) {
            return Some(value);
        }
        match name {
            "client-name" => Some(PropValue::Str(self.app_name.clone())),
            "provide-clock" => Some(PropValue::Bool(self.provide_clock)),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    crate::dmabufmap::require_readable_domain(&frame, ELEMENT_NAME)?;
                    self.skew.measure(&self.clock);
                    let frame = self.skew.correct_frame(frame, ELEMENT_NAME)?;
                    let link = self.link.as_ref().ok_or(G2gError::NotConfigured)?;
                    if let Some(device) = self.output.take_switch() {
                        link.send(
                            DeviceCmd::SwitchDevice(device),
                            G2gError::Hardware(HardwareError::PulseAudio(-1)),
                        )
                        .await?;
                    }
                    // Awaited, not blocking: a full queue is the device's
                    // back-pressure and yields to the executor.
                    link.send(
                        DeviceCmd::Samples(frame),
                        G2gError::Hardware(HardwareError::PulseAudio(-1)),
                    )
                    .await?;
                    Ok(())
                }
                // A mid-stream format change can't be honoured on an open
                // stream; only a caps identical to the configured one passes.
                PipelinePacket::CapsChanged(c) => {
                    pulse_spec(&c)?;
                    Ok(())
                }
                PipelinePacket::Flush | PipelinePacket::Segment(_) => Ok(()),
                PipelinePacket::Eos => {
                    // Wait for the queued audio to actually play out, yielding
                    // throughout: a blocking join here stalled every other arm
                    // in the pipeline for the length of the sound.
                    if let Some(link) = self.link.as_mut() {
                        link.finish(
                            DeviceCmd::Shutdown,
                            G2gError::Hardware(HardwareError::PulseAudio(-1)),
                        )
                        .await?;
                    }
                    self.link = None;
                    Ok(())
                }
                // future PipelinePacket variants: no-op (terminal sink).
                _ => Ok(()),
            }
        })
    }
}

fn accepted_pcm_caps() -> CapsSet {
    pcm_caps_any_rate_and_channels(|format| FORMATS.iter().any(|(opened, _)| *opened == format))
}

impl PadTemplates for PulseSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(accepted_pcm_caps())])
    }
}

// =================================================================
// Worker thread: blocking libpulse simple write
// =================================================================

fn open_stream(app_name: &str, device: &str, spec: &Spec) -> Result<Simple, G2gError> {
    let map = pulse_map(spec.channels);
    let attr = buffer_attr(spec);
    Simple::new(
        None,     // default server
        app_name, // application name
        Direction::Playback,
        (!device.is_empty()).then_some(device),
        "playback", // stream description
        spec,
        map.as_ref(),
        Some(&attr),
    )
    .map_err(|e| G2gError::Hardware(HardwareError::PulseAudio(e.0)))
}

fn play(
    simple: Simple,
    spec: &Spec,
    rx: &Receiver<DeviceCmd>,
    written: &AtomicU64,
    clock: &DriftClock,
    clock_offset: &mut DeviceClockOffset,
) -> PlayEnd {
    let end = loop {
        match rx.recv() {
            Ok(DeviceCmd::Samples(frame)) => {
                let payload = match crate::dmabufmap::frame_bytes(&frame, ELEMENT_NAME) {
                    Ok(payload) => payload,
                    Err(error) => break PlayEnd::Failed(error),
                };
                let bytes = payload.as_slice();
                if let Err(e) = simple.write(bytes) {
                    return PlayEnd::Failed(G2gError::Hardware(HardwareError::PulseAudio(e.0)));
                }
                written.fetch_add(bytes.len() as u64, Ordering::Relaxed);
                // After the blocking write returns (at ~the device rate), feed
                // the drift clock one playout observation.
                discipline_clock(&simple, spec, written, clock, clock_offset);
            }
            Ok(DeviceCmd::SwitchDevice(device)) => break PlayEnd::SwitchDevice(device),
            Ok(DeviceCmd::Shutdown) | Err(_) => break PlayEnd::Shutdown,
        }
    };
    // Play out whatever is still buffered, then stop.
    let _ = simple.drain();
    end
}

/// Feed the drift clock one `(monotonic_now, played_ns)` observation. Bytes
/// actually played = bytes handed to `pa_simple_write` minus the latency the
/// server still reports (the queue plus the device's own, pulse's analog of
/// `snd_pcm_delay`), both converted to time through the stream spec. That is the
/// true playout position, which drifts from wall time at the hardware's real
/// rate. The local time is sampled next to the latency query so the pair lines up.
fn discipline_clock(
    simple: &Simple,
    spec: &Spec,
    written: &AtomicU64,
    clock: &DriftClock,
    clock_offset: &mut DeviceClockOffset,
) {
    let local_ns = clock.reference_now();
    // A failed latency query (stream not running yet) yields no usable playout
    // position; skip rather than feed a bogus sample.
    let Ok(latency) = simple.get_latency() else {
        return;
    };
    let written_us = spec.bytes_to_usec(written.load(Ordering::Relaxed)).0;
    let played_us = written_us.saturating_sub(latency.0);
    if played_us == 0 {
        return;
    }
    let master_ns = clock_offset.master_ns(clock, local_ns, played_us.saturating_mul(1_000));
    clock.observe(local_ns, master_ns);
}

#[cfg(test)]
mod tests {
    use super::*;

    use g2g_core::AudioFormat;

    #[test]
    fn intercept_accepts_pcm_rejects_compressed() {
        let sink = PulseSink::new();
        let pcm = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        assert_eq!(sink.intercept_caps(&pcm), Ok(pcm));
        let opus = Caps::Audio {
            format: AudioFormat::Opus,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        assert_eq!(sink.intercept_caps(&opus), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn buffer_attr_bounds_the_server_buffer_to_the_target_time() {
        use libpulse_binding::sample::Format as PaFormat;
        let spec = Spec {
            format: PaFormat::S16le,
            channels: 2,
            rate: 48_000,
        };
        let attr = buffer_attr(&spec);
        // 200 ms of 48 kHz stereo s16 = 9600 frames * 4 bytes, and 20 ms of it
        // per request: the default (~2 s) would never block the writer.
        assert_eq!(attr.tlength, 200 * 48 * 4);
        assert_eq!(attr.minreq, 20 * 48 * 4);
        // server defaults for the rest (fragsize is record-only).
        assert_eq!(
            (attr.maxlength, attr.prebuf, attr.fragsize),
            (u32::MAX, u32::MAX, u32::MAX)
        );
    }

    #[test]
    fn provides_an_audio_master_clock_by_default() {
        let sink = PulseSink::new();
        let cand = sink.provide_clock().expect("audio sink offers a clock");
        // AudioProvider so audio outranks a video sink's plain Provider.
        assert_eq!(cand.priority, ClockPriority::AudioProvider);
    }

    #[test]
    fn provide_clock_property_toggles_the_candidate() {
        let mut sink = PulseSink::new();
        assert_eq!(
            sink.get_property("provide-clock"),
            Some(PropValue::Bool(true))
        );

        sink.set_property("provide-clock", PropValue::Bool(false))
            .unwrap();
        assert_eq!(
            sink.get_property("provide-clock"),
            Some(PropValue::Bool(false))
        );
        assert!(
            sink.provide_clock().is_none(),
            "disabled sink offers no clock"
        );

        // Wrong value type is rejected, not silently accepted.
        assert_eq!(
            sink.set_property("provide-clock", PropValue::Str("yes".into())),
            Err(PropError::Type)
        );
    }

    #[test]
    fn client_name_property_sets_the_application_name() {
        let mut sink = PulseSink::new();
        assert_eq!(
            sink.get_property("client-name"),
            Some(PropValue::Str("glass2glass".into()))
        );
        sink.set_property("client-name", PropValue::Str("g2g-test".into()))
            .unwrap();
        assert_eq!(sink.app_name, "g2g-test");
    }

    #[test]
    fn pad_template_is_pcm_sink_only() {
        use g2g_core::{PadDirection, PadTemplates};
        let sink = PulseSink::pad_template(PadDirection::Sink).expect("has sink pad");
        let pcm = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        assert!(matches!(sink.caps, g2g_core::PadCaps::Fixed(ref s) if s.accepts(&pcm)));
        assert!(PulseSink::pad_template(PadDirection::Source).is_none());
    }
}
