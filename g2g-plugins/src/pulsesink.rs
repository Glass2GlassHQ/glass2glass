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

use std::sync::mpsc::{self, Receiver, SyncSender};
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
    AsyncElement, Caps, CapsConstraint, CapsSet, ClockCandidate, ClockPriority, ConfigureOutcome,
    DriftClock, ElementMetadata, G2gError, HardwareError, MonotonicClock, OutputSink, PadTemplate,
    PadTemplates, PipelineClock, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::pulsepcm::{pulse_map, pulse_spec, FORMATS};

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

/// Worker command. `Samples` carries one buffer of interleaved PCM bytes in the
/// negotiated format; `Shutdown` asks the worker to drain and stop.
enum WorkerCmd {
    Samples(Vec<u8>),
    Shutdown,
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::pulsesink::PulseSink;
///
/// let sink = PulseSink::with_app_name("my-player");
/// ```
pub struct PulseSink {
    app_name: String,
    /// Bounded, non-blocking link to the device thread. The executor is
    /// cooperative, so neither the hand-off nor the end-of-stream drain may
    /// block it (see [`crate::audioworker`]).
    link: Option<crate::audioworker::WorkerLink<WorkerCmd>>,
    caps: Option<Caps>,
    bytes_written: Arc<AtomicU64>,
    /// Playout-disciplined master clock (M884, the `AlsaSink` M590 mechanism
    /// over pulse's latency query). The worker feeds it
    /// `(monotonic_now, played_ns)` observations so its `now_ns()` tracks the
    /// real playout rate; a video sink slaves to it when it is elected.
    clock: Arc<DriftClock>,
    /// Whether to offer [`clock`](Self::clock) to the pipeline's clock election
    /// (the `provide-clock` property, default on, GStreamer's `basesink`).
    provide_clock: bool,
}

impl core::fmt::Debug for PulseSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PulseSink")
            .field("app_name", &self.app_name)
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
            link: None,
            caps: None,
            bytes_written: Arc::new(AtomicU64::new(0)),
            clock: Arc::new(DriftClock::new(Arc::new(MonotonicClock))),
            provide_clock: true,
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

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        pulse_spec(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    /// PCM only. `Caps::Audio` has no open dims, so the per-rate/channel
    /// acceptance rides the legacy intercept bridge, as in `WasapiSink`.
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::LegacySink(Box::new(|c: &Caps| {
            pulse_spec(c)?;
            Ok(c.clone())
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let spec = pulse_spec(absolute_caps)?;

        if self.link.as_ref().is_some_and(|l| l.is_running()) {
            if self.caps.as_ref() == Some(absolute_caps) {
                return Ok(ConfigureOutcome::Accepted);
            }
            self.shutdown();
        }

        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<(), i32>>(1);
        let written = Arc::clone(&self.bytes_written);
        let app_name = self.app_name.clone();
        // Only discipline the clock when we actually offer it; otherwise the
        // per-buffer latency query is wasted work no one reads.
        let clock = self.provide_clock.then(|| Arc::clone(&self.clock));

        let link = crate::audioworker::WorkerLink::spawn("g2g-pulsesink", move |rx| {
            worker_main(&app_name, spec, rx, written, clock, ready_tx);
        })?;

        // The worker reports whether the server connection opened; a host with
        // no PulseAudio server fails loud here rather than dropping audio.
        match ready_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(())) => {}
            // `link`'s Drop reaps the worker, so a failed open needs no join.
            Ok(Err(code)) => return Err(G2gError::Hardware(HardwareError::PulseAudio(code))),
            Err(_) => return Err(G2gError::Hardware(HardwareError::PulseAudio(-1))),
        }

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
                "provide-clock",
                PropKind::Bool,
                "Provide a playout-disciplined clock so audio is the A/V sync master",
            )
            .with_default("true"),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
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
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let samples = slice.to_vec();
                    let link = self.link.as_ref().ok_or(G2gError::NotConfigured)?;
                    // Awaited, not blocking: a full queue is the device's
                    // back-pressure and yields to the executor.
                    link.send(
                        WorkerCmd::Samples(samples),
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
                            WorkerCmd::Shutdown,
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

impl PadTemplates for PulseSink {
    /// Terminal PCM sink pad. `Caps::Audio` has no open dims, so the template
    /// pins the common shapes per PCM format, as in `WasapiSink`.
    fn pad_templates() -> Vec<PadTemplate> {
        let pcm = |(format, _)| Caps::Audio {
            format,
            channels: 2,
            sample_rate: 48_000,
        };
        Vec::from([PadTemplate::sink(CapsSet::from_alternatives(
            FORMATS.map(pcm).to_vec(),
        ))])
    }
}

// =================================================================
// Worker thread: blocking libpulse simple write
// =================================================================

fn worker_main(
    app_name: &str,
    spec: Spec,
    rx: Receiver<WorkerCmd>,
    written: Arc<AtomicU64>,
    clock: Option<Arc<DriftClock>>,
    ready: SyncSender<Result<(), i32>>,
) {
    let map = pulse_map(spec.channels);
    let attr = buffer_attr(&spec);
    let simple = match Simple::new(
        None,     // default server
        app_name, // application name
        Direction::Playback,
        None,       // default device
        "playback", // stream description
        &spec,
        map.as_ref(),
        Some(&attr),
    ) {
        Ok(s) => {
            let _ = ready.send(Ok(()));
            s
        }
        Err(e) => {
            let _ = ready.send(Err(e.0));
            return;
        }
    };

    let mut closing = false;
    while !closing {
        match rx.recv() {
            Ok(WorkerCmd::Samples(bytes)) => {
                if simple.write(&bytes).is_err() {
                    break;
                }
                written.fetch_add(bytes.len() as u64, Ordering::Relaxed);
                // After the blocking write returns (at ~the device rate), feed
                // the drift clock one playout observation.
                if let Some(clock) = clock.as_deref() {
                    discipline_clock(&simple, &spec, &written, clock);
                }
            }
            Ok(WorkerCmd::Shutdown) | Err(_) => closing = true,
        }
    }
    // Play out whatever is still buffered, then stop.
    let _ = simple.drain();
}

/// Feed the drift clock one `(monotonic_now, played_ns)` observation. Bytes
/// actually played = bytes handed to `pa_simple_write` minus the latency the
/// server still reports (the queue plus the device's own, pulse's analog of
/// `snd_pcm_delay`), both converted to time through the stream spec. That is the
/// true playout position, which drifts from wall time at the hardware's real
/// rate. The local time is sampled next to the latency query so the pair lines up.
fn discipline_clock(simple: &Simple, spec: &Spec, written: &AtomicU64, clock: &DriftClock) {
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
    clock.observe(local_ns, played_us.saturating_mul(1_000));
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
        };
        assert_eq!(sink.intercept_caps(&pcm), Ok(pcm));
        let opus = Caps::Audio {
            format: AudioFormat::Opus,
            channels: 2,
            sample_rate: 48_000,
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
        };
        assert!(matches!(sink.caps, g2g_core::PadCaps::Fixed(ref s) if s.accepts(&pcm)));
        assert!(PulseSink::pad_template(PadDirection::Source).is_none());
    }
}
