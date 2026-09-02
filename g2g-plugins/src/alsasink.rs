//! Linux ALSA render sink. The audible-output end of the audio path on Linux,
//! the analog of the Windows-only [`WasapiSink`](crate::wasapisink::WasapiSink).
//! Consumes interleaved PCM (`PcmU8` / `PcmS16Le` / `PcmS24Le` / `PcmS32Le` /
//! `PcmF32Le`) `DataFrame`s and plays them on an ALSA PCM device (`default` by
//! default) via libasound.
//!
//! ## Channel order
//!
//! The worker reads the device channel map after `hw_params` and permutes each
//! buffer into it (see `alsapcm`), so surround content lands on the right
//! speakers.
//!
//! ## Threading
//!
//! `snd_pcm_writei` is a blocking call, so (like `WasapiSink`) all of it lives
//! on a dedicated worker spun up at `configure_pipeline`. The sink struct holds
//! only `Send` handles (an mpsc sender plus a shared counter); PCM bytes cross
//! to the worker by value.
//!
//! ## Pacing
//!
//! ALSA's blocking `writei` *is* the pacing: it returns once the ring buffer
//! has space, i.e. at the device rate. The source pushes faster than real time
//! (`AudioTestSrc` emits without sleeping), so the worker queues bursts and the
//! blocking write feeds them out at the hardware clock. On `Eos` the worker
//! drains the ring (`snd_pcm_drain`) so the tail is not cut off.

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicU64, Ordering};

use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::Arc;

use std::time::Duration;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use alsa::pcm::{Access, HwParams, PCM};
use alsa::{Direction, ValueOr};

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ClockCandidate, ClockPriority, ConfigureOutcome,
    DriftClock, ElementMetadata, G2gError, HardwareError, MonotonicClock, OutputSink, PadTemplate,
    PadTemplates, PipelineClock, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::alsapcm::{alsa_params, device_permutation, permute, PcmConfig, FORMATS};
use crate::audioconvert::sample_bytes;

fn alsa_err(code: i32) -> G2gError {
    G2gError::Hardware(HardwareError::Alsa(code))
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
/// use g2g_plugins::alsasink::AlsaSink;
///
/// // gst-launch equivalent: alsasink device=hw:0,0
/// let sink = AlsaSink::with_device("hw:0,0");
/// assert_eq!(sink.frames_rendered(), 0);
/// ```
pub struct AlsaSink {
    device: String,
    /// Bounded, non-blocking link to the device thread. The executor is
    /// cooperative, so neither the hand-off nor the end-of-stream drain may
    /// block it (see [`crate::audioworker`]).
    link: Option<crate::audioworker::WorkerLink<WorkerCmd>>,
    caps: Option<Caps>,
    frames_rendered: Arc<AtomicU64>,
    /// DAC-disciplined master clock (M590 A/V sync). The worker feeds it
    /// `(monotonic_now, frames_played)` observations so its `now_ns()` tracks
    /// the real playout rate; a video sink slaves to it when it is elected.
    clock: Arc<DriftClock>,
    /// Whether to offer [`clock`](Self::clock) to the pipeline's clock election
    /// (the `provide-clock` property, default on, GStreamer's `basesink`).
    provide_clock: bool,
}

impl core::fmt::Debug for AlsaSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AlsaSink")
            .field("device", &self.device)
            .field("caps", &self.caps)
            .field(
                "frames_rendered",
                &self.frames_rendered.load(Ordering::Relaxed),
            )
            .finish()
    }
}

impl Default for AlsaSink {
    fn default() -> Self {
        Self::new()
    }
}

impl AlsaSink {
    /// Render to the ALSA `default` device.
    pub fn new() -> Self {
        Self::with_device("default")
    }

    /// Render to a named ALSA PCM device (e.g. `hw:0,0`, `plughw:1`).
    pub fn with_device(device: impl Into<String>) -> Self {
        Self {
            device: device.into(),
            link: None,
            caps: None,
            frames_rendered: Arc::new(AtomicU64::new(0)),
            clock: Arc::new(DriftClock::new(Arc::new(MonotonicClock))),
            provide_clock: true,
        }
    }

    /// Count of sample frames written to the device. Useful in tests.
    pub fn frames_rendered(&self) -> u64 {
        self.frames_rendered.load(Ordering::Relaxed)
    }

    /// The DAC-disciplined clock this sink offers to election. Exposed for
    /// tests / introspection; its `now_ns()` tracks real playout once the
    /// worker has observed the device.
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

impl Drop for AlsaSink {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl AsyncElement for AlsaSink {
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
        alsa_params(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    /// PCM only. `Caps::Audio` has no open dims, so the per-rate/channel
    /// acceptance rides the legacy intercept bridge, as in `WasapiSink`.
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::LegacySink(Box::new(|c: &Caps| {
            alsa_params(c)?;
            Ok(c.clone())
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let cfg = alsa_params(absolute_caps)?;

        if self.link.as_ref().is_some_and(|l| l.is_running()) {
            if self.caps.as_ref() == Some(absolute_caps) {
                return Ok(ConfigureOutcome::Accepted);
            }
            self.shutdown();
        }

        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<(), i32>>(1);
        let rendered = Arc::clone(&self.frames_rendered);
        let device = self.device.clone();
        // Only discipline the clock when we actually offer it; otherwise the
        // per-buffer `delay()` probe is wasted work no one reads.
        let clock = self.provide_clock.then(|| Arc::clone(&self.clock));

        let link = crate::audioworker::WorkerLink::spawn("g2g-alsasink", move |rx| {
            worker_main(&device, cfg, rx, rendered, clock, ready_tx);
        })?;

        // The worker reports whether the device opened; a host with no ALSA
        // device fails loud here rather than silently dropping audio.
        match ready_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(())) => {}
            // `link`'s Drop reaps the worker, so a failed open needs no join.
            Ok(Err(code)) => return Err(G2gError::Hardware(HardwareError::Alsa(code))),
            Err(_) => return Err(G2gError::Hardware(HardwareError::Alsa(-1))),
        }

        self.link = Some(link);
        self.caps = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    /// Offer the DAC-disciplined [`clock`](Self::clock) as an
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
            "ALSA audio sink",
            "Sink/Audio",
            "Plays interleaved PCM via ALSA",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new(
                "device",
                PropKind::Str,
                "ALSA PCM device (e.g. default, hw:0,0, plughw:1)",
            )
            .with_default("default"),
            PropertySpec::new(
                "provide-clock",
                PropKind::Bool,
                "Provide a DAC-disciplined clock so audio is the A/V sync master",
            )
            .with_default("true"),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "device" => {
                self.device = value.as_str().ok_or(PropError::Type)?.into();
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
            "device" => Some(PropValue::Str(self.device.clone())),
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
                    let slice = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    let samples = slice.to_vec();
                    let link = self.link.as_ref().ok_or(G2gError::NotConfigured)?;
                    // Awaited, not blocking: a full queue is the device's
                    // back-pressure and yields to the executor.
                    link.send(
                        WorkerCmd::Samples(samples),
                        G2gError::Hardware(HardwareError::Alsa(-1)),
                    )
                    .await?;
                    Ok(())
                }
                // A mid-stream format change can't be honoured on an open
                // device; only a caps identical to the configured one passes.
                PipelinePacket::CapsChanged(c) => {
                    alsa_params(&c)?;
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
                            G2gError::Hardware(HardwareError::Alsa(-1)),
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

impl PadTemplates for AlsaSink {
    /// Terminal PCM sink pad. `Caps::Audio` has no open dims, so the template
    /// pins the common shapes per PCM format, as in `WasapiSink`.
    fn pad_templates() -> Vec<PadTemplate> {
        let pcm = |(format, _)| Caps::Audio {
            format,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        Vec::from([PadTemplate::sink(CapsSet::from_alternatives(
            FORMATS.map(pcm).to_vec(),
        ))])
    }
}

// =================================================================
// Worker thread: blocking ALSA writei
// =================================================================

fn worker_main(
    device: &str,
    cfg: PcmConfig,
    rx: Receiver<WorkerCmd>,
    rendered: Arc<AtomicU64>,
    clock: Option<Arc<DriftClock>>,
    ready: SyncSender<Result<(), i32>>,
) {
    let pcm = match open_pcm(device, cfg) {
        Ok(pcm) => {
            let _ = ready.send(Ok(()));
            pcm
        }
        Err(code) => {
            let _ = ready.send(Err(code));
            return;
        }
    };
    let perm = device_permutation(&pcm, cfg.channels);

    let mut closing = false;
    while !closing {
        match rx.recv() {
            Ok(WorkerCmd::Samples(bytes)) => {
                if write_all(&pcm, cfg, perm.as_deref(), &bytes, &rendered).is_err() {
                    break;
                }
                // After the blocking write returns (at ~the device rate), feed
                // the drift clock one playout observation.
                if let Some(clock) = clock.as_deref() {
                    discipline_clock(&pcm, cfg.rate, &rendered, clock);
                }
            }
            Ok(WorkerCmd::Shutdown) | Err(_) => closing = true,
        }
    }
    // Play out whatever is still buffered, then stop.
    let _ = pcm.drain();
}

/// Feed the drift clock one `(monotonic_now, played_ns)` observation. Frames
/// actually played = frames handed to `writei` minus the `snd_pcm_delay` still
/// queued in the ring; that is the true DAC playout position, which drifts from
/// wall time at the hardware's real rate. The local time is sampled next to the
/// `delay()` probe so the pair lines up.
fn discipline_clock(pcm: &PCM, rate: u32, rendered: &AtomicU64, clock: &DriftClock) {
    let local_ns = clock.reference_now();
    // A failed / negative delay probe (device not running yet) yields no usable
    // playout position; skip rather than feed a bogus sample.
    let Ok(delay) = pcm.delay() else { return };
    let buffered = delay.max(0) as u64;
    let played = rendered.load(Ordering::Relaxed).saturating_sub(buffered);
    if played == 0 {
        return;
    }
    let master_ns = (u128::from(played) * 1_000_000_000 / u128::from(rate)) as u64;
    let outcome = clock.observe(local_ns, master_ns);
    // Pacing diagnosis: how the DAC clock the video slaves to actually moves,
    // and whether the outlier gate dropped this reading.
    g2g_core::g2g_log!(
        g2g_core::log::Target::category("AlsaSink"),
        "clock local={}ms master={}ms buffered={} slope={:.6} {:?}",
        local_ns / 1_000_000,
        master_ns / 1_000_000,
        buffered,
        clock.slope(),
        outcome
    );
}

/// Open and configure the PCM device for blocking interleaved playback.
/// Returns the ALSA errno on failure.
fn open_pcm(device: &str, cfg: PcmConfig) -> Result<PCM, i32> {
    let pcm = PCM::new(device, Direction::Playback, false).map_err(|e| e.errno())?;
    {
        let hwp = HwParams::any(&pcm).map_err(|e| e.errno())?;
        hwp.set_channels(cfg.channels).map_err(|e| e.errno())?;
        hwp.set_rate(cfg.rate, ValueOr::Nearest)
            .map_err(|e| e.errno())?;
        hwp.set_format(cfg.fmt).map_err(|e| e.errno())?;
        hwp.set_access(Access::RWInterleaved)
            .map_err(|e| e.errno())?;
        // Bound the ring so `writei` blocks (paces) at the device rate. Without
        // this some backends (pipewire-alsa) expose a very large default buffer,
        // so the whole stream queues without ever blocking, which both defeats
        // the pacing this sink relies on and pins `snd_pcm_delay` at the full
        // backlog, so the M590 playout-clock discipline sees zero frames played.
        // ~200 ms buffer / ~20 ms period; `_near` so a device that cannot honour
        // the exact size picks its closest instead of failing configure.
        hwp.set_period_time_near(20_000, ValueOr::Nearest)
            .map_err(|e| e.errno())?;
        hwp.set_buffer_time_near(200_000, ValueOr::Nearest)
            .map_err(|e| e.errno())?;
        pcm.hw_params(&hwp).map_err(|e| e.errno())?;
    }
    pcm.prepare().map_err(|e| e.errno())?;
    Ok(pcm)
}

/// Write a whole interleaved buffer, looping over partial writes and recovering
/// from underruns (`-EPIPE`). The bytes go to `writei` as-is: the device was
/// opened with the matching little-endian ALSA format, so no per-format
/// reinterpretation is needed and 3-byte S24 works like the rest.
fn write_all(
    pcm: &PCM,
    cfg: PcmConfig,
    perm: Option<&[usize]>,
    bytes: &[u8],
    rendered: &AtomicU64,
) -> Result<(), G2gError> {
    let sample = sample_bytes(cfg.format);
    let frame = sample * cfg.channels as usize;
    if frame == 0 {
        return Err(G2gError::CapsMismatch);
    }
    let permuted;
    let buf = match perm {
        Some(p) => {
            permuted = permute(bytes, p, sample);
            &permuted[..]
        }
        None => &bytes[..bytes.len() / frame * frame],
    };

    let io = pcm.io_bytes();
    let mut off = 0usize;
    while off < buf.len() {
        match io.writei(&buf[off..]) {
            Ok(frames) => {
                rendered.fetch_add(frames as u64, Ordering::Relaxed);
                off += frames * frame;
            }
            Err(e) => {
                // Underrun / suspend: recover and retry the remainder.
                pcm.try_recover(e, true).map_err(|e| alsa_err(e.errno()))?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use g2g_core::AudioFormat;

    #[test]
    fn intercept_accepts_pcm_rejects_compressed() {
        let sink = AlsaSink::new();
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
    fn pad_template_is_pcm_sink_only() {
        use g2g_core::{PadDirection, PadTemplates};
        let sink = AlsaSink::pad_template(PadDirection::Sink).expect("has sink pad");
        let pcm = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        assert!(matches!(sink.caps, g2g_core::PadCaps::Fixed(ref s) if s.accepts(&pcm)));
        assert!(AlsaSink::pad_template(PadDirection::Source).is_none());
    }

    #[test]
    fn with_device_sets_name() {
        let sink = AlsaSink::with_device("hw:0,0");
        assert_eq!(sink.device, "hw:0,0");
    }

    #[test]
    fn provides_an_audio_master_clock_by_default() {
        use g2g_core::ClockPriority;
        let sink = AlsaSink::new();
        let cand = sink.provide_clock().expect("audio sink offers a clock");
        // AudioProvider so audio outranks a video sink's plain Provider.
        assert_eq!(cand.priority, ClockPriority::AudioProvider);
    }

    #[test]
    fn provide_clock_property_toggles_the_candidate() {
        let mut sink = AlsaSink::new();
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
}
