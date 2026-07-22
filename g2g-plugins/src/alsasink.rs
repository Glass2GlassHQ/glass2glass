//! Linux ALSA render sink. The audible-output end of the audio path on Linux,
//! the analog of the Windows-only [`WasapiSink`](crate::wasapisink::WasapiSink).
//! Consumes interleaved PCM (`PcmS16Le` / `PcmF32Le`) `DataFrame`s and plays
//! them on an ALSA PCM device (`default` by default) via libasound.
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

use std::sync::mpsc::{self, Receiver, Sender, SyncSender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use alsa::pcm::{Access, Format, HwParams, PCM};
use alsa::{Direction, ValueOr};

use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ClockCandidate, ClockPriority,
    ConfigureOutcome, DriftClock, ElementMetadata, G2gError, HardwareError, MonotonicClock,
    OutputSink, PadTemplate, PadTemplates, PipelineClock, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec,
};

/// Negotiated PCM parameters: (ALSA sample format, channels, rate). Compressed
/// audio (AAC / Opus) is rejected structurally, as in `WasapiSink`.
fn alsa_params(caps: &Caps) -> Result<(Format, u32, u32), G2gError> {
    let Caps::Audio {
        format,
        channels,
        sample_rate,
    } = caps
    else {
        return Err(G2gError::CapsMismatch);
    };
    let fmt = match format {
        AudioFormat::PcmS16Le => Format::S16LE,
        AudioFormat::PcmF32Le => Format::FloatLE,
        AudioFormat::Aac | AudioFormat::Opus => return Err(G2gError::CapsMismatch),
        _ => return Err(G2gError::CapsMismatch),
    };
    Ok((fmt, u32::from(*channels), *sample_rate))
}

/// Worker command. `Samples` carries one buffer of interleaved PCM bytes in the
/// negotiated format; `Shutdown` asks the worker to drain and stop.
enum WorkerCmd {
    Samples(Vec<u8>),
    Shutdown,
}

/// Negotiated PCM device parameters passed to the worker as one unit (keeps the
/// worker signature under clippy's argument cap).
#[derive(Clone, Copy, Debug)]
struct PcmConfig {
    fmt: Format,
    channels: u32,
    rate: u32,
}

pub struct AlsaSink {
    device: String,
    cmd_tx: Option<Sender<WorkerCmd>>,
    worker: Option<JoinHandle<()>>,
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
            cmd_tx: None,
            worker: None,
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

    fn shutdown(&mut self) {
        if let Some(tx) = self.cmd_tx.take() {
            let _ = tx.send(WorkerCmd::Shutdown);
        }
        if let Some(join) = self.worker.take() {
            let _ = join.join();
        }
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
        let (fmt, channels, rate) = alsa_params(absolute_caps)?;

        if self.worker.is_some() {
            if self.caps.as_ref() == Some(absolute_caps) {
                return Ok(ConfigureOutcome::Accepted);
            }
            self.shutdown();
        }

        let (tx, rx) = mpsc::channel::<WorkerCmd>();
        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<(), i32>>(1);
        let rendered = Arc::clone(&self.frames_rendered);
        let device = self.device.clone();
        // Only discipline the clock when we actually offer it; otherwise the
        // per-buffer `delay()` probe is wasted work no one reads.
        let clock = self.provide_clock.then(|| Arc::clone(&self.clock));
        let cfg = PcmConfig {
            fmt,
            channels,
            rate,
        };

        let join = thread::Builder::new()
            .name(String::from("g2g-alsasink"))
            .spawn(move || {
                worker_main(&device, cfg, rx, rendered, clock, ready_tx);
            })
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

        // The worker reports whether the device opened; a host with no ALSA
        // device fails loud here rather than silently dropping audio.
        match ready_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(())) => {}
            Ok(Err(code)) => {
                let _ = join.join();
                return Err(G2gError::Hardware(HardwareError::Alsa(code)));
            }
            Err(_) => {
                let _ = join.join();
                return Err(G2gError::Hardware(HardwareError::Alsa(-1)));
            }
        }

        self.cmd_tx = Some(tx);
        self.worker = Some(join);
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
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let tx = self.cmd_tx.as_ref().ok_or(G2gError::NotConfigured)?;
                    tx.send(WorkerCmd::Samples(slice.to_vec()))
                        .map_err(|_| G2gError::Hardware(HardwareError::Alsa(-1)))?;
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
                    self.shutdown();
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
        let pcm = |format| Caps::Audio {
            format,
            channels: 2,
            sample_rate: 48_000,
        };
        Vec::from([PadTemplate::sink(CapsSet::from_alternatives(Vec::from([
            pcm(AudioFormat::PcmS16Le),
            pcm(AudioFormat::PcmF32Le),
        ])))])
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
    let PcmConfig {
        fmt,
        channels,
        rate,
    } = cfg;
    let pcm = match open_pcm(device, fmt, channels, rate) {
        Ok(pcm) => {
            let _ = ready.send(Ok(()));
            pcm
        }
        Err(code) => {
            let _ = ready.send(Err(code));
            return;
        }
    };

    let ch = channels as usize;
    let mut closing = false;
    while !closing {
        match rx.recv() {
            Ok(WorkerCmd::Samples(bytes)) => {
                if write_all(&pcm, fmt, ch, &bytes, &rendered).is_err() {
                    break;
                }
                // After the blocking write returns (at ~the device rate), feed
                // the drift clock one playout observation.
                if let Some(clock) = clock.as_deref() {
                    discipline_clock(&pcm, rate, &rendered, clock);
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
    clock.observe(local_ns, master_ns);
}

/// Open and configure the PCM device for blocking interleaved playback.
/// Returns the ALSA errno on failure.
fn open_pcm(device: &str, fmt: Format, channels: u32, rate: u32) -> Result<PCM, i32> {
    let pcm = PCM::new(device, Direction::Playback, false).map_err(|e| e.errno())?;
    {
        let hwp = HwParams::any(&pcm).map_err(|e| e.errno())?;
        hwp.set_channels(channels).map_err(|e| e.errno())?;
        hwp.set_rate(rate, ValueOr::Nearest)
            .map_err(|e| e.errno())?;
        hwp.set_format(fmt).map_err(|e| e.errno())?;
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
/// from underruns (`-EPIPE`). The byte buffer is reinterpreted into the typed
/// samples ALSA's `writei` expects, endian-safe via `from_le_bytes`.
fn write_all(
    pcm: &PCM,
    fmt: Format,
    channels: usize,
    bytes: &[u8],
    rendered: &AtomicU64,
) -> Result<(), G2gError> {
    match fmt {
        Format::S16LE => {
            let samples: Vec<i16> = bytes
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                .collect();
            let io = pcm.io_i16().map_err(|e| alsa_err(e.errno()))?;
            write_typed(pcm, |buf| io.writei(buf), &samples, channels, rendered)
        }
        Format::FloatLE => {
            let samples: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let io = pcm.io_f32().map_err(|e| alsa_err(e.errno()))?;
            write_typed(pcm, |buf| io.writei(buf), &samples, channels, rendered)
        }
        _ => Err(G2gError::CapsMismatch),
    }
}

/// Drive one typed `writei` to completion. `write` is the format-specific
/// closure; `samples` is interleaved (items, not frames).
fn write_typed<S>(
    pcm: &PCM,
    mut write: impl FnMut(&[S]) -> Result<usize, alsa::Error>,
    samples: &[S],
    channels: usize,
    rendered: &AtomicU64,
) -> Result<(), G2gError> {
    let mut off = 0usize;
    while off < samples.len() {
        match write(&samples[off..]) {
            Ok(frames) => {
                rendered.fetch_add(frames as u64, Ordering::Relaxed);
                off += frames * channels;
            }
            Err(e) => {
                // Underrun / suspend: recover and retry the remainder.
                pcm.try_recover(e, true).map_err(|e| alsa_err(e.errno()))?;
            }
        }
    }
    Ok(())
}

fn alsa_err(code: i32) -> G2gError {
    G2gError::Hardware(HardwareError::Alsa(code))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alsa_params_maps_formats_and_rejects_compressed() {
        let s16 = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
        };
        assert_eq!(alsa_params(&s16), Ok((Format::S16LE, 2, 48_000)));
        let f32 = Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: 1,
            sample_rate: 44_100,
        };
        assert_eq!(alsa_params(&f32), Ok((Format::FloatLE, 1, 44_100)));
        let aac = Caps::Audio {
            format: AudioFormat::Aac,
            channels: 2,
            sample_rate: 48_000,
        };
        assert_eq!(alsa_params(&aac), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn intercept_accepts_pcm_rejects_compressed() {
        let sink = AlsaSink::new();
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
    fn pad_template_is_pcm_sink_only() {
        use g2g_core::{PadDirection, PadTemplates};
        let sink = AlsaSink::pad_template(PadDirection::Sink).expect("has sink pad");
        let pcm = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
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
