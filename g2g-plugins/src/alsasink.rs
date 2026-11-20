// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

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

use std::sync::mpsc::Receiver;
use std::sync::Arc;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use crate::alsa_ffi::{Direction, Pcm};

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ClockCandidate, ClockSync, ConfigureOutcome,
    DeviceClockOffset, DriftClock, ElementMetadata, G2gError, HardwareError, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::alsapcm::{alsa_params, device_permutation, permute, PcmConfig, FORMATS};
use crate::audiooutput::{
    DeviceCmd, DeviceStep, OutputDevice, PausableDevice, PcmGeometry, PlayEnd, PlayoutGate,
    PlayoutLink,
};
use g2g_core::pcm_caps_any_rate_and_channels;

fn alsa_err(code: i32) -> G2gError {
    G2gError::Hardware(HardwareError::Alsa(code))
}

const ELEMENT_NAME: &str = "AlsaSink";
const DEFAULT_DEVICE: &str = "default";

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
    output: OutputDevice,
    /// Bounded, non-blocking link to the device thread. The executor is
    /// cooperative, so neither the hand-off nor the end-of-stream drain may
    /// block it (see [`crate::audioworker`]).
    link: PlayoutLink,
    caps: Option<Caps>,
    frames_rendered: Arc<AtomicU64>,
    playout: crate::audioskew::PlayoutClock,
}

impl core::fmt::Debug for AlsaSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AlsaSink")
            .field("output", &self.output)
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
        Self::with_device(DEFAULT_DEVICE)
    }

    /// Render to a named ALSA PCM device (e.g. `hw:0,0`, `plughw:1`).
    pub fn with_device(device: impl Into<String>) -> Self {
        Self {
            output: OutputDevice::new("device", device),
            link: PlayoutLink::new(G2gError::Hardware(HardwareError::Alsa(-1))),
            caps: None,
            frames_rendered: Arc::new(AtomicU64::new(0)),
            playout: crate::audioskew::PlayoutClock::default(),
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
        self.playout.clock()
    }
}

impl AsyncElement for AlsaSink {
    fn latency(&self) -> g2g_core::LatencyReport {
        self.link.latency()
    }

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
        alsa_params(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(accepted_pcm_caps())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let cfg = alsa_params(absolute_caps)?;
        crate::alsa_ffi::require_library()?;

        if self.link.is_running() && self.caps.as_ref() == Some(absolute_caps) {
            return Ok(ConfigureOutcome::Accepted);
        }

        let rendered = Arc::clone(&self.frames_rendered);
        // disciplined even when not offered, skew slaving reads it
        let clock = self.playout.clock();
        self.playout.configure(absolute_caps)?;
        let frame_bytes = cfg
            .format
            .bytes_per_sample()
            .ok_or(G2gError::CapsMismatch)? as u64
            * u64::from(cfg.channels);
        let fallback = self.output.fallback(
            frame_bytes * u64::from(cfg.rate),
            Arc::clone(&self.frames_rendered),
            frame_bytes,
            |frame| {
                crate::dmabufmap::frame_bytes(frame, ELEMENT_NAME)
                    .ok()
                    .map(|bytes| bytes.as_slice().len())
            },
        );

        let mut clock_offset = DeviceClockOffset::default();
        self.link.open(
            "g2g-alsasink",
            absolute_caps,
            &mut self.output,
            fallback,
            move |device| open_pcm(device, cfg).map_err(alsa_err),
            move |pcm, rx, gate| play(pcm, cfg, rx, gate, &rendered, &clock, &mut clock_offset),
        )?;
        self.caps = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    /// Offer the DAC-disciplined [`clock`](Self::clock) as an
    /// [`AudioProvider`](g2g_core::ClockPriority::AudioProvider) so audio becomes the
    /// pipeline master (video slaves to it), unless `provide-clock` is off.
    fn provide_clock(&self) -> Option<ClockCandidate> {
        self.playout.candidate()
    }

    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.link.set_clock_sync(sync.clone());
        self.playout.set_clock_sync(sync);
    }

    fn adopt_position(
        &mut self,
        position: g2g_core::progress::SinkPosition,
    ) -> Option<g2g_core::progress::SinkPosition> {
        self.link.adopt_position(position)
    }

    fn set_bus(&mut self, bus: g2g_core::bus::BusHandle) {
        self.output.set_bus(bus);
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "ALSA audio sink",
            "Sink/Audio",
            "Plays interleaved PCM via ALSA",
            crate::AARON_BOXER,
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new(
                "device",
                PropKind::Str,
                "ALSA PCM device (e.g. default, hw:0,0, plughw:1)",
            )
            .with_default(DEFAULT_DEVICE),
            crate::audioskew::PROVIDE_CLOCK_PROPERTY,
            crate::audioskew::SLAVE_METHOD_PROPERTY,
            crate::audioskew::DRIFT_TOLERANCE_PROPERTY,
            crate::audiooutput::CONTINUE_ON_ERROR_PROPERTY,
            g2g_core::PROCESSING_DEADLINE_PROPERTY,
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if let Some(result) = self.playout.set_property(name, &value) {
            return result;
        }
        if let Some(result) = self.link.set_property(name, &value) {
            return result;
        }
        self.output
            .set_property(name, &value, self.link.is_running())
            .unwrap_or(Err(PropError::Unknown))
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        if let Some(value) = self.link.get_property(name) {
            return Some(value);
        }
        if let Some(value) = self.playout.get_property(name) {
            return Some(value);
        }
        self.output.get_property(name)
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
                    let frame = self.playout.correct_frame(frame, ELEMENT_NAME)?;
                    self.link.send(frame, &mut self.output).await
                }
                // A mid-stream format change can't be honoured on an open
                // device; only a caps identical to the configured one passes.
                PipelinePacket::CapsChanged(c) => {
                    alsa_params(&c)?;
                    Ok(())
                }
                PipelinePacket::Flush => self.link.flush().await,
                PipelinePacket::Segment(segment) => {
                    self.link.set_segment(segment);
                    Ok(())
                }
                PipelinePacket::Eos => self.link.finish().await,
                // future PipelinePacket variants: no-op (terminal sink).
                _ => Ok(()),
            }
        })
    }
}

fn accepted_pcm_caps() -> CapsSet {
    pcm_caps_any_rate_and_channels(|format| FORMATS.iter().any(|(opened, _)| *opened == format))
}

impl PadTemplates for AlsaSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(accepted_pcm_caps())])
    }
}

// =================================================================
// Worker thread: blocking ALSA writei
// =================================================================

fn play(
    pcm: Pcm,
    cfg: PcmConfig,
    rx: &Receiver<DeviceCmd>,
    gate: &mut PlayoutGate,
    rendered: &AtomicU64,
    clock: &DriftClock,
    clock_offset: &mut DeviceClockOffset,
) -> PlayEnd {
    let perm = device_permutation(&pcm, cfg.channels);
    let mut permuted = Vec::new();
    let geometry = match PcmGeometry::new(cfg.format, cfg.channels, cfg.rate) {
        Ok(geometry) => geometry,
        Err(error) => return PlayEnd::Failed(error),
    };
    let mut device = AlsaPause {
        pcm: &pcm,
        rate: cfg.rate,
        paused_in_device: false,
    };
    let end = loop {
        match gate.recv(rx) {
            Ok(None) => {}
            Ok(Some(DeviceStep::Samples(frame))) => {
                let bytes = match crate::dmabufmap::frame_bytes(&frame, ELEMENT_NAME) {
                    Ok(bytes) => bytes,
                    Err(error) => break PlayEnd::Failed(error),
                };
                let played = gate.play_in_slices(
                    rx,
                    bytes.as_slice(),
                    geometry,
                    &mut device,
                    |device, slice| {
                        write_all(
                            device.pcm,
                            cfg,
                            perm.as_deref(),
                            &mut permuted,
                            slice,
                            rendered,
                        )?;
                        // After the blocking write returns (at ~the device rate), feed
                        // the drift clock one playout observation.
                        discipline_clock(device.pcm, cfg.rate, rendered, clock, clock_offset);
                        Ok(())
                    },
                );
                if let Err(error) = played {
                    return PlayEnd::Failed(error);
                }
            }
            Ok(Some(DeviceStep::Flush)) => {
                if let Err(error) = pcm.drop_pending().and_then(|()| pcm.prepare()) {
                    return PlayEnd::Failed(alsa_err(error));
                }
            }
            Ok(Some(DeviceStep::SwitchDevice(device))) => break PlayEnd::SwitchDevice(device),
            Ok(Some(DeviceStep::Shutdown)) | Err(_) => break PlayEnd::Shutdown,
        }
    };
    // Play out whatever is still buffered, then stop.
    let _ = pcm.drain();
    end
}

struct AlsaPause<'a> {
    pcm: &'a Pcm,
    rate: u32,
    paused_in_device: bool,
}

impl PausableDevice for AlsaPause<'_> {
    fn pause(&mut self) -> Result<(), G2gError> {
        // alsa devices without hardware pause refuse it, then play out their buffer and underrun
        self.paused_in_device = self.pcm.pause(true).is_ok();
        Ok(())
    }

    fn resume(&mut self) -> Result<(), G2gError> {
        if self.paused_in_device {
            self.pcm.pause(false).map_err(alsa_err)?;
        }
        Ok(())
    }

    fn still_playing_after_pause_ns(&self) -> Result<u64, G2gError> {
        match self.paused_in_device {
            true => Ok(0),
            false => self.delay_ns(),
        }
    }

    fn delay_ns(&self) -> Result<u64, G2gError> {
        // alsa refuses the delay query on a device that stopped, which plays nothing
        let delay = self.pcm.delay().unwrap_or(0).max(0) as u64;
        Ok(crate::audioconvert::samples_to_ns(delay, self.rate))
    }
}

/// Feed the drift clock one `(monotonic_now, played_ns)` observation. Frames
/// actually played = frames handed to `writei` minus the `snd_pcm_delay` still
/// queued in the ring; that is the true DAC playout position, which drifts from
/// wall time at the hardware's real rate. The local time is sampled next to the
/// `delay()` probe so the pair lines up.
fn discipline_clock(
    pcm: &Pcm,
    rate: u32,
    rendered: &AtomicU64,
    clock: &DriftClock,
    clock_offset: &mut DeviceClockOffset,
) {
    let local_ns = clock.reference_now();
    // A failed / negative delay probe (device not running yet) yields no usable
    // playout position; skip rather than feed a bogus sample.
    let Ok(delay) = pcm.delay() else { return };
    let buffered = delay.max(0) as u64;
    let played = rendered.load(Ordering::Relaxed).saturating_sub(buffered);
    if played == 0 {
        return;
    }
    let master_ns = crate::audioskew::played_master_ns(clock, clock_offset, local_ns, played, rate);
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
pub(crate) fn default_device_opens() -> Result<(), String> {
    crate::alsa_ffi::library_installed()?;
    // gst's autoaudiosink skips a sink whose device fails to open
    Pcm::open(DEFAULT_DEVICE, Direction::Playback, true)
        .map(drop)
        .map_err(|code| {
            alloc::format!("the {DEFAULT_DEVICE} playback device does not open: error {code}")
        })
}

fn open_pcm(device: &str, cfg: PcmConfig) -> Result<Pcm, i32> {
    let pcm = Pcm::open(device, Direction::Playback, false)?;
    {
        let hwp = pcm.hardware_parameters()?;
        hwp.set_channels(cfg.channels)?;
        hwp.set_rate(cfg.rate)?;
        hwp.set_format(cfg.fmt)?;
        hwp.set_access_interleaved()?;
        // Bound the ring so `writei` blocks (paces) at the device rate. Without
        // this some backends (pipewire-alsa) expose a very large default buffer,
        // so the whole stream queues without ever blocking, which both defeats
        // the pacing this sink relies on and pins `snd_pcm_delay` at the full
        // backlog, so the M590 playout-clock discipline sees zero frames played.
        // ~200 ms buffer / ~20 ms period; `_near` so a device that cannot honour
        // the exact size picks its closest instead of failing configure.
        hwp.set_period_time_near(20_000)?;
        hwp.set_buffer_time_near(200_000)?;
        pcm.apply(&hwp)?;
    }
    pcm.prepare()?;
    Ok(pcm)
}

/// Write a whole interleaved buffer, looping over partial writes and recovering
/// from underruns (`-EPIPE`). The bytes go to `writei` as-is: the device was
/// opened with the matching little-endian ALSA format, so no per-format
/// reinterpretation is needed and 3-byte S24 works like the rest.
fn write_all(
    pcm: &Pcm,
    cfg: PcmConfig,
    perm: Option<&[usize]>,
    permuted: &mut Vec<u8>,
    bytes: &[u8],
    rendered: &AtomicU64,
) -> Result<(), G2gError> {
    let sample = cfg
        .format
        .bytes_per_sample()
        .ok_or(G2gError::CapsMismatch)?;
    let frame = sample * cfg.channels as usize;
    if frame == 0 {
        return Err(G2gError::CapsMismatch);
    }
    let buf = match perm {
        Some(p) => {
            permute(bytes, p, sample, permuted);
            &permuted[..]
        }
        None => &bytes[..bytes.len() / frame * frame],
    };

    let mut off = 0usize;
    while off < buf.len() {
        let remaining = &buf[off..];
        match pcm.write_interleaved(remaining, remaining.len() / frame) {
            Ok(frames) => {
                rendered.fetch_add(frames as u64, Ordering::Relaxed);
                off += frames * frame;
            }
            Err(e) => {
                // Underrun / suspend: recover and retry the remainder.
                pcm.recover(e).map_err(alsa_err)?;
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
        assert_eq!(sink.output.name(), "hw:0,0");
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
