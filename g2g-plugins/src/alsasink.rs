//! Linux ALSA render sink. The audible-output end of the audio path on Linux,
//! the analog of the Windows-only [`WasapiSink`](crate::wasapisink::WasapiSink).
//! Consumes interleaved PCM (`PcmU8` / `PcmS16Le` / `PcmS24Le` / `PcmS32Le` /
//! `PcmF32Le`) `DataFrame`s and plays them on an ALSA PCM device (`default` by
//! default) via libasound.
//!
//! ## Channel order
//!
//! Pipeline PCM is interleaved in the WAV / ffmpeg order
//! ([`ChannelLayout::default_for`]), but ALSA reports its own per-device map:
//! a 5.1 device is typically `FL FR RL RR FC LFE`, not our `FL FR FC LFE BL BR`.
//! The worker reads the device map after `hw_params` and permutes each buffer
//! into it, so surround content lands on the right speakers.
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

use alsa::pcm::{Access, ChmapPosition, Format, HwParams, PCM};
use alsa::{Direction, ValueOr};

use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ChannelLayout, ChannelPosition,
    ClockCandidate, ClockPriority, ConfigureOutcome, DriftClock, ElementMetadata, G2gError,
    HardwareError, MonotonicClock, OutputSink, PadTemplate, PadTemplates, PipelineClock,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::audioconvert::sample_bytes;

/// The PCM sample formats this sink opens a device with, and the ALSA format
/// each maps to. `PcmS24Le` is our 3-byte packed layout, ALSA's `S24_3LE`
/// (not `S24LE`, which is 24 bits inside a 32-bit container).
const FORMATS: [(AudioFormat, Format); 5] = [
    (AudioFormat::PcmU8, Format::U8),
    (AudioFormat::PcmS16Le, Format::S16LE),
    (AudioFormat::PcmS24Le, Format::S243LE),
    (AudioFormat::PcmS32Le, Format::S32LE),
    (AudioFormat::PcmF32Le, Format::FloatLE),
];

/// Negotiated PCM device parameters passed to the worker as one unit (keeps the
/// worker signature under clippy's argument cap).
#[derive(Clone, Copy, Debug, PartialEq)]
struct PcmConfig {
    /// Pipeline sample format: drives the byte-level frame arithmetic.
    format: AudioFormat,
    /// The ALSA format the device is opened with.
    fmt: Format,
    channels: u32,
    rate: u32,
}

/// Negotiated PCM parameters. Compressed audio (AAC / Opus) is rejected
/// structurally, as in `WasapiSink`.
fn alsa_params(caps: &Caps) -> Result<PcmConfig, G2gError> {
    let Caps::Audio {
        format,
        channels,
        sample_rate,
    } = caps
    else {
        return Err(G2gError::CapsMismatch);
    };
    let (_, fmt) = FORMATS
        .iter()
        .find(|(f, _)| f == format)
        .ok_or(G2gError::CapsMismatch)?;
    Ok(PcmConfig {
        format: *format,
        fmt: *fmt,
        channels: u32::from(*channels),
        rate: *sample_rate,
    })
}

/// Worker command. `Samples` carries one buffer of interleaved PCM bytes in the
/// negotiated format; `Shutdown` asks the worker to drain and stop.
enum WorkerCmd {
    Samples(Vec<u8>),
    Shutdown,
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
        let cfg = alsa_params(absolute_caps)?;

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
    clock.observe(local_ns, master_ns);
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

/// Our speaker position for an ALSA channel-map position. `None` for one
/// outside the [`ChannelLayout`] table (or an unpositioned channel), which
/// makes the caller fall back to writing the buffer straight through.
fn our_position(p: ChmapPosition) -> Option<ChannelPosition> {
    Some(match p {
        ChmapPosition::FL => ChannelPosition::Fl,
        ChmapPosition::FR => ChannelPosition::Fr,
        ChmapPosition::FC | ChmapPosition::Mono => ChannelPosition::Fc,
        ChmapPosition::LFE => ChannelPosition::Lfe,
        ChmapPosition::RL => ChannelPosition::Bl,
        ChmapPosition::RR => ChannelPosition::Br,
        ChmapPosition::SL => ChannelPosition::Sl,
        ChmapPosition::SR => ChannelPosition::Sr,
        ChmapPosition::RC => ChannelPosition::Bc,
        ChmapPosition::FLC => ChannelPosition::Flc,
        ChmapPosition::FRC => ChannelPosition::Frc,
        _ => return None,
    })
}

/// The permutation from our interleave order (the default layout for
/// `channels`) into the device's reported channel map: `perm[d]` is the source
/// channel that feeds device channel `d`. `None` when the two orders already
/// agree, the device reports no map, or a device position falls outside our
/// layout: the buffer then goes out unpermuted. Must be called after
/// `hw_params`, which is when ALSA fixes the map.
fn device_permutation(pcm: &PCM, channels: u32) -> Option<Vec<usize>> {
    let layout = ChannelLayout::default_for(u8::try_from(channels).ok()?)?;
    let positions: Vec<ChmapPosition> = (&pcm.get_chmap().ok()?).into();
    if positions.len() != channels as usize {
        return None;
    }
    let mut perm = Vec::with_capacity(positions.len());
    for p in positions {
        perm.push(layout.index_of(our_position(p)?)?);
    }
    if perm.iter().enumerate().all(|(d, s)| d == *s) {
        return None;
    }
    Some(perm)
}

/// Reorder interleaved frames into the device's channel order: output channel
/// `d` of each frame takes source channel `perm[d]`. A ragged trailing partial
/// frame is dropped (it cannot be written as a frame anyway).
fn permute(bytes: &[u8], perm: &[usize], sample: usize) -> Vec<u8> {
    let frame = sample * perm.len();
    let mut out = Vec::with_capacity(bytes.len());
    for f in bytes.chunks_exact(frame) {
        for &s in perm {
            out.extend_from_slice(&f[s * sample..][..sample]);
        }
    }
    out
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

fn alsa_err(code: i32) -> G2gError {
    G2gError::Hardware(HardwareError::Alsa(code))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(format: AudioFormat, channels: u8, rate: u32) -> Caps {
        Caps::Audio {
            format,
            channels,
            sample_rate: rate,
        }
    }

    #[test]
    fn alsa_params_maps_formats_and_rejects_compressed() {
        let p = alsa_params(&caps(AudioFormat::PcmS16Le, 2, 48_000)).unwrap();
        assert_eq!((p.fmt, p.channels, p.rate), (Format::S16LE, 2, 48_000));
        let p = alsa_params(&caps(AudioFormat::PcmF32Le, 1, 44_100)).unwrap();
        assert_eq!((p.fmt, p.channels, p.rate), (Format::FloatLE, 1, 44_100));
        // 24-bit is the 3-byte packed ALSA format, not the 32-bit container.
        assert_eq!(
            alsa_params(&caps(AudioFormat::PcmS24Le, 6, 48_000))
                .unwrap()
                .fmt,
            Format::S243LE
        );
        assert_eq!(
            alsa_params(&caps(AudioFormat::PcmS32Le, 2, 48_000))
                .unwrap()
                .fmt,
            Format::S32LE
        );
        assert_eq!(
            alsa_params(&caps(AudioFormat::PcmU8, 2, 8_000))
                .unwrap()
                .fmt,
            Format::U8
        );
        assert_eq!(
            alsa_params(&caps(AudioFormat::Aac, 2, 48_000)),
            Err(G2gError::CapsMismatch)
        );
        // a raw format the sink does not open a device with is rejected too.
        assert_eq!(
            alsa_params(&caps(AudioFormat::Mulaw, 1, 8_000)),
            Err(G2gError::CapsMismatch)
        );
    }

    #[test]
    fn permute_reorders_frames_into_the_device_map() {
        // 5.1: our FL FR FC LFE BL BR -> the usual ALSA FL FR RL RR FC LFE.
        let perm = [0usize, 1, 4, 5, 2, 3];
        // one frame of s16 channel markers 0..5.
        let src: Vec<u8> = (0i16..6).flat_map(|c| c.to_le_bytes()).collect();
        let out = permute(&src, &perm, 2);
        let got: Vec<i16> = out
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(got, [0, 1, 4, 5, 2, 3]);
    }

    #[test]
    fn permute_handles_three_byte_samples_and_drops_a_ragged_tail() {
        // S24 is 3 bytes: a swap of a stereo frame moves 3 bytes at a time.
        let src = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let out = permute(&src, &[1, 0], 3);
        assert_eq!(out, [4, 5, 6, 1, 2, 3]);
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
