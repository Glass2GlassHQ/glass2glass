//! Linux ALSA capture source: the input mirror of
//! [`AlsaSink`](crate::alsasink::AlsaSink) and the non-PipeWire sibling of
//! [`PipeWireSrc`](crate::pipewiresrc::PipeWireSrc). Reads interleaved PCM from
//! an ALSA capture device (`default` by default) via libasound and emits
//! `DataFrame`s, so a microphone / line-in feeds a g2g pipeline the way
//! `AudioTestSrc` feeds a synthetic tone.
//!
//! The requested format / rate / channels are set exactly (`snd_pcm_hw_params`
//! with an exact rate), so the produced caps are deterministic from the
//! properties and negotiation needs no device round-trip: a launch line parses
//! and solves on a host with no sound card, and only `run` fails loud. Point it
//! at `default` (which is `plug:` on a PipeWire / dmix host) for automatic
//! conversion; a raw `hw:0,0` accepts only what the card natively supports.
//!
//! ## Channel order
//!
//! The device reports its own channel map, so the worker permutes each captured
//! period from the device order back into our interleave order (the inverse of
//! what `AlsaSink` applies, see `alsapcm`).
//!
//! ## Threading
//!
//! `snd_pcm_readi` is a blocking call, so it runs on a dedicated worker spun up
//! in `run`; captured periods cross to the async loop over a channel, which
//! stamps timing and pushes them.

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, Ordering};

use std::sync::mpsc as std_mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use alsa::pcm::{Access, HwParams, PCM};
use alsa::{Direction, ValueOr};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, FrameTiming,
    G2gError, HardwareError, LatencyReport, MemoryDomain, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::alsapcm::{alsa_params, device_permutation, invert, permute, PcmConfig, FORMATS};
use crate::audioconvert::{audio_format_from_str, audio_format_to_str};

/// Ring-buffer span, microseconds (gst `alsasrc buffer-time`).
const DEFAULT_BUFFER_US: u32 = 200_000;
/// One period, microseconds (gst `alsasrc latency-time`): the read granularity.
const DEFAULT_LATENCY_US: u32 = 10_000;

/// Upper bound on a device-reported period, in sample frames. The period size
/// comes back from `snd_pcm_hw_params_get_period_size`, so it decides an
/// allocation: cap it rather than trusting whatever the driver reports. 1 M
/// frames is ~21 s at 48 kHz, far past any real period.
const MAX_PERIOD_FRAMES: usize = 1 << 20;

/// # Example
///
/// ```no_run
/// use g2g_plugins::alsasrc::AlsaSrc;
///
/// // gst-launch equivalent: alsasrc device=hw:0 ! ...
/// let source = AlsaSrc::new().with_device("hw:0").with_rate(48_000);
/// ```
#[derive(Debug)]
pub struct AlsaSrc {
    device: String,
    format: AudioFormat,
    channels: u8,
    rate: u32,
    buffer_us: u32,
    latency_us: u32,
    /// `u64::MAX` = capture until error or downstream shutdown; else stop after
    /// N periods and emit EOS (the bounded / test path).
    num_buffers: u64,
    configured: bool,
}

impl Default for AlsaSrc {
    fn default() -> Self {
        Self::new()
    }
}

impl AlsaSrc {
    /// Capture S16LE stereo at 48 kHz from the ALSA `default` device, until
    /// downstream stops.
    pub fn new() -> Self {
        Self {
            device: String::from("default"),
            format: AudioFormat::PcmS16Le,
            channels: 2,
            rate: 48_000,
            buffer_us: DEFAULT_BUFFER_US,
            latency_us: DEFAULT_LATENCY_US,
            num_buffers: u64::MAX,
            configured: false,
        }
    }

    /// Capture from a named ALSA PCM device (e.g. `hw:0,0`, `plughw:1`).
    pub fn with_device(mut self, device: impl Into<String>) -> Self {
        self.device = device.into();
        self
    }

    /// Request a PCM sample format (any of the five [`FORMATS`] map).
    pub fn with_format(mut self, format: AudioFormat) -> Self {
        self.format = format;
        self
    }

    /// Request a sample rate in Hz.
    pub fn with_rate(mut self, rate: u32) -> Self {
        self.rate = rate;
        self
    }

    /// Request a channel count.
    pub fn with_channels(mut self, channels: u8) -> Self {
        self.channels = channels;
        self
    }

    /// Ring-buffer span in microseconds (gst `buffer-time`).
    pub fn with_buffer_time(mut self, us: u32) -> Self {
        self.buffer_us = us;
        self
    }

    /// Period span in microseconds (gst `latency-time`): one captured buffer.
    pub fn with_latency_time(mut self, us: u32) -> Self {
        self.latency_us = us;
        self
    }

    /// Stop after `n` captured periods and emit EOS.
    pub fn with_num_buffers(mut self, n: u64) -> Self {
        self.num_buffers = n;
        self
    }

    fn caps(&self) -> Result<Caps, G2gError> {
        let caps = Caps::Audio {
            format: self.format,
            channels: self.channels,
            sample_rate: self.rate,
        };
        // Reject a format no ALSA device opens (compressed, mulaw, ...) up front.
        alsa_params(&caps)?;
        Ok(caps)
    }
}

impl SourceLoop for AlsaSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(self.caps())
    }

    /// Produces the fixed PCM caps the device is opened with, so a chain built
    /// on the mic takes the native arc-consistency path (mirrors `PipeWireSrc`).
    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(
            self.caps()
                .map(|c| CapsConstraint::Produces(CapsSet::one(c))),
        )
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.caps()?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    /// Live source: one period is device-driven, so report it as the live
    /// latency this source contributes.
    fn latency(&self) -> LatencyReport {
        LatencyReport::live(u64::from(self.latency_us) * 1_000, None)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "ALSA audio source",
            "Source/Audio",
            "Captures interleaved PCM from an ALSA device",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        ALSASRC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "device" => self.device = value.as_str().ok_or(PropError::Type)?.to_string(),
            "format" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.format = audio_format_from_str(s).ok_or(PropError::Value)?;
            }
            "samplerate" => self.rate = value.as_uint().ok_or(PropError::Type)? as u32,
            "channels" => self.channels = value.as_uint().ok_or(PropError::Type)? as u8,
            "buffer-time" => self.buffer_us = value.as_uint().ok_or(PropError::Type)? as u32,
            "latency-time" => self.latency_us = value.as_uint().ok_or(PropError::Type)? as u32,
            "num-buffers" => crate::numbuffers::set_num_buffers(&mut self.num_buffers, &value)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "device" => Some(PropValue::Str(self.device.clone())),
            "format" => Some(PropValue::Str(audio_format_to_str(self.format).into())),
            "samplerate" => Some(PropValue::Uint(u64::from(self.rate))),
            "channels" => Some(PropValue::Uint(u64::from(self.channels))),
            "buffer-time" => Some(PropValue::Uint(u64::from(self.buffer_us))),
            "latency-time" => Some(PropValue::Uint(u64::from(self.latency_us))),
            "num-buffers" => Some(crate::numbuffers::get_num_buffers(self.num_buffers)),
            _ => None,
        }
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let cfg = alsa_params(&self.caps()?)?;
            let frame_bytes = cfg.frame_bytes();
            if frame_bytes == 0 {
                return Err(G2gError::CapsMismatch);
            }
            let limit = self.num_buffers;

            // Captured periods cross from the blocking worker to here.
            let (audio_tx, mut audio_rx) = tokio::sync::mpsc::unbounded_channel::<Vec<u8>>();
            let (ready_tx, ready_rx) = std_mpsc::sync_channel::<Result<(), i32>>(1);
            let stop = Arc::new(AtomicBool::new(false));

            let device = self.device.clone();
            let timing = (self.buffer_us, self.latency_us);
            let worker_stop = Arc::clone(&stop);
            let worker = thread::Builder::new()
                .name(String::from("g2g-alsasrc"))
                .spawn(move || {
                    worker_main(&device, cfg, timing, audio_tx, worker_stop, ready_tx);
                })
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

            // The worker reports whether the device opened; a host with no ALSA
            // capture device fails loud here rather than pushing silence.
            let ready = ready_rx.recv_timeout(Duration::from_secs(5));
            if !matches!(ready, Ok(Ok(()))) {
                stop.store(true, Ordering::Relaxed);
                let _ = worker.join();
                let code = match ready {
                    Ok(Err(code)) => code,
                    _ => -1,
                };
                return Err(G2gError::Hardware(HardwareError::Alsa(code)));
            }

            let mut seq = 0u64;
            let mut frames_total = 0u64;
            let mut downstream_open = true;
            while seq < limit {
                let Some(bytes) = audio_rx.recv().await else {
                    break; // worker ended
                };
                let n_frames = (bytes.len() / frame_bytes) as u64;
                if n_frames == 0 {
                    continue;
                }
                let pts_ns = frames_total * 1_000_000_000 / u64::from(cfg.rate);
                let end_ns = (frames_total + n_frames) * 1_000_000_000 / u64::from(cfg.rate);
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                    timing: FrameTiming {
                        pts_ns,
                        dts_ns: pts_ns,
                        duration_ns: end_ns - pts_ns,
                        capture_ns: pts_ns,
                        arrival_ns: g2g_core::metrics::monotonic_ns(),
                        keyframe: false, // audio: every buffer is independent
                    },
                    sequence: seq,
                    meta: Default::default(),
                };
                if out.push(PipelinePacket::DataFrame(frame)).await.is_err() {
                    downstream_open = false;
                    break;
                }
                frames_total += n_frames;
                seq += 1;
            }

            // Stop the blocking read loop and reap the worker.
            stop.store(true, Ordering::Relaxed);
            drop(audio_rx);
            let _ = worker.join();

            if downstream_open {
                out.push(PipelinePacket::Eos).await?;
            }
            Ok(seq)
        })
    }
}

impl PadTemplates for AlsaSrc {
    /// Produces PCM; a constructed instance fixes the format / rate / channels.
    /// `Caps::Audio` has no open dims, so the template pins the common stereo /
    /// 48 kHz shape per format, as in `AlsaSink`.
    fn pad_templates() -> Vec<PadTemplate> {
        let pcm = |(format, _)| Caps::Audio {
            format,
            channels: 2,
            sample_rate: 48_000,
        };
        Vec::from([PadTemplate::source(CapsSet::from_alternatives(
            FORMATS.map(pcm).to_vec(),
        ))])
    }
}

/// `AlsaSrc`'s settable properties. `device`, `buffer-time` and `latency-time`
/// match gst `alsasrc`; `num-buffers` matches gst `basesrc`. Rate / channels /
/// format are caps on the gst side, so they take the g2g `audiotestsrc` names.
static ALSASRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "device",
        PropKind::Str,
        "ALSA PCM device (e.g. default, hw:0,0, plughw:1)",
    )
    .with_default("default"),
    PropertySpec::new(
        "format",
        PropKind::Str,
        "capture sample format: S16LE | F32LE | S24LE | S32LE | U8",
    )
    .with_default("S16LE"),
    PropertySpec::new("samplerate", PropKind::Uint, "samples per second").with_default("48000"),
    PropertySpec::new("channels", PropKind::Uint, "channel count").with_default("2"),
    PropertySpec::new(
        "buffer-time",
        PropKind::Uint,
        "ring buffer span in microseconds",
    )
    .with_default("200000"),
    PropertySpec::new(
        "latency-time",
        PropKind::Uint,
        "period span in microseconds (one captured buffer)",
    )
    .with_default("10000"),
    PropertySpec::new(
        "num-buffers",
        PropKind::Int,
        "periods to capture then EOS (-1 = forever)",
    )
    .with_default("-1"),
];

// =================================================================
// Worker thread: blocking ALSA readi
// =================================================================

fn worker_main(
    device: &str,
    cfg: PcmConfig,
    timing: (u32, u32),
    audio_tx: tokio::sync::mpsc::UnboundedSender<Vec<u8>>,
    stop: Arc<AtomicBool>,
    ready: std_mpsc::SyncSender<Result<(), i32>>,
) {
    let (pcm, period_frames) = match open_pcm(device, cfg, timing) {
        Ok(open) => {
            let _ = ready.send(Ok(()));
            open
        }
        Err(code) => {
            let _ = ready.send(Err(code));
            return;
        }
    };
    // Device order -> our interleave order: the reverse of what AlsaSink applies.
    let perm = device_permutation(&pcm, cfg.channels).map(invert);
    let sample = cfg.frame_bytes() / cfg.channels as usize;

    let io = pcm.io_bytes();
    let mut buf = vec![0u8; period_frames * cfg.frame_bytes()];
    while !stop.load(Ordering::Relaxed) {
        let frames = match io.readi(&mut buf) {
            // A driver could report more frames than the buffer holds; clamp
            // before it becomes a slice length.
            Ok(frames) => frames.min(period_frames),
            // Overrun / suspend: recover and retry, else give up.
            Err(e) => {
                if pcm.try_recover(e, true).is_err() {
                    break;
                }
                continue;
            }
        };
        if frames == 0 {
            continue;
        }
        let captured = &buf[..frames * cfg.frame_bytes()];
        let chunk = match perm.as_deref() {
            Some(p) => permute(captured, p, sample),
            None => captured.to_vec(),
        };
        if audio_tx.send(chunk).is_err() {
            break; // consumer dropped
        }
    }
    let _ = pcm.drop();
}

/// Open and configure the device for blocking interleaved capture. Returns the
/// PCM and its period size in sample frames, or the ALSA errno on failure. The
/// rate / channels / format are set exactly, so a device that cannot deliver
/// them fails here instead of silently producing something else.
fn open_pcm(device: &str, cfg: PcmConfig, timing: (u32, u32)) -> Result<(PCM, usize), i32> {
    let (buffer_us, latency_us) = timing;
    let pcm = PCM::new(device, Direction::Capture, false).map_err(|e| e.errno())?;
    let period_frames = {
        let hwp = HwParams::any(&pcm).map_err(|e| e.errno())?;
        hwp.set_channels(cfg.channels).map_err(|e| e.errno())?;
        hwp.set_rate(cfg.rate, ValueOr::Nearest)
            .map_err(|e| e.errno())?;
        hwp.set_format(cfg.fmt).map_err(|e| e.errno())?;
        hwp.set_access(Access::RWInterleaved)
            .map_err(|e| e.errno())?;
        // `_near` so a device that cannot honour the exact span picks its
        // closest instead of failing the open. The period is the read
        // granularity, i.e. one emitted buffer.
        hwp.set_period_time_near(latency_us, ValueOr::Nearest)
            .map_err(|e| e.errno())?;
        hwp.set_buffer_time_near(buffer_us, ValueOr::Nearest)
            .map_err(|e| e.errno())?;
        pcm.hw_params(&hwp).map_err(|e| e.errno())?;
        // Device-reported: it sizes the read buffer, so bound it before it
        // becomes an allocation.
        let frames = hwp.get_period_size().map_err(|e| e.errno())?;
        let frames = usize::try_from(frames).map_err(|_| EINVAL)?;
        if frames == 0 || frames > MAX_PERIOD_FRAMES {
            return Err(EINVAL);
        }
        frames
    };
    pcm.prepare().map_err(|e| e.errno())?;
    pcm.start().map_err(|e| e.errno())?;
    Ok((pcm, period_frames))
}

/// The errno reported for a device whose period size we refuse. Spelled out so
/// this module needs no libc dependency.
const EINVAL: i32 = -22;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builders_set_requested_config() {
        let src = AlsaSrc::new()
            .with_device("hw:1,0")
            .with_format(AudioFormat::PcmF32Le)
            .with_rate(44_100)
            .with_channels(1)
            .with_buffer_time(100_000)
            .with_latency_time(20_000)
            .with_num_buffers(5);
        assert_eq!(src.device, "hw:1,0");
        assert_eq!(src.format, AudioFormat::PcmF32Le);
        assert_eq!((src.channels, src.rate), (1, 44_100));
        assert_eq!((src.buffer_us, src.latency_us), (100_000, 20_000));
        assert_eq!(src.num_buffers, 5);
    }

    #[test]
    fn caps_reflect_request_and_reject_non_pcm() {
        let src = AlsaSrc::new().with_rate(16_000).with_channels(1);
        assert_eq!(
            src.caps(),
            Ok(Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: 1,
                sample_rate: 16_000,
            })
        );
        let bad = AlsaSrc::new().with_format(AudioFormat::Opus);
        assert_eq!(bad.caps(), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn defaults_match_the_declared_property_defaults() {
        let src = AlsaSrc::new();
        for spec in ALSASRC_PROPS {
            let declared = spec.default.expect("every alsasrc prop declares a default");
            let live = src.get_property(spec.name).expect("readable");
            let live = match live {
                PropValue::Str(s) => s,
                PropValue::Uint(u) => alloc::format!("{u}"),
                PropValue::Int(i) => alloc::format!("{i}"),
                other => panic!("unexpected kind for {}: {other:?}", spec.name),
            };
            assert_eq!(live, declared, "{} default", spec.name);
        }
    }

    #[test]
    fn latency_tracks_the_period_property() {
        let mut src = AlsaSrc::new();
        assert_eq!(src.latency().min_ns, 10_000_000);
        src.set_property("latency-time", PropValue::Uint(25_000))
            .unwrap();
        assert_eq!(src.latency().min_ns, 25_000_000);
    }

    #[test]
    fn pad_template_is_pcm_source_only() {
        use g2g_core::PadDirection;
        let source = AlsaSrc::pad_template(PadDirection::Source).expect("has source pad");
        let pcm = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
        };
        assert!(matches!(source.caps, g2g_core::PadCaps::Fixed(ref s) if s.accepts(&pcm)));
        assert!(AlsaSrc::pad_template(PadDirection::Sink).is_none());
    }
}
