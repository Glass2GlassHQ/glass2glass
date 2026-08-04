//! Linux PulseAudio capture source: the input mirror of
//! [`PulseSink`](crate::pulsesink::PulseSink) and the higher-level sibling of
//! [`AlsaSrc`](crate::alsasrc::AlsaSrc). Records interleaved PCM from a
//! PulseAudio (or PipeWire-pulse) source and emits `DataFrame`s.
//!
//! The requested format / rate / channels are what the server converts to, so
//! the produced caps are deterministic from the properties and negotiation needs
//! no server round-trip: a launch line parses and solves with no server running,
//! and only `run` fails loud. An empty `device` records the server's default
//! source, which on a desktop exists even with no microphone (the output
//! monitor).
//!
//! ## Threading
//!
//! `pa_simple_read` blocks until a whole fragment has been recorded, so it runs
//! on a dedicated worker spun up in `run`, exactly as `PulseSink` does its
//! writes. Fragments cross to the async loop over a channel, which stamps timing
//! and pushes them.

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

use libpulse_binding::def::BufferAttr;
use libpulse_binding::sample::Spec;
use libpulse_binding::stream::Direction;
use libpulse_simple_binding::Simple;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, FrameTiming,
    G2gError, HardwareError, LatencyReport, MemoryDomain, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::audioconvert::{audio_format_from_str, audio_format_to_str, sample_bytes};
use crate::pulsepcm::{opt_name, pulse_map, pulse_spec, FORMATS};

/// Server-side record buffer span, microseconds (gst `pulsesrc buffer-time`).
const DEFAULT_BUFFER_US: u32 = 200_000;
/// One fragment, microseconds (gst `pulsesrc latency-time`): the read granularity.
const DEFAULT_LATENCY_US: u32 = 10_000;

/// Upper bound on a fragment, in sample frames, so an absurd `latency-time`
/// cannot ask for a giant allocation. 1 M frames is ~21 s at 48 kHz.
const MAX_FRAGMENT_FRAMES: usize = 1 << 20;

/// What the worker needs to open and read the record stream. Grouped so the
/// worker signature stays under clippy's argument cap.
#[derive(Debug, Clone)]
struct StreamConfig {
    server: String,
    device: String,
    client_name: String,
    spec: Spec,
    attr: BufferAttr,
    /// Bytes read per `pa_simple_read`, i.e. one emitted buffer.
    fragment_bytes: usize,
}

#[derive(Debug)]
pub struct PulseSrc {
    /// Empty = the default server (`PULSE_SERVER` / the local socket).
    server: String,
    /// Empty = the server's default source.
    device: String,
    client_name: String,
    format: AudioFormat,
    channels: u8,
    rate: u32,
    buffer_us: u32,
    latency_us: u32,
    /// `u64::MAX` = record until error or downstream shutdown; else stop after
    /// N fragments and emit EOS (the bounded / test path).
    num_buffers: u64,
    configured: bool,
}

impl Default for PulseSrc {
    fn default() -> Self {
        Self::new()
    }
}

impl PulseSrc {
    /// Record S16LE stereo at 48 kHz from the default server's default source,
    /// until downstream stops.
    pub fn new() -> Self {
        Self {
            server: String::new(),
            device: String::new(),
            client_name: String::from("glass2glass"),
            format: AudioFormat::PcmS16Le,
            channels: 2,
            rate: 48_000,
            buffer_us: DEFAULT_BUFFER_US,
            latency_us: DEFAULT_LATENCY_US,
            num_buffers: u64::MAX,
            configured: false,
        }
    }

    /// Record from a named source (`pactl list sources short`); empty = default.
    pub fn with_device(mut self, device: impl Into<String>) -> Self {
        self.device = device.into();
        self
    }

    /// Connect to a named server instead of the default one.
    pub fn with_server(mut self, server: impl Into<String>) -> Self {
        self.server = server.into();
        self
    }

    /// Record under a custom application name (shown in the mixer).
    pub fn with_client_name(mut self, name: impl Into<String>) -> Self {
        self.client_name = name.into();
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

    /// Server-side buffer span in microseconds (gst `buffer-time`).
    pub fn with_buffer_time(mut self, us: u32) -> Self {
        self.buffer_us = us;
        self
    }

    /// Fragment span in microseconds (gst `latency-time`): one recorded buffer.
    pub fn with_latency_time(mut self, us: u32) -> Self {
        self.latency_us = us;
        self
    }

    /// Stop after `n` recorded fragments and emit EOS.
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
        // Reject a format no pulse stream carries (compressed, mulaw, ...).
        pulse_spec(&caps)?;
        Ok(caps)
    }

    /// The stream parameters, with the two time properties folded into a
    /// [`BufferAttr`]. The playback-only fields are `u32::MAX` ("server picks"),
    /// which is what a record stream wants: only `maxlength` and `fragsize`
    /// apply to it.
    fn stream_config(&self) -> Result<StreamConfig, G2gError> {
        let spec = pulse_spec(&self.caps()?)?;
        let frame_bytes = sample_bytes(self.format) * self.channels as usize;
        if frame_bytes == 0 {
            return Err(G2gError::CapsMismatch);
        }
        let bytes_for = |us: u32| -> Option<usize> {
            let frames = usize::try_from(u64::from(us) * u64::from(self.rate) / 1_000_000).ok()?;
            if frames == 0 || frames > MAX_FRAGMENT_FRAMES {
                return None;
            }
            frames.checked_mul(frame_bytes)
        };
        let fragment_bytes = bytes_for(self.latency_us).ok_or(G2gError::CapsMismatch)?;
        let maxlength = bytes_for(self.buffer_us).ok_or(G2gError::CapsMismatch)?;
        Ok(StreamConfig {
            server: self.server.clone(),
            device: self.device.clone(),
            client_name: self.client_name.clone(),
            spec,
            attr: BufferAttr {
                maxlength: u32::try_from(maxlength).unwrap_or(u32::MAX),
                tlength: u32::MAX,
                prebuf: u32::MAX,
                minreq: u32::MAX,
                fragsize: u32::try_from(fragment_bytes).unwrap_or(u32::MAX),
            },
            fragment_bytes,
        })
    }
}

impl SourceLoop for PulseSrc {
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

    /// Produces the fixed PCM caps the server converts to, so a chain built on
    /// the mic takes the native arc-consistency path (mirrors `PipeWireSrc`).
    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(
            self.caps()
                .map(|c| CapsConstraint::Produces(CapsSet::one(c))),
        )
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.stream_config()?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    /// Live source: one fragment is server-driven, so report it as the live
    /// latency this source contributes.
    fn latency(&self) -> LatencyReport {
        LatencyReport::live(u64::from(self.latency_us) * 1_000, None)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "PulseAudio audio source",
            "Source/Audio",
            "Captures interleaved PCM via PulseAudio",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        PULSESRC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "server" => self.server = value.as_str().ok_or(PropError::Type)?.to_string(),
            "device" => self.device = value.as_str().ok_or(PropError::Type)?.to_string(),
            "client-name" => self.client_name = value.as_str().ok_or(PropError::Type)?.to_string(),
            "format" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.format = audio_format_from_str(s).ok_or(PropError::Value)?;
            }
            "samplerate" => self.rate = value.as_uint().ok_or(PropError::Type)? as u32,
            "channels" => self.channels = value.as_uint().ok_or(PropError::Type)? as u8,
            "buffer-time" => self.buffer_us = value.as_uint().ok_or(PropError::Type)? as u32,
            "latency-time" => self.latency_us = value.as_uint().ok_or(PropError::Type)? as u32,
            "num-buffers" => {
                let n = value.as_int().ok_or(PropError::Type)?;
                self.num_buffers = if n < 0 { u64::MAX } else { n as u64 };
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "server" => Some(PropValue::Str(self.server.clone())),
            "device" => Some(PropValue::Str(self.device.clone())),
            "client-name" => Some(PropValue::Str(self.client_name.clone())),
            "format" => Some(PropValue::Str(audio_format_to_str(self.format).into())),
            "samplerate" => Some(PropValue::Uint(u64::from(self.rate))),
            "channels" => Some(PropValue::Uint(u64::from(self.channels))),
            "buffer-time" => Some(PropValue::Uint(u64::from(self.buffer_us))),
            "latency-time" => Some(PropValue::Uint(u64::from(self.latency_us))),
            "num-buffers" => Some(PropValue::Int(if self.num_buffers == u64::MAX {
                -1
            } else {
                self.num_buffers as i64
            })),
            _ => None,
        }
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let cfg = self.stream_config()?;
            let frame_bytes = sample_bytes(self.format) * self.channels as usize;
            let rate = u64::from(self.rate);
            let limit = self.num_buffers;

            // Recorded fragments cross from the blocking worker to here.
            let (audio_tx, mut audio_rx) = tokio::sync::mpsc::unbounded_channel::<Vec<u8>>();
            let (ready_tx, ready_rx) = std_mpsc::sync_channel::<Result<(), i32>>(1);
            let stop = Arc::new(AtomicBool::new(false));

            let worker_stop = Arc::clone(&stop);
            let worker = thread::Builder::new()
                .name(String::from("g2g-pulsesrc"))
                .spawn(move || worker_main(cfg, audio_tx, worker_stop, ready_tx))
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

            // The worker reports whether the record stream opened; a host with
            // no PulseAudio server fails loud here rather than pushing silence.
            let ready = ready_rx.recv_timeout(Duration::from_secs(5));
            if !matches!(ready, Ok(Ok(()))) {
                stop.store(true, Ordering::Relaxed);
                let _ = worker.join();
                let code = match ready {
                    Ok(Err(code)) => code,
                    _ => -1,
                };
                return Err(G2gError::Hardware(HardwareError::PulseAudio(code)));
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
                let pts_ns = frames_total * 1_000_000_000 / rate;
                let end_ns = (frames_total + n_frames) * 1_000_000_000 / rate;
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

impl PadTemplates for PulseSrc {
    /// Produces PCM; a constructed instance fixes the format / rate / channels.
    /// `Caps::Audio` has no open dims, so the template pins the common stereo /
    /// 48 kHz shape per format, as in `PulseSink`.
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

/// `PulseSrc`'s settable properties. `server`, `device`, `client-name`,
/// `buffer-time` and `latency-time` match gst `pulsesrc`; `num-buffers` matches
/// gst `basesrc`. Rate / channels / format are caps on the gst side, so they
/// take the g2g `audiotestsrc` names.
static PULSESRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "server",
        PropKind::Str,
        "PulseAudio server to connect to (empty = default)",
    )
    .with_default(""),
    PropertySpec::new(
        "device",
        PropKind::Str,
        "source name to record from (empty = server default)",
    )
    .with_default(""),
    PropertySpec::new(
        "client-name",
        PropKind::Str,
        "application name shown in the mixer",
    )
    .with_default("glass2glass"),
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
        "server-side buffer span in microseconds",
    )
    .with_default("200000"),
    PropertySpec::new(
        "latency-time",
        PropKind::Uint,
        "fragment span in microseconds (one recorded buffer)",
    )
    .with_default("10000"),
    PropertySpec::new(
        "num-buffers",
        PropKind::Int,
        "fragments to record then EOS (-1 = forever)",
    )
    .with_default("-1"),
];

// =================================================================
// Worker thread: blocking libpulse simple read
// =================================================================

fn worker_main(
    cfg: StreamConfig,
    audio_tx: tokio::sync::mpsc::UnboundedSender<Vec<u8>>,
    stop: Arc<AtomicBool>,
    ready: std_mpsc::SyncSender<Result<(), i32>>,
) {
    let map = pulse_map(cfg.spec.channels);
    let simple = match Simple::new(
        opt_name(&cfg.server),
        &cfg.client_name,
        Direction::Record,
        opt_name(&cfg.device),
        "record", // stream description
        &cfg.spec,
        map.as_ref(),
        Some(&cfg.attr),
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

    let mut buf = vec![0u8; cfg.fragment_bytes];
    while !stop.load(Ordering::Relaxed) {
        // `pa_simple_read` fills the whole slice or fails; a short read is not
        // a thing, so the fragment length is ours, never the server's.
        if simple.read(&mut buf).is_err() {
            break;
        }
        if audio_tx.send(buf.clone()).is_err() {
            break; // consumer dropped
        }
    }
    // Drop whatever is still queued server-side rather than draining it.
    let _ = simple.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builders_set_requested_config() {
        let src = PulseSrc::new()
            .with_server("unix:/run/user/1000/pulse/native")
            .with_device("alsa_output.monitor")
            .with_client_name("probe")
            .with_format(AudioFormat::PcmF32Le)
            .with_rate(44_100)
            .with_channels(1)
            .with_buffer_time(100_000)
            .with_latency_time(20_000)
            .with_num_buffers(5);
        assert_eq!(src.server, "unix:/run/user/1000/pulse/native");
        assert_eq!(src.device, "alsa_output.monitor");
        assert_eq!(src.client_name, "probe");
        assert_eq!(src.format, AudioFormat::PcmF32Le);
        assert_eq!((src.channels, src.rate), (1, 44_100));
        assert_eq!((src.buffer_us, src.latency_us), (100_000, 20_000));
        assert_eq!(src.num_buffers, 5);
    }

    #[test]
    fn caps_reflect_request_and_reject_non_pcm() {
        let src = PulseSrc::new().with_rate(16_000).with_channels(1);
        assert_eq!(
            src.caps(),
            Ok(Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: 1,
                sample_rate: 16_000,
            })
        );
        let bad = PulseSrc::new().with_format(AudioFormat::Opus);
        assert_eq!(bad.caps(), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn buffer_attr_folds_the_time_properties_into_bytes() {
        // 10 ms of 48 kHz stereo s16 = 480 frames x 4 bytes.
        let cfg = PulseSrc::new().stream_config().expect("default config");
        assert_eq!(cfg.fragment_bytes, 480 * 4);
        assert_eq!(cfg.attr.fragsize, 480 * 4);
        assert_eq!(cfg.attr.maxlength, 9_600 * 4);
        // playback-only fields left to the server.
        assert_eq!(
            (cfg.attr.tlength, cfg.attr.prebuf, cfg.attr.minreq),
            (u32::MAX, u32::MAX, u32::MAX)
        );
        // empty names mean "server default", not a literal empty name.
        assert_eq!(opt_name(&cfg.server), None);
        assert_eq!(opt_name(&cfg.device), None);
    }

    #[test]
    fn a_fragment_that_rounds_to_nothing_is_rejected() {
        // Under one frame of audio: there is no buffer to read.
        let src = PulseSrc::new().with_latency_time(0);
        assert_eq!(src.stream_config().err(), Some(G2gError::CapsMismatch));
    }

    #[test]
    fn defaults_match_the_declared_property_defaults() {
        let src = PulseSrc::new();
        for spec in PULSESRC_PROPS {
            let declared = spec
                .default
                .expect("every pulsesrc prop declares a default");
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
    fn latency_tracks_the_fragment_property() {
        let mut src = PulseSrc::new();
        assert_eq!(src.latency().min_ns, 10_000_000);
        src.set_property("latency-time", PropValue::Uint(25_000))
            .unwrap();
        assert_eq!(src.latency().min_ns, 25_000_000);
    }

    #[test]
    fn pad_template_is_pcm_source_only() {
        use g2g_core::PadDirection;
        let source = PulseSrc::pad_template(PadDirection::Source).expect("has source pad");
        let pcm = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
        };
        assert!(matches!(source.caps, g2g_core::PadCaps::Fixed(ref s) if s.accepts(&pcm)));
        assert!(PulseSrc::pad_template(PadDirection::Sink).is_none());
    }
}
