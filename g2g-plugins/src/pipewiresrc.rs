//! Linux PipeWire audio capture source. The capture sibling of
//! [`PipeWireSink`](crate::pipewiresink::PipeWireSink): connects an input stream
//! to the PipeWire graph and streams interleaved PCM (`PcmS16Le` / `PcmF32Le`)
//! `DataFrame`s downstream. The modern Linux microphone path (PipeWire replaces
//! v4l2 + PulseAudio + the screen-capture DBus dance); video / screen capture is
//! a follow-up on the same element.
//!
//! PipeWire is a callback-driven main loop pinned to one thread, so (like
//! `V4l2Src`) the loop runs on a dedicated worker thread that feeds the async
//! `run` loop over a channel. We request a fixed PCM format; the PipeWire
//! adapter converts the device to it, so the produced caps are deterministic
//! (no async `param_changed` round-trip needed for v1).

use core::future::Future;
use core::pin::Pin;

use std::sync::Arc;

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AudioFormat, Caps, CapsConstraint, CapsSet, ClockCandidate, ClockPriority, ConfigureOutcome,
    DriftClock, DriftObservation, ElementMetadata, FrameTiming, G2gError, HardwareError,
    LatencyReport, MemoryDomain, MonotonicClock, OutputSink, PadTemplate, PadTemplates,
    PipelineClock, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use pipewire as pw;
use pw::spa;

use crate::audioconvert::{audio_format_from_str, audio_format_to_str};
use crate::pwaudio::{format_pod_bytes, frame_bytes, pw_params};

const DEFAULT_RATE: u32 = 48_000;
const DEFAULT_CHANNELS: u8 = 2;

/// How often the clock-discipline trace prints, out of the once-per-quantum
/// stream of observations.
const CLOCK_LOG_INTERVAL_NS: u64 = 1_000_000_000;

/// Control message to the loop thread (quit on teardown).
enum Ctrl {
    Terminate,
}

/// Loop-thread to element messages.
enum FromWorker {
    /// One captured PCM buffer.
    Buffer(Vec<u8>),
    /// The stream went to the error state. A `target-object` the daemon cannot
    /// resolve lands here rather than capturing some other node.
    Failed(G2gError),
}

/// # Example
///
/// ```no_run
/// use g2g_core::AudioFormat;
/// use g2g_plugins::pipewiresrc::PipeWireSrc;
///
/// let element = PipeWireSrc::new()
///     .with_format(AudioFormat::PcmF32Le)
///     .with_rate(48_000);
/// ```
#[derive(Debug)]
pub struct PipeWireSrc {
    /// Node to capture from (`node.name` or object serial); empty = the default
    /// audio source the session manager picks.
    target: String,
    format: AudioFormat,
    channels: u8,
    rate: u32,
    /// `u64::MAX` = run until error or downstream shutdown; else stop after this
    /// many frames (PipeWire buffers) and emit EOS. The bounded-capture / test
    /// path.
    frame_limit: u64,
    configured: bool,
    /// Capture-disciplined clock. The realtime callback feeds it
    /// `(monotonic_now, graph_ticks_ns)` observations from
    /// `pw_stream_get_time_n`, so its `now_ns()` tracks the graph driver's real
    /// rate rather than wall time.
    clock: Arc<DriftClock>,
    /// Whether to offer [`clock`](Self::clock) to the pipeline's clock election
    /// (the `provide-clock` property, default on).
    provide_clock: bool,
}

/// What the loop thread needs to open the capture stream.
struct StreamCfg {
    format: spa::param::audio::AudioFormat,
    channels: u32,
    rate: u32,
    stride: usize,
    target: String,
    /// Present when the element offers its clock to election; the process
    /// callback then disciplines it from the stream time.
    clock: Option<Arc<DriftClock>>,
}

impl Default for PipeWireSrc {
    fn default() -> Self {
        Self::new()
    }
}

impl PipeWireSrc {
    /// Capture S16LE stereo at 48 kHz from the default source by default.
    pub fn new() -> Self {
        Self {
            target: String::new(),
            format: AudioFormat::PcmS16Le,
            channels: DEFAULT_CHANNELS,
            rate: DEFAULT_RATE,
            frame_limit: u64::MAX,
            configured: false,
            clock: Arc::new(DriftClock::new(Arc::new(MonotonicClock))),
            provide_clock: true,
        }
    }

    /// The capture-disciplined clock this source offers to election. Exposed for
    /// tests / introspection; its `now_ns()` tracks the graph's real rate once
    /// the realtime callback has observed the stream.
    pub fn clock(&self) -> Arc<DriftClock> {
        Arc::clone(&self.clock)
    }

    /// Capture from a specific node: its `node.name` or its object serial.
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = target.into();
        self
    }

    /// Request a PCM sample format (`PcmS16Le` or `PcmF32Le`).
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

    /// Stop after `n` captured buffers and emit EOS (0 emits EOS without opening
    /// the stream). Without this the source runs until an error or until
    /// downstream drops.
    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.frame_limit = n;
        self
    }

    fn caps(&self) -> Result<Caps, G2gError> {
        let caps = Caps::Audio {
            format: self.format,
            channels: self.channels,
            sample_rate: self.rate,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        // Reject a non-PCM request up front (the SPA mapping is PCM-only).
        pw_params(&caps)?;
        Ok(caps)
    }
}

impl SourceLoop for PipeWireSrc {
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

    /// Produces the fixed PCM caps we ask the graph to convert to, so a chain
    /// built on the mic takes the native arc-consistency path (mirrors `V4l2Src`).
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

    /// Live source: one buffer period is device-driven, so report a small live
    /// latency hint rather than zero.
    fn latency(&self) -> LatencyReport {
        LatencyReport::live(0, None)
    }

    /// Offer the capture-disciplined [`clock`](Self::clock) as a plain
    /// [`Provider`](ClockPriority::Provider), one rank below the audio sinks'
    /// [`AudioProvider`](ClockPriority::AudioProvider): in a duplex pipeline the
    /// playout rate is the one nobody can adjust, so the sink stays master and
    /// capture slaves to it. In a capture-only pipeline nothing else offers a
    /// clock, so this one wins and the pipeline runs on the graph's real rate
    /// instead of the monotonic fallback.
    fn provide_clock(&self) -> Option<ClockCandidate> {
        if !self.provide_clock {
            return None;
        }
        let clock: Arc<dyn PipelineClock + Send + Sync> = self.clock.clone();
        Some(ClockCandidate::new(ClockPriority::Provider, clock))
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "PipeWire audio source",
            "Source/Audio",
            "Captures interleaved PCM from a PipeWire node",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        PIPEWIRESRC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "target-object" => self.target = value.as_str().ok_or(PropError::Type)?.to_string(),
            "format" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.format = audio_format_from_str(s).ok_or(PropError::Value)?;
            }
            "samplerate" => self.rate = value.as_uint().ok_or(PropError::Type)? as u32,
            "channels" => self.channels = value.as_uint().ok_or(PropError::Type)? as u8,
            "num-buffers" => crate::numbuffers::set_num_buffers(&mut self.frame_limit, &value)?,
            "provide-clock" => self.provide_clock = value.as_bool().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "target-object" => Some(PropValue::Str(self.target.clone())),
            "format" => Some(PropValue::Str(audio_format_to_str(self.format).into())),
            "samplerate" => Some(PropValue::Uint(u64::from(self.rate))),
            "channels" => Some(PropValue::Uint(u64::from(self.channels))),
            "num-buffers" => Some(crate::numbuffers::get_num_buffers(self.frame_limit)),
            "provide-clock" => Some(PropValue::Bool(self.provide_clock)),
            _ => None,
        }
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            if crate::numbuffers::finished_at_zero_limit(self.frame_limit, out).await? {
                return Ok(0);
            }
            let (spa_format, channels, rate) =
                pw_params(&self.caps()?).map_err(|_| G2gError::NotConfigured)?;
            let stride = frame_bytes(spa_format, channels);
            let cfg = StreamCfg {
                format: spa_format,
                channels,
                rate,
                stride,
                target: self.target.clone(),
                // Only discipline the clock when we actually offer it; otherwise
                // the per-callback time probe is wasted work no one reads.
                clock: self.provide_clock.then(|| Arc::clone(&self.clock)),
            };
            let limit = self.frame_limit;

            // Captured PCM buffers cross from the loop thread to here.
            let (audio_tx, mut audio_rx) = tokio::sync::mpsc::unbounded_channel::<FromWorker>();
            // Control + a setup-result handshake (surface a connect failure).
            let (ctrl_tx, ctrl_rx) = pw::channel::channel::<Ctrl>();
            let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel::<Result<(), i32>>(1);

            let handle = std::thread::Builder::new()
                .name(String::from("g2g-pipewiresrc"))
                .spawn(move || {
                    worker_main(cfg, audio_tx, ctrl_rx, ready_tx);
                })
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

            // Block briefly for the stream to connect (sync ioctl-equivalent).
            match ready_rx.recv_timeout(std::time::Duration::from_secs(5)) {
                Ok(Ok(())) => {}
                Ok(Err(code)) => {
                    let _ = ctrl_tx.send(Ctrl::Terminate);
                    let _ = handle.join();
                    return Err(G2gError::Hardware(HardwareError::PipeWire(code)));
                }
                Err(_) => {
                    let _ = ctrl_tx.send(Ctrl::Terminate);
                    let _ = handle.join();
                    return Err(G2gError::Hardware(HardwareError::PipeWire(-1)));
                }
            }

            let frame_dur = if rate > 0 {
                1_000_000_000u64 / rate as u64
            } else {
                0
            };
            let mut seq = 0u64;
            let mut frames_total = 0u64; // sample frames, for PTS
            let mut downstream_open = true;
            let mut failure = None;

            while seq < limit {
                let Some(msg) = audio_rx.recv().await else {
                    break; // worker ended
                };
                let bytes = match msg {
                    FromWorker::Buffer(bytes) => bytes,
                    FromWorker::Failed(e) => {
                        failure = Some(e);
                        break;
                    }
                };
                if bytes.len() < stride {
                    continue;
                }
                let n_frames = (bytes.len() / stride) as u64;
                let arrival_ns = g2g_core::metrics::monotonic_ns();
                let pts = if rate > 0 {
                    frames_total * 1_000_000_000 / rate as u64
                } else {
                    0
                };
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                    timing: FrameTiming {
                        pts_ns: pts,
                        dts_ns: pts,
                        duration_ns: n_frames * frame_dur,
                        capture_ns: pts,
                        arrival_ns,
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

            // Stop the loop and reap the worker.
            let _ = ctrl_tx.send(Ctrl::Terminate);
            let _ = handle.join();

            if let Some(e) = failure {
                return Err(e);
            }
            if downstream_open {
                out.push(PipelinePacket::Eos).await?;
            }
            Ok(seq)
        })
    }
}

impl PadTemplates for PipeWireSrc {
    /// Produces PCM; a constructed instance fixes the format / rate / channels.
    fn pad_templates() -> Vec<PadTemplate> {
        let pcm = |format| Caps::Audio {
            format,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        Vec::from([PadTemplate::source(CapsSet::from_alternatives(Vec::from(
            [pcm(AudioFormat::PcmS16Le), pcm(AudioFormat::PcmF32Le)],
        )))])
    }
}

/// `PipeWireSrc`'s settable properties. `target-object` matches gst
/// `pipewiresrc`; rate / channels / format are caps on the gst side, so they take
/// the same names as the ALSA / PulseAudio sources, and `num-buffers` matches gst
/// `basesrc`.
static PIPEWIRESRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "target-object",
        PropKind::Str,
        "node name or object serial to capture from (empty = default)",
    )
    .with_default(""),
    PropertySpec::new(
        "format",
        PropKind::Str,
        "capture sample format: S16LE | F32LE | S24LE | S32LE | U8",
    )
    .with_default("S16LE"),
    PropertySpec::new("samplerate", PropKind::Uint, "samples per second").with_default("48000"),
    PropertySpec::new("channels", PropKind::Uint, "channel count").with_default("2"),
    PropertySpec::new(
        "num-buffers",
        PropKind::Int,
        "buffers to capture then EOS (-1 = forever)",
    )
    .with_default("-1")
    .with_range("-1", "9223372036854775807"),
    PropertySpec::new(
        "provide-clock",
        PropKind::Bool,
        "Provide a capture-disciplined clock for the pipeline's clock election",
    )
    .with_default("true"),
];

// =================================================================
// Worker thread: the PipeWire capture main loop
// =================================================================

fn worker_main(
    cfg: StreamCfg,
    audio_tx: tokio::sync::mpsc::UnboundedSender<FromWorker>,
    ctrl_rx: pw::channel::Receiver<Ctrl>,
    ready: std::sync::mpsc::SyncSender<Result<(), i32>>,
) {
    if let Err(code) = build_and_run(&cfg, audio_tx, ctrl_rx, &ready) {
        let _ = ready.send(Err(code));
    }
}

fn build_and_run(
    cfg: &StreamCfg,
    audio_tx: tokio::sync::mpsc::UnboundedSender<FromWorker>,
    ctrl_rx: pw::channel::Receiver<Ctrl>,
    ready: &std::sync::mpsc::SyncSender<Result<(), i32>>,
) -> Result<(), i32> {
    let stride = cfg.stride;
    pw::init();
    let mainloop = pw::main_loop::MainLoop::new(None).map_err(|_| -1)?;
    let context = pw::context::Context::new(&mainloop).map_err(|_| -1)?;
    let core = context.connect(None).map_err(|_| -1)?;

    let mut props = pw::properties::properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE => "Music",
    };
    if !cfg.target.is_empty() {
        // spelled out because pipewire-rs gates its TARGET_OBJECT constant
        // behind a crate feature this build does not enable
        props.insert("target.object", cfg.target.as_str());
    }
    let stream = pw::stream::Stream::new(&core, "g2g-pipewiresrc", props).map_err(|_| -1)?;

    let error_tx = audio_tx.clone();
    let drift = cfg.clock.clone();
    let mut last_clock_log_ns = 0u64;
    let _listener = stream
        .add_local_listener_with_user_data(())
        // A stream the daemon cannot route (a `target-object` it cannot resolve)
        // goes to the error state after connect succeeded: surface it so the
        // capture fails instead of waiting for buffers that never come.
        .state_changed(move |_, (), _old, new| {
            if let pw::stream::StreamState::Error(_) = new {
                let _ = error_tx.send(FromWorker::Failed(G2gError::Hardware(
                    HardwareError::PipeWire(-1),
                )));
            }
        })
        .process(move |stream, ()| {
            if let Some(clock) = drift.as_deref() {
                discipline_clock(stream, clock, &mut last_clock_log_ns);
            }
            let Some(mut buffer) = stream.dequeue_buffer() else {
                return;
            };
            let datas = buffer.datas_mut();
            if datas.is_empty() {
                return;
            }
            let data = &mut datas[0];
            let offset = data.chunk().offset() as usize;
            let size = data.chunk().size() as usize;
            if size < stride {
                return;
            }
            if let Some(slice) = data.data() {
                let end = (offset + size).min(slice.len());
                if end > offset {
                    // Copy the valid region out and hand it to the async side.
                    let _ = audio_tx.send(FromWorker::Buffer(slice[offset..end].to_vec()));
                }
            }
        })
        .register()
        .map_err(|_| -1)?;

    let values = format_pod_bytes(cfg.format, cfg.channels, cfg.rate);
    let mut params = [spa::pod::Pod::from_bytes(&values).ok_or(-1)?];
    stream
        .connect(
            spa::utils::Direction::Input,
            None,
            pw::stream::StreamFlags::AUTOCONNECT
                | pw::stream::StreamFlags::MAP_BUFFERS
                | pw::stream::StreamFlags::RT_PROCESS,
            &mut params,
        )
        .map_err(|_| -1)?;

    let weak = mainloop.downgrade();
    let _recv = ctrl_rx.attach(mainloop.loop_(), move |_ctrl| {
        if let Some(ml) = weak.upgrade() {
            ml.quit();
        }
    });

    let _ = ready.send(Ok(()));
    mainloop.run();
    Ok(())
}

/// Feed the drift clock one `(monotonic_now, graph_time_ns)` observation from
/// the stream's own time report. `pw_time.ticks` is the graph driver's monotonic
/// sample counter, which advances at the capture device's real rate, so it is
/// the timeline this source's frame counter actually runs on. It is independent
/// of how promptly we drain the buffers, so a dropped or late callback never
/// skews it; the constant graph delay from the microphone lands in the affine
/// offset the fit absorbs. The local time is sampled next to the probe so the
/// pair lines up.
///
/// The callback runs once per graph quantum (~47 Hz at 1024 frames / 48 kHz),
/// so the trace is throttled to [`CLOCK_LOG_INTERVAL_NS`]; `last_log_ns` carries
/// the caller's stamp of the last line.
fn discipline_clock(stream: &pw::stream::StreamRef, clock: &DriftClock, last_log_ns: &mut u64) {
    let local_ns = clock.reference_now();
    let Some(time) = crate::pwaudio::stream_time(stream) else {
        return;
    };
    let Some(outcome) =
        observe_stream_time(clock, local_ns, time.ticks, time.rate.num, time.rate.denom)
    else {
        return;
    };
    if local_ns.saturating_sub(*last_log_ns) < CLOCK_LOG_INTERVAL_NS {
        return;
    }
    *last_log_ns = local_ns;
    g2g_core::g2g_log!(
        g2g_core::log::Target::category("PipeWireSrc"),
        "clock local={}ms ticks={} delay={} slope={:.6} observations={} {:?}",
        local_ns / 1_000_000,
        time.ticks,
        time.delay,
        clock.slope(),
        clock.observations(),
        outcome
    );
}

/// Fold one stream-time report into the drift fit: `ticks` counted at the
/// `num / denom` rate is the master timeline, `local_ns` the reference instant
/// it was read at. `None` when the report carries no usable position (a stream
/// that has not started still reads zero ticks, and a report with no rate cannot
/// be converted), so the caller skips instead of feeding a bogus sample.
fn observe_stream_time(
    clock: &DriftClock,
    local_ns: u64,
    ticks: u64,
    rate_num: u32,
    rate_denom: u32,
) -> Option<DriftObservation> {
    if ticks == 0 || rate_denom == 0 {
        return None;
    }
    let master_ns =
        (u128::from(ticks) * u128::from(rate_num) * 1_000_000_000 / u128::from(rate_denom)) as u64;
    Some(clock.observe(local_ns, master_ns))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builders_set_requested_config() {
        let src = PipeWireSrc::new()
            .with_target("g2g-node")
            .with_format(AudioFormat::PcmF32Le)
            .with_rate(44_100)
            .with_channels(1)
            .with_frame_limit(5);
        assert_eq!(src.target, "g2g-node");
        assert_eq!(src.format, AudioFormat::PcmF32Le);
        assert_eq!((src.channels, src.rate), (1, 44_100));
        assert_eq!(src.frame_limit, 5);
        // The builder spells the limit the way the property does: 0 is a count
        // of zero, not a second way to say "forever".
        assert_eq!(PipeWireSrc::new().with_frame_limit(0).frame_limit, 0);
        assert_eq!(PipeWireSrc::new().frame_limit, u64::MAX);
    }

    #[test]
    fn properties_round_trip_through_the_launch_path() {
        let mut src = PipeWireSrc::new();
        for (name, value) in [
            ("target-object", PropValue::Str("mic0".to_string())),
            ("format", PropValue::Str("F32LE".to_string())),
            ("samplerate", PropValue::Uint(44_100)),
            ("channels", PropValue::Uint(1)),
            ("num-buffers", PropValue::Int(20)),
        ] {
            src.set_property(name, value.clone()).expect("known prop");
            assert_eq!(src.get_property(name), Some(value));
        }
        assert_eq!(src.target, "mic0");
        assert_eq!(src.format, AudioFormat::PcmF32Le);
        assert_eq!((src.channels, src.rate), (1, 44_100));
        assert_eq!(src.frame_limit, 20);
        // -1 is no limit, in both directions
        src.set_property("num-buffers", PropValue::Int(-1))
            .expect("known prop");
        assert_eq!(src.frame_limit, u64::MAX);
        assert_eq!(src.get_property("num-buffers"), Some(PropValue::Int(-1)));
        // a sample format the element cannot open a stream with is an error
        assert_eq!(
            src.set_property("format", PropValue::Str("AAC".to_string())),
            Err(PropError::Value)
        );
        assert_eq!(
            src.set_property("nope", PropValue::Uint(1)),
            Err(PropError::Unknown)
        );
        // every declared property is handled by both halves
        for spec in src.properties() {
            assert!(
                src.get_property(spec.name).is_some(),
                "unhandled property {}",
                spec.name
            );
        }
    }

    /// The properties reach the pod the stream connects with: rate, channels and
    /// sample format all come back out of the serialized SPA audio format.
    #[test]
    fn properties_reach_the_connect_pod() {
        use spa::param::audio::{AudioFormat as SpaAudioFormat, AudioInfoRaw};

        let mut src = PipeWireSrc::new();
        src.set_property("format", PropValue::Str("S32LE".to_string()))
            .unwrap();
        src.set_property("samplerate", PropValue::Uint(96_000))
            .unwrap();
        src.set_property("channels", PropValue::Uint(6)).unwrap();

        let (format, channels, rate) = pw_params(&src.caps().expect("caps")).expect("pcm maps");
        let bytes = format_pod_bytes(format, channels, rate);
        let pod = spa::pod::Pod::from_bytes(&bytes).expect("pod bytes parse");
        let mut info = AudioInfoRaw::new();
        info.parse(pod).expect("spa parses our audio format");
        assert_eq!(info.format(), SpaAudioFormat::S32LE);
        assert_eq!(info.rate(), 96_000);
        assert_eq!(info.channels(), 6);
    }

    #[test]
    fn caps_reflect_request_and_reject_compressed() {
        let src = PipeWireSrc::new().with_rate(16_000).with_channels(1);
        assert_eq!(
            src.caps(),
            Ok(Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: 1,
                sample_rate: 16_000,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
            })
        );
        let bad = PipeWireSrc::new().with_format(AudioFormat::Opus);
        assert_eq!(bad.caps(), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn provides_a_capture_clock_below_the_sinks() {
        let src = PipeWireSrc::new();
        let cand = src.provide_clock().expect("pipewiresrc offers a clock");
        // Below AudioProvider so a duplex pipeline keeps the sink's playout
        // clock as master, above SystemFallback so capture-only elects this.
        assert_eq!(cand.priority, ClockPriority::Provider);
        assert!(ClockPriority::Provider < ClockPriority::AudioProvider);
        assert!(ClockPriority::Provider > ClockPriority::SystemFallback);

        let mut src = src;
        src.set_property("provide-clock", PropValue::Bool(false))
            .unwrap();
        assert!(
            src.provide_clock().is_none(),
            "disabled src offers no clock"
        );
        assert_eq!(
            src.set_property("provide-clock", PropValue::Str("yes".into())),
            Err(PropError::Type)
        );
    }

    /// The discipline fold, driven with synthetic `pw_time` reports: a graph
    /// driver running 100 ppm fast should show up as a slope of ~1.0001, which
    /// is the whole point of the clock (the pipeline paces to the capture
    /// device, not wall time).
    #[test]
    fn stream_time_reports_fit_the_real_capture_rate() {
        const RATE_DENOM: u32 = 48_000;
        const QUANTUM_TICKS: u64 = 1024;
        const GRAPH_PPM_FAST: f64 = 1.0001;
        /// One quantum of reference time: the nominal quantum period, ~21.3 ms.
        /// The graph counts more than `QUANTUM_TICKS` in it, which is the drift.
        const QUANTUM_NS: u64 = QUANTUM_TICKS * 1_000_000_000 / RATE_DENOM as u64;

        let clock = DriftClock::new(Arc::new(MonotonicClock));
        let base_local_ns = 5_000_000_000u64;
        for quantum in 1..=DriftClock::DEFAULT_WINDOW as u64 {
            let local_ns = base_local_ns + quantum * QUANTUM_NS;
            let ticks = ((quantum * QUANTUM_TICKS) as f64 * GRAPH_PPM_FAST).round() as u64;
            assert_eq!(
                observe_stream_time(&clock, local_ns, ticks, 1, RATE_DENOM),
                Some(DriftObservation::Folded),
                "clean ramp sample {quantum} should fold"
            );
        }
        let slope = clock.slope();
        assert!(
            (1.00005..1.00015).contains(&slope),
            "capture-rate estimate {slope} did not converge on 1.0001"
        );
    }

    #[test]
    fn a_report_with_no_position_feeds_no_sample() {
        let clock = DriftClock::new(Arc::new(MonotonicClock));
        // A stream that has not started reads zero ticks, and a report with no
        // rate cannot be converted to a timeline.
        assert_eq!(observe_stream_time(&clock, 1_000, 0, 1, 48_000), None);
        assert_eq!(observe_stream_time(&clock, 1_000, 1_024, 1, 0), None);
        assert_eq!(clock.observations(), 0);
    }

    #[test]
    fn pad_template_is_pcm_source_only() {
        use g2g_core::{PadDirection, PadTemplates};
        assert!(PipeWireSrc::pad_template(PadDirection::Source).is_some());
        assert!(PipeWireSrc::pad_template(PadDirection::Sink).is_none());
    }
}
