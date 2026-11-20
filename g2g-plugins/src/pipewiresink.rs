// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Linux PipeWire audio render sink. Plays interleaved PCM (`PcmU8` /
//! `PcmS16Le` / `PcmS24Le` / `PcmS32Le` / `PcmF32Le`) through the PipeWire
//! graph, the modern Linux media layer and the third Linux audio output
//! alongside [`AlsaSink`](crate::alsasink::AlsaSink) and
//! [`PulseSink`](crate::pulsesink::PulseSink).
//!
//! ## Channel order
//!
//! The connected format carries an explicit SPA position array built from our
//! interleave order, so a 5.1 / 7.1 stream is routed by speaker rather than
//! connected unpositioned.
//!
//! ## Threading
//!
//! PipeWire is a callback-driven main loop pinned to one thread, so the loop
//! runs on its own thread with a realtime `process` callback that drains a
//! shared PCM byte queue. A device worker, the same one the ALSA / Pulse /
//! WASAPI sinks use, fills that queue: it holds the first buffer until its
//! start time, stops filling while the pipeline is paused, and drops queued
//! audio on a flush.
//!
//! ## Pacing
//!
//! The byte queue holds 200 ms, like `AlsaSink`'s ring, and the device worker
//! waits for room, so the graph's pull rate back-pressures the pipeline. While
//! the pipeline is paused the callback plays silence and leaves the queue as it
//! is.

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use std::collections::VecDeque;
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use pipewire as pw;
use pw::spa;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ClockCandidate, ClockSync, ConfigureOutcome,
    DeviceClockOffset, DriftClock, ElementMetadata, G2gError, HardwareError, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::audiooutput::{
    DeviceCmd, DeviceStep, OutputDevice, PausableDevice, PcmGeometry, PlayEnd, PlayoutGate,
    PlayoutLink,
};
use crate::pwaudio::{format_pod_bytes, frame_bytes, pw_params, ticks_to_ns, FORMATS};
use g2g_core::pcm_caps_any_rate_and_channels;

const ELEMENT_NAME: &str = "PipeWireSink";

/// How often the clock-discipline trace prints, out of the once-per-quantum
/// stream of observations.
const CLOCK_LOG_INTERVAL_NS: u64 = 1_000_000_000;

/// Control message to the loop thread. Only `Terminate` for now (quit the loop).
enum Ctrl {
    Terminate,
}

// bounds the waits on the graph so a stalled stream can't hang the pipeline
const STALLED_STREAM_LIMIT: Duration = Duration::from_secs(2);

const QUEUED_AUDIO_NS: u64 = 200_000_000;

const ROOM_POLL_INTERVAL: Duration = Duration::from_millis(5);

fn pipewire_err() -> G2gError {
    G2gError::Hardware(HardwareError::PipeWire(-1))
}

#[derive(Debug, Default)]
struct StreamShared {
    queue: Mutex<VecDeque<u8>>,
    paused: AtomicBool,
    graph_delay_ns: AtomicU64,
}

impl StreamShared {
    fn queue(&self) -> Result<MutexGuard<'_, VecDeque<u8>>, G2gError> {
        self.queue.lock().map_err(|_| pipewire_err())
    }
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::pipewiresink::PipeWireSink;
///
/// let sink = PipeWireSink::new().with_target("alsa_output.pci-0000_00_1f.3.analog-stereo");
/// ```
pub struct PipeWireSink {
    /// Node to play to (`node.name` or object serial); empty = the default sink
    /// the session manager picks.
    output: OutputDevice,
    link: PlayoutLink,
    caps: Option<Caps>,
    bytes_queued: Arc<AtomicU64>,
    playout: crate::audioskew::PlayoutClock,
}

/// What the loop thread needs to open the playback stream.
#[derive(Clone)]
struct StreamCfg {
    format: spa::param::audio::AudioFormat,
    channels: u32,
    rate: u32,
    stride: usize,
    /// The process callback disciplines it from the stream time.
    clock: Arc<DriftClock>,
}

impl core::fmt::Debug for PipeWireSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PipeWireSink")
            .field("output", &self.output)
            .field("caps", &self.caps)
            .field("bytes_queued", &self.bytes_queued.load(Ordering::Relaxed))
            .finish()
    }
}

impl Default for PipeWireSink {
    fn default() -> Self {
        Self::new()
    }
}

impl PipeWireSink {
    pub fn new() -> Self {
        Self {
            output: OutputDevice::new("target-object", ""),
            link: PlayoutLink::new(pipewire_err()),
            caps: None,
            bytes_queued: Arc::new(AtomicU64::new(0)),
            playout: crate::audioskew::PlayoutClock::default(),
        }
    }

    /// Play to a specific node: its `node.name` or its object serial.
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.output = OutputDevice::new("target-object", target);
        self
    }

    /// The playout-disciplined clock this sink offers to election. Exposed for
    /// tests / introspection; its `now_ns()` tracks the stream's graph time
    /// once the realtime callback has observed the device.
    pub fn clock(&self) -> Arc<DriftClock> {
        self.playout.clock()
    }

    /// Total PCM bytes handed to the stream's queue.
    pub fn bytes_queued(&self) -> u64 {
        self.bytes_queued.load(Ordering::Relaxed)
    }
}

impl AsyncElement for PipeWireSink {
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
        pw_params(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(accepted_pcm_caps())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (format, channels, rate) = pw_params(absolute_caps)?;
        if self.link.is_running() && self.caps.as_ref() == Some(absolute_caps) {
            return Ok(ConfigureOutcome::Accepted);
        }

        self.playout.configure(absolute_caps)?;
        let stride = frame_bytes(format, channels);
        let cfg = StreamCfg {
            format,
            channels,
            rate,
            stride,
            // disciplined even when not offered, skew slaving reads it
            clock: self.playout.clock(),
        };
        let geometry = PcmGeometry::of(absolute_caps)?;
        let queue_capacity =
            crate::audioconvert::ns_to_samples(QUEUED_AUDIO_NS, rate) as usize * stride;
        let fallback = self.output.fallback(
            (stride * rate as usize) as u64,
            Arc::clone(&self.bytes_queued),
            1,
            |frame| {
                crate::dmabufmap::frame_bytes(frame, ELEMENT_NAME)
                    .ok()
                    .map(|bytes| bytes.as_slice().len())
            },
        );
        let bytes_queued = Arc::clone(&self.bytes_queued);
        self.link.open(
            "g2g-pipewiresink",
            absolute_caps,
            &mut self.output,
            fallback,
            move |target| PwStream::open(&cfg, target),
            move |stream, rx, gate| {
                let queue = QueuedStream {
                    shared: &stream.shared,
                    geometry,
                    capacity: queue_capacity,
                };
                play(queue, rx, gate, &bytes_queued)
            },
        )?;
        self.caps = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    /// Offer the playout-disciplined [`clock`](Self::clock) as an
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
            "PipeWire audio sink",
            "Sink/Audio",
            "Plays interleaved PCM through a PipeWire node",
            crate::AARON_BOXER,
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new(
                "target-object",
                PropKind::Str,
                "node name or object serial to play to (empty = default)",
            )
            .with_default(""),
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
                // A mid-stream format change can't be honoured on an open stream;
                // only a caps identical to the configured one passes.
                PipelinePacket::CapsChanged(c) => {
                    pw_params(&c)?;
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

impl PadTemplates for PipeWireSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(accepted_pcm_caps())])
    }
}

// =================================================================
// Device worker: fills the queue the realtime callback drains
// =================================================================

struct QueuedStream<'a> {
    shared: &'a StreamShared,
    geometry: PcmGeometry,
    capacity: usize,
}

impl QueuedStream<'_> {
    fn push(&self, pcm: &[u8]) -> Result<(), G2gError> {
        let deadline = Instant::now() + STALLED_STREAM_LIMIT;
        loop {
            let mut queue = self.shared.queue()?;
            if queue.len() + pcm.len() <= self.capacity.max(pcm.len()) {
                queue.extend(pcm);
                return Ok(());
            }
            drop(queue);
            if Instant::now() >= deadline {
                return Err(pipewire_err());
            }
            thread::sleep(ROOM_POLL_INTERVAL);
        }
    }

    fn wait_until_played(&self) -> Result<(), G2gError> {
        let deadline = Instant::now() + STALLED_STREAM_LIMIT;
        while !self.shared.queue()?.is_empty() && Instant::now() < deadline {
            thread::sleep(ROOM_POLL_INTERVAL);
        }
        Ok(())
    }
}

impl PausableDevice for QueuedStream<'_> {
    fn pause(&mut self) -> Result<(), G2gError> {
        self.shared.paused.store(true, Ordering::Release);
        Ok(())
    }

    fn resume(&mut self) -> Result<(), G2gError> {
        self.shared.paused.store(false, Ordering::Release);
        Ok(())
    }

    fn still_playing_after_pause_ns(&self) -> Result<u64, G2gError> {
        Ok(self.shared.graph_delay_ns.load(Ordering::Acquire))
    }

    fn delay_ns(&self) -> Result<u64, G2gError> {
        let queued_ns = self.geometry.bytes_to_ns(self.shared.queue()?.len());
        Ok(queued_ns + self.shared.graph_delay_ns.load(Ordering::Acquire))
    }
}

fn play(
    mut queue: QueuedStream<'_>,
    rx: &Receiver<DeviceCmd>,
    gate: &mut PlayoutGate,
    bytes_queued: &AtomicU64,
) -> PlayEnd {
    let geometry = queue.geometry;
    loop {
        let result = match gate.recv(rx) {
            Ok(None) => Ok(()),
            Ok(Some(DeviceStep::Samples(frame))) => {
                crate::dmabufmap::frame_bytes(&frame, ELEMENT_NAME).and_then(|payload| {
                    let pcm = payload.as_slice();
                    gate.play_in_slices(rx, pcm, geometry, &mut queue, |queue, slice| {
                        queue.push(slice)?;
                        bytes_queued.fetch_add(slice.len() as u64, Ordering::Relaxed);
                        Ok(())
                    })
                })
            }
            Ok(Some(DeviceStep::Flush)) => queue.shared.queue().map(|mut queue| queue.clear()),
            Ok(Some(DeviceStep::SwitchDevice(device))) => return PlayEnd::SwitchDevice(device),
            Ok(Some(DeviceStep::Shutdown)) => {
                return match queue.wait_until_played() {
                    Ok(()) => PlayEnd::Shutdown,
                    Err(error) => PlayEnd::Failed(error),
                };
            }
            // the element aborted: stop without playing out
            Err(_) => return PlayEnd::Shutdown,
        };
        if let Err(error) = result {
            return PlayEnd::Failed(error);
        }
    }
}

// =================================================================
// Loop thread: the PipeWire main loop
// =================================================================

struct PwStream {
    ctrl_tx: pw::channel::Sender<Ctrl>,
    loop_thread: Option<JoinHandle<()>>,
    shared: Arc<StreamShared>,
}

impl PwStream {
    fn open(cfg: &StreamCfg, target: &str) -> Result<Self, G2gError> {
        let (ctrl_tx, ctrl_rx) = pw::channel::channel::<Ctrl>();
        let (ready_tx, ready_rx) = sync_channel::<Result<(), i32>>(1);
        let shared = Arc::new(StreamShared::default());
        let loop_shared = Arc::clone(&shared);
        let cfg = cfg.clone();
        let target = String::from(target);
        let join = thread::Builder::new()
            .name(String::from("g2g-pipewireloop"))
            .spawn(move || {
                if let Err(code) = build_and_run(&cfg, &target, loop_shared, ctrl_rx, &ready_tx) {
                    // a send after the loop already reported ready lands on a closed channel
                    let _ = ready_tx.send(Err(code));
                }
            })
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

        let code = match ready_rx.recv_timeout(STALLED_STREAM_LIMIT) {
            Ok(Ok(())) => {
                return Ok(Self {
                    ctrl_tx,
                    loop_thread: Some(join),
                    shared,
                })
            }
            Ok(Err(code)) => code,
            Err(_) => -1,
        };
        let _ = ctrl_tx.send(Ctrl::Terminate);
        let _ = join.join();
        Err(G2gError::Hardware(HardwareError::PipeWire(code)))
    }
}

impl Drop for PwStream {
    fn drop(&mut self) {
        let _ = self.ctrl_tx.send(Ctrl::Terminate);
        if let Some(join) = self.loop_thread.take() {
            let _ = join.join();
        }
    }
}

fn build_and_run(
    cfg: &StreamCfg,
    target: &str,
    shared: Arc<StreamShared>,
    ctrl_rx: pw::channel::Receiver<Ctrl>,
    ready: &std::sync::mpsc::SyncSender<Result<(), i32>>,
) -> Result<(), i32> {
    let stride = cfg.stride;
    pw::init();
    let mainloop = pw::main_loop::MainLoopRc::new(None).map_err(|_| -1)?;
    let context = pw::context::ContextBox::new(mainloop.loop_(), None).map_err(|_| -1)?;
    let core = context.connect(None).map_err(|_| -1)?;

    let mut props = pw::properties::properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_ROLE => "Music",
        *pw::keys::MEDIA_CATEGORY => "Playback",
    };
    if !target.is_empty() {
        // spelled out because pipewire-rs gates its TARGET_OBJECT constant
        // behind a crate feature this build does not enable
        props.insert("target.object", target);
    }
    let stream = pw::stream::StreamBox::new(&core, "g2g-pipewiresink", props).map_err(|_| -1)?;

    let drift = cfg.clock.clone();
    let mut last_clock_log_ns = 0u64;
    // pipewire ticks restart from zero with each stream
    let mut clock_offset = DeviceClockOffset::default();
    let _listener = stream
        .add_local_listener_with_user_data(())
        .process(move |stream, ()| {
            if let Some(time) = crate::pwaudio::stream_time(stream) {
                let delay_ticks = u64::try_from(time.delay).unwrap_or(0);
                if let Some(delay_ns) = ticks_to_ns(delay_ticks, time.rate) {
                    shared.graph_delay_ns.store(delay_ns, Ordering::Release);
                }
                discipline_clock(
                    stream,
                    &time,
                    &drift,
                    &mut clock_offset,
                    &mut last_clock_log_ns,
                );
            }
            let Some(mut buffer) = stream.dequeue_buffer() else {
                return;
            };
            // one graph cycle of frames, 0 when PipeWire suggests no amount
            let requested_bytes = match buffer.requested() {
                0 => usize::MAX,
                frames => usize::try_from(frames).map_or(usize::MAX, |f| f.saturating_mul(stride)),
            };
            let datas = buffer.datas_mut();
            if datas.is_empty() {
                return;
            }
            let data = &mut datas[0];
            let size = if let Some(slice) = data.data() {
                // Fill whole-frame-aligned capacity from the queue; pad the
                // remainder with silence so an underrun or a pause is quiet, not garbage.
                let cap = (slice.len().min(requested_bytes) / stride) * stride;
                let mut filled = 0;
                if !shared.paused.load(Ordering::Acquire) {
                    if let Ok(mut queue) = shared.queue.lock() {
                        filled = queue.len().min(cap);
                        for (out, byte) in slice[..filled].iter_mut().zip(queue.drain(..filled)) {
                            *out = byte;
                        }
                    }
                }
                slice[filled..cap].fill(0);
                cap
            } else {
                0
            };
            let chunk = data.chunk_mut();
            *chunk.offset_mut() = 0;
            *chunk.stride_mut() = stride as _;
            *chunk.size_mut() = size as _;
        })
        .register()
        .map_err(|_| -1)?;

    let values = format_pod_bytes(cfg.format, cfg.channels, cfg.rate);
    let mut params = [spa::pod::Pod::from_bytes(&values).ok_or(-1)?];
    stream
        .connect(
            spa::utils::Direction::Output,
            None,
            pw::stream::StreamFlags::AUTOCONNECT
                | pw::stream::StreamFlags::MAP_BUFFERS
                | pw::stream::StreamFlags::RT_PROCESS,
            &mut params,
        )
        .map_err(|_| -1)?;

    // Quit the loop when the element sends `Terminate` on teardown.
    let weak = mainloop.downgrade();
    let _recv = ctrl_rx.attach(mainloop.loop_(), move |_ctrl| {
        if let Some(ml) = weak.upgrade() {
            ml.quit();
        }
    });

    // Setup succeeded; unblock the device worker, then run the loop.
    let _ = ready.send(Ok(()));
    mainloop.run();
    Ok(())
}

/// Feed the drift clock one `(monotonic_now, graph_time_ns)` observation from
/// the stream's own time report. `pw_time.ticks` is the graph driver's
/// monotonic sample counter, which advances at the device's real rate, so it
/// is the master timeline the pipeline slaves to. It is independent of our
/// byte queue, so pauses and flushes never skew it; the constant graph
/// delay to the speaker lands in the affine offset the fit absorbs. The local
/// time is the report's own sample instant (`pw_time.now`), so a callback that
/// runs late does not shift the pair.
///
/// The callback runs once per graph quantum (~47 Hz at 1024 frames / 48 kHz),
/// so the trace is throttled to [`CLOCK_LOG_INTERVAL_NS`]; `last_log_ns` carries
/// the caller's stamp of the last line.
fn discipline_clock(
    stream: &pw::stream::Stream,
    time: &pw::sys::pw_time,
    clock: &DriftClock,
    clock_offset: &mut DeviceClockOffset,
    last_log_ns: &mut u64,
) {
    let Some(local_ns) = crate::pwaudio::reference_ns_at_sample(stream, clock, time) else {
        return;
    };
    // A not-yet-running stream (ticks still zero) or a report with no rate
    // yields no usable position; skip rather than feed a bogus sample.
    if time.ticks == 0 {
        return;
    }
    let Some(graph_ns) = ticks_to_ns(time.ticks, time.rate) else {
        return;
    };
    let master_ns = clock_offset.master_ns(clock, local_ns, graph_ns);
    let outcome = clock.observe(local_ns, master_ns);
    if local_ns.saturating_sub(*last_log_ns) < CLOCK_LOG_INTERVAL_NS {
        return;
    }
    *last_log_ns = local_ns;
    g2g_core::g2g_log!(
        g2g_core::log::Target::category("PipeWireSink"),
        "clock local={}ms master={}ms delay={} slope={:.6} observations={} {:?}",
        local_ns / 1_000_000,
        master_ns / 1_000_000,
        time.delay,
        clock.slope(),
        clock.observations(),
        outcome
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::AudioFormat;

    #[test]
    fn intercept_accepts_pcm_rejects_compressed() {
        let sink = PipeWireSink::new();
        let pcm = Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        assert_eq!(sink.intercept_caps(&pcm), Ok(pcm));
        let aac = Caps::Audio {
            format: AudioFormat::Aac,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        assert_eq!(sink.intercept_caps(&aac), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn properties_round_trip_through_the_launch_path() {
        let mut sink = PipeWireSink::new();
        assert_eq!(
            sink.get_property("target-object"),
            Some(PropValue::Str(String::new()))
        );
        sink.set_property("target-object", PropValue::Str(String::from("spk0")))
            .expect("known prop");
        assert_eq!(sink.output.name(), "spk0");
        assert_eq!(
            sink.get_property("target-object"),
            Some(PropValue::Str(String::from("spk0")))
        );
        assert_eq!(
            sink.set_property("target-object", PropValue::Uint(1)),
            Err(PropError::Type)
        );
        assert_eq!(
            sink.set_property("nope", PropValue::Uint(1)),
            Err(PropError::Unknown)
        );
        // the builder is the same knob
        assert_eq!(
            PipeWireSink::new().with_target("spk1").output.name(),
            "spk1"
        );
        // every declared property is handled by both halves
        for spec in sink.properties() {
            assert!(
                sink.get_property(spec.name).is_some(),
                "unhandled property {}",
                spec.name
            );
        }
    }

    #[test]
    fn pad_template_is_pcm_sink_only() {
        use g2g_core::{PadDirection, PadTemplates};
        let sink = PipeWireSink::pad_template(PadDirection::Sink).expect("has sink pad");
        let pcm = Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        };
        assert!(matches!(sink.caps, g2g_core::PadCaps::Fixed(ref s) if s.accepts(&pcm)));
        assert!(PipeWireSink::pad_template(PadDirection::Source).is_none());
    }
}
