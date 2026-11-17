// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::string::String;

use g2g_core::bus::{BusHandle, BusMessage};
use g2g_core::{G2gError, PropError, PropKind, PropValue, PropertySpec};

#[cfg(any(
    all(
        target_os = "linux",
        any(feature = "alsa-sink", feature = "pulse-sink")
    ),
    all(target_os = "windows", feature = "wasapi-sink")
))]
pub(crate) use device_worker::{
    run_device_worker, DeviceCmd, DeviceStep, PcmGeometry, PendingFlushes, PlayEnd, PlayoutGate,
    PlayoutStart,
};

pub(crate) const CONTINUE_ON_ERROR_PROPERTY: PropertySpec = PropertySpec::new(
    "continue-on-error",
    PropKind::Bool,
    "Post a warning and keep consuming buffers in real time when the device fails to open, fails a write or is removed",
)
.with_default("false");

// The device-name property of one sink, switched in order with the samples while it plays.
#[derive(Debug)]
pub(crate) struct OutputDevice {
    property: &'static str,
    name: String,
    switch_pending: bool,
    continue_on_error: bool,
    bus: Option<BusHandle>,
}

impl OutputDevice {
    pub(crate) fn new(property: &'static str, name: impl Into<String>) -> Self {
        Self {
            property,
            name: name.into(),
            switch_pending: false,
            continue_on_error: false,
            bus: None,
        }
    }

    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn set_bus(&mut self, bus: BusHandle) {
        self.bus = Some(bus);
    }

    pub(crate) fn set_property(
        &mut self,
        name: &str,
        value: &PropValue,
        playing: bool,
    ) -> Option<Result<(), PropError>> {
        if name == self.property {
            let Some(device) = value.as_str() else {
                return Some(Err(PropError::Type));
            };
            if device != self.name {
                self.name = device.into();
                self.switch_pending = playing;
            }
            return Some(Ok(()));
        }
        if name == CONTINUE_ON_ERROR_PROPERTY.name {
            return Some(
                value
                    .as_bool()
                    .map(|on| self.continue_on_error = on)
                    .ok_or(PropError::Type),
            );
        }
        None
    }

    pub(crate) fn get_property(&self, name: &str) -> Option<PropValue> {
        if name == self.property {
            return Some(PropValue::Str(self.name.clone()));
        }
        (name == CONTINUE_ON_ERROR_PROPERTY.name).then_some(PropValue::Bool(self.continue_on_error))
    }

    // the stream that just started plays the current name, so nothing is left to switch
    pub(crate) fn opened(&mut self) {
        self.switch_pending = false;
    }

    pub(crate) fn take_switch(&mut self) -> Option<String> {
        core::mem::take(&mut self.switch_pending).then(|| self.name.clone())
    }

    #[cfg(all(target_os = "linux", feature = "pipewire"))]
    pub(crate) fn continue_on_error(&self) -> bool {
        self.continue_on_error
    }

    #[cfg(all(target_os = "linux", feature = "pipewire"))]
    pub(crate) fn report(&self, error: G2gError) {
        report_on(self.bus.as_ref(), error);
    }
}

fn report_on(bus: Option<&BusHandle>, error: G2gError) {
    if let Some(bus) = bus {
        bus.try_post(BusMessage::Warning(error));
    }
}

#[cfg(any(
    all(
        target_os = "linux",
        any(feature = "alsa-sink", feature = "pulse-sink")
    ),
    all(target_os = "windows", feature = "wasapi-sink")
))]
mod device_worker {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::mpsc::{Receiver, SyncSender};
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

    use alloc::string::String;
    use alloc::vec::Vec;

    use g2g_core::bus::BusHandle;
    use g2g_core::frame::Frame;
    use g2g_core::memory::{MemoryDomain, SystemSlice};
    use g2g_core::{Caps, ClockSync, G2gError, PipelineClock, PresentationPacer, Segment};

    use super::{report_on, OutputDevice};
    use crate::audioconvert::{ns_to_samples, samples_to_ns};
    use crate::audioworker::WorkerLink;

    pub(crate) enum DeviceCmd {
        Samples(Frame),
        StartAt(StartAt),
        // the device drops what it buffered, and nothing queued before this plays
        Flush,
        SwitchDevice(String),
        Shutdown,
    }

    pub(crate) enum DeviceStep {
        Samples(Frame),
        Flush,
        SwitchDevice(String),
        Shutdown,
    }

    pub(crate) struct StartAt {
        deadline_ns: u64,
        clock: Arc<dyn PipelineClock + Send + Sync>,
        geometry: PcmGeometry,
    }

    #[derive(Debug, Clone, Copy)]
    pub(crate) struct PcmGeometry {
        bytes_per_frame: usize,
        rate: u32,
    }

    impl PcmGeometry {
        pub(crate) fn of(caps: &Caps) -> Result<Self, G2gError> {
            let Caps::Audio {
                format,
                channels,
                sample_rate,
                ..
            } = caps
            else {
                return Err(G2gError::CapsMismatch);
            };
            let bytes_per_sample = format.bytes_per_sample().ok_or(G2gError::CapsMismatch)?;
            Ok(Self {
                bytes_per_frame: bytes_per_sample * usize::from(*channels),
                rate: *sample_rate,
            })
        }

        fn duration_ns(self, frame: &Frame) -> u64 {
            let bytes = frame.domain.as_system_slice().map_or(0, <[u8]>::len);
            samples_to_ns((bytes / self.bytes_per_frame) as u64, self.rate)
        }

        fn trim_front(self, frame: Frame, ns: u64) -> Option<Frame> {
            let samples = ns_to_samples(ns, self.rate);
            if samples == 0 {
                return Some(frame);
            }
            let bytes = frame.domain.as_system_slice()?;
            let cut = (samples as usize).saturating_mul(self.bytes_per_frame);
            if cut >= bytes.len() {
                return None;
            }
            let rest = bytes[cut..].to_vec().into_boxed_slice();
            let mut timing = frame.timing;
            timing.pts_ns = timing
                .pts_ns
                .saturating_add(samples_to_ns(samples, self.rate));
            Some(Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(rest)),
                timing,
                ..frame
            })
        }
    }

    #[derive(Debug, Default)]
    pub(crate) struct PlayoutStart {
        pacer: PresentationPacer,
        started: bool,
        held: Vec<Frame>,
    }

    impl PlayoutStart {
        pub(crate) fn set_clock_sync(&mut self, sync: ClockSync) {
            self.pacer.set_clock_sync(sync);
        }

        pub(crate) fn set_segment(&mut self, segment: Segment) {
            self.pacer.set_segment(segment);
        }

        pub(crate) async fn send(
            &mut self,
            link: &WorkerLink<DeviceCmd>,
            frame: Frame,
            geometry: PcmGeometry,
            err: G2gError,
        ) -> Result<(), G2gError> {
            for cmd in self.admit(frame, geometry) {
                link.send(cmd, err.clone()).await?;
            }
            Ok(())
        }

        pub(crate) async fn flush(
            &mut self,
            link: &WorkerLink<DeviceCmd>,
            pending_flushes: &PendingFlushes,
            err: G2gError,
        ) -> Result<(), G2gError> {
            self.pacer.flush();
            self.started = false;
            self.held.clear();
            pending_flushes.send(link, err).await
        }

        pub(crate) async fn send_held(
            &mut self,
            link: &WorkerLink<DeviceCmd>,
            err: G2gError,
        ) -> Result<(), G2gError> {
            for cmd in self.take_held() {
                link.send(cmd, err.clone()).await?;
            }
            Ok(())
        }

        fn admit(&mut self, frame: Frame, geometry: PcmGeometry) -> Vec<DeviceCmd> {
            if self.started {
                return Vec::from([DeviceCmd::Samples(frame)]);
            }
            self.held.push(frame);
            let Some(sync) = self.pacer.clock_sync() else {
                return self.take_held();
            };
            if sync.awaits_play_edge() {
                return Vec::new();
            }
            let clock = Arc::clone(&sync.clock);
            // samples before the segment start are not played, as gstreamer's audio sink clips them
            let segment_start = self.pacer.segment().map_or(0, |segment| segment.start);
            while let Some(first) = self.held.first() {
                let before_start = segment_start.saturating_sub(first.timing.pts_ns);
                let first = self.held.remove(0);
                if let Some(rest) = geometry.trim_front(first, before_start) {
                    self.held.insert(0, rest);
                    break;
                }
            }
            let Some(first_pts) = self.held.first().map(|first| first.timing.pts_ns) else {
                return Vec::new();
            };
            let deadline = self.pacer.deadline_ns(first_pts);
            let held = self.take_held();
            let Some(deadline_ns) = deadline else {
                return held;
            };
            let mut cmds = Vec::with_capacity(held.len() + 1);
            cmds.push(DeviceCmd::StartAt(StartAt {
                deadline_ns,
                clock,
                geometry,
            }));
            cmds.extend(held);
            cmds
        }

        fn take_held(&mut self) -> Vec<DeviceCmd> {
            self.started = true;
            self.held.drain(..).map(DeviceCmd::Samples).collect()
        }
    }

    #[derive(Debug, Clone, Default)]
    pub(crate) struct PendingFlushes(Arc<AtomicU64>);

    impl PendingFlushes {
        async fn send(&self, link: &WorkerLink<DeviceCmd>, err: G2gError) -> Result<(), G2gError> {
            self.0.fetch_add(1, Ordering::AcqRel);
            link.send(DeviceCmd::Flush, err).await
        }

        fn pending(&self) -> bool {
            self.0.load(Ordering::Acquire) > 0
        }
    }

    const FLUSH_CHECK_DURING_START_WAIT: Duration = Duration::from_millis(10);

    // thread::sleep wakes up to a scheduler tick late
    const START_LATENESS_KEPT_NS: u64 = 5_000_000;

    pub(crate) struct PlayoutGate {
        pending_flushes: PendingFlushes,
        start: Option<StartAt>,
    }

    impl PlayoutGate {
        fn new(pending_flushes: PendingFlushes) -> Self {
            Self {
                pending_flushes,
                start: None,
            }
        }

        pub(crate) fn admit(&mut self, cmd: DeviceCmd) -> Option<DeviceStep> {
            match cmd {
                // samples queued ahead of a flush belong to the position before a seek
                DeviceCmd::Samples(_) | DeviceCmd::StartAt(_) if self.pending_flushes.pending() => {
                    None
                }
                DeviceCmd::Flush => {
                    self.pending_flushes.0.fetch_sub(1, Ordering::AcqRel);
                    self.start = None;
                    Some(DeviceStep::Flush)
                }
                DeviceCmd::StartAt(start) => {
                    self.start = Some(start);
                    None
                }
                DeviceCmd::Samples(frame) => self.start_playing(frame).map(DeviceStep::Samples),
                DeviceCmd::SwitchDevice(device) => Some(DeviceStep::SwitchDevice(device)),
                DeviceCmd::Shutdown => Some(DeviceStep::Shutdown),
            }
        }

        fn start_playing(&mut self, frame: Frame) -> Option<Frame> {
            let Some(start) = self.start.take() else {
                return Some(frame);
            };
            loop {
                let now = start.clock.now_ns();
                if now >= start.deadline_ns {
                    break;
                }
                if self.pending_flushes.pending() {
                    return None;
                }
                thread::sleep(
                    Duration::from_nanos(start.deadline_ns - now)
                        .min(FLUSH_CHECK_DURING_START_WAIT),
                );
            }
            if self.pending_flushes.pending() {
                return None;
            }
            let late_ns = start.clock.now_ns().saturating_sub(start.deadline_ns);
            if late_ns < START_LATENESS_KEPT_NS {
                return Some(frame);
            }
            let duration_ns = start.geometry.duration_ns(&frame);
            let trimmed = start.geometry.trim_front(frame, late_ns);
            if trimmed.is_none() {
                // the next buffer starts where this one ended
                self.start = Some(StartAt {
                    deadline_ns: start.deadline_ns + duration_ns,
                    ..start
                });
            }
            trimmed
        }
    }

    pub(crate) enum PlayEnd {
        Shutdown,
        SwitchDevice(String),
        Failed(G2gError),
    }

    #[derive(Debug, Clone)]
    pub(crate) struct DeviceFallback {
        continue_on_error: bool,
        bus: Option<BusHandle>,
        bytes_per_second: u64,
        // the sink's own playout counter, advanced in the sink's unit while no device plays
        rendered: Arc<AtomicU64>,
        bytes_per_rendered_unit: u64,
        payload_len: fn(&Frame) -> Option<usize>,
    }

    impl OutputDevice {
        pub(crate) fn fallback(
            &self,
            bytes_per_second: u64,
            rendered: Arc<AtomicU64>,
            bytes_per_rendered_unit: u64,
            payload_len: fn(&Frame) -> Option<usize>,
        ) -> DeviceFallback {
            DeviceFallback {
                continue_on_error: self.continue_on_error,
                bus: self.bus.clone(),
                bytes_per_second,
                rendered,
                bytes_per_rendered_unit,
                payload_len,
            }
        }
    }

    pub(crate) fn run_device_worker<Device>(
        first_device: String,
        rx: Receiver<DeviceCmd>,
        pending_flushes: PendingFlushes,
        fallback: DeviceFallback,
        ready: SyncSender<Result<(), G2gError>>,
        mut open: impl FnMut(&str) -> Result<Device, G2gError>,
        mut play: impl FnMut(Device, &Receiver<DeviceCmd>, &mut PlayoutGate) -> PlayEnd,
    ) {
        let mut gate = PlayoutGate::new(pending_flushes);
        let mut ready = Some(ready);
        let mut device = first_device;
        loop {
            let end = match open(&device) {
                Ok(opened) => {
                    if let Some(ready) = ready.take() {
                        let _ = ready.send(Ok(()));
                    }
                    play(opened, &rx, &mut gate)
                }
                Err(error) => PlayEnd::Failed(error),
            };
            let error = match end {
                PlayEnd::Shutdown => return,
                PlayEnd::SwitchDevice(next) => {
                    device = next;
                    continue;
                }
                PlayEnd::Failed(error) => error,
            };
            if !fallback.continue_on_error {
                if let Some(ready) = ready.take() {
                    let _ = ready.send(Err(error));
                }
                return;
            }
            report_on(fallback.bus.as_ref(), error);
            if let Some(ready) = ready.take() {
                let _ = ready.send(Ok(()));
            }
            match play_without_device(&rx, &mut gate, &fallback) {
                Some(next) => device = next,
                None => return,
            }
        }
    }

    fn play_without_device(
        rx: &Receiver<DeviceCmd>,
        gate: &mut PlayoutGate,
        fallback: &DeviceFallback,
    ) -> Option<String> {
        let mut first_sample_at: Option<Instant> = None;
        let mut consumed_bytes = 0u64;
        loop {
            match rx.recv().map(|cmd| gate.admit(cmd)) {
                Ok(None | Some(DeviceStep::Flush)) => {}
                Ok(Some(DeviceStep::Samples(frame))) => {
                    let started = *first_sample_at.get_or_insert_with(Instant::now);
                    let bytes = (fallback.payload_len)(&frame)? as u64;
                    consumed_bytes += bytes;
                    fallback
                        .rendered
                        .fetch_add(bytes / fallback.bytes_per_rendered_unit, Ordering::Relaxed);
                    let due_ns = u128::from(consumed_bytes) * 1_000_000_000
                        / u128::from(fallback.bytes_per_second);
                    let due = started + Duration::from_nanos(due_ns as u64);
                    if let Some(wait) = due.checked_duration_since(Instant::now()) {
                        thread::sleep(wait);
                    }
                }
                Ok(Some(DeviceStep::SwitchDevice(next))) => return Some(next),
                Ok(Some(DeviceStep::Shutdown)) | Err(_) => return None,
            }
        }
    }
}

#[cfg(all(
    test,
    any(
        all(
            target_os = "linux",
            any(feature = "alsa-sink", feature = "pulse-sink")
        ),
        all(target_os = "windows", feature = "wasapi-sink")
    )
))]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::mpsc::{sync_channel, Receiver};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use crate::audioworker::WorkerLink;
    use alloc::vec::Vec;
    use g2g_core::bus::Bus;
    use g2g_core::frame::Frame;
    use g2g_core::memory::{MemoryDomain, SystemSlice};

    const BYTES_PER_SECOND: u64 = 48_000 * 4;

    fn samples(bytes: usize) -> DeviceCmd {
        DeviceCmd::Samples(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(
                alloc::vec![0u8; bytes].into_boxed_slice(),
            )),
            timing: Default::default(),
            sequence: 0,
            meta: Default::default(),
        })
    }

    // the device boundary: records which device each buffer reached, and refuses names starting with "missing"
    fn spawn_worker(
        output: &OutputDevice,
        rendered: Arc<AtomicU64>,
        played: Arc<Mutex<Vec<String>>>,
    ) -> (
        std::sync::mpsc::SyncSender<DeviceCmd>,
        Receiver<Result<(), G2gError>>,
        std::thread::JoinHandle<()>,
    ) {
        let (tx, rx) = sync_channel(64);
        let (ready_tx, ready_rx) = sync_channel(1);
        let fallback = output.fallback(BYTES_PER_SECOND, rendered, 1, |frame| {
            frame.domain.as_system_slice().map(<[u8]>::len)
        });
        let first = String::from(output.name());
        let worker = std::thread::spawn(move || {
            run_device_worker(
                first,
                rx,
                PendingFlushes::default(),
                fallback,
                ready_tx,
                |device| match device.starts_with("missing") {
                    true => Err(G2gError::Hardware(g2g_core::HardwareError::Alsa(-2))),
                    false => Ok(String::from(device)),
                },
                |device, rx, gate| loop {
                    match rx.recv().map(|cmd| gate.admit(cmd)) {
                        Ok(None | Some(DeviceStep::Flush)) => {}
                        Ok(Some(DeviceStep::Samples(_))) => {
                            played.lock().unwrap().push(device.clone())
                        }
                        Ok(Some(DeviceStep::SwitchDevice(next))) => {
                            return PlayEnd::SwitchDevice(next)
                        }
                        Ok(Some(DeviceStep::Shutdown)) | Err(_) => return PlayEnd::Shutdown,
                    }
                },
            );
        });
        (tx, ready_rx, worker)
    }

    #[test]
    fn setting_the_device_while_playing_switches_once_in_order() {
        let mut output = OutputDevice::new("device", "first");
        let rendered = Arc::new(AtomicU64::new(0));
        let played = Arc::new(Mutex::new(Vec::new()));
        let (tx, ready, worker) = spawn_worker(&output, rendered, Arc::clone(&played));
        assert_eq!(ready.recv().unwrap(), Ok(()));
        output.opened();

        tx.send(samples(4)).unwrap();
        let same = PropValue::Str("first".into());
        assert_eq!(output.set_property("device", &same, true), Some(Ok(())));
        assert_eq!(output.take_switch(), None);
        let second = PropValue::Str("second".into());
        assert_eq!(output.set_property("device", &second, true), Some(Ok(())));
        let switch = output.take_switch().expect("a changed device switches");
        assert_eq!(output.take_switch(), None);
        tx.send(DeviceCmd::SwitchDevice(switch)).unwrap();
        tx.send(samples(4)).unwrap();
        tx.send(DeviceCmd::Shutdown).unwrap();
        worker.join().unwrap();

        assert_eq!(*played.lock().unwrap(), ["first", "second"]);
    }

    #[derive(Debug, PartialEq)]
    enum DeviceEvent {
        Samples { pts_ns: u64, len: usize },
        Flush,
    }

    type DeviceLog = Arc<Mutex<Vec<(DeviceEvent, Instant)>>>;

    fn spawn_recording_worker(
        pending_flushes: PendingFlushes,
        open: Receiver<()>,
    ) -> (WorkerLink<DeviceCmd>, DeviceLog) {
        let output = OutputDevice::new("device", "first");
        let fallback = output.fallback(BYTES_PER_SECOND, Arc::new(AtomicU64::new(0)), 1, |frame| {
            frame.domain.as_system_slice().map(<[u8]>::len)
        });
        let log = DeviceLog::default();
        let device_log = Arc::clone(&log);
        let (ready_tx, _ready_rx) = sync_channel(1);
        let link = WorkerLink::spawn("test-audio", move |rx| {
            run_device_worker(
                String::from("first"),
                rx,
                pending_flushes,
                fallback,
                ready_tx,
                |device| {
                    open.recv().unwrap();
                    Ok(String::from(device))
                },
                |_, rx, gate| loop {
                    let event = match rx.recv().map(|cmd| gate.admit(cmd)) {
                        Ok(None) => continue,
                        Ok(Some(DeviceStep::Samples(frame))) => DeviceEvent::Samples {
                            pts_ns: frame.timing.pts_ns,
                            len: frame.domain.as_system_slice().unwrap().len(),
                        },
                        Ok(Some(DeviceStep::Flush)) => DeviceEvent::Flush,
                        Ok(Some(DeviceStep::SwitchDevice(next))) => {
                            return PlayEnd::SwitchDevice(next)
                        }
                        Ok(Some(DeviceStep::Shutdown)) | Err(_) => return PlayEnd::Shutdown,
                    };
                    device_log.lock().unwrap().push((event, Instant::now()));
                },
            );
        })
        .unwrap();
        (link, log)
    }

    fn pcm_frame(pts_ns: u64, bytes: usize) -> Frame {
        let DeviceCmd::Samples(mut frame) = samples(bytes) else {
            unreachable!("samples builds a Samples command")
        };
        frame.timing.pts_ns = pts_ns;
        frame
    }

    const STEREO_S16_RATE: u32 = 48_000;
    const STEREO_S16_BYTES_PER_FRAME: usize = 4;

    fn stereo_s16() -> PcmGeometry {
        PcmGeometry::of(&g2g_core::Caps::Audio {
            format: g2g_core::AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: STEREO_S16_RATE,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        })
        .unwrap()
    }

    fn err() -> G2gError {
        G2gError::Hardware(g2g_core::HardwareError::Alsa(-1))
    }

    fn events(log: &DeviceLog) -> Vec<DeviceEvent> {
        log.lock()
            .unwrap()
            .drain(..)
            .map(|(event, _)| event)
            .collect()
    }

    #[test]
    fn samples_queued_before_a_flush_never_play() {
        const BEFORE_FLUSH: usize = 4;
        const AFTER_FLUSH: usize = 8;
        let pending_flushes = PendingFlushes::default();
        // the device opens only once the samples and the flush are queued behind it
        let (open_tx, open_rx) = sync_channel::<()>(1);
        let (mut link, log) = spawn_recording_worker(pending_flushes.clone(), open_rx);
        let mut start = PlayoutStart::default();
        g2g_core::runtime::block_on(async {
            for _ in 0..2 {
                let frame = pcm_frame(0, BEFORE_FLUSH);
                start.send(&link, frame, stereo_s16(), err()).await.unwrap();
            }
            start.flush(&link, &pending_flushes, err()).await.unwrap();
            let frame = pcm_frame(0, AFTER_FLUSH);
            start.send(&link, frame, stereo_s16(), err()).await.unwrap();
            open_tx.send(()).unwrap();
            link.finish(DeviceCmd::Shutdown, err()).await.unwrap();
        });

        assert_eq!(
            events(&log),
            [
                DeviceEvent::Flush,
                DeviceEvent::Samples {
                    pts_ns: 0,
                    len: AFTER_FLUSH
                }
            ]
        );
    }

    #[test]
    fn a_preroll_buffer_waits_for_the_play_edge_then_its_deadline() {
        const BUFFER_BYTES: usize = 4_800;
        const START_DELAY_NS: u64 = 50_000_000;
        let clock: Arc<dyn g2g_core::PipelineClock + Send + Sync> =
            Arc::new(g2g_core::MonotonicClock);
        let anchor = g2g_core::clock::PlayAnchor::new();
        let mut start = PlayoutStart::default();
        start.set_clock_sync(g2g_core::ClockSync::with_play_anchor(
            Arc::clone(&clock),
            0,
            anchor.clone(),
        ));
        start.set_segment(g2g_core::Segment::new());
        let (open_tx, open_rx) = sync_channel::<()>(1);
        open_tx.send(()).unwrap();
        let (mut link, log) = spawn_recording_worker(PendingFlushes::default(), open_rx);
        let geometry = stereo_s16();
        let preroll = pcm_frame(0, BUFFER_BYTES);
        let next_pts = crate::audioconvert::samples_to_ns(
            (BUFFER_BYTES / STEREO_S16_BYTES_PER_FRAME) as u64,
            STEREO_S16_RATE,
        );
        let stamped_at = g2g_core::runtime::block_on(async {
            start.send(&link, preroll, geometry, err()).await.unwrap();
            std::thread::sleep(Duration::from_nanos(START_DELAY_NS));
            assert!(
                events(&log).is_empty(),
                "a preroll buffer played before Playing"
            );

            let stamped_at = Instant::now();
            anchor.stamp(clock.now_ns() + START_DELAY_NS);
            let next = pcm_frame(next_pts, BUFFER_BYTES);
            start.send(&link, next, geometry, err()).await.unwrap();
            link.finish(DeviceCmd::Shutdown, err()).await.unwrap();
            stamped_at
        });

        let log = log.lock().unwrap();
        let played: Vec<&DeviceEvent> = log.iter().map(|(event, _)| event).collect();
        assert_eq!(
            played,
            [
                &DeviceEvent::Samples {
                    pts_ns: 0,
                    len: BUFFER_BYTES
                },
                &DeviceEvent::Samples {
                    pts_ns: next_pts,
                    len: BUFFER_BYTES
                }
            ]
        );
        assert!(log[0].1 >= stamped_at + Duration::from_nanos(START_DELAY_NS));
    }

    #[test]
    fn samples_past_their_start_are_dropped_from_the_front() {
        const BASE_NS: u64 = 1_000_000_000;
        const LATE_NS: u64 = 10_000_000;
        const BUFFER_BYTES: usize = 19_200;
        let manual = crate::testutil::ManualClock::default();
        manual.set(BASE_NS + LATE_NS);
        let anchor = g2g_core::clock::PlayAnchor::new();
        anchor.stamp(BASE_NS);
        let mut start = PlayoutStart::default();
        start.set_clock_sync(g2g_core::ClockSync::with_play_anchor(
            Arc::new(manual),
            0,
            anchor,
        ));
        start.set_segment(g2g_core::Segment::new());
        let (open_tx, open_rx) = sync_channel::<()>(1);
        open_tx.send(()).unwrap();
        let (mut link, log) = spawn_recording_worker(PendingFlushes::default(), open_rx);
        let geometry = stereo_s16();
        g2g_core::runtime::block_on(async {
            let frame = pcm_frame(0, BUFFER_BYTES);
            start.send(&link, frame, geometry, err()).await.unwrap();
            link.finish(DeviceCmd::Shutdown, err()).await.unwrap();
        });

        let late_bytes = crate::audioconvert::ns_to_samples(LATE_NS, STEREO_S16_RATE) as usize
            * STEREO_S16_BYTES_PER_FRAME;
        assert_eq!(
            events(&log),
            [DeviceEvent::Samples {
                pts_ns: LATE_NS,
                len: BUFFER_BYTES - late_bytes
            }]
        );
    }

    #[test]
    fn a_device_set_before_playing_needs_no_switch() {
        let mut output = OutputDevice::new("device", "first");
        let second = PropValue::Str("second".into());
        assert_eq!(output.set_property("device", &second, false), Some(Ok(())));
        assert_eq!(output.take_switch(), None);
        assert_eq!(output.name(), "second");
    }

    #[test]
    fn a_failed_open_errors_without_continue_on_error() {
        let output = OutputDevice::new("device", "missing");
        let (_tx, ready, worker) = spawn_worker(
            &output,
            Arc::new(AtomicU64::new(0)),
            Arc::new(Mutex::new(Vec::new())),
        );
        assert_eq!(
            ready.recv().unwrap(),
            Err(G2gError::Hardware(g2g_core::HardwareError::Alsa(-2)))
        );
        worker.join().unwrap();
    }

    #[test]
    fn continue_on_error_warns_then_consumes_in_real_time_until_a_device_opens() {
        let (bus, handle) = Bus::new(8);
        let mut output = OutputDevice::new("device", "missing");
        output.set_bus(handle);
        let on = PropValue::Bool(true);
        assert_eq!(
            output.set_property("continue-on-error", &on, false),
            Some(Ok(()))
        );
        let rendered = Arc::new(AtomicU64::new(0));
        let played = Arc::new(Mutex::new(Vec::new()));
        let (tx, ready, worker) = spawn_worker(&output, Arc::clone(&rendered), Arc::clone(&played));
        assert_eq!(ready.recv().unwrap(), Ok(()));
        assert!(matches!(
            bus.try_recv(),
            Some(BusMessage::Warning(G2gError::Hardware(
                g2g_core::HardwareError::Alsa(-2)
            )))
        ));

        // a tenth of a second of audio with no device still takes a tenth of a second
        let tenth_of_a_second = (BYTES_PER_SECOND / 10) as usize;
        let started = Instant::now();
        tx.send(samples(tenth_of_a_second)).unwrap();
        tx.send(DeviceCmd::SwitchDevice(String::from("back")))
            .unwrap();
        tx.send(samples(4)).unwrap();
        tx.send(DeviceCmd::Shutdown).unwrap();
        worker.join().unwrap();

        assert!(started.elapsed() >= Duration::from_millis(100));
        assert_eq!(rendered.load(Ordering::Relaxed), tenth_of_a_second as u64);
        assert_eq!(*played.lock().unwrap(), ["back"]);
    }
}
