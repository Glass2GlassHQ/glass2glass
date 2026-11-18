// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::string::String;

use g2g_core::bus::{BusHandle, BusMessage};
use g2g_core::{G2gError, PropError, PropKind, PropValue, PropertySpec};

#[cfg(not(target_os = "windows"))]
pub(crate) use device_worker::PcmGeometry;
pub(crate) use device_worker::{
    DeviceCmd, DeviceStep, PausableDevice, PlayEnd, PlayoutGate, PlayoutLink,
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
}

fn report_on(bus: Option<&BusHandle>, error: G2gError) {
    if let Some(bus) = bus {
        bus.try_post(BusMessage::Warning(error));
    }
}

mod device_worker {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::mpsc::{self, Receiver, RecvError, RecvTimeoutError, SyncSender};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};

    use alloc::boxed::Box;
    use alloc::string::String;
    use alloc::vec::Vec;

    use g2g_core::bus::BusHandle;
    use g2g_core::frame::Frame;
    use g2g_core::memory::{MemoryDomain, SystemSlice};
    use g2g_core::progress::SinkPosition;
    use g2g_core::{AudioFormat, Caps, ClockSync, G2gError, PresentationPacer, Segment};

    use super::{report_on, OutputDevice};
    use crate::audioconvert::{ns_to_samples, samples_to_ns, silence_byte};
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
        paused_total_ns: u64,
        sync: ClockSync,
        segment: Option<Segment>,
        geometry: PcmGeometry,
    }

    impl StartAt {
        // a pause before the start moves the start back by its length
        fn deadline_after_pauses_ns(&self) -> u64 {
            self.deadline_ns
                + self
                    .sync
                    .paused_total_ns()
                    .saturating_sub(self.paused_total_ns)
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub(crate) struct PcmGeometry {
        bytes_per_frame: usize,
        rate: u32,
        silence: u8,
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
            Self::new(*format, u32::from(*channels), *sample_rate)
        }

        pub(crate) fn new(format: AudioFormat, channels: u32, rate: u32) -> Result<Self, G2gError> {
            let bytes_per_sample = format.bytes_per_sample().ok_or(G2gError::CapsMismatch)?;
            let bytes_per_frame = bytes_per_sample * channels as usize;
            if bytes_per_frame == 0 || rate == 0 {
                return Err(G2gError::CapsMismatch);
            }
            Ok(Self {
                bytes_per_frame,
                rate,
                silence: silence_byte(format),
            })
        }

        #[cfg(target_os = "android")]
        pub(crate) fn bytes_per_frame(self) -> usize {
            self.bytes_per_frame
        }

        pub(crate) fn bytes_to_ns(self, bytes: usize) -> u64 {
            samples_to_ns((bytes / self.bytes_per_frame) as u64, self.rate)
        }

        #[cfg(not(target_os = "windows"))]
        fn pause_check_slices<'a>(self, bytes: &'a [u8]) -> core::slice::Chunks<'a, u8> {
            let frames = (self.rate / PAUSE_CHECKS_PER_SECOND).max(1) as usize;
            bytes.chunks(frames * self.bytes_per_frame)
        }

        fn duration_ns(self, frame: &Frame) -> u64 {
            self.bytes_to_ns(frame.domain.as_system_slice().map_or(0, <[u8]>::len))
        }

        fn clip_to_start(self, frame: Frame, start_ns: u64) -> Option<Frame> {
            let before_start_ns = start_ns.saturating_sub(frame.timing.pts_ns);
            if before_start_ns == 0 {
                return Some(frame);
            }
            let mut rest = self.trim_front(frame, before_start_ns)?;
            // gstreamer's audio clipping stamps the first kept sample at the segment start, a part sample before it rounds away
            rest.timing.pts_ns = start_ns;
            Some(rest)
        }

        fn prepend_silence(self, frame: Frame, ns: u64) -> Option<Frame> {
            let samples = ns_to_samples(ns, self.rate);
            let bytes = frame.domain.as_system_slice()?;
            let mut padded = alloc::vec![self.silence; samples as usize * self.bytes_per_frame];
            padded.extend_from_slice(bytes);
            let mut timing = frame.timing;
            timing.pts_ns = timing
                .pts_ns
                .saturating_sub(samples_to_ns(samples, self.rate));
            Some(Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(padded.into_boxed_slice())),
                timing,
                ..frame
            })
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

    // a pause stops the device within one slice, however long the buffers upstream are
    #[cfg(not(target_os = "windows"))]
    const PAUSE_CHECKS_PER_SECOND: u32 = 50;

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
            self.reset();
            pending_flushes.send(link, err).await
        }

        fn reset(&mut self) {
            self.pacer.flush();
            self.started = false;
            self.held.clear();
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
            let sync = sync.clone();
            // samples before the segment start are not played, as gstreamer's audio sink clips them
            let segment_start = self.pacer.segment().map_or(0, |segment| segment.start);
            while !self.held.is_empty() {
                let first = self.held.remove(0);
                if let Some(rest) = geometry.clip_to_start(first, segment_start) {
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
                paused_total_ns: sync.paused_total_ns(),
                sync,
                segment: self.pacer.segment(),
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

    // the stream time of the sample the device plays now, once the sink has adopted the pipeline's position handle
    #[derive(Debug, Clone, Default)]
    pub(crate) struct PlayedPosition(Arc<Mutex<Option<SinkPosition>>>);

    impl PlayedPosition {
        fn adopt(&self, position: SinkPosition) {
            *self.0.lock().expect("played position lock") = Some(position);
        }

        fn set_at_pts(&self, segment: Option<&Segment>, pts_ns: u64) {
            if let Some(position) = &*self.0.lock().expect("played position lock") {
                position.set_at_pts(segment, pts_ns);
            }
        }

        fn forget(&self) {
            if let Some(position) = &*self.0.lock().expect("played position lock") {
                position.forget();
            }
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

    const DEVICE_OPEN_TIMEOUT: Duration = Duration::from_secs(5);

    struct RunningWorker {
        link: WorkerLink<DeviceCmd>,
        geometry: PcmGeometry,
    }

    type SpawnWorker = Box<
        dyn FnMut(
                String,
                PlayoutGate,
                SyncSender<Result<(), G2gError>>,
            ) -> Result<WorkerLink<DeviceCmd>, G2gError>
            + Send,
    >;

    struct WorkerSpawn {
        geometry: PcmGeometry,
        spawn: SpawnWorker,
    }

    pub(crate) struct PlayoutLink {
        worker: Option<RunningWorker>,
        // what opens the device again for a seek after the end of the stream shut it
        worker_spawn: Option<WorkerSpawn>,
        pending_flushes: PendingFlushes,
        played_position: PlayedPosition,
        start: PlayoutStart,
        worker_error: G2gError,
    }

    impl core::fmt::Debug for PlayoutLink {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("PlayoutLink")
                .field("running", &self.is_running())
                .finish_non_exhaustive()
        }
    }

    impl PlayoutLink {
        pub(crate) fn new(worker_error: G2gError) -> Self {
            Self {
                worker: None,
                worker_spawn: None,
                pending_flushes: PendingFlushes::default(),
                played_position: PlayedPosition::default(),
                start: PlayoutStart::default(),
                worker_error,
            }
        }

        pub(crate) fn is_running(&self) -> bool {
            self.worker
                .as_ref()
                .is_some_and(|worker| worker.link.is_running())
        }

        pub(crate) fn set_clock_sync(&mut self, sync: ClockSync) {
            self.start.set_clock_sync(sync);
        }

        pub(crate) fn set_segment(&mut self, segment: Segment) {
            self.start.set_segment(segment);
        }

        // without a clock the worker never learns when a sample plays
        pub(crate) fn adopt_position(&mut self, position: SinkPosition) -> Option<SinkPosition> {
            if !self.start.pacer.is_paced() {
                return Some(position);
            }
            self.played_position.adopt(position);
            None
        }

        pub(crate) fn open<Device, Open, Play>(
            &mut self,
            thread_name: &str,
            caps: &Caps,
            output: &mut OutputDevice,
            fallback: DeviceFallback,
            open: Open,
            play: Play,
        ) -> Result<(), G2gError>
        where
            Open: FnMut(&str) -> Result<Device, G2gError> + Clone + Send + 'static,
            Play: FnMut(Device, &Receiver<DeviceCmd>, &mut PlayoutGate) -> PlayEnd
                + Clone
                + Send
                + 'static,
        {
            self.abort();
            let geometry = PcmGeometry::of(caps)?;
            let thread_name = String::from(thread_name);
            let spawn = move |device: String, gate: PlayoutGate, ready: SyncSender<_>| {
                let (fallback, open, play) = (fallback.clone(), open.clone(), play.clone());
                WorkerLink::spawn(&thread_name, move |rx| {
                    run_device_worker(device, rx, gate, fallback, ready, open, play);
                })
            };
            self.worker_spawn = Some(WorkerSpawn {
                geometry,
                spawn: Box::new(spawn),
            });
            self.start_worker(output)
        }

        fn start_worker(&mut self, output: &mut OutputDevice) -> Result<(), G2gError> {
            self.pending_flushes = PendingFlushes::default();
            let gate = PlayoutGate::new(self.pending_flushes.clone(), self.played_position.clone());
            let worker_spawn = self.worker_spawn.as_mut().ok_or(G2gError::NotConfigured)?;
            let (ready_tx, ready_rx) = mpsc::sync_channel(1);
            let link = (worker_spawn.spawn)(String::from(output.name()), gate, ready_tx)?;
            let geometry = worker_spawn.geometry;
            // a host with no audio device fails here instead of dropping audio
            match ready_rx.recv_timeout(DEVICE_OPEN_TIMEOUT) {
                Ok(Ok(())) => {}
                Ok(Err(error)) => return Err(error),
                Err(_) => return Err(self.worker_error.clone()),
            }
            output.opened();
            self.worker = Some(RunningWorker { link, geometry });
            Ok(())
        }

        pub(crate) async fn send(
            &mut self,
            frame: Frame,
            output: &mut OutputDevice,
        ) -> Result<(), G2gError> {
            if self.worker.is_none() {
                self.start_worker(output)?;
            }
            let worker = self.worker.as_ref().expect("started above");
            if let Some(device) = output.take_switch() {
                let switch = DeviceCmd::SwitchDevice(device);
                worker.link.send(switch, self.worker_error.clone()).await?;
            }
            let error = self.worker_error.clone();
            self.start
                .send(&worker.link, frame, worker.geometry, error)
                .await
        }

        pub(crate) async fn flush(&mut self) -> Result<(), G2gError> {
            self.played_position.forget();
            let Some(worker) = self.worker.as_ref() else {
                self.worker_spawn.as_ref().ok_or(G2gError::NotConfigured)?;
                self.start.reset();
                return Ok(());
            };
            let error = self.worker_error.clone();
            self.start
                .flush(&worker.link, &self.pending_flushes, error)
                .await
        }

        pub(crate) async fn finish(&mut self) -> Result<(), G2gError> {
            let Some(mut worker) = self.worker.take() else {
                return Ok(());
            };
            let error = self.worker_error.clone();
            self.start.send_held(&worker.link, error.clone()).await?;
            worker.link.finish(DeviceCmd::Shutdown, error).await
        }

        pub(crate) fn abort(&mut self) {
            self.worker = None;
        }
    }

    const COMMAND_CHECK_INTERVAL: Duration = Duration::from_millis(10);

    // thread::sleep wakes up to a scheduler tick late
    const START_LATENESS_KEPT_NS: u64 = 5_000_000;

    // pipewire-alsa's delay jumps by a quantum and stalls for a few milliseconds after an unpause
    const RESYNC_SETTLE_NS: u64 = 100_000_000;
    // alsa delay readings through pipewire jitter by about nine milliseconds
    const RESYNC_TOLERANCE_NS: i128 = 10_000_000;

    pub(crate) trait PausableDevice {
        fn pause(&mut self) -> Result<(), G2gError>;
        fn resume(&mut self) -> Result<(), G2gError>;
        fn still_playing_after_pause_ns(&self) -> Result<u64, G2gError>;
        fn delay_ns(&self) -> Result<u64, G2gError>;
    }

    #[derive(Debug, Clone, Copy)]
    struct Timeline {
        // clock time minus pts of audio that plays on time
        offset_ns: i128,
        paused_total_ns: u64,
        segment: Option<Segment>,
        geometry: PcmGeometry,
    }

    impl Timeline {
        fn offset_now_ns(&self, sync: &ClockSync) -> i128 {
            self.offset_ns + i128::from(sync.paused_total_ns().saturating_sub(self.paused_total_ns))
        }
    }

    pub(crate) struct PlayoutGate {
        pending_flushes: PendingFlushes,
        played_position: PlayedPosition,
        start: Option<StartAt>,
        sync: Option<ClockSync>,
        timeline: Option<Timeline>,
        handed_out_end_pts_ns: u64,
        resync_at_ns: Option<u64>,
        // positive drops that much audio from the next buffer, negative plays that much silence first
        correction_ns: i128,
        // commands received while waiting on a start or a pause, played back in order
        backlog: VecDeque<DeviceCmd>,
        disconnected: bool,
    }

    impl PlayoutGate {
        pub(crate) fn new(
            pending_flushes: PendingFlushes,
            played_position: PlayedPosition,
        ) -> Self {
            Self {
                pending_flushes,
                played_position,
                start: None,
                sync: None,
                timeline: None,
                handed_out_end_pts_ns: 0,
                resync_at_ns: None,
                correction_ns: 0,
                backlog: VecDeque::new(),
                disconnected: false,
            }
        }

        pub(crate) fn recv(
            &mut self,
            rx: &Receiver<DeviceCmd>,
        ) -> Result<Option<DeviceStep>, RecvError> {
            if self.disconnected {
                return Err(RecvError);
            }
            let cmd = match self.backlog.pop_front() {
                Some(cmd) => cmd,
                None => rx.recv()?,
            };
            Ok(self.admit(cmd, rx))
        }

        #[cfg(all(target_os = "windows", feature = "wasapi-sink"))]
        pub(crate) fn recv_timeout(
            &mut self,
            rx: &Receiver<DeviceCmd>,
            timeout: Duration,
        ) -> Result<Option<DeviceStep>, RecvTimeoutError> {
            if self.disconnected {
                return Err(RecvTimeoutError::Disconnected);
            }
            let cmd = match self.backlog.pop_front() {
                Some(cmd) => cmd,
                None => rx.recv_timeout(timeout)?,
            };
            Ok(self.admit(cmd, rx))
        }

        #[cfg(not(target_os = "windows"))]
        pub(crate) fn play_in_slices<Device: PausableDevice>(
            &mut self,
            rx: &Receiver<DeviceCmd>,
            pcm: &[u8],
            geometry: PcmGeometry,
            device: &mut Device,
            mut write: impl FnMut(&mut Device, &[u8]) -> Result<(), G2gError>,
        ) -> Result<(), G2gError> {
            let mut unwritten = pcm.len();
            for slice in geometry.pause_check_slices(pcm) {
                let unwritten_ns = geometry.bytes_to_ns(unwritten);
                self.publish_played_position(device, unwritten_ns)?;
                self.pause_or_resync(rx, device, unwritten_ns)?;
                if self.discards_queued_audio() {
                    break;
                }
                write(device, slice)?;
                unwritten -= slice.len();
            }
            Ok(())
        }

        // returns whether the device was paused
        pub(crate) fn pause_or_resync(
            &mut self,
            rx: &Receiver<DeviceCmd>,
            device: &mut impl PausableDevice,
            unwritten_ns: u64,
        ) -> Result<bool, G2gError> {
            let Some(sync) = self.sync.clone() else {
                return Ok(false);
            };
            if self.discards_queued_audio() {
                return Ok(false);
            }
            if sync.is_playing() {
                self.resync(&sync, device, unwritten_ns)?;
                return Ok(false);
            }
            let paused_total_ns = sync.paused_total_ns();
            device.pause()?;
            let stopped_at_ns = sync.now_ns() + device.still_playing_after_pause_ns()?;
            while !sync.is_playing() && !self.discards_queued_audio() {
                self.wait_for_command(rx, COMMAND_CHECK_INTERVAL);
            }
            if sync.is_stopped() {
                return Ok(true);
            }
            // the audio played after the pipeline paused is made up by staying stopped that much longer
            let resume_at_ns =
                stopped_at_ns + sync.paused_total_ns().saturating_sub(paused_total_ns);
            while !self.discards_queued_audio() {
                let now_ns = sync.now_ns();
                if now_ns >= resume_at_ns {
                    break;
                }
                let wait = Duration::from_nanos(resume_at_ns - now_ns);
                self.wait_for_command(rx, wait.min(COMMAND_CHECK_INTERVAL));
            }
            device.resume()?;
            self.resync_at_ns = Some(sync.now_ns() + RESYNC_SETTLE_NS);
            Ok(true)
        }

        fn resync(
            &mut self,
            sync: &ClockSync,
            device: &impl PausableDevice,
            unwritten_ns: u64,
        ) -> Result<(), G2gError> {
            let (Some(resync_at_ns), Some(timeline)) = (self.resync_at_ns, self.timeline) else {
                return Ok(());
            };
            let now_ns = sync.now_ns();
            if now_ns < resync_at_ns {
                return Ok(());
            }
            self.resync_at_ns = None;
            let playing_pts_ns = self.playing_pts_ns(device, unwritten_ns)?;
            let late_ns = i128::from(now_ns) - playing_pts_ns - timeline.offset_now_ns(sync);
            if late_ns.abs() > RESYNC_TOLERANCE_NS {
                self.correction_ns = late_ns;
            }
            Ok(())
        }

        fn playing_pts_ns(
            &self,
            device: &impl PausableDevice,
            unwritten_ns: u64,
        ) -> Result<i128, G2gError> {
            let not_yet_played_ns = unwritten_ns + device.delay_ns()?;
            Ok(i128::from(self.handed_out_end_pts_ns) - i128::from(not_yet_played_ns))
        }

        pub(crate) fn publish_played_position(
            &self,
            device: &impl PausableDevice,
            unwritten_ns: u64,
        ) -> Result<(), G2gError> {
            let Some(timeline) = self.timeline else {
                return Ok(());
            };
            if self.discards_queued_audio() {
                return Ok(());
            }
            let Ok(playing_pts_ns) = u64::try_from(self.playing_pts_ns(device, unwritten_ns)?)
            else {
                return Ok(());
            };
            self.played_position
                .set_at_pts(timeline.segment.as_ref(), playing_pts_ns);
            Ok(())
        }

        // a pending flush, a closed queue or a stopped pipeline means the rest of the buffer being played must not play
        pub(crate) fn discards_queued_audio(&self) -> bool {
            self.pending_flushes.pending()
                || self.disconnected
                || self.sync.as_ref().is_some_and(ClockSync::is_stopped)
        }

        fn wait_for_command(&mut self, rx: &Receiver<DeviceCmd>, timeout: Duration) {
            match rx.recv_timeout(timeout) {
                Ok(cmd) => self.backlog.push_back(cmd),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => self.disconnected = true,
            }
        }

        fn admit(&mut self, cmd: DeviceCmd, rx: &Receiver<DeviceCmd>) -> Option<DeviceStep> {
            match cmd {
                // samples queued ahead of a flush belong to the position before a seek
                DeviceCmd::Samples(_) | DeviceCmd::StartAt(_) if self.pending_flushes.pending() => {
                    None
                }
                DeviceCmd::Flush => {
                    self.pending_flushes.0.fetch_sub(1, Ordering::AcqRel);
                    self.start = None;
                    self.timeline = None;
                    self.resync_at_ns = None;
                    self.correction_ns = 0;
                    Some(DeviceStep::Flush)
                }
                DeviceCmd::StartAt(start) => {
                    self.sync = Some(start.sync.clone());
                    self.start = Some(start);
                    None
                }
                DeviceCmd::Samples(frame) => {
                    let frame = self.start_playing(frame, rx)?;
                    let frame = self.correct(frame)?;
                    if let Some(timeline) = self.timeline {
                        self.handed_out_end_pts_ns =
                            frame.timing.pts_ns + timeline.geometry.duration_ns(&frame);
                    }
                    Some(DeviceStep::Samples(frame))
                }
                DeviceCmd::SwitchDevice(device) => Some(DeviceStep::SwitchDevice(device)),
                DeviceCmd::Shutdown => Some(DeviceStep::Shutdown),
            }
        }

        fn start_playing(&mut self, frame: Frame, rx: &Receiver<DeviceCmd>) -> Option<Frame> {
            let Some(start) = self.start.take() else {
                return Some(frame);
            };
            loop {
                if self.discards_queued_audio() {
                    return None;
                }
                let now_ns = start.sync.now_ns();
                let left =
                    Duration::from_nanos(start.deadline_after_pauses_ns().saturating_sub(now_ns));
                if left.is_zero() && start.sync.is_playing() {
                    break;
                }
                let wait = match left.is_zero() {
                    true => COMMAND_CHECK_INTERVAL,
                    false => left.min(COMMAND_CHECK_INTERVAL),
                };
                self.wait_for_command(rx, wait);
            }
            let deadline_ns = start.deadline_after_pauses_ns();
            let now_ns = start.sync.now_ns();
            self.timeline = Some(Timeline {
                offset_ns: i128::from(start.deadline_ns) - i128::from(frame.timing.pts_ns),
                paused_total_ns: start.paused_total_ns,
                segment: start.segment,
                geometry: start.geometry,
            });
            self.resync_at_ns = Some(now_ns + RESYNC_SETTLE_NS);
            let late_ns = now_ns.saturating_sub(deadline_ns);
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

        fn correct(&mut self, frame: Frame) -> Option<Frame> {
            let Some(timeline) = self.timeline else {
                return Some(frame);
            };
            let correction_ns = self.correction_ns;
            if correction_ns < 0 {
                self.correction_ns = 0;
                return timeline
                    .geometry
                    .prepend_silence(frame, correction_ns.unsigned_abs() as u64);
            }
            let duration_ns = timeline.geometry.duration_ns(&frame);
            let trimmed = timeline.geometry.trim_front(frame, correction_ns as u64);
            self.correction_ns = match trimmed {
                Some(_) => 0,
                None => correction_ns - i128::from(duration_ns),
            };
            trimmed
        }
    }

    struct NoDevice;

    impl PausableDevice for NoDevice {
        fn pause(&mut self) -> Result<(), G2gError> {
            Ok(())
        }

        fn resume(&mut self) -> Result<(), G2gError> {
            Ok(())
        }

        fn still_playing_after_pause_ns(&self) -> Result<u64, G2gError> {
            Ok(0)
        }

        fn delay_ns(&self) -> Result<u64, G2gError> {
            Ok(0)
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
        mut gate: PlayoutGate,
        fallback: DeviceFallback,
        ready: SyncSender<Result<(), G2gError>>,
        mut open: impl FnMut(&str) -> Result<Device, G2gError>,
        mut play: impl FnMut(Device, &Receiver<DeviceCmd>, &mut PlayoutGate) -> PlayEnd,
    ) {
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
            match gate.recv(rx) {
                Ok(None | Some(DeviceStep::Flush)) => {}
                Ok(Some(DeviceStep::Samples(frame))) => {
                    let bytes = (fallback.payload_len)(&frame)? as u64;
                    let unconsumed_ns = (u128::from(bytes) * 1_000_000_000
                        / u128::from(fallback.bytes_per_second))
                        as u64;
                    gate.publish_played_position(&NoDevice, unconsumed_ns)
                        .ok()?;
                    if gate
                        .pause_or_resync(rx, &mut NoDevice, unconsumed_ns)
                        .ok()?
                    {
                        first_sample_at = None;
                        consumed_bytes = 0;
                    }
                    if gate.discards_queued_audio() {
                        continue;
                    }
                    let started = *first_sample_at.get_or_insert_with(Instant::now);
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

#[cfg(test)]
mod tests {
    use super::device_worker::*;
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
                PlayoutGate::new(PendingFlushes::default(), PlayedPosition::default()),
                fallback,
                ready_tx,
                |device| match device.starts_with("missing") {
                    true => Err(G2gError::Hardware(g2g_core::HardwareError::Alsa(-2))),
                    false => Ok(String::from(device)),
                },
                |device, rx, gate| loop {
                    match gate.recv(rx) {
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
                PlayoutGate::new(pending_flushes, PlayedPosition::default()),
                fallback,
                ready_tx,
                |device| {
                    open.recv().unwrap();
                    Ok(String::from(device))
                },
                |_, rx, gate| loop {
                    let event = match gate.recv(rx) {
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

    fn stereo_s16_caps() -> g2g_core::Caps {
        g2g_core::Caps::Audio {
            format: g2g_core::AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: STEREO_S16_RATE,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    }

    fn stereo_s16() -> PcmGeometry {
        PcmGeometry::of(&stereo_s16_caps()).unwrap()
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
    fn a_buffer_starting_part_of_a_sample_before_the_segment_waits_for_its_deadline() {
        // a seek target between two samples at 48 kHz
        const SEGMENT_START_NS: u64 = 1_000_010_000;
        const BUFFER_BYTES: usize = 4_800;
        let clock: Arc<dyn g2g_core::PipelineClock + Send + Sync> =
            Arc::new(g2g_core::MonotonicClock);
        let anchor = g2g_core::clock::PlayAnchor::new();
        let mut start = PlayoutStart::default();
        start.set_clock_sync(g2g_core::ClockSync::with_play_anchor(
            Arc::clone(&clock),
            0,
            anchor.clone(),
        ));
        start.set_segment(g2g_core::Segment::for_flush_seek(
            &g2g_core::Seek::flush_to(SEGMENT_START_NS),
            None,
        ));
        let (open_tx, open_rx) = sync_channel::<()>(1);
        open_tx.send(()).unwrap();
        let (mut link, log) = spawn_recording_worker(PendingFlushes::default(), open_rx);
        let one_sample_ns = buffer_ns(STEREO_S16_BYTES_PER_FRAME);
        let frame = pcm_frame(SEGMENT_START_NS - one_sample_ns / 2, BUFFER_BYTES);
        g2g_core::runtime::block_on(async {
            start.send(&link, frame, stereo_s16(), err()).await.unwrap();
            anchor.stamp(clock.now_ns());
            let next = pcm_frame(SEGMENT_START_NS + buffer_ns(BUFFER_BYTES), BUFFER_BYTES);
            start.send(&link, next, stereo_s16(), err()).await.unwrap();
            link.finish(DeviceCmd::Shutdown, err()).await.unwrap();
        });

        assert_eq!(
            events(&log)[0],
            DeviceEvent::Samples {
                pts_ns: SEGMENT_START_NS,
                len: BUFFER_BYTES
            },
            "the buffer played without a deadline"
        );
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

    #[derive(Debug, PartialEq)]
    enum PauseEvent {
        Samples { pts_ns: u64, len: usize },
        Pause,
        Resume { at_ns: u64 },
    }

    // the device boundary: logs pauses, resumes and buffers, and reports the amounts the test sets
    struct RecordingDevice {
        log: Arc<Mutex<Vec<PauseEvent>>>,
        clock: crate::testutil::ManualClock,
        left_to_play_ns: u64,
        delay_ns: Arc<AtomicU64>,
    }

    impl PausableDevice for RecordingDevice {
        fn pause(&mut self) -> Result<(), G2gError> {
            self.log.lock().unwrap().push(PauseEvent::Pause);
            Ok(())
        }

        fn resume(&mut self) -> Result<(), G2gError> {
            let at_ns = g2g_core::PipelineClock::now_ns(&self.clock);
            self.log.lock().unwrap().push(PauseEvent::Resume { at_ns });
            Ok(())
        }

        fn still_playing_after_pause_ns(&self) -> Result<u64, G2gError> {
            Ok(self.left_to_play_ns)
        }

        fn delay_ns(&self) -> Result<u64, G2gError> {
            Ok(self.delay_ns.load(Ordering::SeqCst))
        }
    }

    fn spawn_pausing_worker(
        mut device: RecordingDevice,
        pending_flushes: PendingFlushes,
    ) -> WorkerLink<DeviceCmd> {
        let output = OutputDevice::new("device", "first");
        let fallback = output.fallback(BYTES_PER_SECOND, Arc::new(AtomicU64::new(0)), 1, |frame| {
            frame.domain.as_system_slice().map(<[u8]>::len)
        });
        let (ready_tx, _ready_rx) = sync_channel(1);
        WorkerLink::spawn("test-audio", move |rx| {
            run_device_worker(
                String::from("first"),
                rx,
                PlayoutGate::new(pending_flushes, PlayedPosition::default()),
                fallback,
                ready_tx,
                |_| Ok(()),
                |(), rx, gate| loop {
                    match gate.recv(rx) {
                        Ok(Some(DeviceStep::Samples(frame))) => {
                            gate.pause_or_resync(rx, &mut device, 0).unwrap();
                            let pts_ns = frame.timing.pts_ns;
                            let len = frame.domain.as_system_slice().unwrap().len();
                            let event = PauseEvent::Samples { pts_ns, len };
                            device.log.lock().unwrap().push(event);
                        }
                        Ok(None | Some(DeviceStep::Flush)) => {}
                        Ok(Some(DeviceStep::SwitchDevice(next))) => {
                            return PlayEnd::SwitchDevice(next)
                        }
                        Ok(Some(DeviceStep::Shutdown)) | Err(_) => return PlayEnd::Shutdown,
                    }
                },
            );
        })
        .unwrap()
    }

    fn playing_start(
        manual: &crate::testutil::ManualClock,
        anchor: &g2g_core::clock::PlayAnchor,
    ) -> PlayoutStart {
        let mut start = PlayoutStart::default();
        start.set_clock_sync(g2g_core::ClockSync::with_play_anchor(
            Arc::new(crate::testutil::ManualClock(Arc::clone(&manual.0))),
            0,
            anchor.clone(),
        ));
        start.set_segment(g2g_core::Segment::new());
        start
    }

    fn wait_for(log: &Mutex<Vec<PauseEvent>>, len: usize) {
        let started = Instant::now();
        while log.lock().unwrap().len() < len {
            assert!(started.elapsed() < Duration::from_secs(5), "{log:?}");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn buffer_ns(bytes: usize) -> u64 {
        crate::audioconvert::samples_to_ns(
            (bytes / STEREO_S16_BYTES_PER_FRAME) as u64,
            STEREO_S16_RATE,
        )
    }

    #[test]
    fn a_pause_stops_the_device_until_play_plus_the_audio_it_played_past_the_pause() {
        const BASE_NS: u64 = 1_000_000_000;
        const PAUSED_AT_NS: u64 = BASE_NS + 100_000_000;
        const PAUSE_NS: u64 = 2_000_000_000;
        const LEFT_TO_PLAY_NS: u64 = 30_000_000;
        const BUFFER_BYTES: usize = 4_800;
        let manual = crate::testutil::ManualClock::default();
        manual.set(BASE_NS);
        let anchor = g2g_core::clock::PlayAnchor::new();
        anchor.stamp(BASE_NS);
        let mut start = playing_start(&manual, &anchor);
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut link = spawn_pausing_worker(
            RecordingDevice {
                log: Arc::clone(&log),
                clock: crate::testutil::ManualClock(Arc::clone(&manual.0)),
                left_to_play_ns: LEFT_TO_PLAY_NS,
                delay_ns: Arc::default(),
            },
            PendingFlushes::default(),
        );
        let geometry = stereo_s16();
        let next_pts = buffer_ns(BUFFER_BYTES);
        g2g_core::runtime::block_on(async {
            let first = pcm_frame(0, BUFFER_BYTES);
            start.send(&link, first, geometry, err()).await.unwrap();
            wait_for(&log, 1);

            manual.set(PAUSED_AT_NS);
            anchor.pause(PAUSED_AT_NS);
            let next = pcm_frame(next_pts, BUFFER_BYTES);
            start.send(&link, next, geometry, err()).await.unwrap();
            wait_for(&log, 2);

            let played_at_ns = PAUSED_AT_NS + PAUSE_NS;
            manual.set(played_at_ns);
            anchor.resume(played_at_ns);
            std::thread::sleep(Duration::from_millis(50));
            assert_eq!(
                log.lock().unwrap().len(),
                2,
                "resumed before the audio played past the pause was made up"
            );
            manual.set(played_at_ns + LEFT_TO_PLAY_NS);
            link.finish(DeviceCmd::Shutdown, err()).await.unwrap();
        });

        assert_eq!(
            *log.lock().unwrap(),
            [
                PauseEvent::Samples {
                    pts_ns: 0,
                    len: BUFFER_BYTES
                },
                PauseEvent::Pause,
                PauseEvent::Resume {
                    at_ns: PAUSED_AT_NS + PAUSE_NS + LEFT_TO_PLAY_NS
                },
                PauseEvent::Samples {
                    pts_ns: next_pts,
                    len: BUFFER_BYTES
                },
            ]
        );
    }

    #[test]
    fn audio_measured_late_against_the_clock_is_dropped_from_the_next_buffer() {
        const BASE_NS: u64 = 1_000_000_000;
        const BUFFER_BYTES: usize = 96_000;
        const LATE_NS: u64 = 30_000_000;
        const MEASURED_AFTER_NS: u64 = 150_000_000;
        let manual = crate::testutil::ManualClock::default();
        manual.set(BASE_NS);
        let anchor = g2g_core::clock::PlayAnchor::new();
        anchor.stamp(BASE_NS);
        let mut start = playing_start(&manual, &anchor);
        let log = Arc::new(Mutex::new(Vec::new()));
        let delay_ns = Arc::new(AtomicU64::new(0));
        let mut link = spawn_pausing_worker(
            RecordingDevice {
                log: Arc::clone(&log),
                clock: crate::testutil::ManualClock(Arc::clone(&manual.0)),
                left_to_play_ns: 0,
                delay_ns: Arc::clone(&delay_ns),
            },
            PendingFlushes::default(),
        );
        let geometry = stereo_s16();
        let buffer = buffer_ns(BUFFER_BYTES);
        g2g_core::runtime::block_on(async {
            start
                .send(&link, pcm_frame(0, BUFFER_BYTES), geometry, err())
                .await
                .unwrap();
            wait_for(&log, 1);

            // two buffers handed out, and the device still has to play everything after the late position
            manual.set(BASE_NS + MEASURED_AFTER_NS);
            let playing_pts = MEASURED_AFTER_NS - LATE_NS;
            delay_ns.store(2 * buffer - playing_pts, Ordering::SeqCst);
            start
                .send(&link, pcm_frame(buffer, BUFFER_BYTES), geometry, err())
                .await
                .unwrap();
            start
                .send(&link, pcm_frame(2 * buffer, BUFFER_BYTES), geometry, err())
                .await
                .unwrap();
            link.finish(DeviceCmd::Shutdown, err()).await.unwrap();
        });

        let late_bytes = crate::audioconvert::ns_to_samples(LATE_NS, STEREO_S16_RATE) as usize
            * STEREO_S16_BYTES_PER_FRAME;
        assert_eq!(
            log.lock().unwrap()[2],
            PauseEvent::Samples {
                pts_ns: 2 * buffer + LATE_NS,
                len: BUFFER_BYTES - late_bytes
            }
        );
    }

    // the device boundary: reports a fixed amount still to play and plays nothing
    #[cfg(not(target_os = "windows"))]
    struct DelayedDevice {
        delay_ns: u64,
    }

    #[cfg(not(target_os = "windows"))]
    impl PausableDevice for DelayedDevice {
        fn pause(&mut self) -> Result<(), G2gError> {
            Ok(())
        }

        fn resume(&mut self) -> Result<(), G2gError> {
            Ok(())
        }

        fn still_playing_after_pause_ns(&self) -> Result<u64, G2gError> {
            Ok(0)
        }

        fn delay_ns(&self) -> Result<u64, G2gError> {
            Ok(self.delay_ns)
        }
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn an_adopted_position_is_the_sample_the_device_plays_now() {
        const BASE_NS: u64 = 1_000_000_000;
        // ten milliseconds, inside one of the slices a pause is checked between
        const BUFFER_BYTES: usize = 1_920;
        const DEVICE_DELAY_NS: u64 = 4_000_000;
        let manual = crate::testutil::ManualClock::default();
        manual.set(BASE_NS);
        let anchor = g2g_core::clock::PlayAnchor::new();
        anchor.stamp(BASE_NS);
        let mut link = PlayoutLink::new(err());
        link.set_clock_sync(g2g_core::ClockSync::with_play_anchor(
            Arc::new(crate::testutil::ManualClock(Arc::clone(&manual.0))),
            0,
            anchor.clone(),
        ));
        link.set_segment(g2g_core::Segment::new());
        let progress = g2g_core::progress::PipelineProgress::new();
        assert!(link.adopt_position(progress.sink_position()).is_none());
        let mut output = OutputDevice::new("device", "first");
        let fallback = output.fallback(BYTES_PER_SECOND, Arc::default(), 1, |frame| {
            frame.domain.as_system_slice().map(<[u8]>::len)
        });
        let geometry = stereo_s16();
        link.open(
            "test-audio",
            &stereo_s16_caps(),
            &mut output,
            fallback,
            |_| {
                Ok(DelayedDevice {
                    delay_ns: DEVICE_DELAY_NS,
                })
            },
            move |mut device, rx, gate| loop {
                match gate.recv(rx) {
                    Ok(Some(DeviceStep::Samples(frame))) => {
                        let pcm = frame.domain.as_system_slice().unwrap().to_vec();
                        gate.play_in_slices(rx, &pcm, geometry, &mut device, |_, _| Ok(()))
                            .unwrap();
                    }
                    Ok(None | Some(DeviceStep::Flush)) => {}
                    Ok(Some(DeviceStep::SwitchDevice(next))) => return PlayEnd::SwitchDevice(next),
                    Ok(Some(DeviceStep::Shutdown)) | Err(_) => return PlayEnd::Shutdown,
                }
            },
        )
        .unwrap();
        let buffer = buffer_ns(BUFFER_BYTES);
        g2g_core::runtime::block_on(async {
            for pts_ns in [0, buffer] {
                let frame = pcm_frame(pts_ns, BUFFER_BYTES);
                link.send(frame, &mut output).await.unwrap();
            }
            link.finish().await.unwrap();
        });

        // before the second buffer is written, the device holds the delay and the second buffer is still to come
        assert_eq!(progress.position(), Some(buffer - DEVICE_DELAY_NS));
    }

    #[test]
    fn a_flush_after_the_end_of_the_stream_plays_on_a_device_opened_again() {
        const BUFFER_BYTES: usize = 1_920;
        let opens = Arc::new(AtomicU64::new(0));
        let played = Arc::new(Mutex::new(Vec::new()));
        let mut link = PlayoutLink::new(err());
        let mut output = OutputDevice::new("device", "first");
        let fallback = output.fallback(BYTES_PER_SECOND, Arc::default(), 1, |frame| {
            frame.domain.as_system_slice().map(<[u8]>::len)
        });
        let opened = Arc::clone(&opens);
        let log = Arc::clone(&played);
        link.open(
            "test-audio",
            &stereo_s16_caps(),
            &mut output,
            fallback,
            move |_| {
                opened.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
            move |(), rx, gate| loop {
                match gate.recv(rx) {
                    Ok(Some(DeviceStep::Samples(frame))) => {
                        log.lock().unwrap().push(frame.timing.pts_ns)
                    }
                    Ok(None | Some(DeviceStep::Flush)) => {}
                    Ok(Some(DeviceStep::SwitchDevice(next))) => return PlayEnd::SwitchDevice(next),
                    Ok(Some(DeviceStep::Shutdown)) | Err(_) => return PlayEnd::Shutdown,
                }
            },
        )
        .unwrap();
        let seek_target = buffer_ns(BUFFER_BYTES);
        g2g_core::runtime::block_on(async {
            link.send(pcm_frame(0, BUFFER_BYTES), &mut output)
                .await
                .unwrap();
            link.finish().await.unwrap();
            link.flush().await.unwrap();
            link.send(pcm_frame(seek_target, BUFFER_BYTES), &mut output)
                .await
                .unwrap();
            link.finish().await.unwrap();
        });

        assert_eq!(*played.lock().unwrap(), [0, seek_target]);
        assert_eq!(opens.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn a_pipeline_stopped_while_paused_ends_playout() {
        const BASE_NS: u64 = 1_000_000_000;
        const PAUSED_AT_NS: u64 = BASE_NS + 100_000_000;
        const BUFFER_BYTES: usize = 4_800;
        const FINISH_TIMEOUT: Duration = Duration::from_secs(5);
        let manual = crate::testutil::ManualClock::default();
        manual.set(BASE_NS);
        let anchor = g2g_core::clock::PlayAnchor::new();
        anchor.stamp(BASE_NS);
        let mut start = playing_start(&manual, &anchor);
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut link = spawn_pausing_worker(
            RecordingDevice {
                log: Arc::clone(&log),
                clock: crate::testutil::ManualClock(Arc::clone(&manual.0)),
                left_to_play_ns: 0,
                delay_ns: Arc::default(),
            },
            PendingFlushes::default(),
        );
        let geometry = stereo_s16();
        g2g_core::runtime::block_on(async {
            start
                .send(&link, pcm_frame(0, BUFFER_BYTES), geometry, err())
                .await
                .unwrap();
            wait_for(&log, 1);
            manual.set(PAUSED_AT_NS);
            anchor.pause(PAUSED_AT_NS);
            let next = pcm_frame(buffer_ns(BUFFER_BYTES), BUFFER_BYTES);
            start.send(&link, next, geometry, err()).await.unwrap();
            wait_for(&log, 2);
        });
        anchor.clear();
        let (finished_tx, finished_rx) = sync_channel(1);
        std::thread::spawn(move || {
            g2g_core::runtime::block_on(link.finish(DeviceCmd::Shutdown, err())).unwrap();
            finished_tx.send(()).unwrap();
        });

        finished_rx
            .recv_timeout(FINISH_TIMEOUT)
            .expect("playout ended once the pipeline stopped");
        assert!(!log
            .lock()
            .unwrap()
            .iter()
            .any(|event| matches!(event, PauseEvent::Resume { .. })));
    }

    #[test]
    fn a_seek_while_paused_plays_on_at_the_next_play() {
        const BASE_NS: u64 = 1_000_000_000;
        const PAUSED_AT_NS: u64 = BASE_NS + 100_000_000;
        const PLAYED_AT_NS: u64 = PAUSED_AT_NS + 1_000_000_000;
        const BUFFER_BYTES: usize = 4_800;
        let manual = crate::testutil::ManualClock::default();
        manual.set(BASE_NS);
        let anchor = g2g_core::clock::PlayAnchor::new();
        anchor.stamp(BASE_NS);
        let mut start = playing_start(&manual, &anchor);
        let log = Arc::new(Mutex::new(Vec::new()));
        let pending_flushes = PendingFlushes::default();
        let mut link = spawn_pausing_worker(
            RecordingDevice {
                log: Arc::clone(&log),
                clock: crate::testutil::ManualClock(Arc::clone(&manual.0)),
                left_to_play_ns: 0,
                delay_ns: Arc::default(),
            },
            pending_flushes.clone(),
        );
        let geometry = stereo_s16();
        let second_pts = buffer_ns(BUFFER_BYTES);
        g2g_core::runtime::block_on(async {
            start
                .send(&link, pcm_frame(0, BUFFER_BYTES), geometry, err())
                .await
                .unwrap();
            wait_for(&log, 1);
            manual.set(PAUSED_AT_NS);
            anchor.pause(PAUSED_AT_NS);
            start
                .send(&link, pcm_frame(second_pts, BUFFER_BYTES), geometry, err())
                .await
                .unwrap();
            wait_for(&log, 2);

            anchor.restart();
            // the seek reaches the sink after the paused worker has seen the anchor restart
            std::thread::sleep(Duration::from_millis(50));
            start.flush(&link, &pending_flushes, err()).await.unwrap();
            start
                .send(&link, pcm_frame(0, BUFFER_BYTES), geometry, err())
                .await
                .unwrap();
            manual.set(PLAYED_AT_NS);
            anchor.stamp(PLAYED_AT_NS);
            start
                .send(&link, pcm_frame(second_pts, BUFFER_BYTES), geometry, err())
                .await
                .unwrap();
            link.finish(DeviceCmd::Shutdown, err()).await.unwrap();
        });

        let log = log.lock().unwrap();
        let after_the_seek = &log[log.len().saturating_sub(2)..];
        assert_eq!(
            after_the_seek,
            [
                PauseEvent::Samples {
                    pts_ns: 0,
                    len: BUFFER_BYTES
                },
                PauseEvent::Samples {
                    pts_ns: second_pts,
                    len: BUFFER_BYTES
                },
            ],
            "{log:?}"
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
