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
pub(crate) use device_worker::{run_device_worker, DeviceCmd, PlayEnd};

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

    use g2g_core::bus::BusHandle;
    use g2g_core::frame::Frame;
    use g2g_core::G2gError;

    use super::{report_on, OutputDevice};

    pub(crate) enum DeviceCmd {
        Samples(Frame),
        SwitchDevice(String),
        Shutdown,
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
        fallback: DeviceFallback,
        ready: SyncSender<Result<(), G2gError>>,
        mut open: impl FnMut(&str) -> Result<Device, G2gError>,
        mut play: impl FnMut(Device, &Receiver<DeviceCmd>) -> PlayEnd,
    ) {
        let mut ready = Some(ready);
        let mut device = first_device;
        loop {
            let end = match open(&device) {
                Ok(opened) => {
                    if let Some(ready) = ready.take() {
                        let _ = ready.send(Ok(()));
                    }
                    play(opened, &rx)
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
            match play_without_device(&rx, &fallback) {
                Some(next) => device = next,
                None => return,
            }
        }
    }

    fn play_without_device(rx: &Receiver<DeviceCmd>, fallback: &DeviceFallback) -> Option<String> {
        let mut first_sample_at: Option<Instant> = None;
        let mut consumed_bytes = 0u64;
        loop {
            match rx.recv() {
                Ok(DeviceCmd::Samples(frame)) => {
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
                Ok(DeviceCmd::SwitchDevice(next)) => return Some(next),
                Ok(DeviceCmd::Shutdown) | Err(_) => return None,
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
                fallback,
                ready_tx,
                |device| match device.starts_with("missing") {
                    true => Err(G2gError::Hardware(g2g_core::HardwareError::Alsa(-2))),
                    false => Ok(String::from(device)),
                },
                |device, rx| loop {
                    match rx.recv() {
                        Ok(DeviceCmd::Samples(_)) => played.lock().unwrap().push(device.clone()),
                        Ok(DeviceCmd::SwitchDevice(next)) => return PlayEnd::SwitchDevice(next),
                        Ok(DeviceCmd::Shutdown) | Err(_) => return PlayEnd::Shutdown,
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
