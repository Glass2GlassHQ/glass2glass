//! Device discovery (M938): the `GstDeviceProvider` / `GstDeviceMonitor`
//! analog. A [`DeviceProvider`] probes one backend (v4l2, ALSA, PipeWire,
//! GPU adapters) for the capture / render / compute devices it can see; a
//! [`DeviceMonitor`] aggregates providers, filters by device class and caps,
//! and (started) watches for hotplug, emitting [`DeviceEvent`]s on its own
//! channel (GStreamer's monitor posts on a private bus for the same reason:
//! a monitor is application-side, not part of a running pipeline).
//!
//! A [`Device`] does not own an element factory: it names the launch element
//! plus the `key=value` properties that select it, so construction rides the
//! same [`Registry`] + [`PropertySpec::parse_value`] path as `parse_launch`,
//! and [`Device::launch_fragment`] drops straight into a text pipeline.
//!
//! Class strings are `/`-separated, GStreamer-style: `Video/Source`,
//! `Audio/Sink`, `Video/Source/Network`, and the g2g extension `Compute/GPU`.
//! A filter matches when every one of its parts appears among the device's
//! parts (case-insensitive), so `Source` matches both `Video/Source` and
//! `Audio/Source`.

extern crate std;

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::time::Duration;
use std::sync::{Condvar, Mutex};
use std::thread::JoinHandle;

use crate::caps::{Caps, CapsSet};
use crate::element::DynAsyncElement;
use crate::error::G2gError;
use crate::property::PropError;

use super::channel::{bounded, Receiver, Sender};
use super::{DynSourceLoop, Registry};

/// One discovered device: what to show the user, how to filter it, and how to
/// build the element that uses it.
#[derive(Debug, Clone, PartialEq)]
pub struct Device {
    /// Human-readable name, e.g. `"Integrated Camera: Integrated C"`.
    pub display_name: String,
    /// `/`-separated class, e.g. `"Video/Source"`; see the module docs.
    pub klass: String,
    /// Identity stable across replug / reboot where the backend offers one
    /// (USB bus info, ALSA card longname, PipeWire `object.serial` is NOT
    /// stable, so providers prefer `node.name`). Falls back to the volatile
    /// handle when nothing better exists.
    pub persistent_id: String,
    /// The formats the device was probed to produce / accept, highest
    /// preference first.
    pub caps: CapsSet,
    /// Launch name of the element that drives this device, e.g. `"v4l2src"`.
    pub element: &'static str,
    /// Properties selecting this device on that element, textual `key=value`
    /// pairs parsed through the element's own [`PropertySpec`]s.
    pub props: Vec<(String, String)>,
    /// Backend detail for display only (bus path, driver, sample formats),
    /// never fed to the element.
    pub detail: Vec<(String, String)>,
    /// Name of the provider that found this device.
    pub provider: &'static str,
}

/// The element a [`Device::create`] built: sources have a different dyn trait
/// than transforms / sinks, mirroring the [`Registry`] factory split.
pub enum DeviceElement {
    /// A capture device (camera, microphone).
    Source(Box<dyn DynSourceLoop>),
    /// A render / transform device (speaker, display, decoder).
    Element(Box<dyn DynAsyncElement>),
}

impl core::fmt::Debug for DeviceElement {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DeviceElement::Source(_) => f.write_str("DeviceElement::Source"),
            DeviceElement::Element(_) => f.write_str("DeviceElement::Element"),
        }
    }
}

impl Device {
    /// Whether this device carries every class part of `filter`
    /// (case-insensitive), GStreamer's `gst_device_has_classes` semantics:
    /// `"Video/Source"` requires both parts, `"Source"` matches any source.
    pub fn has_classes(&self, filter: &str) -> bool {
        filter
            .split('/')
            .filter(|part| !part.is_empty())
            .all(|part| {
                self.klass
                    .split('/')
                    .any(|have| have.eq_ignore_ascii_case(part))
            })
    }

    /// Whether any probed caps alternative intersects `caps`.
    pub fn caps_overlap(&self, caps: &Caps) -> bool {
        self.caps
            .alternatives()
            .iter()
            .any(|c| c.intersect(caps).is_ok())
    }

    /// The `gst-launch` text fragment that builds this device's element,
    /// e.g. `v4l2src device=/dev/video0`. Values with whitespace are quoted.
    pub fn launch_fragment(&self) -> String {
        let mut out = String::from(self.element);
        for (key, value) in &self.props {
            out.push(' ');
            out.push_str(key);
            out.push('=');
            if value.chars().any(char::is_whitespace) {
                out.push('"');
                out.push_str(value);
                out.push('"');
            } else {
                out.push_str(value);
            }
        }
        out
    }

    /// Build the element for this device from `registry` and apply
    /// [`props`](Self::props), each parsed through the element's declared
    /// [`PropertySpec`](crate::PropertySpec). `CapsMismatch` when the launch
    /// name is not registered; `ControlBinding` when a property is unknown to
    /// the element or its value does not parse (the provider and the element
    /// disagree, a bug on one side).
    pub fn create(&self, registry: &Registry) -> Result<DeviceElement, G2gError> {
        if let Some(mut src) = registry.make_source(self.element) {
            apply_props(src.properties(), &self.props, |name, value| {
                src.set_property(name, value)
            })?;
            return Ok(DeviceElement::Source(src));
        }
        if let Some(mut el) = registry.make_element(self.element) {
            apply_props(el.properties(), &self.props, |name, value| {
                el.set_property(name, value)
            })?;
            return Ok(DeviceElement::Element(el));
        }
        Err(G2gError::CapsMismatch)
    }
}

/// Parse each textual prop against `specs` and apply it via `set`.
fn apply_props(
    specs: &'static [crate::property::PropertySpec],
    props: &[(String, String)],
    mut set: impl FnMut(&str, crate::property::PropValue) -> Result<(), PropError>,
) -> Result<(), G2gError> {
    for (key, raw) in props {
        let spec = specs
            .iter()
            .find(|s| s.name == key.as_str())
            .ok_or(G2gError::ControlBinding)?;
        let value = spec
            .parse_value(raw)
            .map_err(|_| G2gError::ControlBinding)?;
        set(key, value).map_err(|_| G2gError::ControlBinding)?;
    }
    Ok(())
}

/// A hotplug change observed by a started [`DeviceMonitor`].
#[derive(Debug, Clone, PartialEq)]
pub enum DeviceEvent {
    /// A device appeared (also emitted once per device present at start, so a
    /// consumer needs no separate initial probe).
    Added(Device),
    /// A device disappeared, named by its provider + persistent id.
    Removed {
        /// [`Device::provider`] of the removed device.
        provider: &'static str,
        /// [`Device::persistent_id`] of the removed device.
        persistent_id: String,
    },
    /// A device's probed description changed in place (same persistent id).
    Changed(Device),
}

/// Filter-applying producer handle a provider's native watch posts through;
/// handed to [`DeviceProvider::watch`] by the monitor so provider events pass
/// the same class / caps filters as [`DeviceMonitor::probe`].
#[derive(Clone)]
pub struct DeviceSink {
    tx: Sender<DeviceEvent>,
    filters: Arc<Vec<DeviceFilter>>,
}

impl core::fmt::Debug for DeviceSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DeviceSink").finish_non_exhaustive()
    }
}

impl DeviceSink {
    /// Post one event, blocking until the consumer has room. `false` once the
    /// monitor is gone (the watch should shut down), or when the event's
    /// device is filtered out (not an error; keep watching).
    pub fn post(&self, event: DeviceEvent) -> bool {
        let passes = match &event {
            DeviceEvent::Added(d) | DeviceEvent::Changed(d) => matches_filters(&self.filters, d),
            DeviceEvent::Removed { .. } => true,
        };
        if !passes {
            return true;
        }
        super::blocking::block_on(self.tx.send(event)).is_ok()
    }
}

/// Keeps a provider's native watch alive; dropping it stops the watch (quits
/// the backend loop, joins its thread).
pub struct WatchGuard {
    stop: Option<Box<dyn FnOnce() + Send>>,
}

impl WatchGuard {
    /// Wrap the watch's stop action.
    pub fn new(stop: impl FnOnce() + Send + 'static) -> Self {
        Self {
            stop: Some(Box::new(stop)),
        }
    }
}

impl core::fmt::Debug for WatchGuard {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("WatchGuard")
    }
}

impl Drop for WatchGuard {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            stop();
        }
    }
}

/// One discovery backend. `Send` so the monitor can move it onto a poll
/// thread.
pub trait DeviceProvider: Send {
    /// Short backend name (`"v4l2"`, `"alsa"`, `"pipewire"`, `"gpu"`).
    fn name(&self) -> &'static str;

    /// Enumerate the devices this backend currently sees. Blocking (device
    /// opens, format ioctls); callers off the async path use
    /// [`run_blocking`](crate::runtime)-style offload.
    fn probe(&self) -> Result<Vec<Device>, G2gError>;

    /// Start a native event watch, posting the CURRENT device set as
    /// [`DeviceEvent::Added`] first and hotplug changes after. `Ok(None)`
    /// (the default) means the backend has no event source and the monitor
    /// falls back to poll-and-diff over [`probe`](Self::probe).
    fn watch(&self, sink: DeviceSink) -> Result<Option<WatchGuard>, G2gError> {
        let _ = sink;
        Ok(None)
    }
}

/// One monitor filter: a class string plus an optional caps constraint, OR-ed
/// with the other filters (`gst_device_monitor_add_filter` semantics).
#[derive(Debug, Clone)]
struct DeviceFilter {
    classes: String,
    caps: Option<Caps>,
}

fn matches_filters(filters: &[DeviceFilter], device: &Device) -> bool {
    if filters.is_empty() {
        return true;
    }
    filters.iter().any(|f| {
        device.has_classes(&f.classes) && f.caps.as_ref().is_none_or(|c| device.caps_overlap(c))
    })
}

/// Everything one [`DeviceMonitor::probe`] pass produced: the (filtered)
/// devices, plus per-provider errors so a dead backend (no PipeWire daemon)
/// degrades to a warning instead of failing the whole probe.
#[derive(Debug)]
pub struct ProbeOutcome {
    /// Devices that passed the monitor's filters.
    pub devices: Vec<Device>,
    /// Providers whose probe failed, with the failure.
    pub errors: Vec<(&'static str, G2gError)>,
}

/// Aggregates [`DeviceProvider`]s behind class / caps filters; the
/// `GstDeviceMonitor` analog. One-shot [`probe`](Self::probe), or
/// [`start`](Self::start) for hotplug.
pub struct DeviceMonitor {
    providers: Vec<Box<dyn DeviceProvider>>,
    filters: Vec<DeviceFilter>,
    poll_interval: Option<Duration>,
}

impl core::fmt::Debug for DeviceMonitor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DeviceMonitor")
            .field("providers", &self.providers.len())
            .field("filters", &self.filters)
            .field("poll_interval", &self.poll_interval)
            .finish()
    }
}

impl Default for DeviceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceMonitor {
    /// An empty monitor. Providers without a native event source are polled
    /// every 2 s once started; see [`set_poll_interval`](Self::set_poll_interval).
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
            filters: Vec::new(),
            poll_interval: Some(Duration::from_secs(2)),
        }
    }

    /// Register one provider, returning `&mut self` to chain calls.
    pub fn register(&mut self, provider: Box<dyn DeviceProvider>) -> &mut Self {
        self.providers.push(provider);
        self
    }

    /// Add one filter: `classes` (see [`Device::has_classes`]) plus an
    /// optional caps constraint. Filters OR; no filters means everything.
    pub fn add_filter(&mut self, classes: &str, caps: Option<Caps>) -> &mut Self {
        self.filters.push(DeviceFilter {
            classes: classes.to_string(),
            caps,
        });
        self
    }

    /// Poll cadence for providers without a native watch; `None` disables the
    /// fallback (those providers then report only their initial device set).
    pub fn set_poll_interval(&mut self, interval: Option<Duration>) -> &mut Self {
        self.poll_interval = interval;
        self
    }

    /// Probe every provider once, filtered. Blocking.
    pub fn probe(&self) -> ProbeOutcome {
        let mut devices = Vec::new();
        let mut errors = Vec::new();
        for provider in &self.providers {
            match provider.probe() {
                Ok(found) => devices.extend(
                    found
                        .into_iter()
                        .filter(|d| matches_filters(&self.filters, d)),
                ),
                Err(e) => errors.push((provider.name(), e)),
            }
        }
        ProbeOutcome { devices, errors }
    }

    /// Start watching: native watches where a provider has one, poll-and-diff
    /// threads elsewhere. Every present device arrives as an initial
    /// [`DeviceEvent::Added`]. The returned monitor stops on drop.
    pub fn start(self) -> RunningMonitor {
        let (tx, rx) = bounded::<DeviceEvent>(EVENT_CAPACITY);
        let filters = Arc::new(self.filters);
        let stop = Arc::new(StopFlag::default());
        let mut guards = Vec::new();
        let mut threads = Vec::new();
        let mut watch_errors = Vec::new();

        for provider in self.providers {
            let sink = DeviceSink {
                tx: tx.clone(),
                filters: filters.clone(),
            };
            match provider.watch(sink.clone()) {
                Ok(Some(guard)) => guards.push(guard),
                Ok(None) => {
                    threads.push(spawn_poll(provider, sink, self.poll_interval, stop.clone()))
                }
                Err(e) => {
                    watch_errors.push((provider.name(), e));
                    threads.push(spawn_poll(provider, sink, self.poll_interval, stop.clone()));
                }
            }
        }

        RunningMonitor {
            rx,
            stop,
            threads,
            _guards: guards,
            watch_errors,
        }
    }
}

/// Event backlog of a started monitor. Hotplug is rare; 64 absorbs the
/// initial burst of a large device list without unbounded growth.
const EVENT_CAPACITY: usize = 64;

#[derive(Debug, Default)]
struct StopFlag {
    stopped: Mutex<bool>,
    cv: Condvar,
}

impl StopFlag {
    fn stop(&self) {
        *self.stopped.lock().unwrap() = true;
        self.cv.notify_all();
    }

    /// Wait up to `timeout`; `true` once stopped.
    fn wait(&self, timeout: Duration) -> bool {
        let guard = self.stopped.lock().unwrap();
        if *guard {
            return true;
        }
        let (guard, _) = self.cv.wait_timeout(guard, timeout).unwrap();
        *guard
    }

    fn is_stopped(&self) -> bool {
        *self.stopped.lock().unwrap()
    }
}

/// Poll-and-diff watcher for a provider without native events: emit the
/// initial set, then re-probe on the interval and diff by persistent id.
fn spawn_poll(
    provider: Box<dyn DeviceProvider>,
    sink: DeviceSink,
    interval: Option<Duration>,
    stop: Arc<StopFlag>,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let mut known: Vec<Device> = provider.probe().unwrap_or_default();
        for device in &known {
            if !sink.post(DeviceEvent::Added(device.clone())) {
                return;
            }
        }
        let Some(interval) = interval else { return };
        loop {
            if stop.wait(interval) {
                return;
            }
            // A transient probe failure (backend restarting) keeps the last
            // known set rather than reporting every device removed.
            let Ok(now) = provider.probe() else { continue };
            for old in &known {
                if !now.iter().any(|d| d.persistent_id == old.persistent_id) {
                    let removed = DeviceEvent::Removed {
                        provider: old.provider,
                        persistent_id: old.persistent_id.clone(),
                    };
                    if !sink.post(removed) {
                        return;
                    }
                }
            }
            for device in &now {
                match known
                    .iter()
                    .find(|k| k.persistent_id == device.persistent_id)
                {
                    None => {
                        if !sink.post(DeviceEvent::Added(device.clone())) {
                            return;
                        }
                    }
                    Some(old) if old != device => {
                        if !sink.post(DeviceEvent::Changed(device.clone())) {
                            return;
                        }
                    }
                    Some(_) => {}
                }
            }
            known = now;
            if stop.is_stopped() {
                return;
            }
        }
    })
}

/// A started [`DeviceMonitor`]: drain [`DeviceEvent`]s from it; dropping (or
/// [`stop`](Self::stop)) ends every watch and joins the poll threads.
#[derive(Debug)]
pub struct RunningMonitor {
    rx: Receiver<DeviceEvent>,
    stop: Arc<StopFlag>,
    threads: Vec<JoinHandle<()>>,
    _guards: Vec<WatchGuard>,
    /// Providers whose native watch failed to start; each fell back to
    /// polling, so their devices still arrive.
    pub watch_errors: Vec<(&'static str, G2gError)>,
}

impl RunningMonitor {
    /// Non-blocking drain of one event; `None` when empty.
    pub fn try_recv(&self) -> Option<DeviceEvent> {
        self.rx.try_recv()
    }

    /// Await the next event; `None` once every watch has ended.
    pub async fn recv(&self) -> Option<DeviceEvent> {
        self.rx.recv().await
    }

    /// Stop watching and join the watcher threads.
    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        self.stop.stop();
        self._guards.clear();
        for handle in self.threads.drain(..) {
            let _ = handle.join();
        }
    }
}

impl Drop for RunningMonitor {
    fn drop(&mut self) {
        self.shutdown();
    }
}
