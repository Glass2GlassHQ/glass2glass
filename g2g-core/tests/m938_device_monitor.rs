//! M938: device discovery. The `Device` model (class / caps filtering, launch
//! fragment, registry construction), the `DeviceMonitor` one-shot probe with
//! per-provider error isolation, and the started monitor's hotplug paths:
//! poll-and-diff for providers without events, the filtered `DeviceSink` for
//! providers with a native watch, initial-`Added` semantics for both.
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use core::time::Duration;
use std::sync::{Arc, Mutex};

use g2g_core::runtime::{
    block_on, Device, DeviceElement, DeviceEvent, DeviceMonitor, DeviceProvider, DeviceSink,
    Registry, SourceFactory, SourceLoop, WatchGuard,
};
use g2g_core::{
    Caps, CapsSet, ConfigureOutcome, Dim, G2gError, OutputSink, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec, Rate, RawVideoFormat,
};

fn video_caps(width: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Yuyv,
        width: Dim::Fixed(width),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

fn camera(id: &str, name: &str) -> Device {
    Device {
        display_name: name.into(),
        klass: "Video/Source".into(),
        persistent_id: id.into(),
        caps: CapsSet::one(video_caps(640)),
        element: "mockcamsrc",
        props: vec![("device".into(), format!("/dev/{id}"))],
        detail: vec![("driver".into(), "mock".into())],
        provider: "mock",
    }
}

fn speaker(id: &str) -> Device {
    Device {
        display_name: "Mock Speaker".into(),
        klass: "Audio/Sink".into(),
        persistent_id: id.into(),
        caps: CapsSet::from_alternatives(vec![]),
        element: "mockspeakersink",
        props: vec![],
        detail: vec![],
        provider: "mock",
    }
}

/// Provider over a shared mutable device list, so hotplug tests mutate the
/// backend between poll rounds.
struct MockProvider {
    devices: Arc<Mutex<Vec<Device>>>,
    fail: bool,
}

impl DeviceProvider for MockProvider {
    fn name(&self) -> &'static str {
        "mock"
    }
    fn probe(&self) -> Result<Vec<Device>, G2gError> {
        if self.fail {
            return Err(G2gError::Shutdown);
        }
        Ok(self.devices.lock().unwrap().clone())
    }
}

#[test]
fn class_filter_matches_gstreamer_semantics() {
    let cam = camera("video0", "Cam");
    assert!(cam.has_classes("Video/Source"));
    assert!(cam.has_classes("Source"));
    assert!(cam.has_classes("video"));
    assert!(cam.has_classes("Source/Video"));
    assert!(!cam.has_classes("Audio/Source"));
    assert!(!cam.has_classes("Video/Sink"));
}

#[test]
fn caps_filter_uses_intersection() {
    let cam = camera("video0", "Cam");
    assert!(cam.caps_overlap(&video_caps(640)));
    // Same format, different fixed geometry: no intersection.
    assert!(!cam.caps_overlap(&video_caps(1920)));
}

#[test]
fn launch_fragment_quotes_whitespace() {
    let mut cam = camera("video0", "Cam");
    assert_eq!(cam.launch_fragment(), "mockcamsrc device=/dev/video0");
    cam.props.push(("label".into(), "front door".into()));
    assert_eq!(
        cam.launch_fragment(),
        "mockcamsrc device=/dev/video0 label=\"front door\""
    );
}

#[test]
fn probe_filters_and_isolates_provider_errors() {
    let mut monitor = DeviceMonitor::new();
    monitor
        .register(Box::new(MockProvider {
            devices: Arc::new(Mutex::new(vec![camera("video0", "Cam"), speaker("card0")])),
            fail: false,
        }))
        .register(Box::new(MockProvider {
            devices: Arc::new(Mutex::new(Vec::new())),
            fail: true,
        }))
        .add_filter("Video/Source", None);
    let outcome = monitor.probe();
    assert_eq!(outcome.devices.len(), 1);
    assert_eq!(outcome.devices[0].persistent_id, "video0");
    assert_eq!(outcome.errors, vec![("mock", G2gError::Shutdown)]);
}

#[test]
fn probe_caps_filter() {
    let mut monitor = DeviceMonitor::new();
    monitor
        .register(Box::new(MockProvider {
            devices: Arc::new(Mutex::new(vec![camera("video0", "Cam")])),
            fail: false,
        }))
        .add_filter("Video/Source", Some(video_caps(1920)));
    assert!(monitor.probe().devices.is_empty());
}

// --- Device::create through a real Registry --------------------------------

/// Records the `device` property a `Device::create` applies.
#[derive(Default)]
struct MockCamSrc {
    device: String,
}

const MOCKCAM_PROPS: &[PropertySpec] = &[PropertySpec::new("device", PropKind::Str, "device node")];

impl SourceLoop for MockCamSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(video_caps(640)))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::Eos).await?;
            Ok(0)
        })
    }
    fn properties(&self) -> &'static [PropertySpec] {
        MOCKCAM_PROPS
    }
    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "device" => {
                self.device = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }
    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "device" => Some(PropValue::Str(self.device.clone())),
            _ => None,
        }
    }
}

#[test]
fn create_builds_source_and_applies_props() {
    let mut registry = Registry::new();
    registry.register_source(SourceFactory::new("mockcamsrc", video_caps(640), || {
        Box::new(MockCamSrc::default())
    }));
    let cam = camera("video1", "Cam");
    let element = cam.create(&registry).expect("registered source");
    let DeviceElement::Source(src) = element else {
        panic!("camera should build a source");
    };
    assert_eq!(
        src.get_property("device"),
        Some(PropValue::Str("/dev/video1".into()))
    );
}

#[test]
fn create_rejects_unknown_element_and_bad_prop() {
    let registry = Registry::new();
    let cam = camera("video0", "Cam");
    assert!(matches!(cam.create(&registry), Err(G2gError::CapsMismatch)));

    let mut registry = Registry::new();
    registry.register_source(SourceFactory::new("mockcamsrc", video_caps(640), || {
        Box::new(MockCamSrc::default())
    }));
    let mut cam = camera("video0", "Cam");
    cam.props = vec![("no-such-prop".into(), "x".into())];
    assert!(matches!(
        cam.create(&registry),
        Err(G2gError::ControlBinding)
    ));
}

// --- started monitor: poll-and-diff hotplug --------------------------------

fn drain_until(monitor: &g2g_core::runtime::RunningMonitor, want: usize) -> Vec<DeviceEvent> {
    let mut events = Vec::new();
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while events.len() < want && std::time::Instant::now() < deadline {
        match monitor.try_recv() {
            Some(ev) => events.push(ev),
            None => std::thread::sleep(Duration::from_millis(5)),
        }
    }
    events
}

#[test]
fn poll_watch_emits_initial_added_removed_changed() {
    let devices = Arc::new(Mutex::new(vec![camera("video0", "Cam A")]));
    let mut monitor = DeviceMonitor::new();
    monitor
        .register(Box::new(MockProvider {
            devices: devices.clone(),
            fail: false,
        }))
        .set_poll_interval(Some(Duration::from_millis(20)));
    let running = monitor.start();

    let initial = drain_until(&running, 1);
    assert_eq!(initial, vec![DeviceEvent::Added(camera("video0", "Cam A"))]);

    // Replace video0's description and add video1: one Changed, one Added.
    *devices.lock().unwrap() = vec![camera("video0", "Cam B"), camera("video1", "Cam C")];
    let events = drain_until(&running, 2);
    assert!(events.contains(&DeviceEvent::Changed(camera("video0", "Cam B"))));
    assert!(events.contains(&DeviceEvent::Added(camera("video1", "Cam C"))));

    devices.lock().unwrap().clear();
    let events = drain_until(&running, 2);
    assert!(events.contains(&DeviceEvent::Removed {
        provider: "mock",
        persistent_id: "video0".into(),
    }));
    assert!(events.contains(&DeviceEvent::Removed {
        provider: "mock",
        persistent_id: "video1".into(),
    }));

    running.stop();
}

#[test]
fn started_monitor_applies_filters_to_poll_events() {
    let devices = Arc::new(Mutex::new(vec![camera("video0", "Cam"), speaker("card0")]));
    let mut monitor = DeviceMonitor::new();
    monitor
        .register(Box::new(MockProvider {
            devices,
            fail: false,
        }))
        .add_filter("Audio/Sink", None)
        .set_poll_interval(None);
    let running = monitor.start();
    let events = drain_until(&running, 1);
    assert_eq!(events, vec![DeviceEvent::Added(speaker("card0"))]);
    // Nothing else passes the filter, and interval None means no re-poll.
    assert_eq!(running.try_recv(), None);
}

// --- started monitor: native watch path ------------------------------------

/// Provider with a native watch: a thread posts a canned Added + Removed pair
/// through the filtering sink, and the guard's drop is observable.
struct NativeProvider {
    stopped: Arc<Mutex<bool>>,
}

impl DeviceProvider for NativeProvider {
    fn name(&self) -> &'static str {
        "native"
    }
    fn probe(&self) -> Result<Vec<Device>, G2gError> {
        Ok(vec![camera("video9", "Native Cam")])
    }
    fn watch(&self, sink: DeviceSink) -> Result<Option<WatchGuard>, G2gError> {
        let stopped = self.stopped.clone();
        let handle = std::thread::spawn(move || {
            sink.post(DeviceEvent::Added(camera("video9", "Native Cam")));
            sink.post(DeviceEvent::Added(speaker("card9")));
            sink.post(DeviceEvent::Removed {
                provider: "mock",
                persistent_id: "video9".into(),
            });
        });
        Ok(Some(WatchGuard::new(move || {
            let _ = handle.join();
            *stopped.lock().unwrap() = true;
        })))
    }
}

#[test]
fn native_watch_events_pass_the_monitor_filters() {
    let stopped = Arc::new(Mutex::new(false));
    let mut monitor = DeviceMonitor::new();
    monitor
        .register(Box::new(NativeProvider {
            stopped: stopped.clone(),
        }))
        .add_filter("Video/Source", None);
    let running = monitor.start();
    assert!(running.watch_errors.is_empty());

    // The speaker is filtered out; Removed always passes.
    let events = drain_until(&running, 2);
    assert_eq!(
        events,
        vec![
            DeviceEvent::Added(camera("video9", "Native Cam")),
            DeviceEvent::Removed {
                provider: "mock",
                persistent_id: "video9".into(),
            },
        ]
    );

    running.stop();
    assert!(*stopped.lock().unwrap());
}

#[test]
fn drop_with_full_event_queue_does_not_deadlock() {
    // More initial devices than the event channel holds, so the poll watcher
    // is mid-post (blocked on a full queue) when the monitor drops. Shutdown
    // must close the channel before joining or this hangs forever.
    let many: Vec<Device> = (0..200)
        .map(|i| camera(&format!("video{i}"), "Cam"))
        .collect();
    let mut monitor = DeviceMonitor::new();
    monitor
        .register(Box::new(MockProvider {
            devices: Arc::new(Mutex::new(many)),
            fail: false,
        }))
        .set_poll_interval(None);
    let running = monitor.start();
    // Let the watcher fill the queue and block on it.
    std::thread::sleep(Duration::from_millis(50));
    drop(running);
}

#[test]
fn recv_is_awaitable() {
    let devices = Arc::new(Mutex::new(vec![camera("video0", "Cam")]));
    let mut monitor = DeviceMonitor::new();
    monitor
        .register(Box::new(MockProvider {
            devices,
            fail: false,
        }))
        .set_poll_interval(None);
    let running = monitor.start();
    let event = block_on(running.recv());
    assert_eq!(event, Some(DeviceEvent::Added(camera("video0", "Cam"))));
}
