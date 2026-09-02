//! PipeWire device discovery (M939): the [`DeviceProvider`] over the PipeWire
//! graph. Nodes carrying a capture / render `media.class` become devices whose
//! element is the matching g2g PipeWire element, selected with `target-object`.
//!
//! The only Linux backend with a native event source: [`watch`](PipeWireDeviceProvider::watch)
//! registers a registry listener, so the hotplug path is push, not poll.
//!
//! The main loop is thread-affine, so both `probe` and `watch` own a dedicated
//! thread with their own connection (same shape as the capture elements).

use core::cell::RefCell;
use core::time::Duration;

use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use g2g_core::runtime::{Device, DeviceEvent, DeviceProvider, DeviceSink, WatchGuard};
use g2g_core::{AudioFormat, Caps, CapsSet, Dim, G2gError, HardwareError, Interlace, Rate};

use pipewire as pw;
use pw::registry::GlobalObject;
use pw::spa::utils::dict::DictRef;
use pw::types::ObjectType;

use crate::pwaudio::pw_params;
use crate::pwvideo::{supported_formats, MAX_DIM};

/// [`Device::provider`] for everything this backend finds.
const PROVIDER: &str = "pipewire";

/// `media.class` prefixes we turn into devices, with the device class (the same
/// string) and the element that opens the node. A prefix match takes
/// `Audio/Source/Virtual` as an audio source; everything else (`Audio/Duplex`,
/// the `Stream/...` classes of application streams) is not an endpoint we drive.
const NODE_CLASSES: [(&str, &str); 3] = [
    ("Audio/Source", "pipewiresrc"),
    ("Audio/Sink", "pipewiresink"),
    ("Video/Source", "pipewirevideosrc"),
];

/// Node props copied into [`Device::detail`] for display. The full node dict
/// carries dozens of transient keys, so only these are worth showing.
const DETAIL_KEYS: [&str; 5] = [
    "media.class",
    "object.serial",
    "device.api",
    "device.description",
    "node.nick",
];

/// The PCM formats an audio device advertises, filtered through the elements'
/// own SPA mapping so the provider cannot offer one they refuse to open.
fn pcm_formats() -> [AudioFormat; 5] {
    g2g_core::pcm_formats()
}

/// Upper framerate bound of an advertised video range. Nothing in the graph
/// produces faster, and an unbounded `Rate::Any` would not survive fixate.
const MAX_FPS: u32 = 1_000;

/// Enumeration roundtrip bound. The daemon answers a sync in milliseconds, so a
/// loop still running after this will not finish; quit rather than wedge the
/// probe thread.
const PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// How long [`watch`](PipeWireDeviceProvider::watch) waits for its thread to
/// connect before reporting the backend unreachable.
const WATCH_READY_TIMEOUT: Duration = Duration::from_secs(5);

fn pw_error() -> G2gError {
    G2gError::Hardware(HardwareError::PipeWire(-1))
}

/// Discovers PipeWire audio / video nodes. Stateless: each call opens its own
/// connection on its own thread.
#[derive(Debug, Default, Clone, Copy)]
pub struct PipeWireDeviceProvider;

impl PipeWireDeviceProvider {
    pub fn new() -> Self {
        Self
    }
}

impl DeviceProvider for PipeWireDeviceProvider {
    fn name(&self) -> &'static str {
        PROVIDER
    }

    fn probe(&self) -> Result<Vec<Device>, G2gError> {
        std::thread::Builder::new()
            .name(String::from("g2g-pwdevice-probe"))
            .spawn(probe_main)
            .map_err(|_| pw_error())?
            .join()
            .map_err(|_| pw_error())?
    }

    fn watch(&self, sink: DeviceSink) -> Result<Option<WatchGuard>, G2gError> {
        let (ctrl_tx, ctrl_rx) = pw::channel::channel::<()>();
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel::<Result<(), G2gError>>(1);

        let handle = std::thread::Builder::new()
            .name(String::from("g2g-pwdevice-watch"))
            .spawn(move || {
                if let Err(e) = watch_main(sink, ctrl_rx, &ready_tx) {
                    let _ = ready_tx.send(Err(e));
                }
            })
            .map_err(|_| pw_error())?;

        // block until the listener is registered, so a dead daemon is an error
        // from watch() and the monitor can fall back to polling
        match wait_ready(&ready_rx) {
            Ok(()) => {}
            Err(e) => {
                let _ = ctrl_tx.send(());
                let _ = handle.join();
                return Err(e);
            }
        }

        Ok(Some(WatchGuard::new(move || {
            let _ = ctrl_tx.send(());
            let _ = handle.join();
        })))
    }
}

/// Flatten the watch thread's setup handshake (timeout included) to one result.
fn wait_ready(rx: &std::sync::mpsc::Receiver<Result<(), G2gError>>) -> Result<(), G2gError> {
    match rx.recv_timeout(WATCH_READY_TIMEOUT) {
        Ok(res) => res,
        Err(_) => Err(pw_error()),
    }
}

// =================================================================
// Node props -> Device
// =================================================================

/// Whether `class` is `prefix` or one of its subclasses (`Audio/Source/Virtual`),
/// without matching a longer name that merely starts with the same letters.
fn class_matches(class: &str, prefix: &str) -> bool {
    match class.strip_prefix(prefix) {
        Some(rest) => rest.is_empty() || rest.starts_with('/'),
        None => false,
    }
}

/// Map one node's props to a device. `None` for anything that is not a capture
/// / render endpoint we can open: a class outside [`NODE_CLASSES`], or a node
/// with no `node.name` to target it by.
fn map_node(props: &[(&str, &str)]) -> Option<Device> {
    let get = |key: &str| {
        props
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| *v)
            .filter(|v| !v.is_empty())
    };

    let class = get("media.class")?;
    let (klass, element) = NODE_CLASSES
        .iter()
        .find(|(prefix, _)| class_matches(class, prefix))?;

    // node.name is the one id that survives a daemon restart: object.serial and
    // the registry global id are both per-session
    let name = get("node.name")?;
    let display_name = get("node.description")
        .or_else(|| get("node.nick"))
        .unwrap_or(name);

    let detail = DETAIL_KEYS
        .iter()
        .filter_map(|key| get(key).map(|v| ((*key).to_string(), v.to_string())))
        .collect();

    Some(Device {
        display_name: display_name.to_string(),
        klass: (*klass).to_string(),
        persistent_id: name.to_string(),
        caps: if klass.starts_with("Video") {
            video_caps()
        } else {
            audio_caps()
        },
        element,
        props: Vec::from([("target-object".to_string(), name.to_string())]),
        detail,
        provider: PROVIDER,
    })
}

/// What a PipeWire audio node can deliver through the graph, which is not its
/// hardware mode: the adapter resamples and remixes, so every PCM format the
/// elements open a stream with is reachable at the element default 48 kHz
/// stereo.
fn audio_caps() -> CapsSet {
    let alternatives: Vec<Caps> = pcm_formats()
        .iter()
        .map(|format| Caps::Audio {
            format: *format,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        })
        .filter(|caps| pw_params(caps).is_ok())
        .collect();
    CapsSet::from_alternatives(alternatives)
}

/// What a PipeWire video node can deliver. Geometry and rate are the node's
/// business and cost a stream connect to learn, so the ranges are the element's
/// bounds rather than a probed mode.
fn video_caps() -> CapsSet {
    let alternatives: Vec<Caps> = supported_formats()
        .map(|format| Caps::RawVideo {
            format,
            width: Dim::Range {
                min: 1,
                max: MAX_DIM,
            },
            height: Dim::Range {
                min: 1,
                max: MAX_DIM,
            },
            framerate: Rate::Range {
                min_q16: 1 << 16,
                max_q16: MAX_FPS << 16,
            },
            interlace: Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        })
        .collect();
    CapsSet::from_alternatives(alternatives)
}

/// A device for a registry global, or `None` when it is not a media node.
fn device_from_global(obj: &GlobalObject<&DictRef>) -> Option<Device> {
    if obj.type_ != ObjectType::Node {
        return None;
    }
    let pairs: Vec<(&str, &str)> = obj.props?.iter().collect();
    map_node(&pairs)
}

// =================================================================
// Loop threads: one-shot probe and the hotplug watch
// =================================================================

fn probe_main() -> Result<Vec<Device>, G2gError> {
    pw::init();
    let mainloop = pw::main_loop::MainLoop::new(None).map_err(|_| pw_error())?;
    let context = pw::context::Context::new(&mainloop).map_err(|_| pw_error())?;
    let core = context.connect(None).map_err(|_| pw_error())?;
    let registry = core.get_registry().map_err(|_| pw_error())?;

    let found = Rc::new(RefCell::new(Vec::new()));
    let collect = found.clone();
    let _listener = registry
        .add_listener_local()
        .global(move |obj| {
            if let Some(device) = device_from_global(obj) {
                collect.borrow_mut().push(device);
            }
        })
        .register();

    // the daemon answers a sync only after every global it had already sent, so
    // `done` for this seq is the end-of-enumeration fence
    let fence = core.sync(0).map_err(|_| pw_error())?.seq();
    let weak = mainloop.downgrade();
    let _core_listener = core
        .add_listener_local()
        .done(move |id, seq| {
            if id == pw::core::PW_ID_CORE && seq.seq() == fence {
                if let Some(ml) = weak.upgrade() {
                    ml.quit();
                }
            }
        })
        .register();

    let weak = mainloop.downgrade();
    let timer = mainloop.loop_().add_timer(move |_| {
        if let Some(ml) = weak.upgrade() {
            ml.quit();
        }
    });
    timer
        .update_timer(Some(PROBE_TIMEOUT), None)
        .into_result()
        .map_err(|_| pw_error())?;

    mainloop.run();
    Ok(found.take())
}

fn watch_main(
    sink: DeviceSink,
    ctrl_rx: pw::channel::Receiver<()>,
    ready: &std::sync::mpsc::SyncSender<Result<(), G2gError>>,
) -> Result<(), G2gError> {
    pw::init();
    let mainloop = pw::main_loop::MainLoop::new(None).map_err(|_| pw_error())?;
    let context = pw::context::Context::new(&mainloop).map_err(|_| pw_error())?;
    let core = context.connect(None).map_err(|_| pw_error())?;
    let registry = core.get_registry().map_err(|_| pw_error())?;

    // global id -> persistent id: the remove event carries the id alone, so the
    // mapping has to be kept from the add
    let known: Rc<RefCell<BTreeMap<u32, String>>> = Rc::new(RefCell::new(BTreeMap::new()));

    let add_known = known.clone();
    let add_sink = sink.clone();
    let add_quit = mainloop.downgrade();
    let remove_quit = mainloop.downgrade();

    // a fresh listener is replayed every existing global, so the initial Added
    // set arrives without a separate probe
    let _listener = registry
        .add_listener_local()
        .global(move |obj| {
            let Some(device) = device_from_global(obj) else {
                return;
            };
            add_known
                .borrow_mut()
                .insert(obj.id, device.persistent_id.clone());
            // post blocks while the monitor's queue is full: the loop stalls
            // rather than dropping a hotplug event
            if !add_sink.post(DeviceEvent::Added(device)) {
                if let Some(ml) = add_quit.upgrade() {
                    ml.quit();
                }
            }
        })
        .global_remove(move |id| {
            let Some(persistent_id) = known.borrow_mut().remove(&id) else {
                return;
            };
            let removed = DeviceEvent::Removed {
                provider: PROVIDER,
                persistent_id,
            };
            if !sink.post(removed) {
                if let Some(ml) = remove_quit.upgrade() {
                    ml.quit();
                }
            }
        })
        .register();

    let weak = mainloop.downgrade();
    let _recv = ctrl_rx.attach(mainloop.loop_(), move |()| {
        if let Some(ml) = weak.upgrade() {
            ml.quit();
        }
    });

    let _ = ready.send(Ok(()));
    mainloop.run();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::runtime::DeviceElement;

    fn audio_source_props() -> Vec<(&'static str, &'static str)> {
        Vec::from([
            ("media.class", "Audio/Source"),
            ("node.name", "alsa_input.pci-0000_00_1f.3.analog-stereo"),
            ("node.description", "Built-in Audio Analog Stereo"),
            ("node.nick", "Built-in"),
            ("object.serial", "42"),
            ("device.api", "alsa"),
        ])
    }

    #[test]
    fn audio_source_maps_to_the_capture_element() {
        let device = map_node(&audio_source_props()).expect("audio source maps");
        assert_eq!(device.element, "pipewiresrc");
        assert_eq!(device.klass, "Audio/Source");
        assert_eq!(device.provider, "pipewire");
        assert_eq!(device.display_name, "Built-in Audio Analog Stereo");
        assert_eq!(
            device.persistent_id,
            "alsa_input.pci-0000_00_1f.3.analog-stereo"
        );
        assert_eq!(
            device.props,
            Vec::from([(
                "target-object".to_string(),
                "alsa_input.pci-0000_00_1f.3.analog-stereo".to_string()
            )])
        );
        // every PCM format the elements can open, at the element defaults
        assert_eq!(device.caps.alternatives().len(), pcm_formats().len());
        for caps in device.caps.alternatives() {
            assert!(matches!(
                caps,
                Caps::Audio {
                    channels: 2,
                    sample_rate: 48_000,
                    ..
                }
            ));
        }
    }

    #[test]
    fn sink_and_video_classes_route_to_their_elements() {
        let sink = map_node(&[
            ("media.class", "Audio/Sink"),
            ("node.name", "alsa_output.analog-stereo"),
        ])
        .expect("audio sink maps");
        assert_eq!(sink.element, "pipewiresink");
        assert_eq!(sink.klass, "Audio/Sink");

        let cam = map_node(&[
            ("media.class", "Video/Source"),
            ("node.name", "v4l2_input.0"),
        ])
        .expect("video source maps");
        assert_eq!(cam.element, "pipewirevideosrc");
        assert_eq!(cam.klass, "Video/Source");
        assert_eq!(cam.caps.alternatives().len(), supported_formats().count());
        for caps in cam.caps.alternatives() {
            let Caps::RawVideo {
                width,
                height,
                framerate,
                ..
            } = caps
            else {
                panic!("video device carries raw video caps");
            };
            // ranges, never Any: an advertised Any does not survive fixate
            assert_eq!(
                *width,
                Dim::Range {
                    min: 1,
                    max: MAX_DIM
                }
            );
            assert_eq!(
                *height,
                Dim::Range {
                    min: 1,
                    max: MAX_DIM
                }
            );
            assert!(matches!(framerate, Rate::Range { .. }));
        }
    }

    #[test]
    fn class_matching_takes_subclasses_and_drops_the_rest() {
        let virtual_mic = map_node(&[
            ("media.class", "Audio/Source/Virtual"),
            ("node.name", "virtual-mic"),
        ]);
        assert_eq!(
            virtual_mic.map(|d| d.element),
            Some("pipewiresrc"),
            "a subclass is still an audio source"
        );

        for class in [
            "Audio/Duplex",
            "Stream/Output/Audio",
            "Stream/Input/Video",
            "Video/Sink",
            "Audio/Sourcery",
            "Midi/Bridge",
        ] {
            assert!(
                map_node(&[("media.class", class), ("node.name", "n")]).is_none(),
                "{class} is not a device we drive"
            );
        }
        // a node with no media.class is not an endpoint either
        assert!(map_node(&[("node.name", "n")]).is_none());
    }

    #[test]
    fn display_name_falls_back_to_nick_then_name() {
        let nick = map_node(&[
            ("media.class", "Audio/Source"),
            ("node.name", "mic0"),
            ("node.nick", "Desk mic"),
        ])
        .expect("maps");
        assert_eq!(nick.display_name, "Desk mic");

        let bare =
            map_node(&[("media.class", "Audio/Source"), ("node.name", "mic0")]).expect("maps");
        assert_eq!(bare.display_name, "mic0");

        // an empty value is no value
        let empty = map_node(&[
            ("media.class", "Audio/Source"),
            ("node.name", "mic0"),
            ("node.description", ""),
            ("node.nick", "Desk mic"),
        ])
        .expect("maps");
        assert_eq!(empty.display_name, "Desk mic");
    }

    #[test]
    fn a_node_without_a_name_is_skipped() {
        assert!(map_node(&[("media.class", "Audio/Source"), ("object.serial", "7")]).is_none());
        assert!(map_node(&[("media.class", "Audio/Source"), ("node.name", "")]).is_none());
    }

    #[test]
    fn detail_carries_the_display_only_keys() {
        let device = map_node(&audio_source_props()).expect("maps");
        for (key, value) in [
            ("media.class", "Audio/Source"),
            ("object.serial", "42"),
            ("device.api", "alsa"),
        ] {
            assert!(
                device.detail.iter().any(|(k, v)| k == key && v == value),
                "detail is missing {key}"
            );
        }
        // detail is display-only, never fed to the element
        assert!(device.detail.iter().all(|(k, _)| k != "target-object"));
    }

    /// The provider and the elements agree: every mapped device's element +
    /// props build through the registry, which is what `Device::create` and a
    /// launch line both do.
    #[test]
    fn every_mapped_device_builds_its_element() {
        let registry = crate::registry::default_registry();
        for (class, name) in [
            ("Audio/Source", "mic0"),
            ("Audio/Sink", "spk0"),
            ("Video/Source", "cam0"),
        ] {
            let device = map_node(&[("media.class", class), ("node.name", name)]).expect("maps");
            let built = device.create(&registry).expect("device builds its element");
            let target = match &built {
                DeviceElement::Source(src) => src.get_property("target-object"),
                DeviceElement::Element(el) => el.get_property("target-object"),
            };
            assert_eq!(
                target,
                Some(g2g_core::PropValue::Str(name.to_string())),
                "{class} element did not take the device's target-object"
            );
            assert_eq!(device.launch_fragment(), {
                let mut s = String::from(device.element);
                s.push_str(" target-object=");
                s.push_str(name);
                s
            });
        }
    }

    /// Live smoke: skips cleanly when no daemon is reachable.
    #[test]
    fn probe_and_watch_against_a_running_daemon() {
        use g2g_core::runtime::DeviceMonitor;

        let provider = PipeWireDeviceProvider::new();
        let Ok(devices) = provider.probe() else {
            return; // no PipeWire here
        };
        for device in &devices {
            assert_eq!(device.provider, "pipewire");
            assert!(!device.persistent_id.is_empty());
            assert!(!device.caps.alternatives().is_empty());
        }

        // the same set has to arrive as initial Added events from the native
        // watch, and the guard has to stop the loop thread on drop
        let mut monitor = DeviceMonitor::new();
        monitor.register(alloc::boxed::Box::new(PipeWireDeviceProvider::new()));
        let running = monitor.start();
        assert!(running.watch_errors.is_empty(), "native watch started");
        let mut added = Vec::new();
        for _ in 0..50 {
            while let Some(event) = running.try_recv() {
                if let DeviceEvent::Added(d) = event {
                    added.push(d.persistent_id);
                }
            }
            if added.len() >= devices.len() {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        for device in &devices {
            assert!(
                added.contains(&device.persistent_id),
                "watch replayed {}",
                device.persistent_id
            );
        }
        drop(running);
    }
}
