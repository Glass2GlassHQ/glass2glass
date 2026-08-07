//! WASAPI endpoint discovery (M943): the Windows audio half of the
//! [`DeviceMonitor`](g2g_core::runtime::DeviceMonitor), the counterpart of
//! [`alsadevice`](crate::alsadevice) on Linux. `IMMDeviceEnumerator` lists the
//! active render and capture endpoints; a render endpoint drives
//! [`WasapiSink`](crate::wasapisink), a capture endpoint
//! [`WasapiSrc`](crate::wasapisrc), each selected by the endpoint id string
//! Windows keeps stable across replug and reboot.
//!
//! This is the one Windows backend with a native event source:
//! [`watch`](WasapiDeviceProvider::watch) registers an `IMMNotificationClient`,
//! so endpoint hotplug is push. Cameras have no such callback, so
//! [`mfdevice`](crate::mfdevice) is polled.
//!
//! Caps come from each endpoint's shared-mode mix format, the one shape a
//! shared-mode client can open it as. An endpoint that will not activate is
//! still listed, with empty caps.
//!
//! The endpoint selector both elements use lives in
//! [`wasapipcm`](crate::wasapipcm), so the id this provider reports and the id
//! `device=` accepts cannot drift.
//!
//! Windows-only and compile-checked cross-target from Linux; the enumeration
//! and the notification callback are owed a run on a Windows host.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use std::sync::mpsc as std_mpsc;
use std::thread;
use std::time::Duration;

use windows::core::{implement, PCWSTR};
use windows::Win32::Devices::FunctionDiscovery::PKEY_Device_FriendlyName;
use windows::Win32::Foundation::PROPERTYKEY;
use windows::Win32::Media::Audio::{
    eRender, EDataFlow, ERole, IAudioClient, IMMDevice, IMMDeviceEnumerator, IMMNotificationClient,
    IMMNotificationClient_Impl, MMDeviceEnumerator, DEVICE_STATE, DEVICE_STATE_ACTIVE,
};
use windows::Win32::System::Com::StructuredStorage::PropVariantClear;
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoTaskMemFree, CoUninitialize, CLSCTX_ALL,
    CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED, STGM_READ,
};
use windows::Win32::System::Variant::VT_LPWSTR;

use g2g_core::runtime::{Device, DeviceEvent, DeviceProvider, DeviceSink, WatchGuard};
use g2g_core::{Caps, CapsSet, G2gError, HardwareError};

use crate::wasapipcm::{audio_config_from_format, audio_err};

/// [`Device::provider`] for everything this backend finds.
const PROVIDER: &str = "wasapi";

/// How long the enumeration thread may take before the probe gives up.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// How often the watch thread looks for its stop signal while no endpoint
/// notification arrives.
const STOP_POLL: Duration = Duration::from_millis(250);

/// Lists the active WASAPI render / capture endpoints.
#[derive(Debug, Default, Clone, Copy)]
pub struct WasapiDeviceProvider;

impl WasapiDeviceProvider {
    pub fn new() -> Self {
        Self
    }
}

impl DeviceProvider for WasapiDeviceProvider {
    fn name(&self) -> &'static str {
        PROVIDER
    }

    fn probe(&self) -> Result<Vec<Device>, G2gError> {
        // WASAPI is COM and the monitor's poll thread is not initialised for
        // it, so the enumeration owns a thread of its own.
        let (tx, rx) = std_mpsc::sync_channel::<Result<Vec<Device>, G2gError>>(1);
        thread::Builder::new()
            .name(String::from("g2g-wasapidevice-probe"))
            .spawn(move || {
                // SAFETY: COM init on this thread, balanced before it exits.
                let result = unsafe {
                    let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
                    let r = probe_endpoints();
                    CoUninitialize();
                    r
                };
                let _ = tx.send(result);
            })
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        rx.recv_timeout(PROBE_TIMEOUT)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?
    }

    /// Register an `IMMNotificationClient` and report endpoint changes as they
    /// happen. The callback thread re-probes on each notification and diffs,
    /// because the callback carries only an id: cheap, since hotplug is rare.
    fn watch(&self, sink: DeviceSink) -> Result<Option<WatchGuard>, G2gError> {
        let (stop_tx, stop_rx) = std_mpsc::channel::<()>();
        let (ready_tx, ready_rx) = std_mpsc::sync_channel::<Result<(), G2gError>>(1);
        let handle = thread::Builder::new()
            .name(String::from("g2g-wasapidevice-watch"))
            .spawn(move || {
                if let Err(e) = watch_main(&sink, &stop_rx, &ready_tx) {
                    let _ = ready_tx.send(Err(e));
                }
            })
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

        // Block until the callback is registered, so a failure here is an
        // error from watch() and the monitor falls back to polling.
        match ready_rx.recv_timeout(PROBE_TIMEOUT) {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                let _ = stop_tx.send(());
                let _ = handle.join();
                return Err(e);
            }
            Err(_) => {
                let _ = stop_tx.send(());
                let _ = handle.join();
                return Err(G2gError::Hardware(HardwareError::Other));
            }
        }

        Ok(Some(WatchGuard::new(move || {
            let _ = stop_tx.send(());
            let _ = handle.join();
        })))
    }
}

// =================================================================
// Enumeration
// =================================================================

/// # Safety
/// Must run on a COM-initialised thread.
unsafe fn probe_endpoints() -> Result<Vec<Device>, G2gError> {
    // SAFETY: COM enumeration on the owning thread.
    unsafe {
        let enumerator: IMMDeviceEnumerator =
            CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_INPROC_SERVER).map_err(audio_err)?;
        let mut devices = Vec::new();
        for dataflow in listed_dataflows() {
            let collection = enumerator
                .EnumAudioEndpoints(dataflow, DEVICE_STATE_ACTIVE)
                .map_err(audio_err)?;
            let count = collection.GetCount().map_err(audio_err)?;
            for index in 0..count {
                let Ok(endpoint) = collection.Item(index) else {
                    continue;
                };
                if let Some(device) = describe_endpoint(&endpoint, dataflow) {
                    devices.push(device);
                }
            }
        }
        Ok(devices)
    }
}

/// The endpoint directions this build lists: each needs the element that
/// drives it compiled in.
fn listed_dataflows() -> Vec<EDataFlow> {
    #[allow(unused_mut)]
    let mut out = Vec::new();
    #[cfg(feature = "wasapi-src")]
    out.push(windows::Win32::Media::Audio::eCapture);
    #[cfg(feature = "wasapi-sink")]
    out.push(eRender);
    out
}

/// One endpoint as a [`Device`]; `None` when it has no id to select it by.
///
/// # Safety
/// Must run on a COM-initialised thread.
unsafe fn describe_endpoint(endpoint: &IMMDevice, dataflow: EDataFlow) -> Option<Device> {
    // SAFETY: property + activation calls on the owning thread; every string
    // the API allocates is copied out and freed here.
    unsafe {
        let id_ptr = endpoint.GetId().ok()?;
        if id_ptr.is_null() {
            return None;
        }
        let id = id_ptr.to_string().ok();
        CoTaskMemFree(Some(id_ptr.as_ptr().cast()));
        let id = id?;

        let name = friendly_name(endpoint).unwrap_or_else(|| id.clone());
        let (element, klass) = if dataflow == eRender {
            ("wasapisink", "Audio/Sink")
        } else {
            ("wasapisrc", "Audio/Source")
        };
        let (caps, detail) = match mix_format_caps(endpoint) {
            Some((caps, rate, channels)) => (
                CapsSet::one(caps),
                Vec::from([
                    ("mix-rate".to_string(), rate.to_string()),
                    ("mix-channels".to_string(), channels.to_string()),
                ]),
            ),
            None => (
                CapsSet::from_alternatives(Vec::new()),
                Vec::from([(
                    "probe-error".to_string(),
                    "endpoint would not report a mix format".to_string(),
                )]),
            ),
        };

        Some(Device {
            display_name: name,
            klass: klass.to_string(),
            persistent_id: id.clone(),
            caps,
            element,
            props: Vec::from([("device".to_string(), id)]),
            detail,
            provider: PROVIDER,
        })
    }
}

/// `PKEY_Device_FriendlyName` off the endpoint's property store.
///
/// # Safety
/// Must run on a COM-initialised thread.
unsafe fn friendly_name(endpoint: &IMMDevice) -> Option<String> {
    // SAFETY: the PROPVARIANT is read only for the string type it reports and
    // is cleared before returning.
    unsafe {
        let store = endpoint.OpenPropertyStore(STGM_READ).ok()?;
        let mut value = store.GetValue(&PKEY_Device_FriendlyName).ok()?;
        let inner = &value.Anonymous.Anonymous;
        let name = if inner.vt == VT_LPWSTR && !inner.Anonymous.pwszVal.is_null() {
            inner.Anonymous.pwszVal.to_string().ok()
        } else {
            None
        };
        let _ = PropVariantClear(&mut value);
        name
    }
}

/// The endpoint's shared-mode mix format as caps, plus the rate and channel
/// count for display. `None` when it cannot be activated or the format is one
/// the elements do not carry.
///
/// # Safety
/// Must run on a COM-initialised thread.
unsafe fn mix_format_caps(endpoint: &IMMDevice) -> Option<(Caps, u32, u8)> {
    // SAFETY: activation on the owning thread; the format the API allocates is
    // read then freed.
    unsafe {
        let client: IAudioClient = endpoint.Activate(CLSCTX_ALL, None).ok()?;
        let fmt_ptr = client.GetMixFormat().ok()?;
        if fmt_ptr.is_null() {
            return None;
        }
        // the same mapping the elements apply, so the provider cannot
        // advertise a shape they refuse to open.
        let config = audio_config_from_format(&*fmt_ptr).ok();
        CoTaskMemFree(Some(fmt_ptr.cast()));
        let config = config?;
        Some((
            Caps::Audio {
                format: config.format,
                channels: config.channels,
                sample_rate: config.sample_rate,
            },
            config.sample_rate,
            config.channels,
        ))
    }
}

// =================================================================
// Hotplug: IMMNotificationClient
// =================================================================

/// The endpoint callback. Every method just nudges the watch thread; the
/// re-probe happens there, off the COM callback (which must not block).
#[implement(IMMNotificationClient)]
struct EndpointNotifier {
    /// COM may call the notification methods from any thread, and an mpsc
    /// `Sender` is Send but not Sync, so the lock is what makes `&self` sends
    /// legal rather than a nicety.
    changed: std::sync::Mutex<std_mpsc::Sender<()>>,
}

impl EndpointNotifier {
    fn nudge(&self) {
        if let Ok(changed) = self.changed.lock() {
            let _ = changed.send(());
        }
    }
}

impl IMMNotificationClient_Impl for EndpointNotifier_Impl {
    fn OnDeviceStateChanged(
        &self,
        _id: &PCWSTR,
        _state: DEVICE_STATE,
    ) -> windows::core::Result<()> {
        self.nudge();
        Ok(())
    }

    fn OnDeviceAdded(&self, _id: &PCWSTR) -> windows::core::Result<()> {
        self.nudge();
        Ok(())
    }

    fn OnDeviceRemoved(&self, _id: &PCWSTR) -> windows::core::Result<()> {
        self.nudge();
        Ok(())
    }

    fn OnDefaultDeviceChanged(
        &self,
        _flow: EDataFlow,
        _role: ERole,
        _id: &PCWSTR,
    ) -> windows::core::Result<()> {
        Ok(())
    }

    fn OnPropertyValueChanged(
        &self,
        _id: &PCWSTR,
        _key: &PROPERTYKEY,
    ) -> windows::core::Result<()> {
        Ok(())
    }
}

/// Watch thread: register the callback, emit the initial set, then re-probe
/// and diff on every notification until the guard drops.
fn watch_main(
    sink: &DeviceSink,
    stop: &std_mpsc::Receiver<()>,
    ready: &std_mpsc::SyncSender<Result<(), G2gError>>,
) -> Result<(), G2gError> {
    // SAFETY: COM init on this thread, balanced by the CoUninitialize below.
    unsafe {
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
    }
    let result = watch_loop(sink, stop, ready);
    // SAFETY: balances the initialise above.
    unsafe { CoUninitialize() };
    result
}

fn watch_loop(
    sink: &DeviceSink,
    stop: &std_mpsc::Receiver<()>,
    ready: &std_mpsc::SyncSender<Result<(), G2gError>>,
) -> Result<(), G2gError> {
    let (changed_tx, changed_rx) = std_mpsc::channel::<()>();
    let notifier: IMMNotificationClient = EndpointNotifier {
        changed: std::sync::Mutex::new(changed_tx),
    }
    .into();

    // SAFETY: COM object creation and registration on the owning thread; the
    // callback is unregistered before this function returns.
    let enumerator: IMMDeviceEnumerator = unsafe {
        CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_INPROC_SERVER).map_err(audio_err)?
    };
    // SAFETY: as above.
    unsafe {
        enumerator
            .RegisterEndpointNotificationCallback(&notifier)
            .map_err(audio_err)?;
    }
    let _ = ready.try_send(Ok(()));

    // SAFETY: enumeration on the owning thread.
    let mut known = unsafe { probe_endpoints() }.unwrap_or_default();
    for device in &known {
        if !sink.post(DeviceEvent::Added(device.clone())) {
            break;
        }
    }

    loop {
        if stop.try_recv().is_ok() {
            break;
        }
        // Only a notification triggers a re-probe: probing activates every
        // endpoint, so a timeout must fall through to the stop check above,
        // never to the enumeration below.
        match changed_rx.recv_timeout(STOP_POLL) {
            Ok(()) => {}
            Err(std_mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std_mpsc::RecvTimeoutError::Disconnected) => break,
        }
        // Coalesce the burst a single replug produces.
        while changed_rx.try_recv().is_ok() {}
        // SAFETY: enumeration on the owning thread.
        let Ok(now) = (unsafe { probe_endpoints() }) else {
            continue;
        };
        if !post_diff(&known, &now, &mut |event| sink.post(event)) {
            break;
        }
        known = now;
    }

    // SAFETY: unregisters the callback registered above, before it is dropped.
    unsafe {
        let _ = enumerator.UnregisterEndpointNotificationCallback(&notifier);
    }
    Ok(())
}

/// Post the difference between two probes; `false` once the consumer is gone.
fn post_diff(known: &[Device], now: &[Device], post: &mut impl FnMut(DeviceEvent) -> bool) -> bool {
    for old in known {
        if !now.iter().any(|d| d.persistent_id == old.persistent_id) {
            let removed = DeviceEvent::Removed {
                provider: old.provider,
                persistent_id: old.persistent_id.clone(),
            };
            if !post(removed) {
                return false;
            }
        }
    }
    for device in now {
        let event = match known
            .iter()
            .find(|k| k.persistent_id == device.persistent_id)
        {
            None => DeviceEvent::Added(device.clone()),
            Some(old) if old != device => DeviceEvent::Changed(device.clone()),
            Some(_) => continue,
        };
        if !post(event) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn endpoint(id: &str) -> Device {
        Device {
            display_name: id.to_string(),
            klass: "Audio/Source".to_string(),
            persistent_id: id.to_string(),
            caps: CapsSet::from_alternatives(Vec::new()),
            element: "wasapisrc",
            props: Vec::from([("device".to_string(), id.to_string())]),
            detail: Vec::new(),
            provider: PROVIDER,
        }
    }

    #[test]
    fn the_diff_reports_only_what_moved() {
        let mut seen = Vec::new();
        assert!(post_diff(
            &[endpoint("a"), endpoint("b")],
            &[endpoint("b"), endpoint("c")],
            &mut |event| {
                seen.push(event);
                true
            }
        ));
        assert_eq!(
            seen,
            [
                DeviceEvent::Removed {
                    provider: PROVIDER,
                    persistent_id: "a".to_string()
                },
                DeviceEvent::Added(endpoint("c")),
            ]
        );

        // an endpoint whose description changed under the same id is a change,
        // not an add, so a listing updates in place.
        let mut renamed = endpoint("b");
        renamed.display_name = "Headset".to_string();
        let mut seen = Vec::new();
        post_diff(&[endpoint("b")], &[renamed.clone()], &mut |event| {
            seen.push(event);
            true
        });
        assert_eq!(seen, [DeviceEvent::Changed(renamed)]);
    }

    #[test]
    fn a_gone_consumer_stops_the_diff() {
        let mut posts = 0;
        assert!(!post_diff(
            &[],
            &[endpoint("a"), endpoint("b")],
            &mut |_| {
                posts += 1;
                false
            }
        ));
        assert_eq!(posts, 1);
    }

    #[test]
    fn every_listed_direction_has_its_element_compiled_in() {
        let listed = listed_dataflows();
        #[cfg(feature = "wasapi-src")]
        assert!(listed.contains(&windows::Win32::Media::Audio::eCapture));
        #[cfg(not(feature = "wasapi-src"))]
        assert!(!listed.contains(&windows::Win32::Media::Audio::eCapture));
        #[cfg(feature = "wasapi-sink")]
        assert!(listed.contains(&eRender));
        #[cfg(not(feature = "wasapi-sink"))]
        assert!(!listed.contains(&eRender));
    }
}
