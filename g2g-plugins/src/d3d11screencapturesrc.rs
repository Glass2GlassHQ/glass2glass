// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::future::Future;
use core::pin::Pin;

use std::sync::mpsc as std_mpsc;
use std::thread;
use std::time::{Duration, Instant};

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use tokio::sync::watch;

use windows::core::Interface;
use windows::Win32::Foundation::{HMODULE, POINT};
use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_UNKNOWN;
use windows::Win32::Graphics::Direct3D11::{
    D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D, D3D11_CPU_ACCESS_READ,
    D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_MAPPED_SUBRESOURCE, D3D11_MAP_READ, D3D11_SDK_VERSION,
    D3D11_TEXTURE2D_DESC, D3D11_USAGE_STAGING,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_MODE_ROTATION_IDENTITY, DXGI_MODE_ROTATION_UNSPECIFIED,
    DXGI_SAMPLE_DESC,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, IDXGIAdapter1, IDXGIFactory1, IDXGIOutput, IDXGIOutput1,
    IDXGIOutputDuplication, DXGI_ERROR_NOT_CURRENTLY_AVAILABLE, DXGI_ERROR_NOT_FOUND,
    DXGI_ERROR_WAIT_TIMEOUT, DXGI_OUTDUPL_FRAME_INFO,
};
use windows::Win32::Graphics::Gdi::{MonitorFromPoint, MONITOR_DEFAULTTOPRIMARY};
use windows::Win32::UI::HiDpi::{
    SetThreadDpiAwarenessContext, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2,
};

use crate::mfcommon::mf_err;
use g2g_core::frame::Frame;
use g2g_core::g2g_error;
use g2g_core::log::Target;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, CaptureAnchor, ClockSync, Colorimetry, ConfigureOutcome, Dim,
    ElementMetadata, FrameTiming, G2gError, HardwareError, Interlace, LatencyReport, MemoryDomain,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PixelAspect, PropError, PropKind,
    PropValue, PropertySpec, Rate, RawVideoFormat,
};

const CATEGORY: &str = "d3d11screencapturesrc";
const PRIMARY_MONITOR: i32 = -1;
const DEFAULT_FPS: u32 = 60;
const BYTES_PER_PIXEL: usize = 4;
const NANOS_PER_SECOND: u64 = 1_000_000_000;
const WORKER_START_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MonitorBounds {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug)]
struct SelectedOutput {
    adapter: IDXGIAdapter1,
    output: IDXGIOutput,
    bounds: MonitorBounds,
}

pub fn with_physical_pixels<T>(run: impl FnOnce() -> T) -> T {
    // windows reports monitor coordinates scaled by the display DPI to threads that are not DPI aware
    // SAFETY: the call only changes this thread's DPI awareness, and the previous value is restored below.
    let previous =
        unsafe { SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) };
    let result = run();
    // SAFETY: restores the awareness this thread had before the call above.
    unsafe { SetThreadDpiAwarenessContext(previous) };
    result
}

fn desktop_outputs() -> Result<Vec<(IDXGIAdapter1, IDXGIOutput)>, G2gError> {
    // SAFETY: DXGI factory and enumeration calls take no pointers from this code.
    unsafe {
        let factory: IDXGIFactory1 = CreateDXGIFactory1().map_err(mf_err)?;
        let mut outputs = Vec::new();
        for adapter_index in 0.. {
            let adapter = match factory.EnumAdapters1(adapter_index) {
                Ok(adapter) => adapter,
                Err(e) if e.code() == DXGI_ERROR_NOT_FOUND => break,
                Err(e) => return Err(mf_err(e)),
            };
            for output_index in 0.. {
                match adapter.EnumOutputs(output_index) {
                    Ok(output) => outputs.push((adapter.clone(), output)),
                    Err(e) if e.code() == DXGI_ERROR_NOT_FOUND => break,
                    // EnumOutputs fails this way in a session with no desktop, such as ssh
                    Err(e) if e.code() == DXGI_ERROR_NOT_CURRENTLY_AVAILABLE => {
                        return Err(G2gError::DeviceNotFound)
                    }
                    Err(e) => return Err(mf_err(e)),
                }
            }
        }
        Ok(outputs)
    }
}

fn select_output(monitor_index: i32) -> Result<SelectedOutput, G2gError> {
    let log = Target::category(CATEGORY);
    // SAFETY: MonitorFromPoint takes the point by value and returns a handle without side effects.
    let primary = unsafe { MonitorFromPoint(POINT { x: 0, y: 0 }, MONITOR_DEFAULTTOPRIMARY) };
    let mut attached = Vec::new();
    for (adapter, output) in desktop_outputs()? {
        // SAFETY: GetDesc fills and returns a plain struct.
        let desc = unsafe { output.GetDesc() }.map_err(mf_err)?;
        if desc.AttachedToDesktop.as_bool() {
            attached.push((adapter, output, desc));
        }
    }
    let chosen = if monitor_index == PRIMARY_MONITOR {
        attached
            .into_iter()
            .find(|(_, _, desc)| desc.Monitor == primary)
    } else {
        usize::try_from(monitor_index)
            .ok()
            .and_then(|index| attached.into_iter().nth(index))
    };
    let Some((adapter, output, desc)) = chosen else {
        g2g_error!(log, "no desktop monitor has index {monitor_index}");
        return Err(G2gError::DeviceNotFound);
    };
    if desc.Rotation != DXGI_MODE_ROTATION_IDENTITY
        && desc.Rotation != DXGI_MODE_ROTATION_UNSPECIFIED
    {
        g2g_error!(
            log,
            "monitor {monitor_index} is rotated, which capture does not support"
        );
        return Err(G2gError::CapsMismatch);
    }
    let rect = desc.DesktopCoordinates;
    let side = |extent: i32| u32::try_from(extent).map_err(|_| G2gError::CapsMismatch);
    Ok(SelectedOutput {
        adapter,
        output,
        bounds: MonitorBounds {
            x: rect.left,
            y: rect.top,
            width: side(rect.right - rect.left)?,
            height: side(rect.bottom - rect.top)?,
        },
    })
}

pub fn monitor_bounds(monitor_index: i32) -> Result<MonitorBounds, G2gError> {
    with_physical_pixels(|| select_output(monitor_index).map(|selected| selected.bounds))
}

#[derive(Debug)]
pub struct D3D11ScreenCaptureSrc {
    monitor_index: i32,
    fps: u32,
    frame_limit: u64,
    bounds: Option<MonitorBounds>,
    configured: bool,
    clock_sync: Option<ClockSync>,
}

impl Default for D3D11ScreenCaptureSrc {
    fn default() -> Self {
        Self::new()
    }
}

impl D3D11ScreenCaptureSrc {
    pub fn new() -> Self {
        Self {
            monitor_index: PRIMARY_MONITOR,
            fps: DEFAULT_FPS,
            frame_limit: u64::MAX,
            bounds: None,
            configured: false,
            clock_sync: None,
        }
    }

    pub fn with_monitor_index(mut self, monitor_index: i32) -> Self {
        self.monitor_index = monitor_index;
        self
    }

    pub fn with_fps(mut self, fps: u32) -> Self {
        assert!(fps > 0, "capture rate must be positive");
        self.fps = fps;
        self
    }

    pub fn with_frame_limit(mut self, frames: u64) -> Self {
        self.frame_limit = frames;
        self
    }

    fn period(&self) -> Duration {
        Duration::from_nanos(NANOS_PER_SECOND / u64::from(self.fps))
    }

    fn probe(&mut self) -> Result<Caps, G2gError> {
        let bounds = match self.bounds {
            Some(bounds) => bounds,
            None => *self.bounds.insert(monitor_bounds(self.monitor_index)?),
        };
        Ok(Caps::RawVideo {
            format: RawVideoFormat::Bgra8,
            width: Dim::Fixed(bounds.width),
            height: Dim::Fixed(bounds.height),
            framerate: Rate::Fixed(self.fps << 16),
            interlace: Interlace::Progressive,
            colorimetry: Colorimetry::SRGB,
            pixel_aspect: PixelAspect::SQUARE,
        })
    }
}

static D3D11SCREENCAPTURESRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "monitor-index",
        PropKind::Int,
        "position among the desktop monitors (-1 = primary)",
    )
    .with_default("-1"),
    PropertySpec::new(
        "fps",
        PropKind::Uint,
        "frames per second, repeating the last image when the desktop has not changed",
    )
    .with_default("60"),
    PropertySpec::new(
        "num-buffers",
        PropKind::Int,
        "frames to capture then EOS (-1 = forever)",
    )
    .with_default("-1"),
];

impl SourceLoop for D3D11ScreenCaptureSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(self.probe())
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(
            self.probe()
                .map(|caps| CapsConstraint::Produces(CapsSet::one(caps))),
        )
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if self.bounds.is_none() {
            return Err(G2gError::NotConfigured);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Direct3D11 screen capture source",
            "Source/Video",
            "Captures a desktop monitor through DXGI Desktop Duplication as BGRA",
            crate::AARON_BOXER,
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        D3D11SCREENCAPTURESRC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "monitor-index" => {
                let index = value.as_int().ok_or(PropError::Type)?;
                self.monitor_index = i32::try_from(index).map_err(|_| PropError::Value)?;
                self.bounds = None;
            }
            "fps" => {
                let fps = value.as_uint().ok_or(PropError::Type)?;
                self.fps = u32::try_from(fps)
                    .ok()
                    .filter(|&fps| fps > 0)
                    .ok_or(PropError::Value)?;
            }
            "num-buffers" => crate::numbuffers::set_num_buffers(&mut self.frame_limit, &value)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "monitor-index" => Some(PropValue::Int(i64::from(self.monitor_index))),
            "fps" => Some(PropValue::Uint(u64::from(self.fps))),
            "num-buffers" => Some(crate::numbuffers::get_num_buffers(self.frame_limit)),
            _ => None,
        }
    }

    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.clock_sync = Some(sync);
    }

    fn latency(&self) -> LatencyReport {
        LatencyReport::live(self.period().as_nanos() as u64, None)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let bounds = self.bounds.ok_or(G2gError::NotConfigured)?;
            let limit = self.frame_limit;
            if crate::numbuffers::finished_at_zero_limit(limit, out).await? {
                return Ok(0);
            }
            let period = self.period();
            let monitor_index = self.monitor_index;
            // a frame the pipeline has not taken yet is replaced by the newer one
            let (frame_sender, mut frame_receiver) = watch::channel::<Option<CapturedFrame>>(None);
            let (ready_sender, ready_receiver) = std_mpsc::sync_channel(1);
            let worker = thread::Builder::new()
                .name(String::from("g2g-d3d11screencapturesrc"))
                .spawn(move || {
                    let result = with_physical_pixels(|| {
                        capture(monitor_index, bounds, period, &frame_sender, &ready_sender)
                    });
                    if let Err(e) = &result {
                        let _ = ready_sender.try_send(Err(e.clone()));
                    }
                    result
                })
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
            match ready_receiver.recv_timeout(WORKER_START_TIMEOUT) {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    let _ = worker.join();
                    return Err(e);
                }
                Err(_) => return Err(G2gError::Timeout),
            }

            let period_ns = period.as_nanos() as u64;
            let clock_sync = self.clock_sync.clone();
            let mut anchor = CaptureAnchor::new();
            let mut first_capture_ns = None;
            let mut sequence = 0u64;
            let mut downstream_open = true;
            while sequence < limit {
                if frame_receiver.changed().await.is_err() {
                    break;
                }
                let captured = match frame_receiver.borrow_and_update().as_ref() {
                    Some(captured) => CapturedFrame {
                        image: captured.image.share(),
                        capture_ns: captured.capture_ns,
                    },
                    None => continue,
                };
                let elapsed =
                    captured.capture_ns - *first_capture_ns.get_or_insert(captured.capture_ns);
                let pts = match &clock_sync {
                    Some(sync) => anchor.stamp(sync, elapsed, period_ns),
                    None => elapsed,
                };
                let frame = Frame {
                    domain: captured.image,
                    timing: FrameTiming {
                        pts_ns: pts,
                        dts_ns: pts,
                        duration_ns: period_ns,
                        capture_ns: pts,
                        arrival_ns: captured.capture_ns,
                        keyframe: true,
                        ..FrameTiming::default()
                    },
                    sequence,
                    meta: Default::default(),
                };
                if out.push(PipelinePacket::DataFrame(frame)).await.is_err() {
                    downstream_open = false;
                    break;
                }
                sequence += 1;
            }
            drop(frame_receiver);
            let captured = worker
                .join()
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
            if downstream_open {
                captured?;
                out.push(PipelinePacket::Eos).await?;
            }
            Ok(sequence)
        })
    }
}

impl PadTemplates for D3D11ScreenCaptureSrc {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::source(CapsSet::one(Caps::RawVideo {
            format: RawVideoFormat::Bgra8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: Interlace::Progressive,
            colorimetry: Colorimetry::SRGB,
            pixel_aspect: PixelAspect::SQUARE,
        }))])
    }
}

#[derive(Debug)]
struct CapturedFrame {
    image: MemoryDomain,
    capture_ns: u64,
}

#[derive(Debug)]
struct Duplication {
    duplication: IDXGIOutputDuplication,
    context: ID3D11DeviceContext,
    staging: ID3D11Texture2D,
    width: u32,
    height: u32,
}

fn create_device(adapter: &IDXGIAdapter1) -> Result<ID3D11Device, G2gError> {
    let mut device = None;
    // SAFETY: the out-param is a valid local, written by the call before it is read.
    unsafe {
        D3D11CreateDevice(
            adapter,
            D3D_DRIVER_TYPE_UNKNOWN,
            HMODULE::default(),
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            None,
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            None,
        )
        .map_err(mf_err)?;
    }
    device.ok_or(G2gError::Hardware(HardwareError::Other))
}

fn duplicate(selected: &SelectedOutput) -> Result<Duplication, G2gError> {
    let log = Target::category(CATEGORY);
    let device = create_device(&selected.adapter)?;
    // SAFETY: COM calls on interfaces this thread created, with out-params that are valid locals.
    unsafe {
        let context = device.GetImmediateContext().map_err(mf_err)?;
        let output: IDXGIOutput1 = selected.output.cast().map_err(mf_err)?;
        let duplication = output.DuplicateOutput(&device).map_err(mf_err)?;
        let mode = duplication.GetDesc().ModeDesc;
        let bounds = selected.bounds;
        if (mode.Width, mode.Height) != (bounds.width, bounds.height) {
            g2g_error!(
                log,
                "duplicated image is {}x{} but the monitor was {}x{}",
                mode.Width,
                mode.Height,
                bounds.width,
                bounds.height
            );
            return Err(G2gError::CapsMismatch);
        }
        let desc = D3D11_TEXTURE2D_DESC {
            Width: mode.Width,
            Height: mode.Height,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_STAGING,
            BindFlags: 0,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            MiscFlags: 0,
        };
        let mut staging = None;
        device
            .CreateTexture2D(&desc, None, Some(&mut staging))
            .map_err(mf_err)?;
        Ok(Duplication {
            duplication,
            context,
            staging: staging.ok_or(G2gError::Hardware(HardwareError::Other))?,
            width: mode.Width,
            height: mode.Height,
        })
    }
}

impl Duplication {
    fn copy_out(&self, texture: &ID3D11Texture2D) -> Result<Vec<u8>, G2gError> {
        let row_bytes = self.width as usize * BYTES_PER_PIXEL;
        let rows = self.height as usize;
        let mut image = Vec::with_capacity(row_bytes * rows);
        // SAFETY: the staging texture is mapped for reading between Map and Unmap, and each row read
        // stays inside RowPitch * height bytes of the mapping.
        unsafe {
            self.context.CopyResource(&self.staging, texture);
            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            self.context
                .Map(&self.staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
                .map_err(mf_err)?;
            let base = mapped.pData.cast::<u8>().cast_const();
            for row in 0..rows {
                let start = base.add(row * mapped.RowPitch as usize);
                image.extend_from_slice(core::slice::from_raw_parts(start, row_bytes));
            }
            self.context.Unmap(&self.staging, 0);
        }
        Ok(image)
    }

    fn next_image(&self, timeout: Duration) -> Result<Option<Vec<u8>>, G2gError> {
        // AcquireNextFrame counts whole milliseconds, and rounding down returns before the repeat is due
        let timeout_ms = u32::try_from(timeout.as_micros().div_ceil(1000))
            .expect("a capture period fits in u32 milliseconds");
        let mut info = DXGI_OUTDUPL_FRAME_INFO::default();
        let mut resource = None;
        // SAFETY: both out-params are valid locals, and every acquired frame is released below.
        unsafe {
            match self
                .duplication
                .AcquireNextFrame(timeout_ms, &mut info, &mut resource)
            {
                Ok(()) => {}
                Err(e) if e.code() == DXGI_ERROR_WAIT_TIMEOUT => return Ok(None),
                Err(e) => return Err(mf_err(e)),
            }
            // DXGI reports a frame with only a pointer update with LastPresentTime 0
            let image = match resource {
                Some(resource) if info.LastPresentTime != 0 => resource
                    .cast::<ID3D11Texture2D>()
                    .map_err(mf_err)
                    .and_then(|texture| self.copy_out(&texture))
                    .map(Some),
                _ => Ok(None),
            };
            self.duplication.ReleaseFrame().map_err(mf_err)?;
            image
        }
    }
}

fn capture(
    monitor_index: i32,
    bounds: MonitorBounds,
    period: Duration,
    frames: &watch::Sender<Option<CapturedFrame>>,
    ready: &std_mpsc::SyncSender<Result<(), G2gError>>,
) -> Result<(), G2gError> {
    let log = Target::category(CATEGORY);
    let selected = select_output(monitor_index)?;
    if selected.bounds != bounds {
        g2g_error!(
            log,
            "monitor {monitor_index} changed from {bounds:?} to {:?}",
            selected.bounds
        );
        return Err(G2gError::CapsMismatch);
    }
    let duplication = duplicate(&selected)?;
    let _ = ready.try_send(Ok(()));
    let mut last_image: Option<MemoryDomain> = None;
    let mut last_change: Option<Instant> = None;
    let mut last_sent = Instant::now();
    loop {
        if let Some(changed_at) = last_change {
            let next_change = changed_at + period;
            let now = Instant::now();
            if next_change > now {
                thread::sleep(next_change - now);
            }
        }
        let repeat_due = last_sent + period;
        let changed =
            match duplication.next_image(repeat_due.saturating_duration_since(Instant::now())) {
                Ok(changed) => changed,
                Err(e) => {
                    g2g_error!(log, "desktop duplication failed: {e:?}");
                    return Err(e);
                }
            };
        let now = Instant::now();
        if let Some(image) = changed {
            let mut image = MemoryDomain::System(SystemSlice::from_boxed(image.into_boxed_slice()));
            image.make_shareable();
            last_image = Some(image);
            last_change = Some(now);
        } else if now < repeat_due {
            continue;
        }
        let Some(image) = &last_image else {
            continue;
        };
        let captured = CapturedFrame {
            image: image.share(),
            capture_ns: g2g_core::metrics::monotonic_ns(),
        };
        if frames.send(Some(captured)).is_err() {
            return Ok(());
        }
        last_sent = now;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_declared_property_round_trips() {
        let mut source = D3D11ScreenCaptureSrc::new();
        SourceLoop::set_property(&mut source, "monitor-index", PropValue::Int(1)).expect("index");
        SourceLoop::set_property(&mut source, "fps", PropValue::Uint(30)).expect("fps");
        SourceLoop::set_property(&mut source, "num-buffers", PropValue::Int(5)).expect("limit");
        assert_eq!(source.monitor_index, 1);
        assert_eq!(source.fps, 30);
        assert_eq!(source.frame_limit, 5);
        assert_eq!(
            SourceLoop::set_property(&mut source, "fps", PropValue::Uint(0)),
            Err(PropError::Value)
        );
        for spec in D3D11SCREENCAPTURESRC_PROPS {
            assert!(
                SourceLoop::get_property(&source, spec.name).is_some(),
                "{} is declared but not readable",
                spec.name
            );
        }
    }

    #[test]
    fn missing_monitor_fails_the_probe() {
        let mut source = D3D11ScreenCaptureSrc::new().with_monitor_index(i32::MAX);
        assert_eq!(source.probe(), Err(G2gError::DeviceNotFound));
    }
}
