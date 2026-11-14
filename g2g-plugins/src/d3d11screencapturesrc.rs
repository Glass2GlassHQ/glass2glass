// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::future::Future;
use core::pin::Pin;

use std::time::Duration;

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;

use windows::core::Interface;
use windows::Win32::Foundation::{HMODULE, POINT};
use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_UNKNOWN;
use windows::Win32::Graphics::Direct3D11::{
    D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext, ID3D11Multithread, ID3D11Texture2D,
    D3D11_BIND_FLAG, D3D11_BIND_RENDER_TARGET, D3D11_BIND_SHADER_RESOURCE, D3D11_CPU_ACCESS_READ,
    D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_CREATE_DEVICE_FLAG, D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
    D3D11_MAPPED_SUBRESOURCE, D3D11_MAP_READ, D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC, D3D11_USAGE,
    D3D11_USAGE_DEFAULT, D3D11_USAGE_STAGING,
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

use crate::d3d11texture::{texture_frame, TextureOwner};
use crate::desktopcapture::{run_desktop_capture, DesktopCaptureSettings, DesktopGrabber};
use crate::mfcommon::mf_err;
use crate::texturepool::take_from_pool;
use g2g_core::g2g_error;
use g2g_core::log::Target;
use g2g_core::memory::{DomainSet, MemoryDomainKind, SystemSlice};
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AllocationParams, Caps, CapsConstraint, CapsSet, ClockSync, Colorimetry, ConfigureOutcome, Dim,
    ElementMetadata, G2gError, HardwareError, Interlace, LatencyReport, MemoryDomain, OutputSink,
    PadTemplate, PadTemplates, PixelAspect, PropError, PropKind, PropValue, PropertySpec, Rate,
    RawVideoFormat,
};

const CATEGORY: &str = "d3d11screencapturesrc";
const PRIMARY_MONITOR: i32 = -1;
const DEFAULT_FPS: u32 = 60;
const BYTES_PER_PIXEL: usize = 4;
const NANOS_PER_SECOND: u64 = 1_000_000_000;
const OUTPUT_DOMAINS: DomainSet =
    DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::D3D11Texture);

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
    out_domain: MemoryDomainKind,
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
            out_domain: MemoryDomainKind::System,
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
            "Captures a desktop monitor through DXGI Desktop Duplication as BGRA, in system memory or a D3D11 texture",
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

    fn output_memory(&self) -> MemoryDomainKind {
        self.out_domain
    }

    fn output_domains(&self) -> DomainSet {
        OUTPUT_DOMAINS
    }

    fn configure_allocation(&mut self, params: &AllocationParams) {
        if let Ok(resolved) = params.clone().resolve_for_producer(OUTPUT_DOMAINS) {
            self.out_domain = resolved.domain;
        }
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
            let monitor_index = self.monitor_index;
            let out_domain = self.out_domain;
            let settings = DesktopCaptureSettings {
                thread_name: "g2g-d3d11screencapturesrc",
                period: self.period(),
                frame_limit: limit,
                clock_sync: self.clock_sync.clone(),
            };
            run_desktop_capture(settings, out, move |worker| {
                with_physical_pixels(|| {
                    let mut duplication = open_duplication(monitor_index, bounds, out_domain)?;
                    worker.pace(&mut duplication).inspect_err(|e| {
                        g2g_error!(
                            Target::category(CATEGORY),
                            "desktop duplication failed: {e:?}"
                        )
                    })
                })
            })
            .await
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
enum ImageCopy {
    System { staging: ID3D11Texture2D },
    Texture { pool: Vec<Arc<TextureOwner>> },
}

#[derive(Debug)]
struct Duplication {
    duplication: IDXGIOutputDuplication,
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    copy: ImageCopy,
    width: u32,
    height: u32,
}

fn create_device(
    adapter: &IDXGIAdapter1,
    flags: D3D11_CREATE_DEVICE_FLAG,
) -> Result<ID3D11Device, G2gError> {
    let mut device = None;
    // SAFETY: the out-param is a valid local, written by the call before it is read.
    unsafe {
        D3D11CreateDevice(
            adapter,
            D3D_DRIVER_TYPE_UNKNOWN,
            HMODULE::default(),
            flags,
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

fn bgra_texture(
    device: &ID3D11Device,
    width: u32,
    height: u32,
    usage: D3D11_USAGE,
    bind: D3D11_BIND_FLAG,
    cpu_access: u32,
) -> Result<ID3D11Texture2D, G2gError> {
    let desc = D3D11_TEXTURE2D_DESC {
        Width: width,
        Height: height,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: usage,
        BindFlags: bind.0 as u32,
        CPUAccessFlags: cpu_access,
        MiscFlags: 0,
    };
    let mut texture = None;
    // SAFETY: `desc` and the out-param are valid locals for the call.
    unsafe {
        device
            .CreateTexture2D(&desc, None, Some(&mut texture))
            .map_err(mf_err)?;
    }
    texture.ok_or(G2gError::Hardware(HardwareError::Other))
}

fn duplicate(
    selected: &SelectedOutput,
    out_domain: MemoryDomainKind,
) -> Result<Duplication, G2gError> {
    let log = Target::category(CATEGORY);
    let to_texture = out_domain == MemoryDomainKind::D3D11Texture;
    let flags = match to_texture {
        // the texture goes to a d3d11 video processor on this device
        true => D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
        false => D3D11_CREATE_DEVICE_BGRA_SUPPORT,
    };
    let device = create_device(&selected.adapter, flags)?;
    // SAFETY: COM calls on interfaces this thread created, with out-params that are valid locals.
    unsafe {
        let context = device.GetImmediateContext().map_err(mf_err)?;
        if to_texture {
            // d3d11 immediate contexts are not thread safe, and the consumer of the texture uses this one too
            let multithread: ID3D11Multithread = context.cast().map_err(mf_err)?;
            let _ = multithread.SetMultithreadProtected(true);
        }
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
        let copy = match to_texture {
            true => ImageCopy::Texture { pool: Vec::new() },
            false => ImageCopy::System {
                staging: bgra_texture(
                    &device,
                    mode.Width,
                    mode.Height,
                    D3D11_USAGE_STAGING,
                    D3D11_BIND_FLAG(0),
                    D3D11_CPU_ACCESS_READ.0 as u32,
                )?,
            },
        };
        Ok(Duplication {
            duplication,
            device,
            context,
            copy,
            width: mode.Width,
            height: mode.Height,
        })
    }
}

impl Duplication {
    fn copy_out(&mut self, texture: &ID3D11Texture2D) -> Result<MemoryDomain, G2gError> {
        let staging = match &self.copy {
            ImageCopy::System { staging } => staging,
            ImageCopy::Texture { .. } => return self.copy_to_texture(texture),
        };
        let row_bytes = self.width as usize * BYTES_PER_PIXEL;
        let rows = self.height as usize;
        let mut image = Vec::with_capacity(row_bytes * rows);
        // SAFETY: the staging texture is mapped for reading between Map and Unmap, and each row read
        // stays inside RowPitch * height bytes of the mapping.
        unsafe {
            self.context.CopyResource(staging, texture);
            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            self.context
                .Map(staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
                .map_err(mf_err)?;
            let base = mapped.pData.cast::<u8>().cast_const();
            for row in 0..rows {
                let start = base.add(row * mapped.RowPitch as usize);
                image.extend_from_slice(core::slice::from_raw_parts(start, row_bytes));
            }
            self.context.Unmap(staging, 0);
        }
        Ok(MemoryDomain::System(SystemSlice::from_boxed(
            image.into_boxed_slice(),
        )))
    }

    fn copy_to_texture(&mut self, texture: &ID3D11Texture2D) -> Result<MemoryDomain, G2gError> {
        let ImageCopy::Texture { pool } = &mut self.copy else {
            unreachable!("copy_out sends only texture capture here");
        };
        // dxgi reuses the duplicated texture for the next frame, so each frame gets a copy
        let copy = take_from_pool(pool, || {
            bgra_texture(
                &self.device,
                self.width,
                self.height,
                D3D11_USAGE_DEFAULT,
                D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE,
                0,
            )
            .map(TextureOwner)
        })?;
        // SAFETY: both textures belong to this device and match in size and format.
        unsafe { self.context.CopyResource(&copy.0, texture) };
        Ok(texture_frame(
            &self.device,
            copy,
            self.width,
            self.height,
            DXGI_FORMAT_B8G8R8A8_UNORM,
        ))
    }
}

impl DesktopGrabber for Duplication {
    fn next_image(&mut self) -> Result<Option<MemoryDomain>, G2gError> {
        let mut info = DXGI_OUTDUPL_FRAME_INFO::default();
        let mut resource = None;
        // SAFETY: both out-params are valid locals, and every acquired frame is released below.
        unsafe {
            // AcquireNextFrame holds the device lock while it waits, and the converter and encoder share the device
            match self
                .duplication
                .AcquireNextFrame(0, &mut info, &mut resource)
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

fn open_duplication(
    monitor_index: i32,
    bounds: MonitorBounds,
    out_domain: MemoryDomainKind,
) -> Result<Duplication, G2gError> {
    let selected = select_output(monitor_index)?;
    if selected.bounds != bounds {
        g2g_error!(
            Target::category(CATEGORY),
            "monitor {monitor_index} changed from {bounds:?} to {:?}",
            selected.bounds
        );
        return Err(G2gError::CapsMismatch);
    }
    duplicate(&selected, out_domain)
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
