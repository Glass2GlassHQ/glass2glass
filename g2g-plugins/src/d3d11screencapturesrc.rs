// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::cell::Cell;
use core::future::Future;
use core::pin::Pin;

use std::thread_local;
use std::time::{Duration, Instant};

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use tokio::sync::watch;

use windows::core::Interface;
use windows::Win32::Foundation::{E_ACCESSDENIED, GENERIC_ALL, HMODULE, POINT};
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
    IDXGIOutputDuplication, DXGI_ERROR_ACCESS_LOST, DXGI_ERROR_NOT_CURRENTLY_AVAILABLE,
    DXGI_ERROR_NOT_FOUND, DXGI_ERROR_SESSION_DISCONNECTED, DXGI_ERROR_WAIT_TIMEOUT,
    DXGI_OUTDUPL_FRAME_INFO, DXGI_OUTDUPL_POINTER_SHAPE_INFO, DXGI_OUTDUPL_POINTER_SHAPE_TYPE,
    DXGI_OUTDUPL_POINTER_SHAPE_TYPE_COLOR, DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MASKED_COLOR,
    DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MONOCHROME,
};
use windows::Win32::Graphics::Gdi::{MonitorFromPoint, MONITOR_DEFAULTTOPRIMARY};
use windows::Win32::System::StationsAndDesktops::{
    CloseDesktop, OpenInputDesktop, SetThreadDesktop, DESKTOP_ACCESS_FLAGS, DESKTOP_CONTROL_FLAGS,
    HDESK,
};
use windows::Win32::UI::HiDpi::{
    SetThreadDpiAwarenessContext, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2,
};

use crate::cursorshape::CursorShape;
use crate::d3d11texture::{texture_frame, TextureOwner};
use crate::desktopcapture::{run_desktop_capture, DesktopCaptureSettings, DesktopGrabber};
use crate::mfcommon::mf_err;
use crate::texturepool::take_from_pool;
use g2g_core::log::Target;
use g2g_core::memory::{DomainSet, MemoryDomainKind, SystemSlice};
use g2g_core::runtime::SourceLoop;
use g2g_core::{g2g_error, g2g_info, g2g_warn};
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

thread_local! {
    static OPENED_INPUT_DESKTOP: Cell<Option<HDESK>> = const { Cell::new(None) };
}

pub fn switch_to_input_desktop() -> windows::core::Result<()> {
    // SAFETY: the opened handle is closed on failure, or kept until the next switch replaces it.
    unsafe {
        let desktop = OpenInputDesktop(
            DESKTOP_CONTROL_FLAGS(0),
            false,
            DESKTOP_ACCESS_FLAGS(GENERIC_ALL.0),
        )?;
        if let Err(e) = SetThreadDesktop(desktop) {
            CloseDesktop(desktop)?;
            return Err(e);
        }
        // CloseDesktop fails on the desktop this thread is assigned to
        match OPENED_INPUT_DESKTOP.replace(Some(desktop)) {
            Some(previous) => CloseDesktop(previous),
            None => Ok(()),
        }
    }
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
    cursor: watch::Sender<Option<CursorShape>>,
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
            cursor: watch::channel(None).0,
        }
    }

    // None while the pointer is hidden or before the first frame reports it
    pub fn cursor_shapes(&self) -> watch::Receiver<Option<CursorShape>> {
        self.cursor.subscribe()
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
        Ok(bgra_caps(bounds.width, bounds.height, self.fps))
    }
}

fn bgra_caps(width: u32, height: u32, fps: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Bgra8,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        framerate: Rate::Fixed(fps << 16),
        interlace: Interlace::Progressive,
        colorimetry: Colorimetry::SRGB,
        pixel_aspect: PixelAspect::SQUARE,
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
    crate::numbuffers::num_buffers_property("frames to capture then EOS (-1 = forever)"),
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
            let fps = self.fps;
            let out_domain = self.out_domain;
            let cursor = self.cursor.clone();
            let settings = DesktopCaptureSettings {
                thread_name: "g2g-d3d11screencapturesrc",
                period: self.period(),
                frame_limit: limit,
                clock_sync: self.clock_sync.clone(),
            };
            run_desktop_capture(settings, out, move |worker| {
                with_physical_pixels(|| {
                    let mut duplication =
                        open_duplication(monitor_index, bounds, fps, out_domain, cursor)?;
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
    duplication: Option<IDXGIOutputDuplication>,
    output: IDXGIOutput1,
    next_reopen: Instant,
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    copy: ImageCopy,
    width: u32,
    height: u32,
    fps: u32,
    resized: Option<Caps>,
    pointer: PointerTracker,
}

#[derive(Debug)]
struct PointerTracker {
    shape: Option<CursorShape>,
    visible: bool,
    published: watch::Sender<Option<CursorShape>>,
}

impl PointerTracker {
    fn publish(&self) {
        let shown = self.shape.clone().filter(|_| self.visible);
        self.published.send_if_modified(|current| {
            let changed = *current != shown;
            *current = shown;
            changed
        });
    }
}

fn shape_rows(buffer: &[u8], pitch: usize, row_bytes: usize, rows: usize) -> Option<Vec<&[u8]>> {
    (0..rows)
        .map(|row| buffer.get(row * pitch..)?.get(..row_bytes))
        .collect()
}

fn mask_bit(row: &[u8], x: usize) -> bool {
    row[x / 8] & (0x80 >> (x % 8)) != 0
}

const OPAQUE_BLACK: [u8; 4] = [0, 0, 0, 0xff];
const OPAQUE_WHITE: [u8; 4] = [0xff, 0xff, 0xff, 0xff];
const TRANSPARENT: [u8; 4] = [0, 0, 0, 0];
const MASK_XOR: u8 = 0xff;

fn cursor_shape(info: &DXGI_OUTDUPL_POINTER_SHAPE_INFO, buffer: &[u8]) -> Option<CursorShape> {
    let width = info.Width as usize;
    let pitch = info.Pitch as usize;
    let color_rows = |rows| shape_rows(buffer, pitch, width * BYTES_PER_PIXEL, rows);
    let (height, pixels): (u32, Vec<[u8; 4]>) =
        match DXGI_OUTDUPL_POINTER_SHAPE_TYPE(info.Type as i32) {
            DXGI_OUTDUPL_POINTER_SHAPE_TYPE_COLOR => {
                let rows = color_rows(info.Height as usize)?;
                let pixels = rows
                    .iter()
                    .flat_map(|row| row.as_chunks::<BYTES_PER_PIXEL>().0)
                    .map(|bgra| [bgra[2], bgra[1], bgra[0], bgra[3]])
                    .collect();
                (info.Height, pixels)
            }
            DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MASKED_COLOR => {
                let rows = color_rows(info.Height as usize)?;
                let pixels = rows
                    .iter()
                    .flat_map(|row| row.as_chunks::<BYTES_PER_PIXEL>().0)
                    .map(|bgra| match (bgra[3], bgra[..3] == [0, 0, 0]) {
                        (MASK_XOR, true) => TRANSPARENT,
                        // windows inverts the screen under xor pixels, which a client cursor draws black
                        (MASK_XOR, false) => OPAQUE_BLACK,
                        _ => [bgra[2], bgra[1], bgra[0], 0xff],
                    })
                    .collect();
                (info.Height, pixels)
            }
            DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MONOCHROME => {
                // monochrome shapes stack the AND mask rows above the XOR mask rows
                let height = info.Height / 2;
                let rows = shape_rows(buffer, pitch, width.div_ceil(8), info.Height as usize)?;
                let (and_rows, xor_rows) = rows.split_at(height as usize);
                let pixels = and_rows
                    .iter()
                    .zip(xor_rows)
                    .flat_map(|(and_row, xor_row)| {
                        (0..width).map(|x| match (mask_bit(and_row, x), mask_bit(xor_row, x)) {
                            (false, false) => OPAQUE_BLACK,
                            (false, true) => OPAQUE_WHITE,
                            (true, false) => TRANSPARENT,
                            // windows inverts the screen under these, which a client cursor draws black
                            (true, true) => OPAQUE_BLACK,
                        })
                    })
                    .collect();
                (height, pixels)
            }
            _ => return None,
        };
    Some(CursorShape {
        width: info.Width,
        height,
        hotspot_x: info.HotSpot.x,
        hotspot_y: info.HotSpot.y,
        rgba: pixels.concat(),
    })
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

fn image_copy(
    device: &ID3D11Device,
    out_domain: MemoryDomainKind,
    width: u32,
    height: u32,
) -> Result<ImageCopy, G2gError> {
    if out_domain == MemoryDomainKind::D3D11Texture {
        return Ok(ImageCopy::Texture { pool: Vec::new() });
    }
    let staging = bgra_texture(
        device,
        width,
        height,
        D3D11_USAGE_STAGING,
        D3D11_BIND_FLAG(0),
        D3D11_CPU_ACCESS_READ.0 as u32,
    )?;
    Ok(ImageCopy::System { staging })
}

fn duplicate(
    selected: &SelectedOutput,
    fps: u32,
    out_domain: MemoryDomainKind,
    cursor: watch::Sender<Option<CursorShape>>,
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
        switch_to_input_desktop().map_err(mf_err)?;
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
        let copy = image_copy(&device, out_domain, mode.Width, mode.Height)?;
        Ok(Duplication {
            duplication: Some(duplication),
            output,
            next_reopen: Instant::now(),
            device,
            context,
            copy,
            width: mode.Width,
            height: mode.Height,
            fps,
            resized: None,
            pointer: PointerTracker {
                shape: None,
                visible: false,
                published: cursor,
            },
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

impl Duplication {
    fn update_pointer(
        &mut self,
        duplication: &IDXGIOutputDuplication,
        info: &DXGI_OUTDUPL_FRAME_INFO,
    ) {
        if info.LastMouseUpdateTime != 0 {
            self.pointer.visible = info.PointerPosition.Visible.as_bool();
        }
        if info.PointerShapeBufferSize > 0 {
            let mut buffer = vec![0u8; info.PointerShapeBufferSize as usize];
            let mut required = 0;
            let mut shape_info = DXGI_OUTDUPL_POINTER_SHAPE_INFO::default();
            // SAFETY: the buffer holds the size DXGI reported for this frame, and the out-params are valid locals.
            let read = unsafe {
                duplication.GetFramePointerShape(
                    info.PointerShapeBufferSize,
                    buffer.as_mut_ptr().cast(),
                    &mut required,
                    &mut shape_info,
                )
            };
            match read.ok().and_then(|()| cursor_shape(&shape_info, &buffer)) {
                Some(shape) => self.pointer.shape = Some(shape),
                None => g2g_warn!(
                    Target::category(CATEGORY),
                    "cannot read the pointer shape {shape_info:?}"
                ),
            }
        }
        self.pointer.publish();
    }
}

const DESKTOP_UNAVAILABLE_ERRORS: [windows::core::HRESULT; 4] = [
    // processes not running as SYSTEM get E_ACCESSDENIED while a UAC prompt or the lock screen is up
    E_ACCESSDENIED,
    DXGI_ERROR_ACCESS_LOST,
    DXGI_ERROR_NOT_CURRENTLY_AVAILABLE,
    DXGI_ERROR_SESSION_DISCONNECTED,
];

const REOPEN_INTERVAL: Duration = Duration::from_millis(100);

impl Duplication {
    fn reopen(&mut self) -> Result<(), G2gError> {
        let now = Instant::now();
        if now < self.next_reopen {
            return Ok(());
        }
        self.next_reopen = now + REOPEN_INTERVAL;
        // SAFETY: DuplicateOutput takes a device this thread created and returns an owned interface.
        let reopened = switch_to_input_desktop()
            .and_then(|()| unsafe { self.output.DuplicateOutput(&self.device) });
        let duplication = match reopened {
            Ok(duplication) => duplication,
            Err(e) if DESKTOP_UNAVAILABLE_ERRORS.contains(&e.code()) => return Ok(()),
            Err(e) => return Err(mf_err(e)),
        };
        // SAFETY: GetDesc fills a struct and takes no pointers from this code.
        let mode = unsafe { duplication.GetDesc() }.ModeDesc;
        if (mode.Width, mode.Height) != (self.width, self.height) {
            self.resize(mode.Width, mode.Height)?;
        }
        g2g_info!(Target::category(CATEGORY), "desktop duplication reopened");
        self.duplication = Some(duplication);
        Ok(())
    }

    fn resize(&mut self, width: u32, height: u32) -> Result<(), G2gError> {
        g2g_info!(
            Target::category(CATEGORY),
            "desktop is {width}x{height} after a desktop switch, was {}x{}",
            self.width,
            self.height
        );
        let out_domain = match self.copy {
            ImageCopy::System { .. } => MemoryDomainKind::System,
            ImageCopy::Texture { .. } => MemoryDomainKind::D3D11Texture,
        };
        self.copy = image_copy(&self.device, out_domain, width, height)?;
        self.width = width;
        self.height = height;
        self.resized = Some(bgra_caps(width, height, self.fps));
        Ok(())
    }

    fn lose_access(&mut self) {
        g2g_warn!(
            Target::category(CATEGORY),
            "desktop duplication lost access, repeating the last image until it reopens"
        );
        self.duplication = None;
    }
}

impl DesktopGrabber for Duplication {
    fn resized_caps(&mut self) -> Option<Caps> {
        self.resized.take()
    }

    fn next_image(&mut self) -> Result<Option<MemoryDomain>, G2gError> {
        if self.duplication.is_none() {
            self.reopen()?;
        }
        let Some(duplication) = self.duplication.clone() else {
            return Ok(None);
        };
        let mut info = DXGI_OUTDUPL_FRAME_INFO::default();
        let mut resource = None;
        // SAFETY: both out-params are valid locals, and every acquired frame is released below.
        unsafe {
            // AcquireNextFrame holds the device lock while it waits, and the converter and encoder share the device
            match duplication.AcquireNextFrame(0, &mut info, &mut resource) {
                Ok(()) => {}
                Err(e) if e.code() == DXGI_ERROR_WAIT_TIMEOUT => return Ok(None),
                Err(e) if e.code() == DXGI_ERROR_ACCESS_LOST => {
                    self.lose_access();
                    return Ok(None);
                }
                Err(e) => return Err(mf_err(e)),
            }
            self.update_pointer(&duplication, &info);
            // DXGI reports a frame with only a pointer update with LastPresentTime 0
            let image = match resource {
                Some(resource) if info.LastPresentTime != 0 => resource
                    .cast::<ID3D11Texture2D>()
                    .map_err(mf_err)
                    .and_then(|texture| self.copy_out(&texture))
                    .map(Some),
                _ => Ok(None),
            };
            match duplication.ReleaseFrame() {
                Ok(()) => image,
                Err(e) if e.code() == DXGI_ERROR_ACCESS_LOST => {
                    self.lose_access();
                    Ok(None)
                }
                Err(e) => Err(mf_err(e)),
            }
        }
    }
}

fn open_duplication(
    monitor_index: i32,
    bounds: MonitorBounds,
    fps: u32,
    out_domain: MemoryDomainKind,
    cursor: watch::Sender<Option<CursorShape>>,
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
    duplicate(&selected, fps, out_domain, cursor)
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

    fn shape_info(
        shape_type: DXGI_OUTDUPL_POINTER_SHAPE_TYPE,
        width: u32,
        height: u32,
        pitch: u32,
    ) -> DXGI_OUTDUPL_POINTER_SHAPE_INFO {
        DXGI_OUTDUPL_POINTER_SHAPE_INFO {
            Type: shape_type.0 as u32,
            Width: width,
            Height: height,
            Pitch: pitch,
            HotSpot: POINT { x: 1, y: 0 },
        }
    }

    #[test]
    fn color_pointer_rows_skip_the_pitch_padding_and_swap_to_rgba() {
        const PADDING: [u8; 4] = [9, 9, 9, 9];
        let buffer = [[1, 2, 3, 4], [5, 6, 7, 8], PADDING].concat();
        let info = shape_info(DXGI_OUTDUPL_POINTER_SHAPE_TYPE_COLOR, 2, 1, 12);
        let shape = cursor_shape(&info, &buffer).expect("color shape");
        assert_eq!((shape.width, shape.height), (2, 1));
        assert_eq!((shape.hotspot_x, shape.hotspot_y), (1, 0));
        assert_eq!(shape.rgba, [3, 2, 1, 4, 7, 6, 5, 8]);
    }

    #[test]
    fn monochrome_pointer_masks_map_to_black_white_and_transparent() {
        // AND row 0011, XOR row 0101, one byte per row
        let buffer = [0b0011_0000, 0b0101_0000];
        let info = shape_info(DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MONOCHROME, 4, 2, 1);
        let shape = cursor_shape(&info, &buffer).expect("monochrome shape");
        assert_eq!((shape.width, shape.height), (4, 1));
        let expected = [OPAQUE_BLACK, OPAQUE_WHITE, TRANSPARENT, OPAQUE_BLACK].concat();
        assert_eq!(shape.rgba, expected);
    }

    #[test]
    fn masked_color_pointer_keeps_opaque_pixels_and_drops_empty_xor_pixels() {
        let buffer = [[1, 2, 3, 0], [0, 0, 0, MASK_XOR], [4, 5, 6, MASK_XOR]].concat();
        let info = shape_info(DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MASKED_COLOR, 3, 1, 12);
        let shape = cursor_shape(&info, &buffer).expect("masked shape");
        let expected = [[3, 2, 1, 0xff], TRANSPARENT, OPAQUE_BLACK].concat();
        assert_eq!(shape.rgba, expected);
    }

    #[test]
    fn a_pointer_buffer_shorter_than_its_rows_is_refused() {
        let info = shape_info(DXGI_OUTDUPL_POINTER_SHAPE_TYPE_COLOR, 2, 2, 8);
        assert_eq!(cursor_shape(&info, &[0; 12]), None);
    }

    #[test]
    fn missing_monitor_fails_the_probe() {
        let mut source = D3D11ScreenCaptureSrc::new().with_monitor_index(i32::MAX);
        assert_eq!(source.probe(), Err(G2gError::DeviceNotFound));
    }
}
