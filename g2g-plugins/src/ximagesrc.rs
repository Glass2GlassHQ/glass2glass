// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::fmt::Debug;
use core::future::Future;
use core::pin::Pin;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use x11rb::connection::Connection;
use x11rb::protocol::damage::{ConnectionExt as _, ReportLevel};
use x11rb::protocol::shm::ConnectionExt as _;
use x11rb::protocol::xfixes::{ConnectionExt as _, CursorNotifyMask};
use x11rb::protocol::xproto::{
    AtomEnum, ChangeWindowAttributesAux, ConnectionExt as _, EventMask, ImageFormat, ImageOrder,
    Window,
};
use x11rb::protocol::Event;
use x11rb::rust_connection::RustConnection;

use g2g_core::log::Target;
use g2g_core::memory::{DomainSet, MemoryDomainKind, SystemSlice};
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, ClockSync, Colorimetry, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    HardwareError, Interlace, LatencyReport, MemoryDomain, OutputSink, PadTemplate, PadTemplates,
    PixelAspect, PropError, PropKind, PropValue, PropertySpec, Rate, RawVideoFormat,
};

use crate::desktopcapture::{run_desktop_capture, DesktopCaptureSettings, DesktopGrabber};
use crate::videosizeandrate::{VideoSizeAndRate, MAX_REQUESTED_FPS, REQUESTED_DIMENSION};

const CATEGORY: &str = "ximagesrc";
// gst's ximagesrc fixates an open framerate to 25/1
const DEFAULT_FPS: u32 = 25;
const REQUESTED_FRAMERATE: Rate = Rate::Range {
    min_q16: 1 << 16,
    max_q16: MAX_REQUESTED_FPS << 16,
};
const FORMAT: RawVideoFormat = RawVideoFormat::Bgrx8;
const BYTES_PER_PIXEL: usize = 4;
const ALL_PLANES: u32 = !0;
const MIN_SHM_VERSION: (u16, u16) = (1, 2);
const DAMAGE_VERSION: (u32, u32) = (1, 1);
const XFIXES_VERSION: (u32, u32) = (5, 0);
const RED_MASK: u32 = 0x00FF_0000;
const GREEN_MASK: u32 = 0x0000_FF00;
const BLUE_MASK: u32 = 0x0000_00FF;
const ROOT_WINDOW: u64 = 0;

fn x11_failure(context: &str, error: impl Debug) -> G2gError {
    g2g_core::g2g_error!(Target::category(CATEGORY), "{context}: {error:?}");
    G2gError::Hardware(HardwareError::Other)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CaptureArea {
    window: Window,
    x: i16,
    y: i16,
    width: u16,
    height: u16,
}

impl CaptureArea {
    fn bytes(&self) -> usize {
        usize::from(self.width) * usize::from(self.height) * BYTES_PER_PIXEL
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Region {
    start_x: u32,
    start_y: u32,
    end_x: u32,
    end_y: u32,
}

fn region_span(start: u32, end: u32, extent: u16) -> Option<(i16, u16)> {
    // gst's startx..endx is inclusive, and an end of 0 is the far edge
    let last = if end == 0 {
        u32::from(extent).checked_sub(1)?
    } else {
        end
    };
    if start > last || last >= u32::from(extent) {
        return None;
    }
    let offset = i16::try_from(start).ok()?;
    let length = u16::try_from(last - start + 1).ok()?;
    Some((offset, length))
}

#[derive(Debug, Clone, Default)]
struct CaptureTarget {
    display_name: Option<String>,
    xid: u64,
    xname: Option<String>,
    region: Region,
}

fn connect(display_name: Option<&str>) -> Result<(RustConnection, usize), G2gError> {
    x11rb::connect(display_name).map_err(|error| x11_failure("connecting to the X display", error))
}

fn window_name(connection: &RustConnection, window: Window) -> Result<Vec<u8>, G2gError> {
    let reply = connection
        .get_property(false, window, AtomEnum::WM_NAME, AtomEnum::ANY, 0, u32::MAX)
        .map_err(|error| x11_failure("reading a window name", error))?
        .reply()
        .map_err(|error| x11_failure("reading a window name", error))?;
    Ok(reply.value)
}

fn find_window_by_name(
    connection: &RustConnection,
    root: Window,
    name: &str,
) -> Result<Option<Window>, G2gError> {
    // gst matches WM_NAME exactly, searching the window tree depth first
    let mut pending = Vec::from([root]);
    while let Some(window) = pending.pop() {
        if window_name(connection, window)? == name.as_bytes() {
            return Ok(Some(window));
        }
        let tree = connection
            .query_tree(window)
            .map_err(|error| x11_failure("listing child windows", error))?
            .reply()
            .map_err(|error| x11_failure("listing child windows", error))?;
        pending.extend(tree.children.iter().rev());
    }
    Ok(None)
}

fn require_bgrx(
    connection: &RustConnection,
    screen: usize,
    window: Window,
) -> Result<(), G2gError> {
    let setup = connection.setup();
    let attributes = connection
        .get_window_attributes(window)
        .map_err(|error| x11_failure("reading the window attributes", error))?
        .reply()
        .map_err(|error| x11_failure("reading the window attributes", error))?;
    let depth = setup.roots[screen]
        .allowed_depths
        .iter()
        .find(|depth| {
            depth
                .visuals
                .iter()
                .any(|visual| visual.visual_id == attributes.visual)
        })
        .ok_or(G2gError::CapsMismatch)?;
    let visual = depth
        .visuals
        .iter()
        .find(|visual| visual.visual_id == attributes.visual)
        .ok_or(G2gError::CapsMismatch)?;
    let bits_per_pixel = setup
        .pixmap_formats
        .iter()
        .find(|format| format.depth == depth.depth)
        .map(|format| format.bits_per_pixel);
    let bgrx = setup.image_byte_order == ImageOrder::LSB_FIRST
        && bits_per_pixel == Some(32)
        && (visual.red_mask, visual.green_mask, visual.blue_mask)
            == (RED_MASK, GREEN_MASK, BLUE_MASK);
    if !bgrx {
        g2g_core::g2g_error!(
            Target::category(CATEGORY),
            "the window's pixels are not 32-bit BGRx: depth {}, {bits_per_pixel:?} bits per pixel",
            depth.depth
        );
        return Err(G2gError::CapsMismatch);
    }
    Ok(())
}

fn capture_area(target: &CaptureTarget) -> Result<CaptureArea, G2gError> {
    let (connection, screen) = connect(target.display_name.as_deref())?;
    let root = connection.setup().roots[screen].root;
    let window = match (&target.xname, target.xid) {
        (Some(name), _) => find_window_by_name(&connection, root, name)?.ok_or_else(|| {
            g2g_core::g2g_error!(Target::category(CATEGORY), "no window is named {name:?}");
            G2gError::CapsMismatch
        })?,
        (None, ROOT_WINDOW) => root,
        (None, xid) => Window::try_from(xid).map_err(|_| G2gError::CapsMismatch)?,
    };
    require_bgrx(&connection, screen, window)?;
    let geometry = connection
        .get_geometry(window)
        .map_err(|error| x11_failure("reading the window size", error))?
        .reply()
        .map_err(|error| x11_failure("reading the window size", error))?;
    let region = target.region;
    let spans = region_span(region.start_x, region.end_x, geometry.width).zip(region_span(
        region.start_y,
        region.end_y,
        geometry.height,
    ));
    let Some(((x, width), (y, height))) = spans else {
        g2g_core::g2g_error!(
            Target::category(CATEGORY),
            "the region {region:?} is outside the {}x{} window",
            geometry.width,
            geometry.height
        );
        return Err(G2gError::CapsMismatch);
    };
    Ok(CaptureArea {
        window,
        x,
        y,
        width,
        height,
    })
}

#[derive(Debug)]
struct SharedImage {
    fd: OwnedFd,
    address: *mut libc::c_void,
    length: usize,
}

impl SharedImage {
    fn new(length: usize) -> Result<Self, G2gError> {
        // SAFETY: memfd_create takes a NUL-terminated name and returns a fresh descriptor or -1.
        let raw = unsafe { libc::memfd_create(c"g2g-ximagesrc".as_ptr(), libc::MFD_CLOEXEC) };
        if raw < 0 {
            return Err(x11_failure(
                "creating the shared image",
                std::io::Error::last_os_error(),
            ));
        }
        // SAFETY: raw is the descriptor memfd_create just returned and nothing else owns it.
        let fd = unsafe { OwnedFd::from_raw_fd(raw) };
        let size = libc::off_t::try_from(length).map_err(|_| G2gError::CapsMismatch)?;
        // SAFETY: fd is an open memfd.
        if unsafe { libc::ftruncate(fd.as_raw_fd(), size) } < 0 {
            return Err(x11_failure(
                "sizing the shared image",
                std::io::Error::last_os_error(),
            ));
        }
        // SAFETY: a null hint lets the kernel place the mapping, and fd is a memfd of length bytes.
        let address = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                length,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if address == libc::MAP_FAILED {
            return Err(x11_failure(
                "mapping the shared image",
                std::io::Error::last_os_error(),
            ));
        }
        Ok(Self {
            fd,
            address,
            length,
        })
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: the mapping covers length readable bytes for as long as self lives.
        unsafe { core::slice::from_raw_parts(self.address.cast::<u8>(), self.length) }
    }
}

impl Drop for SharedImage {
    fn drop(&mut self) {
        // SAFETY: address and length are the mapping new made, unmapped only here.
        unsafe { libc::munmap(self.address, self.length) };
    }
}

// SAFETY: the mapping is plain memory owned by this value, and only the capture thread reads it.
unsafe impl Send for SharedImage {}

#[derive(Debug)]
struct CursorImage {
    width: usize,
    hot_x: i32,
    hot_y: i32,
    // xfixes cursor pixels are premultiplied ARGB
    pixels: Vec<u32>,
}

fn blend_channel(destination: u8, source: u32, alpha: u32) -> u8 {
    let blended = source + u32::from(destination) * (255 - alpha) / 255;
    blended.min(255) as u8
}

fn draw_cursor(picture: &mut [u8], area: CaptureArea, cursor: &CursorImage, x: i32, y: i32) {
    let picture_width = i32::from(area.width);
    let picture_height = i32::from(area.height);
    let left = x - cursor.hot_x - i32::from(area.x);
    let top = y - cursor.hot_y - i32::from(area.y);
    for (row, pixels) in cursor.pixels.chunks_exact(cursor.width).enumerate() {
        let picture_y = top + row as i32;
        if !(0..picture_height).contains(&picture_y) {
            continue;
        }
        for (column, &argb) in pixels.iter().enumerate() {
            let picture_x = left + column as i32;
            if !(0..picture_width).contains(&picture_x) {
                continue;
            }
            let offset = (picture_y * picture_width + picture_x) as usize * BYTES_PER_PIXEL;
            let alpha = argb >> 24;
            let pixel = &mut picture[offset..offset + 3];
            for (channel, shift) in pixel.iter_mut().zip([0, 8, 16]) {
                *channel = blend_channel(*channel, (argb >> shift) & 0xFF, alpha);
            }
        }
    }
}

#[derive(Debug)]
struct PointerOverlay {
    cursor: Option<CursorImage>,
    position: Option<(i32, i32)>,
}

struct XGrabber {
    connection: RustConnection,
    target: CaptureTarget,
    framerate_q16: u32,
    area: CaptureArea,
    segment: u32,
    image: SharedImage,
    damage: Option<u32>,
    pointer: Option<PointerOverlay>,
    captured_once: bool,
    resized: Option<Caps>,
}

#[derive(Debug, Default)]
struct XEvents {
    damaged: bool,
    cursor_changed: bool,
    reconfigured: bool,
}

fn attach_image(connection: &RustConnection, bytes: usize) -> Result<(u32, SharedImage), G2gError> {
    let image = SharedImage::new(bytes)?;
    let segment = connection
        .generate_id()
        .map_err(|error| x11_failure("allocating the shm segment id", error))?;
    let shared_fd = image
        .fd
        .try_clone()
        .map_err(|error| x11_failure("duplicating the shared image", error))?;
    connection
        .shm_attach_fd(segment, shared_fd, false)
        .map_err(|error| x11_failure("attaching the shared image", error))?
        .check()
        .map_err(|error| x11_failure("attaching the shared image", error))?;
    Ok((segment, image))
}

impl Debug for XGrabber {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("XGrabber")
            .field("area", &self.area)
            .field("damage", &self.damage)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone)]
struct GrabSettings {
    target: CaptureTarget,
    framerate_q16: u32,
    area: CaptureArea,
    show_pointer: bool,
    use_damage: bool,
}

impl XGrabber {
    fn open(settings: GrabSettings) -> Result<Self, G2gError> {
        let (connection, _) = connect(settings.target.display_name.as_deref())?;
        let area = settings.area;
        let version = connection
            .shm_query_version()
            .map_err(|error| x11_failure("querying MIT-SHM", error))?
            .reply()
            .map_err(|error| x11_failure("querying MIT-SHM", error))?;
        if (version.major_version, version.minor_version) < MIN_SHM_VERSION {
            return Err(x11_failure("MIT-SHM is older than 1.2", version));
        }
        let (segment, image) = attach_image(&connection, area.bytes())?;
        // a resized window or screen sends ConfigureNotify to clients that select structure events on it
        connection
            .change_window_attributes(
                area.window,
                &ChangeWindowAttributesAux::new().event_mask(EventMask::STRUCTURE_NOTIFY),
            )
            .map_err(|error| x11_failure("watching the window size", error))?
            .check()
            .map_err(|error| x11_failure("watching the window size", error))?;
        let extensions_needed = settings.use_damage || settings.show_pointer;
        if extensions_needed {
            let (major, minor) = XFIXES_VERSION;
            connection
                .xfixes_query_version(major, minor)
                .map_err(|error| x11_failure("querying XFIXES", error))?
                .reply()
                .map_err(|error| x11_failure("querying XFIXES", error))?;
        }
        let damage = match settings.use_damage {
            true => Some(watch_damage(&connection, area.window)?),
            false => None,
        };
        let pointer = match settings.show_pointer {
            true => {
                connection
                    .xfixes_select_cursor_input(area.window, CursorNotifyMask::DISPLAY_CURSOR)
                    .map_err(|error| x11_failure("watching the cursor", error))?
                    .check()
                    .map_err(|error| x11_failure("watching the cursor", error))?;
                Some(PointerOverlay {
                    cursor: None,
                    position: None,
                })
            }
            false => None,
        };
        Ok(Self {
            connection,
            target: settings.target,
            framerate_q16: settings.framerate_q16,
            area,
            segment,
            image,
            damage,
            pointer,
            captured_once: false,
            resized: None,
        })
    }

    fn follow_resize(&mut self) -> Result<(), G2gError> {
        let area = capture_area(&self.target)?;
        if (area.width, area.height) != (self.area.width, self.area.height) {
            let (segment, image) = attach_image(&self.connection, area.bytes())?;
            // the X server also detaches the segment when the connection closes
            let _ = self.connection.shm_detach(self.segment);
            self.segment = segment;
            self.image = image;
            self.resized = caps_for(
                FORMAT,
                Dim::Fixed(u32::from(area.width)),
                Dim::Fixed(u32::from(area.height)),
                Rate::Fixed(self.framerate_q16),
            );
        }
        self.area = area;
        Ok(())
    }

    fn cursor_image(&self) -> Result<CursorImage, G2gError> {
        let reply = self
            .connection
            .xfixes_get_cursor_image()
            .map_err(|error| x11_failure("reading the cursor", error))?
            .reply()
            .map_err(|error| x11_failure("reading the cursor", error))?;
        let width = usize::from(reply.width);
        if width == 0 || reply.cursor_image.len() != width * usize::from(reply.height) {
            return Err(x11_failure(
                "the cursor image size",
                reply.cursor_image.len(),
            ));
        }
        Ok(CursorImage {
            width,
            hot_x: i32::from(reply.xhot),
            hot_y: i32::from(reply.yhot),
            pixels: reply.cursor_image,
        })
    }

    fn pointer_position(&self) -> Result<Option<(i32, i32)>, G2gError> {
        let reply = self
            .connection
            .query_pointer(self.area.window)
            .map_err(|error| x11_failure("reading the pointer position", error))?
            .reply()
            .map_err(|error| x11_failure("reading the pointer position", error))?;
        Ok(reply
            .same_screen
            .then(|| (i32::from(reply.win_x), i32::from(reply.win_y))))
    }

    fn take_events(&mut self) -> Result<XEvents, G2gError> {
        let mut events = XEvents::default();
        while let Some(event) = self
            .connection
            .poll_for_event()
            .map_err(|error| x11_failure("reading X events", error))?
        {
            match event {
                Event::DamageNotify(_) => events.damaged = true,
                Event::XfixesCursorNotify(_) => events.cursor_changed = true,
                Event::ConfigureNotify(notify) if notify.window == self.area.window => {
                    events.reconfigured = true;
                }
                Event::Error(error) => {
                    return Err(x11_failure("the X server failed a request", error))
                }
                _ => {}
            }
        }
        Ok(events)
    }
}

fn watch_damage(connection: &RustConnection, window: Window) -> Result<u32, G2gError> {
    let (major, minor) = DAMAGE_VERSION;
    connection
        .damage_query_version(major, minor)
        .map_err(|error| x11_failure("querying DAMAGE", error))?
        .reply()
        .map_err(|error| x11_failure("querying DAMAGE", error))?;
    let damage = connection
        .generate_id()
        .map_err(|error| x11_failure("allocating the damage id", error))?;
    connection
        .damage_create(damage, window, ReportLevel::NON_EMPTY)
        .map_err(|error| x11_failure("watching damage", error))?
        .check()
        .map_err(|error| x11_failure("watching damage", error))?;
    Ok(damage)
}

impl DesktopGrabber for XGrabber {
    fn next_image(&mut self) -> Result<Option<MemoryDomain>, G2gError> {
        let XEvents {
            damaged,
            cursor_changed,
            reconfigured,
        } = self.take_events()?;
        if reconfigured {
            self.follow_resize()?;
        }
        let mut changed =
            !self.captured_once || self.damage.is_none() || damaged || self.resized.is_some();
        if let (Some(damage), true) = (self.damage, damaged) {
            // damage reports once until the damaged region is cleared
            self.connection
                .damage_subtract(damage, x11rb::NONE, x11rb::NONE)
                .map_err(|error| x11_failure("clearing damage", error))?;
        }
        if let Some(mut pointer) = self.pointer.take() {
            if cursor_changed || pointer.cursor.is_none() {
                pointer.cursor = Some(self.cursor_image()?);
                changed = true;
            }
            let position = self.pointer_position()?;
            changed |= position != pointer.position;
            pointer.position = position;
            self.pointer = Some(pointer);
        }
        if !changed {
            return Ok(None);
        }
        let area = self.area;
        let reply = self
            .connection
            .shm_get_image(
                area.window,
                area.x,
                area.y,
                area.width,
                area.height,
                ALL_PLANES,
                ImageFormat::Z_PIXMAP.into(),
                self.segment,
                0,
            )
            .map_err(|error| x11_failure("capturing the window", error))?
            .reply()
            .map_err(|error| x11_failure("capturing the window", error))?;
        let length = area.bytes();
        if usize::try_from(reply.size).ok() != Some(length) {
            return Err(x11_failure("the captured image size", reply.size));
        }
        let mut picture = self.image.bytes()[..length].to_owned().into_boxed_slice();
        if let Some(PointerOverlay {
            cursor: Some(cursor),
            position: Some((x, y)),
        }) = &self.pointer
        {
            draw_cursor(&mut picture, area, cursor, *x, *y);
        }
        self.captured_once = true;
        Ok(Some(MemoryDomain::System(SystemSlice::from_boxed(picture))))
    }

    fn resized_caps(&mut self) -> Option<Caps> {
        self.resized.take()
    }
}

impl Drop for XGrabber {
    fn drop(&mut self) {
        // the X server also detaches the segment when the connection closes
        let _ = self.connection.shm_detach(self.segment);
    }
}

#[derive(Debug)]
pub struct XImageSrc {
    target: CaptureTarget,
    show_pointer: bool,
    use_damage: bool,
    size_and_rate: Option<VideoSizeAndRate>,
    framerate_q16: u32,
    area: Option<CaptureArea>,
    frame_limit: u64,
    configured: bool,
    clock_sync: Option<ClockSync>,
}

impl Default for XImageSrc {
    fn default() -> Self {
        Self::new()
    }
}

impl XImageSrc {
    pub fn new() -> Self {
        Self {
            target: CaptureTarget::default(),
            show_pointer: true,
            use_damage: true,
            size_and_rate: None,
            framerate_q16: DEFAULT_FPS << 16,
            area: None,
            frame_limit: u64::MAX,
            configured: false,
            clock_sync: None,
        }
    }

    pub fn with_display_name(mut self, display_name: impl Into<String>) -> Self {
        self.target.display_name = Some(display_name.into());
        self
    }

    pub fn with_xid(mut self, xid: u32) -> Self {
        self.target.xid = u64::from(xid);
        self
    }

    pub fn with_num_buffers(mut self, frame_limit: u64) -> Self {
        self.frame_limit = frame_limit;
        self
    }

    fn probe(&mut self) -> Result<VideoSizeAndRate, G2gError> {
        if let Some(size_and_rate) = self.size_and_rate {
            return Ok(size_and_rate);
        }
        let area = capture_area(&self.target)?;
        let mut size_and_rate = VideoSizeAndRate::new(
            u32::from(area.width),
            u32::from(area.height),
            self.framerate_q16,
        );
        size_and_rate.pin_width(u32::from(area.width));
        size_and_rate.pin_height(u32::from(area.height));
        self.area = Some(area);
        self.size_and_rate = Some(size_and_rate);
        Ok(size_and_rate)
    }

    fn forget_probe(&mut self) {
        self.size_and_rate = None;
        self.area = None;
    }

    fn period_ns(&self) -> u64 {
        g2g_core::caps::framerate_q16_period_ns(self.framerate_q16).unwrap_or(0)
    }
}

pub(crate) const DECLARED_CAPS: Caps = Caps::RawVideo {
    format: FORMAT,
    width: REQUESTED_DIMENSION,
    height: REQUESTED_DIMENSION,
    framerate: REQUESTED_FRAMERATE,
    interlace: Interlace::Progressive,
    colorimetry: Colorimetry::SRGB,
    pixel_aspect: PixelAspect::SQUARE,
};

fn caps_for(format: RawVideoFormat, width: Dim, height: Dim, framerate: Rate) -> Option<Caps> {
    Some(Caps::RawVideo {
        format,
        width,
        height,
        framerate,
        interlace: Interlace::Progressive,
        colorimetry: Colorimetry::SRGB,
        pixel_aspect: PixelAspect::SQUARE,
    })
}

static XIMAGESRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "display-name",
        PropKind::Str,
        "X display to capture, as in DISPLAY (unset = DISPLAY)",
    ),
    PropertySpec::new(
        "xid",
        PropKind::Uint,
        "window to capture by id (0 = the whole screen)",
    )
    .with_default("0"),
    PropertySpec::new(
        "xname",
        PropKind::Str,
        "window to capture by its exact WM_NAME, instead of xid",
    ),
    PropertySpec::new("startx", PropKind::Uint, "left edge of the captured area").with_default("0"),
    PropertySpec::new("starty", PropKind::Uint, "top edge of the captured area").with_default("0"),
    PropertySpec::new(
        "endx",
        PropKind::Uint,
        "right edge of the captured area, inclusive (0 = right edge of the window)",
    )
    .with_default("0"),
    PropertySpec::new(
        "endy",
        PropKind::Uint,
        "bottom edge of the captured area, inclusive (0 = bottom edge of the window)",
    )
    .with_default("0"),
    PropertySpec::new(
        "show-pointer",
        PropKind::Bool,
        "draw the mouse pointer into the frames",
    )
    .with_default("true"),
    PropertySpec::new(
        "use-damage",
        PropKind::Bool,
        "capture only after XDamage reports a change, instead of every frame period",
    )
    .with_default("true"),
    crate::numbuffers::num_buffers_property("frames to capture then EOS (-1 = forever)"),
];

impl SourceLoop for XImageSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        let caps = self.probe().and_then(|size_and_rate| {
            caps_for(
                FORMAT,
                Dim::Fixed(size_and_rate.width()),
                Dim::Fixed(size_and_rate.height()),
                Rate::Fixed(size_and_rate.framerate_q16()),
            )
            .ok_or(G2gError::CapsMismatch)
        });
        core::future::ready(caps)
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        let produced = self.probe().map(|size_and_rate| {
            CapsConstraint::Produces(size_and_rate.produced_caps(
                &[FORMAT],
                REQUESTED_FRAMERATE,
                caps_for,
            ))
        });
        core::future::ready(produced)
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let mut size_and_rate = self.probe()?;
        if !size_and_rate
            .produced_caps(&[FORMAT], REQUESTED_FRAMERATE, caps_for)
            .accepts(absolute_caps)
        {
            return Err(G2gError::CapsMismatch);
        }
        size_and_rate.configure(absolute_caps)?;
        self.framerate_q16 = size_and_rate.framerate_q16();
        self.size_and_rate = Some(size_and_rate);
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "X11 screen capture source",
            "Source/Video",
            "Captures an X11 screen or window through MIT-SHM as BGRx",
            crate::AARON_BOXER,
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        XIMAGESRC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let as_u32 = |value: &PropValue| {
            value
                .as_uint()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or(PropError::Value)
        };
        let as_string =
            |value: &PropValue| value.as_str().map(str::to_owned).ok_or(PropError::Type);
        match name {
            "display-name" => self.target.display_name = Some(as_string(&value)?),
            "xid" => self.target.xid = u64::from(as_u32(&value)?),
            "xname" => self.target.xname = Some(as_string(&value)?),
            "startx" => self.target.region.start_x = as_u32(&value)?,
            "starty" => self.target.region.start_y = as_u32(&value)?,
            "endx" => self.target.region.end_x = as_u32(&value)?,
            "endy" => self.target.region.end_y = as_u32(&value)?,
            "show-pointer" => self.show_pointer = value.as_bool().ok_or(PropError::Type)?,
            "use-damage" => self.use_damage = value.as_bool().ok_or(PropError::Type)?,
            "num-buffers" => crate::numbuffers::set_num_buffers(&mut self.frame_limit, &value)?,
            _ => return Err(PropError::Unknown),
        }
        self.forget_probe();
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        let region = self.target.region;
        match name {
            "display-name" => self.target.display_name.clone().map(PropValue::Str),
            "xid" => Some(PropValue::Uint(self.target.xid)),
            "xname" => self.target.xname.clone().map(PropValue::Str),
            "startx" => Some(PropValue::Uint(region.start_x.into())),
            "starty" => Some(PropValue::Uint(region.start_y.into())),
            "endx" => Some(PropValue::Uint(region.end_x.into())),
            "endy" => Some(PropValue::Uint(region.end_y.into())),
            "show-pointer" => Some(PropValue::Bool(self.show_pointer)),
            "use-damage" => Some(PropValue::Bool(self.use_damage)),
            "num-buffers" => Some(crate::numbuffers::get_num_buffers(self.frame_limit)),
            _ => None,
        }
    }

    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.clock_sync = Some(sync);
    }

    fn output_memory(&self) -> MemoryDomainKind {
        MemoryDomainKind::System
    }

    fn output_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System)
    }

    fn latency(&self) -> LatencyReport {
        LatencyReport::live(self.period_ns(), None)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let area = self.area.ok_or(G2gError::NotConfigured)?;
            if crate::numbuffers::finished_at_zero_limit(self.frame_limit, out).await? {
                return Ok(0);
            }
            let grab = GrabSettings {
                target: self.target.clone(),
                framerate_q16: self.framerate_q16,
                area,
                show_pointer: self.show_pointer,
                use_damage: self.use_damage,
            };
            let settings = DesktopCaptureSettings {
                thread_name: "g2g-ximagesrc",
                period: core::time::Duration::from_nanos(self.period_ns()),
                frame_limit: self.frame_limit,
                clock_sync: self.clock_sync.clone(),
            };
            run_desktop_capture(settings, out, move |worker| {
                let mut grabber = XGrabber::open(grab)?;
                worker.pace(&mut grabber)
            })
            .await
        })
    }
}

impl PadTemplates for XImageSrc {
    fn pad_templates() -> Vec<PadTemplate> {
        let caps =
            caps_for(FORMAT, Dim::Any, Dim::Any, Rate::Any).expect("caps_for builds every format");
        Vec::from([PadTemplate::source(g2g_core::CapsSet::one(caps))])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::vec;

    const AREA: CaptureArea = CaptureArea {
        window: 0,
        x: 0,
        y: 0,
        width: 4,
        height: 4,
    };

    #[test]
    fn every_declared_property_round_trips() {
        let mut source = XImageSrc::new();
        for spec in XIMAGESRC_PROPS {
            let value = match spec.kind {
                PropKind::Str => PropValue::Str(String::from("test")),
                PropKind::Uint => PropValue::Uint(3),
                PropKind::Bool => PropValue::Bool(false),
                PropKind::Int => PropValue::Int(5),
                kind => panic!("{} has a kind the test does not cover: {kind:?}", spec.name),
            };
            source
                .set_property(spec.name, value.clone())
                .unwrap_or_else(|error| panic!("{} takes {value:?}: {error:?}", spec.name));
            assert_eq!(source.get_property(spec.name), Some(value), "{}", spec.name);
        }
    }

    #[test]
    fn an_end_of_zero_reaches_the_far_edge_and_ends_are_inclusive() {
        assert_eq!(region_span(0, 0, 320), Some((0, 320)));
        assert_eq!(region_span(10, 109, 320), Some((10, 100)));
        assert_eq!(region_span(10, 0, 320), Some((10, 310)));
    }

    #[test]
    fn a_region_outside_the_window_is_refused() {
        assert_eq!(region_span(320, 0, 320), None);
        assert_eq!(region_span(0, 320, 320), None);
        assert_eq!(region_span(50, 40, 320), None);
    }

    #[test]
    fn the_cursor_is_blended_at_its_hotspot_and_clipped_to_the_picture() {
        let mut picture = vec![0u8; AREA.bytes()];
        let opaque_red = 0xFFFF_0000;
        let half_green = 0x8000_8000;
        let cursor = CursorImage {
            width: 2,
            hot_x: 1,
            hot_y: 0,
            pixels: vec![opaque_red, half_green],
        };
        let pixel = |picture: &[u8], x: usize, y: usize| {
            let offset = (y * usize::from(AREA.width) + x) * BYTES_PER_PIXEL;
            [picture[offset], picture[offset + 1], picture[offset + 2]]
        };
        draw_cursor(&mut picture, AREA, &cursor, 3, 3);
        assert_eq!(pixel(&picture, 2, 3), [0, 0, 0xFF]);
        assert_eq!(pixel(&picture, 3, 3), [0, 0x80, 0]);
        let last_column = usize::from(AREA.width) - 1;
        draw_cursor(&mut picture, AREA, &cursor, 4, 0);
        assert_eq!(pixel(&picture, last_column, 0), [0, 0, 0xFF]);
    }
}
