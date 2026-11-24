// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::borrow::Cow;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use g2g_core::log::Target;
use g2g_core::memory::OwnedDmaBuf;
use g2g_core::{G2gError, HardwareError};
use g2g_plugins::cursorshape::{CursorShape, CURSOR_BYTES_PER_PIXEL};
use g2g_plugins::windowinput::{WindowInput, WHEEL_PIXELS_PER_NOTCH, WHEEL_UNITS_PER_NOTCH};
use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::allocator::Fourcc;
use smithay::backend::input::{Axis, AxisSource, ButtonState, InputTime, KeyState};
use smithay::backend::renderer::damage::OutputDamageTracker;
use smithay::backend::renderer::element::surface::{
    render_elements_from_surface_tree, WaylandSurfaceRenderElement,
};
use smithay::backend::renderer::element::Kind;
use smithay::backend::renderer::pixman::PixmanRenderer;
use smithay::backend::renderer::utils::{on_commit_buffer_handler, with_renderer_surface_state};
use smithay::backend::renderer::{
    Bind, ImportAll, ImportDma, Offscreen, Renderer as SmithayRenderer,
};
use smithay::desktop::space::render_output;
use smithay::desktop::utils::send_frames_surface_tree;
use smithay::desktop::{PopupKind, PopupManager, Space, Window, WindowSurfaceType};
use smithay::input::dnd::{
    DnDGrab, DndFocus, DndGrabHandler, DndTarget, GrabType, OfferData, Source,
};
use smithay::input::keyboard::{
    FilterResult, KeyboardTarget, Keycode, KeysymHandle, ModifiersState, XkbConfig,
};
use smithay::input::pointer::{
    AxisFrame, ButtonEvent, CursorIcon, CursorImageStatus, CursorImageSurfaceData, Focus,
    GestureHoldBeginEvent, GestureHoldEndEvent, GesturePinchBeginEvent, GesturePinchEndEvent,
    GesturePinchUpdateEvent, GestureSwipeBeginEvent, GestureSwipeEndEvent, GestureSwipeUpdateEvent,
    MotionEvent, PointerTarget, RelativeMotionEvent,
};
use smithay::input::{Seat, SeatHandler, SeatState};
use smithay::output::{Mode, Output, PhysicalProperties, Subpixel};
use smithay::reexports::calloop::channel::{Channel, Event as ChannelEvent};
use smithay::reexports::calloop::generic::Generic;
use smithay::reexports::calloop::{
    EventLoop, Interest, LoopHandle, Mode as TriggerMode, PostAction,
};
use smithay::reexports::pixman::Image;
use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_toplevel;
use smithay::reexports::wayland_server::backend::ClientData;
use smithay::reexports::wayland_server::protocol::{
    wl_buffer, wl_seat, wl_shm, wl_surface::WlSurface,
};
use smithay::reexports::wayland_server::Resource;
use smithay::reexports::wayland_server::{Client, Display, DisplayHandle};
use smithay::utils::{
    IsAlive, Logical, Physical, Point, Rectangle, Serial, Transform, SERIAL_COUNTER,
};
use smithay::wayland::buffer::BufferHandler;
use smithay::wayland::compositor::{
    get_parent, is_sync_subsurface, with_states, CompositorClientState, CompositorHandler,
    CompositorState, SurfaceAttributes,
};
use smithay::wayland::dmabuf::{DmabufGlobal, DmabufHandler, DmabufState, ImportNotifier};
use smithay::wayland::output::{OutputHandler, OutputManagerState};
use smithay::wayland::pointer_constraints::PointerConstraintsHandler;
use smithay::wayland::seat::WaylandFocus;
use smithay::wayland::selection::data_device::{
    set_data_device_focus, DataDeviceHandler, DataDeviceState, WaylandDndGrabHandler, WlOfferData,
};
use smithay::wayland::selection::SelectionHandler;
use smithay::wayland::shell::xdg::{
    PopupSurface, PositionerState, ToplevelSurface, XdgShellHandler, XdgShellState,
    XdgToplevelSurfaceData,
};
use smithay::wayland::shm::{with_buffer_contents, BufferAccessError, ShmHandler, ShmState};
use smithay::wayland::socket::ListeningSocketSource;
use smithay::wayland::xwayland_shell::{XWaylandShellHandler, XWaylandShellState};
use smithay::xwayland::xwm::{Reorder, ResizeEdge, XwmId, XwmOfferData};
use smithay::xwayland::{
    X11Surface, X11Wm, XWayland, XWaylandClientData, XWaylandEvent, XwmHandler,
};
use tokio::sync::{oneshot, watch};
use xcursor::CursorTheme;

use crate::gpu::{GpuFormat, GpuOutput, GpuRenderer};
use crate::waylanddisplaysrc::DisplayNames;

const LOG_CATEGORY: &str = "WaylandDisplaySrc";
// xkb keycodes are evdev codes offset by 8
const XKB_KEYCODE_OFFSET: u32 = 8;
const KEY_REPEAT_DELAY_MS: i32 = 200;
const KEY_REPEAT_RATE_HZ: i32 = 25;
// libXcursor's default theme and nominal size when XCURSOR_THEME and XCURSOR_SIZE are unset
const DEFAULT_CURSOR_THEME: &str = "default";
const DEFAULT_CURSOR_SIZE: u32 = 24;
pub(crate) const BACKGROUND: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
pub(crate) const BYTES_PER_PIXEL: usize = 4;
pub(crate) const OUTPUT_FOURCC: Fourcc = Fourcc::Abgr8888;

#[derive(Debug, Clone, Copy)]
pub(crate) struct DisplaySize {
    pub(crate) width: u32,
    pub(crate) height: u32,
}

#[derive(Debug)]
pub(crate) struct Pacing {
    pub(crate) frame_period: Duration,
    pub(crate) repeat_after: Duration,
}

#[derive(Debug, Clone)]
pub(crate) enum Picture {
    System(Vec<u8>),
    DmaBuf(OwnedDmaBuf),
}

pub(crate) type FrameReply = oneshot::Sender<Result<Picture, G2gError>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SessionOutput {
    pub(crate) output: GpuOutput,
    pub(crate) format: GpuFormat,
}

#[derive(Debug)]
pub(crate) enum DisplayRequest {
    Attach {
        output: SessionOutput,
        reply: oneshot::Sender<Result<(), G2gError>>,
    },
    Frame(FrameReply),
    Detach,
}

#[derive(Debug, Clone)]
pub(crate) enum Rendering {
    Software,
    Gpu { render_node: PathBuf },
}

#[derive(Debug)]
pub(crate) struct DisplaySettings {
    pub(crate) size: DisplaySize,
    pub(crate) pacing: Pacing,
    pub(crate) rendering: Rendering,
    pub(crate) xwayland: bool,
}

pub(crate) type NamesReply = oneshot::Sender<Result<DisplayNames, G2gError>>;

pub(crate) struct DisplayChannels {
    pub(crate) requests: Channel<DisplayRequest>,
    pub(crate) input: Channel<WindowInput>,
    pub(crate) names: NamesReply,
    pub(crate) cursor_shapes: watch::Sender<Option<CursorShape>>,
}

pub(crate) fn hardware_error(what: &str, error: impl core::fmt::Debug) -> G2gError {
    g2g_core::g2g_error!(Target::category(LOG_CATEGORY), "{what}: {error:?}");
    G2gError::Hardware(HardwareError::Other)
}

pub(crate) fn run_display(
    settings: DisplaySettings,
    channels: DisplayChannels,
) -> Result<(), G2gError> {
    let DisplayChannels {
        requests,
        input,
        names,
        cursor_shapes,
    } = channels;
    let xwayland = settings.xwayland;
    let (mut event_loop, mut state) = match start_display(settings, requests, input, cursor_shapes)
    {
        Ok(started) => started,
        Err(error) => {
            let _ = names.send(Err(error.clone()));
            return Err(error);
        }
    };
    state.names_reply = Some(names);
    if xwayland {
        if let Err(error) = spawn_xwayland(&mut state) {
            state.fail_startup(error.clone());
            return Err(error);
        }
    } else {
        state.publish_names(None);
    }
    while !state.stopped {
        let timeout = state.time_until_next_frame(Instant::now());
        event_loop
            .dispatch(timeout, &mut state)
            .map_err(|error| hardware_error("wayland event loop", error))?;
        state.serve_frame_request(Instant::now());
        state
            .display_handle
            .flush_clients()
            .map_err(|error| hardware_error("flushing wayland clients", error))?;
    }
    Ok(())
}

fn spawn_xwayland(state: &mut DisplayState) -> Result<(), G2gError> {
    let (xwayland, client) = XWayland::spawn(
        &state.display_handle,
        None,
        std::iter::empty::<(String, String)>(),
        std::iter::empty::<String>(),
        true,
        Stdio::null(),
        Stdio::null(),
        |_| (),
    )
    .map_err(|error| hardware_error("starting Xwayland", error))?;
    state
        .loop_handle
        .insert_source(xwayland, move |event, _, state| match event {
            XWaylandEvent::Ready {
                x11_socket,
                display_number,
            } => state.start_window_manager(x11_socket, display_number, client.clone()),
            XWaylandEvent::Error => {
                state.fail_startup(hardware_error("Xwayland exited during startup", ()));
            }
        })
        .map_err(|error| hardware_error("watching Xwayland", error))?;
    Ok(())
}

fn start_display(
    settings: DisplaySettings,
    requests: Channel<DisplayRequest>,
    input: Channel<WindowInput>,
    cursor_shapes: watch::Sender<Option<CursorShape>>,
) -> Result<(EventLoop<'static, DisplayState>, DisplayState), G2gError> {
    let event_loop: EventLoop<DisplayState> =
        EventLoop::try_new().map_err(|error| hardware_error("creating the event loop", error))?;
    let display: Display<DisplayState> =
        Display::new().map_err(|error| hardware_error("creating the wayland display", error))?;
    let display_handle = display.handle();

    let listening_socket = ListeningSocketSource::new_auto()
        .map_err(|error| hardware_error("opening the wayland socket", error))?;
    let socket_name = listening_socket
        .socket_name()
        .to_string_lossy()
        .into_owned();
    let loop_handle = event_loop.handle();
    loop_handle
        .insert_source(listening_socket, |stream, _, state| {
            if let Err(error) = state
                .display_handle
                .insert_client(stream, Arc::new(ClientState::default()))
            {
                g2g_core::g2g_warn!(
                    Target::category(LOG_CATEGORY),
                    "refused a wayland client: {error}"
                );
            }
        })
        .map_err(|error| hardware_error("listening on the wayland socket", error))?;
    loop_handle
        .insert_source(
            Generic::new(display, Interest::READ, TriggerMode::Level),
            |_, display, state| {
                // SAFETY: the display stays inside this event source and is never dropped while dispatching.
                unsafe { display.get_mut() }
                    .dispatch_clients(state)
                    .map(|_| PostAction::Continue)
            },
        )
        .map_err(|error| hardware_error("dispatching wayland clients", error))?;
    loop_handle
        .insert_source(requests, |event, _, state| match event {
            ChannelEvent::Msg(DisplayRequest::Attach { output, reply }) => {
                // the session drops its receiver when its run is cancelled
                let _ = reply.send(state.attach(output));
            }
            ChannelEvent::Msg(DisplayRequest::Frame(reply)) => state.pending_reply = Some(reply),
            ChannelEvent::Msg(DisplayRequest::Detach) => state.detach(),
            ChannelEvent::Closed => state.stopped = true,
        })
        .map_err(|error| hardware_error("receiving display requests", error))?;
    loop_handle
        .insert_source(input, |event, _, state| {
            if let ChannelEvent::Msg(input) = event {
                state.handle_input(input);
            }
        })
        .map_err(|error| hardware_error("receiving input", error))?;

    let mut state = DisplayState::new(
        display_handle,
        loop_handle,
        socket_name,
        settings,
        cursor_shapes,
    )?;
    state.publish_cursor();
    Ok((event_loop, state))
}

#[derive(Debug, Default)]
struct ClientState {
    compositor_state: CompositorClientState,
}

impl ClientData for ClientState {}

#[derive(Debug, Clone, PartialEq)]
enum FocusedSurface {
    Wayland(WlSurface),
    X11(Box<X11Surface>),
}

impl FocusedSurface {
    fn of_window(window: &Window) -> Option<Self> {
        if let Some(toplevel) = window.toplevel() {
            return Some(Self::Wayland(toplevel.wl_surface().clone()));
        }
        window
            .x11_surface()
            .map(|surface| Self::X11(Box::new(surface.clone())))
    }
}

impl IsAlive for FocusedSurface {
    fn alive(&self) -> bool {
        match self {
            Self::Wayland(surface) => surface.alive(),
            Self::X11(surface) => surface.alive(),
        }
    }
}

impl WaylandFocus for FocusedSurface {
    fn wl_surface(&self) -> Option<Cow<'_, WlSurface>> {
        match self {
            Self::Wayland(surface) => Some(Cow::Borrowed(surface)),
            Self::X11(surface) => WaylandFocus::wl_surface(&**surface),
        }
    }
}

impl KeyboardTarget<DisplayState> for FocusedSurface {
    fn enter(
        &self,
        seat: &Seat<DisplayState>,
        state: &mut DisplayState,
        keys: Vec<KeysymHandle<'_>>,
        serial: Serial,
    ) {
        match self {
            Self::Wayland(surface) => KeyboardTarget::enter(surface, seat, state, keys, serial),
            Self::X11(surface) => KeyboardTarget::enter(&**surface, seat, state, keys, serial),
        }
    }

    fn leave(&self, seat: &Seat<DisplayState>, state: &mut DisplayState, serial: Serial) {
        match self {
            Self::Wayland(surface) => KeyboardTarget::leave(surface, seat, state, serial),
            Self::X11(surface) => KeyboardTarget::leave(&**surface, seat, state, serial),
        }
    }

    fn key(
        &self,
        seat: &Seat<DisplayState>,
        state: &mut DisplayState,
        key: KeysymHandle<'_>,
        key_state: KeyState,
        serial: Serial,
        time: InputTime,
    ) {
        match self {
            Self::Wayland(surface) => {
                KeyboardTarget::key(surface, seat, state, key, key_state, serial, time);
            }
            Self::X11(surface) => {
                KeyboardTarget::key(&**surface, seat, state, key, key_state, serial, time);
            }
        }
    }

    fn modifiers(
        &self,
        seat: &Seat<DisplayState>,
        state: &mut DisplayState,
        modifiers: ModifiersState,
        serial: Serial,
    ) {
        match self {
            Self::Wayland(surface) => {
                KeyboardTarget::modifiers(surface, seat, state, modifiers, serial);
            }
            Self::X11(surface) => {
                KeyboardTarget::modifiers(&**surface, seat, state, modifiers, serial);
            }
        }
    }
}

impl FocusedSurface {
    fn pointer_target(&self) -> &dyn PointerTarget<DisplayState> {
        match self {
            Self::Wayland(surface) => surface,
            Self::X11(surface) => &**surface,
        }
    }
}

impl PointerTarget<DisplayState> for FocusedSurface {
    fn enter(&self, seat: &Seat<DisplayState>, state: &mut DisplayState, event: &MotionEvent) {
        self.pointer_target().enter(seat, state, event);
    }

    fn motion(&self, seat: &Seat<DisplayState>, state: &mut DisplayState, event: &MotionEvent) {
        self.pointer_target().motion(seat, state, event);
    }

    fn relative_motion(
        &self,
        seat: &Seat<DisplayState>,
        state: &mut DisplayState,
        event: &RelativeMotionEvent,
    ) {
        self.pointer_target().relative_motion(seat, state, event);
    }

    fn button(&self, seat: &Seat<DisplayState>, state: &mut DisplayState, event: &ButtonEvent) {
        self.pointer_target().button(seat, state, event);
    }

    fn axis(&self, seat: &Seat<DisplayState>, state: &mut DisplayState, frame: AxisFrame) {
        self.pointer_target().axis(seat, state, frame);
    }

    fn frame(&self, seat: &Seat<DisplayState>, state: &mut DisplayState) {
        self.pointer_target().frame(seat, state);
    }

    fn gesture_swipe_begin(
        &self,
        seat: &Seat<DisplayState>,
        state: &mut DisplayState,
        event: &GestureSwipeBeginEvent,
    ) {
        self.pointer_target()
            .gesture_swipe_begin(seat, state, event);
    }

    fn gesture_swipe_update(
        &self,
        seat: &Seat<DisplayState>,
        state: &mut DisplayState,
        event: &GestureSwipeUpdateEvent,
    ) {
        self.pointer_target()
            .gesture_swipe_update(seat, state, event);
    }

    fn gesture_swipe_end(
        &self,
        seat: &Seat<DisplayState>,
        state: &mut DisplayState,
        event: &GestureSwipeEndEvent,
    ) {
        self.pointer_target().gesture_swipe_end(seat, state, event);
    }

    fn gesture_pinch_begin(
        &self,
        seat: &Seat<DisplayState>,
        state: &mut DisplayState,
        event: &GesturePinchBeginEvent,
    ) {
        self.pointer_target()
            .gesture_pinch_begin(seat, state, event);
    }

    fn gesture_pinch_update(
        &self,
        seat: &Seat<DisplayState>,
        state: &mut DisplayState,
        event: &GesturePinchUpdateEvent,
    ) {
        self.pointer_target()
            .gesture_pinch_update(seat, state, event);
    }

    fn gesture_pinch_end(
        &self,
        seat: &Seat<DisplayState>,
        state: &mut DisplayState,
        event: &GesturePinchEndEvent,
    ) {
        self.pointer_target().gesture_pinch_end(seat, state, event);
    }

    fn gesture_hold_begin(
        &self,
        seat: &Seat<DisplayState>,
        state: &mut DisplayState,
        event: &GestureHoldBeginEvent,
    ) {
        self.pointer_target().gesture_hold_begin(seat, state, event);
    }

    fn gesture_hold_end(
        &self,
        seat: &Seat<DisplayState>,
        state: &mut DisplayState,
        event: &GestureHoldEndEvent,
    ) {
        self.pointer_target().gesture_hold_end(seat, state, event);
    }

    fn leave(
        &self,
        seat: &Seat<DisplayState>,
        state: &mut DisplayState,
        serial: Serial,
        time: InputTime,
    ) {
        self.pointer_target().leave(seat, state, serial, time);
    }
}

#[derive(Debug)]
enum DropOffer<S: Source> {
    Wayland(WlOfferData<S>),
    X11(XwmOfferData<S>),
}

impl<S: Source> OfferData for DropOffer<S> {
    fn disable(&self) {
        match self {
            Self::Wayland(offer) => offer.disable(),
            Self::X11(offer) => offer.disable(),
        }
    }

    fn drop(&self) {
        match self {
            Self::Wayland(offer) => offer.drop(),
            Self::X11(offer) => offer.drop(),
        }
    }

    fn validated(&self) -> bool {
        match self {
            Self::Wayland(offer) => offer.validated(),
            Self::X11(offer) => offer.validated(),
        }
    }
}

fn wayland_offer<S: Source>(
    offer: Option<&mut DropOffer<S>>,
) -> Option<Option<&mut WlOfferData<S>>> {
    match offer {
        None => Some(None),
        Some(DropOffer::Wayland(offer)) => Some(Some(offer)),
        Some(DropOffer::X11(_)) => None,
    }
}

fn x11_offer<S: Source>(offer: Option<&mut DropOffer<S>>) -> Option<Option<&mut XwmOfferData<S>>> {
    match offer {
        None => Some(None),
        Some(DropOffer::X11(offer)) => Some(Some(offer)),
        Some(DropOffer::Wayland(_)) => None,
    }
}

impl DndFocus<DisplayState> for FocusedSurface {
    type OfferData<S>
        = DropOffer<S>
    where
        S: Source;

    fn enter<S: Source>(
        &self,
        state: &mut DisplayState,
        display_handle: &DisplayHandle,
        source: Arc<S>,
        seat: &Seat<DisplayState>,
        location: Point<f64, Logical>,
        serial: &Serial,
    ) -> Option<DropOffer<S>> {
        match self {
            Self::Wayland(surface) => DndFocus::enter(
                surface,
                state,
                display_handle,
                source,
                seat,
                location,
                serial,
            )
            .map(DropOffer::Wayland),
            Self::X11(surface) => DndFocus::enter(
                &**surface,
                state,
                display_handle,
                source,
                seat,
                location,
                serial,
            )
            .map(DropOffer::X11),
        }
    }

    fn motion<S: Source>(
        &self,
        state: &mut DisplayState,
        offer: Option<&mut DropOffer<S>>,
        seat: &Seat<DisplayState>,
        location: Point<f64, Logical>,
        time: InputTime,
    ) {
        match self {
            Self::Wayland(surface) => {
                let Some(offer) = wayland_offer(offer) else {
                    return;
                };
                DndFocus::motion(surface, state, offer, seat, location, time);
            }
            Self::X11(surface) => {
                let Some(offer) = x11_offer(offer) else {
                    return;
                };
                DndFocus::motion(&**surface, state, offer, seat, location, time);
            }
        }
    }

    fn leave<S: Source>(
        &self,
        state: &mut DisplayState,
        offer: Option<&mut DropOffer<S>>,
        seat: &Seat<DisplayState>,
    ) {
        match self {
            Self::Wayland(surface) => {
                let Some(offer) = wayland_offer(offer) else {
                    return;
                };
                DndFocus::leave(surface, state, offer, seat);
            }
            Self::X11(surface) => {
                let Some(offer) = x11_offer(offer) else {
                    return;
                };
                DndFocus::leave(&**surface, state, offer, seat);
            }
        }
    }

    fn drop<S: Source>(
        &self,
        state: &mut DisplayState,
        offer: Option<&mut DropOffer<S>>,
        seat: &Seat<DisplayState>,
    ) {
        match self {
            Self::Wayland(surface) => {
                let Some(offer) = wayland_offer(offer) else {
                    return;
                };
                DndFocus::drop(surface, state, offer, seat);
            }
            Self::X11(surface) => {
                let Some(offer) = x11_offer(offer) else {
                    return;
                };
                DndFocus::drop(&**surface, state, offer, seat);
            }
        }
    }
}

struct SoftwareRenderer {
    pixman: PixmanRenderer,
    image: Image<'static, 'static>,
    rendered_once: bool,
}

enum Renderer {
    Software(SoftwareRenderer),
    Gpu(Box<GpuRenderer>),
}

struct DisplayState {
    display_handle: DisplayHandle,
    loop_handle: LoopHandle<'static, DisplayState>,
    socket_name: String,
    names_reply: Option<NamesReply>,
    size: DisplaySize,
    pacing: Pacing,
    started: Instant,
    compositor_state: CompositorState,
    xdg_shell_state: XdgShellState,
    shm_state: ShmState,
    dmabuf_state: DmabufState,
    _dmabuf_global: Option<DmabufGlobal>,
    data_device_state: DataDeviceState,
    _output_manager_state: OutputManagerState,
    xwayland_shell_state: XWaylandShellState,
    window_manager: Option<X11Wm>,
    seat_state: SeatState<DisplayState>,
    seat: Seat<DisplayState>,
    space: Space<Window>,
    popups: PopupManager,
    output: Output,
    renderer: Renderer,
    damage_tracker: OutputDamageTracker,
    pending_reply: Option<FrameReply>,
    damaged: bool,
    last_frame: Option<(Instant, Picture)>,
    held_buttons: Vec<u32>,
    drag_icon: Option<DragIcon>,
    cursor_status: CursorImageStatus,
    cursor_theme: CursorTheme,
    cursor_size: u32,
    cursor_shapes: watch::Sender<Option<CursorShape>>,
    stopped: bool,
}

#[derive(Debug)]
enum CursorError {
    NotInTheme(CursorIcon),
    Read(std::io::Error),
    NotXcursor,
    NoRendererState,
    NotShm(BufferAccessError),
    Format(wl_shm::Format),
    Size,
}

impl core::fmt::Display for CursorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotInTheme(icon) => write!(f, "the cursor theme has no {}", icon.name()),
            Self::Read(error) => write!(f, "reading the cursor file: {error}"),
            Self::NotXcursor => write!(f, "the cursor file is not an xcursor file"),
            Self::NoRendererState => write!(f, "the cursor surface has no buffer state"),
            Self::NotShm(error) => write!(f, "the cursor buffer is not wl_shm: {error:?}"),
            Self::Format(format) => write!(f, "the cursor buffer format {format:?} is not argb"),
            Self::Size => write!(f, "the cursor buffer size does not fit its pool"),
        }
    }
}

// wl_shm and xcursor pixels are little-endian argb with premultiplied alpha
fn straight_rgba(bgra_premultiplied: [u8; 4]) -> [u8; 4] {
    let [blue, green, red, alpha] = bgra_premultiplied;
    if alpha == 0 {
        return [0; 4];
    }
    let straight = |channel: u8| (u32::from(channel) * 255 / u32::from(alpha)).min(255) as u8;
    [straight(red), straight(green), straight(blue), alpha]
}

fn named_cursor(
    theme: &CursorTheme,
    size: u32,
    icon: CursorIcon,
) -> Result<CursorShape, CursorError> {
    let path = std::iter::once(icon.name())
        .chain(icon.alt_names().iter().copied())
        .find_map(|name| theme.load_icon(name))
        .ok_or(CursorError::NotInTheme(icon))?;
    let file = std::fs::read(path).map_err(CursorError::Read)?;
    let images = xcursor::parser::parse_xcursor(&file).ok_or(CursorError::NotXcursor)?;
    let image = images
        .iter()
        .min_by_key(|image| image.size.abs_diff(size))
        .ok_or(CursorError::NotXcursor)?;
    let hotspot = |pixel: u32| i32::try_from(pixel).map_err(|_| CursorError::Size);
    Ok(CursorShape {
        width: image.width,
        height: image.height,
        hotspot_x: hotspot(image.xhot)?,
        hotspot_y: hotspot(image.yhot)?,
        rgba: image
            .pixels_rgba
            .as_chunks::<CURSOR_BYTES_PER_PIXEL>()
            .0
            .iter()
            .flat_map(|pixel| straight_rgba(*pixel))
            .collect(),
    })
}

fn surface_cursor(surface: &WlSurface) -> Result<Option<CursorShape>, CursorError> {
    let buffer = with_renderer_surface_state(surface, |state| state.buffer().cloned())
        .ok_or(CursorError::NoRendererState)?;
    // wayland hides the pointer over a cursor surface with no buffer
    let Some(buffer) = buffer else {
        return Ok(None);
    };
    let hotspot = with_states(surface, |states| {
        states
            .data_map
            .get::<CursorImageSurfaceData>()
            .expect("smithay gives every cursor surface its attributes")
            .lock()
            .expect("no thread panics holding the cursor attributes")
            .hotspot
    });
    let pixels = with_buffer_contents(&buffer, |pool, pool_len, data| {
        let opaque = match data.format {
            wl_shm::Format::Argb8888 => false,
            wl_shm::Format::Xrgb8888 => true,
            format => return Err(CursorError::Format(format)),
        };
        let size = |value: i32| usize::try_from(value).map_err(|_| CursorError::Size);
        let (offset, width, height, stride) = (
            size(data.offset)?,
            size(data.width)?,
            size(data.height)?,
            size(data.stride)?,
        );
        let row_bytes = width * CURSOR_BYTES_PER_PIXEL;
        let end = height
            .checked_mul(stride)
            .and_then(|bytes| bytes.checked_add(offset))
            .filter(|end| *end <= pool_len && row_bytes <= stride)
            .ok_or(CursorError::Size)?;
        // SAFETY: smithay maps pool_len bytes at pool for the duration of this callback, and end is within them.
        let bytes = unsafe { std::slice::from_raw_parts(pool, end) };
        let rgba = bytes[offset..]
            .chunks(stride)
            .take(height)
            .flat_map(|row| row[..row_bytes].as_chunks::<CURSOR_BYTES_PER_PIXEL>().0)
            .flat_map(|&[blue, green, red, alpha]| {
                let alpha = if opaque { u8::MAX } else { alpha };
                straight_rgba([blue, green, red, alpha])
            })
            .collect::<Vec<u8>>();
        Ok((data.width, data.height, rgba))
    })
    .map_err(CursorError::NotShm)??;
    let (width, height, rgba) = pixels;
    let dimension = |value: i32| u32::try_from(value).map_err(|_| CursorError::Size);
    Ok(Some(CursorShape {
        width: dimension(width)?,
        height: dimension(height)?,
        hotspot_x: hotspot.x,
        hotspot_y: hotspot.y,
        rgba,
    }))
}

#[derive(Debug)]
struct DragIcon {
    surface: WlSurface,
    // wayland clients move the hotspot by attaching the icon buffer with an offset
    offset: Point<i32, Logical>,
}

pub(crate) fn drag_icon_elements<R>(
    renderer: &mut R,
    drag_icon: Option<(&WlSurface, Point<i32, Physical>)>,
) -> Vec<WaylandSurfaceRenderElement<R>>
where
    R: SmithayRenderer + ImportAll,
    R::TextureId: Clone + 'static,
{
    let Some((surface, location)) = drag_icon else {
        return Vec::new();
    };
    render_elements_from_surface_tree(renderer, surface, location, 1.0, 1.0, Kind::Unspecified)
}

fn render_software(
    software: &mut SoftwareRenderer,
    output: &Output,
    space: &Space<Window>,
    drag_icon: Option<(&WlSurface, Point<i32, Physical>)>,
    damage_tracker: &mut OutputDamageTracker,
) -> Result<(), G2gError> {
    let age = usize::from(software.rendered_once);
    let drag_icon = drag_icon_elements(&mut software.pixman, drag_icon);
    let mut framebuffer = software
        .pixman
        .bind(&mut software.image)
        .map_err(|error| hardware_error("binding the output picture", error))?;
    let result = render_output::<_, WaylandSurfaceRenderElement<PixmanRenderer>, _, _>(
        output,
        &mut software.pixman,
        &mut framebuffer,
        1.0,
        age,
        [space],
        &drag_icon,
        damage_tracker,
        BACKGROUND,
    )
    .map_err(|error| hardware_error("rendering the output", error))?;
    result
        .sync
        .wait()
        .map_err(|error| hardware_error("waiting for the render", error))?;
    software.rendered_once = true;
    Ok(())
}

fn copy_picture(image: &Image<'static, 'static>, size: DisplaySize) -> Vec<u8> {
    let row_bytes = size.width as usize * BYTES_PER_PIXEL;
    let stride = image.stride();
    let height = size.height as usize;
    // SAFETY: pixman owns stride * height bytes at data() for as long as the image lives, and nothing writes them during this borrow.
    let bits = unsafe {
        std::slice::from_raw_parts(image.data().cast::<u8>().cast_const(), stride * height)
    };
    let mut picture = Vec::with_capacity(row_bytes * height);
    for row in bits.chunks_exact(stride) {
        picture.extend_from_slice(&row[..row_bytes]);
    }
    picture
}

impl DisplayState {
    fn new(
        display_handle: DisplayHandle,
        loop_handle: LoopHandle<'static, DisplayState>,
        socket_name: String,
        settings: DisplaySettings,
        cursor_shapes: watch::Sender<Option<CursorShape>>,
    ) -> Result<Self, G2gError> {
        let DisplaySettings {
            size,
            pacing,
            rendering,
            ..
        } = settings;
        let compositor_state = CompositorState::new::<Self>(&display_handle);
        let xdg_shell_state = XdgShellState::new::<Self>(&display_handle);
        let shm_state = ShmState::new::<Self>(&display_handle, Vec::new());
        let data_device_state = DataDeviceState::new::<Self>(&display_handle);
        let output_manager_state = OutputManagerState::new_with_xdg_output::<Self>(&display_handle);
        let xwayland_shell_state = XWaylandShellState::new::<Self>(&display_handle);
        let mut seat_state = SeatState::new();
        let mut seat = seat_state.new_wl_seat(&display_handle, "seat0");
        seat.add_keyboard(
            XkbConfig::default(),
            KEY_REPEAT_DELAY_MS,
            KEY_REPEAT_RATE_HZ,
        )
        .map_err(|error| hardware_error("loading the keymap", error))?;
        seat.add_pointer();

        let pixel_size = (
            i32::try_from(size.width).map_err(|_| G2gError::CapsMismatch)?,
            i32::try_from(size.height).map_err(|_| G2gError::CapsMismatch)?,
        );
        let output = Output::new(
            "g2g-0".into(),
            PhysicalProperties {
                size: (0, 0).into(),
                subpixel: Subpixel::Unknown,
                make: "glass2glass".into(),
                model: "waylanddisplaysrc".into(),
                serial_number: String::new(),
            },
        );
        let refresh_millihertz =
            Duration::from_secs(1).as_nanos() * 1000 / pacing.frame_period.as_nanos().max(1);
        let mode = Mode {
            size: pixel_size.into(),
            refresh: i32::try_from(refresh_millihertz).map_err(|_| G2gError::CapsMismatch)?,
        };
        output.create_global::<Self>(&display_handle);
        output.change_current_state(
            Some(mode),
            Some(Transform::Normal),
            None,
            Some((0, 0).into()),
        );
        output.set_preferred(mode);
        let mut space = Space::default();
        space.map_output(&output, (0, 0));

        let renderer = match rendering {
            Rendering::Software => {
                let mut pixman = PixmanRenderer::new()
                    .map_err(|error| hardware_error("creating the renderer", error))?;
                let image = pixman
                    .create_buffer(OUTPUT_FOURCC, pixel_size.into())
                    .map_err(|error| hardware_error("allocating the output picture", error))?;
                Renderer::Software(SoftwareRenderer {
                    pixman,
                    image,
                    rendered_once: false,
                })
            }
            Rendering::Gpu { render_node } => {
                Renderer::Gpu(Box::new(GpuRenderer::open(&render_node, size)?))
            }
        };
        let mut dmabuf_state = DmabufState::new();
        let dmabuf_global = match &renderer {
            Renderer::Software(_) => None,
            Renderer::Gpu(gpu) => Some(dmabuf_state.create_global_with_default_feedback::<Self>(
                &display_handle,
                &gpu.dmabuf_feedback()?,
            )),
        };
        let damage_tracker = OutputDamageTracker::from_output(&output);
        let cursor_theme_name =
            std::env::var("XCURSOR_THEME").unwrap_or_else(|_| String::from(DEFAULT_CURSOR_THEME));
        let cursor_size = std::env::var("XCURSOR_SIZE")
            .ok()
            .and_then(|size| size.parse().ok())
            .unwrap_or(DEFAULT_CURSOR_SIZE);

        Ok(Self {
            display_handle,
            loop_handle,
            socket_name,
            names_reply: None,
            size,
            pacing,
            started: Instant::now(),
            compositor_state,
            xdg_shell_state,
            shm_state,
            dmabuf_state,
            _dmabuf_global: dmabuf_global,
            data_device_state,
            _output_manager_state: output_manager_state,
            xwayland_shell_state,
            window_manager: None,
            seat_state,
            seat,
            space,
            popups: PopupManager::default(),
            output,
            renderer,
            damage_tracker,
            pending_reply: None,
            damaged: true,
            last_frame: None,
            held_buttons: Vec::new(),
            drag_icon: None,
            cursor_status: CursorImageStatus::default_named(),
            cursor_theme: CursorTheme::load(&cursor_theme_name),
            cursor_size,
            cursor_shapes,
            stopped: false,
        })
    }

    fn attach(&mut self, output: SessionOutput) -> Result<(), G2gError> {
        match &mut self.renderer {
            Renderer::Software(_) => {
                let software = SessionOutput {
                    output: GpuOutput::System,
                    format: GpuFormat::Rgba,
                };
                if output != software {
                    return Err(G2gError::CapsMismatch);
                }
            }
            Renderer::Gpu(gpu) => gpu.select_output(output)?,
        }
        self.pending_reply = None;
        self.last_frame = None;
        self.damaged = true;
        Ok(())
    }

    fn detach(&mut self) {
        self.pending_reply = None;
        self.release_held_input(InputTime::now());
    }

    fn publish_cursor(&mut self) {
        let shape = match &self.cursor_status {
            CursorImageStatus::Hidden => Ok(None),
            CursorImageStatus::Named(icon) => {
                named_cursor(&self.cursor_theme, self.cursor_size, *icon).map(Some)
            }
            CursorImageStatus::Surface(surface) => surface_cursor(surface),
        };
        let shape = match shape {
            Ok(shape) => shape,
            Err(error) => {
                g2g_core::g2g_warn!(
                    Target::category(LOG_CATEGORY),
                    "kept the previous cursor: {error}"
                );
                return;
            }
        };
        self.cursor_shapes.send_if_modified(|current| {
            if *current == shape {
                return false;
            }
            *current = shape;
            true
        });
    }

    fn publish_names(&mut self, x11_display: Option<String>) {
        if let Some(reply) = self.names_reply.take() {
            // the element drops its receiver when the run ends
            let _ = reply.send(Ok(DisplayNames {
                wayland_socket: self.socket_name.clone(),
                x11_display,
            }));
        }
    }

    fn fail_startup(&mut self, error: G2gError) {
        if let Some(reply) = self.names_reply.take() {
            let _ = reply.send(Err(error));
        }
        self.stopped = true;
    }

    fn start_window_manager(
        &mut self,
        x11_socket: UnixStream,
        display_number: u32,
        client: Client,
    ) {
        let started = X11Wm::start_wm(
            self.loop_handle.clone(),
            &self.display_handle,
            x11_socket,
            client,
        );
        match started {
            Ok(window_manager) => {
                self.window_manager = Some(window_manager);
                self.publish_names(Some(format!(":{display_number}")));
            }
            Err(error) => {
                self.fail_startup(hardware_error("starting the X11 window manager", error))
            }
        }
    }

    fn output_rectangle(&self) -> Rectangle<i32, Logical> {
        self.space
            .output_geometry(&self.output)
            .expect("the output is mapped at construction")
    }

    fn x11_window(&self, surface: &X11Surface) -> Option<Window> {
        self.space
            .elements()
            .find(|window| window.x11_surface() == Some(surface))
            .cloned()
    }

    fn next_frame_due(&self) -> Option<Instant> {
        self.pending_reply.as_ref()?;
        let Some((rendered_at, _)) = &self.last_frame else {
            return Some(self.started);
        };
        let wait = if self.damaged {
            self.pacing.frame_period
        } else {
            self.pacing.repeat_after
        };
        Some(*rendered_at + wait)
    }

    fn time_until_next_frame(&self, now: Instant) -> Option<Duration> {
        self.next_frame_due()
            .map(|due| due.saturating_duration_since(now))
    }

    fn serve_frame_request(&mut self, now: Instant) {
        let Some(due) = self.next_frame_due() else {
            return;
        };
        if now < due {
            return;
        }
        let picture = match self.last_frame.take() {
            Some((_, picture)) if !self.damaged => Ok(picture),
            _ => self.render(),
        };
        if let Ok(picture) = &picture {
            self.last_frame = Some((now, picture.clone()));
        }
        if let Some(reply) = self.pending_reply.take() {
            // the element drops its receiver when the run ends
            let _ = reply.send(picture);
        }
    }

    fn render(&mut self) -> Result<Picture, G2gError> {
        let pointer = self.seat.get_pointer().expect("the seat has a pointer");
        let drag_icon = self.drag_icon.as_ref().map(|icon| {
            let location = pointer.current_location() + icon.offset.to_f64();
            (&icon.surface, location.to_physical(1.0).to_i32_round())
        });
        let picture = match &mut self.renderer {
            Renderer::Software(software) => {
                render_software(
                    software,
                    &self.output,
                    &self.space,
                    drag_icon,
                    &mut self.damage_tracker,
                )?;
                Picture::System(copy_picture(&software.image, self.size))
            }
            Renderer::Gpu(gpu) => gpu.render(
                &self.output,
                &self.space,
                drag_icon,
                &mut self.damage_tracker,
            )?,
        };
        self.damaged = false;

        let elapsed = self.started.elapsed();
        for window in self.space.elements() {
            window.send_frame(&self.output, elapsed, Some(Duration::ZERO), |_, _| {
                Some(self.output.clone())
            });
        }
        let drag_icon = self.drag_icon.as_ref().map(|icon| &icon.surface);
        let cursor = match &self.cursor_status {
            CursorImageStatus::Surface(surface) => Some(surface),
            _ => None,
        };
        for surface in drag_icon.into_iter().chain(cursor) {
            send_frames_surface_tree(
                surface,
                &self.output,
                elapsed,
                Some(Duration::ZERO),
                |_, _| Some(self.output.clone()),
            );
        }
        self.space.refresh();
        self.popups.cleanup();
        Ok(picture)
    }

    fn surface_under(
        &self,
        location: Point<f64, Logical>,
    ) -> Option<(FocusedSurface, Point<f64, Logical>)> {
        let (window, window_location) = self.space.element_under(location)?;
        let (surface, surface_location) =
            window.surface_under(location - window_location.to_f64(), WindowSurfaceType::ALL)?;
        // smithay drops onto X11 windows through XDND only when the pointer focus is the X11Surface
        let focus = match window.x11_surface() {
            Some(x11_surface) => FocusedSurface::X11(Box::new(x11_surface.clone())),
            None => FocusedSurface::Wayland(surface),
        };
        Some((focus, (surface_location + window_location).to_f64()))
    }

    fn raise_override_redirect_windows(&mut self) {
        // X11 menus and tooltips are override-redirect and must stay above every managed window
        let menus: Vec<Window> = self
            .space
            .elements()
            .filter(|window| {
                window
                    .x11_surface()
                    .is_some_and(X11Surface::is_override_redirect)
            })
            .cloned()
            .collect();
        for menu in menus {
            self.space.raise_element(&menu, false);
        }
    }

    fn focus_window_under_pointer(&mut self, serial: Serial) {
        let pointer = self.seat.get_pointer().expect("the seat has a pointer");
        let Some(window) = self
            .space
            .element_under(pointer.current_location())
            .map(|(window, _)| window.clone())
        else {
            return;
        };
        self.space.raise_element(&window, true);
        self.raise_override_redirect_windows();
        self.damaged = true;
        let surface = FocusedSurface::of_window(&window);
        let keyboard = self.seat.get_keyboard().expect("the seat has a keyboard");
        keyboard.set_focus(self, surface, serial);
        for window in self.space.elements() {
            if let Some(toplevel) = window.toplevel() {
                toplevel.send_pending_configure();
            }
        }
    }

    fn handle_input(&mut self, input: WindowInput) {
        let time = InputTime::now();
        match input {
            WindowInput::Key { scancode, down } => {
                let key_state = if down {
                    KeyState::Pressed
                } else {
                    KeyState::Released
                };
                self.key(Keycode::new(scancode + XKB_KEYCODE_OFFSET), key_state, time);
            }
            WindowInput::Pointer { x, y } => {
                let location = Point::from((f64::from(x), f64::from(y)));
                let under = self.surface_under(location);
                let pointer = self.seat.get_pointer().expect("the seat has a pointer");
                let motion = MotionEvent {
                    location,
                    serial: SERIAL_COUNTER.next_serial(),
                    time,
                };
                pointer.motion(self, under, &motion);
                pointer.frame(self);
                if self.drag_icon.is_some() {
                    self.damaged = true;
                }
            }
            WindowInput::Button { button, down } => {
                let serial = SERIAL_COUNTER.next_serial();
                let pointer = self.seat.get_pointer().expect("the seat has a pointer");
                if down && !pointer.is_grabbed() {
                    self.focus_window_under_pointer(serial);
                }
                let state = if down {
                    ButtonState::Pressed
                } else {
                    ButtonState::Released
                };
                self.button(button.evdev_code(), state, serial, time);
            }
            WindowInput::Wheel { dx, dy } => {
                // linux wheel units scroll up for positive values, wayland axis values scroll down
                let horizontal = dx;
                let vertical = -dy;
                let mut frame = AxisFrame::new(time).source(AxisSource::Wheel);
                for (axis, units) in [(Axis::Horizontal, horizontal), (Axis::Vertical, vertical)] {
                    if units == 0 {
                        continue;
                    }
                    let pixels = f64::from(units) / f64::from(WHEEL_UNITS_PER_NOTCH)
                        * WHEEL_PIXELS_PER_NOTCH;
                    frame = frame.value(axis, pixels).v120(axis, units);
                }
                let pointer = self.seat.get_pointer().expect("the seat has a pointer");
                pointer.axis(self, frame);
                pointer.frame(self);
            }
            WindowInput::Focus(true) => {}
            WindowInput::Focus(false) => self.release_held_input(time),
        }
    }

    fn key(&mut self, keycode: Keycode, key_state: KeyState, time: InputTime) {
        let keyboard = self.seat.get_keyboard().expect("the seat has a keyboard");
        keyboard.input::<(), _>(
            self,
            keycode,
            key_state,
            SERIAL_COUNTER.next_serial(),
            time,
            |_, _, _| FilterResult::Forward,
        );
    }

    fn button(&mut self, button: u32, state: ButtonState, serial: Serial, time: InputTime) {
        self.held_buttons.retain(|held| *held != button);
        if state == ButtonState::Pressed {
            self.held_buttons.push(button);
        }
        let pointer = self.seat.get_pointer().expect("the seat has a pointer");
        pointer.button(
            self,
            &ButtonEvent {
                serial,
                time,
                button,
                state,
            },
        );
        pointer.frame(self);
    }

    fn release_held_input(&mut self, time: InputTime) {
        let keyboard = self.seat.get_keyboard().expect("the seat has a keyboard");
        for keycode in keyboard.pressed_keys() {
            self.key(keycode, KeyState::Released, time);
        }
        for button in core::mem::take(&mut self.held_buttons) {
            self.button(
                button,
                ButtonState::Released,
                SERIAL_COUNTER.next_serial(),
                time,
            );
        }
    }

    fn focus_top_window(&mut self) {
        let surface = self
            .space
            .elements()
            .last()
            .and_then(FocusedSurface::of_window);
        let keyboard = self.seat.get_keyboard().expect("the seat has a keyboard");
        keyboard.set_focus(self, surface, SERIAL_COUNTER.next_serial());
    }
}

impl CompositorHandler for DisplayState {
    fn compositor_state(&mut self) -> &mut CompositorState {
        &mut self.compositor_state
    }

    fn client_compositor_state<'a>(&self, client: &'a Client) -> &'a CompositorClientState {
        // smithay inserts the Xwayland client with its own client data
        if let Some(xwayland) = client.get_data::<XWaylandClientData>() {
            return &xwayland.compositor_state;
        }
        &client
            .get_data::<ClientState>()
            .expect("the socket inserts every other client with ClientState")
            .compositor_state
    }

    fn commit(&mut self, surface: &WlSurface) {
        on_commit_buffer_handler::<Self>(surface);
        if matches!(&self.cursor_status, CursorImageStatus::Surface(cursor) if cursor == surface) {
            self.publish_cursor();
            return;
        }
        self.damaged = true;
        if !is_sync_subsurface(surface) {
            let mut root = surface.clone();
            while let Some(parent) = get_parent(&root) {
                root = parent;
            }
            if let Some(window) = self.window_for(&root) {
                window.on_commit();
            }
        }
        if let Some(window) = self.window_for(surface) {
            let initial_configure_sent = with_states(surface, |states| {
                states
                    .data_map
                    .get::<XdgToplevelSurfaceData>()
                    .and_then(|data| data.lock().ok().map(|data| data.initial_configure_sent))
                    .unwrap_or(true)
            });
            if !initial_configure_sent {
                if let Some(toplevel) = window.toplevel() {
                    toplevel.send_configure();
                }
            }
        }
        if let Some(icon) = self
            .drag_icon
            .as_mut()
            .filter(|icon| icon.surface == *surface)
        {
            icon.offset += with_states(surface, |states| {
                states
                    .cached_state
                    .get::<SurfaceAttributes>()
                    .current()
                    .buffer_delta
                    .take()
                    .unwrap_or_default()
            });
        }
        self.popups.commit(surface);
        if let Some(PopupKind::Xdg(popup)) = self.popups.find_popup(surface) {
            if !popup.is_initial_configure_sent() {
                // xdg-shell always allows the initial configure
                let _ = popup.send_configure();
            }
        }
    }
}

impl DisplayState {
    fn window_for(&self, surface: &WlSurface) -> Option<Window> {
        self.space
            .elements()
            .find(|window| window.wl_surface().as_deref() == Some(surface))
            .cloned()
    }
}

impl BufferHandler for DisplayState {
    fn buffer_destroyed(&mut self, _buffer: &wl_buffer::WlBuffer) {}
}

impl DmabufHandler for DisplayState {
    fn dmabuf_state(&mut self) -> &mut DmabufState {
        &mut self.dmabuf_state
    }

    fn dmabuf_imported(
        &mut self,
        _global: &DmabufGlobal,
        dmabuf: Dmabuf,
        notifier: ImportNotifier,
    ) {
        let imported = match &mut self.renderer {
            Renderer::Software(software) => software.pixman.import_dmabuf(&dmabuf, None).is_ok(),
            Renderer::Gpu(gpu) => gpu.gles.import_dmabuf(&dmabuf, None).is_ok(),
        };
        if imported {
            // the client may disconnect before the reply
            let _ = notifier.successful::<Self>();
        } else {
            notifier.failed();
        }
    }
}

impl ShmHandler for DisplayState {
    fn shm_state(&self) -> &ShmState {
        &self.shm_state
    }
}

impl XdgShellHandler for DisplayState {
    fn xdg_shell_state(&mut self) -> &mut XdgShellState {
        &mut self.xdg_shell_state
    }

    fn new_toplevel(&mut self, surface: ToplevelSurface) {
        let output_size = self
            .output
            .current_mode()
            .map(|mode| mode.size.to_logical(1));
        surface.with_pending_state(|state| {
            state.size = output_size;
            state.states.set(xdg_toplevel::State::Fullscreen);
        });
        let wl_surface = surface.wl_surface().clone();
        self.space
            .map_element(Window::new_wayland_window(surface), (0, 0), true);
        self.raise_override_redirect_windows();
        let keyboard = self.seat.get_keyboard().expect("the seat has a keyboard");
        keyboard.set_focus(
            self,
            Some(FocusedSurface::Wayland(wl_surface)),
            SERIAL_COUNTER.next_serial(),
        );
        self.damaged = true;
    }

    fn toplevel_destroyed(&mut self, surface: ToplevelSurface) {
        if let Some(window) = self.window_for(surface.wl_surface()) {
            self.space.unmap_elem(&window);
        }
        self.focus_top_window();
        self.damaged = true;
    }

    fn new_popup(&mut self, surface: PopupSurface, _positioner: PositionerState) {
        if let Err(error) = self.popups.track_popup(PopupKind::Xdg(surface)) {
            g2g_core::g2g_warn!(Target::category(LOG_CATEGORY), "ignored a popup: {error:?}");
        }
    }

    fn popup_destroyed(&mut self, _surface: PopupSurface) {
        self.damaged = true;
    }

    fn grab(&mut self, _surface: PopupSurface, _seat: wl_seat::WlSeat, _serial: Serial) {}

    fn reposition_request(
        &mut self,
        surface: PopupSurface,
        positioner: PositionerState,
        token: u32,
    ) {
        surface.with_pending_state(|state| {
            state.geometry = positioner.get_geometry();
            state.positioner = positioner;
        });
        surface.send_repositioned(token);
    }
}

impl SeatHandler for DisplayState {
    type KeyboardFocus = FocusedSurface;
    type PointerFocus = FocusedSurface;
    type TouchFocus = WlSurface;

    fn seat_state(&mut self) -> &mut SeatState<Self> {
        &mut self.seat_state
    }

    fn cursor_image(&mut self, _seat: &Seat<Self>, image: CursorImageStatus) {
        self.cursor_status = image;
        self.publish_cursor();
    }

    fn focus_changed(&mut self, seat: &Seat<Self>, focused: Option<&FocusedSurface>) {
        let client = focused
            .and_then(WaylandFocus::wl_surface)
            .and_then(|surface| self.display_handle.get_client(surface.id()).ok());
        set_data_device_focus(&self.display_handle, seat, client);
        let Some(FocusedSurface::X11(surface)) = focused else {
            return;
        };
        let Some(window_manager) = self.window_manager.as_mut() else {
            return;
        };
        let raised = window_manager
            .raise_window(&**surface)
            .and_then(|()| surface.set_activated(true));
        if let Err(error) = raised {
            g2g_core::g2g_warn!(
                Target::category(LOG_CATEGORY),
                "could not activate an X11 window: {error}"
            );
        }
    }
}

impl SelectionHandler for DisplayState {
    type SelectionUserData = ();
}

impl DataDeviceHandler for DisplayState {
    fn data_device_state(&mut self) -> &mut DataDeviceState {
        &mut self.data_device_state
    }
}

impl DndGrabHandler for DisplayState {
    fn dropped(
        &mut self,
        _target: Option<DndTarget<'_, Self>>,
        _validated: bool,
        _seat: Seat<Self>,
        _location: Point<f64, Logical>,
    ) {
        self.drag_icon = None;
        self.damaged = true;
    }

    fn cancelled(&mut self, _seat: Seat<Self>, _location: Point<f64, Logical>) {
        self.drag_icon = None;
        self.damaged = true;
    }
}

impl WaylandDndGrabHandler for DisplayState {
    fn dnd_requested<S: Source>(
        &mut self,
        source: S,
        icon: Option<WlSurface>,
        seat: Seat<Self>,
        serial: Serial,
        grab_type: GrabType,
    ) {
        let pointer = seat.get_pointer().expect("the seat has a pointer");
        let (GrabType::Pointer, Some(start_data)) = (grab_type, pointer.grab_start_data()) else {
            source.cancel();
            return;
        };
        self.drag_icon = icon.map(|surface| DragIcon {
            surface,
            offset: Point::default(),
        });
        let grab = DnDGrab::new_pointer(&self.display_handle, start_data, source, seat);
        pointer.set_grab(self, grab, serial, Focus::Keep);
    }
}

impl OutputHandler for DisplayState {}

impl XWaylandShellHandler for DisplayState {
    fn xwayland_shell_state(&mut self) -> &mut XWaylandShellState {
        &mut self.xwayland_shell_state
    }
}

fn warn_x11(what: &str, error: impl core::fmt::Display) {
    g2g_core::g2g_warn!(Target::category(LOG_CATEGORY), "{what}: {error}");
}

impl XwmHandler for DisplayState {
    fn xwm_state(&mut self, _: XwmId) -> &mut X11Wm {
        self.window_manager
            .as_mut()
            .expect("X11 events arrive only from a running window manager")
    }

    fn new_window(&mut self, _: XwmId, _: X11Surface) {}

    fn new_override_redirect_window(&mut self, _: XwmId, _: X11Surface) {}

    fn map_window_request(&mut self, _: XwmId, surface: X11Surface) {
        let output = self.output_rectangle();
        if let Err(error) = surface.set_mapped(true) {
            warn_x11("could not map an X11 window", error);
            return;
        }
        if let Err(error) = surface.set_fullscreen(true) {
            warn_x11("could not make an X11 window fullscreen", error);
        }
        if let Err(error) = surface.configure(output) {
            warn_x11("could not size an X11 window", error);
        }
        self.space
            .map_element(Window::new_x11_window(surface.clone()), output.loc, true);
        self.raise_override_redirect_windows();
        let keyboard = self.seat.get_keyboard().expect("the seat has a keyboard");
        keyboard.set_focus(
            self,
            Some(FocusedSurface::X11(Box::new(surface))),
            SERIAL_COUNTER.next_serial(),
        );
        self.damaged = true;
    }

    fn mapped_override_redirect_window(&mut self, _: XwmId, surface: X11Surface) {
        let location = surface.last_configure().loc;
        self.space
            .map_element(Window::new_x11_window(surface), location, true);
        self.damaged = true;
    }

    fn unmapped_window(&mut self, _: XwmId, surface: X11Surface) {
        if let Some(window) = self.x11_window(&surface) {
            self.space.unmap_elem(&window);
        }
        if !surface.is_override_redirect() {
            if let Err(error) = surface.set_mapped(false) {
                warn_x11("could not unmap an X11 window", error);
            }
        }
        self.focus_top_window();
        self.damaged = true;
    }

    fn destroyed_window(&mut self, _: XwmId, _: X11Surface) {}

    fn configure_request(
        &mut self,
        _: XwmId,
        surface: X11Surface,
        _: Option<i32>,
        _: Option<i32>,
        _: Option<u32>,
        _: Option<u32>,
        _: Option<Reorder>,
    ) {
        // managed windows stay fullscreen whatever size they ask for
        if let Err(error) = surface.configure(self.output_rectangle()) {
            warn_x11("could not size an X11 window", error);
        }
    }

    fn configure_notify(
        &mut self,
        _: XwmId,
        surface: X11Surface,
        geometry: Rectangle<i32, Logical>,
        _: Option<u32>,
    ) {
        if let Some(window) = self.x11_window(&surface) {
            self.space.map_element(window, geometry.loc, false);
            self.raise_override_redirect_windows();
            self.damaged = true;
        }
    }

    fn resize_request(&mut self, _: XwmId, _: X11Surface, _: u32, _: ResizeEdge) {}

    fn move_request(&mut self, _: XwmId, _: X11Surface, _: u32) {}

    fn disconnected(&mut self, _: XwmId) {
        self.window_manager = None;
    }
}

impl PointerConstraintsHandler for DisplayState {}

smithay::delegate_dispatch2!(DisplayState);

impl core::fmt::Debug for DisplayState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // smithay's state types have no Debug
        f.debug_struct("DisplayState")
            .field("socket_name", &self.socket_name)
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}
