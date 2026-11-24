// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use g2g_core::log::Target;
use g2g_core::memory::OwnedDmaBuf;
use g2g_core::{G2gError, HardwareError};
use g2g_plugins::windowinput::{WindowInput, WHEEL_PIXELS_PER_NOTCH, WHEEL_UNITS_PER_NOTCH};
use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::allocator::Fourcc;
use smithay::backend::input::{Axis, AxisSource, ButtonState, InputTime, KeyState};
use smithay::backend::renderer::damage::OutputDamageTracker;
use smithay::backend::renderer::element::surface::WaylandSurfaceRenderElement;
use smithay::backend::renderer::pixman::PixmanRenderer;
use smithay::backend::renderer::utils::on_commit_buffer_handler;
use smithay::backend::renderer::{Bind, ImportDma, Offscreen};
use smithay::desktop::space::render_output;
use smithay::desktop::{PopupKind, PopupManager, Space, Window, WindowSurfaceType};
use smithay::input::dnd::DndGrabHandler;
use smithay::input::keyboard::{FilterResult, Keycode, XkbConfig};
use smithay::input::pointer::{AxisFrame, ButtonEvent, MotionEvent};
use smithay::input::{Seat, SeatHandler, SeatState};
use smithay::output::{Mode, Output, PhysicalProperties, Subpixel};
use smithay::reexports::calloop::channel::{Channel, Event as ChannelEvent};
use smithay::reexports::calloop::generic::Generic;
use smithay::reexports::calloop::{EventLoop, Interest, Mode as TriggerMode, PostAction};
use smithay::reexports::pixman::Image;
use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_toplevel;
use smithay::reexports::wayland_server::backend::ClientData;
use smithay::reexports::wayland_server::protocol::{wl_buffer, wl_seat, wl_surface::WlSurface};
use smithay::reexports::wayland_server::Resource;
use smithay::reexports::wayland_server::{Client, Display, DisplayHandle};
use smithay::utils::{Logical, Point, Serial, Transform, SERIAL_COUNTER};
use smithay::wayland::buffer::BufferHandler;
use smithay::wayland::compositor::{
    get_parent, is_sync_subsurface, with_states, CompositorClientState, CompositorHandler,
    CompositorState,
};
use smithay::wayland::dmabuf::{DmabufGlobal, DmabufHandler, DmabufState, ImportNotifier};
use smithay::wayland::output::{OutputHandler, OutputManagerState};
use smithay::wayland::pointer_constraints::PointerConstraintsHandler;
use smithay::wayland::selection::data_device::{
    set_data_device_focus, DataDeviceHandler, DataDeviceState, WaylandDndGrabHandler,
};
use smithay::wayland::selection::SelectionHandler;
use smithay::wayland::shell::xdg::{
    PopupSurface, PositionerState, ToplevelSurface, XdgShellHandler, XdgShellState,
    XdgToplevelSurfaceData,
};
use smithay::wayland::shm::{ShmHandler, ShmState};
use smithay::wayland::socket::ListeningSocketSource;
use tokio::sync::oneshot;

use crate::gpu::{GpuFormat, GpuOutput, GpuRenderer};

const LOG_CATEGORY: &str = "WaylandDisplaySrc";
// xkb keycodes are evdev codes offset by 8
const XKB_KEYCODE_OFFSET: u32 = 8;
const KEY_REPEAT_DELAY_MS: i32 = 200;
const KEY_REPEAT_RATE_HZ: i32 = 25;
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

pub(crate) type FrameReply = oneshot::Sender<Picture>;

#[derive(Debug, Clone)]
pub(crate) enum Rendering {
    Software,
    Gpu {
        render_node: PathBuf,
        output: GpuOutput,
        format: GpuFormat,
    },
}

pub(crate) struct DisplayChannels {
    pub(crate) frame_requests: Channel<FrameReply>,
    pub(crate) input: Channel<WindowInput>,
    pub(crate) socket_name: oneshot::Sender<Result<String, G2gError>>,
}

pub(crate) fn hardware_error(what: &str, error: impl core::fmt::Debug) -> G2gError {
    g2g_core::g2g_error!(Target::category(LOG_CATEGORY), "{what}: {error:?}");
    G2gError::Hardware(HardwareError::Other)
}

pub(crate) fn run_display(
    size: DisplaySize,
    pacing: Pacing,
    rendering: Rendering,
    channels: DisplayChannels,
) -> Result<(), G2gError> {
    let DisplayChannels {
        frame_requests,
        input,
        socket_name,
    } = channels;
    let started = start_display(size, pacing, rendering, frame_requests, input);
    let (mut event_loop, mut state) = match started {
        Ok(started) => started,
        Err(error) => {
            let _ = socket_name.send(Err(error.clone()));
            return Err(error);
        }
    };
    let _ = socket_name.send(Ok(state.socket_name.clone()));
    while !state.stopped {
        let timeout = state.time_until_next_frame(Instant::now());
        event_loop
            .dispatch(timeout, &mut state)
            .map_err(|error| hardware_error("wayland event loop", error))?;
        state.serve_frame_request(Instant::now())?;
        state
            .display_handle
            .flush_clients()
            .map_err(|error| hardware_error("flushing wayland clients", error))?;
    }
    Ok(())
}

fn start_display(
    size: DisplaySize,
    pacing: Pacing,
    rendering: Rendering,
    frame_requests: Channel<FrameReply>,
    input: Channel<WindowInput>,
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
        .insert_source(frame_requests, |event, _, state| match event {
            ChannelEvent::Msg(reply) => state.pending_reply = Some(reply),
            ChannelEvent::Closed => state.stopped = true,
        })
        .map_err(|error| hardware_error("receiving frame requests", error))?;
    loop_handle
        .insert_source(input, |event, _, state| {
            if let ChannelEvent::Msg(input) = event {
                state.handle_input(input);
            }
        })
        .map_err(|error| hardware_error("receiving input", error))?;

    let state = DisplayState::new(display_handle, socket_name, size, pacing, rendering)?;
    Ok((event_loop, state))
}

#[derive(Debug, Default)]
struct ClientState {
    compositor_state: CompositorClientState,
}

impl ClientData for ClientState {}

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
    socket_name: String,
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
    stopped: bool,
}

fn render_software(
    software: &mut SoftwareRenderer,
    output: &Output,
    space: &Space<Window>,
    damage_tracker: &mut OutputDamageTracker,
) -> Result<(), G2gError> {
    let age = usize::from(software.rendered_once);
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
        &[],
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
        socket_name: String,
        size: DisplaySize,
        pacing: Pacing,
        rendering: Rendering,
    ) -> Result<Self, G2gError> {
        let compositor_state = CompositorState::new::<Self>(&display_handle);
        let xdg_shell_state = XdgShellState::new::<Self>(&display_handle);
        let shm_state = ShmState::new::<Self>(&display_handle, Vec::new());
        let data_device_state = DataDeviceState::new::<Self>(&display_handle);
        let output_manager_state = OutputManagerState::new_with_xdg_output::<Self>(&display_handle);
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
            Rendering::Gpu {
                render_node,
                output,
                format,
            } => Renderer::Gpu(Box::new(GpuRenderer::open(
                &render_node,
                size,
                output,
                format,
            )?)),
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

        Ok(Self {
            display_handle,
            socket_name,
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
            stopped: false,
        })
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

    fn serve_frame_request(&mut self, now: Instant) -> Result<(), G2gError> {
        let Some(due) = self.next_frame_due() else {
            return Ok(());
        };
        if now < due {
            return Ok(());
        }
        let picture = match self.last_frame.take() {
            Some((_, picture)) if !self.damaged => picture,
            _ => self.render()?,
        };
        if let Some(reply) = self.pending_reply.take() {
            // the element drops its receiver when the run ends
            let _ = reply.send(picture.clone());
        }
        self.last_frame = Some((now, picture));
        Ok(())
    }

    fn render(&mut self) -> Result<Picture, G2gError> {
        let picture = match &mut self.renderer {
            Renderer::Software(software) => {
                render_software(
                    software,
                    &self.output,
                    &self.space,
                    &mut self.damage_tracker,
                )?;
                Picture::System(copy_picture(&software.image, self.size))
            }
            Renderer::Gpu(gpu) => {
                gpu.render(&self.output, &self.space, &mut self.damage_tracker)?
            }
        };
        self.damaged = false;

        let elapsed = self.started.elapsed();
        for window in self.space.elements() {
            window.send_frame(&self.output, elapsed, Some(Duration::ZERO), |_, _| {
                Some(self.output.clone())
            });
        }
        self.space.refresh();
        self.popups.cleanup();
        Ok(picture)
    }

    fn surface_under(
        &self,
        location: Point<f64, Logical>,
    ) -> Option<(WlSurface, Point<f64, Logical>)> {
        let (window, window_location) = self.space.element_under(location)?;
        window
            .surface_under(location - window_location.to_f64(), WindowSurfaceType::ALL)
            .map(|(surface, surface_location)| {
                (surface, (surface_location + window_location).to_f64())
            })
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
        let surface = window
            .toplevel()
            .map(|toplevel| toplevel.wl_surface().clone());
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
            .and_then(|window| window.toplevel())
            .map(|toplevel| toplevel.wl_surface().clone());
        let keyboard = self.seat.get_keyboard().expect("the seat has a keyboard");
        keyboard.set_focus(self, surface, SERIAL_COUNTER.next_serial());
    }
}

impl CompositorHandler for DisplayState {
    fn compositor_state(&mut self) -> &mut CompositorState {
        &mut self.compositor_state
    }

    fn client_compositor_state<'a>(&self, client: &'a Client) -> &'a CompositorClientState {
        &client
            .get_data::<ClientState>()
            .expect("every client is inserted with ClientState")
            .compositor_state
    }

    fn commit(&mut self, surface: &WlSurface) {
        on_commit_buffer_handler::<Self>(surface);
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
            .find(|window| {
                window
                    .toplevel()
                    .is_some_and(|toplevel| toplevel.wl_surface() == surface)
            })
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
        let keyboard = self.seat.get_keyboard().expect("the seat has a keyboard");
        keyboard.set_focus(self, Some(wl_surface), SERIAL_COUNTER.next_serial());
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
    type KeyboardFocus = WlSurface;
    type PointerFocus = WlSurface;
    type TouchFocus = WlSurface;

    fn seat_state(&mut self) -> &mut SeatState<Self> {
        &mut self.seat_state
    }

    fn focus_changed(&mut self, seat: &Seat<Self>, focused: Option<&WlSurface>) {
        let client = focused.and_then(|surface| self.display_handle.get_client(surface.id()).ok());
        set_data_device_focus(&self.display_handle, seat, client);
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

impl DndGrabHandler for DisplayState {}

impl WaylandDndGrabHandler for DisplayState {}

impl OutputHandler for DisplayState {}

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
