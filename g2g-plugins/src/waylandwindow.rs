// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Shared Wayland window worker for the display sinks that own their window.
//!
//! Wayland (and the graphics API drawn on it) is single-thread-affine, so each
//! such sink runs its window on a dedicated worker thread. Everything that thread
//! does apart from getting one frame's pixels onto the surface is the same for
//! every sink: connect to the compositor, map an `xdg_toplevel`, build the
//! renderer over that surface, then loop on a `calloop` channel drawing each
//! handed-over frame and presenting it. That is [`run_window`]; the sink supplies
//! a [`WindowRenderer`] for the graphics half (EGL + GL ES for
//! [`crate::glwindow`], a `wgpu::Surface` for [`crate::wgpupresent`]).
//!
//! The first `xdg` configure signals the sink's readiness handshake, and a frame
//! that arrives before the surface is mappable is held and drawn then.

use core::sync::atomic::{AtomicBool, Ordering};
use std::io::{Read, Write};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use alloc::boxed::Box;
use alloc::string::String;

use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState, FrameCallbackData},
    data_device_manager::{
        data_device::{DataDevice, DataDeviceData, DataDeviceHandler},
        data_offer::{DataOfferHandler, DragOffer},
        data_source::{CopyPasteSource, DataSourceHandler},
        DataDeviceManagerState, WritePipe,
    },
    delegate_dispatch2, delegate_registry,
    output::{OutputHandler, OutputState},
    reexports::calloop::{
        channel::{Channel, Event as ChanEvent},
        EventLoop,
    },
    reexports::calloop_wayland_source::WaylandSource,
    reexports::client::{
        globals::registry_queue_init,
        protocol::{
            wl_data_device::WlDataDevice, wl_data_device_manager::DndAction,
            wl_data_source::WlDataSource, wl_keyboard, wl_output, wl_pointer, wl_seat, wl_shm,
            wl_surface,
        },
        Connection, Proxy, QueueHandle,
    },
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    seat::{
        keyboard::{KeyEvent, KeyboardHandler, Keysym, Modifiers, RawModifiers},
        pointer::{AxisScroll, PointerEvent, PointerEventKind, PointerHandler},
        Capability, SeatHandler, SeatState,
    },
    shell::{
        xdg::{
            window::{Window, WindowConfigure, WindowDecorations, WindowHandler},
            XdgShell,
        },
        WaylandSurface,
    },
    shm::{
        slot::{Buffer, SlotPool},
        Shm, ShmHandler,
    },
};

use crate::cursorshape::{CursorShape, CURSOR_BYTES_PER_PIXEL};
use crate::windowinput::{
    PointerButton, WindowInput, CLIPBOARD_TEXT_MIME_TYPES, WHEEL_PIXELS_PER_NOTCH,
    WHEEL_UNITS_PER_NOTCH,
};
use crate::windowworker::{
    ClipboardLink, PresentReports, WindowParams, WindowRenderer, WorkerChannels, WorkerCmd,
};
use crate::worker_ready::Handshake;
use g2g_core::log::Target;
#[cfg(feature = "wgpu-present")]
use g2g_core::Caps;
use g2g_core::G2gError;

/// Open the Wayland window, build the sink's renderer over its surface, and run
/// the present loop until the sink sends `Shutdown` (or the compositor closes the
/// window). `build_renderer` is called once the surface exists, on this thread,
/// because a graphics context is bound to the thread that made it. Runs on the
/// sink's worker thread and returns when the loop exits.
pub(crate) fn run_window<R: WindowRenderer>(
    build_renderer: impl FnOnce(
        &Connection,
        &wl_surface::WlSurface,
        u32,
        u32,
    ) -> Result<R, Box<dyn std::error::Error>>,
    params: WindowParams,
    channels: WorkerChannels<Channel<WorkerCmd<R::Frame>>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let WindowParams {
        width,
        height,
        title,
        app_id,
        fullscreen,
        log_tag,
    } = params;
    let conn = Connection::connect_to_env()?;
    let (globals, event_queue) = registry_queue_init(&conn)?;
    let qh = event_queue.handle();

    let mut event_loop: EventLoop<WindowWorker<R>> = EventLoop::try_new()?;
    let loop_handle = event_loop.handle();
    WaylandSource::new(conn.clone(), event_queue).insert(loop_handle.clone())?;

    let compositor = CompositorState::bind(&globals, &qh)?;
    let xdg_shell = XdgShell::bind(&globals, &qh)?;
    let shm = Shm::bind(&globals, &qh)?;
    let cursor_surface = compositor.create_surface(&qh);

    let surface = compositor.create_surface(&qh);
    let window = xdg_shell.create_window(surface, WindowDecorations::RequestServer, &qh);
    window.set_title(&title);
    window.set_app_id(&app_id);
    window.set_min_size(Some((width, height)));
    if fullscreen {
        window.set_fullscreen(None);
    }
    window.commit();

    let renderer = build_renderer(&conn, window.wl_surface(), width, height)?;
    let clipboard =
        channels
            .clipboard
            .and_then(|link| match DataDeviceManagerState::bind(&globals, &qh) {
                Ok(manager) => Some(WindowClipboard {
                    link,
                    manager,
                    device: None,
                    source: None,
                    offered: None,
                    offer_pending: false,
                    input_serial: None,
                    synced: Arc::default(),
                }),
                Err(e) => {
                    g2g_core::g2g_warn!(
                        Target::category(log_tag),
                        "the compositor has no clipboard: {e}"
                    );
                    None
                }
            });

    let mut state = WindowWorker {
        registry_state: RegistryState::new(&globals),
        output_state: OutputState::new(&globals, &qh),
        renderer,
        window,
        qh: qh.clone(),
        configured: false,
        exit: false,
        ready: Some(channels.ready),
        reports: PresentReports {
            presented: channels.presented,
            latency: channels.latency,
            log_tag,
        },
        window_closed: channels.window_closed,
        pending: None,
        seat_state: SeatState::new(&globals, &qh),
        input: channels.input,
        keyboard: None,
        pointer: None,
        shm,
        cursor: PointerCursor {
            shapes: channels.cursor_shapes,
            surface: cursor_surface,
            pool: None,
            buffer: None,
            enter_serial: None,
        },
        clipboard,
    };

    loop_handle.insert_source(
        channels.rx,
        |event, _, state: &mut WindowWorker<R>| match event {
            ChanEvent::Msg(WorkerCmd::Frame {
                frame,
                arrival_ns,
                ack,
            }) => {
                if state.configured {
                    state.draw(frame, arrival_ns, ack);
                } else {
                    state.pending = Some((frame, arrival_ns, ack));
                }
            }
            #[cfg(feature = "wgpu-present")]
            ChanEvent::Msg(WorkerCmd::Reconfigure {
                caps,
                width,
                height,
            }) => state.reconfigure(&caps, width, height),
            ChanEvent::Msg(WorkerCmd::Shutdown) | ChanEvent::Closed => {
                state.exit = true;
            }
        },
    )?;

    while !state.exit {
        event_loop.dispatch(Some(Duration::from_millis(100)), &mut state)?;
        state.follow_cursor_shape();
        state.follow_remote_clipboard();
    }
    Ok(())
}

// =================================================================
// Worker state + SCTK handlers
// =================================================================

struct WindowWorker<R: WindowRenderer> {
    registry_state: RegistryState,
    output_state: OutputState,
    /// Declared before `window`: the renderer's surface is built over the
    /// `wl_surface` and has to be torn down before it.
    renderer: R,
    window: Window,
    qh: QueueHandle<WindowWorker<R>>,
    configured: bool,
    exit: bool,
    ready: Option<Arc<Handshake>>,
    reports: PresentReports,
    window_closed: Arc<AtomicBool>,
    /// Frame that arrived before the surface was mappable.
    pending: Option<(R::Frame, u64, tokio::sync::oneshot::Sender<()>)>,
    seat_state: SeatState,
    input: Option<tokio::sync::mpsc::UnboundedSender<WindowInput>>,
    keyboard: Option<wl_keyboard::WlKeyboard>,
    pointer: Option<wl_pointer::WlPointer>,
    shm: Shm,
    cursor: PointerCursor,
    clipboard: Option<WindowClipboard>,
}

struct WindowClipboard {
    link: ClipboardLink,
    manager: DataDeviceManagerState,
    device: Option<DataDevice>,
    source: Option<CopyPasteSource>,
    // the text this window's selection serves
    offered: Option<String>,
    offer_pending: bool,
    // wayland lets a client set the selection only with the serial of a recent input event
    input_serial: Option<u32>,
    // the text last set or read in either direction, so neither side echoes it back
    synced: Arc<Mutex<Option<String>>>,
}

struct PointerCursor {
    shapes: Option<crate::windowworker::CursorShapes>,
    surface: wl_surface::WlSurface,
    pool: Option<SlotPool>,
    // wayland reads the attached buffer until another replaces it
    buffer: Option<Buffer>,
    enter_serial: Option<u32>,
}

fn premultiplied_argb(rgba: &[u8]) -> impl Iterator<Item = [u8; 4]> + '_ {
    // wl_shm ARGB8888 is premultiplied and little-endian, so each pixel is B, G, R, A in memory
    rgba.as_chunks::<CURSOR_BYTES_PER_PIXEL>()
        .0
        .iter()
        .map(|&[r, g, b, a]| {
            let scale = |c: u8| (u16::from(c) * u16::from(a) / u16::from(u8::MAX)) as u8;
            [scale(b), scale(g), scale(r), a]
        })
}

impl<R: WindowRenderer> WindowWorker<R> {
    /// Draw one frame and present it. Signals `ack` after the present returns
    /// (compositor-paced backpressure).
    fn draw(&mut self, frame: R::Frame, arrival_ns: u64, ack: tokio::sync::oneshot::Sender<()>) {
        let drawn = self.draw_inner(&frame);
        self.reports.acknowledge(drawn, frame, arrival_ns, ack);
    }

    #[cfg(feature = "wgpu-present")]
    /// Point the renderer at the new caps and ask the compositor for the size
    /// that goes with them. A renderer that cannot take them leaves the window
    /// showing a stale layout, so the worker exits instead and the pipeline fails
    /// on the next frame.
    fn reconfigure(&mut self, caps: &Caps, width: u32, height: u32) {
        if let Err(e) = self.renderer.reconfigure(caps) {
            g2g_core::g2g_error!(
                Target::category(self.reports.log_tag),
                "cannot apply the new caps to the window: {e:?}"
            );
            self.exit = true;
            return;
        }
        self.window.set_min_size(Some((width, height)));
        self.window.commit();
    }

    fn follow_remote_clipboard(&mut self) {
        let Some(clipboard) = &mut self.clipboard else {
            return;
        };
        if !clipboard.link.remote_texts.has_changed().unwrap_or(false) {
            return;
        }
        let Some(text) = clipboard.link.remote_texts.borrow_and_update().clone() else {
            return;
        };
        clipboard.offered = Some(text);
        clipboard.offer_pending = true;
        self.offer_clipboard();
    }

    fn note_input_serial(&mut self, serial: u32) {
        let Some(clipboard) = &mut self.clipboard else {
            return;
        };
        clipboard.input_serial = Some(serial);
        if clipboard.offer_pending {
            self.offer_clipboard();
        }
    }

    fn offer_clipboard(&mut self) {
        let Some(clipboard) = &mut self.clipboard else {
            return;
        };
        let (Some(text), Some(device), Some(serial)) = (
            &clipboard.offered,
            &clipboard.device,
            clipboard.input_serial,
        ) else {
            return;
        };
        *clipboard
            .synced
            .lock()
            .expect("no thread panics holding the synced text") = Some(text.clone());
        let source = clipboard
            .manager
            .create_copy_paste_source(&self.qh, CLIPBOARD_TEXT_MIME_TYPES);
        source.set_selection(device, serial);
        clipboard.source = Some(source);
        clipboard.offer_pending = false;
    }

    fn follow_cursor_shape(&mut self) {
        let Some(shapes) = &mut self.cursor.shapes else {
            return;
        };
        if shapes.has_changed().unwrap_or(false) {
            shapes.mark_unchanged();
            self.show_cursor();
        }
    }

    fn show_cursor(&mut self) {
        let (Some(shapes), Some(pointer), Some(serial)) =
            (&self.cursor.shapes, &self.pointer, self.cursor.enter_serial)
        else {
            return;
        };
        let shape = shapes.borrow().clone();
        let Some(shape) = shape else {
            pointer.set_cursor(serial, None, 0, 0);
            return;
        };
        if let Err(e) = self.draw_cursor(&shape) {
            g2g_core::g2g_warn!(
                Target::category(self.reports.log_tag),
                "cannot draw the {}x{} cursor: {e}",
                shape.width,
                shape.height
            );
            return;
        }
        if let Some(pointer) = &self.pointer {
            pointer.set_cursor(
                serial,
                Some(&self.cursor.surface),
                shape.hotspot_x,
                shape.hotspot_y,
            );
        }
    }

    fn draw_cursor(&mut self, shape: &CursorShape) -> Result<(), Box<dyn std::error::Error>> {
        let width = i32::try_from(shape.width)?;
        let height = i32::try_from(shape.height)?;
        let stride = width * CURSOR_BYTES_PER_PIXEL as i32;
        let pool = match &mut self.cursor.pool {
            Some(pool) => pool,
            None => self
                .cursor
                .pool
                .insert(SlotPool::new(shape.rgba.len().max(1), &self.shm)?),
        };
        let (buffer, canvas) =
            pool.create_buffer(width, height, stride, wl_shm::Format::Argb8888)?;
        for (target, pixel) in canvas
            .as_chunks_mut::<CURSOR_BYTES_PER_PIXEL>()
            .0
            .iter_mut()
            .zip(premultiplied_argb(&shape.rgba))
        {
            *target = pixel;
        }
        let surface = &self.cursor.surface;
        surface.attach(Some(buffer.wl_buffer()), 0, 0);
        surface.damage_buffer(0, 0, width, height);
        surface.commit();
        self.cursor.buffer = Some(buffer);
        Ok(())
    }

    fn send_input(&mut self, event: WindowInput) {
        let Some(input) = &self.input else {
            return;
        };
        if input.send(event).is_err() {
            self.input = None;
        }
    }

    fn draw_inner(&mut self, frame: &R::Frame) -> Result<(), G2gError> {
        // Subscribe to the next frame callback (compositor pacing) before the
        // present commits the surface.
        let surface = self.window.wl_surface().clone();
        surface.frame(&self.qh, FrameCallbackData(surface.clone()));
        self.renderer.present(frame)
    }
}

impl<R: WindowRenderer> CompositorHandler for WindowWorker<R> {
    fn scale_factor_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: i32,
    ) {
    }
    fn transform_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: wl_output::Transform,
    ) {
    }
    fn frame(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: u32) {
        // Compositor released the buffer; pacing is handled by the per-frame
        // ack in `draw`, so nothing extra is needed here.
    }
    fn surface_enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: &wl_output::WlOutput,
    ) {
    }
    fn surface_leave(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: &wl_output::WlOutput,
    ) {
    }
}

impl<R: WindowRenderer> WindowHandler for WindowWorker<R> {
    fn request_close(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &Window) {
        self.window_closed.store(true, Ordering::Release);
        self.exit = true;
    }

    fn configure(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &Window,
        configure: WindowConfigure,
        _serial: u32,
    ) {
        if let (Some(width), Some(height)) = configure.new_size {
            self.renderer.resize(width.get(), height.get());
        }
        let was_first = !self.configured;
        self.configured = true;
        if was_first {
            if let Some(ready) = self.ready.take() {
                ready.notify();
            }
            if let Some((frame, arrival_ns, ack)) = self.pending.take() {
                self.draw(frame, arrival_ns, ack);
            }
        }
    }
}

impl<R: WindowRenderer> OutputHandler for WindowWorker<R> {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }
    fn new_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn update_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn output_destroyed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
}

impl<R: WindowRenderer> ProvidesRegistryState for WindowWorker<R> {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState, SeatState,];
}

impl<R: WindowRenderer> SeatHandler for WindowWorker<R> {
    fn seat_state(&mut self) -> &mut SeatState {
        &mut self.seat_state
    }

    fn new_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {}

    fn new_capability(
        &mut self,
        _: &Connection,
        qh: &QueueHandle<Self>,
        seat: wl_seat::WlSeat,
        capability: Capability,
    ) {
        if self.input.is_none() {
            return;
        }
        let log = Target::category(self.reports.log_tag);
        match capability {
            Capability::Keyboard if self.keyboard.is_none() => {
                match self.seat_state.get_keyboard(qh, &seat, None) {
                    Ok(keyboard) => self.keyboard = Some(keyboard),
                    Err(e) => g2g_core::g2g_error!(log, "cannot read the keyboard: {e}"),
                }
                if let Some(clipboard) = &mut self.clipboard {
                    clipboard.device = Some(clipboard.manager.get_data_device(qh, &seat));
                }
            }
            Capability::Pointer if self.pointer.is_none() => {
                match self.seat_state.get_pointer(qh, &seat) {
                    Ok(pointer) => self.pointer = Some(pointer),
                    Err(e) => g2g_core::g2g_error!(log, "cannot read the pointer: {e}"),
                }
            }
            _ => {}
        }
    }

    fn remove_capability(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: wl_seat::WlSeat,
        capability: Capability,
    ) {
        match capability {
            Capability::Keyboard => {
                if let Some(keyboard) = self.keyboard.take() {
                    keyboard.release();
                }
            }
            Capability::Pointer => {
                if let Some(pointer) = self.pointer.take() {
                    pointer.release();
                }
            }
            _ => {}
        }
    }

    fn remove_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {}
}

impl<R: WindowRenderer> KeyboardHandler for WindowWorker<R> {
    fn enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: &wl_surface::WlSurface,
        serial: u32,
        _: &[u32],
        _: &[Keysym],
    ) {
        self.note_input_serial(serial);
        self.send_input(WindowInput::Focus(true));
    }

    fn leave(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: &wl_surface::WlSurface,
        _: u32,
    ) {
        self.send_input(WindowInput::Focus(false));
    }

    fn press_key(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        serial: u32,
        event: KeyEvent,
    ) {
        self.note_input_serial(serial);
        self.send_input(WindowInput::Key {
            scancode: event.raw_code,
            down: true,
        });
    }

    fn repeat_key(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        _: KeyEvent,
    ) {
    }

    fn release_key(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        event: KeyEvent,
    ) {
        self.send_input(WindowInput::Key {
            scancode: event.raw_code,
            down: false,
        });
    }

    fn update_modifiers(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        _: Modifiers,
        _: RawModifiers,
        _: u32,
    ) {
    }
}

fn wheel_units(scroll: &AxisScroll) -> i32 {
    if scroll.value120 != 0 {
        return scroll.value120;
    }
    if scroll.discrete != 0 {
        return scroll.discrete.saturating_mul(WHEEL_UNITS_PER_NOTCH);
    }
    let notches = scroll.absolute / WHEEL_PIXELS_PER_NOTCH;
    (notches * f64::from(WHEEL_UNITS_PER_NOTCH)).round() as i32
}

fn wheel_input(horizontal: &AxisScroll, vertical: &AxisScroll) -> Option<WindowInput> {
    let dx = wheel_units(horizontal);
    // wayland scrolls down for a positive vertical value, linux REL_WHEEL scrolls up
    let dy = wheel_units(vertical).saturating_neg();
    (dx != 0 || dy != 0).then_some(WindowInput::Wheel { dx, dy })
}

impl<R: WindowRenderer> PointerHandler for WindowWorker<R> {
    fn pointer_frame(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_pointer::WlPointer,
        events: &[PointerEvent],
    ) {
        for event in events {
            match event.kind {
                PointerEventKind::Enter { .. } | PointerEventKind::Motion { .. } => {
                    if let PointerEventKind::Enter { serial } = event.kind {
                        self.cursor.enter_serial = Some(serial);
                        self.show_cursor();
                    }
                    let (x, y) = event.position;
                    if let Some((x, y)) = self.renderer.picture_point(x, y) {
                        self.send_input(WindowInput::Pointer { x, y });
                    }
                }
                PointerEventKind::Press { button, .. }
                | PointerEventKind::Release { button, .. } => {
                    let down = matches!(event.kind, PointerEventKind::Press { .. });
                    if let Some(button) = PointerButton::from_evdev_code(button) {
                        self.send_input(WindowInput::Button { button, down });
                    }
                }
                PointerEventKind::Axis {
                    horizontal,
                    vertical,
                    ..
                } => {
                    if let Some(wheel) = wheel_input(&horizontal, &vertical) {
                        self.send_input(wheel);
                    }
                }
                PointerEventKind::Leave { .. } => {}
            }
        }
    }
}

impl<R: WindowRenderer> DataDeviceHandler for WindowWorker<R> {
    fn enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &WlDataDevice,
        _: f64,
        _: f64,
        _: &wl_surface::WlSurface,
    ) {
    }

    fn leave(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataDevice) {}

    fn motion(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataDevice, _: f64, _: f64) {}

    fn selection(&mut self, _: &Connection, _: &QueueHandle<Self>, data_device: &WlDataDevice) {
        let Some(clipboard) = &self.clipboard else {
            return;
        };
        let Some(offer) = data_device
            .data::<DataDeviceData>()
            .and_then(DataDeviceData::selection_offer)
        else {
            return;
        };
        let mime_type = offer.with_mime_types(|offered| {
            CLIPBOARD_TEXT_MIME_TYPES
                .iter()
                .find(|text| offered.iter().any(|mime| mime == *text))
                .map(|text| String::from(*text))
        });
        let Some(mime_type) = mime_type else {
            return;
        };
        let log = Target::category(self.reports.log_tag);
        let mut pipe = match offer.receive(mime_type) {
            Ok(pipe) => pipe,
            Err(e) => {
                g2g_core::g2g_warn!(log, "cannot read the clipboard: {e}");
                return;
            }
        };
        let synced = Arc::clone(&clipboard.synced);
        let local_texts = clipboard.link.local_texts.clone();
        // the copying client writes the text, and that is this thread when the window set the selection
        std::thread::spawn(move || {
            let mut text = String::new();
            if let Err(e) = pipe.read_to_string(&mut text) {
                g2g_core::g2g_warn!(log, "cannot read the clipboard: {e}");
                return;
            }
            let mut synced = synced
                .lock()
                .expect("no thread panics holding the synced text");
            if synced.as_deref() == Some(text.as_str()) {
                return;
            }
            *synced = Some(text.clone());
            let _ = local_texts.send(text);
        });
    }

    fn drop_performed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataDevice) {}
}

impl<R: WindowRenderer> DataOfferHandler for WindowWorker<R> {
    fn source_actions(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &mut DragOffer,
        _: DndAction,
    ) {
    }

    fn selected_action(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &mut DragOffer,
        _: DndAction,
    ) {
    }
}

impl<R: WindowRenderer> DataSourceHandler for WindowWorker<R> {
    fn accept_mime(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &WlDataSource,
        _: Option<String>,
    ) {
    }

    fn send_request(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        source: &WlDataSource,
        _: String,
        mut pipe: WritePipe,
    ) {
        let Some(clipboard) = &self.clipboard else {
            return;
        };
        let ours = clipboard
            .source
            .as_ref()
            .is_some_and(|ours| ours.inner() == source);
        let Some(text) = clipboard.offered.as_ref().filter(|_| ours) else {
            return;
        };
        if let Err(e) = pipe.write_all(text.as_bytes()) {
            g2g_core::g2g_warn!(
                Target::category(self.reports.log_tag),
                "cannot hand out the clipboard: {e}"
            );
        }
    }

    fn cancelled(&mut self, _: &Connection, _: &QueueHandle<Self>, source: &WlDataSource) {
        let Some(clipboard) = &mut self.clipboard else {
            return;
        };
        if clipboard
            .source
            .as_ref()
            .is_some_and(|ours| ours.inner() == source)
        {
            clipboard.source = None;
        }
    }

    fn dnd_dropped(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataSource) {}

    fn dnd_finished(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataSource) {}

    fn action(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataSource, _: DndAction) {}
}

delegate_dispatch2!(@<R: WindowRenderer> WindowWorker<R>);
delegate_registry!(@<R: WindowRenderer> WindowWorker<R>);

impl<R: WindowRenderer> ShmHandler for WindowWorker<R> {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::windowworker::present_on_worker;
    use g2g_core::runtime::block_on;
    use g2g_core::HardwareError;
    use smithay_client_toolkit::reexports::calloop::channel::channel;

    fn scroll(value120: i32, discrete: i32, absolute: f64) -> AxisScroll {
        AxisScroll {
            absolute,
            discrete,
            value120,
            ..AxisScroll::default()
        }
    }

    #[test]
    fn cursor_pixels_become_premultiplied_bgra_in_memory() {
        let rgba = [[200, 100, 50, 255], [200, 100, 50, 0], [255, 0, 255, 128]].concat();
        let argb: alloc::vec::Vec<[u8; 4]> = premultiplied_argb(&rgba).collect();
        assert_eq!(
            argb,
            [[50, 100, 200, 255], [0, 0, 0, 0], [128, 0, 128, 128]]
        );
    }

    #[test]
    fn wheel_down_one_notch_is_minus_120_and_falls_back_to_discrete_then_pixels() {
        let none = AxisScroll::default();
        let one_notch_down = scroll(WHEEL_UNITS_PER_NOTCH, 1, WHEEL_PIXELS_PER_NOTCH);
        assert_eq!(
            wheel_input(&none, &one_notch_down),
            Some(WindowInput::Wheel {
                dx: 0,
                dy: -WHEEL_UNITS_PER_NOTCH
            })
        );
        let old_compositor_right = scroll(0, 2, 2.0 * WHEEL_PIXELS_PER_NOTCH);
        assert_eq!(
            wheel_input(&old_compositor_right, &none),
            Some(WindowInput::Wheel {
                dx: 2 * WHEEL_UNITS_PER_NOTCH,
                dy: 0
            })
        );
        let touchpad_half_notch_up = scroll(0, 0, -WHEEL_PIXELS_PER_NOTCH / 2.0);
        assert_eq!(
            wheel_input(&none, &touchpad_half_notch_up),
            Some(WindowInput::Wheel {
                dx: 0,
                dy: WHEEL_UNITS_PER_NOTCH / 2
            })
        );
        assert_eq!(
            wheel_input(&none, &none),
            None,
            "a scroll stop moves nothing"
        );
    }

    #[test]
    fn a_frame_for_a_window_the_user_closed_fails_with_window_closed() {
        let (tx, rx) = channel::<WorkerCmd<()>>();
        drop(rx);
        let window_closed = AtomicBool::new(true);
        assert_eq!(
            block_on(present_on_worker(&tx, (), 0, &window_closed)),
            Err(G2gError::WindowClosed)
        );
        window_closed.store(false, Ordering::Release);
        assert_eq!(
            block_on(present_on_worker(&tx, (), 0, &window_closed)),
            Err(G2gError::Hardware(HardwareError::Other))
        );
    }
}
