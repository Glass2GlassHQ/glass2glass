// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

use g2g_core::PipelineClock;
use smithay_client_toolkit::compositor::{CompositorHandler, CompositorState};
use smithay_client_toolkit::output::{OutputHandler, OutputState};
use smithay_client_toolkit::reexports::client::globals::registry_queue_init;
use smithay_client_toolkit::reexports::client::protocol::{
    wl_keyboard, wl_output, wl_pointer, wl_seat, wl_shm, wl_surface,
};
use smithay_client_toolkit::reexports::client::{Connection, QueueHandle};
use smithay_client_toolkit::registry::{ProvidesRegistryState, RegistryState};
use smithay_client_toolkit::seat::keyboard::{
    KeyEvent, KeyboardHandler, Keysym, Modifiers, RawModifiers,
};
use smithay_client_toolkit::seat::pointer::{PointerEvent, PointerEventKind, PointerHandler};
use smithay_client_toolkit::seat::{Capability, SeatHandler, SeatState};
use smithay_client_toolkit::shell::xdg::window::{
    Window, WindowConfigure, WindowDecorations, WindowHandler,
};
use smithay_client_toolkit::shell::xdg::XdgShell;
use smithay_client_toolkit::shell::WaylandSurface;
use smithay_client_toolkit::shm::slot::SlotPool;
use smithay_client_toolkit::shm::{Shm, ShmHandler};
use smithay_client_toolkit::{delegate_registry, registry_handlers};
use tokio::sync::{mpsc, watch};

pub(crate) const WIDTH: u32 = 64;
pub(crate) const HEIGHT: u32 = 48;
pub(crate) const FRAMERATE: u32 = 30;
pub(crate) const BYTES_PER_PIXEL: usize = 4;
pub(crate) const UPPER_COLOR_ARGB: u32 = 0xFF20_80C0;
pub(crate) const LOWER_COLOR_ARGB: u32 = 0xFFC0_4010;
pub(crate) const TIMEOUT: Duration = Duration::from_secs(10);

pub(crate) struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ClientEvent {
    Globals(Vec<String>),
    Configured { width: u32, height: u32 },
    KeyboardEnter,
    Key { code: u32, pressed: bool },
    PointerMotion { x: f64, y: f64 },
    Button { code: u32, pressed: bool },
}

struct Client {
    registry_state: RegistryState,
    seat_state: SeatState,
    output_state: OutputState,
    shm: Shm,
    pool: SlotPool,
    window: Window,
    drawn: bool,
    keyboard: Option<wl_keyboard::WlKeyboard>,
    pointer: Option<wl_pointer::WlPointer>,
    events: mpsc::UnboundedSender<ClientEvent>,
}

impl Client {
    fn draw(&mut self, width: u32, height: u32) {
        let stride = width as i32 * BYTES_PER_PIXEL as i32;
        let (buffer, canvas) = self
            .pool
            .create_buffer(
                width as i32,
                height as i32,
                stride,
                wl_shm::Format::Argb8888,
            )
            .expect("the pool holds one picture");
        let upper_bytes = (height / 2) as usize * stride as usize;
        let (upper, lower) = canvas.split_at_mut(upper_bytes);
        for (half, color) in [(upper, UPPER_COLOR_ARGB), (lower, LOWER_COLOR_ARGB)] {
            for pixel in half.as_chunks_mut::<BYTES_PER_PIXEL>().0 {
                *pixel = color.to_le_bytes();
            }
        }
        let surface = self.window.wl_surface();
        surface.damage_buffer(0, 0, width as i32, height as i32);
        buffer.attach_to(surface).expect("attaches");
        self.window.commit();
        self.drawn = true;
    }
}

pub(crate) async fn connect_client(
    socket_names: &mut watch::Receiver<String>,
) -> mpsc::UnboundedReceiver<ClientEvent> {
    let runtime_dir = PathBuf::from(
        std::env::var_os("XDG_RUNTIME_DIR").expect("XDG_RUNTIME_DIR names where sockets go"),
    );
    let socket_name = tokio::time::timeout(TIMEOUT, socket_names.wait_for(|name| !name.is_empty()))
        .await
        .expect("the display opens its socket")
        .expect("the source is alive")
        .clone();
    let (event_sender, events) = mpsc::unbounded_channel();
    let socket = runtime_dir.join(socket_name);
    std::thread::spawn(move || run_client(socket, event_sender));
    events
}

pub(crate) async fn client_globals(
    events: &mut mpsc::UnboundedReceiver<ClientEvent>,
) -> Vec<String> {
    let first = tokio::time::timeout(TIMEOUT, events.recv())
        .await
        .expect("the client connects")
        .expect("the client sends its globals first");
    let ClientEvent::Globals(interfaces) = first else {
        panic!("the first client event is the global list, got {first:?}");
    };
    interfaces
}

fn rgb(argb: u32) -> [u8; 3] {
    let [blue, green, red, _] = argb.to_le_bytes();
    [red, green, blue]
}

fn rgb_at(picture: &[u8], stride: usize, x: u32, y: u32) -> [u8; 3] {
    let offset = y as usize * stride + x as usize * BYTES_PER_PIXEL;
    let mut pixel = [0; 3];
    pixel.copy_from_slice(&picture[offset..offset + 3]);
    pixel
}

pub(crate) fn shows_client(picture: &[u8], stride: usize) -> bool {
    let x = WIDTH / 2;
    rgb_at(picture, stride, x, HEIGHT / 4) == rgb(UPPER_COLOR_ARGB)
        && rgb_at(picture, stride, x, HEIGHT * 3 / 4) == rgb(LOWER_COLOR_ARGB)
}

fn run_client(socket: PathBuf, events: mpsc::UnboundedSender<ClientEvent>) {
    let stream = UnixStream::connect(socket).expect("connects to the display socket");
    let connection = Connection::from_socket(stream).expect("wayland handshake");
    let (globals, mut queue) = registry_queue_init::<Client>(&connection).expect("registry");
    let qh = queue.handle();
    let interfaces = globals
        .contents()
        .with_list(|list| list.iter().map(|global| global.interface.clone()).collect());
    let _ = events.send(ClientEvent::Globals(interfaces));
    let compositor = CompositorState::bind(&globals, &qh).expect("wl_compositor");
    let xdg_shell = XdgShell::bind(&globals, &qh).expect("xdg_wm_base");
    let shm = Shm::bind(&globals, &qh).expect("wl_shm");
    let window =
        xdg_shell.create_window(compositor.create_surface(&qh), WindowDecorations::None, &qh);
    window.commit();
    let pool_bytes = WIDTH as usize * HEIGHT as usize * BYTES_PER_PIXEL;
    let mut client = Client {
        registry_state: RegistryState::new(&globals),
        seat_state: SeatState::new(&globals, &qh),
        output_state: OutputState::new(&globals, &qh),
        pool: SlotPool::new(pool_bytes, &shm).expect("shm pool"),
        shm,
        window,
        drawn: false,
        keyboard: None,
        pointer: None,
        events,
    };
    // the dispatch fails once the display closes the connection at the end of the test
    while queue.blocking_dispatch(&mut client).is_ok() {}
}

impl CompositorHandler for Client {
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
    fn frame(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: u32) {}
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

impl WindowHandler for Client {
    fn request_close(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &Window) {}

    fn configure(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &Window,
        configure: WindowConfigure,
        _: u32,
    ) {
        let width = configure
            .new_size
            .0
            .expect("the display sends a width")
            .get();
        let height = configure
            .new_size
            .1
            .expect("the display sends a height")
            .get();
        let _ = self.events.send(ClientEvent::Configured { width, height });
        if !self.drawn {
            self.draw(width, height);
        }
    }
}

impl SeatHandler for Client {
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
        if capability == Capability::Keyboard && self.keyboard.is_none() {
            self.keyboard = Some(
                self.seat_state
                    .get_keyboard(qh, &seat, None)
                    .expect("keyboard"),
            );
        }
        if capability == Capability::Pointer && self.pointer.is_none() {
            self.pointer = Some(self.seat_state.get_pointer(qh, &seat).expect("pointer"));
        }
    }
    fn remove_capability(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: wl_seat::WlSeat,
        _: Capability,
    ) {
    }
    fn remove_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {}
}

impl KeyboardHandler for Client {
    fn enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: &wl_surface::WlSurface,
        _: u32,
        _: &[u32],
        _: &[Keysym],
    ) {
        let _ = self.events.send(ClientEvent::KeyboardEnter);
    }
    fn leave(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: &wl_surface::WlSurface,
        _: u32,
    ) {
    }
    fn press_key(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_keyboard::WlKeyboard,
        _: u32,
        event: KeyEvent,
    ) {
        let _ = self.events.send(ClientEvent::Key {
            code: event.raw_code,
            pressed: true,
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
        let _ = self.events.send(ClientEvent::Key {
            code: event.raw_code,
            pressed: false,
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

impl PointerHandler for Client {
    fn pointer_frame(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_pointer::WlPointer,
        events: &[PointerEvent],
    ) {
        for event in events {
            let client_event = match event.kind {
                PointerEventKind::Enter { .. } | PointerEventKind::Motion { .. } => {
                    ClientEvent::PointerMotion {
                        x: event.position.0,
                        y: event.position.1,
                    }
                }
                PointerEventKind::Press { button, .. } => ClientEvent::Button {
                    code: button,
                    pressed: true,
                },
                PointerEventKind::Release { button, .. } => ClientEvent::Button {
                    code: button,
                    pressed: false,
                },
                PointerEventKind::Leave { .. } | PointerEventKind::Axis { .. } => continue,
            };
            let _ = self.events.send(client_event);
        }
    }
}

impl OutputHandler for Client {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }
    fn new_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn update_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
    fn output_destroyed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
}

impl ShmHandler for Client {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm
    }
}

delegate_registry!(Client);

impl ProvidesRegistryState for Client {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }
    registry_handlers![OutputState, SeatState];
}

smithay_client_toolkit::delegate_dispatch2!(Client);

pub(crate) async fn wait_for_client_event(
    events: &mut mpsc::UnboundedReceiver<ClientEvent>,
    expected: &ClientEvent,
) {
    let found = tokio::time::timeout(TIMEOUT, async {
        while let Some(event) = events.recv().await {
            if &event == expected {
                return true;
            }
        }
        false
    })
    .await;
    assert_eq!(found, Ok(true), "the client receives {expected:?}");
}
