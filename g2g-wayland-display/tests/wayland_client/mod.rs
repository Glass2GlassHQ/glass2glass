// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

use g2g_core::runtime::{run_graph, GraphNode};
use g2g_core::Graph;
use g2g_core::PipelineClock;
use g2g_plugins::appsink::{register_appsink_pull, AppSink, AppSinkPull};
use g2g_plugins::windowinput::{PointerButton, WindowInput};
use g2g_wayland_display::waylanddisplaysrc::DisplayNames;
use g2g_wayland_display::waylanddisplaysrc::{InputSender, WaylandDisplaySrc};
use smithay_client_toolkit::compositor::{CompositorHandler, CompositorState};
use smithay_client_toolkit::data_device_manager::data_device::{
    DataDevice, DataDeviceData, DataDeviceHandler,
};
use smithay_client_toolkit::data_device_manager::data_offer::{DataOfferHandler, DragOffer};
use smithay_client_toolkit::data_device_manager::data_source::{
    CopyPasteSource, DataSourceHandler, DragSource,
};
use smithay_client_toolkit::data_device_manager::{DataDeviceManagerState, WritePipe};
use smithay_client_toolkit::dispatch2::Dispatch2;
use smithay_client_toolkit::output::{OutputHandler, OutputState};
use smithay_client_toolkit::reexports::client::globals::registry_queue_init;
use smithay_client_toolkit::reexports::client::protocol::wl_data_device::WlDataDevice;
use smithay_client_toolkit::reexports::client::protocol::wl_data_device_manager::DndAction;
use smithay_client_toolkit::reexports::client::protocol::wl_data_source::WlDataSource;
use smithay_client_toolkit::reexports::client::protocol::{
    wl_keyboard, wl_output, wl_pointer, wl_seat, wl_shm, wl_surface, wl_touch,
};
use smithay_client_toolkit::reexports::client::{Connection, Proxy, QueueHandle, WEnum};
use smithay_client_toolkit::registry::{ProvidesRegistryState, RegistryState};
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

pub(crate) const WIDTH: u32 = 160;
pub(crate) const HEIGHT: u32 = 128;
pub(crate) const FRAMERATE: u32 = 30;
pub(crate) const BYTES_PER_PIXEL: usize = 4;
pub(crate) const UPPER_COLOR_ARGB: u32 = 0xFF20_80C0;
pub(crate) const LOWER_COLOR_ARGB: u32 = 0xFFC0_4010;
pub(crate) const TIMEOUT: Duration = Duration::from_secs(10);
// the client starts a drag when this button goes down over its window
const DRAG_BUTTON: PointerButton = PointerButton::Middle;
const DRAG_ICON_SIZE: u32 = 16;
const DRAG_ICON_COLOR_ARGB: u32 = 0xFF30_C050;
pub(crate) const DRAG_PAYLOAD: &str = "dragged text";
const TEXT_MIME_TYPE: &str = "text/plain;charset=utf-8";
// linux input-event-codes.h KEY_C, the client copies CLIPBOARD_TEXT when it goes down
pub(crate) const COPY_KEY: u32 = 46;
pub(crate) const CLIPBOARD_TEXT: &str = "copied text";
const DRAG_START: (u32, u32) = (20, 20);
// the client sets this cursor whenever the pointer enters its window
pub(crate) const CURSOR_SIZE: u32 = 8;
pub(crate) const CURSOR_COLOR_ARGB: u32 = 0xFF10_A0E0;
pub(crate) const CURSOR_HOTSPOT: (i32, i32) = (2, 3);
const DRAG_END: (u32, u32) = (60, 30);

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
    DropAccepted,
    Dropped(String),
    Clipboard(String),
    TouchDown { id: i32, x: f64, y: f64 },
    TouchMotion { id: i32, x: f64, y: f64 },
    TouchUp { id: i32 },
    TouchCancel,
}

struct Client {
    registry_state: RegistryState,
    seat_state: SeatState,
    output_state: OutputState,
    compositor: CompositorState,
    data_device_manager: DataDeviceManagerState,
    data_device: Option<DataDevice>,
    drag: Option<(DragSource, wl_surface::WlSurface)>,
    copy: Option<CopyPasteSource>,
    cursor: Option<wl_surface::WlSurface>,
    shm: Shm,
    pool: SlotPool,
    window: Window,
    drawn: bool,
    keyboard: Option<wl_keyboard::WlKeyboard>,
    pointer: Option<wl_pointer::WlPointer>,
    touch: Option<wl_touch::WlTouch>,
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

    fn set_cursor(&mut self, qh: &QueueHandle<Self>, serial: u32) {
        let surface = self
            .cursor
            .get_or_insert_with(|| self.compositor.create_surface(qh))
            .clone();
        let (hotspot_x, hotspot_y) = CURSOR_HOTSPOT;
        self.pointer
            .as_ref()
            .expect("the pointer entered")
            .set_cursor(serial, Some(&surface), hotspot_x, hotspot_y);
        let side = CURSOR_SIZE as i32;
        let stride = side * BYTES_PER_PIXEL as i32;
        let (buffer, canvas) = self
            .pool
            .create_buffer(side, side, stride, wl_shm::Format::Argb8888)
            .expect("the pool grows for the cursor");
        for pixel in canvas.as_chunks_mut::<BYTES_PER_PIXEL>().0 {
            *pixel = CURSOR_COLOR_ARGB.to_le_bytes();
        }
        surface.damage_buffer(0, 0, side, side);
        buffer.attach_to(&surface).expect("attaches");
        surface.commit();
    }

    fn copy(&mut self, qh: &QueueHandle<Self>, serial: u32) {
        let source = self
            .data_device_manager
            .create_copy_paste_source(qh, [TEXT_MIME_TYPE]);
        source.set_selection(
            self.data_device
                .as_ref()
                .expect("the seat has a data device"),
            serial,
        );
        self.copy = Some(source);
    }

    fn start_drag(&mut self, qh: &QueueHandle<Self>, serial: u32) {
        let data_device = self
            .data_device
            .as_ref()
            .expect("the seat has a data device");
        let source = self.data_device_manager.create_drag_and_drop_source(
            qh,
            [TEXT_MIME_TYPE],
            DndAction::Copy,
        );
        let icon = self.compositor.create_surface(qh);
        let side = DRAG_ICON_SIZE as i32;
        let stride = side * BYTES_PER_PIXEL as i32;
        let (buffer, canvas) = self
            .pool
            .create_buffer(side, side, stride, wl_shm::Format::Argb8888)
            .expect("the pool grows for the icon");
        for pixel in canvas.as_chunks_mut::<BYTES_PER_PIXEL>().0 {
            *pixel = DRAG_ICON_COLOR_ARGB.to_le_bytes();
        }
        icon.damage_buffer(0, 0, side, side);
        buffer.attach_to(&icon).expect("attaches");
        icon.commit();
        source.start_drag(data_device, self.window.wl_surface(), Some(&icon), serial);
        self.drag = Some((source, icon));
    }
}

pub(crate) async fn connect_client(
    display_names: &mut watch::Receiver<Option<DisplayNames>>,
) -> mpsc::UnboundedReceiver<ClientEvent> {
    let runtime_dir = PathBuf::from(
        std::env::var_os("XDG_RUNTIME_DIR").expect("XDG_RUNTIME_DIR names where sockets go"),
    );
    let socket_name = tokio::time::timeout(TIMEOUT, display_names.wait_for(Option::is_some))
        .await
        .expect("the display opens its socket")
        .expect("the source is alive")
        .as_ref()
        .expect("waited for the names")
        .wayland_socket
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

pub(crate) fn rgb(argb: u32) -> [u8; 3] {
    let [blue, green, red, _] = argb.to_le_bytes();
    [red, green, blue]
}

pub(crate) fn rgb_at(picture: &[u8], stride: usize, x: u32, y: u32) -> [u8; 3] {
    let offset = y as usize * stride + x as usize * BYTES_PER_PIXEL;
    let mut pixel = [0; 3];
    pixel.copy_from_slice(&picture[offset..offset + 3]);
    pixel
}

fn rgb_under_drag_icon(picture: &[u8], (pointer_x, pointer_y): (u32, u32)) -> [u8; 3] {
    let middle = DRAG_ICON_SIZE / 2;
    let stride = WIDTH as usize * BYTES_PER_PIXEL;
    rgb_at(picture, stride, pointer_x + middle, pointer_y + middle)
}

pub(crate) async fn wait_for_picture(pull: &AppSinkPull, shows: impl Fn(&[u8]) -> bool) -> bool {
    tokio::time::timeout(TIMEOUT, async {
        loop {
            let frame = pull.pull().await.expect("the source keeps producing");
            let picture = frame.domain.as_system_slice().expect("system memory");
            if shows(picture) {
                return;
            }
        }
    })
    .await
    .is_ok()
}

async fn wait_for_drag_icon(pull: &AppSinkPull, pointer: (u32, u32), argb: u32) -> bool {
    wait_for_picture(pull, |picture| {
        rgb_under_drag_icon(picture, pointer) == rgb(argb)
    })
    .await
}

fn send(input: &InputSender, event: WindowInput) {
    input.send(event).expect("the display takes input");
}

pub(crate) async fn drag_icon_to(pull: &AppSinkPull, input: &InputSender, end: (u32, u32)) {
    let (start_x, start_y) = DRAG_START;
    let (end_x, end_y) = end;
    send(
        input,
        WindowInput::Pointer {
            x: start_x,
            y: start_y,
        },
    );
    send(
        input,
        WindowInput::Button {
            button: DRAG_BUTTON,
            down: true,
        },
    );
    // the drag starts once the client answers the press, and only motion after that reaches a drop target
    assert!(
        wait_for_drag_icon(pull, DRAG_START, DRAG_ICON_COLOR_ARGB).await,
        "a frame shows the drag icon where the drag starts"
    );
    send(input, WindowInput::Pointer { x: end_x, y: end_y });
    assert!(
        wait_for_drag_icon(pull, end, DRAG_ICON_COLOR_ARGB).await,
        "the drag icon follows the pointer"
    );
}

pub(crate) fn release_drag(input: &InputSender) {
    send(
        input,
        WindowInput::Button {
            button: DRAG_BUTTON,
            down: false,
        },
    );
}

// the source must output system memory at WIDTH x HEIGHT
pub(crate) async fn drag_reaches_the_drop_target(source: WaylandDisplaySrc, channel: &str) {
    let input = source.input_sender();
    let mut display_names = source.display_names();
    let pull = register_appsink_pull(channel);
    let mut graph: Graph<GraphNode> = Graph::new();
    let source_node = graph.add_source(GraphNode::source(source));
    let sink_node = graph.add_sink(GraphNode::element(AppSink::new().with_channel(channel)));
    graph.link(source_node, sink_node).expect("links");
    let checks = async {
        let mut events = connect_client(&mut display_names).await;
        client_globals(&mut events).await;
        let stride = WIDTH as usize * BYTES_PER_PIXEL;
        assert!(
            wait_for_picture(&pull, |picture| shows_client(picture, stride)).await,
            "a frame shows the client's window"
        );
        drag_icon_to(&pull, &input, DRAG_END).await;
        // smithay cancels a drop the target has not accepted yet
        wait_for_client_event(&mut events, &ClientEvent::DropAccepted).await;
        release_drag(&input);
        wait_for_client_event(&mut events, &ClientEvent::Dropped(DRAG_PAYLOAD.into())).await;
        assert!(
            wait_for_drag_icon(&pull, DRAG_END, UPPER_COLOR_ARGB).await,
            "the drag icon goes away after the drop"
        );
    };
    tokio::select! {
        result = run_graph(graph, &ZeroClock, 2) => panic!("the graph ended first: {result:?}"),
        () = checks => {}
    }
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
    let data_device_manager =
        DataDeviceManagerState::bind(&globals, &qh).expect("wl_data_device_manager");
    let window =
        xdg_shell.create_window(compositor.create_surface(&qh), WindowDecorations::None, &qh);
    window.commit();
    let pool_bytes = WIDTH as usize * HEIGHT as usize * BYTES_PER_PIXEL;
    let mut client = Client {
        registry_state: RegistryState::new(&globals),
        seat_state: SeatState::new(&globals, &qh),
        output_state: OutputState::new(&globals, &qh),
        compositor,
        data_device_manager,
        data_device: None,
        drag: None,
        copy: None,
        cursor: None,
        pool: SlotPool::new(pool_bytes, &shm).expect("shm pool"),
        shm,
        window,
        drawn: false,
        keyboard: None,
        pointer: None,
        touch: None,
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
            self.keyboard = Some(seat.get_keyboard(qh, KeyboardEvents));
        }
        if capability == Capability::Pointer && self.pointer.is_none() {
            self.pointer = Some(self.seat_state.get_pointer(qh, &seat).expect("pointer"));
            self.data_device = Some(self.data_device_manager.get_data_device(qh, &seat));
        }
        if capability == Capability::Touch && self.touch.is_none() {
            self.touch = Some(seat.get_touch(qh, TouchEvents));
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

struct KeyboardEvents;

impl Dispatch2<wl_keyboard::WlKeyboard, Client> for KeyboardEvents {
    fn event(
        &self,
        client: &mut Client,
        _: &wl_keyboard::WlKeyboard,
        event: wl_keyboard::Event,
        _: &Connection,
        qh: &QueueHandle<Client>,
    ) {
        let event = match event {
            wl_keyboard::Event::Enter { .. } => ClientEvent::KeyboardEnter,
            wl_keyboard::Event::Key {
                key, state, serial, ..
            } => {
                let pressed = state == WEnum::Value(wl_keyboard::KeyState::Pressed);
                if key == COPY_KEY && pressed {
                    client.copy(qh, serial);
                }
                ClientEvent::Key { code: key, pressed }
            }
            _ => return,
        };
        let _ = client.events.send(event);
    }
}

struct TouchEvents;

impl Dispatch2<wl_touch::WlTouch, Client> for TouchEvents {
    fn event(
        &self,
        client: &mut Client,
        _: &wl_touch::WlTouch,
        event: wl_touch::Event,
        _: &Connection,
        _: &QueueHandle<Client>,
    ) {
        let event = match event {
            wl_touch::Event::Down { id, x, y, .. } => ClientEvent::TouchDown { id, x, y },
            wl_touch::Event::Motion { id, x, y, .. } => ClientEvent::TouchMotion { id, x, y },
            wl_touch::Event::Up { id, .. } => ClientEvent::TouchUp { id },
            wl_touch::Event::Cancel => ClientEvent::TouchCancel,
            _ => return,
        };
        let _ = client.events.send(event);
    }
}

impl PointerHandler for Client {
    fn pointer_frame(
        &mut self,
        _: &Connection,
        qh: &QueueHandle<Self>,
        _: &wl_pointer::WlPointer,
        events: &[PointerEvent],
    ) {
        for event in events {
            if let PointerEventKind::Enter { serial } = event.kind {
                self.set_cursor(qh, serial);
            }
            let client_event = match event.kind {
                PointerEventKind::Enter { .. } | PointerEventKind::Motion { .. } => {
                    ClientEvent::PointerMotion {
                        x: event.position.0,
                        y: event.position.1,
                    }
                }
                PointerEventKind::Press { button, serial, .. } => {
                    if button == DRAG_BUTTON.evdev_code() {
                        self.start_drag(qh, serial);
                    }
                    ClientEvent::Button {
                        code: button,
                        pressed: true,
                    }
                }
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

impl DataDeviceHandler for Client {
    fn enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        data_device: &WlDataDevice,
        _: f64,
        _: f64,
        _: &wl_surface::WlSurface,
    ) {
        let offer = drag_offer(data_device);
        offer.accept_mime_type(offer.serial, Some(String::from(TEXT_MIME_TYPE)));
        offer.set_actions(DndAction::Copy, DndAction::Copy);
    }
    fn leave(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataDevice) {}
    fn motion(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataDevice, _: f64, _: f64) {}
    fn selection(&mut self, _: &Connection, _: &QueueHandle<Self>, data_device: &WlDataDevice) {
        let Some(offer) = data_device
            .data::<DataDeviceData>()
            .and_then(DataDeviceData::selection_offer)
        else {
            return;
        };
        if !offer.with_mime_types(|mime_types| mime_types.iter().any(|mime| mime == TEXT_MIME_TYPE))
        {
            return;
        }
        let mut pipe = offer
            .receive(String::from(TEXT_MIME_TYPE))
            .expect("the selection offers text");
        let events = self.events.clone();
        // this thread also serves the selection when this client copied it
        std::thread::spawn(move || {
            let mut text = String::new();
            pipe.read_to_string(&mut text).expect("reads the clipboard");
            let _ = events.send(ClientEvent::Clipboard(text));
        });
    }
    fn drop_performed(
        &mut self,
        connection: &Connection,
        _: &QueueHandle<Self>,
        data_device: &WlDataDevice,
    ) {
        let offer = drag_offer(data_device);
        let mut pipe = offer
            .receive(String::from(TEXT_MIME_TYPE))
            .expect("the dropped offer has the mime type");
        let connection = connection.clone();
        let events = self.events.clone();
        // this thread also serves the drag source, which writes the payload
        std::thread::spawn(move || {
            let mut payload = String::new();
            pipe.read_to_string(&mut payload)
                .expect("reads the payload");
            // smithay drops an X11 source's transfer when the target finishes before the X11 drop arrives
            offer.finish();
            let _ = connection.flush();
            let _ = events.send(ClientEvent::Dropped(payload));
        });
    }
}

fn drag_offer(data_device: &WlDataDevice) -> DragOffer {
    data_device
        .data::<DataDeviceData>()
        .and_then(DataDeviceData::drag_offer)
        .expect("a drag is over the window")
}

impl DataOfferHandler for Client {
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
        action: DndAction,
    ) {
        if action == DndAction::Copy {
            let _ = self.events.send(ClientEvent::DropAccepted);
        }
    }
}

impl DataSourceHandler for Client {
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
        let copied = self
            .copy
            .as_ref()
            .is_some_and(|copy| copy.inner() == source);
        let payload = if copied { CLIPBOARD_TEXT } else { DRAG_PAYLOAD };
        pipe.write_all(payload.as_bytes())
            .expect("writes the payload");
    }
    fn cancelled(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataSource) {
        self.drag = None;
    }
    fn dnd_dropped(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataSource) {}
    fn dnd_finished(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataSource) {
        self.drag = None;
    }
    fn action(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &WlDataSource, _: DndAction) {}
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
