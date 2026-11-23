// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "linux", feature = "display"))]

use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

use g2g_core::runtime::{parse_launch, run_graph, GraphNode};
use g2g_core::{Graph, PipelineClock};
use g2g_plugins::appsink::{register_appsink_pull, AppSink};
use g2g_plugins::registry::default_registry;
use g2g_plugins::windowinput::{PointerButton, WindowInput};
use g2g_wayland_display::waylanddisplaysrc::WaylandDisplaySrc;
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
use tokio::sync::mpsc;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;
const FRAMERATE: u32 = 30;
const BYTES_PER_PIXEL: usize = 4;
const CLIENT_COLOR_ARGB: u32 = 0xFF20_80C0;
const TIMEOUT: Duration = Duration::from_secs(10);
// gtk 4 refuses to open a display that lacks this interface
const GTK_REQUIRED_INTERFACE: &str = "wl_data_device_manager";
// linux input-event-codes.h KEY_A
const KEY_A: u32 = 30;
const POINTER_X: u32 = 10;
const POINTER_Y: u32 = 20;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[derive(Debug, Clone, PartialEq)]
enum ClientEvent {
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
        for pixel in canvas.as_chunks_mut::<BYTES_PER_PIXEL>().0 {
            *pixel = CLIENT_COLOR_ARGB.to_le_bytes();
        }
        let surface = self.window.wl_surface();
        surface.damage_buffer(0, 0, width as i32, height as i32);
        buffer.attach_to(surface).expect("attaches");
        self.window.commit();
        self.drawn = true;
    }
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

async fn wait_for_client_event(
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

fn centre_pixel(picture: &[u8]) -> &[u8] {
    let row_bytes = WIDTH as usize * BYTES_PER_PIXEL;
    let offset = HEIGHT as usize / 2 * row_bytes + WIDTH as usize / 2 * BYTES_PER_PIXEL;
    &picture[offset..offset + BYTES_PER_PIXEL]
}

#[tokio::test]
async fn a_client_window_reaches_the_frames_and_input_reaches_the_client() {
    let runtime_dir = PathBuf::from(
        std::env::var_os("XDG_RUNTIME_DIR").expect("XDG_RUNTIME_DIR names where sockets go"),
    );
    let source = WaylandDisplaySrc::new()
        .with_size(WIDTH, HEIGHT)
        .with_framerate(FRAMERATE);
    let input = source.input_sender();
    let mut socket_names = source.socket_names();
    let pull = register_appsink_pull("m2534_frames");
    let mut graph: Graph<GraphNode> = Graph::new();
    let source_node = graph.add_source(GraphNode::source(source));
    let sink_node = graph.add_sink(GraphNode::element(
        AppSink::new().with_channel("m2534_frames"),
    ));
    graph.link(source_node, sink_node).expect("links");
    let checks = async {
        let socket_name =
            tokio::time::timeout(TIMEOUT, socket_names.wait_for(|name| !name.is_empty()))
                .await
                .expect("the display opens its socket")
                .expect("the source is alive")
                .clone();
        let (event_sender, mut events) = mpsc::unbounded_channel();
        let socket = runtime_dir.join(socket_name);
        std::thread::spawn(move || run_client(socket, event_sender));

        let interfaces = tokio::time::timeout(TIMEOUT, events.recv())
            .await
            .expect("the client connects")
            .expect("the client sends its globals first");
        let ClientEvent::Globals(interfaces) = interfaces else {
            panic!("the first client event is the global list, got {interfaces:?}");
        };
        assert!(
            interfaces.iter().any(|name| name == GTK_REQUIRED_INTERFACE),
            "the display offers {GTK_REQUIRED_INTERFACE}: {interfaces:?}"
        );

        // BGRx bytes of the client's ARGB colour
        let [blue, green, red, _] = CLIENT_COLOR_ARGB.to_le_bytes();
        let shows_client = tokio::time::timeout(TIMEOUT, async {
            loop {
                let frame = pull.pull().await.expect("the source keeps producing");
                let picture = frame.domain.as_system_slice().expect("system memory");
                assert_eq!(
                    picture.len(),
                    WIDTH as usize * HEIGHT as usize * BYTES_PER_PIXEL
                );
                if centre_pixel(picture)[..3] == [blue, green, red] {
                    return;
                }
            }
        })
        .await;
        assert!(shows_client.is_ok(), "a frame shows the client's window");

        wait_for_client_event(
            &mut events,
            &ClientEvent::Configured {
                width: WIDTH,
                height: HEIGHT,
            },
        )
        .await;
        wait_for_client_event(&mut events, &ClientEvent::KeyboardEnter).await;
        for event in [
            WindowInput::Pointer {
                x: POINTER_X,
                y: POINTER_Y,
            },
            WindowInput::Button {
                button: PointerButton::Left,
                down: true,
            },
            WindowInput::Key {
                scancode: KEY_A,
                down: true,
            },
        ] {
            input.send(event).expect("the display takes input");
        }
        wait_for_client_event(
            &mut events,
            &ClientEvent::PointerMotion {
                x: f64::from(POINTER_X),
                y: f64::from(POINTER_Y),
            },
        )
        .await;
        wait_for_client_event(
            &mut events,
            &ClientEvent::Button {
                code: PointerButton::Left.evdev_code(),
                pressed: true,
            },
        )
        .await;
        wait_for_client_event(
            &mut events,
            &ClientEvent::Key {
                code: KEY_A,
                pressed: true,
            },
        )
        .await;

        input
            .send(WindowInput::Focus(false))
            .expect("the display takes input");
        wait_for_client_event(
            &mut events,
            &ClientEvent::Key {
                code: KEY_A,
                pressed: false,
            },
        )
        .await;
        wait_for_client_event(
            &mut events,
            &ClientEvent::Button {
                code: PointerButton::Left.evdev_code(),
                pressed: false,
            },
        )
        .await;
    };
    tokio::select! {
        result = run_graph(graph, &ZeroClock, 2) => panic!("the graph ended first: {result:?}"),
        () = checks => {}
    }
}

#[tokio::test]
async fn a_launch_line_sets_the_size_and_frame_count() {
    const LAUNCH_WIDTH: u32 = 32;
    const LAUNCH_HEIGHT: u32 = 24;
    let pull = register_appsink_pull("m2534_launch_frames");
    let mut registry = default_registry();
    g2g_wayland_display::register(&mut registry);
    let graph = parse_launch(
        &registry,
        &format!(
            "waylanddisplaysrc width={LAUNCH_WIDTH} height={LAUNCH_HEIGHT} framerate={FRAMERATE} \
             num-buffers=1 ! appsink channel=m2534_launch_frames"
        ),
    )
    .expect("parses");
    let stats = tokio::time::timeout(TIMEOUT, run_graph(graph, &ZeroClock, 2))
        .await
        .expect("one frame ends the run")
        .expect("runs");
    assert_eq!(stats.frames_emitted, 1);
    let frame = pull.pull().await.expect("the frame reaches the sink");
    let picture = frame.domain.as_system_slice().expect("system memory");
    assert_eq!(
        picture.len(),
        LAUNCH_WIDTH as usize * LAUNCH_HEIGHT as usize * BYTES_PER_PIXEL
    );
    assert!(
        pull.pull().await.is_none(),
        "the stream ends after one frame"
    );
}
