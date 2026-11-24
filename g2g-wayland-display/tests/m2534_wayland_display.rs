// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "linux", feature = "display"))]

mod wayland_client;

use g2g_core::runtime::{parse_launch, run_graph, GraphNode};
use g2g_core::Graph;
use g2g_plugins::appsink::{register_appsink_pull, AppSink};
use g2g_plugins::cursorshape::CursorShape;
use g2g_plugins::registry::default_registry;
use g2g_plugins::windowinput::{PointerButton, WindowInput};
use g2g_wayland_display::waylanddisplaysrc::{
    WaylandDisplay, WaylandDisplaySettings, WaylandDisplaySrc, SOFTWARE_RENDERING,
};
use tokio::sync::mpsc;
use wayland_client::{
    client_globals, connect_client, drag_icon_to, drag_reaches_the_drop_target, release_drag, rgb,
    rgb_at, shows_client, wait_for_client_event, wait_for_picture, ClientEvent, ZeroClock,
    BYTES_PER_PIXEL, CLIPBOARD_TEXT, COPY_KEY, CURSOR_COLOR_ARGB, CURSOR_HOTSPOT, CURSOR_SIZE,
    DRAG_PAYLOAD, FRAMERATE, HEIGHT, TIMEOUT, WIDTH,
};
use x11rb::connection::Connection;
use x11rb::protocol::xproto::{
    Atom, AtomEnum, ClientMessageEvent, ConnectionExt, CreateWindowAux, EventMask, PropMode,
    SelectionNotifyEvent, SelectionRequestEvent, Window, WindowClass, SELECTION_NOTIFY_EVENT,
};
use x11rb::protocol::Event;
use x11rb::rust_connection::RustConnection;
use x11rb::wrapper::ConnectionExt as _;
use x11rb::COPY_DEPTH_FROM_PARENT;

// gtk 4 refuses to open a display that lacks this interface
const GTK_REQUIRED_INTERFACE: &str = "wl_data_device_manager";
// linux input-event-codes.h KEY_A
const KEY_A: u32 = 30;
const POINTER_X: u32 = 10;
const POINTER_Y: u32 = 20;
// the X11 drop target covers the right of the screen, the Wayland client stays visible on the left
const X11_TARGET_LEFT: u32 = 100;
const X11_TARGET_RGB: [u8; 3] = [0x50, 0x10, 0x90];
const X11_DROP_POINT: (u32, u32) = (120, 30);
const WAYLAND_DRAG_POINT: (u32, u32) = (40, 30);
const XDND_VERSION: u32 = 5;
const XDND_STATUS_ACCEPTS: u32 = 1;
const XDND_FINISHED_ACCEPTED: u32 = 1;
const PROPERTY_WORDS: u32 = 1024;

#[tokio::test]
async fn a_client_window_reaches_the_frames_and_input_reaches_the_client() {
    let source = WaylandDisplaySrc::new()
        .with_size(WIDTH, HEIGHT)
        .with_framerate(FRAMERATE)
        .with_render_node(SOFTWARE_RENDERING);
    let input = source.input_sender();
    let mut display_names = source.display_names();
    let pull = register_appsink_pull("m2534_frames");
    let mut graph: Graph<GraphNode> = Graph::new();
    let source_node = graph.add_source(GraphNode::source(source));
    let sink_node = graph.add_sink(GraphNode::element(
        AppSink::new().with_channel("m2534_frames"),
    ));
    graph.link(source_node, sink_node).expect("links");
    let checks = async {
        let mut events = connect_client(&mut display_names).await;
        let interfaces = client_globals(&mut events).await;
        assert!(
            interfaces.iter().any(|name| name == GTK_REQUIRED_INTERFACE),
            "the display offers {GTK_REQUIRED_INTERFACE}: {interfaces:?}"
        );

        let client_shown = tokio::time::timeout(TIMEOUT, async {
            loop {
                let frame = pull.pull().await.expect("the source keeps producing");
                let picture = frame.domain.as_system_slice().expect("system memory");
                assert_eq!(
                    picture.len(),
                    WIDTH as usize * HEIGHT as usize * BYTES_PER_PIXEL
                );
                if shows_client(picture, WIDTH as usize * BYTES_PER_PIXEL) {
                    return;
                }
            }
        })
        .await;
        assert!(client_shown.is_ok(), "a frame shows the client's window");

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
             render-node={SOFTWARE_RENDERING} num-buffers=1 ! appsink channel=m2534_launch_frames"
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

#[tokio::test]
async fn a_drag_shows_its_icon_and_drops_its_data() {
    let source = WaylandDisplaySrc::new()
        .with_size(WIDTH, HEIGHT)
        .with_framerate(FRAMERATE)
        .with_render_node(SOFTWARE_RENDERING);
    drag_reaches_the_drop_target(source, "m2541_software_drag").await;
}

#[derive(Debug, PartialEq)]
enum XdndEvent {
    Position,
    Dropped(String),
    Refused,
}

struct XdndAtoms {
    aware: Atom,
    enter: Atom,
    position: Atom,
    status: Atom,
    drop: Atom,
    finished: Atom,
    selection: Atom,
    action_copy: Atom,
    dropped_data: Atom,
}

fn intern(connection: &RustConnection, name: &str) -> Atom {
    connection
        .intern_atom(false, name.as_bytes())
        .expect("sends the request")
        .reply()
        .expect("interns the atom")
        .atom
}

fn send_client_message(
    connection: &RustConnection,
    to: Window,
    message_type: Atom,
    data: [u32; 5],
) {
    connection
        .send_event(
            false,
            to,
            EventMask::NO_EVENT,
            ClientMessageEvent::new(32, to, message_type, data),
        )
        .expect("sends the message");
    connection.flush().expect("flushes");
}

impl XdndAtoms {
    fn intern(connection: &RustConnection) -> Self {
        Self {
            aware: intern(connection, "XdndAware"),
            enter: intern(connection, "XdndEnter"),
            position: intern(connection, "XdndPosition"),
            status: intern(connection, "XdndStatus"),
            drop: intern(connection, "XdndDrop"),
            finished: intern(connection, "XdndFinished"),
            selection: intern(connection, "XdndSelection"),
            action_copy: intern(connection, "XdndActionCopy"),
            dropped_data: intern(connection, "G2G_DROPPED_DATA"),
        }
    }
}

// an override-redirect window over the right of the screen, as X11 menus are
fn map_menu_window(
    connection: &RustConnection,
    screen_number: usize,
    event_mask: EventMask,
) -> Window {
    let screen = &connection.setup().roots[screen_number];
    let [red, green, blue] = X11_TARGET_RGB;
    let window = connection.generate_id().expect("allocates a window id");
    connection
        .create_window(
            COPY_DEPTH_FROM_PARENT,
            window,
            screen.root,
            X11_TARGET_LEFT as i16,
            0,
            (WIDTH - X11_TARGET_LEFT) as u16,
            HEIGHT as u16,
            0,
            WindowClass::INPUT_OUTPUT,
            screen.root_visual,
            &CreateWindowAux::new()
                .background_pixel(u32::from_be_bytes([0, red, green, blue]))
                .override_redirect(1)
                .event_mask(event_mask),
        )
        .expect("creates a window");
    connection.map_window(window).expect("maps the window");
    connection.flush().expect("flushes");
    window
}

fn run_xdnd_target(display: String, events: mpsc::UnboundedSender<XdndEvent>) {
    let (connection, screen_number) = x11rb::connect(Some(&display)).expect("connects to Xwayland");
    let atoms = XdndAtoms::intern(&connection);
    let window = map_menu_window(&connection, screen_number, EventMask::NO_EVENT);
    connection
        .change_property32(
            PropMode::REPLACE,
            window,
            atoms.aware,
            AtomEnum::ATOM,
            &[XDND_VERSION],
        )
        .expect("marks the window as a drop target");
    connection.flush().expect("flushes");
    let mut offered_type = None;
    let mut source = None;
    while let Ok(event) = connection.wait_for_event() {
        let event = match event {
            Event::ClientMessage(message) => {
                let data = message.data.as_data32();
                if message.type_ == atoms.enter {
                    offered_type = Some(data[2]);
                    continue;
                }
                if message.type_ == atoms.position {
                    source = Some(data[0]);
                    let status = [window, XDND_STATUS_ACCEPTS, 0, 0, atoms.action_copy];
                    send_client_message(&connection, data[0], atoms.status, status);
                    XdndEvent::Position
                } else if message.type_ == atoms.drop {
                    let target = offered_type.expect("XdndEnter names the offered type");
                    connection
                        .convert_selection(
                            window,
                            atoms.selection,
                            target,
                            atoms.dropped_data,
                            data[2],
                        )
                        .expect("asks for the dropped data");
                    connection.flush().expect("flushes");
                    continue;
                } else {
                    continue;
                }
            }
            Event::SelectionNotify(notify) if notify.property == x11rb::NONE => XdndEvent::Refused,
            Event::SelectionNotify(_) => {
                let property = connection
                    .get_property(
                        true,
                        window,
                        atoms.dropped_data,
                        AtomEnum::ANY,
                        0,
                        PROPERTY_WORDS,
                    )
                    .expect("sends the request")
                    .reply()
                    .expect("reads the dropped data");
                let source = source.expect("XdndPosition names the drag source");
                let finished = [window, XDND_FINISHED_ACCEPTED, atoms.action_copy, 0, 0];
                send_client_message(&connection, source, atoms.finished, finished);
                XdndEvent::Dropped(String::from_utf8(property.value).expect("utf-8 text"))
            }
            _ => continue,
        };
        if events.send(event).is_err() {
            return;
        }
    }
}

async fn wait_for_xdnd_event(events: &mut mpsc::UnboundedReceiver<XdndEvent>, expected: XdndEvent) {
    let received = tokio::time::timeout(TIMEOUT, events.recv())
        .await
        .expect("the X11 client answers in time");
    assert_eq!(received, Some(expected));
}

#[tokio::test]
async fn a_wayland_drag_drops_onto_an_x11_window() {
    let source = WaylandDisplaySrc::new()
        .with_size(WIDTH, HEIGHT)
        .with_framerate(FRAMERATE)
        .with_render_node(SOFTWARE_RENDERING)
        .with_xwayland();
    let input = source.input_sender();
    let mut display_names = source.display_names();
    let pull = register_appsink_pull("m2557_x11_drop");
    let mut graph: Graph<GraphNode> = Graph::new();
    let source_node = graph.add_source(GraphNode::source(source));
    let sink_node = graph.add_sink(GraphNode::element(
        AppSink::new().with_channel("m2557_x11_drop"),
    ));
    graph.link(source_node, sink_node).expect("links");
    let checks = async {
        let mut events = connect_client(&mut display_names).await;
        client_globals(&mut events).await;
        let stride = WIDTH as usize * BYTES_PER_PIXEL;
        assert!(
            wait_for_picture(&pull, |picture| shows_client(picture, stride)).await,
            "a frame shows the Wayland client's window"
        );
        // a press raises the window under it, so the X11 window maps on top once the drag holds the pointer
        drag_icon_to(&pull, &input, WAYLAND_DRAG_POINT).await;
        let x11_display = display_names
            .borrow()
            .as_ref()
            .and_then(|names| names.x11_display.clone())
            .expect("the source runs Xwayland");
        let (xdnd_sender, mut xdnd_events) = mpsc::unbounded_channel();
        std::thread::spawn(move || run_xdnd_target(x11_display, xdnd_sender));
        let (drop_x, drop_y) = X11_DROP_POINT;
        assert!(
            wait_for_picture(&pull, |picture| {
                rgb_at(picture, stride, drop_x, drop_y) == X11_TARGET_RGB
            })
            .await,
            "a frame shows the X11 window"
        );
        input
            .send(WindowInput::Pointer {
                x: drop_x,
                y: drop_y,
            })
            .expect("the display takes input");
        wait_for_xdnd_event(&mut xdnd_events, XdndEvent::Position).await;
        // smithay sends the next XdndPosition only after it has read the accepting XdndStatus
        input
            .send(WindowInput::Pointer {
                x: drop_x + 1,
                y: drop_y,
            })
            .expect("the display takes input");
        wait_for_xdnd_event(&mut xdnd_events, XdndEvent::Position).await;
        release_drag(&input);
        wait_for_xdnd_event(&mut xdnd_events, XdndEvent::Dropped(DRAG_PAYLOAD.into())).await;
    };
    tokio::select! {
        result = run_graph(graph, &ZeroClock, 2) => panic!("the graph ended first: {result:?}"),
        () = checks => {}
    }
}

async fn start_software_display() -> WaylandDisplay {
    WaylandDisplay::start(WaylandDisplaySettings {
        width: WIDTH,
        height: HEIGHT,
        framerate_q16: FRAMERATE << 16,
        render_node: String::from(SOFTWARE_RENDERING),
        xwayland: false,
    })
    .await
    .expect("the display starts")
}

fn session_graph(display: &WaylandDisplay, channel: &str) -> Graph<GraphNode> {
    let mut graph: Graph<GraphNode> = Graph::new();
    let source_node = graph.add_source(GraphNode::source(WaylandDisplaySrc::on_display(display)));
    let sink_node = graph.add_sink(GraphNode::element(AppSink::new().with_channel(channel)));
    graph.link(source_node, sink_node).expect("links");
    graph
}

#[tokio::test]
async fn a_shared_display_keeps_its_client_across_runs_and_releases_held_keys() {
    let display = start_software_display().await;
    let input = display.input_sender();
    let mut display_names = WaylandDisplaySrc::on_display(&display).display_names();
    let stride = WIDTH as usize * BYTES_PER_PIXEL;
    let first_frames = register_appsink_pull("m2558_first_run");
    let mut events = connect_client(&mut display_names).await;
    client_globals(&mut events).await;
    let first_run = async {
        assert!(
            wait_for_picture(&first_frames, |picture| shows_client(picture, stride)).await,
            "a frame of the first run shows the client's window"
        );
        wait_for_client_event(&mut events, &ClientEvent::KeyboardEnter).await;
        input
            .send(WindowInput::Key {
                scancode: KEY_A,
                down: true,
            })
            .expect("the display takes input");
        wait_for_client_event(
            &mut events,
            &ClientEvent::Key {
                code: KEY_A,
                pressed: true,
            },
        )
        .await;
    };
    tokio::select! {
        result = run_graph(session_graph(&display, "m2558_first_run"), &ZeroClock, 2) => {
            panic!("the first run ended first: {result:?}")
        }
        () = first_run => {}
    }
    wait_for_client_event(
        &mut events,
        &ClientEvent::Key {
            code: KEY_A,
            pressed: false,
        },
    )
    .await;

    let second_frames = register_appsink_pull("m2558_second_run");
    let second_run = async {
        assert!(
            wait_for_picture(&second_frames, |picture| shows_client(picture, stride)).await,
            "a frame of the second run shows the window the client drew during the first"
        );
    };
    tokio::select! {
        result = run_graph(session_graph(&display, "m2558_second_run"), &ZeroClock, 2) => {
            panic!("the second run ended first: {result:?}")
        }
        () = second_run => {}
    }
}

#[tokio::test]
async fn a_shared_display_publishes_the_cursor_its_client_sets() {
    let display = start_software_display().await;
    let input = display.input_sender();
    let mut cursor_shapes = display.cursor_shapes();
    let mut display_names = WaylandDisplaySrc::on_display(&display).display_names();
    let frames = register_appsink_pull("m2558_cursor");
    let stride = WIDTH as usize * BYTES_PER_PIXEL;
    let mut events = connect_client(&mut display_names).await;
    client_globals(&mut events).await;
    let [red, green, blue] = rgb(CURSOR_COLOR_ARGB);
    let (hotspot_x, hotspot_y) = CURSOR_HOTSPOT;
    let expected = CursorShape {
        width: CURSOR_SIZE,
        height: CURSOR_SIZE,
        hotspot_x,
        hotspot_y,
        rgba: [red, green, blue, u8::MAX].repeat((CURSOR_SIZE * CURSOR_SIZE) as usize),
    };
    let checks = async {
        assert!(
            wait_for_picture(&frames, |picture| shows_client(picture, stride)).await,
            "a frame shows the client's window"
        );
        input
            .send(WindowInput::Pointer {
                x: POINTER_X,
                y: POINTER_Y,
            })
            .expect("the display takes input");
        let published = tokio::time::timeout(
            TIMEOUT,
            cursor_shapes.wait_for(|shape| shape.as_ref() == Some(&expected)),
        )
        .await;
        assert!(
            matches!(published, Ok(Ok(_))),
            "the display publishes the client's cursor"
        );
    };
    tokio::select! {
        result = run_graph(session_graph(&display, "m2558_cursor"), &ZeroClock, 2) => {
            panic!("the run ended first: {result:?}")
        }
        () = checks => {}
    }
}

#[tokio::test]
async fn a_press_on_a_window_leaves_an_x11_menu_above_it() {
    const FRAMES_AFTER_PRESS: usize = 2;
    let source = WaylandDisplaySrc::new()
        .with_size(WIDTH, HEIGHT)
        .with_framerate(FRAMERATE)
        .with_render_node(SOFTWARE_RENDERING)
        .with_xwayland();
    let input = source.input_sender();
    let mut display_names = source.display_names();
    let pull = register_appsink_pull("m2561_menu_stacking");
    let mut graph: Graph<GraphNode> = Graph::new();
    let source_node = graph.add_source(GraphNode::source(source));
    let sink_node = graph.add_sink(GraphNode::element(
        AppSink::new().with_channel("m2561_menu_stacking"),
    ));
    graph.link(source_node, sink_node).expect("links");
    let checks = async {
        let mut events = connect_client(&mut display_names).await;
        client_globals(&mut events).await;
        let stride = WIDTH as usize * BYTES_PER_PIXEL;
        assert!(
            wait_for_picture(&pull, |picture| shows_client(picture, stride)).await,
            "a frame shows the Wayland client's window"
        );
        let x11_display = display_names
            .borrow()
            .as_ref()
            .and_then(|names| names.x11_display.clone())
            .expect("the source runs Xwayland");
        let (xdnd_sender, _xdnd_events) = mpsc::unbounded_channel();
        std::thread::spawn(move || run_xdnd_target(x11_display, xdnd_sender));
        let (menu_x, menu_y) = X11_DROP_POINT;
        let shows_menu = |picture: &[u8]| rgb_at(picture, stride, menu_x, menu_y) == X11_TARGET_RGB;
        assert!(
            wait_for_picture(&pull, shows_menu).await,
            "a frame shows the X11 menu"
        );

        input
            .send(WindowInput::Pointer {
                x: POINTER_X,
                y: POINTER_Y,
            })
            .expect("the display takes input");
        for down in [true, false] {
            input
                .send(WindowInput::Button {
                    button: PointerButton::Left,
                    down,
                })
                .expect("the display takes input");
        }
        wait_for_client_event(
            &mut events,
            &ClientEvent::Button {
                code: PointerButton::Left.evdev_code(),
                pressed: false,
            },
        )
        .await;
        for _ in 0..FRAMES_AFTER_PRESS {
            let frame = pull.pull().await.expect("the source keeps producing");
            let picture = frame.domain.as_system_slice().expect("system memory");
            assert!(
                shows_menu(picture),
                "the menu stays above the pressed window"
            );
        }
    };
    tokio::select! {
        result = run_graph(graph, &ZeroClock, 2) => panic!("the graph ended first: {result:?}"),
        () = checks => {}
    }
}

#[derive(Debug, PartialEq)]
enum XdndSourceEvent {
    Pressed,
    Entered,
}

// XDND sources send to the top-level window under the pointer, which during a drag is smithay's proxy
fn window_under(connection: &RustConnection, root: Window, x: i16, y: i16) -> Option<Window> {
    let child = connection
        .translate_coordinates(root, root, x, y)
        .expect("sends the request")
        .reply()
        .expect("finds the window under the pointer")
        .child;
    (child != x11rb::NONE).then_some(child)
}

// answers an X11 selection request with text, and with the targets list when asked for it
fn serve_text(connection: &RustConnection, request: &SelectionRequestEvent, text: &str) {
    let targets = intern(connection, "TARGETS");
    let utf8_string = intern(connection, "UTF8_STRING");
    let property = if request.target == targets {
        connection
            .change_property32(
                PropMode::REPLACE,
                request.requestor,
                request.property,
                AtomEnum::ATOM,
                &[targets, utf8_string],
            )
            .expect("writes the targets");
        request.property
    } else if request.target == utf8_string {
        connection
            .change_property8(
                PropMode::REPLACE,
                request.requestor,
                request.property,
                request.target,
                text.as_bytes(),
            )
            .expect("writes the text");
        request.property
    } else {
        x11rb::NONE
    };
    let notify = SelectionNotifyEvent {
        response_type: SELECTION_NOTIFY_EVENT,
        sequence: 0,
        time: request.time,
        requestor: request.requestor,
        selection: request.selection,
        target: request.target,
        property,
    };
    connection
        .send_event(false, request.requestor, EventMask::NO_EVENT, notify)
        .expect("answers the request");
    connection.flush().expect("flushes");
}

fn run_xdnd_source(display: String, events: mpsc::UnboundedSender<XdndSourceEvent>) {
    let (connection, screen_number) = x11rb::connect(Some(&display)).expect("connects to Xwayland");
    let atoms = XdndAtoms::intern(&connection);
    let utf8_string = intern(&connection, "UTF8_STRING");
    let root = connection.setup().roots[screen_number].root;
    let window = map_menu_window(
        &connection,
        screen_number,
        EventMask::BUTTON_PRESS | EventMask::BUTTON_RELEASE | EventMask::POINTER_MOTION,
    );
    let mut proxy = None;
    let mut dragging = false;
    while let Ok(event) = connection.wait_for_event() {
        let event = match event {
            Event::ButtonPress(press) => {
                connection
                    .set_selection_owner(window, atoms.selection, press.time)
                    .expect("takes the drag selection");
                connection.flush().expect("flushes");
                dragging = true;
                XdndSourceEvent::Pressed
            }
            Event::MotionNotify(motion) if dragging => {
                if proxy.is_none() {
                    proxy = window_under(&connection, root, motion.root_x, motion.root_y)
                        .filter(|under| *under != window);
                    if let Some(proxy) = proxy {
                        let enter = [window, XDND_VERSION << 24, utf8_string, 0, 0];
                        send_client_message(&connection, proxy, atoms.enter, enter);
                        if events.send(XdndSourceEvent::Entered).is_err() {
                            return;
                        }
                    }
                }
                if let Some(proxy) = proxy {
                    let at =
                        (u32::from(motion.root_x as u16) << 16) | u32::from(motion.root_y as u16);
                    let position = [window, 0, at, motion.time, atoms.action_copy];
                    send_client_message(&connection, proxy, atoms.position, position);
                }
                continue;
            }
            Event::ButtonRelease(release) => {
                if let Some(proxy) = proxy {
                    send_client_message(
                        &connection,
                        proxy,
                        atoms.drop,
                        [window, 0, release.time, 0, 0],
                    );
                }
                continue;
            }
            Event::SelectionRequest(request) => {
                serve_text(&connection, &request, DRAG_PAYLOAD);
                continue;
            }
            _ => continue,
        };
        if events.send(event).is_err() {
            return;
        }
    }
}

#[tokio::test]
async fn an_x11_drag_drops_onto_a_wayland_window() {
    const MOTION_INTERVAL: std::time::Duration = std::time::Duration::from_millis(10);
    let source = WaylandDisplaySrc::new()
        .with_size(WIDTH, HEIGHT)
        .with_framerate(FRAMERATE)
        .with_render_node(SOFTWARE_RENDERING)
        .with_xwayland();
    let input = source.input_sender();
    let mut display_names = source.display_names();
    let pull = register_appsink_pull("m2562_x11_drag");
    let mut graph: Graph<GraphNode> = Graph::new();
    let source_node = graph.add_source(GraphNode::source(source));
    let sink_node = graph.add_sink(GraphNode::element(
        AppSink::new().with_channel("m2562_x11_drag"),
    ));
    graph.link(source_node, sink_node).expect("links");
    let send = |event| input.send(event).expect("the display takes input");
    let checks = async {
        let mut events = connect_client(&mut display_names).await;
        client_globals(&mut events).await;
        let stride = WIDTH as usize * BYTES_PER_PIXEL;
        assert!(
            wait_for_picture(&pull, |picture| shows_client(picture, stride)).await,
            "a frame shows the Wayland client's window"
        );
        let x11_display = display_names
            .borrow()
            .as_ref()
            .and_then(|names| names.x11_display.clone())
            .expect("the source runs Xwayland");
        let (source_sender, mut source_events) = mpsc::unbounded_channel();
        std::thread::spawn(move || run_xdnd_source(x11_display, source_sender));
        let (start_x, start_y) = X11_DROP_POINT;
        assert!(
            wait_for_picture(&pull, |picture| {
                rgb_at(picture, stride, start_x, start_y) == X11_TARGET_RGB
            })
            .await,
            "a frame shows the X11 window"
        );
        send(WindowInput::Pointer {
            x: start_x,
            y: start_y,
        });
        send(WindowInput::Button {
            button: PointerButton::Left,
            down: true,
        });
        let pressed = tokio::time::timeout(TIMEOUT, source_events.recv()).await;
        assert_eq!(pressed.ok().flatten(), Some(XdndSourceEvent::Pressed));
        let (end_x, end_y) = WAYLAND_DRAG_POINT;
        // smithay maps its drop proxy once it sees the new drag selection, and the app looks for it on each motion
        let entered = tokio::time::timeout(TIMEOUT, async {
            for x in (end_x..start_x).rev().cycle() {
                send(WindowInput::Pointer { x, y: end_y });
                tokio::time::sleep(MOTION_INTERVAL).await;
                if let Ok(event) = source_events.try_recv() {
                    return event;
                }
            }
            unreachable!("the pointer moves forever")
        })
        .await;
        assert_eq!(entered.ok(), Some(XdndSourceEvent::Entered));
        send(WindowInput::Pointer { x: end_x, y: end_y });
        wait_for_client_event(&mut events, &ClientEvent::DropAccepted).await;
        send(WindowInput::Button {
            button: PointerButton::Left,
            down: false,
        });
        wait_for_client_event(&mut events, &ClientEvent::Dropped(DRAG_PAYLOAD.into())).await;
    };
    tokio::select! {
        result = run_graph(graph, &ZeroClock, 2) => panic!("the graph ended first: {result:?}"),
        () = checks => {}
    }
}

const REMOTE_CLIPBOARD_TEXT: &str = "text from the remote client";
const X11_CLIPBOARD_TEXT: &str = "text an X11 app copied";
const SELECTION_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(20);

async fn start_xwayland_display() -> WaylandDisplay {
    WaylandDisplay::start(WaylandDisplaySettings {
        width: WIDTH,
        height: HEIGHT,
        framerate_q16: FRAMERATE << 16,
        render_node: String::from(SOFTWARE_RENDERING),
        xwayland: true,
    })
    .await
    .expect("the display starts")
}

async fn wait_for_clipboard(display: &WaylandDisplay, expected: &str) {
    let mut texts = display.clipboard_texts();
    let published = tokio::time::timeout(
        TIMEOUT,
        texts.wait_for(|text| text.as_deref() == Some(expected)),
    )
    .await;
    assert!(
        matches!(published, Ok(Ok(_))),
        "the display publishes {expected:?}"
    );
}

#[tokio::test]
async fn a_wayland_copy_is_published_and_a_remote_clipboard_reaches_wayland_clients() {
    let display = start_software_display().await;
    let input = display.input_sender();
    let mut display_names = WaylandDisplaySrc::on_display(&display).display_names();
    let mut events = connect_client(&mut display_names).await;
    client_globals(&mut events).await;
    wait_for_client_event(&mut events, &ClientEvent::KeyboardEnter).await;

    input
        .send(WindowInput::Key {
            scancode: COPY_KEY,
            down: true,
        })
        .expect("the display takes input");
    wait_for_clipboard(&display, CLIPBOARD_TEXT).await;

    display
        .set_clipboard(String::from(REMOTE_CLIPBOARD_TEXT))
        .expect("the display takes the clipboard");
    wait_for_client_event(
        &mut events,
        &ClientEvent::Clipboard(String::from(REMOTE_CLIPBOARD_TEXT)),
    )
    .await;
}

fn run_x11_copier(display: String, text: &'static str) {
    let (connection, screen_number) = x11rb::connect(Some(&display)).expect("connects to Xwayland");
    let clipboard = intern(&connection, "CLIPBOARD");
    let window = map_menu_window(&connection, screen_number, EventMask::NO_EVENT);
    connection
        .set_selection_owner(window, clipboard, x11rb::CURRENT_TIME)
        .expect("takes the clipboard");
    connection.flush().expect("flushes");
    while let Ok(event) = connection.wait_for_event() {
        if let Event::SelectionRequest(request) = event {
            serve_text(&connection, &request, text);
        }
    }
}

fn x11_clipboard_text(display: String) -> String {
    let (connection, screen_number) = x11rb::connect(Some(&display)).expect("connects to Xwayland");
    let clipboard = intern(&connection, "CLIPBOARD");
    let utf8_string = intern(&connection, "UTF8_STRING");
    let pasted = intern(&connection, "G2G_PASTED");
    let window = map_menu_window(&connection, screen_number, EventMask::NO_EVENT);
    // the window manager takes the clipboard once the display offers the remote text
    while connection
        .get_selection_owner(clipboard)
        .expect("sends the request")
        .reply()
        .expect("reads the owner")
        .owner
        == x11rb::NONE
    {
        std::thread::sleep(SELECTION_POLL_INTERVAL);
    }
    connection
        .convert_selection(window, clipboard, utf8_string, pasted, x11rb::CURRENT_TIME)
        .expect("asks for the clipboard");
    connection.flush().expect("flushes");
    loop {
        let event = connection.wait_for_event().expect("Xwayland stays up");
        let Event::SelectionNotify(notify) = event else {
            continue;
        };
        assert_ne!(notify.property, x11rb::NONE, "the clipboard owner answers");
        let property = connection
            .get_property(true, window, pasted, AtomEnum::ANY, 0, PROPERTY_WORDS)
            .expect("sends the request")
            .reply()
            .expect("reads the clipboard");
        return String::from_utf8(property.value).expect("utf-8 text");
    }
}

fn x11_display_name(display: &WaylandDisplay) -> String {
    display
        .names()
        .x11_display
        .clone()
        .expect("the display runs Xwayland")
}

#[tokio::test]
async fn an_x11_copy_is_published() {
    let display = start_xwayland_display().await;
    let x11_display = x11_display_name(&display);
    std::thread::spawn(move || run_x11_copier(x11_display, X11_CLIPBOARD_TEXT));
    wait_for_clipboard(&display, X11_CLIPBOARD_TEXT).await;
}

#[tokio::test]
async fn a_remote_clipboard_reaches_x11_apps() {
    let display = start_xwayland_display().await;
    display
        .set_clipboard(String::from(REMOTE_CLIPBOARD_TEXT))
        .expect("the display takes the clipboard");
    let x11_display = x11_display_name(&display);
    let pasted = tokio::time::timeout(
        TIMEOUT,
        tokio::task::spawn_blocking(move || x11_clipboard_text(x11_display)),
    )
    .await
    .expect("the X11 app reads the clipboard in time")
    .expect("the X11 app's thread finishes");
    assert_eq!(pasted, REMOTE_CLIPBOARD_TEXT);
}
