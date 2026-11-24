// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "linux", feature = "display"))]

use std::time::Duration;

use g2g_core::runtime::{run_graph, GraphNode};
use g2g_core::{Graph, PipelineClock};
use g2g_plugins::appsink::{register_appsink_pull, AppSink};
use g2g_plugins::windowinput::WindowInput;
use g2g_wayland_display::waylanddisplaysrc::WaylandDisplaySrc;
use tokio::sync::mpsc;
use x11rb::connection::Connection;
use x11rb::protocol::xproto::{ConnectionExt, CreateWindowAux, EventMask, WindowClass};
use x11rb::protocol::Event;
use x11rb::COPY_DEPTH_FROM_PARENT;

const WIDTH: u32 = 160;
const HEIGHT: u32 = 128;
const FRAMERATE: u32 = 30;
const BYTES_PER_PIXEL: usize = 4;
const TIMEOUT: Duration = Duration::from_secs(10);
const SOFTWARE_RENDERING: &str = "software";
const WINDOW_RGB: [u8; 3] = [0x20, 0x80, 0xC0];
// linux input-event-codes.h KEY_A
const KEY_A: u32 = 30;
// X11 keycodes are evdev codes offset by 8
const X11_KEYCODE_OFFSET: u32 = 8;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[derive(Debug, PartialEq)]
enum X11Event {
    FocusIn,
    KeyPress(u8),
}

fn run_x11_client(display: String, events: mpsc::UnboundedSender<X11Event>) {
    let (connection, screen_number) = x11rb::connect(Some(&display)).expect("connects to Xwayland");
    let screen = &connection.setup().roots[screen_number];
    let [red, green, blue] = WINDOW_RGB;
    let background = u32::from_be_bytes([0, red, green, blue]);
    let window = connection.generate_id().expect("allocates a window id");
    connection
        .create_window(
            COPY_DEPTH_FROM_PARENT,
            window,
            screen.root,
            0,
            0,
            WIDTH as u16,
            HEIGHT as u16,
            0,
            WindowClass::INPUT_OUTPUT,
            screen.root_visual,
            &CreateWindowAux::new()
                .background_pixel(background)
                .event_mask(EventMask::KEY_PRESS | EventMask::FOCUS_CHANGE),
        )
        .expect("creates a window");
    connection.map_window(window).expect("maps the window");
    connection.flush().expect("flushes");
    while let Ok(event) = connection.wait_for_event() {
        let event = match event {
            Event::FocusIn(_) => X11Event::FocusIn,
            Event::KeyPress(press) => X11Event::KeyPress(press.detail),
            _ => continue,
        };
        if events.send(event).is_err() {
            return;
        }
    }
}

async fn wait_for_x11_event(events: &mut mpsc::UnboundedReceiver<X11Event>, expected: X11Event) {
    let found = tokio::time::timeout(TIMEOUT, async {
        while let Some(event) = events.recv().await {
            if event == expected {
                return true;
            }
        }
        false
    })
    .await;
    assert_eq!(found, Ok(true), "the X11 client receives {expected:?}");
}

fn center_rgb(picture: &[u8]) -> [u8; 3] {
    let center = (HEIGHT as usize / 2 * WIDTH as usize + WIDTH as usize / 2) * BYTES_PER_PIXEL;
    [picture[center], picture[center + 1], picture[center + 2]]
}

#[tokio::test]
async fn an_x11_window_reaches_the_frames_and_keys_reach_it() {
    let source = WaylandDisplaySrc::new()
        .with_size(WIDTH, HEIGHT)
        .with_framerate(FRAMERATE)
        .with_render_node(SOFTWARE_RENDERING)
        .with_xwayland();
    let input = source.input_sender();
    let mut display_names = source.display_names();
    let pull = register_appsink_pull("m2537_frames");
    let mut graph: Graph<GraphNode> = Graph::new();
    let source_node = graph.add_source(GraphNode::source(source));
    let sink_node = graph.add_sink(GraphNode::element(
        AppSink::new().with_channel("m2537_frames"),
    ));
    graph.link(source_node, sink_node).expect("links");
    let checks = async {
        let x11_display = tokio::time::timeout(TIMEOUT, display_names.wait_for(Option::is_some))
            .await
            .expect("Xwayland starts")
            .expect("the source is alive")
            .as_ref()
            .and_then(|names| names.x11_display.clone())
            .expect("the source runs Xwayland");
        let (event_sender, mut events) = mpsc::unbounded_channel();
        std::thread::spawn(move || run_x11_client(x11_display, event_sender));

        let window_shown = tokio::time::timeout(TIMEOUT, async {
            loop {
                let frame = pull.pull().await.expect("the source keeps producing");
                let picture = frame.domain.as_system_slice().expect("system memory");
                if center_rgb(picture) == WINDOW_RGB {
                    return;
                }
            }
        })
        .await;
        assert!(window_shown.is_ok(), "a frame shows the X11 window");

        wait_for_x11_event(&mut events, X11Event::FocusIn).await;
        input
            .send(WindowInput::Key {
                scancode: KEY_A,
                down: true,
            })
            .expect("the display takes input");
        let keycode = u8::try_from(KEY_A + X11_KEYCODE_OFFSET).expect("fits an X11 keycode");
        wait_for_x11_event(&mut events, X11Event::KeyPress(keycode)).await;
    };
    tokio::select! {
        result = run_graph(graph, &ZeroClock, 2) => panic!("the graph ended first: {result:?}"),
        () = checks => {}
    }
}
