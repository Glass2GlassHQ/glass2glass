// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "linux", feature = "display"))]

mod render_node;

use std::time::Duration;

use g2g_core::runtime::{parse_source, run_graph, GraphNode};
use g2g_core::{Graph, PipelineClock};
use g2g_plugins::registry::default_registry;
use g2g_plugins::wgpupresent::WgpuPresentSink;
use g2g_plugins::windowinput::{PointerButton, WindowInput};
use g2g_wayland_display::waylanddisplaysrc::{
    WaylandDisplay, WaylandDisplaySettings, SOFTWARE_RENDERING,
};
use tokio::sync::mpsc;

const WIDTH: u32 = 160;
const HEIGHT: u32 = 128;
const FRAMERATE: u32 = 30;
const TIMEOUT: Duration = Duration::from_secs(10);
// linux input-event-codes.h
const KEY_ESC: u32 = 1;
const KEY_A: u32 = 30;
const POINTER_X: u32 = 40;
const POINTER_Y: u32 = 30;
const POINTER_RETRY: Duration = Duration::from_millis(20);

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

async fn next_key_or_button(window: &mut mpsc::UnboundedReceiver<WindowInput>) -> WindowInput {
    loop {
        let input = tokio::time::timeout(TIMEOUT, window.recv())
            .await
            .expect("the window reports input in time")
            .expect("the window stays open");
        if matches!(input, WindowInput::Key { .. } | WindowInput::Button { .. }) {
            return input;
        }
    }
}

#[tokio::test]
async fn escape_releases_the_keyboard_grab_and_a_click_takes_it_again() {
    // wgpu needs a GPU, and machines without one have no render node
    if render_node::render_nodes().is_empty() {
        return;
    }
    let desktop = WaylandDisplay::start(WaylandDisplaySettings {
        width: WIDTH,
        height: HEIGHT,
        framerate_q16: FRAMERATE << 16,
        render_node: String::from(SOFTWARE_RENDERING),
        xwayland: false,
    })
    .await
    .expect("the desktop starts");
    // the present window connects to the compositor WAYLAND_DISPLAY names
    std::env::set_var("WAYLAND_DISPLAY", &desktop.names().wayland_socket);
    let desktop_input = desktop.input_sender();
    let press = |input: WindowInput| desktop_input.send(input).expect("the desktop takes input");
    let tap = |scancode: u32| {
        for down in [true, false] {
            press(WindowInput::Key { scancode, down });
        }
    };

    let mut sink = WgpuPresentSink::new()
        .with_title("m2566")
        .with_keyboard_grab();
    let mut window = sink.input_events();
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = parse_source(
        &default_registry(),
        &format!("videotestsrc is-live=true width={WIDTH} height={HEIGHT}"),
    )
    .expect("parses the source");
    let source_node = graph.add_source(GraphNode::Source(source));
    let sink_node = graph.add_sink(GraphNode::element(sink));
    graph.link(source_node, sink_node).expect("links");

    let checks = async {
        let focused = tokio::time::timeout(TIMEOUT, window.recv()).await;
        assert_eq!(
            focused.ok().flatten(),
            Some(WindowInput::Focus(true)),
            "the window takes the grab with the focus"
        );
        // grabbed: Escape releases the grab and stays local
        tap(KEY_ESC);
        // released: Escape goes through like any key
        tap(KEY_ESC);
        for down in [true, false] {
            assert_eq!(
                next_key_or_button(&mut window).await,
                WindowInput::Key {
                    scancode: KEY_ESC,
                    down
                }
            );
        }
        // the compositor gives keyboard focus at once, the pointer finds the window only after it draws
        let pointed = tokio::time::timeout(TIMEOUT, async {
            loop {
                press(WindowInput::Pointer {
                    x: POINTER_X,
                    y: POINTER_Y,
                });
                tokio::time::sleep(POINTER_RETRY).await;
                while let Ok(input) = window.try_recv() {
                    if matches!(input, WindowInput::Pointer { .. }) {
                        return;
                    }
                }
            }
        })
        .await;
        assert!(pointed.is_ok(), "the pointer reaches the window");
        for down in [true, false] {
            press(WindowInput::Button {
                button: PointerButton::Left,
                down,
            });
        }
        // grabbed again by the click
        tap(KEY_ESC);
        press(WindowInput::Key {
            scancode: KEY_A,
            down: true,
        });
        let expected = [
            WindowInput::Button {
                button: PointerButton::Left,
                down: true,
            },
            WindowInput::Button {
                button: PointerButton::Left,
                down: false,
            },
            WindowInput::Key {
                scancode: KEY_A,
                down: true,
            },
        ];
        for expected in expected {
            assert_eq!(next_key_or_button(&mut window).await, expected);
        }
    };
    tokio::select! {
        result = run_graph(graph, &ZeroClock, 2) => panic!("the graph ended first: {result:?}"),
        () = checks => {}
    }
}
