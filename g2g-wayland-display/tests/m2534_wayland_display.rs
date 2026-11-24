// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "linux", feature = "display"))]

mod wayland_client;

use g2g_core::runtime::{parse_launch, run_graph, GraphNode};
use g2g_core::Graph;
use g2g_plugins::appsink::{register_appsink_pull, AppSink};
use g2g_plugins::registry::default_registry;
use g2g_plugins::windowinput::{PointerButton, WindowInput};
use g2g_wayland_display::waylanddisplaysrc::{WaylandDisplaySrc, SOFTWARE_RENDERING};
use wayland_client::{
    client_globals, connect_client, drag_reaches_the_drop_target, shows_client,
    wait_for_client_event, ClientEvent, ZeroClock, BYTES_PER_PIXEL, FRAMERATE, HEIGHT, TIMEOUT,
    WIDTH,
};

// gtk 4 refuses to open a display that lacks this interface
const GTK_REQUIRED_INTERFACE: &str = "wl_data_device_manager";
// linux input-event-codes.h KEY_A
const KEY_A: u32 = 30;
const POINTER_X: u32 = 10;
const POINTER_Y: u32 = 20;

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
