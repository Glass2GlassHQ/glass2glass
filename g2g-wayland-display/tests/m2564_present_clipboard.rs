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
use g2g_wayland_display::waylanddisplaysrc::{
    WaylandDisplay, WaylandDisplaySettings, SOFTWARE_RENDERING,
};
use tokio::sync::watch;

const WIDTH: u32 = 160;
const HEIGHT: u32 = 128;
const FRAMERATE: u32 = 30;
const TIMEOUT: Duration = Duration::from_secs(10);
const HOST_TEXT: &str = "text the host copied";
const DESKTOP_TEXT: &str = "text the user copied on the client's desktop";

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn a_present_window_shares_its_desktop_clipboard() {
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

    let mut sink = WgpuPresentSink::new().with_title("m2564");
    // the window binds the keyboard, whose serials set the selection, only when input is read
    let _input = sink.input_events();
    let (host_texts, remote_texts) = watch::channel(None);
    let mut local_copies = sink.sync_clipboard(remote_texts);
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
        let mut desktop_texts = desktop.clipboard_texts();
        host_texts.send_replace(Some(String::from(HOST_TEXT)));
        let published = tokio::time::timeout(
            TIMEOUT,
            desktop_texts.wait_for(|text| text.as_deref() == Some(HOST_TEXT)),
        )
        .await;
        assert!(
            matches!(published, Ok(Ok(_))),
            "the window's selection reaches the desktop"
        );

        desktop
            .set_clipboard(String::from(DESKTOP_TEXT))
            .expect("the desktop takes the clipboard");
        let copied = tokio::time::timeout(TIMEOUT, local_copies.recv()).await;
        assert_eq!(
            copied.ok().flatten().as_deref(),
            Some(DESKTOP_TEXT),
            "the window reports the desktop's copy, and not the host text it set itself"
        );
    };
    tokio::select! {
        result = run_graph(graph, &ZeroClock, 2) => panic!("the graph ended first: {result:?}"),
        () = checks => {}
    }
}
