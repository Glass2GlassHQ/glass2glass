// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use g2g_core::log::Target;
use g2g_core::runtime::SourceLoop;
use g2g_core::runtime::{
    parse_chain, parse_element, parse_source, GraphNode, GraphNodeRef, ParseError, RunBuilder,
};
use g2g_core::{Graph, PropValue};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;
use g2g_plugins::remotewttrack::{RemoteWtTrackSink, RemoteWtTrackSrc};

use crate::lineargraph::linear_graph;

const AUDIO_CATEGORY: &str = "g2g-desktop-audio";
// audio delay grows with link depth
const LINK_CAPACITY: usize = 2;

// pulse names the default output's monitor source @DEFAULT_MONITOR@
#[cfg(target_os = "linux")]
pub(crate) const CAPTURE: &str = "pulsesrc device=@DEFAULT_MONITOR@";
#[cfg(target_os = "windows")]
pub(crate) const CAPTURE: &str = "wasapisrc loopback=true";
const ENCODE: &str = "audioconvert ! opusenc frame-size=10 audio-type=restricted-lowdelay";
const DECODE: &str = "opusdec ! audioconvert";
// frames later than this behind the fastest recent one are dropped
const JITTER_LATENCY_MS: u64 = 60;
// pulsesink starts playing once buffer-time of audio is queued
#[cfg(target_os = "linux")]
const PLAYBACK: &str = "pulsesink buffer-time=40000";
#[cfg(target_os = "windows")]
const PLAYBACK: &str = "wasapisink";

fn send_graph(track: RemoteWtTrackSink) -> Result<Graph<GraphNode>, ParseError> {
    let registry = default_registry();
    let capture = parse_source(&registry, CAPTURE)?;
    let encode = parse_chain(&registry, ENCODE)?;
    Ok(linear_graph(
        GraphNodeRef::Source(capture),
        encode.into_iter().map(GraphNodeRef::Element),
        GraphNodeRef::element(track),
    ))
}

fn play_graph(mut track: RemoteWtTrackSrc) -> Result<Graph<GraphNode>, ParseError> {
    track
        .set_property("latency", PropValue::Uint(JITTER_LATENCY_MS))
        .expect("the track source takes a latency in milliseconds");
    let registry = default_registry();
    let decode = parse_chain(&registry, DECODE)?;
    let playback = parse_element(&registry, PLAYBACK)?;
    Ok(linear_graph(
        GraphNodeRef::source(track),
        decode.into_iter().map(GraphNodeRef::Element),
        GraphNodeRef::Element(playback),
    ))
}

async fn run(graph: Result<Graph<GraphNode>, ParseError>) {
    let log = Target::category(AUDIO_CATEGORY);
    let graph = match graph {
        Ok(graph) => graph,
        Err(e) => {
            g2g_core::g2g_warn!(log, "audio pipeline: {e}");
            return;
        }
    };
    let clock = WallClock::new();
    if let Err(e) = RunBuilder::new(graph, &clock, LINK_CAPACITY).run().await {
        g2g_core::g2g_warn!(log, "audio stopped: {e:?}");
    }
}

// host audio on the video graph's runtime measured 1 ms p50 and 3 ms p99 added to each video frame
pub(crate) fn send(track: RemoteWtTrackSink) {
    let log = Target::category(AUDIO_CATEGORY);
    let spawned = std::thread::Builder::new()
        .name(AUDIO_CATEGORY.into())
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build();
            match runtime {
                Ok(runtime) => runtime.block_on(run(send_graph(track))),
                Err(e) => g2g_core::g2g_warn!(log, "audio runtime: {e}"),
            }
        });
    if let Err(e) = spawned {
        g2g_core::g2g_warn!(log, "audio thread: {e}");
    }
}

pub(crate) async fn play(track: Option<RemoteWtTrackSrc>) {
    if let Some(track) = track {
        run(play_graph(track)).await;
    }
}
