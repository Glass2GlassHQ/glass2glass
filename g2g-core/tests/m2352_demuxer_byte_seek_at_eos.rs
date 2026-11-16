// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{
    attach_seek_controller, block_on, eos_then_wait_for_seek, ElementSeekRoute, GraphNode,
    RunBuilder, SeekController, SeekFormat, SourceLoop,
};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Frame, FrameTiming, G2gError, Graph, MultiOutputElement,
    MultiOutputSink, OutputSink, PipelineClock, PipelinePacket, Seek,
};

const LINK_CAPACITY: usize = 2;
const STREAM_LEN: u64 = 8;
// the demuxer reads the stream a second time from here, as one reads a moov stored after its mdat
const REREAD_FROM: u64 = 3;
const DEMUX_PORT_COUNT: u8 = 1;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps() -> Caps {
    Caps::ByteStream {
        encoding: g2g_core::ByteStreamEncoding::Mp4,
    }
}

// one frame per byte offset, each stamped with its offset
#[derive(Default)]
struct ByteSource {
    seek: Option<SeekController>,
}

impl SourceLoop for ByteSource {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(caps()))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn seek_format(&self) -> Option<SeekFormat> {
        Some(SeekFormat::Bytes)
    }

    fn set_seek(&mut self, controller: SeekController) {
        self.seek = Some(controller);
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let seek = self.seek.clone().expect("the test attaches a controller");
            let mut position = 0;
            loop {
                if let Some(taken) = seek.take_seek(Ok) {
                    out.push(PipelinePacket::Flush).await?;
                    position = taken.start;
                }
                if position < STREAM_LEN {
                    let frame = Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 1]))),
                        FrameTiming {
                            pts_ns: position,
                            ..FrameTiming::default()
                        },
                        position,
                    );
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                    position += 1;
                    continue;
                }
                if !eos_then_wait_for_seek(Some(&seek), out).await? {
                    return Ok(position);
                }
            }
        })
    }
}

// passes frames on, and at its first Eos byte-seeks its source to read the stream again
#[derive(Default)]
struct Rereader {
    upstream: Option<SeekController>,
    reread_asked: bool,
    awaiting_own_flush: bool,
}

impl Rereader {
    // whether the packet goes on to the output
    fn passes(&mut self, packet: &PipelinePacket) -> bool {
        match packet {
            PipelinePacket::Flush if self.awaiting_own_flush => {
                self.awaiting_own_flush = false;
                false
            }
            PipelinePacket::Eos if !self.reread_asked => {
                self.reread_asked = true;
                self.awaiting_own_flush = true;
                self.upstream
                    .as_ref()
                    .expect("the run gives the demuxer its byte source")
                    .seek(Seek::flush_to(REREAD_FROM));
                false
            }
            PipelinePacket::Eos => false,
            _ => true,
        }
    }
}

impl AsyncElement for Rereader {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn seek_route(&self) -> Option<ElementSeekRoute> {
        Some(ElementSeekRoute::UpstreamBytes)
    }

    fn set_seek(&mut self, _app: SeekController, upstream: SeekController) {
        self.upstream = Some(upstream);
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if self.passes(&packet) {
                out.push(packet).await?;
            }
            Ok(())
        })
    }
}

#[derive(Default)]
struct RereadingDemux(Rereader);

impl MultiOutputElement for RereadingDemux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn seek_route(&self) -> Option<ElementSeekRoute> {
        Some(ElementSeekRoute::UpstreamBytes)
    }

    fn set_seek(&mut self, _app: SeekController, upstream: SeekController) {
        self.0.upstream = Some(upstream);
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if self.0.passes(&packet) {
                out.push_to(0, packet).await?;
            }
            Ok(())
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Seen {
    Frame(u64),
    Eos,
}

struct RecordingSink {
    seen: Arc<Mutex<Vec<Seen>>>,
}

impl AsyncElement for RecordingSink {
    type ProcessFuture<'a>
        = core::future::Ready<Result<(), G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let seen = match packet {
            PipelinePacket::DataFrame(frame) => Some(Seen::Frame(frame.timing.pts_ns)),
            PipelinePacket::Eos => Some(Seen::Eos),
            _ => None,
        };
        self.seen.lock().unwrap().extend(seen);
        core::future::ready(Ok(()))
    }
}

fn read_twice_then_ended() -> Vec<Seen> {
    (0..STREAM_LEN)
        .chain(REREAD_FROM..STREAM_LEN)
        .map(Seen::Frame)
        .chain([Seen::Eos])
        .collect()
}

fn run_to_end(mut graph: Graph<GraphNode>) {
    let controller = attach_seek_controller(&mut graph).expect("the demuxer can seek");
    block_on(
        RunBuilder::new(graph, &ZeroClock, LINK_CAPACITY)
            .with_seek_controller(&controller)
            .run(),
    )
    .expect("the run ends at the end of the second read");
}

#[test]
fn a_transform_demuxer_rereads_its_source_after_the_eos() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_source(GraphNode::source(ByteSource::default()));
    let demux = graph.add_transform(GraphNode::element(Rereader::default()));
    let sink = graph.add_sink(GraphNode::element(RecordingSink { seen: seen.clone() }));
    graph.link(source, demux).unwrap();
    graph.link(demux, sink).unwrap();
    run_to_end(graph);
    assert_eq!(*seen.lock().unwrap(), read_twice_then_ended());
}

#[test]
fn a_fan_out_demuxer_keeps_its_ports_open_through_its_own_byte_seek() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_source(GraphNode::source(ByteSource::default()));
    let demux = graph.add_demux(
        GraphNode::demux(RereadingDemux::default()),
        DEMUX_PORT_COUNT,
    );
    let sink = graph.add_sink(GraphNode::element(RecordingSink { seen: seen.clone() }));
    graph.link(source, demux.input()).unwrap();
    graph.link(demux.out(0), sink).unwrap();
    run_to_end(graph);
    assert_eq!(*seen.lock().unwrap(), read_twice_then_ended());
}
