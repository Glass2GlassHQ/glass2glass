//! M1023: auto-plug around a decoder that turns out not to decode the stream.
//!
//! The search picks a decoder from pad templates and memory domains, which say
//! nothing about whether the hardware behind it will accept a particular
//! bitstream. When it will not, the run dies where GStreamer's `decodebin` would
//! have tried the next candidate. The pieces for that live here: the runner names
//! the failing element on the bus, the registry maps that instance back to its
//! factory, and `parse_launch_avoiding` re-plugs the line without it. This drives
//! all three, which is the fallback an application performs.
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;

use g2g_core::bus::{Bus, BusMessage};
use g2g_core::runtime::{
    block_on, parse_launch, parse_launch_avoiding, run_graph_with_bus, ElementFactory,
    LaunchFactory, Registry, SourceFactory, SourceLoop,
};
use g2g_core::{
    AsyncElement, Caps, CapsSet, ConfigureOutcome, Dim, G2gError, Interlace, OutputSink,
    PadTemplate, PipelineClock, PipelinePacket, Rate, RawVideoFormat, VideoCodec,
};

/// The runs are not paced; nothing here reads the clock.
struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// The line under test: nothing in it names a decoder, so both runs get whatever
/// the auto-plug search picks.
const PIPELINE: &str = "h264src ! decodebin ! countsink";

/// Link depth for the run; the source pushes a handful of frames.
const LINK_CAPACITY: usize = 2;

fn h264() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(64),
        height: Dim::Fixed(48),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn nv12() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(64),
        height: Dim::Fixed(48),
        framerate: Rate::Fixed(30 << 16),
        interlace: Interlace::Any,
    }
}

/// Frames the source pushes before end of stream.
const FRAMES: u64 = 4;

struct H264Source;

impl SourceLoop for H264Source {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(h264()))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for i in 0..FRAMES {
                let frame = g2g_core::Frame {
                    domain: g2g_core::MemoryDomain::System(
                        g2g_core::memory::SystemSlice::from_boxed(alloc_boxed_slice()),
                    ),
                    timing: Default::default(),
                    sequence: i,
                    meta: Default::default(),
                };
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(FRAMES)
        })
    }
}

fn alloc_boxed_slice() -> Box<[u8]> {
    vec![0u8; 8].into_boxed_slice()
}

/// A decoder stand-in. `refuses` models a decoder the hardware turns out not to
/// run: negotiation is fine and the failure appears on the first frame, exactly
/// where a driver rejects a stream it cannot decode.
struct StubDecoder {
    refuses: bool,
}

impl AsyncElement for StubDecoder {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, _upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(nv12())
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let refuses = self.refuses;
        Box::pin(async move {
            if refuses && matches!(packet, PipelinePacket::DataFrame(_)) {
                return Err(G2gError::CapsMismatch);
            }
            out.push(packet).await.map(|_| ())
        })
    }
}

struct CountSink;

impl AsyncElement for CountSink {
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

    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move { Ok(()) })
    }
}

fn decoder_templates() -> Vec<PadTemplate> {
    Vec::from([
        PadTemplate::sink(CapsSet::one(h264())),
        PadTemplate::source(CapsSet::one(nv12())),
    ])
}

/// The refusing decoder is the one the search picks (it is the only candidate
/// producing the domain the sink is happy with, and it registers first), so a
/// plain run always hits it.
fn registry() -> Registry {
    let mut registry = Registry::new();
    registry
        .register_source(SourceFactory::new("h264src", h264(), || {
            Box::new(H264Source)
        }))
        .register(ElementFactory::new(
            "refusingdec",
            decoder_templates(),
            |_out| Box::new(StubDecoder { refuses: true }),
        ))
        .register(ElementFactory::new(
            "workingdec",
            decoder_templates(),
            |_out| Box::new(StubDecoder { refuses: false }),
        ))
        .register_launch(LaunchFactory::new(
            "refusingdec",
            decoder_templates(),
            || Box::new(StubDecoder { refuses: true }),
        ))
        .register_launch(LaunchFactory::new(
            "workingdec",
            decoder_templates(),
            || Box::new(StubDecoder { refuses: false }),
        ))
        .register_launch(LaunchFactory::new(
            "countsink",
            Vec::from([PadTemplate::sink(CapsSet::one(nv12()))]),
            || Box::new(CountSink),
        ));
    registry
}

/// Run `graph` and return the element the bus names as having failed, if it did.
fn run_reporting_failure(graph: g2g_core::Graph<g2g_core::runtime::GraphNode>) -> Option<String> {
    let (bus, handle) = Bus::new(64);
    let clock = ZeroClock;
    let result = block_on(run_graph_with_bus(graph, &clock, LINK_CAPACITY, &handle));
    let mut failed = None;
    while let Some(message) = bus.try_recv() {
        if let BusMessage::ElementError { element, .. } = message {
            failed = Some(element);
        }
    }
    if result.is_ok() {
        assert!(failed.is_none(), "a run that succeeded named no failure");
    }
    failed
}

#[test]
fn a_refused_stream_replugs_onto_the_other_decoder() {
    let registry = registry();

    // First run: the search picks the decoder that refuses the stream, and the
    // bus says which element that was.
    let graph = parse_launch(&registry, PIPELINE).expect("the line parses");
    let failed = run_reporting_failure(graph).expect("the run failed and named the element");
    let factory = registry
        .factory_of_instance(&failed)
        .expect("the failing instance maps back to its factory");
    assert_eq!(
        factory, "refusingdec",
        "the element named on the bus is the decoder that refused"
    );

    // The fallback: re-plug the same line with that factory ruled out.
    let graph = parse_launch_avoiding(&registry, PIPELINE, &[factory])
        .expect("the line parses without the refused decoder");
    let (bus, handle) = Bus::new(64);
    let clock = ZeroClock;
    let stats = block_on(run_graph_with_bus(graph, &clock, LINK_CAPACITY, &handle))
        .expect("the other decoder runs the stream");
    assert_eq!(
        stats.frames_consumed, FRAMES,
        "every frame reaches the sink"
    );
    while let Some(message) = bus.try_recv() {
        assert!(
            !matches!(message, BusMessage::ElementError { .. }),
            "the fallback run names no failing element"
        );
    }
}

#[test]
fn ruling_out_every_decoder_fails_the_parse_instead_of_choosing_one() {
    // Nothing is silently substituted: with no candidate left the line does not
    // parse, so a caller retrying in a loop terminates.
    let registry = registry();
    assert!(
        parse_launch_avoiding(&registry, PIPELINE, &["refusingdec", "workingdec"]).is_err(),
        "no decoder left means no chain"
    );
}

#[test]
fn an_instance_name_maps_back_to_the_factory_that_built_it() {
    let registry = registry();
    assert_eq!(
        registry.factory_of_instance("StubDecoder0"),
        Some("refusingdec"),
        "the first factory whose element carries that log category"
    );
    assert_eq!(registry.factory_of_instance("NoSuchElement3"), None);
    assert_eq!(registry.factory_of_instance("7"), None);
}
