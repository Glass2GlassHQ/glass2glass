//! M1018: a launch line's `decodebin` picks its decoder from the memory the
//! consumer declares it takes, the text-parser counterpart of the graph-side
//! derivation (M989). Two H.264 decoders are registered, a CPU one first and a
//! `Cuda`-producing one second, so registration order alone would always pick
//! the CPU decoder; only the sink at the end of the line differs between cases.
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Mutex, MutexGuard};

use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::runtime::{
    parse_launch, ElementFactory, LaunchFactory, Registry, SourceFactory, SourceLoop,
    UriSourceFactory,
};
use g2g_core::{
    AsyncElement, Caps, CapsSet, ConfigureOutcome, Dim, G2gError, Interlace, OutputSink,
    PadTemplate, PipelinePacket, Rate, RawVideoFormat, VideoCodec,
};

fn h264() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

fn nv12() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        interlace: Interlace::Any,
    }
}

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
            out.push(PipelinePacket::Eos).await?;
            Ok(0)
        })
    }
}

/// H.264 -> NV12 decoder stand-in that records its name when the launch parser
/// builds it, so the parsed line reports which decoder the search chose.
struct StubDecoder;

impl StubDecoder {
    fn built(name: &'static str) -> Box<dyn g2g_core::element::DynAsyncElement> {
        decoders_built().lock().unwrap().push(name);
        Box::new(StubDecoder)
    }
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
        Box::pin(async move { out.push(packet).await.map(|_| ()) })
    }
}

/// Raw-video sink accepting exactly `accepted` memory.
struct DomainSink {
    accepted: DomainSet,
}

impl AsyncElement for DomainSink {
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

    fn input_domains(&self) -> DomainSet {
        self.accepted
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move { out.push(packet).await.map(|_| ()) })
    }
}

fn decoder_templates() -> Vec<PadTemplate> {
    Vec::from([
        PadTemplate::sink(CapsSet::one(h264())),
        PadTemplate::source(CapsSet::one(nv12())),
    ])
}

fn sink_templates() -> Vec<PadTemplate> {
    Vec::from([PadTemplate::sink(CapsSet::one(nv12()))])
}

/// The decoder names the launch parser has constructed, in order.
fn decoders_built() -> &'static Mutex<Vec<&'static str>> {
    static BUILT: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    &BUILT
}

/// The recording registry is process-global, so the cases run one at a time.
fn serialized() -> MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// CPU decoder registered first, `Cuda` decoder second: registration order alone
/// picks the CPU one, so any GPU pick came from the consumer's declaration.
fn registry() -> Registry {
    let mut registry = Registry::new();
    registry
        .register_source(SourceFactory::new("h264src", h264(), || {
            Box::new(H264Source)
        }))
        .register(ElementFactory::new("cpudec", decoder_templates(), |_out| {
            StubDecoder::built("cpudec")
        }))
        .register(
            ElementFactory::new("gpudec", decoder_templates(), |_out| {
                StubDecoder::built("gpudec")
            })
            .produces(MemoryDomainKind::Cuda),
        )
        .register_launch(LaunchFactory::new("cpudec", decoder_templates(), || {
            StubDecoder::built("cpudec")
        }))
        .register_launch(LaunchFactory::new("gpudec", decoder_templates(), || {
            StubDecoder::built("gpudec")
        }))
        .register_launch(LaunchFactory::new("cudasink", sink_templates(), || {
            Box::new(DomainSink {
                accepted: DomainSet::only(MemoryDomainKind::Cuda),
            })
        }))
        .register_launch(LaunchFactory::new("plainsink", sink_templates(), || {
            Box::new(DomainSink {
                accepted: DomainSet::ALL,
            })
        }))
        .register_uri(UriSourceFactory::new("test", |_uri| {
            Ok((Box::new(H264Source), h264()))
        }));
    registry
}

/// Parse `line` and report the decoders its `decodebin` built.
fn decoders_of(line: &str) -> Vec<&'static str> {
    let _guard = serialized();
    decoders_built().lock().unwrap().clear();
    parse_launch(&registry(), line).expect("line parses");
    decoders_built().lock().unwrap().clone()
}

#[test]
fn a_gpu_consumer_picks_the_gpu_decoder() {
    assert_eq!(
        decoders_of("h264src ! decodebin ! cudasink"),
        ["gpudec"],
        "a Cuda-only sink picks the Cuda decoder with no domain named on the line"
    );
}

#[test]
fn a_consumer_declaring_nothing_keeps_the_cpu_decoder() {
    assert_eq!(
        decoders_of("h264src ! decodebin ! plainsink"),
        ["cpudec"],
        "an undeclared sink leaves the plain selection alone"
    );
}

#[test]
fn a_decodebin_at_the_end_of_a_line_keeps_the_cpu_decoder() {
    // No consumer to read: the plain selection, not a failure.
    assert_eq!(decoders_of("h264src ! decodebin"), ["cpudec"]);
}

#[test]
fn playbin_reads_the_sink_it_was_given() {
    assert_eq!(
        decoders_of("playbin uri=test://stream video-sink=cudasink"),
        ["gpudec"],
        "playbin names its own sink, so the decoder follows that sink's memory"
    );
    assert_eq!(
        decoders_of("playbin uri=test://stream video-sink=plainsink"),
        ["cpudec"]
    );
}

#[test]
fn uridecodebin_reads_the_element_after_it() {
    assert_eq!(
        decoders_of("uridecodebin uri=test://stream ! cudasink"),
        ["gpudec"]
    );
    assert_eq!(
        decoders_of("uridecodebin uri=test://stream ! plainsink"),
        ["cpudec"]
    );
}
