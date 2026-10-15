#![cfg(all(feature = "std", feature = "runtime"))]

use core::cell::RefCell;
use core::future::Future;
use core::pin::Pin;

use g2g_core::runtime::{
    parse_launch, ElementFactory, LaunchFactory, Registry, SourceFactory, SourceLoop,
};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, Interlace,
    OutputSink, PadTemplate, PipelinePacket, Rate, RawVideoFormat, VideoCodec,
};

const PIPELINE: &str = "h264src ! decodebin ! i420sink";

fn h264() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn raw(format: RawVideoFormat) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        interlace: Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

type BoxedResult<'a, T> = Pin<Box<dyn Future<Output = Result<T, G2gError>> + 'a>>;

struct H264Source;

impl SourceLoop for H264Source {
    type RunFuture<'a>
        = BoxedResult<'a, u64>
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

    fn run<'a>(&'a mut self, _out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async { Ok(0) })
    }
}

struct Stub {
    accepted: Option<Caps>,
}

impl AsyncElement for Stub {
    type ProcessFuture<'a>
        = BoxedResult<'a, ()>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        match &self.accepted {
            Some(accepted) => accepted.intersect(upstream),
            None => Ok(upstream.clone()),
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        match &self.accepted {
            Some(accepted) => CapsConstraint::Identity(CapsSet::one(accepted.clone())),
            None => CapsConstraint::IdentityAny,
        }
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

thread_local! {
    static BUILT_DECODERS: RefCell<Vec<(&'static str, Caps)>> = const { RefCell::new(Vec::new()) };
}

fn record_build(name: &'static str, output: &Caps) -> Box<Stub> {
    BUILT_DECODERS.with(|built| built.borrow_mut().push((name, output.clone())));
    Box::new(Stub { accepted: None })
}

fn decoders_built_by(registry: &Registry) -> Vec<(&'static str, Caps)> {
    BUILT_DECODERS.with(|built| built.take());
    parse_launch(registry, PIPELINE).expect("the line parses");
    BUILT_DECODERS.with(|built| built.take())
}

fn decoder_templates(outputs: &[RawVideoFormat]) -> Vec<PadTemplate> {
    Vec::from([
        PadTemplate::sink(CapsSet::one(h264())),
        PadTemplate::source(CapsSet::from_alternatives(
            outputs.iter().map(|format| raw(*format)).collect(),
        )),
    ])
}

fn registry_with(decoders: impl FnOnce(&mut Registry)) -> Registry {
    let mut registry = Registry::new();
    registry
        .register_source(SourceFactory::new("h264src", h264(), || {
            Box::new(H264Source)
        }))
        .register_launch(LaunchFactory::new(
            "i420sink",
            Vec::from([PadTemplate::sink(CapsSet::one(raw(RawVideoFormat::I420)))]),
            || {
                Box::new(Stub {
                    accepted: Some(raw(RawVideoFormat::I420)),
                })
            },
        ));
    decoders(&mut registry);
    registry
}

#[test]
fn a_multi_format_decoder_is_built_for_the_format_its_consumer_takes() {
    // NV12 first, so a search that ignores the consumer builds an NV12 decoder
    let registry = registry_with(|registry| {
        registry.register(ElementFactory::new(
            "h264dec",
            decoder_templates(&[RawVideoFormat::Nv12, RawVideoFormat::I420]),
            |output| record_build("h264dec", output),
        ));
    });
    assert_eq!(
        decoders_built_by(&registry),
        [("h264dec", raw(RawVideoFormat::I420))]
    );
}

#[test]
fn a_pinned_decoder_is_kept_when_its_consumer_rejects_its_output() {
    let mut registry = registry_with(|registry| {
        registry
            .register(ElementFactory::new(
                "i420dec",
                decoder_templates(&[RawVideoFormat::I420]),
                |output| record_build("i420dec", output),
            ))
            .register(ElementFactory::new(
                "nv12dec",
                decoder_templates(&[RawVideoFormat::Nv12]),
                |output| record_build("nv12dec", output),
            ));
    });
    assert_eq!(
        decoders_built_by(&registry),
        [("i420dec", raw(RawVideoFormat::I420))]
    );

    registry
        .apply_rank_spec("nv12dec:max")
        .expect("nv12dec is registered");
    assert_eq!(
        decoders_built_by(&registry),
        [("nv12dec", raw(RawVideoFormat::Nv12))]
    );
}
