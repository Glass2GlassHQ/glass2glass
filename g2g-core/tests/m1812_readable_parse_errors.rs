// A decodebin with no decoder for its input names the stream in gst caps syntax, the
// codec and picture size, rather than the Rust Debug form of the caps.
#![cfg(all(feature = "std", feature = "runtime"))]

use core::future::Future;
use core::pin::Pin;

use g2g_core::runtime::{parse_launch, LaunchFactory, Registry, SourceFactory, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, OutputSink, PadTemplate,
    PipelinePacket, Rate, VideoCodec,
};

const WIDTH: u32 = 352;
const HEIGHT: u32 = 288;

fn vp9() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::Vp9,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Range {
            min_q16: 1 << 16,
            max_q16: 240 << 16,
        },
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

type BoxedResult<'a, T> = Pin<Box<dyn Future<Output = Result<T, G2gError>> + 'a>>;

struct Vp9Source;

impl SourceLoop for Vp9Source {
    type RunFuture<'a>
        = BoxedResult<'a, u64>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(vp9()))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, _out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async { Ok(0) })
    }
}

struct AnySink;

impl AsyncElement for AnySink {
    type ProcessFuture<'a>
        = BoxedResult<'a, ()>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
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

#[test]
fn a_missing_decoder_names_the_codec_and_size() {
    let mut registry = Registry::new();
    registry
        .register_source(SourceFactory::new("vp9src", vp9(), || Box::new(Vp9Source)))
        .register_launch(LaunchFactory::new(
            "anysink",
            Vec::from([PadTemplate::sink_any()]),
            || Box::new(AnySink),
        ));
    let error = parse_launch(&registry, "vp9src ! decodebin ! anysink")
        .expect_err("no decoder is registered for VP9");
    let message = error.to_string();
    assert!(
        message.contains(&format!("video/x-vp9,width={WIDTH},height={HEIGHT}")),
        "{message}"
    );
    assert!(!message.contains("CompressedVideo"), "{message}");
}
