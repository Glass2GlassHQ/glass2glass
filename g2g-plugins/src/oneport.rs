use alloc::boxed::Box;
use core::task::{Context, Poll};

use g2g_core::element::BoxFuture;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, ConfigureOutcome, ElementMetadata, G2gError, MultiOutputSink, MultiOutputSource,
    OutputSink, PipelinePacket, PropError, PropValue, PropertySpec, PushOutcome,
};

#[derive(Debug)]
pub struct OnePortSource<S> {
    source: S,
    properties: &'static [PropertySpec],
    metadata: ElementMetadata,
}

impl<S: MultiOutputSource> OnePortSource<S> {
    // `properties` is the part of the source's properties the single-output form exposes
    pub fn wrap(source: S, properties: &'static [PropertySpec], metadata: ElementMetadata) -> Self {
        assert_eq!(source.output_count(), 1, "a one-port source has one output");
        Self {
            source,
            properties,
            metadata,
        }
    }

    pub fn map(self, build: impl FnOnce(S) -> S) -> Self {
        Self::wrap(build(self.source), self.properties, self.metadata)
    }

    fn exposes(&self, name: &str) -> bool {
        self.properties.iter().any(|spec| spec.name == name)
    }
}

impl<S: MultiOutputSource> SourceLoop for OnePortSource<S> {
    type RunFuture<'a>
        = BoxFuture<'a, Result<u64, G2gError>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = BoxFuture<'a, Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        Box::pin(async move {
            self.source.resolve_output_caps().await?;
            self.source.output_caps(0)
        })
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move { self.source.run(&mut OnePortSink(out)).await })
    }

    fn latency(&self) -> g2g_core::query::LatencyReport {
        self.source.latency()
    }

    fn metadata(&self) -> ElementMetadata {
        self.metadata
    }

    fn properties(&self) -> &'static [PropertySpec] {
        self.properties
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if !self.exposes(name) {
            return Err(PropError::Unknown);
        }
        self.source.set_property(name, value)
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        self.exposes(name)
            .then(|| self.source.get_property(name))
            .flatten()
    }
}

struct OnePortSink<'a>(&'a mut dyn OutputSink);

impl MultiOutputSink for OnePortSink<'_> {
    fn poll_push_to(
        &mut self,
        cx: &mut Context<'_>,
        port: usize,
        packet: &mut Option<PipelinePacket>,
    ) -> Poll<Result<PushOutcome, G2gError>> {
        assert_eq!(port, 0, "a one-port element pushed to port {port}");
        self.0.poll_push(cx, packet)
    }

    fn begin_push_to(&mut self, _port: usize) {
        self.0.begin_push();
    }

    fn port_count(&self) -> usize {
        1
    }
}
