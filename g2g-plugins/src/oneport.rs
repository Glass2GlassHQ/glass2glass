use alloc::boxed::Box;
use alloc::string::String;
use core::task::{Context, Poll};

use g2g_core::element::BoxFuture;
use g2g_core::memory::DomainSet;
use g2g_core::query::LatencyReport;
use g2g_core::runtime::{ElementSeekRoute, SeekController, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, G2gError,
    MultiOutputElement, MultiOutputSink, MultiOutputSource, OutputSink, PipelinePacket, PropError,
    PropValue, PropertySpec, PushOutcome,
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

#[derive(Debug)]
pub struct OnePortElement<E> {
    element: E,
    log_category: &'static str,
}

impl<E: MultiOutputElement> OnePortElement<E> {
    pub fn wrap(element: E, log_category: &'static str) -> Self {
        assert!(
            element.port_output_caps(0).is_some() && element.port_output_caps(1).is_none(),
            "a one-port element is a demuxer with caps for port 0 alone"
        );
        Self {
            element,
            log_category,
        }
    }

    pub fn map(self, build: impl FnOnce(E) -> E) -> Self {
        Self::wrap(build(self.element), self.log_category)
    }
}

impl<E: MultiOutputElement> AsyncElement for OnePortElement<E> {
    type ProcessFuture<'a>
        = BoxFuture<'a, Result<(), G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        self.element.intercept_caps(upstream_caps)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let accepted = self.element.caps_constraint_as_input();
        let output = self.element.port_output_caps(0);
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| {
            CapsSet::from_alternatives(
                output
                    .iter()
                    .filter(|_| accepted.accepts(input))
                    .cloned()
                    .collect(),
            )
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.element.configure_pipeline(absolute_caps)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move { self.element.process(packet, &mut OnePortSink(out)).await })
    }

    fn input_domains(&self) -> DomainSet {
        self.element.input_domains()
    }

    fn latency(&self) -> LatencyReport {
        self.element.latency()
    }

    fn seek_route(&self) -> Option<ElementSeekRoute> {
        self.element.seek_route()
    }

    fn set_seek(&mut self, app: SeekController, upstream: SeekController) {
        self.element.set_seek(app, upstream);
    }

    fn metadata(&self) -> ElementMetadata {
        self.element.metadata()
    }

    fn properties(&self) -> &'static [PropertySpec] {
        self.element.properties()
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        self.element.set_property(name, value)
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        self.element.get_property(name)
    }

    fn set_instance_name(&mut self, name: String) {
        self.element.set_instance_name(name);
    }

    fn set_log_category(&mut self, category: String) {
        self.element.set_log_category(category);
    }

    fn log_category(&self) -> &'static str {
        self.log_category
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
