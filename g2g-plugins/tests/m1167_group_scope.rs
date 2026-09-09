//! M1167: a `togglerecord group=` name joins only the elements of the same
//! `parse_launch` call, so two unrelated pipelines in one process do not start
//! and stop each other, while an application outside every parse still reaches
//! the group its line built. The same-parse half of the contract, two tee
//! branches joined by one name, is
//! `a_named_group_joins_two_branches_of_a_launch_line`.
//!
//! `default_registry` is `std`-gated, so this file is too: run with
//! `cargo test -p g2g-plugins --features std`.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use std::sync::{Mutex, OnceLock};

use g2g_core::runtime::{current_parse_id, parse_launch, run_graph, LaunchFactory, Registry};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, G2gError, OutputSink, PipelineClock,
    PipelinePacket,
};
use g2g_plugins::registry::default_registry;
use g2g_plugins::togglerecord::RecordGroup;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// What both lines' `num-buffers=` asks for, and the link capacity they run on.
const FRAMES: usize = 4;

/// The parse id each `parseprobe` saw as it was built, newest last.
fn observed_parse_ids() -> &'static Mutex<Vec<u64>> {
    static IDS: OnceLock<Mutex<Vec<u64>>> = OnceLock::new();
    IDS.get_or_init(|| Mutex::new(Vec::new()))
}

/// A sink whose only job is to report the parse it was built for.
#[derive(Default)]
struct ParseProbe;

impl AsyncElement for ParseProbe {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
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

fn registry_with_parseprobe() -> Registry {
    let mut reg = default_registry();
    reg.register_launch(LaunchFactory::new("parseprobe", Vec::new(), || {
        observed_parse_ids()
            .lock()
            .unwrap()
            .push(current_parse_id());
        Box::new(ParseProbe)
    }));
    reg
}

#[tokio::test]
async fn two_parses_do_not_share_a_group_name() {
    let reg = default_registry();
    let recording = parse_launch(
        &reg,
        &format!(
            "videotestsrc num-buffers={FRAMES} ! togglerecord group=m1167-take record=true ! fakesink"
        ),
    )
    .expect("the recording line parses");
    // Parsed while the first line's group is still alive, which is the only way
    // the two could have met.
    let idle = parse_launch(
        &reg,
        &format!("videotestsrc num-buffers={FRAMES} ! togglerecord group=m1167-take ! fakesink"),
    )
    .expect("the second line parses");

    let recorded = run_graph(recording, &ZeroClock, FRAMES)
        .await
        .expect("the recording line runs");
    assert_eq!(
        recorded.frames_consumed, FRAMES as u64,
        "`record=true` records every frame of a raw stream"
    );

    let untouched = run_graph(idle, &ZeroClock, FRAMES)
        .await
        .expect("the second line runs");
    assert_eq!(
        untouched.frames_consumed, 0,
        "the other pipeline's `record=true` must not reach this group"
    );
}

#[tokio::test]
async fn an_application_reaches_a_launch_built_group_by_name() {
    let reg = default_registry();
    let graph = parse_launch(
        &reg,
        &format!("videotestsrc num-buffers={FRAMES} ! togglerecord group=m1167-app ! fakesink"),
    )
    .expect("the group line parses");

    // A running arm owns its element, so the group is the application's only
    // hold on the record flag once the line is built.
    let group = RecordGroup::named("m1167-app");
    group.set_record(true);

    let stats = run_graph(graph, &ZeroClock, FRAMES)
        .await
        .expect("the group line runs");
    assert_eq!(
        stats.frames_consumed, FRAMES as u64,
        "the flag the application set is the one the line's element read"
    );
}

#[test]
fn every_parse_reports_its_own_id() {
    assert_eq!(current_parse_id(), 0, "no parse is running on this thread");

    let reg = registry_with_parseprobe();
    parse_launch(&reg, "videotestsrc num-buffers=1 ! parseprobe").expect("first probe line parses");
    parse_launch(&reg, "videotestsrc num-buffers=1 ! parseprobe")
        .expect("second probe line parses");

    let ids = observed_parse_ids().lock().unwrap().clone();
    assert_eq!(ids.len(), 2, "one probe built per parse");
    assert!(ids[0] != 0 && ids[1] != 0, "a parse is never id 0");
    assert_ne!(ids[0], ids[1], "two parses are two scopes");

    assert_eq!(current_parse_id(), 0, "the scope ended with the parse");
}
