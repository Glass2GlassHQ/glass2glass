//! Gap-filling elements end to end through `parse_launch` + `run_graph`: the new
//! audio DSP (`audioamplify` / `level` / `audioecho` / `cutter`), video utility
//! (`gamma` / `deinterlace` / `timeoverlay` / `progressreport`), flow-control
//! (`concat` / `input-selector`), and file (`multifilesink` / `multifilesrc`)
//! elements are registered in `default_registry` and run in text pipelines with
//! their properties parsed from text. The per-element math is unit-tested in each
//! module; this proves the registry wiring and the runner path.

use core::future::Future;
use core::pin::Pin;

use g2g_core::runtime::{parse_launch, run_graph, LaunchFactory, Registry};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, G2gError, OutputSink, PipelineClock,
    PipelinePacket,
};
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// A sink that accepts any packet without `FakeSink`'s monotonic-sequence
/// assertion, so a fan-in element (concat / selector) whose inputs have
/// overlapping sequence numbers is tolerated.
#[derive(Default)]
struct AnySink;

impl AsyncElement for AnySink {
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

fn registry_with_anysink() -> Registry {
    let mut reg = default_registry();
    reg.register_launch(LaunchFactory::new("anysink", Vec::new(), || Box::new(AnySink)));
    reg
}

#[tokio::test]
async fn audio_dsp_chain_runs() {
    let reg = default_registry();
    let graph = parse_launch(
        &reg,
        "audiotestsrc num-buffers=5 ! audioamplify amplification=2.0 amplification-method=clip \
         ! level ! audioecho delay=100000000 intensity=0.4 ! cutter threshold=0.01 ! fakesink",
    )
    .expect("audio DSP pipeline parses");
    let stats = run_graph(graph, &ZeroClock, 4).await.expect("audio DSP pipeline runs");
    assert_eq!(stats.frames_consumed, 5, "all buffers reached the sink");
}

#[tokio::test]
async fn video_utility_chain_runs() {
    let reg = default_registry();
    let graph = parse_launch(
        &reg,
        "videotestsrc num-buffers=4 ! gamma gamma=2.2 ! deinterlace method=linear \
         ! timeoverlay scale=2 ! progressreport silent=true ! fakesink",
    )
    .expect("video utility pipeline parses");
    let stats = run_graph(graph, &ZeroClock, 4).await.expect("video utility pipeline runs");
    assert_eq!(stats.frames_consumed, 4, "all frames reached the sink");
}

#[tokio::test]
async fn concat_plays_two_sources_end_to_end() {
    let reg = registry_with_anysink();
    let graph = parse_launch(
        &reg,
        "videotestsrc num-buffers=4 ! c.   videotestsrc num-buffers=3 ! c.   concat name=c ! anysink",
    )
    .expect("concat pipeline parses");
    let stats = run_graph(graph, &ZeroClock, 4).await.expect("concat pipeline runs");
    assert_eq!(stats.frames_consumed, 7, "4 + 3 frames played in sequence");
}

#[tokio::test]
async fn input_selector_forwards_only_the_active_pad() {
    let reg = registry_with_anysink();
    // active-pad defaults to 0, so only the first source's 4 frames pass; the
    // second source's 3 are dropped.
    let graph = parse_launch(
        &reg,
        "videotestsrc num-buffers=4 ! s.   videotestsrc num-buffers=3 ! s.   input-selector name=s ! anysink",
    )
    .expect("input-selector pipeline parses");
    let stats = run_graph(graph, &ZeroClock, 4).await.expect("input-selector pipeline runs");
    assert_eq!(stats.frames_consumed, 4, "only the active input reached the sink");
}

#[tokio::test]
async fn multifilesink_then_multifilesrc_round_trips() {
    let dir = std::env::temp_dir();
    let pat = dir.join("g2g_gap_seq_%03d.raw").to_string_lossy().into_owned();
    for i in 0..3 {
        let _ = std::fs::remove_file(pat.replace("%03d", &format!("{i:03}")));
    }

    // Write three raw frames, one file each (buffer mode is the default).
    let reg = default_registry();
    let write = parse_launch(
        &reg,
        &format!("videotestsrc num-buffers=3 ! multifilesink location={pat}"),
    )
    .expect("write pipeline parses");
    let stats = run_graph(write, &ZeroClock, 4).await.expect("write pipeline runs");
    assert_eq!(stats.frames_consumed, 3, "three buffers written");
    for i in 0..3 {
        assert!(
            std::path::Path::new(&pat.replace("%03d", &format!("{i:03}"))).exists(),
            "file {i} written"
        );
    }

    // Read the three files back as a raw byte sequence.
    let read = parse_launch(
        &reg,
        &format!("multifilesrc location={pat} stop-index=2 ! fakesink"),
    )
    .expect("read pipeline parses");
    let stats = run_graph(read, &ZeroClock, 4).await.expect("read pipeline runs");
    assert_eq!(stats.frames_consumed, 3, "three files read back");

    for i in 0..3 {
        let _ = std::fs::remove_file(pat.replace("%03d", &format!("{i:03}")));
    }
}
