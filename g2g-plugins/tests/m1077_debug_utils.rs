//! M1077 debug utilities: `watchdog` fails a stalled run, `capssetter` rewrites
//! the caps a stream carries, `taginject` posts a hand-written tag list, and
//! `rndbuffersize` re-chunks a byte stream at seeded random cut points.
//!
//! `default_registry` and the watchdog's timer are `std`-gated, so this file is
//! too: run with `cargo test -p g2g-plugins --features std`.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};
use std::time::Duration;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{parse_launch, run_graph, run_source_transform_sink, SourceLoop};
use g2g_core::{
    AsyncElement, Bus, BusMessage, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelineClock,
    PipelinePacket, PropValue, PushOutcome, Tag,
};
use g2g_plugins::capsfilter::parse_caps;
use g2g_plugins::capssetter::CapsSetter;
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::registry::default_registry;
use g2g_plugins::rndbuffersize::RndBufferSize;
use g2g_plugins::taginject::TagInject;
use g2g_plugins::watchdog::Watchdog;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> Poll<Result<PushOutcome, G2gError>> {
        self.packets
            .push(packet_slot.take().expect("poll_push without a packet"));
        Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

impl Collect {
    /// The payload of every data frame pushed.
    fn payloads(&self) -> Vec<Vec<u8>> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => {
                    Some(f.domain.as_system_slice().expect("system frame").to_vec())
                }
                _ => None,
            })
            .collect()
    }

    /// The pts of every data frame pushed.
    fn pts(&self) -> Vec<u64> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f.timing.pts_ns),
                _ => None,
            })
            .collect()
    }

    /// Every caps declared downstream.
    fn caps(&self) -> Vec<Caps> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::CapsChanged(c) => Some(c.clone()),
                _ => None,
            })
            .collect()
    }
}

fn byte_frame(bytes: Vec<u8>, pts_ns: u64, sequence: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            ..FrameTiming::default()
        },
        sequence,
    ))
}

// ---------------------------------------------------------------------------
// watchdog
// ---------------------------------------------------------------------------

/// The watchdog's timeout for the stall test, and a stall comfortably past it.
const STALL_TIMEOUT_MS: u64 = 20;
const STALL_MS: u64 = 200;

/// A source that pushes one buffer, stalls longer than the watchdog allows,
/// then pushes another. The stand-in for a live feed that goes silent.
struct StallingSrc {
    configured: bool,
}

impl StallingSrc {
    fn caps() -> Caps {
        Caps::ByteStream {
            encoding: g2g_core::ByteStreamEncoding::MpegTs,
        }
    }
}

impl SourceLoop for StallingSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(Self::caps()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(Self::caps()))))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new("Stalling source", "Source", "Stalls mid-stream", "g2g")
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for sequence in 0..2 {
                out.push(byte_frame(vec![sequence as u8; 4], 0, sequence))
                    .await?;
                tokio::time::sleep(Duration::from_millis(STALL_MS)).await;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(2)
        })
    }
}

#[tokio::test]
async fn watchdog_fails_a_stalled_run_and_reports_it_on_the_bus() {
    let (bus, handle) = Bus::new(8);
    let mut src = StallingSrc { configured: false };
    let mut dog = Watchdog::new()
        .with_timeout_ms(STALL_TIMEOUT_MS)
        .with_bus(handle);
    let mut sink = FakeSink::new();

    let result = run_source_transform_sink(&mut src, &mut dog, &mut sink, &ZeroClock, 4).await;
    assert_eq!(
        result.err(),
        Some(G2gError::Timeout),
        "a stall longer than `timeout` fails the run"
    );

    let mut messages = Vec::new();
    while let Some(m) = bus.try_recv() {
        messages.push(m);
    }
    assert!(
        messages.contains(&BusMessage::Error(G2gError::Timeout)),
        "the application sees the stall on the bus: {messages:?}"
    );
}

/// The timer half: nothing arrives after the first buffer, so only a task
/// sleeping on the deadline can report the stall.
#[tokio::test]
async fn watchdog_reports_a_stall_with_no_packet_in_sight() {
    let (bus, handle) = Bus::new(8);
    let mut dog = Watchdog::new()
        .with_timeout_ms(STALL_TIMEOUT_MS)
        .with_bus(handle);
    dog.configure_pipeline(&StallingSrc::caps())
        .expect("configure");
    let mut out = Collect::default();
    dog.process(byte_frame(vec![0u8; 4], 0, 0), &mut out)
        .await
        .expect("the first buffer passes");

    tokio::time::sleep(Duration::from_millis(STALL_MS)).await;
    let posted: Vec<BusMessage> = std::iter::from_fn(|| bus.try_recv()).collect();
    assert!(
        posted.contains(&BusMessage::Error(G2gError::Timeout)),
        "the deadline is reported while the stream is still stalled: {posted:?}"
    );
}

#[tokio::test]
async fn watchdog_passes_a_stream_inside_its_timeout() {
    let reg = default_registry();
    let graph = parse_launch(
        &reg,
        "fakesrc num-buffers=3 sizemax=64 ! watchdog timeout=10000 ! fakesink",
    )
    .expect("watchdog pipeline parses");
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("a stream well inside the timeout runs");
    assert_eq!(stats.frames_consumed, 3);
}

// ---------------------------------------------------------------------------
// capssetter
// ---------------------------------------------------------------------------

/// The caps arriving at the setter, and the same caps at 60 fps.
const SOURCE_CAPS: &str = "video/x-raw,format=rgba,width=320,height=240,framerate=30/1";
const SOURCE_CAPS_AT_60: &str = "video/x-raw,format=rgba,width=320,height=240,framerate=60/1";
const FRAMERATE_ONLY: &str = "video/x-raw,framerate=60/1";

/// Drive `setter` with the negotiated caps and one frame, returning what
/// downstream saw.
async fn run_setter(setter: &mut CapsSetter, incoming: &Caps) -> Result<Collect, G2gError> {
    setter.configure_pipeline(incoming)?;
    let mut out = Collect::default();
    setter
        .process(PipelinePacket::CapsChanged(incoming.clone()), &mut out)
        .await?;
    setter
        .process(byte_frame(vec![0u8; 4], 0, 0), &mut out)
        .await?;
    Ok(out)
}

#[tokio::test]
async fn capssetter_overwrites_only_the_fields_its_caps_name() {
    let incoming = parse_caps(SOURCE_CAPS).expect("source caps parse");
    let mut setter = CapsSetter::new();
    setter
        .set_property("caps", PropValue::Str(FRAMERATE_ONLY.into()))
        .expect("caps property");

    let out = run_setter(&mut setter, &incoming).await.expect("rewrite");
    assert_eq!(
        out.caps(),
        vec![parse_caps(SOURCE_CAPS_AT_60).expect("expected caps parse")],
        "the framerate is replaced and the geometry survives"
    );
    assert_eq!(out.payloads().len(), 1, "the data passes through");
}

#[tokio::test]
async fn capssetter_runs_in_a_text_pipeline() {
    let reg = default_registry();
    let graph = parse_launch(
        &reg,
        "videotestsrc num-buffers=2 ! capssetter caps=\"video/x-raw,framerate=60/1\" ! fakesink",
    )
    .expect("capssetter pipeline parses");
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("capssetter pipeline runs");
    assert_eq!(stats.frames_consumed, 2);
}

#[tokio::test]
async fn capssetter_replace_swaps_the_caps_outright() {
    let incoming = parse_caps(SOURCE_CAPS).expect("source caps parse");
    let replacement = "video/x-raw,format=nv12,width=64,height=48,framerate=25/1";
    let mut setter = CapsSetter::new();
    setter
        .set_property("caps", PropValue::Str(replacement.into()))
        .expect("caps property");
    setter
        .set_property("replace", PropValue::Bool(true))
        .expect("replace property");

    let out = run_setter(&mut setter, &incoming).await.expect("rewrite");
    assert_eq!(
        out.caps(),
        vec![parse_caps(replacement).expect("replacement parses")]
    );
}

#[tokio::test]
async fn capssetter_join_rejects_another_media_type() {
    let audio = parse_caps("audio/x-opus,channels=2,rate=48000").expect("audio caps parse");
    let mut setter = CapsSetter::new();
    setter
        .set_property("caps", PropValue::Str(FRAMERATE_ONLY.into()))
        .expect("caps property");
    assert_eq!(
        setter.get_property("join"),
        Some(PropValue::Bool(true)),
        "gst capssetter's default"
    );
    assert_eq!(
        run_setter(&mut setter, &audio).await.err(),
        Some(G2gError::CapsMismatch),
        "video fields have nowhere to go on an audio stream"
    );
}

// ---------------------------------------------------------------------------
// taginject
// ---------------------------------------------------------------------------

const INJECTED_TAGS: &str = "title=\"A Title\",artist=Someone";
const INJECTED_TITLE: &str = "A Title";
const INJECTED_ARTIST: &str = "Someone";

#[tokio::test]
async fn taginject_posts_its_taglist_on_the_bus() {
    let (bus, handle) = Bus::new(8);
    let mut inject = TagInject::new().with_bus(handle);
    inject
        .set_property("tags", PropValue::Str(INJECTED_TAGS.into()))
        .expect("tags property");
    let caps = parse_caps(SOURCE_CAPS).expect("source caps parse");
    inject.configure_pipeline(&caps).expect("configure");

    let mut out = Collect::default();
    inject
        .process(byte_frame(vec![7u8; 4], 0, 0), &mut out)
        .await
        .expect("frame passes");
    assert_eq!(out.payloads().len(), 1, "the data passes through");

    let posted = std::iter::from_fn(|| bus.try_recv())
        .find_map(|m| match m {
            BusMessage::Tag { tags, .. } => Some(tags),
            _ => None,
        })
        .expect("the tags reach the application");
    for expected in [
        Tag::Title(INJECTED_TITLE.into()),
        Tag::Artist(INJECTED_ARTIST.into()),
    ] {
        let found = posted
            .tags()
            .iter()
            .find(|t| t.key() == expected.key())
            .unwrap_or_else(|| panic!("`{}` was injected", expected.key()));
        assert_eq!(
            found.value_string(),
            expected.value_string(),
            "a quoted value keeps its space"
        );
    }
}

#[tokio::test]
async fn taginject_without_a_bus_still_passes_the_stream() {
    let reg = default_registry();
    let graph = parse_launch(
        &reg,
        "fakesrc num-buffers=2 sizemax=64 ! taginject tags=\"title=A Title\" ! fakesink",
    )
    .expect("taginject pipeline parses");
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("taginject pipeline runs");
    assert_eq!(stats.frames_consumed, 2, "no bus, no lost frames");
}

// ---------------------------------------------------------------------------
// rndbuffersize
// ---------------------------------------------------------------------------

/// The chunking the launch line below asks for.
const CHUNK_MIN: u64 = 100;
const CHUNK_MAX: u64 = 500;
const CHUNK_SEED: u64 = 7;
/// What `fakesrc num-buffers=4 sizemax=10000` feeds it.
const INPUT_BUFFERS: u64 = 4;
const INPUT_BUFFER_BYTES: usize = 10_000;

/// Re-chunk `INPUT_BUFFERS` buffers through a freshly built element and report
/// what came out.
async fn run_rechunker() -> Collect {
    let mut chunker = RndBufferSize::new();
    for (name, value) in [("min", CHUNK_MIN), ("max", CHUNK_MAX), ("seed", CHUNK_SEED)] {
        chunker
            .set_property(name, PropValue::Uint(value))
            .unwrap_or_else(|e| panic!("`{name}` property: {e:?}"));
    }
    let caps = Caps::ByteStream {
        encoding: g2g_core::ByteStreamEncoding::MpegTs,
    };
    chunker.configure_pipeline(&caps).expect("configure");

    let mut out = Collect::default();
    for sequence in 0..INPUT_BUFFERS {
        let bytes = (0..INPUT_BUFFER_BYTES)
            .map(|i| (i as u64 + sequence * INPUT_BUFFER_BYTES as u64) as u8)
            .collect();
        chunker
            .process(byte_frame(bytes, sequence, sequence), &mut out)
            .await
            .expect("chunking");
    }
    chunker
        .process(PipelinePacket::Eos, &mut out)
        .await
        .expect("tail flush");
    out
}

#[tokio::test]
async fn rndbuffersize_cuts_the_stream_into_sized_chunks() {
    let out = run_rechunker().await;
    let chunks = out.payloads();
    assert!(chunks.len() > 1, "a 40 kB stream is cut up");

    let total: usize = chunks.iter().map(|c| c.len()).sum();
    assert_eq!(
        total,
        INPUT_BUFFERS as usize * INPUT_BUFFER_BYTES,
        "no byte is lost or duplicated"
    );
    let joined: Vec<u8> = chunks.concat();
    let expected: Vec<u8> = (0..total).map(|i| i as u8).collect();
    assert_eq!(joined, expected, "the bytes keep their order");

    let (tail, whole) = chunks.split_last().expect("at least one chunk");
    for chunk in whole {
        assert!(
            (CHUNK_MIN as usize..=CHUNK_MAX as usize).contains(&chunk.len()),
            "chunk of {} bytes is outside [{CHUNK_MIN}, {CHUNK_MAX}]",
            chunk.len()
        );
    }
    assert!(
        tail.len() <= CHUNK_MAX as usize,
        "the tail flushed at EOS may be short, never long"
    );

    let pts = out.pts();
    assert!(
        pts.windows(2).all(|w| w[0] <= w[1]),
        "each chunk carries the pts of the buffer it was cut from"
    );
    assert_eq!(
        pts.last(),
        Some(&(INPUT_BUFFERS - 1)),
        "the tail carries the last input buffer's pts"
    );
}

#[tokio::test]
async fn rndbuffersize_repeats_its_cuts_on_the_same_seed() {
    let first: Vec<usize> = run_rechunker()
        .await
        .payloads()
        .iter()
        .map(|c| c.len())
        .collect();
    let second: Vec<usize> = run_rechunker()
        .await
        .payloads()
        .iter()
        .map(|c| c.len())
        .collect();
    assert_eq!(first, second, "the seed fixes the cut points");
}

#[tokio::test]
async fn rndbuffersize_runs_in_a_text_pipeline() {
    let expected_chunks = run_rechunker().await.payloads().len();
    let reg = default_registry();
    let graph = parse_launch(
        &reg,
        "fakesrc num-buffers=4 sizemax=10000 ! rndbuffersize min=100 max=500 seed=7 ! fakesink",
    )
    .expect("rndbuffersize pipeline parses");
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("rndbuffersize pipeline runs");
    assert_eq!(
        stats.frames_consumed as usize, expected_chunks,
        "the launch line chunks the same stream the same way"
    );
}
