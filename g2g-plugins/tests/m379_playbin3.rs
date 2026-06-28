//! M379 - `playbin3`-equivalent graph builder. `Registry::build_playbin3_graph`
//! assembles `source(uri) -> demux(MkvDemuxN) -> {decode chain -> sink}` per
//! selected stream: the multi-stream counterpart of `build_uridecodebin`. The app
//! derives one `Playbin3Port` per stream from the demux's announced
//! `StreamCollection` (M376) + its selection (M377).
//!
//! Two checks: the builder assembles the right topology (structural, the
//! playbin-family convention, since a full negotiated run of real decoders through
//! a demux branch is the per-branch-negotiation follow-up), and `MkvDemuxN` runs as
//! a first-class `run_graph` demux node feeding per-port branches end-to-end.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{
    is_raw_audio, is_raw_video, run_graph, ElementFactory, GraphNode, Playbin3Error, Playbin3Port,
    Registry, SourceLoop, UriError, UriSourceFactory,
};
use g2g_core::{
    AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, Graph, MultiInputElement,
    OutputSink, PadTemplate, PipelineClock, PushOutcome, RawVideoFormat, Rate, VideoCodec,
};
use g2g_plugins::mkvdemux::{MkvDemuxN, MkvStream};
use g2g_plugins::mkvmuxn::MkvMuxN;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn h264_any() -> Caps {
    Caps::CompressedVideo { codec: VideoCodec::H264, width: Dim::Any, height: Dim::Any, framerate: Rate::Any }
}
fn aac_any() -> Caps {
    Caps::Audio { format: AudioFormat::Aac, channels: 0, sample_rate: 0 }
}
fn raw_video() -> Caps {
    Caps::RawVideo { format: RawVideoFormat::Nv12, width: Dim::Any, height: Dim::Any, framerate: Rate::Any }
}
fn raw_audio() -> Caps {
    Caps::Audio { format: AudioFormat::PcmS16Le, channels: 0, sample_rate: 0 }
}

/// A `mem://` URI source stand-in (its identity is irrelevant to graph assembly,
/// which never runs it): returns a placeholder source + the container byte caps.
fn mem_uri_build(
    _uri: &g2g_core::runtime::Uri,
) -> Result<(Box<dyn g2g_core::runtime::DynSourceLoop>, Caps), UriError> {
    Ok((
        Box::new(g2g_plugins::videotestsrc::VideoTestSrc::new(8, 8, 30, 1)),
        Caps::ByteStream { encoding: g2g_core::ByteStreamEncoding::Matroska },
    ))
}

/// A registry with the `mem://` URI handler and stub H.264 / AAC decoders (so the
/// per-port auto-plug reaches raw), mirroring m196's `h264stub`.
fn registry_with_stubs() -> Registry {
    let mut reg = Registry::new();
    reg.register_uri(UriSourceFactory::new("mem", mem_uri_build));
    reg.register(ElementFactory::new(
        "h264stub",
        Vec::from([PadTemplate::sink(CapsSet::one(h264_any())), PadTemplate::source(CapsSet::one(raw_video()))]),
        |_| Box::new(g2g_plugins::identity::IdentityTransform::new()),
    ));
    reg.register(ElementFactory::new(
        "aacstub",
        Vec::from([PadTemplate::sink(CapsSet::one(aac_any())), PadTemplate::source(CapsSet::one(raw_audio()))]),
        |_| Box::new(g2g_plugins::identity::IdentityTransform::new()),
    ));
    reg
}

/// A trivial accept-anything sink, for both the structural-build branches and the
/// end-to-end run (it tolerates the demux's per-port retyping CapsChanged).
#[derive(Default)]
struct CountSink {
    frames: Arc<AtomicUsize>,
}
impl AsyncElement for CountSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;
    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        if let PipelinePacket::DataFrame(_) = packet {
            self.frames.fetch_add(1, Ordering::Relaxed);
        }
        Box::pin(async { Ok(()) })
    }
}

#[test]
fn build_playbin3_graph_assembles_per_stream_decode_branches() {
    let reg = registry_with_stubs();
    let demux = MkvDemuxN::new(vec![MkvStream::H264, MkvStream::Aac]);
    let ports = vec![
        Playbin3Port { input_caps: h264_any(), target: Box::new(is_raw_video), sink: Box::new(CountSink::default()) },
        Playbin3Port { input_caps: aac_any(), target: Box::new(is_raw_audio), sink: Box::new(CountSink::default()) },
    ];

    let graph = reg
        .build_playbin3_graph("mem://clip.mkv", demux, ports, 6)
        .expect("playbin3 graph assembles");

    // source -> demux(2); each port: demux.out(i) -> stub decoder -> sink.
    // Nodes: source + demux + 2 decoders + 2 sinks = 6.
    assert_eq!(graph.node_count(), 6, "source, demux, two decoders, two sinks");
    // Edges: src->demux, demux->dec0, dec0->sink0, demux->dec1, dec1->sink1 = 5.
    assert_eq!(graph.edges().len(), 5, "one decode branch per selected stream");
}

#[test]
fn build_playbin3_graph_rejects_no_ports() {
    let reg = registry_with_stubs();
    let demux = MkvDemuxN::new(vec![MkvStream::H264]);
    let err = reg.build_playbin3_graph("mem://clip.mkv", demux, Vec::new(), 6).unwrap_err();
    assert!(matches!(err, Playbin3Error::NoPorts), "got {err:?}");
}

#[test]
fn build_playbin3_graph_rejects_unknown_scheme() {
    let reg = registry_with_stubs();
    let demux = MkvDemuxN::new(vec![MkvStream::H264]);
    let ports = vec![Playbin3Port {
        input_caps: h264_any(),
        target: Box::new(is_raw_video),
        sink: Box::new(CountSink::default()),
    }];
    let err = reg.build_playbin3_graph("bogus://x", demux, ports, 6).unwrap_err();
    assert!(matches!(err, Playbin3Error::Uri(UriError::UnknownScheme)), "got {err:?}");
}

// --- end-to-end: MkvDemuxN runs as a run_graph demux node feeding two branches ---

/// A one-shot byte source emitting `bytes` as a single `ByteStream{Matroska}`
/// frame, then EOS. The container source half of the playbin3 graph (the URI
/// handler's real job; here constructed directly so the test owns the bytes).
#[derive(Debug)]
struct BytesSrc {
    bytes: Option<Vec<u8>>,
}
impl SourceLoop for BytesSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>;
    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(Caps::ByteStream { encoding: g2g_core::ByteStreamEncoding::Matroska }))
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let bytes = self.bytes.take().unwrap_or_default();
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                FrameTiming::default(),
                0,
            );
            out.push(PipelinePacket::DataFrame(frame)).await?;
            out.push(PipelinePacket::Eos).await?;
            Ok(1)
        })
    }
}

// --- A/V mux fixture (shared shape with m294 / m377 / m378) ---
#[derive(Default)]
struct Collect {
    bytes: Vec<u8>,
}
impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let MemoryDomain::System(s) = &f.domain {
                    self.bytes.extend_from_slice(s.as_slice());
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}
fn frame(data: Vec<u8>, pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        FrameTiming { pts_ns, dts_ns: pts_ns, ..FrameTiming::default() },
        0,
    ))
}
fn annexb(nals: &[&[u8]]) -> Vec<u8> {
    let mut v = Vec::new();
    for n in nals {
        v.extend_from_slice(&[0, 0, 0, 1]);
        v.extend_from_slice(n);
    }
    v
}
fn adts_au(payload: &[u8]) -> Vec<u8> {
    let frame_len = payload.len() + 7;
    let mut au = vec![
        0xFF,
        0xF1,
        (1 << 6) | (3 << 2),
        ((2 & 3) << 6) | ((frame_len >> 11) & 3) as u8,
        ((frame_len >> 3) & 0xFF) as u8,
        (((frame_len & 7) << 5) as u8) | 0x1F,
        0xFC,
    ];
    au.extend_from_slice(payload);
    au
}
async fn mux_av() -> Vec<u8> {
    let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
    let pps = [0x68u8, 0xce, 0x3c, 0x80];
    let idr = [0x65u8, 0x88, 0x84, 0x00];
    let mut mux = MkvMuxN::new(2);
    mux.configure_pipeline(0, &h264_any()).unwrap();
    mux.configure_pipeline(1, &Caps::Audio { format: AudioFormat::Aac, channels: 2, sample_rate: 48_000 }).unwrap();
    let mut sink = Collect::default();
    mux.process(0, frame(annexb(&[&sps, &pps, &idr]), 0), &mut sink).await.unwrap();
    mux.process(1, frame(adts_au(&[0xA1, 0xA2, 0xA3]), 0), &mut sink).await.unwrap();
    mux.process(0, frame(annexb(&[&[0x41u8, 0x9a, 0x00]]), 33_000_000), &mut sink).await.unwrap();
    mux.process(1, frame(adts_au(&[0xB4, 0xB5]), 21_000_000), &mut sink).await.unwrap();
    mux.process(0, PipelinePacket::Eos, &mut sink).await.unwrap();
    mux.process(1, PipelinePacket::Eos, &mut sink).await.unwrap();
    sink.bytes
}

#[tokio::test]
async fn mkvdemuxn_runs_as_a_graph_demux_node() {
    let file = mux_av().await;
    let video = Arc::new(AtomicUsize::new(0));
    let audio = Arc::new(AtomicUsize::new(0));

    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(BytesSrc { bytes: Some(file) }));
    let demux = g.add_demux(GraphNode::demux(MkvDemuxN::new(vec![MkvStream::H264, MkvStream::Aac])), 2);
    let s0 = g.add_sink(GraphNode::element(CountSink { frames: video.clone() }));
    let s1 = g.add_sink(GraphNode::element(CountSink { frames: audio.clone() }));
    g.link(src, demux.input()).unwrap();
    g.link(demux.out(0), s0).unwrap();
    g.link(demux.out(1), s1).unwrap();

    let stats = run_graph(g, &NullClock, 4).await.expect("playbin3 demux graph runs");
    assert_eq!(stats.frames_consumed, 4, "all four access units reached a branch");
    assert_eq!(video.load(Ordering::Relaxed), 2, "two H.264 access units on the video port");
    assert_eq!(audio.load(Ordering::Relaxed), 2, "two AAC packets on the audio port");
}
