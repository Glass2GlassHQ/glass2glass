//! M134 VP8 parser, end to end through the DAG runner: a source emits a
//! synthetic WebM with a VP8 video track, `mkvdemux` selects it, and `vp8parse`
//! refines the caps from the keyframe before the sink. Exercises the parser
//! inside a real pipeline downstream of the demuxer that produces VP8; the
//! keyframe-header decode itself is unit-tested in `vp8parse`.
//!
//! `default_registry` is `std`-gated; the graph half builds elements directly so
//! it runs on the no_std-baseline test build too.

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_graph, GraphNode, GraphNodeRef, SourceLoop};
use g2g_core::{
    ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, G2gError, Graph,
    MemoryDomain, OutputSink, PipelineClock, PipelinePacket,
};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::mkvdemux::{MkvDemux, MkvStream};
use g2g_plugins::vp8parse::Vp8Parse;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

// --- minimal synthetic EBML builders (mirror the mkvdemux unit tests) ---

fn vint(value: u64) -> Vec<u8> {
    let mut len = 1usize;
    while len < 8 && value >= (1u64 << (7 * len)) - 1 {
        len += 1;
    }
    let mut out = vec![0u8; len];
    let mut v = value;
    for i in (0..len).rev() {
        out[i] = (v & 0xFF) as u8;
        v >>= 8;
    }
    out[0] |= 1 << (8 - len);
    out
}

fn elem(id: &[u8], body: &[u8]) -> Vec<u8> {
    let mut out = id.to_vec();
    out.extend_from_slice(&vint(body.len() as u64));
    out.extend_from_slice(body);
    out
}

fn uint_body(v: u64) -> Vec<u8> {
    if v == 0 {
        return vec![0];
    }
    let mut bytes = v.to_be_bytes().to_vec();
    while bytes.len() > 1 && bytes[0] == 0 {
        bytes.remove(0);
    }
    bytes
}

fn block(track: u64, rel: i16, frame: &[u8]) -> Vec<u8> {
    let mut b = vint(track);
    b.extend_from_slice(&rel.to_be_bytes());
    b.push(0x80); // keyframe, no lacing
    b.extend_from_slice(frame);
    b
}

/// A VP8 key-frame header (RFC 6386 §9.1) for `width` x `height`, plus a short
/// dummy partition payload, so `vp8parse` recovers geometry from real bytes.
fn vp8_keyframe(width: u32, height: u32) -> Vec<u8> {
    vec![
        0x00, 0x00, 0x00, // frame tag: key frame, version 0
        0x9d, 0x01, 0x2a, // start code
        (width & 0xFF) as u8,
        ((width >> 8) & 0x3F) as u8,
        (height & 0xFF) as u8,
        ((height >> 8) & 0x3F) as u8,
        0x00, 0x00, 0x00, 0x00,
    ]
}

/// EBML header + Segment{Tracks(V_VP8), Cluster{ two video blocks }}.
fn synthetic_vp8_webm() -> Vec<u8> {
    let video = {
        let v = [elem(&[0xB0], &uint_body(640)), elem(&[0xBA], &uint_body(360))].concat();
        let body = [elem(&[0xD7], &uint_body(1)), elem(&[0x86], b"V_VP8"), elem(&[0xE0], &v)].concat();
        elem(&[0xAE], &body)
    };
    let tracks = elem(&[0x16, 0x54, 0xAE, 0x6B], &video);
    let interframe = {
        let mut f = vp8_keyframe(640, 360);
        f[0] |= 0x1; // flip to an interframe (no dimensions)
        f
    };
    let cluster = elem(
        &[0x1F, 0x43, 0xB6, 0x75],
        &[
            elem(&[0xE7], &uint_body(0)),
            elem(&[0xA3], &block(1, 0, &vp8_keyframe(640, 360))),
            elem(&[0xA3], &block(1, 20, &interframe)),
        ]
        .concat(),
    );
    let segment = elem(&[0x18, 0x53, 0x80, 0x67], &[tracks, cluster].concat());
    [elem(&[0x1A, 0x45, 0xDF, 0xA3], &[]), segment].concat()
}

/// Emits the whole synthetic WebM as one `ByteStream{Matroska}` frame.
struct MkvSource {
    bytes: Option<Vec<u8>>,
}
impl SourceLoop for MkvSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(Caps::ByteStream { encoding: ByteStreamEncoding::Matroska }))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        }))))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let bytes = self.bytes.take().expect("run once");
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                Default::default(),
                0,
            );
            out.push(PipelinePacket::DataFrame(frame)).await?;
            out.push(PipelinePacket::Eos).await?;
            Ok(1)
        })
    }
}

#[tokio::test]
async fn mkvdemux_feeds_vp8parse_end_to_end() {
    let mut graph: Graph<GraphNode> = Graph::new();
    let src =
        graph.add_source(GraphNodeRef::Source(Box::new(MkvSource { bytes: Some(synthetic_vp8_webm()) })));
    let demux = graph.add_transform(GraphNodeRef::element(MkvDemux::new().with_stream(MkvStream::Vp8)));
    let parse = graph.add_transform(GraphNodeRef::element(Vp8Parse::new()));
    let sink = graph.add_sink(GraphNodeRef::element(FakeSink::new()));
    graph.link(src, demux).unwrap();
    graph.link(demux, parse).unwrap();
    graph.link(parse, sink).unwrap();

    let stats = run_graph(graph, &ZeroClock, 4).await.expect("vp8 pipeline runs");
    assert_eq!(stats.frames_consumed, 2, "keyframe + interframe pass through the parser to the sink");
}

#[cfg(feature = "std")]
#[test]
fn vp8parse_registered_and_constructable() {
    use g2g_plugins::registry::default_registry;
    let reg = default_registry();
    assert!(reg.inspect("vp8parse").is_some(), "vp8parse joins the default registry");
    assert!(reg.make_element("vp8parse").is_some(), "vp8parse builds by name");
}
