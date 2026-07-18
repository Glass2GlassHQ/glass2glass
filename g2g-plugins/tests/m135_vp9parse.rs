//! M135 VP9 parser, end to end through the DAG runner: a source emits a
//! synthetic WebM with a VP9 video track, `mkvdemux` selects it (the default),
//! and `vp9parse` refines the caps from the keyframe before the sink. The
//! uncompressed-header decode itself is unit-tested in `vp9parse`.
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
use g2g_plugins::vp9parse::Vp9Parse;

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

/// A VP9 profile-0 key-frame uncompressed header (BT.709, 4:2:0) for
/// `width` x `height`, MSB-first, so `vp9parse` recovers geometry from real
/// bits.
fn vp9_keyframe(width: u32, height: u32) -> Vec<u8> {
    let mut bits: Vec<u8> = Vec::new();
    let mut push = |value: u32, n: u32| {
        for i in (0..n).rev() {
            bits.push(((value >> i) & 1) as u8);
        }
    };
    push(2, 2); // frame_marker
    push(0, 1); // profile_low_bit
    push(0, 1); // profile_high_bit (profile 0)
    push(0, 1); // show_existing_frame
    push(0, 1); // frame_type = KEY_FRAME
    push(1, 1); // show_frame
    push(0, 1); // error_resilient_mode
    push(0x49, 8); // frame_sync_code
    push(0x83, 8);
    push(0x42, 8);
    push(2, 3); // color_space = CS_BT_709
    push(0, 1); // color_range
    push(width - 1, 16); // frame_width_minus_1
    push(height - 1, 16); // frame_height_minus_1

    let mut out = vec![0u8; bits.len().div_ceil(8)];
    for (i, b) in bits.iter().enumerate() {
        out[i / 8] |= b << (7 - (i % 8));
    }
    out
}

fn synthetic_vp9_webm() -> Vec<u8> {
    let video = {
        let v = [
            elem(&[0xB0], &uint_body(1280)),
            elem(&[0xBA], &uint_body(720)),
        ]
        .concat();
        let body = [
            elem(&[0xD7], &uint_body(1)),
            elem(&[0x86], b"V_VP9"),
            elem(&[0xE0], &v),
        ]
        .concat();
        elem(&[0xAE], &body)
    };
    let tracks = elem(&[0x16, 0x54, 0xAE, 0x6B], &video);
    let interframe = {
        let mut f = vp9_keyframe(1280, 720);
        f[0] |= 0x04; // set frame_type (bit 2 of byte 0) to NON_KEY_FRAME
        f
    };
    let cluster = elem(
        &[0x1F, 0x43, 0xB6, 0x75],
        &[
            elem(&[0xE7], &uint_body(0)),
            elem(&[0xA3], &block(1, 0, &vp9_keyframe(1280, 720))),
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
        core::future::ready(Ok(Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        }))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(
            Caps::ByteStream {
                encoding: ByteStreamEncoding::Matroska,
            },
        ))))
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
async fn mkvdemux_feeds_vp9parse_end_to_end() {
    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_source(GraphNodeRef::Source(Box::new(MkvSource {
        bytes: Some(synthetic_vp9_webm()),
    })));
    let demux = graph.add_transform(GraphNodeRef::element(
        MkvDemux::new().with_stream(MkvStream::Vp9),
    ));
    let parse = graph.add_transform(GraphNodeRef::element(Vp9Parse::new()));
    let sink = graph.add_sink(GraphNodeRef::element(FakeSink::new()));
    graph.link(src, demux).unwrap();
    graph.link(demux, parse).unwrap();
    graph.link(parse, sink).unwrap();

    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("vp9 pipeline runs");
    assert_eq!(
        stats.frames_consumed, 2,
        "keyframe + interframe pass through the parser to the sink"
    );
}

#[cfg(feature = "std")]
#[test]
fn vp9parse_registered_and_constructable() {
    use g2g_plugins::registry::default_registry;
    let reg = default_registry();
    assert!(
        reg.inspect("vp9parse").is_some(),
        "vp9parse joins the default registry"
    );
    assert!(
        reg.make_element("vp9parse").is_some(),
        "vp9parse builds by name"
    );
}
