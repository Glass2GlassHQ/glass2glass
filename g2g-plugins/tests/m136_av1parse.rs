//! M136 AV1 parser, end to end through the DAG runner: a source emits a
//! synthetic WebM with a V_AV1 track, `mkvdemux` selects it, and `av1parse`
//! refines the caps from the sequence-header OBU before the sink. The OBU walk
//! and sequence-header decode are unit-tested in `av1parse`.
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
use g2g_plugins::av1parse::Av1Parse;
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::mkvdemux::{MkvDemux, MkvStream};

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

// --- minimal AV1 OBU builders ---

fn leb128(mut v: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let mut byte = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if v == 0 {
            break;
        }
    }
    out
}

fn bits_for(v: u32) -> u32 {
    if v == 0 {
        1
    } else {
        32 - v.leading_zeros()
    }
}

/// A non-reduced sequence-header OBU payload (one operating point, no timing /
/// decoder-model / display-delay), MSB-first.
fn seq_header_payload(width: u32, height: u32, profile: u32) -> Vec<u8> {
    let mut bits: Vec<u8> = Vec::new();
    let mut push = |value: u32, n: u32| {
        for i in (0..n).rev() {
            bits.push(((value >> i) & 1) as u8);
        }
    };
    push(profile, 3); // seq_profile
    push(0, 1); // still_picture
    push(0, 1); // reduced_still_picture_header
    push(0, 1); // timing_info_present_flag
    push(0, 1); // initial_display_delay_present_flag
    push(0, 5); // operating_points_cnt_minus_1
    push(0, 12); // operating_point_idc[0]
    push(0, 5); // seq_level_idx[0]
    let (wm1, hm1) = (width - 1, height - 1);
    let (nw, nh) = (bits_for(wm1), bits_for(hm1));
    push(nw - 1, 4); // frame_width_bits_minus_1
    push(nh - 1, 4); // frame_height_bits_minus_1
    push(wm1, nw); // max_frame_width_minus_1
    push(hm1, nh); // max_frame_height_minus_1

    let mut out = vec![0u8; bits.len().div_ceil(8)];
    for (i, b) in bits.iter().enumerate() {
        out[i / 8] |= b << (7 - (i % 8));
    }
    out
}

/// Wrap `payload` as a size-delimited OBU of `obu_type`.
fn obu(obu_type: u8, payload: &[u8]) -> Vec<u8> {
    let header = (obu_type << 3) | 0x02; // ext_flag=0, has_size_field=1
    let mut out = vec![header];
    out.extend_from_slice(&leb128(payload.len() as u64));
    out.extend_from_slice(payload);
    out
}

/// Temporal delimiter + sequence-header OBU.
fn av1_keyframe_unit(width: u32, height: u32) -> Vec<u8> {
    let mut tu = obu(2, &[]); // OBU_TEMPORAL_DELIMITER
    tu.extend_from_slice(&obu(1, &seq_header_payload(width, height, 0))); // OBU_SEQUENCE_HEADER
    tu
}

/// Temporal delimiter + a frame OBU only (no sequence header).
fn av1_inter_unit() -> Vec<u8> {
    let mut tu = obu(2, &[]);
    tu.extend_from_slice(&obu(6, &[0xAA, 0xBB, 0xCC])); // OBU_FRAME
    tu
}

fn synthetic_av1_webm() -> Vec<u8> {
    let video = {
        let v = [
            elem(&[0xB0], &uint_body(1920)),
            elem(&[0xBA], &uint_body(1080)),
        ]
        .concat();
        let body = [
            elem(&[0xD7], &uint_body(1)),
            elem(&[0x86], b"V_AV1"),
            elem(&[0xE0], &v),
        ]
        .concat();
        elem(&[0xAE], &body)
    };
    let tracks = elem(&[0x16, 0x54, 0xAE, 0x6B], &video);
    let cluster = elem(
        &[0x1F, 0x43, 0xB6, 0x75],
        &[
            elem(&[0xE7], &uint_body(0)),
            elem(&[0xA3], &block(1, 0, &av1_keyframe_unit(1920, 1080))),
            elem(&[0xA3], &block(1, 20, &av1_inter_unit())),
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
async fn mkvdemux_feeds_av1parse_end_to_end() {
    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_source(GraphNodeRef::Source(Box::new(MkvSource {
        bytes: Some(synthetic_av1_webm()),
    })));
    let demux = graph.add_transform(GraphNodeRef::element(
        MkvDemux::new().with_stream(MkvStream::Av1),
    ));
    let parse = graph.add_transform(GraphNodeRef::element(Av1Parse::new()));
    let sink = graph.add_sink(GraphNodeRef::element(FakeSink::new()));
    graph.link(src, demux).unwrap();
    graph.link(demux, parse).unwrap();
    graph.link(parse, sink).unwrap();

    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("av1 pipeline runs");
    assert_eq!(
        stats.frames_consumed, 2,
        "keyframe + inter unit pass through the parser to the sink"
    );
}

#[cfg(feature = "std")]
#[test]
fn av1parse_registered_and_constructable() {
    use g2g_plugins::registry::default_registry;
    let reg = default_registry();
    assert!(
        reg.inspect("av1parse").is_some(),
        "av1parse joins the default registry"
    );
    assert!(
        reg.make_element("av1parse").is_some(),
        "av1parse builds by name"
    );
}
