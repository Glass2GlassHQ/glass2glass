//! M110 Matroska / WebM demux, end to end through the DAG runner. A source
//! produces a `Caps::ByteStream{Matroska}` stream (a synthetic WebM with a VP9
//! video and an Opus audio track); two `MkvDemux` instances pick it apart, one
//! selecting video, one audio. Proves the new `ByteStream{Matroska}` caps and
//! the demuxer's video / audio outputs negotiate through the solver.

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

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

// --- minimal synthetic EBML builders (mirror the parser unit tests) ---

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

fn synthetic_webm() -> Vec<u8> {
    let video = {
        let v = [elem(&[0xB0], &uint_body(640)), elem(&[0xBA], &uint_body(360))].concat();
        let body = [elem(&[0xD7], &uint_body(1)), elem(&[0x86], b"V_VP9"), elem(&[0xE0], &v)].concat();
        elem(&[0xAE], &body)
    };
    let audio = {
        let mut a = elem(&[0x9F], &uint_body(2));
        a.extend_from_slice(&elem(&[0xB5], &48_000f32.to_be_bytes()));
        let body = [elem(&[0xD7], &uint_body(2)), elem(&[0x86], b"A_OPUS"), elem(&[0xE1], &a)].concat();
        elem(&[0xAE], &body)
    };
    let tracks = elem(&[0x16, 0x54, 0xAE, 0x6B], &[video, audio].concat());
    let cluster = elem(
        &[0x1F, 0x43, 0xB6, 0x75],
        &[
            elem(&[0xE7], &uint_body(0)),
            elem(&[0xA3], &block(1, 0, &[1, 2, 3])),
            elem(&[0xA3], &block(2, 0, &[4, 5])),
            elem(&[0xA3], &block(1, 20, &[6, 7, 8])),
            elem(&[0xA3], &block(2, 20, &[9, 10])),
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

async fn run_selected(stream: MkvStream) -> u64 {
    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_source(GraphNodeRef::Source(Box::new(MkvSource { bytes: Some(synthetic_webm()) })));
    let demux = graph.add_transform(GraphNodeRef::element(MkvDemux::new().with_stream(stream)));
    let sink = graph.add_sink(GraphNodeRef::element(FakeSink::new()));
    graph.link(src, demux).unwrap();
    graph.link(demux, sink).unwrap();
    run_graph(graph, &ZeroClock, 4).await.expect("webm runs").frames_consumed
}

#[tokio::test]
async fn mkvdemux_selects_video_stream() {
    assert_eq!(run_selected(MkvStream::Vp9).await, 2, "two VP9 frames reached the sink");
}

#[tokio::test]
async fn mkvdemux_selects_audio_stream() {
    // Opus leaves as Caps::Audio, a different output variant than the VP9 video,
    // so this exercises the audio caps negotiating through the solver.
    assert_eq!(run_selected(MkvStream::Opus).await, 2, "two Opus frames reached the sink");
}
