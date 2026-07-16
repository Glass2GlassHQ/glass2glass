//! M108 MPEG-TS demux, end to end through the DAG runner: a source that produces
//! a `Caps::ByteStream{MpegTs}` stream feeds `TsDemux`, which negotiates an H.264
//! output and forwards the demuxed access units to a sink. Proves the new
//! `ByteStream` caps variant flows through the solver and the demuxer runs in a
//! real pipeline.

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
use g2g_plugins::mpegts::{STREAM_TYPE_H264, TS_PACKET_LEN};
use g2g_plugins::tsdemux::TsDemux;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

// --- minimal synthetic MPEG-TS builders (mirror the unit tests) ---

fn ts_packet(pid: u16, pusi: bool, payload: &[u8]) -> Vec<u8> {
    const ROOM: usize = TS_PACKET_LEN - 4;
    let mut p = vec![0u8; TS_PACKET_LEN];
    p[0] = 0x47;
    p[1] = if pusi { 0x40 } else { 0x00 } | ((pid >> 8) as u8 & 0x1F);
    p[2] = (pid & 0xFF) as u8;
    let l = payload.len();
    if l == ROOM {
        p[3] = 0x10;
        p[4..].copy_from_slice(payload);
    } else {
        p[3] = 0x30;
        let af_len = ROOM - 1 - l;
        p[4] = af_len as u8;
        if af_len >= 1 {
            p[5] = 0x00;
            for b in p.iter_mut().take(6 + (af_len - 1)).skip(6) {
                *b = 0xFF;
            }
        }
        p[5 + af_len..].copy_from_slice(payload);
    }
    p
}

fn psi(pid: u16, table_id: u8, body: &[u8]) -> Vec<u8> {
    let section_length = body.len() + 4;
    let mut s = vec![table_id, 0xB0 | ((section_length >> 8) as u8 & 0x0F), (section_length & 0xFF) as u8];
    s.extend_from_slice(body);
    s.extend_from_slice(&[0, 0, 0, 0]);
    let mut payload = vec![0u8];
    payload.extend_from_slice(&s);
    ts_packet(pid, true, &payload)
}

fn pes(es: &[u8]) -> Vec<u8> {
    let mut p = vec![0x00, 0x00, 0x01, 0xE0];
    let header = [0x80u8, 0x00, 0x00];
    let len = header.len() + es.len();
    p.push((len >> 8) as u8);
    p.push((len & 0xFF) as u8);
    p.extend_from_slice(&header);
    p.extend_from_slice(es);
    p
}

fn synthetic_ts() -> Vec<u8> {
    let pmt_pid = 0x1000u16;
    let es_pid = 0x0100u16;
    let mut s = Vec::new();
    s.extend_from_slice(&psi(
        0x0000,
        0x00,
        &[0, 1, 0xC1, 0, 0, 0, 1, 0xE0 | (pmt_pid >> 8) as u8 & 0x1F, pmt_pid as u8],
    ));
    s.extend_from_slice(&psi(
        pmt_pid,
        0x02,
        &[
            0x00, 0x01, 0xC1, 0x00, 0x00,
            0xE0 | (es_pid >> 8) as u8 & 0x1F, es_pid as u8,
            0xF0, 0x00,
            STREAM_TYPE_H264,
            0xE0 | (es_pid >> 8) as u8 & 0x1F, es_pid as u8,
            0xF0, 0x00,
        ],
    ));
    // Three H.264 access units, each its own PES.
    for n in 0..3u8 {
        s.extend_from_slice(&ts_packet(es_pid, true, &pes(&[0, 0, 0, 1, 0x65, n])));
    }
    s
}

/// A source that emits the whole synthetic TS as one `ByteStream{MpegTs}` frame.
struct TsSource {
    bytes: Option<Vec<u8>>,
}
impl SourceLoop for TsSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(Caps::ByteStream { encoding: ByteStreamEncoding::MpegTs }))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
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
async fn tsdemux_negotiates_bytestream_and_demuxes_in_runner() {
    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_source(GraphNodeRef::Source(Box::new(TsSource {
        bytes: Some(synthetic_ts()),
    })));
    let demux = graph.add_transform(GraphNodeRef::element(TsDemux::new()));
    let sink = graph.add_sink(GraphNodeRef::element(FakeSink::new()));
    graph.link(src, demux).unwrap();
    graph.link(demux, sink).unwrap();

    let stats = run_graph(graph, &ZeroClock, 4).await.expect("ByteStream -> tsdemux -> sink runs");
    assert_eq!(stats.frames_consumed, 3, "three demuxed H.264 AUs reached the sink");
}
