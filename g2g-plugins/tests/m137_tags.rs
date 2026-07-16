//! M137 stream metadata (tags): end to end through the DAG runner. A source
//! emits a synthetic Ogg/Opus stream whose `OpusTags` header carries
//! VorbisComment metadata; `oggdemux` (with a bus attached) surfaces it as a
//! `BusMessage::Tag` while the audio packets flow to the sink. The VorbisComment
//! parse + `TagList` type are unit-tested in `oggdemux` / g2g-core.

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_graph, GraphNode, GraphNodeRef, SourceLoop};
use g2g_core::{
    Bus, BusMessage, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, G2gError,
    Graph, MemoryDomain, OutputSink, PipelineClock, PipelinePacket, Tag,
};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::oggdemux::OggDemux;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// One Ogg page carrying `packets` (each laced into 255-byte segments).
fn page(header_type: u8, serial: u32, seq: u32, packets: &[&[u8]]) -> Vec<u8> {
    let mut table = Vec::new();
    let mut body = Vec::new();
    for p in packets {
        let mut n = p.len();
        loop {
            let seg = n.min(255);
            table.push(seg as u8);
            n -= seg;
            if seg < 255 {
                break;
            }
        }
        body.extend_from_slice(p);
    }
    let mut out = b"OggS".to_vec();
    out.push(0); // version
    out.push(header_type);
    out.extend_from_slice(&0u64.to_le_bytes()); // granule
    out.extend_from_slice(&serial.to_le_bytes());
    out.extend_from_slice(&seq.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // CRC (ignored on read)
    out.push(table.len() as u8);
    out.extend_from_slice(&table);
    out.extend_from_slice(&body);
    out
}

fn opus_head(channels: u8) -> Vec<u8> {
    let mut h = b"OpusHead".to_vec();
    h.push(1);
    h.push(channels);
    h.extend_from_slice(&[0, 0]);
    h.extend_from_slice(&48_000u32.to_le_bytes());
    h.extend_from_slice(&[0, 0, 0]);
    h
}

fn opus_tags(comments: &[(&str, &str)]) -> Vec<u8> {
    let mut p = b"OpusTags".to_vec();
    let vendor: &[u8] = b"g2g";
    p.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    p.extend_from_slice(vendor);
    p.extend_from_slice(&(comments.len() as u32).to_le_bytes());
    for (k, v) in comments {
        let field = [k.as_bytes(), b"=", v.as_bytes()].concat();
        p.extend_from_slice(&(field.len() as u32).to_le_bytes());
        p.extend_from_slice(&field);
    }
    p
}

fn synthetic_ogg() -> Vec<u8> {
    let serial = 0x0C0F_FEE0;
    let mut s = Vec::new();
    s.extend_from_slice(&page(0x02, serial, 0, &[&opus_head(2)]));
    s.extend_from_slice(&page(0x00, serial, 1, &[&opus_tags(&[("TITLE", "Glass"), ("ARTIST", "g2g")])]));
    s.extend_from_slice(&page(0x00, serial, 2, &[&[0x10, 0x11], &[0x20, 0x21]]));
    s
}

/// Emits the whole synthetic Ogg as one `ByteStream{Ogg}` frame.
struct OggSource {
    bytes: Option<Vec<u8>>,
}
impl SourceLoop for OggSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(Caps::ByteStream { encoding: ByteStreamEncoding::Ogg }))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(Caps::ByteStream {
            encoding: ByteStreamEncoding::Ogg,
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
async fn oggdemux_surfaces_vorbis_comment_tags_on_the_bus() {
    let (bus, handle) = Bus::new(8);
    let stats = {
        let mut graph: Graph<GraphNode> = Graph::new();
        let src =
            graph.add_source(GraphNodeRef::Source(Box::new(OggSource { bytes: Some(synthetic_ogg()) })));
        let demux = graph.add_transform(GraphNodeRef::element(OggDemux::new().with_bus(handle)));
        let sink = graph.add_sink(GraphNodeRef::element(FakeSink::new()));
        graph.link(src, demux).unwrap();
        graph.link(demux, sink).unwrap();
        run_graph(graph, &ZeroClock, 4).await.expect("ogg pipeline runs")
    };
    assert_eq!(stats.frames_consumed, 2, "two Opus packets reached the sink");

    let mut posted = None;
    while let Some(m) = bus.try_recv() {
        if let BusMessage::Tag(t) = m {
            posted = Some(t);
        }
    }
    let tags = posted.expect("oggdemux posted a Tag message");
    assert_eq!(tags.tags(), &[Tag::Title("Glass".into()), Tag::Artist("g2g".into())]);
}
