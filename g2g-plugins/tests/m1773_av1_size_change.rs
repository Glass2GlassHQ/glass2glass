#![cfg(all(feature = "std", any(feature = "dav1d", feature = "rav1d")))]

#[path = "common/collect_sink.rs"]
mod collect_sink;
use collect_sink::CollectSink;

use std::path::PathBuf;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{AsyncElement, ByteStreamEncoding, Caps, PipelineClock};
use g2g_plugins::mkvdemux::{MkvDemux, MkvStream};
use g2g_plugins::registry::default_registry;

// av1-1-b8-03-sizeup cut to its ten 352x288 pictures and the first 1280x720 one
const SIZE_CHANGE_FIXTURE: &str = "av1-1-b8-03-sizeup_11f.mkv";

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

async fn temporal_units_in(path: &PathBuf) -> usize {
    let mut demux = MkvDemux::new().with_stream(MkvStream::Av1);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        })
        .unwrap();
    let file = std::fs::read(path).unwrap();
    let mut sink = CollectSink::default();
    let frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(file.into_boxed_slice())),
        FrameTiming::default(),
        0,
    );
    demux
        .process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
        .unwrap();
    demux.process(PipelinePacket::Eos, &mut sink).await.unwrap();
    sink.packets
        .iter()
        .filter(|packet| matches!(packet, PipelinePacket::DataFrame(_)))
        .count()
}

#[tokio::test]
async fn every_picture_survives_a_mid_stream_size_change() {
    let path = fixture_path(SIZE_CHANGE_FIXTURE);
    let temporal_units = temporal_units_in(&path).await;
    let decoders = [
        #[cfg(feature = "dav1d")]
        "dav1ddec",
        #[cfg(feature = "rav1d")]
        "rav1ddec",
    ];
    for decoder in decoders {
        let line = format!(
            "filesrc location={} ! matroskademux stream=av1 ! av1parse ! {decoder} ! fakesink",
            path.display()
        );
        let graph = parse_launch(&default_registry(), &line)
            .unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
        let stats = run_graph(graph, &ZeroClock, 4)
            .await
            .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
        assert_eq!(stats.frames_consumed, temporal_units as u64, "`{line}`");
    }
}
