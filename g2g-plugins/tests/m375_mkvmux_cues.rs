//! M375 - the Matroska muxer writes a `Cues` index, closing the seek round-trip:
//! `MkvMux` output is seekable by `MkvDemux` through the very index the muxer
//! wrote (the write side of M373's read). The muxer flushes the `Cues` at EOS
//! (after the last Cluster); a streaming muxer cannot place a front `SeekHead`, so
//! seeking relies on the demuxer having read past the Clusters to the `Cues`
//! (exactly the M373 whole-file-then-seek path).

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::element::{AsyncElement, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::SeekController;
use g2g_core::{ByteStreamEncoding, Caps, Dim, G2gError, Rate, Seek, VideoCodec};
use g2g_plugins::mkvdemux::{MkvDemux, MkvStream};
use g2g_plugins::mkvmux::MkvMux;

const CLUSTER_ID: [u8; 4] = [0x1F, 0x43, 0xB6, 0x75];

#[derive(Default)]
struct Capture {
    frames: Vec<Vec<u8>>,
    flushes: usize,
    segments: usize,
}
impl OutputSink for Capture {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(Frame {
                    domain: MemoryDomain::System(s),
                    ..
                }) => {
                    self.frames.push(s.as_slice().to_vec());
                }
                PipelinePacket::Flush => self.flushes += 1,
                PipelinePacket::Segment(_) => self.segments += 1,
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn vp9_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::Vp9,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Any,
    }
}

fn data_frame(bytes: &[u8], pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
        FrameTiming {
            pts_ns,
            ..FrameTiming::default()
        },
        0,
    ))
}

/// Mux four VP9 keyframes into two Clusters (default 1 s span): two at 0/40 ms,
/// two at 1200/1240 ms. Returns the whole Matroska stream the element produced.
async fn mux_two_clusters() -> Vec<u8> {
    let mut mux = MkvMux::new();
    mux.configure_pipeline(&vp9_caps()).unwrap();
    let mut sink = Capture::default();
    mux.process(data_frame(&[0x01], 0), &mut sink)
        .await
        .unwrap();
    mux.process(data_frame(&[0x02], 40_000_000), &mut sink)
        .await
        .unwrap();
    mux.process(data_frame(&[0x04], 1_200_000_000), &mut sink)
        .await
        .unwrap();
    mux.process(data_frame(&[0x05], 1_240_000_000), &mut sink)
        .await
        .unwrap();
    mux.process(PipelinePacket::Eos, &mut sink).await.unwrap();
    sink.frames.concat()
}

/// The absolute byte offset of the Nth (0-based) Cluster element in the stream.
fn nth_cluster_offset(file: &[u8], n: usize) -> usize {
    file.windows(4)
        .enumerate()
        .filter(|(_, w)| *w == CLUSTER_ID)
        .map(|(i, _)| i)
        .nth(n)
        .unwrap()
}

#[tokio::test]
async fn mkvmux_cues_make_its_own_output_seekable() {
    let file = mux_two_clusters().await;

    // The muxer wrote a Cues element (its id) after the Clusters.
    assert!(
        file.windows(4).any(|w| w == [0x1C, 0x53, 0xBB, 0x6B]),
        "a Cues element was written"
    );
    let cluster1_offset = nth_cluster_offset(&file, 1) as u64;

    let byte = SeekController::new(); // stands in for the FileSrc byte-seek channel
    let time = SeekController::new();
    let mut demux = MkvDemux::new()
        .with_stream(MkvStream::Vp9)
        .with_seek(time.clone(), byte.clone());
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        })
        .expect("configure");

    // 1. Feed the whole muxed stream: every frame demuxes and the muxer's Cues
    // index is parsed (read past the Clusters, the path a streamed Cues needs).
    let mut pre = Capture::default();
    demux.process(data_frame(&file, 0), &mut pre).await.unwrap();
    assert_eq!(
        pre.frames,
        vec![vec![0x01u8], vec![0x02], vec![0x04], vec![0x05]]
    );

    // 2. Seek to 1200 ms. With the muxer's Cues parsed, the demuxer byte-seeks
    // straight to the second Cluster (a re-scan would have requested offset 0).
    time.seek(Seek::flush_to(1_200_000_000));
    let mut post = Capture::default();
    demux.process(data_frame(&[], 0), &mut post).await.unwrap(); // dropped awaiting the flush
    let requested = byte
        .take_pending()
        .expect("an upstream byte-seek was requested");
    assert_eq!(
        requested.start, cluster1_offset,
        "the muxer-written Cues steer the seek to Cluster 2, not a re-scan from 0"
    );

    // 3. The byte source flushes and delivers from the new offset: the second
    // Cluster through to the Cues. The demuxer re-syncs at its 1200 ms keyframe.
    demux
        .process(PipelinePacket::Flush, &mut post)
        .await
        .unwrap();
    let cues_offset = file
        .windows(4)
        .position(|w| w == [0x1C, 0x53, 0xBB, 0x6B])
        .unwrap();
    let cluster1_bytes = &file[cluster1_offset as usize..cues_offset];
    demux
        .process(data_frame(cluster1_bytes, 0), &mut post)
        .await
        .unwrap();

    assert_eq!(post.flushes, 1, "the flush is forwarded downstream");
    assert_eq!(
        post.segments, 1,
        "a resume segment is emitted at the landing keyframe"
    );
    assert_eq!(
        post.frames,
        vec![vec![0x04u8], vec![0x05]],
        "re-synced from Cluster 2 via the Cues"
    );
}
