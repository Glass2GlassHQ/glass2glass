// the earliest pts `mkvdemux` gives a file's H.264 track, which is the reorder shift when the video starts at 0

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{AsyncElement, ByteStreamEncoding, Caps};
use g2g_plugins::mkvdemux::{MkvDemux, MkvStream};

use crate::collect_sink::CollectSink;

pub(crate) async fn first_h264_pts_ns(mkv: &[u8]) -> u64 {
    let mut demux = MkvDemux::new().with_stream(MkvStream::H264);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        })
        .expect("mkvdemux accepts a Matroska byte stream");
    let mut out = CollectSink::default();
    let file = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(mkv.to_vec().into_boxed_slice())),
        FrameTiming::default(),
        0,
    );
    demux
        .process(PipelinePacket::DataFrame(file), &mut out)
        .await
        .expect("demux");
    demux
        .process(PipelinePacket::Eos, &mut out)
        .await
        .expect("demux eos");
    out.packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(frame.timing.pts_ns),
            _ => None,
        })
        .min()
        .expect("the file has H.264 frames")
}
