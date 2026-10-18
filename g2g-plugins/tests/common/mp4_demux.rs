use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{AsyncElement, ByteStreamEncoding, Caps};
use g2g_plugins::mp4demux::Mp4Demux;

use crate::collect_sink::CollectSink;

const READ_CHUNK_LEN: usize = 4096;

#[derive(Debug, Default)]
pub(crate) struct Demuxed {
    pub(crate) caps: Vec<Caps>,
    pub(crate) frames: Vec<Vec<u8>>,
}

// feeds a whole file through a default Mp4Demux in filesrc-sized chunks
pub(crate) async fn demux(file: &[u8]) -> Demuxed {
    let mut demux = Mp4Demux::new();
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Mp4,
        })
        .expect("configure");
    let mut sink = CollectSink::default();
    for chunk in file.chunks(READ_CHUNK_LEN) {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(chunk.to_vec().into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        demux
            .process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .expect("demux");
    }
    demux
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("eos");
    let mut out = Demuxed::default();
    for packet in sink.packets {
        match packet {
            PipelinePacket::CapsChanged(caps) => out.caps.push(caps),
            PipelinePacket::DataFrame(frame) => out.frames.push(
                frame
                    .domain
                    .require_system_slice("test")
                    .expect("system frame")
                    .to_vec(),
            ),
            _ => {}
        }
    }
    out
}
