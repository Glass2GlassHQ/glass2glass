#![allow(dead_code)] // not every test file uses every helper here

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, G2gError, MultiOutputElement, MultiOutputSink,
    PropValue, PushOutcome,
};
use g2g_plugins::mp4demux::Mp4Demux;
use g2g_plugins::mp4demuxn::{forwardable_streams, Mp4DemuxN, Mp4Port};

use crate::collect_sink::CollectSink;

const READ_CHUNK_LEN: usize = 4096;

#[derive(Debug, Default)]
pub(crate) struct Demuxed {
    pub(crate) caps: Vec<Caps>,
    pub(crate) frames: Vec<Vec<u8>>,
}

// feeds a whole file through a default Mp4Demux in filesrc-sized chunks
pub(crate) async fn demux(file: &[u8]) -> Demuxed {
    demux_with(Mp4Demux::new(), file).await
}

// as `demux`, with the Mp4Demux emitting the audio track `stream` names
pub(crate) async fn demux_audio(file: &[u8], stream: &str) -> Demuxed {
    let mut demux = Mp4Demux::new();
    demux
        .set_property("stream", PropValue::Str(stream.into()))
        .expect("an audio stream name");
    demux_with(demux, file).await
}

impl Demuxed {
    fn record(&mut self, packet: PipelinePacket) {
        match packet {
            PipelinePacket::CapsChanged(caps) => self.caps.push(caps),
            PipelinePacket::DataFrame(frame) => self.frames.push(
                frame
                    .domain
                    .require_system_slice("test")
                    .expect("system frame")
                    .to_vec(),
            ),
            _ => {}
        }
    }
}

impl MultiOutputSink for Demuxed {
    fn poll_push_to(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        _port: usize,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        self.record(packet_slot.take().expect("poll_push without a packet"));
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }

    fn port_count(&self) -> usize {
        1
    }
}

// feeds a whole file through an Mp4DemuxN with one port, on the file's first forwardable stream
pub(crate) async fn demux_first_port(file: &[u8]) -> Demuxed {
    let streams = forwardable_streams(file);
    let first = streams.first().expect("a forwardable stream");
    let mut demux = Mp4DemuxN::new(vec![Mp4Port {
        track_id: first.track_id,
        caps: first.caps.clone(),
    }]);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff,
        })
        .expect("configure");
    let mut out = Demuxed::default();
    let frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(file.to_vec().into_boxed_slice())),
        FrameTiming::default(),
        0,
    );
    demux
        .process(PipelinePacket::DataFrame(frame), &mut out)
        .await
        .expect("demux");
    demux
        .process(PipelinePacket::Eos, &mut out)
        .await
        .expect("eos");
    out
}

async fn demux_with(mut demux: Mp4Demux, file: &[u8]) -> Demuxed {
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
        out.record(packet);
    }
    out
}
