//! Shared helpers for crate-internal tests: collect packets from an
//! `AsyncElement` without each module reinventing a sink.

use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket,
    PushOutcome,
};

#[derive(Default)]
pub(crate) struct CollectSink {
    pub packets: Vec<PipelinePacket>,
}

impl OutputSink for CollectSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        self.packets
            .push(packet_slot.take().expect("poll_push without a packet"));
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

pub(crate) fn frame(bytes: Vec<u8>) -> Frame {
    frame_with_bytes(0, bytes)
}

pub(crate) fn frame_with_bytes(sequence: u64, bytes: Vec<u8>) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence,
        meta: Default::default(),
    }
}

pub(crate) fn data_bytes(packets: &[PipelinePacket]) -> Vec<u8> {
    packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => {
                Some(f.domain.require_system_slice("test").unwrap().to_vec())
            }
            _ => None,
        })
        .flatten()
        .collect()
}

pub(crate) fn first_caps(packets: &[PipelinePacket]) -> Option<Caps> {
    packets.iter().find_map(|p| match p {
        PipelinePacket::CapsChanged(c) => Some(c.clone()),
        _ => None,
    })
}

/// Configure `element`, push each buffer then EOS, return collected packets.
pub(crate) fn run<E: AsyncElement>(element: &mut E, caps: &Caps, buffers: &[&[u8]]) -> CollectSink {
    element.configure_pipeline(caps).unwrap();
    let mut sink = CollectSink::default();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    for buf in buffers {
        runtime
            .block_on(element.process(PipelinePacket::DataFrame(frame(buf.to_vec())), &mut sink))
            .unwrap();
    }
    runtime
        .block_on(element.process(PipelinePacket::Eos, &mut sink))
        .unwrap();
    sink
}

/// Mux then parse: the samples that come out must equal `samples`, and the
/// parser must announce `audio`.
pub(crate) fn roundtrip<M, P>(
    mut mux: M,
    mut parse: P,
    audio: Caps,
    container: Caps,
    samples: &[u8],
) where
    M: AsyncElement,
    P: AsyncElement,
{
    let written = run(&mut mux, &audio, &[samples]);
    let file = data_bytes(&written.packets);
    let parsed = run(&mut parse, &container, &[&file]);
    assert_eq!(first_caps(&parsed.packets), Some(audio));
    assert_eq!(data_bytes(&parsed.packets), samples);
}

// a FLAC PICTURE block body holding `picture` as a front cover with an empty description and geometry
pub(crate) fn flac_picture_body(media_type: &str, picture: &[u8]) -> Vec<u8> {
    const FRONT_COVER: u32 = 3;
    const GEOMETRY_LEN: usize = 16;
    let mut body = FRONT_COVER.to_be_bytes().to_vec();
    body.extend_from_slice(&(media_type.len() as u32).to_be_bytes());
    body.extend_from_slice(media_type.as_bytes());
    body.extend_from_slice(&0u32.to_be_bytes());
    body.extend_from_slice(&[0; GEOMETRY_LEN]);
    body.extend_from_slice(&(picture.len() as u32).to_be_bytes());
    body.extend_from_slice(picture);
    body
}

pub(crate) const STREAMINFO_TYPE: u8 = 0;

pub(crate) fn flac_block(kind: u8, last: bool, body: &[u8]) -> Vec<u8> {
    use crate::flactag::FLAC_LAST_BLOCK;
    let mut out = Vec::from([if last { FLAC_LAST_BLOCK | kind } else { kind }]);
    out.extend_from_slice(&(body.len() as u32).to_be_bytes()[1..]);
    out.extend_from_slice(body);
    out
}

/// STREAMINFO coding 44100 Hz stereo: a 20-bit rate at byte 10, then
/// channels-1 in the 3 bits behind it.
pub(crate) fn flac_streaminfo() -> Vec<u8> {
    /// STREAMINFO's body is a fixed 34 bytes.
    const STREAMINFO_LEN: usize = 34;
    const SAMPLE_RATE: u32 = 44_100;
    const CHANNELS: u8 = 2;
    let mut body = alloc::vec![0u8; STREAMINFO_LEN];
    body[10] = (SAMPLE_RATE >> 12) as u8;
    body[11] = ((SAMPLE_RATE >> 4) & 0xFF) as u8;
    body[12] = (((SAMPLE_RATE & 0x0F) as u8) << 4) | ((CHANNELS - 1) << 1);
    body
}

// gst-launch-1.0 videotestsrc num-buffers=6 ! video/x-raw,width=64,height=64,framerate=30/1 !
// x264enc key-int-max=3 ! mp4mux fragment-duration=100 streamable=true: two fragments, each with a tfdt
const TWO_FRAGMENTS: &[u8] = include_bytes!("../tests/fixtures/h264_two_fragments.mp4");

// the two-fragment fixture as written, and with each tfdt box renamed to `free`, which keeps every offset
pub(crate) fn two_fragments_with_and_without_tfdt() -> (Vec<u8>, Vec<u8>) {
    let with_tfdt = TWO_FRAGMENTS.to_vec();
    let mut without_tfdt = with_tfdt.clone();
    let tfdt_types: Vec<usize> = with_tfdt
        .windows(4)
        .enumerate()
        .filter(|(_, window)| *window == b"tfdt")
        .map(|(at, _)| at)
        .collect();
    assert_eq!(tfdt_types.len(), 2, "one tfdt per fragment");
    for at in tfdt_types {
        without_tfdt[at..at + 4].copy_from_slice(b"free");
    }
    (with_tfdt, without_tfdt)
}

pub(crate) fn data_pts_ns(packets: &[PipelinePacket]) -> Vec<u64> {
    packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f.timing.pts_ns),
            _ => None,
        })
        .collect()
}
