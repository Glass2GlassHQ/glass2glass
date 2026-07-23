//! M376 - stream discovery, the playbin foundation. A demuxer announces *every*
//! elementary stream the container declares as a `BusMessage::StreamCollection`,
//! out of band on the bus, regardless of which one(s) it forwards. This is the
//! discovery half of the playbin stream-collection model (app-driven selection
//! among them is a follow-up).
//!
//! Mux an A/V Matroska stream (H.264 video + AAC audio) with `MkvMuxN`, feed it to
//! `MkvDemux` with a bus attached, and assert one `StreamCollection` is posted
//! listing both streams with the right kind and caps, exactly once.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AudioFormat, Bus, BusMessage, ByteStreamEncoding, Caps, Dim, G2gError, MultiInputElement,
    OutputSink, PushOutcome, Rate, StreamType, VideoCodec,
};
use g2g_plugins::mkvdemux::MkvDemux;
use g2g_plugins::mkvmuxn::MkvMuxN;

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn aac_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::Aac,
        channels: 2,
        sample_rate: 48_000,
    }
}

#[derive(Default)]
struct CollectSink {
    bytes: Vec<u8>,
}
impl OutputSink for CollectSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.bytes.extend_from_slice(s);
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn frame(data: Vec<u8>, pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            ..FrameTiming::default()
        },
        0,
    ))
}

fn annexb(nals: &[&[u8]]) -> Vec<u8> {
    let mut v = Vec::new();
    for n in nals {
        v.extend_from_slice(&[0, 0, 0, 1]);
        v.extend_from_slice(n);
    }
    v
}

/// A minimal ADTS AAC access unit (7-byte header + payload) at 48 kHz stereo.
fn adts_au(payload: &[u8]) -> Vec<u8> {
    let frame_len = payload.len() + 7;
    let sr_index = 3u8; // 48000
    let channels = 2u8;
    let mut au = vec![
        0xFF,
        0xF1,
        (1 << 6) | (sr_index << 2) | ((channels >> 2) & 1),
        ((channels & 3) << 6) | ((frame_len >> 11) & 3) as u8,
        ((frame_len >> 3) & 0xFF) as u8,
        (((frame_len & 7) << 5) as u8) | 0x1F,
        0xFC,
    ];
    au.extend_from_slice(payload);
    au
}

/// Mux an H.264 + AAC Matroska stream with `MkvMuxN`.
async fn mux_av() -> Vec<u8> {
    let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
    let pps = [0x68u8, 0xce, 0x3c, 0x80];
    let idr = [0x65u8, 0x88, 0x84, 0x00];

    let mut mux = MkvMuxN::new(2);
    mux.configure_pipeline(0, &h264_caps()).unwrap();
    mux.configure_pipeline(1, &aac_caps()).unwrap();
    let mut sink = CollectSink::default();
    mux.process(0, frame(annexb(&[&sps, &pps, &idr]), 0), &mut sink)
        .await
        .unwrap();
    mux.process(1, frame(adts_au(&[0x01, 0x02, 0x03]), 0), &mut sink)
        .await
        .unwrap();
    mux.process(
        0,
        frame(annexb(&[&[0x41u8, 0x9a, 0x00]]), 33_000_000),
        &mut sink,
    )
    .await
    .unwrap();
    mux.process(1, frame(adts_au(&[0x04, 0x05]), 21_000_000), &mut sink)
        .await
        .unwrap();
    mux.process(0, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();
    mux.process(1, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();
    sink.bytes
}

#[tokio::test]
async fn mkvdemux_announces_a_stream_collection_for_all_tracks() {
    let file = mux_av().await;

    let (bus, handle) = Bus::new(16);
    let mut demux = MkvDemux::new().with_bus(handle);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Matroska,
        })
        .expect("configure");

    // Feed the muxed A/V stream; the demuxer parses Tracks and announces the
    // collection (it forwards only one stream, but discovery lists them all).
    let mut sink = CollectSink::default();
    demux
        .process(
            PipelinePacket::DataFrame(Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(file.clone().into_boxed_slice())),
                FrameTiming::default(),
                0,
            )),
            &mut sink,
        )
        .await
        .unwrap();

    let mut collections = Vec::new();
    while let Some(msg) = bus.try_recv() {
        if let BusMessage::StreamCollection(c) = msg {
            collections.push(c);
        }
    }
    assert_eq!(
        collections.len(),
        1,
        "exactly one StreamCollection is announced"
    );
    let c = &collections[0];
    assert_eq!(c.len(), 2, "both the video and audio streams are listed");

    let video: Vec<_> = c.streams_of_type(StreamType::Video).collect();
    assert_eq!(video.len(), 1, "one video stream");
    assert_eq!(video[0].id, "matroska-track-1");
    assert!(
        matches!(
            video[0].caps,
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            }
        ),
        "video stream carries H.264 caps"
    );

    let audio: Vec<_> = c.streams_of_type(StreamType::Audio).collect();
    assert_eq!(audio.len(), 1, "one audio stream");
    assert_eq!(audio[0].id, "matroska-track-2");
    assert!(
        matches!(
            audio[0].caps,
            Caps::Audio {
                format: AudioFormat::Aac,
                sample_rate: 48_000,
                ..
            }
        ),
        "audio stream carries AAC caps"
    );

    // Feeding more data does not re-announce the (unchanged) collection.
    demux
        .process(
            PipelinePacket::DataFrame(Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(file.into_boxed_slice())),
                FrameTiming::default(),
                0,
            )),
            &mut sink,
        )
        .await
        .unwrap();
    let more = core::iter::from_fn(|| bus.try_recv())
        .filter(|m| matches!(m, BusMessage::StreamCollection(_)))
        .count();
    assert_eq!(
        more, 0,
        "the collection is announced once, not on every push"
    );
}
